"""
Bridge narrative graphs for vision model benchmarking.

These plots connect the story from task characteristics through deployment constraints:
1. Length–Latency Correlation (causal bridge: do longer outputs drive latency tails?)
2. Failure/Feasibility Heatmap (systems constraint: what can actually run?)
3. Task Mix Sensitivity (practical tool: what happens if task mix changes?)
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path('/Users/fymor/Downloads/FALL26/SAGE/Benchmarking_DataViz')
OUTPUT_DIR = BASE_DIR / 'bridge_narrative'
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE_NAMES = {
    'dell_output': 'Dell Pro Max GB10',
    'thor_output': 'Jetson Thor',
    'orin_output': 'Jetson AGX Orin'
}

TASK_COLORS = {
    'caption_brief': '#1f77b4',       # blue
    'attributes': '#ff7f0e',          # orange
    'objects_and_counts': '#2ca02c',  # green
    'spatial_relationships': '#d62728', # red
    'scene_context': '#9467bd'        # purple
}

TASK_NAMES = {
    'caption_brief': 'Brief Caption',
    'attributes': 'Attributes',
    'objects_and_counts': 'Count Objects',
    'spatial_relationships': 'Spatial Relationships',
    'scene_context': 'Scene Context'
}

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def normalize_model_name(model_name):
    """Normalize model names across devices (internvl and internvl2_2b → internvl_2b)."""
    if model_name in ['internvl', 'internvl2_2b']:
        return 'internvl_2b'
    return model_name


def load_run_data(device_dir, model_name, run_dir):
    """Load all runs.jsonl data for a specific run."""
    runs_file = device_dir / model_name / run_dir / 'runs.jsonl'
    
    if not runs_file.exists():
        return None
    
    runs = []
    try:
        with open(runs_file, 'r') as f:
            for line in f:
                try:
                    run = json.loads(line.strip())
                    runs.append(run)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error loading {runs_file}: {e}")
        return None
    
    return runs


def extract_all_run_metrics():
    """Extract length, latency, task, success for all runs."""
    all_data = []
    
    for device_key, device_name in DEVICE_NAMES.items():
        device_dir = BASE_DIR / device_key
        if not device_dir.exists():
            continue
        
        # Iterate through all models and runs
        for model_path in device_dir.iterdir():
            if not model_path.is_dir():
                continue
            model_name = model_path.name
            
            for run_path in model_path.iterdir():
                if not run_path.is_dir():
                    continue
                run_name = run_path.name
                
                # Determine precision
                precision = 'FP16' if 'fp16' in run_name else 'BF16'
                
                runs = load_run_data(device_dir, model_name, run_name)
                if runs is None or len(runs) == 0:
                    continue
                
                # Extract metrics from each run
                for run in runs:
                    try:
                        success = run.get('error') is None
                        all_data.append({
                            'device': device_name,
                            'model': normalize_model_name(model_name),
                            'run': run_name,
                            'precision': precision,
                            'task': run.get('task', 'unknown'),
                            'gen_len': run.get('gen_len', 0),
                            'latency_ms': run.get('latency_ms', 0),
                            'input_len': run.get('input_len', 0),
                            'success': success,
                            'error': run.get('error'),
                            'output_text': run.get('output_text', ''),
                        })
                    except Exception as e:
                        continue
    
    return pd.DataFrame(all_data)


def plot_length_latency_correlation():
    """
    Plot 1: Length vs Latency Correlation
    
    Scatter: gen_len (x) vs latency_ms (y), colored by task type.
    This answers: "Are latency tails explained by longer outputs?"
    """
    df = extract_all_run_metrics()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each task type with its color
    for task, color in TASK_COLORS.items():
        task_data = df[df['task'] == task]
        if len(task_data) > 0:
            ax.scatter(
                task_data['gen_len'],
                task_data['latency_ms'],
                alpha=0.5,
                s=50,
                color=color,
                label=TASK_NAMES.get(task, task),
                edgecolors='none'
            )
    
    # Add trend line
    valid_data = df[(df['latency_ms'] > 0) & (df['gen_len'] > 0)]
    if len(valid_data) > 10:
        z = np.polyfit(valid_data['gen_len'], valid_data['latency_ms'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['gen_len'].min(), valid_data['gen_len'].max(), 100)
        ax.plot(x_trend, p(x_trend), 'k--', linewidth=2, alpha=0.7, label=f'Trend (slope={z[0]:.1f} ms/token)')
    
    ax.set_xlabel('Generated Tokens (gen_len)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Length–Latency Correlation\n"Are latency tails explained by longer outputs?"', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_length_latency_correlation.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 01_length_latency_correlation.png")
    plt.close()


def plot_failure_feasibility_heatmap():
    """
    Plot 2: Failure/Feasibility Heatmap
    
    Heatmap: rows=models, cols=devices×precision, values=success rate.
    This answers: "Can it run reliably at all?" and marks deployment boundaries.
    """
    df = extract_all_run_metrics()
    
    # Calculate success rate per model, device, precision
    success_by_config = df.groupby(['model', 'device', 'precision']).agg({
        'success': ['sum', 'count']
    }).reset_index()
    success_by_config.columns = ['model', 'device', 'precision', 'successes', 'total']
    success_by_config['success_rate'] = (success_by_config['successes'] / success_by_config['total'] * 100).round(1)
    
    # Pivot for heatmap: rows=model, columns=device+precision
    success_by_config['config'] = success_by_config['device'].str.replace('Jetson ', '').str.replace('Dell Pro Max ', '') + '\n' + success_by_config['precision']
    
    pivot_data = success_by_config.pivot_table(
        index='model',
        columns='config',
        values='success_rate',
        fill_value=0
    )
    
    # Sort by average success rate
    pivot_data = pivot_data.loc[pivot_data.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Success Rate (%)'},
        vmin=0,
        vmax=100,
        linewidths=0.5,
        ax=ax,
        cbar=True
    )
    
    ax.set_xlabel('Device × Precision', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Failure/Feasibility Heatmap\n"Resource limits create feasibility boundaries that dominate deployment decisions"',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_failure_feasibility_heatmap.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 02_failure_feasibility_heatmap.png")
    plt.close()


def plot_task_mix_sensitivity():
    """
    Plot 3: Task Mix Sensitivity
    
    Bar chart showing predicted latency under different task mixes.
    This answers: "Given your app's task mix, what will happen?"
    """
    df = extract_all_run_metrics()
    
    # Get average latency per task per model
    task_latency = df[df['success']].groupby(['model', 'task'])['latency_ms'].mean().reset_index()
    
    # Define task mixes
    mixes = {
        'Caption-Heavy\n(70% brief, 30% attributes)': {
            'caption_brief': 0.7,
            'attributes': 0.3,
            'objects_and_counts': 0,
            'spatial_relationships': 0,
            'scene_context': 0,
        },
        'Count-Heavy\n(100% counting)': {
            'caption_brief': 0,
            'attributes': 0,
            'objects_and_counts': 1.0,
            'spatial_relationships': 0,
            'scene_context': 0,
        },
        'Spatial-Heavy\n(70% spatial, 30% caption)': {
            'caption_brief': 0.3,
            'attributes': 0,
            'objects_and_counts': 0,
            'spatial_relationships': 0.7,
            'scene_context': 0,
        },
        'Balanced\n(equal weights)': {
            'caption_brief': 0.2,
            'attributes': 0.2,
            'objects_and_counts': 0.2,
            'spatial_relationships': 0.2,
            'scene_context': 0.2,
        }
    }
    
    # Calculate predicted latency for each mix and model
    results = []
    for mix_name, weights in mixes.items():
        for model in df['model'].unique():
            model_tasks = task_latency[task_latency['model'] == model]
            
            predicted_latency = 0
            for task, weight in weights.items():
                task_data = model_tasks[model_tasks['task'] == task]
                if len(task_data) > 0:
                    predicted_latency += task_data['latency_ms'].values[0] * weight
            
            if predicted_latency > 0:
                results.append({
                    'mix': mix_name,
                    'model': model,
                    'predicted_latency_ms': predicted_latency
                })
    
    results_df = pd.DataFrame(results)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))
    
    mix_order = list(mixes.keys())
    models = sorted(results_df['model'].unique())
    x = np.arange(len(mix_order))
    width = 0.12
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model].set_index('mix')
        values = [model_data.loc[mix, 'predicted_latency_ms'] if mix in model_data.index else 0 
                  for mix in mix_order]
        ax.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax.set_xlabel('Task Mix Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Task Mix Sensitivity\n"Given your app\'s task mix, here\'s what will happen"',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(models)-1) / 2)
    ax.set_xticklabels(mix_order, fontsize=10)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_task_mix_sensitivity.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: 03_task_mix_sensitivity.png")
    plt.close()


def generate_bridge_narrative_report():
    """Generate a report explaining the narrative these plots create."""
    df = extract_all_run_metrics()
    
    # Key statistics
    valid_data = df[df['success']]
    
    # Correlation between gen_len and latency
    corr = valid_data['gen_len'].corr(valid_data['latency_ms'])
    
    # Task breakdown
    task_stats = valid_data.groupby('task').agg({
        'latency_ms': ['mean', 'std'],
        'gen_len': 'mean',
        'success': 'count'
    }).round(2)
    
    # Feasibility by device
    feasibility = df.groupby(['device', 'precision']).agg({
        'success': lambda x: (x.sum() / len(x) * 100).round(1)
    }).reset_index()
    feasibility.columns = ['device', 'precision', 'success_rate']
    
    report = f"""
BRIDGE NARRATIVE ANALYSIS
========================

These plots tell a coherent story about vision model deployment:

1. LENGTH–LATENCY CORRELATION (Causal Bridge)
   ────────────────────────────
   
   Finding: Correlation between gen_len and latency = {corr:.3f}
   
   Interpretation:
   - If correlation > 0.7: "Latency tails are strongly driven by output length"
   - If correlation 0.3–0.7: "Output length explains some variance; other factors matter"
   - If correlation < 0.3: "Output length is a minor factor; look at task complexity"
   
   Task Breakdown:
{task_stats.to_string()}
   
   Narrative: "Latency scales with output length; task formulation is a primary driver of runtime variance."

2. FAILURE/FEASIBILITY HEATMAP (Systems Constraint)
   ───────────────────────────
   
   All configurations with <100% success rate are shaded in the heatmap.
   These are not "missing data"—they are "not feasible" under the benchmark conditions.
   
   Feasibility Summary:
{feasibility.to_string()}
   
   Narrative: "Resource limits create feasibility boundaries that dominate deployment decisions."

3. TASK MIX SENSITIVITY (Practical Tool)
   ──────────────────────
   
   This plot shows how model comparison shifts when task mix changes.
   It answers: "Which model is 'best' depends on what your app actually does."
   
   Use case: A vision system that is 50% spatial reasoning will prefer different models
   than one that is 70% brief captions.
   
   Narrative: "Model selection depends on anticipated task mix; no single 'best' model exists."

────────────────────────────────────────────────────────────────────────

RECOMMENDED FIGURE LINEUP FOR PAPER:

For a PAISE short paper, use these 5 figures:

  1. Task → Caption length distribution [from existing analysis]
  2. Length–Latency correlation [NEW: bridge from task to runtime]
  3. Latency tail metrics (P90 or CV) [from existing analysis]
  4. Speed vs detail tradeoff [from existing analysis, now with context]
  5. Failure/Feasibility heatmap [NEW: deployment reality check]

This creates the narrative:
  Task characteristics → Output length → Runtime latency → Deployment feasibility → Model selection

────────────────────────────────────────────────────────────────────────

INTERPRETATION GUIDE FOR REVIEWERS:

Q: "Why do some models have high latency?"
A: "See Figure 2 (Length–Latency): output length is a primary driver. See Figure 3 for 
   task-specific decomposition."

Q: "Which model should I use?"
A: "See Figure 5 (Feasibility): first ensure your target device can run it. Then see Figure 4 
   (Speed–Detail tradeoff): choose based on your latency budget and output quality needs."

Q: "Will this model work for my application?"
A: "See Figure 3 (Task Mix): your specific workload mix determines which model ranks best."

"""
    
    with open(OUTPUT_DIR / 'bridge_narrative_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✓ Saved: bridge_narrative_report.txt")


if __name__ == '__main__':
    print("Generating bridge narrative plots...\n")
    
    print("1. Extracting all run metrics...")
    
    print("2. Creating Length–Latency Correlation plot...")
    plot_length_latency_correlation()
    
    print("3. Creating Failure/Feasibility Heatmap...")
    plot_failure_feasibility_heatmap()
    
    print("4. Creating Task Mix Sensitivity chart...")
    plot_task_mix_sensitivity()
    
    print("5. Generating narrative report...")
    generate_bridge_narrative_report()
    
    print(f"\n✅ All bridge narrative plots saved to: {OUTPUT_DIR}")
    print("\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob('*.png')):
        print(f"  - {f.name}")
    for f in sorted(OUTPUT_DIR.glob('*.txt')):
        print(f"  - {f.name}")
