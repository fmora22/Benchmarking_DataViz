#!/usr/bin/env python3
"""
Generate comprehensive vision model benchmarking plots including:
- Plot 12: Tokens/second results
- Plot 13: Average RAM usage results
- Plot 14: Tokens per second per watt
- Plot 15: Joules per token
- Plot 16: Model load time
Plus cross-device and precision (fp16 vs bf16) comparisons
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200


def normalize_model_name(model_name):
    """Normalize model names across devices (internvl and internvl2_2b → internvl_2b)."""
    if model_name in ['internvl', 'internvl2_2b']:
        return 'internvl_2b'
    return model_name


def read_json(p: Path):
    """Read JSON file."""
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runs_jsonl(filepath):
    """Load all records from runs.jsonl file."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def infer_device_label(run_dir: Path):
    """Infer device name from directory path."""
    device_mapping = {
        'dell_output': 'Dell',
        'thor_output': 'Thor',
        'orin_output': 'Orin',
    }
    
    for part in run_dir.parts:
        if part in device_mapping:
            return device_mapping[part]
    return "unknown"


def infer_dtype_short(dtype: str):
    """Extract short dtype name."""
    if not dtype:
        return "na"
    d = str(dtype).lower()
    if "bfloat16" in d or "bf16" in d:
        return "bf16"
    if "float16" in d or "fp16" in d:
        return "fp16"
    if "float32" in d or "fp32" in d:
        return "fp32"
    return d


def extract_model_name(model_key: str, model_id: str):
    """Extract clean model name."""
    # Use model_key first
    if model_key:
        name = model_key.replace('_', ' ').title()
        # Clean up common patterns
        name = name.replace('Vlm', 'VLM')
        name = name.replace('Vl', 'VL')
        name = name.replace('2b', '2B')
        name = name.replace('3n', '3N')
        return name
    
    # Fallback to model_id
    if '/' in model_id:
        return model_id.split('/')[-1]
    return model_id


def find_run_dirs(roots):
    """Find all benchmark run directories."""
    run_dirs = []
    for r in roots:
        root = Path(r)
        for meta in root.rglob("run_meta.json"):
            run_dir = meta.parent
            # Require both summary.json and runs.jsonl
            if (run_dir / "summary.json").exists() and (run_dir / "runs.jsonl").exists():
                run_dirs.append(run_dir)
    return sorted(set(run_dirs))


def calculate_metrics(run_dir: Path):
    """Calculate all metrics for a single run."""
    meta = read_json(run_dir / "run_meta.json")
    summ = read_json(run_dir / "summary.json")
    
    # Skip runs with all errors
    num_errors = summ.get('num_errors', 0)
    total_runs = summ.get('total_runs', 0)
    if num_errors == total_runs and total_runs > 0:
        return None  # Skip completely failed runs
    
    runs = load_runs_jsonl(run_dir / "runs.jsonl")
    
    # Basic info
    model_key = meta.get('model_key', run_dir.parts[-2])
    model_id = meta.get('model_id', '')
    model_name = normalize_model_name(extract_model_name(model_key, model_id))
    device = infer_device_label(run_dir)
    dtype = infer_dtype_short(meta.get('dtype', ''))
    
    # Calculate total tokens
    total_prompt_tokens = 0
    total_response_tokens = 0
    
    for record in runs:
        if record.get('error') is None:
            # Handle both naming conventions
            prompt_tokens = record.get('prompt_tokens') or record.get('input_len') or 0
            response_tokens = record.get('response_tokens_est') or record.get('gen_len') or 0
            total_prompt_tokens += int(prompt_tokens)
            total_response_tokens += int(response_tokens)
    
    total_tokens = total_prompt_tokens + total_response_tokens
    
    # Calculate average RAM usage (convert MB to GB)
    ram_samples = []
    for record in runs:
        sys_after = record.get('sys_after', {})
        ram_used_mb = sys_after.get('ram_used_mb', 0)
        if ram_used_mb > 0:
            ram_samples.append(float(ram_used_mb) / 1024.0)
    
    avg_ram_gb = float(sum(ram_samples)) / len(ram_samples) if ram_samples else 0.0
    
    # Calculate average power
    power_samples = []
    total_energy_joules = 0.0
    
    for record in runs:
        power_stats = record.get('power_stats')
        if power_stats:
            # Try different power field names
            power_avg = (power_stats.get('power_watts_avg') or 
                        power_stats.get('power_gpu_soc_mean_watts'))
            
            if power_avg is not None:
                power_samples.append(float(power_avg))
            
            # Handle energy
            energy = power_stats.get('energy_joules_est')
            if energy is not None:
                total_energy_joules += float(energy)
            elif power_avg is not None:
                # Calculate energy from power and time
                latency_seconds = float(record.get('latency_ms', 0)) / 1000.0
                total_energy_joules += float(power_avg) * latency_seconds
    
    avg_power_watts = float(sum(power_samples)) / len(power_samples) if power_samples else 0.0
    
    # Timing metrics - handle None values
    duration_seconds = summ.get('total_elapsed_seconds') or 0
    latency_ms_mean = summ.get('latency_ms_mean') or 0
    
    duration_seconds = float(duration_seconds) if duration_seconds is not None else 0.0
    latency_ms_mean = float(latency_ms_mean) if latency_ms_mean is not None else 0.0
    
    # Derived metrics
    tokens_per_sec = float(total_tokens) / duration_seconds if duration_seconds > 0 else 0.0
    tps_per_watt = tokens_per_sec / avg_power_watts if avg_power_watts > 0 else 0.0
    joules_per_token = total_energy_joules / float(total_tokens) if total_tokens > 0 else 0.0
    
    # Extract load time if available (from first run or meta)
    load_time_seconds = None
    if runs:
        # Some benchmarks may record initialization time
        first_run = runs[0]
        load_time_seconds = first_run.get('load_time_seconds') or first_run.get('init_time_seconds')
    
    # Try to get from metadata
    if load_time_seconds is None:
        load_time_seconds = meta.get('load_time_seconds') or meta.get('model_load_time_seconds')
    
    if load_time_seconds is not None:
        load_time_seconds = float(load_time_seconds)
    
    return {
        'run_dir': str(run_dir),
        'model_key': str(model_key),
        'model_name': str(model_name),
        'model_id': str(model_id),
        'device': str(device),
        'dtype': str(dtype),
        'total_tokens': int(total_tokens),
        'prompt_tokens': int(total_prompt_tokens),
        'response_tokens': int(total_response_tokens),
        'tokens_per_sec': float(tokens_per_sec),
        'avg_ram_gb': float(avg_ram_gb),
        'avg_power_watts': float(avg_power_watts),
        'tps_per_watt': float(tps_per_watt),
        'joules_per_token': float(joules_per_token),
        'total_energy_joules': float(total_energy_joules),
        'latency_ms_mean': float(latency_ms_mean),
        'duration_seconds': float(duration_seconds),
        'load_time_seconds': load_time_seconds,
        'num_images': int(meta.get('num_images', 0)),
    }


def style_axes(ax):
    """Apply consistent styling to plot axes."""
    ax.grid(True, axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def plot_12_tokens_per_second(df, output_dir):
    """Plot 12: Vision model tokens/second results."""
    print("Generating Plot 12: Tokens per second...")
    
    # Filter out zero values
    data = df[df['tokens_per_sec'] > 0].copy()
    
    if data.empty:
        print("  ⚠ No tokens/second data available")
        return
    
    # Sort by model and device
    data = data.sort_values(['model_name', 'device', 'dtype'])
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Group by model and device
    grouped = data.groupby(['model_name', 'dtype', 'device'])['tokens_per_sec'].mean().reset_index()
    
    # Create x-axis labels
    models = grouped['model_name'].unique()
    x = np.arange(len(models))
    width = 0.12
    
    # Plot bars for each device+dtype combination
    combinations = []
    for device in sorted(grouped['device'].unique()):
        for dtype in sorted(grouped['dtype'].unique()):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        subset = grouped[(grouped['device'] == device) & (grouped['dtype'] == dtype)]
        values = [subset[subset['model_name'] == m]['tokens_per_sec'].values[0] 
                 if len(subset[subset['model_name'] == m]) > 0 else 0 
                 for m in models]
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Plot 12: Vision Model Tokens/Second Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_12_tokens_per_second.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to plot_12_tokens_per_second.png")


def plot_13_ram_usage(df, output_dir):
    """Plot 13: Vision model average RAM usage results."""
    print("Generating Plot 13: Average RAM usage...")
    
    data = df[df['avg_ram_gb'] > 0].copy()
    
    if data.empty:
        print("  ⚠ No RAM usage data available")
        return
    
    data = data.sort_values(['model_name', 'device', 'dtype'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    grouped = data.groupby(['model_name', 'dtype', 'device'])['avg_ram_gb'].mean().reset_index()
    
    models = grouped['model_name'].unique()
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(grouped['device'].unique()):
        for dtype in sorted(grouped['dtype'].unique()):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        subset = grouped[(grouped['device'] == device) & (grouped['dtype'] == dtype)]
        values = [subset[subset['model_name'] == m]['avg_ram_gb'].values[0] 
                 if len(subset[subset['model_name'] == m]) > 0 else 0 
                 for m in models]
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Average RAM Usage (GB)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Plot 13: Vision Model Average RAM Usage Results', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_13_ram_usage.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to plot_13_ram_usage.png")


def plot_14_tokens_per_sec_per_watt(df, output_dir):
    """Plot 14: Vision model tokens per second per watt."""
    print("Generating Plot 14: Tokens per second per watt...")
    
    data = df[df['tps_per_watt'] > 0].copy()
    
    if data.empty:
        print("  ⚠ No tokens/sec/watt data available")
        return
    
    data = data.sort_values(['model_name', 'device', 'dtype'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    grouped = data.groupby(['model_name', 'dtype', 'device'])['tps_per_watt'].mean().reset_index()
    
    models = grouped['model_name'].unique()
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(grouped['device'].unique()):
        for dtype in sorted(grouped['dtype'].unique()):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        subset = grouped[(grouped['device'] == device) & (grouped['dtype'] == dtype)]
        values = [subset[subset['model_name'] == m]['tps_per_watt'].values[0] 
                 if len(subset[subset['model_name'] == m]) > 0 else 0 
                 for m in models]
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Tokens/sec per Watt', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Plot 14: Vision Model Tokens per Second per Watt (Efficiency)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_14_tokens_per_sec_per_watt.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to plot_14_tokens_per_sec_per_watt.png")


def plot_15_joules_per_token(df, output_dir):
    """Plot 15: Vision model joules per token."""
    print("Generating Plot 15: Joules per token...")
    
    data = df[df['joules_per_token'] > 0].copy()
    
    if data.empty:
        print("  ⚠ No joules/token data available")
        return
    
    data = data.sort_values(['model_name', 'device', 'dtype'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    grouped = data.groupby(['model_name', 'dtype', 'device'])['joules_per_token'].mean().reset_index()
    
    models = grouped['model_name'].unique()
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(grouped['device'].unique()):
        for dtype in sorted(grouped['dtype'].unique()):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        subset = grouped[(grouped['device'] == device) & (grouped['dtype'] == dtype)]
        values = [subset[subset['model_name'] == m]['joules_per_token'].values[0] 
                 if len(subset[subset['model_name'] == m]) > 0 else 0 
                 for m in models]
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Joules per Token (Energy Cost)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Plot 15: Vision Model Joules per Token', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_15_joules_per_token.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to plot_15_joules_per_token.png")


def plot_16_load_time(df, output_dir):
    """Plot 16: Vision model load time."""
    print("Generating Plot 16: Model load time...")
    
    data = df[df['load_time_seconds'].notna() & (df['load_time_seconds'] > 0)].copy()
    
    if data.empty:
        print("  ⚠ No load time data available in the current dataset")
        print("     Note: Load time may not be recorded in all benchmark runs")
        return
    
    data = data.sort_values(['model_name', 'device', 'dtype'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    grouped = data.groupby(['model_name', 'dtype', 'device'])['load_time_seconds'].mean().reset_index()
    
    models = grouped['model_name'].unique()
    x = np.arange(len(models))
    width = 0.12
    
    combinations = []
    for device in sorted(grouped['device'].unique()):
        for dtype in sorted(grouped['dtype'].unique()):
            combinations.append((device, dtype))
    
    for i, (device, dtype) in enumerate(combinations):
        subset = grouped[(grouped['device'] == device) & (grouped['dtype'] == dtype)]
        values = [subset[subset['model_name'] == m]['load_time_seconds'].values[0] 
                 if len(subset[subset['model_name'] == m]) > 0 else 0 
                 for m in models]
        offset = (i - len(combinations)/2) * width
        ax.bar(x + offset, values, width, label=f'{device} {dtype}')
    
    ax.set_ylabel('Load Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Plot 16: Vision Model Load Time', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'plot_16_load_time.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved to plot_16_load_time.png")


def plot_device_comparison(df, output_dir):
    """Create cross-device comparison plots."""
    print("Generating cross-device comparison plots...")
    
    # Heatmap for tokens/sec across devices and models
    data = df[df['tokens_per_sec'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='tokens_per_sec',
            index='model_name',
            columns='device',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Tokens/sec'})
        ax.set_title('Cross-Device Comparison: Tokens per Second', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'device_comparison_tokens_per_sec.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved device_comparison_tokens_per_sec.png")
    
    # Heatmap for efficiency (tokens/sec/watt)
    data = df[df['tps_per_watt'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='tps_per_watt',
            index='model_name',
            columns='device',
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Tokens/sec/W'})
        ax.set_title('Cross-Device Comparison: Efficiency (Tokens/sec/Watt)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Device', fontsize=12, fontweight='bold')
        ax.set_ylabel('Model', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'device_comparison_efficiency.png', dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved device_comparison_efficiency.png")


def plot_precision_comparison(df, output_dir):
    """Create FP16 vs BF16 precision comparison plots."""
    print("Generating precision (FP16 vs BF16) comparison plots...")
    
    # Filter data with both precisions
    fp16_data = df[df['dtype'] == 'fp16'].copy()
    bf16_data = df[df['dtype'] == 'bf16'].copy()
    
    if fp16_data.empty or bf16_data.empty:
        print("  ⚠ Not enough precision data for comparison")
        return
    
    # Tokens/sec comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Tokens/sec by precision
    ax = axes[0, 0]
    data = df[df['tokens_per_sec'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='tokens_per_sec',
            index='model_name',
            columns='dtype',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Tokens per Second', fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Tokens/sec: FP16 vs BF16', fontsize=12, fontweight='bold')
        ax.legend(title='Precision', loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        style_axes(ax)
    
    # Plot 2: RAM usage by precision
    ax = axes[0, 1]
    data = df[df['avg_ram_gb'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='avg_ram_gb',
            index='model_name',
            columns='dtype',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('RAM Usage (GB)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('RAM Usage: FP16 vs BF16', fontsize=12, fontweight='bold')
        ax.legend(title='Precision', loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        style_axes(ax)
    
    # Plot 3: Efficiency by precision
    ax = axes[1, 0]
    data = df[df['tps_per_watt'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='tps_per_watt',
            index='model_name',
            columns='dtype',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Tokens/sec/Watt', fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Efficiency: FP16 vs BF16', fontsize=12, fontweight='bold')
        ax.legend(title='Precision', loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        style_axes(ax)
    
    # Plot 4: Energy cost by precision
    ax = axes[1, 1]
    data = df[df['joules_per_token'] > 0].copy()
    if not data.empty:
        pivot = data.pivot_table(
            values='joules_per_token',
            index='model_name',
            columns='dtype',
            aggfunc='mean'
        )
        pivot.plot(kind='bar', ax=ax, width=0.8)
        ax.set_ylabel('Joules per Token', fontsize=11, fontweight='bold')
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_title('Energy Cost: FP16 vs BF16', fontsize=12, fontweight='bold')
        ax.legend(title='Precision', loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_comparison_fp16_vs_bf16.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved precision_comparison_fp16_vs_bf16.png")


def plot_device_precision_combined(df, output_dir):
    """Create combined device+precision comparison."""
    print("Generating combined device+precision comparison...")
    
    data = df[df['tokens_per_sec'] > 0].copy()
    if data.empty:
        return
    
    # Create combined label
    data['device_dtype'] = data['device'] + ' ' + data['dtype']
    
    # Group by model and device+dtype
    pivot = data.pivot_table(
        values='tokens_per_sec',
        index='model_name',
        columns='device_dtype',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(16, 8))
    pivot.plot(kind='bar', ax=ax, width=0.85)
    ax.set_ylabel('Tokens per Second', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Tokens/sec: All Devices and Precisions', fontsize=14, fontweight='bold')
    ax.legend(title='Device + Precision', loc='upper left', fontsize=9, ncol=2)
    ax.tick_params(axis='x', rotation=45)
    style_axes(ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'combined_device_precision_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved combined_device_precision_comparison.png")


def generate_summary_report(df, output_dir):
    """Generate a summary report of all metrics."""
    print("Generating summary report...")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VISION MODEL BENCHMARKING SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append(f"Total benchmark runs: {len(df)}")
    report_lines.append(f"Models tested: {df['model_name'].nunique()}")
    report_lines.append(f"Devices: {', '.join(sorted(df['device'].unique()))}")
    report_lines.append(f"Precisions: {', '.join(sorted(df['dtype'].unique()))}")
    report_lines.append("")
    
    # Best performers
    report_lines.append("-" * 80)
    report_lines.append("BEST PERFORMERS")
    report_lines.append("-" * 80)
    
    if not df[df['tokens_per_sec'] > 0].empty:
        best_tps = df[df['tokens_per_sec'] > 0].nlargest(3, 'tokens_per_sec')
        report_lines.append("\nTop 3 - Tokens per Second:")
        for idx, row in best_tps.iterrows():
            report_lines.append(f"  {row['model_name']:20s} on {row['device']:10s} ({row['dtype']}): {row['tokens_per_sec']:8.1f} tok/s")
    
    if not df[df['tps_per_watt'] > 0].empty:
        best_eff = df[df['tps_per_watt'] > 0].nlargest(3, 'tps_per_watt')
        report_lines.append("\nTop 3 - Efficiency (Tokens/sec/Watt):")
        for idx, row in best_eff.iterrows():
            report_lines.append(f"  {row['model_name']:20s} on {row['device']:10s} ({row['dtype']}): {row['tps_per_watt']:8.2f} tok/s/W")
    
    if not df[df['joules_per_token'] > 0].empty:
        best_energy = df[df['joules_per_token'] > 0].nsmallest(3, 'joules_per_token')
        report_lines.append("\nTop 3 - Energy Efficiency (Lowest Joules/Token):")
        for idx, row in best_energy.iterrows():
            report_lines.append(f"  {row['model_name']:20s} on {row['device']:10s} ({row['dtype']}): {row['joules_per_token']:8.3f} J/tok")
    
    # Model-by-model breakdown
    report_lines.append("")
    report_lines.append("-" * 80)
    report_lines.append("MODEL-BY-MODEL BREAKDOWN")
    report_lines.append("-" * 80)
    
    for model in sorted(df['model_name'].unique()):
        model_data = df[df['model_name'] == model]
        report_lines.append(f"\n{model}:")
        
        for _, row in model_data.iterrows():
            report_lines.append(f"  {row['device']:10s} {row['dtype']:5s}: "
                              f"TPS={row['tokens_per_sec']:6.1f}, "
                              f"RAM={row['avg_ram_gb']:5.1f}GB, "
                              f"Eff={row['tps_per_watt']:5.2f} tok/s/W, "
                              f"Energy={row['joules_per_token']:6.3f} J/tok")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_file = output_dir / 'benchmark_summary_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    # Also print to console
    print(report_text)
    print(f"\n✓ Saved summary report to {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate vision model benchmark plots')
    parser.add_argument('--roots', nargs='+', required=True,
                       help='Top-level output directories (e.g., dell_output orin_output thor_output)')
    parser.add_argument('--out', required=True,
                       help='Output directory for plots and data')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("VISION MODEL BENCHMARKING - PLOT GENERATION")
    print("=" * 80)
    print(f"\nScanning directories: {', '.join(args.roots)}")
    
    # Find all run directories
    run_dirs = find_run_dirs(args.roots)
    print(f"Found {len(run_dirs)} benchmark runs\n")
    
    if not run_dirs:
        print("ERROR: No benchmark results found!")
        return
    
    # Calculate metrics for all runs
    print("Processing benchmark data...")
    all_metrics = []
    skipped_runs = []
    for run_dir in run_dirs:
        try:
            metrics = calculate_metrics(run_dir)
            if metrics is None:
                skipped_runs.append(run_dir)
                print(f"  ⊘ Skipped {run_dir.parts[-2]:15s} on {infer_device_label(run_dir):10s} (all runs failed)")
                continue
            all_metrics.append(metrics)
            print(f"  ✓ {metrics['model_name']:20s} on {metrics['device']:10s} ({metrics['dtype']})")
        except Exception as e:
            print(f"  ✗ Error processing {run_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    if skipped_runs:
        print(f"\nSkipped {len(skipped_runs)} failed runs")
    
    if not all_metrics:
        print("\nERROR: No valid metrics calculated!")
        return
    
    # Create DataFrame - use json intermediary to avoid pandas/numpy version issues
    import csv
    csv_temp = output_dir / 'temp_metrics.csv'
    
    # Write to CSV
    if all_metrics:
        keys = all_metrics[0].keys()
        with open(csv_temp, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        # Read back as DataFrame
        df = pd.read_csv(csv_temp)
        csv_temp.unlink()  # Delete temp file
    else:
        df = pd.DataFrame()
    
    # Save raw data
    csv_file = output_dir / 'vision_model_metrics.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved metrics to {csv_file}\n")
    
    # Generate all plots
    print("=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    print()
    
    plot_12_tokens_per_second(df, output_dir)
    plot_13_ram_usage(df, output_dir)
    plot_14_tokens_per_sec_per_watt(df, output_dir)
    plot_15_joules_per_token(df, output_dir)
    plot_16_load_time(df, output_dir)
    
    print()
    print("=" * 80)
    print("GENERATING COMPARISON PLOTS")
    print("=" * 80)
    print()
    
    plot_device_comparison(df, output_dir)
    plot_precision_comparison(df, output_dir)
    plot_device_precision_combined(df, output_dir)
    
    print()
    print("=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    print()
    
    generate_summary_report(df, output_dir)
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir.absolute()}")
    print(f"  - Plots: {len(list(output_dir.glob('*.png')))} PNG files")
    print(f"  - Data: vision_model_metrics.csv")
    print(f"  - Report: benchmark_summary_report.txt")


if __name__ == '__main__':
    main()
