# Vision Model Benchmarking Results

Generated on: January 30, 2026

## Overview

This directory contains comprehensive benchmarking plots and analysis for vision language models across three edge devices:
- **Dell Pro Max GB10** - High-performance desktop GPU
- **Jetson Thor** - NVIDIA edge AI platform  
- **Jetson AGX Orin** - NVIDIA edge AI platform

## Devices Compared

The benchmarks compare models across different hardware architectures:
- **Dell Pro Max GB10**: Desktop-class performance baseline
- **Jetson Thor**: High-end edge AI device
- **Jetson AGX Orin**: Mid-range edge AI device

## Models Tested

- **Gemma3N** - Google's efficient vision-language model
- **InternVL / InternVL2-2B** - Open-source vision-language model
- **LLaVA** - Visual instruction tuning model
- **Moondream2** - Lightweight vision-language model
- **Phi** - Microsoft's efficient language model series
- **SmolVLM2** - Compact vision-language model

## Precision Modes

All models were tested with:
- **FP16** (Float16) - Standard half-precision
- **BF16** (BFloat16) - Brain floating-point format

## Generated Plots

### Main Performance Metrics

1. **plot_12_tokens_per_second.png**
   - Tokens per second across all models, devices, and precisions
   - Shows raw throughput performance

2. **plot_13_ram_usage.png**
   - Average RAM usage in GB
   - Important for deployment on memory-constrained devices

3. **plot_14_tokens_per_sec_per_watt.png**
   - Efficiency metric: tokens/sec per watt of power
   - Critical for edge deployment and energy efficiency

4. **plot_15_joules_per_token.png**
   - Energy cost per token generated
   - Lower is better - shows energy efficiency

5. **plot_16_load_time.png** (if available)
   - Model loading time in seconds
   - Note: Not all benchmark runs recorded this metric

### Comparison Plots

6. **device_comparison_tokens_per_sec.png**
   - Heatmap showing tokens/sec across devices and models
   - Easy visual comparison of device performance

7. **device_comparison_efficiency.png**
   - Heatmap showing efficiency (tokens/sec/watt)
   - Highlights most power-efficient configurations

8. **precision_comparison_fp16_vs_bf16.png**
   - 4-panel comparison of FP16 vs BF16 across metrics:
     * Tokens/sec
     * RAM usage
     * Efficiency
     * Energy cost

9. **combined_device_precision_comparison.png**
   - Combined view of all device+precision combinations
   - Comprehensive overview of all test configurations

## Data Files

- **vision_model_metrics.csv** - Complete metrics for all 29 successful runs
  - Columns include: tokens_per_sec, avg_ram_gb, avg_power_watts, tps_per_watt, joules_per_token, etc.
  
- **benchmark_summary_report.txt** - Human-readable summary report
  - Top performers by category
  - Model-by-model breakdown
  - Quick reference statistics

## Key Findings

### Best Performers

**Throughput (Tokens/Second):**
1. SmolVLM2 on Dell (bf16): 795.6 tok/s
2. SmolVLM2 on Dell (fp16): 773.3 tok/s  
3. SmolVLM2 on Thor (bf16): 554.1 tok/s

**Efficiency (Tokens/sec/Watt):**
1. SmolVLM2 on Thor (bf16): 18.78 tok/s/W
2. SmolVLM2 on Thor (fp16): 18.70 tok/s/W
3. SmolVLM2 on Dell (bf16): 18.55 tok/s/W

**Energy Efficiency (Lowest Joules/Token):**
1. SmolVLM2 on Thor (bf16): 0.051 J/tok
2. SmolVLM2 on Thor (fp16): 0.051 J/tok
3. SmolVLM2 on Dell (bf16): 0.052 J/tok

### Observations

- **SmolVLM2** consistently shows the best performance across all metrics
- **Jetson Thor** demonstrates excellent power efficiency despite lower raw throughput than Dell
- **BF16 vs FP16**: Minimal performance difference in most cases, suggesting BF16 is a good choice
- **Dell Pro Max** shows highest raw throughput but lower efficiency than Thor
- **Larger models** (LLaVA, InternVL) show higher energy costs per token

## Usage

To regenerate these plots:

```bash
python vision_model_plots.py --roots dell_output orin_output thor_output --out vision_plots
```

## Notes

- One Gemma3N run on Thor (fp16) was skipped due to complete failure (all 2500 runs failed)
- Load time data was not available in the current benchmark runs
- Power monitoring methods vary by device:
  - Dell: NVIDIA pynvml (GPU power only)
  - Jetson devices: tegrastats (GPU+SOC combined power)

## Requirements

```
python >= 3.8
pandas < 3.0
numpy < 2.0  
matplotlib
seaborn
```
