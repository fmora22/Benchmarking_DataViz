# Caption Quality & Speed Analysis

Generated on: January 30, 2026

## Overview

This analysis examines both the **speed characteristics** and **caption quality** of vision language models across three edge devices. Unlike the basic throughput metrics, this digs into latency distributions, consistency, and the actual quality of generated captions.

## Speed Analysis Plots

### 1. speed_latency_distributions.png
**Box plot showing latency distributions**
- Shows min, max, median, quartiles for each model/device/dtype
- Reveals outliers and variance in inference times
- Helps identify which models have predictable performance

**Key Finding:** Moondream2 shows the fastest and most consistent latency across all configurations.

### 2. speed_latency_p90.png
**P90 latency comparison**
- 90th percentile latency - important for real-time applications
- Shows tail latency performance
- Critical for user experience (90% of requests complete within this time)

**Key Finding:** Moondream2 maintains sub-1200ms P90 latency even on edge devices.

### 3. speed_latency_consistency.png
**Coefficient of Variation (CV) - Lower is better**
- CV = (std_dev / mean) × 100
- Shows how predictable/consistent model performance is
- Important for production deployments

**Key Finding:** Phi on Orin shows best consistency (18.5% CV), followed by Gemma3N on Dell (19.5% CV).

### 4. speed_by_task_type.png
**Latency breakdown by task type**
- Compares: caption_brief, objects_and_counts, spatial_relationships, scene_context, attributes
- Shows which tasks are more computationally expensive
- Reveals model behavior patterns

## Quality Analysis Plots

### 5. quality_caption_lengths.png
**Average caption length by model**
- Longer captions generally indicate more detail
- Shows verbosity differences between models
- Can indicate different training approaches

**Key Finding:** LLaVA generates the most detailed captions (~433 chars), while Moondream2 is more concise (~168 chars).

### 6. quality_caption_length_by_task.png
**Caption length variation by task type**
- Shows how models respond to different prompt types
- Reveals if models follow instructions properly
- Indicates task-specific behavior

**Key Finding:** Different tasks naturally produce different length outputs (as expected), with spatial_relationships and scene_context producing longer descriptions.

### 7. quality_success_rates.png
**Task completion success rate**
- Percentage of tasks completed without errors
- Critical reliability metric
- Shows which configurations are production-ready

**Key Finding:** Dell runs show 100% success rates across all models, while some Jetson configurations have slightly lower rates.

### 8. speed_vs_quality_tradeoff.png
**Scatter plot: Speed vs Caption Detail**
- X-axis: Mean latency (lower = faster)
- Y-axis: Caption length (higher = more detailed)
- Shows the tradeoff between speed and detail
- Different markers for devices, colors for precision

**Key Finding:** Moondream2 offers the best balance - fast with reasonable detail. LLaVA provides maximum detail but at higher latency cost.

## Key Insights

### 🏆 Best for Speed
1. **Moondream2 on Thor (fp16)**: 894ms mean latency
2. **Moondream2 on Thor (bf16)**: 895ms mean latency
3. **Moondream2 on Dell (fp16)**: 952ms mean latency

### 📝 Best for Caption Detail
1. **LLaVA on Thor (bf16)**: 435 chars average
2. **LLaVA on Dell (bf16)**: 433 chars average
3. **LLaVA on Dell (fp16)**: 433 chars average

### ⚡ Most Consistent
1. **Phi on Orin (fp16)**: 18.5% CV
2. **Phi on Orin (bf16)**: 19.1% CV
3. **Gemma3N on Dell (bf16)**: 19.5% CV

### 🎯 Best Speed/Quality Balance
1. **Moondream2 on Thor (fp16)**: Fast (894ms) + Reasonable detail (168 chars)
2. **Moondream2 on Thor (bf16)**: Fast (895ms) + Reasonable detail (168 chars)
3. **Moondream2 on Dell (bf16)**: Fast (954ms) + Reasonable detail (168 chars)

### ✅ Most Reliable
- **Dell configurations**: 100% success rate across all models
- **Thor configurations**: 99-100% success rate
- **Orin configurations**: Variable (some models missing from test suite)

## Model Characteristics

### Moondream2
- **Speed**: ⭐⭐⭐⭐⭐ Fastest overall
- **Detail**: ⭐⭐⭐ Moderate, concise captions
- **Consistency**: ⭐⭐⭐⭐ Good
- **Best for**: Real-time applications, edge deployment, quick summaries

### LLaVA
- **Speed**: ⭐⭐ Slower
- **Detail**: ⭐⭐⭐⭐⭐ Most detailed captions
- **Consistency**: ⭐⭐⭐ Moderate
- **Best for**: When detail matters more than speed, offline processing

### SmolVLM2
- **Speed**: ⭐⭐⭐⭐ Fast (especially token throughput)
- **Detail**: ⭐⭐⭐ Moderate
- **Consistency**: ⭐⭐⭐⭐ Good
- **Best for**: High throughput batch processing, efficient deployment

### Phi
- **Speed**: ⭐⭐ Slower
- **Detail**: ⭐⭐⭐⭐ Good, structured responses
- **Consistency**: ⭐⭐⭐⭐⭐ Most consistent
- **Best for**: Predictable latency requirements, structured outputs

### Gemma3N
- **Speed**: ⭐⭐⭐ Moderate
- **Detail**: ⭐⭐⭐⭐ Good detail
- **Consistency**: ⭐⭐⭐⭐⭐ Very consistent
- **Best for**: Balanced performance, reliable inference

### InternVL
- **Speed**: ⭐⭐⭐ Moderate
- **Detail**: ⭐⭐⭐⭐ Good
- **Consistency**: ⭐⭐⭐ Moderate
- **Best for**: Balanced applications, research

## Task Type Analysis

Different tasks have different computational requirements:

1. **caption_brief**: Fastest, shortest outputs
2. **objects_and_counts**: Fast, structured outputs
3. **attributes**: Moderate speed, descriptive
4. **spatial_relationships**: Slower, more complex reasoning
5. **scene_context**: Variable, depends on scene complexity

## Device Comparison

### Dell Pro Max GB10
- **Strengths**: Highest throughput, 100% success rates, handles all models well
- **Weaknesses**: Higher power consumption
- **Best for**: Development, maximum performance

### Jetson Thor
- **Strengths**: Best power efficiency, good balance of speed and energy
- **Weaknesses**: Slightly higher latency than Dell on some models
- **Best for**: Edge deployment where efficiency matters

### Jetson AGX Orin
- **Strengths**: Moderate power, suitable for edge
- **Weaknesses**: Limited model coverage in current tests, lower throughput
- **Best for**: Cost-sensitive edge deployments

## Precision (FP16 vs BF16)

**Finding**: Minimal difference in speed or quality between FP16 and BF16
- Caption lengths nearly identical
- Latency differences < 5% in most cases
- Success rates equivalent

**Recommendation**: Use BF16 for better numerical stability without performance penalty.

## Recommendations by Use Case

### Real-time Video Analysis
→ **Moondream2 on Thor (fp16)**
- Sub-900ms latency
- Excellent power efficiency
- Reliable performance

### Detailed Image Understanding
→ **LLaVA on Dell (bf16)**
- Rich, detailed captions
- 100% success rate
- Worth the extra latency for quality

### Batch Processing / Throughput
→ **SmolVLM2 on Dell (bf16)**
- Highest tokens/second (795 tok/s)
- Best for processing large datasets
- Great energy efficiency

### Edge Deployment (Power Constrained)
→ **Moondream2 on Thor (bf16)**
- Best efficiency (2.29 tok/s/W)
- Low latency
- Reliable

### Production Reliability
→ **Gemma3N on Dell (bf16)**
- 100% success rate
- Low CV (19.5% - very consistent)
- Good detail quality

## Usage

To regenerate these plots:

```bash
python caption_quality_analysis.py --roots dell_output orin_output thor_output --out caption_analysis
```

## Data Files

- **caption_quality_speed_report.txt**: Detailed text report with rankings
- **8 PNG plots**: Visualizations of speed and quality metrics

## Notes

- Caption length is used as a proxy for detail/quality (longer generally = more descriptive)
- Actual caption quality would require human evaluation or similarity metrics
- Success rate includes all task types, some tasks may be inherently harder
- Latency includes full end-to-end inference time including tokenization and generation
