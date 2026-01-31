from paise_edge_inference_plots import extract_comprehensive_metrics
import numpy as np

df = extract_comprehensive_metrics()

# Get Dell Gemma3N
dell_gemma = df[(df['device'] == 'Dell Pro Max GB10') & (df['model'] == 'gemma3n')]
print("Dell Gemma3N raw data:")
print(dell_gemma[['power_watts', 'latency_ms', 'gen_len']])

# Calculate energy
if len(dell_gemma) > 0:
    row = dell_gemma.iloc[0]
    joules_total = (row['latency_ms'] / 1000) * row['power_watts']
    joules_per_token = joules_total / row['gen_len'] if row['gen_len'] > 0 else joules_total
    print(f"\nCalculated energy:")
    print(f"  joules_total = ({row['latency_ms']} / 1000) * {row['power_watts']} = {joules_total}")
    print(f"  joules_per_token = {joules_total} / {row['gen_len']} = {joules_per_token}")

# Now check what the plot sees
df['joules_total'] = (df['latency_ms'] / 1000) * df['power_watts']
df['joules_per_token'] = df.apply(
    lambda row: (row['joules_total'] / row['gen_len']) if row['gen_len'] > 0 else row['joules_total'],
    axis=1
)
df['joules_per_token'] = df['joules_per_token'].replace([np.inf, -np.inf], np.nan)

energy_filtered = df[df['power_watts'] > 0]
energy_filtered = energy_filtered.dropna(subset=['joules_per_token'])
energy_filtered = energy_filtered[energy_filtered['joules_per_token'] > 0]

dell_gemma_filtered = energy_filtered[(energy_filtered['device'] == 'Dell Pro Max GB10') & (energy_filtered['model'] == 'gemma3n')]
print(f"\nAfter filtering: {len(dell_gemma_filtered)} rows")
if len(dell_gemma_filtered) > 0:
    print(dell_gemma_filtered[['joules_per_token']])
