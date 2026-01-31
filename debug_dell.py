from paise_edge_inference_plots import extract_comprehensive_metrics
import numpy as np

df = extract_comprehensive_metrics()

# Calculate energy per token
df['joules_total'] = (df['latency_ms'] / 1000) * df['power_watts']
df['joules_per_token'] = df.apply(
    lambda row: (row['joules_total'] / row['gen_len']) if row['gen_len'] > 0 else row['joules_total'],
    axis=1
)
df['joules_per_token'] = df['joules_per_token'].replace([np.inf, -np.inf], np.nan)

# Filter
energy_filtered = df[df['power_watts'] > 0]
energy_filtered = energy_filtered.dropna(subset=['joules_per_token'])
energy_filtered = energy_filtered[energy_filtered['joules_per_token'] > 0]

print(f"Total rows after filtering: {len(energy_filtered)}")
print(f"\nDell rows that will be plotted: {len(energy_filtered[energy_filtered['device'] == 'Dell Pro Max GB10'])}")
print("\nDell details:")
dell_data = energy_filtered[energy_filtered['device'] == 'Dell Pro Max GB10']
for _, row in dell_data.iterrows():
    print(f"  {row['model']:15} {row['precision']:5} -> {row['joules_per_token']:8.1f} J")
