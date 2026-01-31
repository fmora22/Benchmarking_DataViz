from paise_edge_inference_plots import extract_comprehensive_metrics
import numpy as np

df = extract_comprehensive_metrics()
print("Original DataFrame length:", len(df))

# Gemma3N rows
gemma_rows = df[df['model'] == 'gemma3n']
print(f"\nGemma3N rows (all): {len(gemma_rows)}")
print(gemma_rows[['device', 'precision', 'latency_ms', 'gen_len', 'power_watts']])

# Calculate energy per token for Gemma3N
df['joules_per_token'] = (df['latency_ms'] / 1000 * df['power_watts']) / (df['gen_len'] + 0.1)
df['joules_per_token'] = df['joules_per_token'].replace([np.inf, -np.inf], np.nan)

gemma_energy = df[df['model'] == 'gemma3n'][['device', 'precision', 'latency_ms', 'gen_len', 'power_watts', 'joules_per_token']]
print(f"\nGemma3N with energy calculation:")
print(gemma_energy)

# After filtering
energy_filtered = df.dropna(subset=['joules_per_token'])
energy_filtered = energy_filtered[energy_filtered['joules_per_token'] > 0]

gemma_filtered = energy_filtered[energy_filtered['model'] == 'gemma3n']
print(f"\nGemma3N after filtering: {len(gemma_filtered)}")
if len(gemma_filtered) > 0:
    print(gemma_filtered[['device', 'precision', 'joules_per_token']])
