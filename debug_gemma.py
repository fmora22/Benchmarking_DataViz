import json
from pathlib import Path
import numpy as np

def extract_power_watts(run):
    if 'power_stats' not in run or run['power_stats'] is None:
        return 0
    
    ps = run['power_stats']
    if not isinstance(ps, dict):
        return 0
    
    if 'power_watts_avg' in ps and ps['power_watts_avg'] and ps['power_watts_avg'] > 0:
        return ps['power_watts_avg']
    
    if 'power_watts_samples' in ps:
        samples = ps['power_watts_samples']
        if samples:
            return np.mean(samples)
    
    return 0

# Test on Gemma3N Dell
device_dir = Path('dell_output')
model_name = 'gemma3n'
run_dir = 'test_bf16_500img_20260108_231557'
runs_file = device_dir / model_name / run_dir / 'runs.jsonl'

with open(runs_file) as f:
    runs = [json.loads(line) for line in f]

successful = [r for r in runs if r.get('error') is None]
print(f"Successful runs: {len(successful)}")

powers = []
for r in successful:
    power = extract_power_watts(r)
    if power > 0:
        powers.append(power)

print(f"Powers collected: {len(powers)}")
print(f"Mean power: {np.mean(powers) if powers else 'None'}")
print(f"Std power: {np.std(powers) if powers else 'None'}")
