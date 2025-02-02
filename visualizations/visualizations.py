import pandas as pd
import matplotlib.pyplot as plt

# Load the varying threshold CSV files
models = ['auto', 'concat', 'encode', 'GRU4REC', 'GRU4REC_concat', 'popular', 'random']
data = {}

for model in models:
    try:
        data[model] = pd.read_csv(f'../statistical_test/varying_thresholds_{model}.csv')
    except FileNotFoundError:
        print(f"Warning: File 'varying_thresholds_{model}.csv' not found. Skipping this model.")

# Plot HR@k
plt.figure(figsize=(8, 6))
for model, df in data.items():
    plt.plot(range(1, 6), df['HR'], marker='o', label=model)

plt.xlabel('k')
plt.ylabel('HR@k')
plt.title('Hit Rate at k')
plt.legend()
plt.grid(True)

# Save HR@k plot
plt.savefig('HR_at_k_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("HR@k plot saved as 'HR_at_k_plot.png'")

# Plot MRR@k
plt.figure(figsize=(8, 6))
for model, df in data.items():
    plt.plot(range(1, 6), df['MRR'], marker='o', label=model)

plt.xlabel('k')
plt.ylabel('MRR@k')
plt.title('Mean Reciprocal Rank at k')
plt.legend()
plt.grid(True)

# Save MRR@k plot
plt.savefig('MRR_at_k_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print("MRR@k plot saved as 'MRR_at_k_plot.png'")
