import pandas as pd
import matplotlib.pyplot as plt
import glob


def visualise(model,seed):
    # Get list of CSV files (adjust the path as needed)
    file_paths = glob.glob(f"../evaluations/varying_thresholds_{model}_{seed}_*.csv")
    # Initialize storage
    data = []
    file_names = []

    for file in file_paths:
        df = pd.read_csv(file)  # Read CSV
        second_row = df.iloc[2]  # Get third row (@k=3)
        data.append(second_row.values)  # Store row values
        file_names.append(file.replace(".csv","").split('_')[-1])  # Extract filename without full path

    # Convert to DataFrame for easier plotting
    data_df = pd.DataFrame(data, columns=df.columns, index=file_names)

    # Plot bar chart
    ax = data_df.T.plot(kind='bar', figsize=(10, 8))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.5f", fontsize=10, padding=3, rotation=90)
    # Labels and title
    plt.ylim(0, 1)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Comparison of learning rates for model {model}@seed:{seed}')
    plt.xticks(rotation=45)
    plt.legend(title="Learning rate", loc="lower right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{model}_seed_{seed}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
