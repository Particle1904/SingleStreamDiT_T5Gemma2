import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from config import Config

def plot_loss(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df['Epoch'], df['Loss'], label='Step Loss', s=5, alpha=0.6, color='#66b3ff')

    ax.set_title(os.path.basename(csv_path).replace('_log.csv', ' Loss Curve'), fontsize=16, color='white')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (L1)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(colors='gray')

    output_path = "loss_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSuccessfully generated loss curve: {output_path}")

if __name__ == "__main__":
    LOG_FILE = Config.log_file
    plot_loss(LOG_FILE)