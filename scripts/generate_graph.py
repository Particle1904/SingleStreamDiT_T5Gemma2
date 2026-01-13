import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
# Import config from the folder above.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
os.chdir(ROOT)
from config import Config

def plot_smoothed_stats(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    plt.style.use('dark_background')
    
    smoothing_span = 10

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    
    df['Loss_Smooth'] = df['Loss'].ewm(span=smoothing_span).mean()

    ax_loss.plot(df['Global_Step'], df['Loss'], color='#66b3ff', alpha=0.15, linewidth=1)
    ax_loss.plot(df['Global_Step'], df['Loss_Smooth'], color='#66b3ff', linewidth=2, label='Smoothed Loss')

    ax_loss.set_title('Training Loss (Smoothed)', fontsize=16, color='white', pad=20)
    ax_loss.set_xlabel('Global Step', fontsize=12)
    ax_loss.set_ylabel('Loss Value', fontsize=12)
    ax_loss.grid(True, linestyle='--', alpha=0.2)
    ax_loss.legend()
    
    loss_out = "loss_curve.png"
    fig_loss.savefig(loss_out, dpi=300, bbox_inches='tight')
    plt.close(fig_loss)
    print(f"Generated Smoothed Loss: {loss_out}")

    if 'Fourier_Gate_Avg' in df.columns:
        fig_gate, ax_gate = plt.subplots(figsize=(10, 6))
        
        df['Gate_Smooth'] = df['Fourier_Gate_Avg'].ewm(span=20).mean()

        ax_gate.plot(df['Global_Step'], df['Fourier_Gate_Avg'], color='#ff9999', alpha=0.2, linewidth=1)
        ax_gate.plot(df['Global_Step'], df['Gate_Smooth'], color='#ff9999', linewidth=2.5, label='Gate Trend')
        
        ax_gate.set_title('Fourier Filter Gate Evolution', fontsize=16, color='white', pad=20)
        ax_gate.set_xlabel('Global Step', fontsize=12)
        ax_gate.set_ylabel('Avg Gate Value', fontsize=12)
        ax_gate.grid(True, linestyle='--', alpha=0.2)
        ax_gate.legend()

        gate_out = "fourier_gate.png"
        fig_gate.savefig(gate_out, dpi=300, bbox_inches='tight')
        plt.close(fig_gate)
        print(f"Generated Smoothed Gate: {gate_out}")
    else:
        print("Warning: 'Fourier_Gate_Avg' column not found.")

if __name__ == "__main__":
    LOG_FILE = Config.log_file
    plot_smoothed_stats(LOG_FILE)