import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os

os.makedirs('docs/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-muted')

INPUT_DIM = 39


# -------------------------------------------------------
# Model Definitions (for parameter count verification)
# -------------------------------------------------------
class ActualEdgeVAE(nn.Module):
    def __init__(self):
        super(ActualEdgeVAE, self).__init__()
        self.enc1   = nn.Linear(INPUT_DIM, 64)
        self.z_mean = nn.Linear(64, 8)
        self.z_log_var = nn.Linear(64, 8)
        self.dec1   = nn.Linear(8, 64)
        self.out    = nn.Linear(64, INPUT_DIM)

    def forward(self, x):
        h = torch.relu(self.enc1(x))
        return self.out(torch.relu(self.dec1(self.z_mean(h))))


class CloudLSTM(nn.Module):
    def __init__(self):
        super(CloudLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_DIM, 128, num_layers=2, batch_first=True)
        self.fc   = nn.Linear(128, INPUT_DIM)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(1))
        return self.fc(out[:, -1, :])


# -------------------------------------------------------
# Figure 4.15 — Edge VAE vs Cloud LSTM Benchmarking
# -------------------------------------------------------
def generate_edge_vs_cloud():
    edge_model  = ActualEdgeVAE()
    cloud_model = CloudLSTM()

    edge_params  = sum(p.numel() for p in edge_model.parameters())
    cloud_params = sum(p.numel() for p in cloud_model.parameters())

    # Measured benchmark values
    edge_kb  = 26.21
    cloud_kb = 873.65
    edge_lat  = 0.1058
    cloud_lat = 0.6387

    print(f"\n{'Metric':<20} | {'Edge VAE (64x8)':<18} | {'Cloud LSTM':<18}")
    print("-" * 62)
    print(f"{'Parameters':<20} | {edge_params:<18} | {cloud_params:<18}")
    print(f"{'Model Size':<20} | {edge_kb:<15} KB | {cloud_kb:<15} KB")
    print(f"{'Latency':<20} | {edge_lat:<15} ms | {cloud_lat:<15} ms")
    print(f"\nRatios: Size {cloud_kb/edge_kb:.1f}x | Speed {cloud_lat/edge_lat:.1f}x")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    models = ['Edge VAE', 'Cloud LSTM']

    # Chart A: Model Size (Log Scale)
    bars1 = ax1.bar(models, [edge_kb, cloud_kb], color=['#2ecc71', '#34495e'], alpha=0.85)
    ax1.set_yscale('log')
    ax1.set_title('Model Size (KB) - Log Scale')
    ax1.set_ylabel('Storage (KB)')
    for i, v in enumerate([edge_kb, cloud_kb]):
        ax1.text(i, v, f'{v:.2f} KB', ha='center', va='bottom', fontweight='bold')

    # Chart B: Inference Latency
    bars2 = ax2.bar(models, [edge_lat, cloud_lat], color=['#27ae60', '#2c3e50'], alpha=0.85)
    ax2.set_title('Inference Latency (ms)')
    ax2.set_ylabel('Latency (ms)')
    for i, v in enumerate([edge_lat, cloud_lat]):
        ax2.text(i, v, f'{v:.4f} ms', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Final Benchmarking (Edge VAE vs. Cloud LSTM)', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig('docs/plots/edge_vs_cloud_log.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/edge_vs_cloud_log.png")


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Phase 4 Figures ---")
    generate_edge_vs_cloud()
    print("--- Done: 1 figure saved to docs/plots/ ---")