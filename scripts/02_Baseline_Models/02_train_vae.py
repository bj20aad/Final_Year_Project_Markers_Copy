import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- CONFIGURATION ---
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
BASE_DIR        = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TRAIN_FILE      = os.path.join(BASE_DIR, 'data', 'clean_benign_train.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'output', 'vae_baseline.pth')
ONNX_SAVE_PATH  = os.path.join(BASE_DIR, 'output', 'vae_baseline.onnx')
SCALER_PATH     = os.path.join(BASE_DIR, 'output', 'scaler_vae.save')

# Baseline hyperparameters (Phase 1 — unoptimised 32x4 architecture)
LATENT_DIM = 4
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3


# --- MODEL DEFINITION ---
class BaselineVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=4):
        super(BaselineVAE, self).__init__()
        self.enc1      = nn.Linear(input_dim, 32)
        self.enc2      = nn.Linear(32, 16)
        self.z_mean    = nn.Linear(16, latent_dim)
        self.z_log_var = nn.Linear(16, latent_dim)
        self.dec1      = nn.Linear(latent_dim, 16)
        self.dec2      = nn.Linear(16, 32)
        self.out       = nn.Linear(32, input_dim)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.relu(self.dec1(z))
        h = self.relu(self.dec2(h))
        return self.sigmoid(self.out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train_baseline_vae():
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        return

    # Load and scale training data
    print("Loading training data...")
    df      = pd.read_csv(TRAIN_FILE)
    scaler  = MinMaxScaler()
    x_train = scaler.fit_transform(df.values)
    joblib.dump(scaler, SCALER_PATH)

    input_dim  = x_train.shape[1]
    dataloader = DataLoader(
        TensorDataset(torch.Tensor(x_train)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Initialise model and optimiser
    model     = BaselineVAE(input_dim, LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print(f"Training Baseline VAE ({EPOCHS} epochs)...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for (data,) in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            mse_loss = nn.functional.mse_loss(recon, data, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss     = mse_loss + kld_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss / len(dataloader.dataset):.4f}")

    # Save weights and export ONNX
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Weights saved: {MODEL_SAVE_PATH}")

    dummy_input = torch.randn(1, input_dim)
    torch.onnx.export(model, dummy_input, ONNX_SAVE_PATH,
                      input_names=['input'], output_names=['output'])
    print(f"ONNX saved: {ONNX_SAVE_PATH}")


if __name__ == "__main__":
    train_baseline_vae()