#First attempt - not used for results in report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION (Optuna Trial 11 best parameters) ---
TRAIN_FILE    = 'data/clean_benign_train.csv'
MODEL_PATH    = 'output/vae_model.pth'
SCALER_PATH   = 'output/scaler_vae.save'
HIDDEN_DIM    = 64
LATENT_DIM    = 8
LEARNING_RATE = 0.00924
EPOCHS        = 10
BATCH_SIZE    = 64


# --- OPTIMISED VAE ARCHITECTURE (64x8) ---
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.enc1      = nn.Linear(input_dim, HIDDEN_DIM)
        self.z_mean    = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.z_log_var = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.dec1      = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.out       = nn.Linear(HIDDEN_DIM, input_dim)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.enc1(x))
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.sigmoid(self.out(self.relu(self.dec1(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def train_optimised_vae():
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        return

    print(f"Training optimised VAE (hidden: {HIDDEN_DIM}, latent: {LATENT_DIM}, lr: {LEARNING_RATE})")

    scaler   = joblib.load(SCALER_PATH)
    df       = pd.read_csv(TRAIN_FILE)
    X_scaled = scaler.transform(df.select_dtypes(include=[np.number]).values).astype(np.float32)

    input_dim  = X_scaled.shape[1]
    dataloader = DataLoader(
        TensorDataset(torch.Tensor(X_scaled)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model     = VAE(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for (x,) in dataloader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            mse_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss     = mse_loss + kld_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS} — loss: {epoch_loss / len(dataloader.dataset):.4f}")

    os.makedirs('output', exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    train_optimised_vae()
