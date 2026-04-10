import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR       = "/Users/benjoel/fypCode"
TEST_FILE      = os.path.join(BASE_DIR, 'data', 'clean_mixed_test.csv')
VAE_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'vae_model.pth')
SCALER_PATH    = os.path.join(BASE_DIR, 'output', 'scaler_vae.save')
HIDDEN_DIM     = 64
LATENT_DIM     = 8
THRESHOLD      = 0.2


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

    def forward(self, x):
        h     = self.relu(self.enc1(x))
        mu    = self.z_mean(h)
        h_dec = self.relu(self.dec1(mu))
        return self.sigmoid(self.out(h_dec))


def diagnose_error_distribution():
    if not os.path.exists(VAE_MODEL_PATH):
        print("ERROR: Model not found — run 03_optimised_vae.py first.")
        return

    df     = pd.read_csv(TEST_FILE)
    y_true = df['Attack_type'].apply(lambda x: 0 if x == 'Normal' else 1).values
    X      = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
               .select_dtypes(include=[np.number])

    scaler   = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X.values).astype(np.float32)

    model = VAE(X_scaled.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        recon = model(torch.Tensor(X_scaled))
        mse   = np.mean(np.power(X_scaled - recon.numpy(), 2), axis=1)

    avg_normal = np.mean(mse[y_true == 0])
    avg_attack = np.mean(mse[y_true == 1])

    print(f"Average reconstruction error — Normal : {avg_normal:.6f}")
    print(f"Average reconstruction error — Attack : {avg_attack:.6f}")

    if avg_attack > THRESHOLD:
        print(f"\nAverage attack error exceeds threshold ({THRESHOLD}) — model is separating classes.")
    else:
        print(f"\nAverage attack error ({avg_attack:.4f}) is below threshold — check model or threshold.")


if __name__ == "__main__":
    diagnose_error_distribution()