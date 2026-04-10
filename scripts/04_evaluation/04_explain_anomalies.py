import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR       = "/Users/benjoel/fypCode"
VAE_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'vae_model.pth')
TEST_FILE      = os.path.join(BASE_DIR, 'data', 'clean_mixed_test.csv')
SCALER_PATH    = os.path.join(BASE_DIR, 'output', 'scaler_vae.save')
HIDDEN_DIM     = 64
LATENT_DIM     = 8
N_SAMPLES      = 3   # number of attack samples to explain
TOP_N_FEATURES = 3   # top contributing features to show per sample


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
        h = self.relu(self.enc1(x))
        return self.sigmoid(self.out(self.relu(self.dec1(self.z_mean(h)))))


def explain_anomalies():
    df      = pd.read_csv(TEST_FILE)
    attacks = df[df['Attack_type'] != 'Normal'].sample(N_SAMPLES)
    X       = attacks.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
                     .select_dtypes(include=[np.number])

    scaler        = joblib.load(SCALER_PATH)
    X_scaled_np   = scaler.transform(X.values).astype(np.float32)
    X_scaled_t    = torch.Tensor(X_scaled_np)

    model = VAE(X_scaled_np.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        recon    = model(X_scaled_t).numpy()
        mse_feat = np.power(X_scaled_np - recon, 2)

    for i in range(len(attacks)):
        print(f"Attack type : {attacks.iloc[i]['Attack_type']}")
        top_idx = mse_feat[i].argsort()[-TOP_N_FEATURES:][::-1]
        for idx in top_idx:
            print(f"  {X.columns[idx]:<35} error: {mse_feat[i][idx]:.4f}")
        print()


if __name__ == "__main__":
    explain_anomalies()