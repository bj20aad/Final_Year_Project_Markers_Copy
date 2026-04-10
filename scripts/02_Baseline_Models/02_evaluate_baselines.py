import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report

# --- CONFIGURATION ---
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

TEST_FILE     = DATA_DIR   / "clean_mixed_test.csv"
KMEANS_PATH   = OUTPUT_DIR / "kmeans_model.pkl"
IFOREST_PATH  = OUTPUT_DIR / "iforest_model.pkl"
VAE_BASE_PATH = OUTPUT_DIR / "vae_baseline.pth"
KMEANS_SC     = OUTPUT_DIR / "scaler_kmeans.pkl"
IFOREST_SC    = OUTPUT_DIR / "scaler_iforest.pkl"
VAE_SC        = OUTPUT_DIR / "scaler_vae.save"

TARGET_NAMES = ['Normal (0)', 'Attack (1)']


# --- BASELINE VAE ARCHITECTURE (32x4, Phase 1) ---
class BaselineVAE(nn.Module):
    def __init__(self, input_dim):
        super(BaselineVAE, self).__init__()
        self.enc1      = nn.Linear(input_dim, 32)
        self.enc2      = nn.Linear(32, 16)
        self.z_mean    = nn.Linear(16, 4)
        self.z_log_var = nn.Linear(16, 4)
        self.dec1      = nn.Linear(4, 16)
        self.dec2      = nn.Linear(16, 32)
        self.out       = nn.Linear(32, input_dim)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        h     = self.relu(self.enc1(x))
        h     = self.relu(self.enc2(h))
        mu    = self.z_mean(h)
        h_dec = self.relu(self.dec1(mu))
        h_dec = self.relu(self.dec2(h_dec))
        return self.sigmoid(self.out(h_dec))


def section(title):
    print(f"\n{'=' * 40}\n{title}\n{'=' * 40}")


def run_full_baseline_evaluation():
    if not TEST_FILE.exists():
        print(f"ERROR: {TEST_FILE} not found.")
        return

    print("Loading test data...")
    df     = pd.read_csv(TEST_FILE)
    y_true = df['Attack_type'].apply(lambda x: 0 if x == 'Normal' else 1).values
    X_raw  = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
               .select_dtypes(include=[np.number])

    # --- K-Means ---
    section("[1/3] K-Means Baseline")
    km_model  = joblib.load(KMEANS_PATH)
    km_scaler = joblib.load(KMEANS_SC)
    X_km      = km_scaler.transform(X_raw.values)
    dist      = np.min(km_model.transform(X_km), axis=1)
    y_km      = (dist > np.percentile(dist, 90)).astype(int)
    print(classification_report(y_true, y_km, target_names=TARGET_NAMES))

    # --- Isolation Forest ---
    section("[2/3] Isolation Forest Baseline")
    if_model  = joblib.load(IFOREST_PATH)
    if_scaler = joblib.load(IFOREST_SC)
    X_if      = if_scaler.transform(X_raw.values)
    y_if      = np.where(if_model.predict(X_if) == -1, 1, 0)
    print(classification_report(y_true, y_if, target_names=TARGET_NAMES))

    # --- Baseline VAE (threshold 0.05 — standard unoptimised baseline) ---
    section("[3/3] Baseline VAE (32x4)")
    vae_scaler = joblib.load(VAE_SC)
    X_vae      = vae_scaler.transform(X_raw.values).astype(np.float32)
    model      = BaselineVAE(X_vae.shape[1])
    model.load_state_dict(torch.load(VAE_BASE_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        recon = model(torch.Tensor(X_vae))
        mse   = np.mean(np.power(X_vae - recon.numpy(), 2), axis=1)
        y_vae = (mse > 0.05).astype(int)
    print(classification_report(y_true, y_vae, target_names=TARGET_NAMES))


if __name__ == "__main__":
    run_full_baseline_evaluation()