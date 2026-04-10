import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import f1_score, accuracy_score

# --- CONFIGURATION ---
BASE_DIR       = "/Users/benjoel/fypCode"
TEST_FILE      = os.path.join(BASE_DIR, 'data', 'clean_mixed_test.csv')
VAE_MODEL_PATH = os.path.join(BASE_DIR, 'output', 'vae_model.pth')
SCALER_PATH    = os.path.join(BASE_DIR, 'output', 'scaler_vae.save')
THRESHOLDS     = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


# --- OPTIMISED VAE ARCHITECTURE (64x8) ---
class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.enc1      = nn.Linear(input_dim, 64)
        self.z_mean    = nn.Linear(64, 8)
        self.z_log_var = nn.Linear(64, 8)
        self.dec1      = nn.Linear(8, 64)
        self.out       = nn.Linear(64, input_dim)
        self.relu      = nn.ReLU()
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        h = self.relu(self.enc1(x))
        return self.sigmoid(self.out(self.relu(self.dec1(self.z_mean(h)))))


def run_threshold_search():
    print("Loading data and model...")
    df     = pd.read_csv(TEST_FILE)
    y_true = df['Attack_type'].apply(lambda x: 0 if x == 'Normal' else 1).values
    X_raw  = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
               .select_dtypes(include=[np.number])

    scaler   = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X_raw.values).astype(np.float32)

    model = VAE(X_scaled.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        recon = model(torch.Tensor(X_scaled))
        mse   = np.mean(np.power(X_scaled - recon.numpy(), 2), axis=1)

    print("\nSearching for optimal classification threshold...")
    print(f"{'Threshold':<12} {'F1':<10} {'Accuracy':<10}")
    print("-" * 32)

    best_f1, best_t = 0, 0
    for t in THRESHOLDS:
        y_pred     = (mse > t).astype(int)
        current_f1 = f1_score(y_true, y_pred)
        acc        = accuracy_score(y_true, y_pred)
        print(f"{t:<12.3f} {current_f1:<10.4f} {acc:<10.4f}")
        if current_f1 > best_f1:
            best_f1, best_t = current_f1, t

    print(f"\nBest threshold: {best_t}  (F1: {best_f1:.4f})")


if __name__ == "__main__":
    run_threshold_search()