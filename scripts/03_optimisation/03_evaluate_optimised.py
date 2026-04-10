import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
BASE_DIR    = '/Users/benjoel/fypCode'
TEST_DATA   = os.path.join(BASE_DIR, 'data/clean_mixed_test.csv')
VAE_PATH    = os.path.join(BASE_DIR, 'output/vae_model.pth')
SCALER_PATH = os.path.join(BASE_DIR, 'output/scaler_vae.save')
THRESHOLD   = 0.2


# --- OPTIMISED VAE ARCHITECTURE (64x8) ---
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enc1      = nn.Linear(39, 64)
        self.z_mean    = nn.Linear(64, 8)
        self.z_log_var = nn.Linear(64, 8)
        self.dec1      = nn.Linear(8, 64)
        self.out       = nn.Linear(64, 39)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.z_mean(h), self.z_log_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return torch.sigmoid(self.out(F.relu(self.dec1(z))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar))


def evaluate_vae():
    print("Loading test data...")
    df     = pd.read_csv(TEST_DATA)
    y_true = (df['Attack_type'] != 'Normal').astype(int).values
    X_test = df.drop(columns=['Attack_type', 'Attack_label', 'Attack_type.1'],
                     errors='ignore').values

    print(f"Loading model and scaler (features: {X_test.shape[1]})...")
    scaler        = joblib.load(SCALER_PATH)
    X_scaled      = torch.FloatTensor(scaler.transform(X_test))
    vae_model     = VAE()
    vae_model.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae_model.eval()

    with torch.no_grad():
        reconstructed = vae_model(X_scaled)
        vae_errors    = torch.mean((X_scaled - reconstructed) ** 2, dim=1).numpy()

    y_pred = (vae_errors > THRESHOLD).astype(int)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Attack'], digits=4))

    # Confusion matrix breakdown for figure and prose verification
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy  = (tp + tn) / len(y_true)
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(f"{'=' * 50}")
    print(f"  Confusion Matrix Breakdown (threshold {THRESHOLD})")
    print(f"{'=' * 50}")
    print(f"  Total samples : {len(y_true):,}")
    print(f"  Normal (y=0)  : {np.sum(y_true == 0):,}")
    print(f"  Attack (y=1)  : {np.sum(y_true == 1):,}")
    print(f"  {'─' * 46}")
    print(f"  True Negatives  (TN) : {tn:,}")
    print(f"  False Positives (FP) : {fp:,}")
    print(f"  False Negatives (FN) : {fn:,}")
    print(f"  True Positives  (TP) : {tp:,}")
    print(f"  {'─' * 46}")
    print(f"  Accuracy  : {accuracy:.6f}")
    print(f"  Recall    : {recall:.6f}")
    print(f"  Precision : {precision:.6f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    evaluate_vae()