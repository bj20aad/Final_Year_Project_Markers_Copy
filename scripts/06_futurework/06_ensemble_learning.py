import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

os.makedirs('docs/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-muted')

# --- Paths ---
BASE_DIR        = '/Users/benjoel/fypCode'
TEST_DATA       = os.path.join(BASE_DIR, 'data/clean_mixed_test.csv')
VAE_PATH        = os.path.join(BASE_DIR, 'output/vae_model.pth')
IF_PATH         = os.path.join(BASE_DIR, 'output/iforest_model.pkl')
VAE_SCALER_PATH = os.path.join(BASE_DIR, 'output/scaler_vae.save')
IF_SCALER_PATH  = os.path.join(BASE_DIR, 'output/scaler_iforest.pkl')


# --- VAE Architecture (64x8) ---
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


# -------------------------------------------------------
# Figure 4.16 — Ensemble Operating Curve
# -------------------------------------------------------
def generate_ensemble_curve():
    # 1. Load data
    print("Loading test data...")
    df     = pd.read_csv(TEST_DATA)
    y_true = (df['Attack_type'] != 'Normal').astype(int).values
    X_test = df.drop(columns=['Attack_type', 'Attack_label', 'Attack_type.1'],
                     errors='ignore').values

    # 2. Load models and scalers
    print(f"Loading models (input dim: {X_test.shape[1]})...")
    vae_scaler = joblib.load(VAE_SCALER_PATH)
    if_scaler  = joblib.load(IF_SCALER_PATH)

    vae_model = VAE()
    vae_model.load_state_dict(torch.load(VAE_PATH, map_location='cpu'))
    vae_model.eval()
    if_model = joblib.load(IF_PATH)

    # 3. VAE predictions
    X_vae = torch.FloatTensor(vae_scaler.transform(X_test))
    with torch.no_grad():
        vae_errors = torch.mean((X_vae - vae_model(X_vae)) ** 2, dim=1).numpy()

    # 4. Isolation Forest predictions
    X_if   = if_scaler.transform(X_test)
    if_raw = if_model.predict(X_if)

    vae_pred = (vae_errors > 0.2).astype(int)
    if_pred  = (if_raw == -1).astype(int)

    or_pred  = np.logical_or(vae_pred, if_pred).astype(int)
    and_pred = np.logical_and(vae_pred, if_pred).astype(int)

    # 5. Weighted voting curve
    w_vae, w_if = 0.63, 0.49

    vae_clean = np.clip(np.nan_to_num(vae_errors, nan=0.0, posinf=10.0), 0, 10)
    if_clean  = np.nan_to_num(-if_model.decision_function(X_if), nan=0.0)

    mms_v = MinMaxScaler(); mms_i = MinMaxScaler()
    norm_vae = mms_v.fit_transform(vae_clean.reshape(-1, 1)).flatten()
    norm_if  = mms_i.fit_transform(if_clean.reshape(-1, 1)).flatten()
    weighted_score = (w_vae * norm_vae) + (w_if * norm_if)

    p_curve, r_curve = [], []
    best_f1, best_p, best_r = 0, 0, 0
    for t in np.linspace(weighted_score.min(), weighted_score.max(), 100):
        w_pred = (weighted_score >= t).astype(int)
        p = precision_score(y_true, w_pred, zero_division=0)
        r = recall_score(y_true, w_pred)
        f = f1_score(y_true, w_pred, zero_division=0)
        p_curve.append(p); r_curve.append(r)
        if f > best_f1:
            best_f1, best_p, best_r = f, p, r

    # 6. Print results table
    print(f"\n{'Strategy':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 58)
    for name, pred in [("OR-Logic", or_pred), ("AND-Logic", and_pred)]:
        p = precision_score(y_true, pred)
        r = recall_score(y_true, pred)
        f = f1_score(y_true, pred)
        print(f"{name:<20} | {p:<10.2f} | {r:<10.2f} | {f:<10.2f}")
    print(f"{'Weighted Voting':<20} | {best_p:<10.2f} | {best_r:<10.2f} | {best_f1:<10.2f}")

    # 7. Plot
    plt.figure(figsize=(9, 6))
    plt.plot(r_curve, p_curve, color='#8e44ad', lw=3,
             label=f'Weighted Hybrid (wVAE={w_vae})')
    plt.scatter(recall_score(y_true, or_pred),  precision_score(y_true, or_pred),
                color='#e74c3c', s=100, label='OR-Logic')
    plt.scatter(recall_score(y_true, and_pred), precision_score(y_true, and_pred),
                color='#2ecc71', s=100, label='AND-Logic')
    plt.title('Ensemble Operating Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('docs/plots/ensemble_pr_curve_real.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/ensemble_pr_curve_real.png")


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Section 4.5 Figures ---")
    generate_ensemble_curve()
    print("--- Done: 1 figure saved to docs/plots/ ---")