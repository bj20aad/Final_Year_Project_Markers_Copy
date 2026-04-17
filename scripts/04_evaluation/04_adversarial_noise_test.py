import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR        = "/Users/benjoel/fypCode"
VAE_MODEL_PATH  = os.path.join(BASE_DIR, 'output', 'vae_model.pth')
TEST_FILE       = os.path.join(BASE_DIR, 'data', 'clean_mixed_test.csv')
SCALER_PATH     = os.path.join(BASE_DIR, 'output', 'scaler_vae.save')
HIDDEN_DIM      = 64
LATENT_DIM      = 8
THRESHOLD       = 0.2
NOISE_LEVELS    = [0.01, 0.05, 0.1]


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


def adversarial_test():
    print("Adversarial robustness test — Gaussian noise injection\n")

    scaler      = joblib.load(SCALER_PATH)
    df          = pd.read_csv(TEST_FILE)
    attack_df   = df[df['Attack_type'] != 'Normal']
    X_raw       = attack_df.drop(columns=['Attack_type', 'Attack_label'],
                                 errors='ignore').select_dtypes(include=[np.number]).values
    X_scaled    = scaler.transform(X_raw).astype(np.float32)

    model = VAE(X_scaled.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    print(f"Attack samples tested : {len(X_scaled):,}\n")
    print(f"{'Noise level':<14} {'Mean MSE':<14} {'Detected':<12} {'Detection rate'}")
    print("-" * 56)

    for noise in NOISE_LEVELS:
        noisy_x = X_scaled + np.random.normal(0, noise, X_scaled.shape).astype(np.float32)
        with torch.no_grad():
            recon = model(torch.Tensor(noisy_x))
            mse   = np.mean(np.power(noisy_x - recon.numpy(), 2), axis=1)
        detected       = (mse > THRESHOLD)
        n_detected     = detected.sum()
        detection_rate = n_detected / len(detected)
        print(f"{noise:<14} {np.mean(mse):<14.4f} {n_detected:<12,} {detection_rate:.4f}")


if __name__ == "__main__":
    adversarial_test()
