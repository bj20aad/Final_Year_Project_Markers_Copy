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

    scaler       = joblib.load(SCALER_PATH)
    df           = pd.read_csv(TEST_FILE)
    attack_packet = df[df['Attack_type'] != 'Normal'].sample(1)
    X_raw        = attack_packet.drop(columns=['Attack_type', 'Attack_label'],
                                      errors='ignore').values
    X_scaled     = scaler.transform(X_raw).astype(np.float32)

    model = VAE(X_scaled.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    print(f"{'Noise level':<14} {'MSE score':<14} {'Result'}")
    print("-" * 42)
    for noise in NOISE_LEVELS:
        noisy_x = X_scaled + np.random.normal(0, noise, X_scaled.shape).astype(np.float32)
        with torch.no_grad():
            recon = model(torch.Tensor(noisy_x))
            mse   = np.mean(np.power(noisy_x - recon.numpy(), 2))
        result = "Detected" if mse > THRESHOLD else "Evasion success"
        print(f"{noise:<14} {mse:<14.4f} {result}")


if __name__ == "__main__":
    adversarial_test()