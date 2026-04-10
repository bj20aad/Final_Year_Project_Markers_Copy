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
        h = self.relu(self.enc1(x))
        return self.sigmoid(self.out(self.relu(self.dec1(self.z_mean(h)))))


def analyze_missed_attacks():
    df    = pd.read_csv(TEST_FILE)
    X_raw = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
               .select_dtypes(include=[np.number])

    scaler   = joblib.load(SCALER_PATH)
    X_scaled = torch.Tensor(scaler.transform(X_raw.values).astype(np.float32))

    model = VAE(X_scaled.shape[1])
    model.load_state_dict(torch.load(VAE_MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        recon = model(X_scaled).numpy()
        mse   = np.mean(np.power(X_scaled.numpy() - recon, 2), axis=1)

    # False negatives: actual attacks whose reconstruction error fell below the threshold
    is_attack = (df['Attack_type'] != 'Normal')
    is_missed = is_attack & (mse <= THRESHOLD)
    missed_df = df[is_missed]

    print(f"Missed attacks: {len(missed_df):,} of {is_attack.sum():,} total attack samples\n")

    if not missed_df.empty:
        counts = missed_df['Attack_type'].value_counts()
        total  = len(missed_df)
        print(f"{'Attack Type':<30} {'Count':>8}  {'%':>6}")
        print("-" * 48)
        for attack_type, count in counts.items():
            print(f"{attack_type:<30} {count:>8,}  {count/total*100:>5.1f}%")
    else:
        print("No attacks missed at this threshold.")


if __name__ == "__main__":
    analyze_missed_attacks()