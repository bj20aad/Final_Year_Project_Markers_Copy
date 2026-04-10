import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIGURATION ---
BASE_DIR      = "/Users/benjoel/fypCode"
SCRIPT_SUBDIR = os.path.join(BASE_DIR, "scripts", "03_optimisation")
TRAIN_FILE    = os.path.join(BASE_DIR, "data", "clean_benign_train.csv")
DB_PATH       = f"sqlite:///{os.path.join(SCRIPT_SUBDIR, 'optuna_study.db')}"
N_TRIALS      = 20
BATCH_SIZE    = 64
EPOCHS_PER_TRIAL = 2


# --- VAE MODEL (parameterised for hyperparameter search) ---
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.enc1      = nn.Linear(input_dim, hidden_dim)
        self.z_mean    = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        self.dec1      = nn.Linear(latent_dim, hidden_dim)
        self.out       = nn.Linear(hidden_dim, input_dim)
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
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def build_dataloader(train_file, batch_size):
    df        = pd.read_csv(train_file)
    scaler    = MinMaxScaler()
    x_train   = scaler.fit_transform(df.values)
    input_dim = x_train.shape[1]
    loader    = DataLoader(
        TensorDataset(torch.Tensor(x_train)),
        batch_size=batch_size,
        shuffle=True
    )
    return loader, input_dim


def make_objective(dataloader, input_dim):
    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 16, 64)
        latent_dim = trial.suggest_int("latent_dim", 2, 10)
        lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        model     = VAE(input_dim, hidden_dim, latent_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()

        total_loss = 0
        for _ in range(EPOCHS_PER_TRIAL):
            for (data,) in dataloader:
                optimizer.zero_grad()
                recon, mu, logvar = model(data)
                mse = nn.functional.mse_loss(recon, data, reduction='sum')
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = mse + kld
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return total_loss / len(dataloader.dataset)
    return objective


def run_optimisation():
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        return

    os.makedirs(SCRIPT_SUBDIR, exist_ok=True)

    print(f"Data source     : {TRAIN_FILE}")
    print(f"Database target : {DB_PATH}")

    dataloader, input_dim = build_dataloader(TRAIN_FILE, BATCH_SIZE)

    study = optuna.create_study(
        study_name="vae_architecture_study",
        storage=DB_PATH,
        load_if_exists=True,
        direction="minimize"
    )

    print(f"\nStarting Bayesian optimisation ({N_TRIALS} trials)...")
    study.optimize(make_objective(dataloader, input_dim),
                   n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\nBest trial loss : {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"\nDatabase saved  : {os.path.join(SCRIPT_SUBDIR, 'optuna_study.db')}")


if __name__ == "__main__":
    run_optimisation()