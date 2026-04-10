import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIGURATION ---
TRAIN_FILE  = 'data/clean_benign_train.csv'
MODEL_PATH  = 'output/kmeans_model.pkl'
SCALER_PATH = 'output/scaler_kmeans.pkl'
K_CLUSTERS  = 5


def train_kmeans():
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        return

    print("Loading training data...")
    df = pd.read_csv(TRAIN_FILE)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(df)

    print(f"Training K-Means with {K_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42)
    kmeans.fit(X_train)

    os.makedirs('output', exist_ok=True)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")


if __name__ == "__main__":
    train_kmeans()