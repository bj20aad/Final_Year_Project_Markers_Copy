import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import os

# --- CONFIGURATION ---
TRAIN_FILE  = 'data/clean_benign_train.csv'
MODEL_PATH  = 'output/iforest_model.pkl'
SCALER_PATH = 'output/scaler_iforest.pkl'


def train_iforest():
    if not os.path.exists(TRAIN_FILE):
        print(f"ERROR: {TRAIN_FILE} not found.")
        return

    print("Loading training data...")
    df = pd.read_csv(TRAIN_FILE)
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(df)

    print("Training Isolation Forest...")
    clf = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1, random_state=42)
    clf.fit(X_train)

    os.makedirs('output', exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Scaler saved: {SCALER_PATH}")


if __name__ == "__main__":
    train_iforest()