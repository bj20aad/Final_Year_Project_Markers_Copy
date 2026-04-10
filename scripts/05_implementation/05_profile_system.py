import time
import psutil
import os
import onnxruntime as ort
import numpy as np
import pandas as pd
import joblib

# --- CONFIGURATION ---
ONNX_PATH   = 'output/vae_model.onnx'
TEST_FILE   = 'data/clean_mixed_test.csv'
SCALER_PATH = 'output/scaler_vae.save'
N_PACKETS   = 1000


def measure_performance():
    if not os.path.exists(ONNX_PATH):
        print(f"ERROR: {ONNX_PATH} not found — run 03_optimised_vae.py first.")
        return

    process    = psutil.Process(os.getpid())
    session    = ort.InferenceSession(ONNX_PATH)
    scaler     = joblib.load(SCALER_PATH)
    input_name = session.get_inputs()[0].name

    df       = pd.read_csv(TEST_FILE).sample(N_PACKETS, random_state=42)
    X        = df.drop(columns=['Attack_type', 'Attack_label'], errors='ignore') \
                 .select_dtypes(include=[np.number])
    X_scaled = scaler.transform(X).astype(np.float32)

    print(f"Profiling inference across {N_PACKETS} packets...")
    latencies = []
    for i in range(len(X_scaled)):
        packet = X_scaled[i:i+1]
        start  = time.time()
        session.run(None, {input_name: packet})
        latencies.append((time.time() - start) * 1000)

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    ram_usage   = process.memory_info().rss / 1024 / 1024

    print(f"\n{'Metric':<25} {'Value'}")
    print("-" * 40)
    print(f"{'Average latency':<25} {avg_latency:.4f} ms")
    print(f"{'P95 latency':<25} {p95_latency:.4f} ms")
    print(f"{'RAM consumption':<25} {ram_usage:.2f} MB")


if __name__ == "__main__":
    measure_performance()