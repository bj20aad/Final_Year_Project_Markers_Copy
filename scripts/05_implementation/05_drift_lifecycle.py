import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import os

# --- CONFIGURATION ---
BASE_DIR  = '/Users/benjoel/fypCode'
TEST_DATA = os.path.join(BASE_DIR, 'data/clean_mixed_test.csv')

# Preferred continuous features for KS testing (falls back to first numeric column)
CANDIDATE_FEATURES = ['frame.len', 'tcp.len', 'ip.len', 'tcp.seq']


def run_drift_audit():
    df = pd.read_csv(TEST_DATA)

    # Select the first available candidate feature, or fall back to the first numeric column
    feature = next(
        (f for f in CANDIDATE_FEATURES if f in df.columns),
        df.select_dtypes(include=[np.number]).columns[0]
    )
    baseline = df[feature].dropna().values
    print(f"KS drift audit — feature: [{feature}]\n")

    # Stage 1: Self-comparison (expected p=1.0 — confirms no internal bias)
    _, p_self = ks_2samp(baseline, baseline)
    print(f"[1] Self-validation          p = {p_self:.2f}  (expected: 1.0)")

    # Stage 2: Split-half consistency (natural variance within the test set)
    half = len(baseline) // 2
    _, p_stable = ks_2samp(baseline[:half], baseline[half:])
    print(f"[2] Split-half consistency   p = {p_stable:.2f}  (expected: > 0.05)")

    # Stage 3: Simulated drift (shift distribution by 2 standard deviations)
    drifted = baseline + (baseline.std() * 2)
    d_drift, p_drift = ks_2samp(baseline, drifted)
    print(f"[3] Adversarial drift        p = {p_drift:.4e}  KS stat = {d_drift:.4f}")

    if p_drift < 0.05:
        print("    >>> Drift detected — monitoring pipeline functioning correctly.")
    else:
        print("    >>> WARNING: Simulated drift was not detected — check feature variance.")


if __name__ == "__main__":
    run_drift_audit()