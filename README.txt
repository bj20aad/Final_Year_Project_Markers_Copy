# Optimising Edge Intelligence: Unsupervised Anomaly Detection in IoT Networks

**Author:** Benjamin Joel  
**Student ID:** 19054139  
**Supervisor:** Frank Förster  
**Module:** 6COM2018 — BSc (Hons) Computer Science Final Year Project  
**Institution:** University of Hertfordshire  
**Academic Year:** 2025–2026

---

## Overview

This repository contains the full experimental codebase for the above final year project. The project investigates the feasibility of deploying an unsupervised anomaly detection system on resource-constrained Industrial IoT (IIoT) edge hardware, without reliance on cloud infrastructure or labelled training data.

A Variational Autoencoder (VAE) was developed, optimised via Bayesian hyperparameter search using the Optuna framework, and evaluated against K-Means Clustering and Isolation Forest baselines on the Edge-IIoT-Set benchmark dataset. The optimised model (64x8 architecture, threshold 0.200) achieved 1.00 attack precision and 85.27% accuracy, with an average inference latency of 0.0198ms — 2,525 times below the 50ms project target.

---

## Repository Structure

```
.
├── README.md
├── optuna_study.db                        # Root-level Optuna study backup
│
├── output/                                # Trained model weights and scalers
│   ├── kmeans_model.pkl
│   ├── iforest_model.pkl
│   ├── vae_baseline.pth                   # Baseline VAE (32x4) weights
│   ├── vae_baseline.onnx
│   ├── vae_model.pth                      # Optimised VAE (64x8) weights
│   ├── vae_model_tuned.pth
│   ├── vae_model.onnx
│   ├── vae_model_quant.onnx               # UINT8 quantised model
│   ├── scaler_kmeans.pkl
│   ├── scaler_iforest.pkl
│   └── scaler_vae.save
│
├── docs/
│   └── plots/                             # All report figures
│       ├── vae_architecture_topology.png  # Figure 3.1
│       ├── phase1_comparison.png          # Figure 4.1
│       ├── baseline_kmeans_cm.png         # Figure 4.2
│       ├── baseline_iforest_cm.png        # Figure 4.3
│       ├── baseline_vae_32x4_cm.png       # Figure 4.4
│       ├── optuna_history.png             # Figure 4.5
│       ├── optuna_importance.png          # Figure 4.6
│       ├── optuna_parallel.png            # Figure 4.7
│       ├── vae_calibration_curve.png      # Figure 4.8
│       ├── total_model_comparison.png     # Figure 4.9
│       ├── phase3_optimised_cm.png        # Figure 4.10
│       ├── error_distribution_log.png     # Figure 4.11
│       ├── missed_attacks_analysis.png    # Figure 4.12
│       ├── figure_4_11_xai_feature_importance.png  # Figure 4.13
│       ├── figure_4_12_adversarial_decay.png        # Figure 4.14
│       ├── edge_vs_cloud_log.png          # Figure 4.15
│       └── ensemble_pr_curve_real.png     # Figure 4.16
│
└── scripts/
    ├── app_api.py
    ├── app_dashboard.py
    │
    ├── 01_Preprocessing/
    │   └── 01_preprocess.py               # Dataset cleaning and train/test split
    │
    ├── 02_Baseline_Models/
    │   ├── 02_train_kmeans.py             # Train K-Means baseline
    │   ├── 02_train_iforest.py            # Train Isolation Forest baseline
    │   ├── 02_train_vae.py                # Train baseline VAE (32x4)
    │   ├── 02_evaluate_baselines.py       # Evaluate all three baselines
    │   ├── 02_baseline_visuals.py         # Generate Phase 1 figures (4.1–4.4)
    │   └── 02_terminal_output.txt         # Verified terminal output from baseline evaluation
    │
    ├── 03_optimisation/
    │   ├── 03_optimised_architecture.py   # Bayesian hyperparameter search (Optuna)
    │   ├── 03_failed_optimised_vae.py     # First training attempt — superseded, not used for results
    │   ├── 03_Optimised_VAE.py            # Threshold calibration search (final version)
    │   ├── 03_evaluate_optimised.py       # Evaluate optimised VAE with confusion matrix
    │   ├── 03_check.py                    # Audit Optuna SQLite database
    │   ├── 03_optimisation_visualised.py  # Generate Phase 2 figures (4.5–4.10)
    │   ├── optimised_arch_diagram.py      # Generate Figure 3.1 (VAE topology)
    │   ├── 03_terminal_output.txt         # Verified terminal output from threshold calibration
    │   └── optuna_study.db                # Persisted Optuna study (20 trials)
    │
    ├── 04_evaluation/
    │   ├── 04_check_error.py              # Diagnose reconstruction error distribution
    │   ├── 04_error_analysis.py           # Breakdown of missed attacks by type
    │   ├── 04_explain_anomalies.py        # XAI feature attribution per attack sample
    │   ├── 04_adversarial_noise_test.py   # Gaussian noise robustness testing
    │   ├── 04_evaluation_visualised.py    # Generate Phase 3 figures (4.11–4.14)
    │   └── 04_terminal_output.txt         # Verified terminal output from XAI feature attribution
    │
    ├── 05_implementation/
    │   ├── 05_profile_system.py           # Edge latency and RAM profiling (ONNX)
    │   ├── 05_quantize_torch.py           # UINT8 dynamic quantisation
    │   ├── 05_train_cloud_model.py        # Cloud LSTM benchmark + Figure 4.15
    │   ├── 05_drift_lifecycle.py          # KS test drift detection pipeline
    │   └── 05_terminal_output.txt         # Verified terminal output from edge profiling and quantisation
    │
    └── 06_futurework/
        └── 06_ensemble_learning.py        # Hybrid ensemble evaluation + Figure 4.16
```

---

## Dataset

This project uses the **Edge-IIoT-Set** dataset (Ferrag et al., 2022), a publicly available benchmark generated from a purpose-built IIoT testbed encompassing 14 attack categories.

**Download:** [IEEE DataPort](https://ieee-dataport.org/documents/edge-iiot-set-new-comprehensive-realistic-cyber-security-dataset-iot-and-iiot-applications)

Once downloaded, place the file at:
```
data/DNN-EdgeIIoT-dataset.csv
```

The preprocessing script will generate the required train and test splits automatically.

---

## Setup

### Requirements

- Python 3.9+
- PyTorch
- Scikit-learn
- Optuna
- ONNX Runtime
- pandas, numpy, matplotlib, seaborn, scipy, psutil, joblib, kaleido

Install dependencies:
```bash
pip install torch scikit-learn optuna onnxruntime pandas numpy matplotlib seaborn scipy psutil joblib kaleido
```

### Directory structure expected

```
fypCode/
├── data/
├── output/
├── docs/plots/
└── scripts/       ← this repository
```

---

## Running the Experiment

Scripts are numbered and intended to be run in order.

### 1. Preprocess the dataset
```bash
python scripts/01_Preprocessing/01_preprocess.py
```
Produces `data/clean_benign_train.csv` and `data/clean_mixed_test.csv`.

### 2. Train and evaluate baselines
```bash
python scripts/02_Baseline_Models/02_train_kmeans.py
python scripts/02_Baseline_Models/02_train_iforest.py
python scripts/02_Baseline_Models/02_train_vae.py
python scripts/02_Baseline_Models/02_evaluate_baselines.py
python scripts/02_Baseline_Models/02_baseline_visuals.py
```
Reference output: `scripts/02_Baseline_Models/02_terminal_output.txt`

### 3. Optimise and evaluate the VAE
```bash
python scripts/03_optimisation/03_optimised_architecture.py   # Bayesian search
python scripts/03_optimisation/03_Optimised_VAE.py            # Threshold calibration
python scripts/03_optimisation/03_evaluate_optimised.py       # Full evaluation
python scripts/03_optimisation/03_optimisation_visualised.py  # Generate figures
python scripts/03_optimisation/optimised_arch_diagram.py      # Architecture diagram
```
Reference output: `scripts/03_optimisation/03_terminal_output.txt`

Note: `03_failed_optimised_vae.py` is an earlier training attempt that was superseded and was not used to produce any results in the report.

### 4. Error analysis and XAI
```bash
python scripts/04_evaluation/04_check_error.py
python scripts/04_evaluation/04_error_analysis.py
python scripts/04_evaluation/04_explain_anomalies.py
python scripts/04_evaluation/04_adversarial_test.py
python scripts/04_evaluation/04_evaluation_visualised.py
```
Reference output: `scripts/04_evaluation/04_terminal_output.txt`

### 5. Edge deployment profiling
```bash
python scripts/05_implementation/05_quantize_torch.py
python scripts/05_implementation/05_profile_system.py
python scripts/05_implementation/05_train_cloud_model.py
python scripts/05_implementation/05_drift_lifecycle.py
```
Reference output: `scripts/05_implementation/05_terminal_output.txt`

### 6. Ensemble evaluation
```bash
python scripts/06_futurework/06_ensemble_learning.py
```

---

## Key Results

| Metric | Value |
|---|---|
| Accuracy | 85.27% |
| Attack Precision | 1.00 |
| Attack Recall | 0.46 |
| Attack F1-Score | 0.63 |
| Average Inference Latency | 0.0198 ms |
| P95 Latency | 0.0210 ms |
| RAM Consumption | 170.98 MB |
| Model Size (ONNX) | 16.40 KB |
| Model Size (UINT8 quantised) | 11.57 KB |
| False Positives | 0 |

---

## Reference

Ferrag, M.A. et al. (2022) 'Edge-IIoT-Set: A new comprehensive realistic cyber security dataset of IoT and IIoT applications for centralized and federated learning', *IEEE Access*, 10, pp. 40,281–40,306.
