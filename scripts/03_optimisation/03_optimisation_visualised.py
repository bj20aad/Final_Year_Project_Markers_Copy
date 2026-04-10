import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import optuna
import optuna.visualization as vis
from pathlib import Path
from sklearn.metrics import confusion_matrix

os.makedirs('docs/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})

DB_PATH = "sqlite:////Users/benjoel/fypCode/scripts/03_optimisation/optuna_study.db"
OUTPUT_DIR = Path("docs/plots")


# -------------------------------------------------------
# Figure 4.5 — Optuna Optimisation History
# -------------------------------------------------------
def generate_optuna_history(study):
    fig = vis.plot_optimization_history(study)
    fig.write_image(OUTPUT_DIR / "optuna_history.png")
    print("Generated: docs/plots/optuna_history.png")


# -------------------------------------------------------
# Figure 4.6 — Hyperparameter Importance Scores
# -------------------------------------------------------
def generate_optuna_importance(study):
    fig = vis.plot_param_importances(study)
    fig.write_image(OUTPUT_DIR / "optuna_importance.png")
    print("Generated: docs/plots/optuna_importance.png")


# -------------------------------------------------------
# Figure 4.7 — Parallel Coordinate Plot
# -------------------------------------------------------
def generate_optuna_parallel(study):
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image(OUTPUT_DIR / "optuna_parallel.png")
    print("Generated: docs/plots/optuna_parallel.png")


# -------------------------------------------------------
# Figure 4.8 — VAE Threshold Calibration Curve
# -------------------------------------------------------
def generate_calibration_curve():
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    prec = [0.36, 0.37, 0.38, 0.39, 0.42, 0.65, 1.00]
    rec  = [0.85, 0.82, 0.75, 0.70, 0.60, 0.52, 0.46]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, prec, marker='o', label='Precision', color='#2ecc71', lw=2)
    plt.plot(thresholds, rec,  marker='s', label='Recall',    color='#e74c3c', lw=2)
    plt.axvline(x=0.2, color='black', ls=':', label='Optimised Threshold (0.2)')
    plt.text(0.185, 0.7, 'Dual-Threshold Paradox', rotation=90, ha='right', fontweight='bold')
    plt.title('VAE Threshold Calibration & Dual-Threshold Paradox')
    plt.xlabel('Security Threshold')
    plt.legend(loc='lower left')
    plt.grid(alpha=0.3)
    plt.savefig('docs/plots/vae_calibration_curve.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/vae_calibration_curve.png")


# -------------------------------------------------------
# Figure 4.9 — All-Model Performance Benchmarking
# -------------------------------------------------------
def generate_model_comparison():
    models = ['K-Means', 'I-Forest', 'Base VAE', 'Optimised VAE']
    p  = [1.00, 0.45, 0.42, 1.00]
    r  = [0.37, 0.53, 0.60, 0.46]
    f1 = [0.54, 0.49, 0.49, 0.63]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 7))
    rects1 = ax.bar(x - width, p,  width, label='Precision', color='#2ecc71')
    rects2 = ax.bar(x,         r,  width, label='Recall',    color='#e74c3c')
    rects3 = ax.bar(x + width, f1, width, label='F1-Score',  color='#3498db')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title('Comparative Performance - Baselines vs Optimised VAE')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.2)
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.02,
                    f'{rect.get_height():.2f}', ha='center', fontsize=9)
    plt.savefig('docs/plots/total_model_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/total_model_comparison.png")


# -------------------------------------------------------
# Figure 4.10 — Optimised VAE Confusion Matrix
# -------------------------------------------------------
def generate_optimised_cm():
    y_true = [0] * 323151 + [1] * 120689
    y_pred = [0] * 323151 + [1] * 55517 + [0] * (120689 - 55517)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'])
    plt.title('Final Confusion Matrix - Optimised VAE (Threshold 0.2)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('docs/plots/phase3_optimised_cm.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/phase3_optimised_cm.png")


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Phase 2 Figures ---")

    # Load Optuna study once for all three Optuna plots
    print("Connecting to Optuna database...")
    try:
        study = optuna.load_study(study_name="vae_architecture_study", storage=DB_PATH)
        generate_optuna_history(study)
        generate_optuna_importance(study)
        generate_optuna_parallel(study)
    except Exception as e:
        print(f"Warning: Could not load Optuna study — {e}")
        print("Skipping Figures 4.5, 4.6, 4.7.")

    generate_calibration_curve()
    generate_model_comparison()
    generate_optimised_cm()

    print("--- Done: 6 figures saved to docs/plots/ ---")