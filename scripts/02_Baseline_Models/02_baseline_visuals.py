import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('docs/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-muted')

# -------------------------------------------------------
# Figure 4.1 — Baseline F1-Score Comparison (Bar Chart)
# -------------------------------------------------------
def generate_f1_comparison():
    plt.figure(figsize=(9, 5))
    plt.bar(['K-Means', 'I-Forest', 'Base VAE'], [0.54, 0.49, 0.49], color='#95a5a6', alpha=0.8)
    plt.title('Baseline Models Performance (F1-Scores)')
    plt.ylabel('F1-Score')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig('docs/plots/phase1_comparison.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/phase1_comparison.png")


# -------------------------------------------------------
# Figures 4.2, 4.3, 4.4 — Individual Confusion Matrices
# -------------------------------------------------------
def generate_confusion_matrices():
    baselines = {
        "K-Means": {
            "matrix": [[323151, 0], [76034, 44655]],
            "filename": "baseline_kmeans_cm.png",
            "title": "Baseline K-Means Confusion Matrix"
        },
        "I-Forest": {
            "matrix": [[245595, 77556], [56724, 63965]],
            "filename": "baseline_iforest_cm.png",
            "title": "Baseline I-Forest Confusion Matrix"
        },
        "VAE_32x4": {
            "matrix": [[222974, 100177], [48276, 72413]],
            "filename": "baseline_vae_32x4_cm.png",
            "title": "Baseline VAE (32x4) Confusion Matrix"
        }
    }

    for name, data in baselines.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(data["matrix"], annot=True, fmt='d', cmap='Greys',
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'],
                    cbar=True)
        plt.title(data["title"], fontsize=14, pad=20)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        save_path = os.path.join('docs/plots', data["filename"])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {save_path}")


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Phase 1 Figures ---")
    generate_f1_comparison()
    generate_confusion_matrices()
    print("--- Done: 4 figures saved to docs/plots/ ---")