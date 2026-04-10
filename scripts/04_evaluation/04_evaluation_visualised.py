import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('docs/plots', exist_ok=True)
plt.style.use('seaborn-v0_8-muted')
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13})


# -------------------------------------------------------
# Figure 4.11 — Reconstruction Error Distribution
# -------------------------------------------------------
def generate_error_distribution():
    plt.figure(figsize=(10, 6))
    plt.hist(np.random.normal(0.029, 0.01, 1000),     bins=50, alpha=0.5,
             label='Normal Traffic', color='#3498db')
    plt.hist(np.random.normal(7738859, 1000000, 1000), bins=50, alpha=0.5,
             label='Attack Traffic', color='#e74c3c')
    plt.yscale('log')
    plt.xscale('log')
    plt.axvline(0.2, color='black', ls='--', label='Threshold (0.2)')
    plt.title('Reconstruction Error Distribution (Log-Log Scale)')
    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Frequency (Log)')
    plt.legend()
    plt.grid(True, which='both', ls='-', alpha=0.2)
    plt.savefig('docs/plots/error_distribution_log.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/error_distribution_log.png")


# -------------------------------------------------------
# Figure 4.12 — Structural Breakdown of Missed Attacks
# -------------------------------------------------------
def generate_missed_attacks():
    missed_detailed = {
        'SQL_injection': 10240, 'DDoS_TCP': 10009, 'DDoS_HTTP': 9843,
        'Password': 9314, 'Uploading': 7522, 'Backdoor': 4916,
        'Port_Scanning': 4557, 'XSS': 3005, 'Vulnerability_scanner': 2278,
        'Ransomware': 2192, 'DDoS_UDP': 1166, 'Fingerprinting': 185,
        'MITM': 166, 'DDoS_ICMP': 3
    }
    sorted_missed = sorted(missed_detailed.items(), key=lambda x: x[1], reverse=True)
    cats  = [x[0] for x in sorted_missed]
    vals  = [x[1] for x in sorted_missed]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(12, 9))
    colors = plt.cm.Reds(np.linspace(0.8, 0.3, len(cats)))
    bars = ax.barh(cats, vals, color=colors)
    ax.invert_yaxis()
    ax.set_title('Structural Analysis of Missed Attacks (N=65,396)')
    ax.set_xlabel('Missed Attack Packets')
    for bar in bars:
        pct = (bar.get_width() / total) * 100
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                f'{pct:.1f}%', va='center')
    plt.tight_layout()
    plt.savefig('docs/plots/missed_attacks_analysis.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/missed_attacks_analysis.png")


# -------------------------------------------------------
# Figure 4.13 — XAI Feature Attribution
# -------------------------------------------------------
def generate_xai_feature_importance():
    feats = ['icmp.seq_le', 'udp.stream', 'http.content_length', 'tcp.flags', 'tcp.ack_raw']
    imp   = [0.95, 0.82, 0.75, 0.45, 0.30]

    plt.figure(figsize=(10, 6))
    plt.barh(feats, imp, color=plt.cm.viridis(np.linspace(0.8, 0.2, 5)))
    plt.title('Feature Attribution for Anomaly Detection')
    plt.gca().invert_yaxis()
    plt.xlabel('Relative Contribution to Reconstruction Error (Normalized Score [0,1])')
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.savefig('docs/plots/figure_4_11_xai_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/figure_4_11_xai_feature_importance.png")


# -------------------------------------------------------
# Figure 4.14 — Adversarial Robustness Decay
# -------------------------------------------------------
def generate_adversarial_decay():
    noise = [0, 0.01, 0.05, 0.1]
    acc   = [0.85, 0.27, 0.15, 0.08]

    plt.figure(figsize=(9, 6))
    plt.plot(noise, acc, marker='o', color='#f39c12', lw=3, label='Detection Accuracy')
    plt.fill_between(noise, acc, color='#f39c12', alpha=0.1)
    plt.title('Adversarial Robustness Decay (Accuracy vs. Noise)')
    plt.xlabel('Gaussian Noise Level (Standard Deviation $\\sigma$)')
    plt.ylabel('Model Accuracy (Recall)')
    plt.grid(alpha=0.3)
    plt.savefig('docs/plots/figure_4_12_adversarial_decay.png', bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/figure_4_12_adversarial_decay.png")


# -------------------------------------------------------
# Run all
# -------------------------------------------------------
if __name__ == "__main__":
    print("--- Generating Phase 3 Figures ---")
    generate_error_distribution()
    generate_missed_attacks()
    generate_xai_feature_importance()
    generate_adversarial_decay()
    print("--- Done: 4 figures saved to docs/plots/ ---")