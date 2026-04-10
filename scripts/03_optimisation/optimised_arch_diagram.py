import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.makedirs('docs/plots', exist_ok=True)


def draw_box(ax, x, y, w, h, label, color='#3498db'):
    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                               edgecolor='black', facecolor=color, alpha=0.7)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label,
            ha='center', va='center', fontweight='bold', fontsize=10)


def draw_vae_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Layers
    draw_box(ax, 1,    4,    1.5, 2,   "Input Layer\n(39 Features)",        '#95a5a6')
    draw_box(ax, 3.5,  3.5,  2,   3,   "Hidden Layer 1\n(64 Neurons)",      '#3498db')
    draw_box(ax, 6.5,  6,    2,   1.5, "Mean (μ)\n(8 Units)",               '#2ecc71')
    draw_box(ax, 6.5,  2.5,  2,   1.5, "Log Var (log σ²)\n(8 Units)",       '#e74c3c')
    draw_box(ax, 9.5,  4.25, 2,   1.5, "Latent Vector (z)\nz = μ + σ · ε", '#f1c40f')
    draw_box(ax, 12.5, 3.5,  2,   3,   "Hidden Layer 2\n(64 Neurons)",      '#3498db')
    draw_box(ax, 15.5, 4,    1.5, 2,   "Output Layer\n(39 Reconstructed)",  '#95a5a6')

    # Arrows
    arrow = dict(arrowstyle='->', lw=1.5, color='black')
    ax.annotate('', xy=(3.5,  5),    xytext=(2.5,  5),    arrowprops=arrow)  # Input -> H1
    ax.annotate('', xy=(6.5,  6.75), xytext=(5.5,  5.5),  arrowprops=arrow)  # H1 -> mu
    ax.annotate('', xy=(6.5,  3.25), xytext=(5.5,  4.5),  arrowprops=arrow)  # H1 -> logvar
    ax.annotate('', xy=(9.5,  5),    xytext=(8.5,  6.75), arrowprops=arrow)  # mu -> z
    ax.annotate('', xy=(9.5,  5),    xytext=(8.5,  3.25), arrowprops=arrow)  # logvar -> z
    ax.annotate('', xy=(12.5, 5),    xytext=(11.5, 5),    arrowprops=arrow)  # z -> H2
    ax.annotate('', xy=(15.5, 5),    xytext=(14.5, 5),    arrowprops=arrow)  # H2 -> Output

    # Loss function annotation
    plt.text(8.5, 9,
             r"Loss Function: $\mathcal{L} = \text{MSE}(x, \hat{x}) + \beta \cdot D_{KL}[q(z|x) || p(z)]$",
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.8))

    # Decision logic annotation
    plt.text(16.25, 2,
             "Decision Logic:\nIf MSE > 0.2 → ANOMALY\nElse → NORMAL",
             ha='center', fontsize=11, fontweight='bold', color='#c0392b',
             bbox=dict(boxstyle='round', facecolor='#f9ebeb'))

    plt.title("Optimised VAE Topology (64x8) with Reparameterisation Flow",
              fontsize=15, pad=20)
    plt.savefig('docs/plots/vae_architecture_topology.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: docs/plots/vae_architecture_topology.png")


if __name__ == "__main__":
    draw_vae_architecture()