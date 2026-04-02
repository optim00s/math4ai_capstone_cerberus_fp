"""
Professional plotting utilities for Math4AI Capstone.
High-quality decision boundaries, loss curves, and analysis plots.
Designed for jury presentation quality.

Fix applied: replaced deprecated plt.cm.get_cmap() with
matplotlib.colormaps[] (matplotlib >= 3.7 compatible).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

# ── Global style settings ──
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Custom colormaps
PROB_CMAP = plt.cm.RdYlBu  # diverging: red → yellow → blue
CLASS_COLORS = ['#E74C3C', '#3498DB']  # red, blue for binary
MULTI_COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12',
                '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4']


def _get_cmap(name, n=None):
    """
    Compatibility wrapper: returns a colormap regardless of matplotlib version.
    Replaces deprecated plt.cm.get_cmap().
    """
    try:
        # matplotlib >= 3.7
        cmap = matplotlib.colormaps[name]
        if n is not None:
            cmap = cmap.resampled(n)
        return cmap
    except AttributeError:
        # matplotlib < 3.7 fallback
        return plt.cm.get_cmap(name, n)


def get_figures_dir():
    """Return path to the figures directory, creating it if needed."""
    fig_dir = Path(__file__).resolve().parents[1] / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


def plot_decision_boundary(model, X, y, title="Decision Boundary",
                           filename=None, resolution=300):
    """Plot 2D decision boundary with probability heatmap and data overlay."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))

    margin = 0.8
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    _, P = model.predict(grid)
    if P.shape[1] == 2:
        prob_map = P[:, 1].reshape(xx.shape)

        im = ax.contourf(xx, yy, prob_map, levels=np.linspace(0, 1, 50),
                         cmap=PROB_CMAP, alpha=0.85)
        ax.contour(xx, yy, prob_map, levels=[0.5],
                   colors='#2C3E50', linewidths=2.5, linestyles='-')
        ax.contour(xx, yy, prob_map, levels=[0.2, 0.35, 0.65, 0.8],
                   colors='#7F8C8D', linewidths=0.6, linestyles='--', alpha=0.5)

        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('P(class = 1)', fontsize=11)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

        for cls_val, color, marker, label in [(0, CLASS_COLORS[0], 'o', 'Class 0'),
                                               (1, CLASS_COLORS[1], 's', 'Class 1')]:
            mask = y == cls_val
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                       edgecolors='white', linewidths=0.5, s=30, alpha=0.85,
                       label=label, zorder=5)
    else:
        preds = np.argmax(P, axis=1).reshape(xx.shape)
        n_classes = P.shape[1]
        # FIX: use _get_cmap() instead of deprecated plt.cm.get_cmap()
        cmap = _get_cmap('tab10', n_classes)
        ax.contourf(xx, yy, preds, levels=np.arange(-0.5, n_classes, 1),
                    cmap=cmap, alpha=0.3)
        ax.contour(xx, yy, preds, colors='k', linewidths=0.5, alpha=0.3)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap,
                             edgecolors='white', linewidths=0.3, s=15, alpha=0.8)
        plt.colorbar(scatter, ax=ax, shrink=0.85, label='Class')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    if P.shape[1] == 2:
        ax.legend(loc='upper left', framealpha=0.9, edgecolor='#BDC3C7')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()

    if filename:
        fig.savefig(get_figures_dir() / filename, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filename}")
    plt.close(fig)


def plot_decision_boundary_comparison(models, X, y, titles,
                                      filename=None, resolution=300):
    """Side-by-side decision boundary comparison."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    margin = 0.8
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    im = None
    for ax, model, title in zip(axes, models, titles):
        _, P = model.predict(grid)
        if P.shape[1] == 2:
            prob_map = P[:, 1].reshape(xx.shape)

            im = ax.contourf(xx, yy, prob_map, levels=np.linspace(0, 1, 50),
                             cmap=PROB_CMAP, alpha=0.85)
            ax.contour(xx, yy, prob_map, levels=[0.5],
                       colors='#2C3E50', linewidths=2.5)
            ax.contour(xx, yy, prob_map, levels=[0.2, 0.35, 0.65, 0.8],
                       colors='#7F8C8D', linewidths=0.5, linestyles='--', alpha=0.4)

            for cls_val, color, marker in [(0, CLASS_COLORS[0], 'o'),
                                            (1, CLASS_COLORS[1], 's')]:
                mask = y == cls_val
                ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                           edgecolors='white', linewidths=0.5, s=25, alpha=0.8,
                           zorder=5)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('P(class = 1)', fontsize=11)
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.subplots_adjust(right=0.90, wspace=0.25)

    if filename:
        fig.savefig(get_figures_dir() / filename, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filename}")
    plt.close(fig)

def plot_capacity_ablation_boundaries(models_dict, X, y,
                                       filename=None, resolution=300):
    """Side-by-side decision boundaries for capacity ablation."""
    widths = sorted(models_dict.keys())
    n = len(widths)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    margin = 0.8
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]

    im = None
    for ax, h_w in zip(axes, widths):
        model = models_dict[h_w]
        _, P = model.predict(grid)
        prob_map = P[:, 1].reshape(xx.shape)

        im = ax.contourf(xx, yy, prob_map, levels=np.linspace(0, 1, 50),
                         cmap=PROB_CMAP, alpha=0.85)
        ax.contour(xx, yy, prob_map, levels=[0.5],
                   colors='#2C3E50', linewidths=2.5)
        ax.contour(xx, yy, prob_map, levels=[0.2, 0.35, 0.65, 0.8],
                   colors='#7F8C8D', linewidths=0.5, linestyles='--', alpha=0.4)

        for cls_val, color, marker in [(0, CLASS_COLORS[0], 'o'),
                                        (1, CLASS_COLORS[1], 's')]:
            mask = y == cls_val
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker,
                       edgecolors='white', linewidths=0.5, s=22, alpha=0.8,
                       zorder=5)

        ax.set_title(f'Hidden Width h = {h_w}', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    if im is not None:
        cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label('P(class = 1)', fontsize=11)

    fig.suptitle('Capacity Ablation: Decision Boundaries on Moons',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.subplots_adjust(right=0.91, wspace=0.22)

    if filename:
        fig.savefig(get_figures_dir() / filename, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filename}")
    plt.close(fig)

def plot_loss_curves(histories, labels, title="Training Dynamics",
                     filename=None):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

    for i, (hist, label) in enumerate(zip(histories, labels)):
        c = palette[i % len(palette)]
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], '-', color=c, alpha=0.45,
                     linewidth=1.2, label=f'{label} train')
        axes[0].plot(epochs, hist['val_loss'], '-', color=c, alpha=1.0,
                     linewidth=2.2, label=f'{label} val')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss', fontweight='bold')
    axes[0].legend(fontsize=8, framealpha=0.9)

    for i, (hist, label) in enumerate(zip(histories, labels)):
        c = palette[i % len(palette)]
        epochs = range(1, len(hist['train_acc']) + 1)
        axes[1].plot(epochs, hist['train_acc'], '-', color=c, alpha=0.45,
                     linewidth=1.2, label=f'{label} train')
        axes[1].plot(epochs, hist['val_acc'], '-', color=c, alpha=1.0,
                     linewidth=2.2, label=f'{label} val')

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy', fontweight='bold')
    axes[1].legend(fontsize=8, framealpha=0.9)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if filename:
        fig.savefig(get_figures_dir() / filename, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filename}")
    plt.close(fig)

def plot_capacity_ablation(histories_dict, title="Capacity Ablation (Moons)",
                           filename=None):
    """Plot loss/accuracy curves for different hidden widths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'2': '#E74C3C', '8': '#F39C12', '32': '#27AE60'}

    for width_str, hist in histories_dict.items():
        c = colors.get(width_str, '#000000')
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], '--', color=c, alpha=0.4, linewidth=1)
        axes[0].plot(epochs, hist['val_loss'], '-', color=c,
                     label=f'h={width_str}', linewidth=2.5)
        axes[1].plot(epochs, hist['train_acc'], '--', color=c, alpha=0.4, linewidth=1)
        axes[1].plot(epochs, hist['val_acc'], '-', color=c,
                     label=f'h={width_str}', linewidth=2.5)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Validation Loss vs. Hidden Width', fontweight='bold')
    axes[0].legend(fontsize=10, framealpha=0.9)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy vs. Hidden Width', fontweight='bold')
    axes[1].legend(fontsize=10, framealpha=0.9)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if filename:
        fig.savefig(get_figures_dir() / filename, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"  Saved: {filename}")
    plt.close(fig)


def plot_optimizer_comparison(histories_dict, title="Optimizer Study (Digits)", filename=None):
    """Validation loss and accuracy for SGD, Momentum, and Adam."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = {'SGD': '#E74C3C', 'Momentum': '#3498DB', 'Adam': '#27AE60'}

    for name, hist in histories_dict.items():
        c = colors.get(name, '#000000')
        ep = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(ep, hist['train_loss'], '-', color=c, alpha=0.35, linewidth=1)
        axes[0].plot(ep, hist['val_loss'],   '-', color=c, linewidth=2.5, label=f'{name} val')
        axes[1].plot(ep, hist['train_acc'],  '-', color=c, alpha=0.35, linewidth=1)
        axes[1].plot(ep, hist['val_acc'],    '-', color=c, linewidth=2.5, label=f'{name} val')

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss: Optimizer Comparison', fontweight='bold')
    axes[0].legend(fontsize=10, framealpha=0.9)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy: Optimizer Comparison', fontweight='bold')
    axes[1].legend(fontsize=10, framealpha=0.9)

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, filename)
    
# ── Track A: PCA / SVD figures ────────────────────────────

def plot_pca_scree(explained_variance_ratio, filename=None):
    """Scree plot — individual EVR bars with cumulative % line.
    Dashed reference at 90% cumulative variance.
    """
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    n = min(len(explained_variance_ratio), 40)
    cumulative = np.cumsum(explained_variance_ratio[:n])

    ax.bar(range(1, n + 1), explained_variance_ratio[:n], alpha=0.65,
           color='#3498DB', edgecolor='#2980B9', linewidth=0.5, label='Individual')
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cumulative * 100, 'o-', color='#E74C3C',
             linewidth=2.5, markersize=4, label='Cumulative %')
    ax2.axhline(y=90, color='#95A5A6', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Cumulative Explained Variance (%)', color='#E74C3C')
    ax2.set_ylim(0, 105)

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio', color='#3498DB')
    ax.set_title('Scree Plot - Digits Data', fontsize=14, fontweight='bold', pad=12)

    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc='center right', framealpha=0.9)

    plt.tight_layout()
    _save(fig, filename)


def plot_pca_2d(X_2d, y, filename=None):
    """Scatter of all digits projected onto the first 2 principal components."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    for cls in np.unique(y):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=MULTI_COLORS[int(cls) % len(MULTI_COLORS)],
                   s=12, alpha=0.65, label=str(int(cls)),
                   edgecolors='white', linewidths=0.2)

    ax.set_xlabel('PC 1', fontsize=12); ax.set_ylabel('PC 2', fontsize=12)
    ax.set_title('2D PCA Projection of Digits Data', fontsize=14, fontweight='bold', pad=12)
    ax.legend(title='Digit', loc='best', ncol=2, fontsize=8,
              title_fontsize=10, framealpha=0.9, edgecolor='#BDC3C7')
    plt.tight_layout()
    _save(fig, filename)


def plot_pca_softmax_comparison(dims, val_accs, val_losses, filename=None):
    """Bar chart: softmax accuracy and CE at different PCA dimensions (m = 10, 20, 40, 64)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    x = np.arange(len(dims))
    labels = [str(d) for d in dims]

    for ax, vals, ylabel, title, offset in zip(
            axes,
            [val_accs, val_losses],
            ['Validation Accuracy', 'Validation Cross-Entropy'],
            ['Accuracy vs PCA Dimensions', 'Loss vs PCA Dimensions'],
            [0.003, 0.005]):
        bars = ax.bar(x, vals, color='#3498DB', alpha=0.85,
                      edgecolor='white', linewidth=1.5, width=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel('PCA Dimensions'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')

    fig.suptitle('Track A: Softmax Regression with PCA Compression',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, filename)