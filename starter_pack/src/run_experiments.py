#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiments:
  1. Core comparison on linear Gaussian, moons, digits
  2. Capacity ablation on moons (h = 2, 8, 32)
  3. Optimizer study on digits (SGD, Momentum, Adam)
  4. Repeated-seed evaluation (5 seeds, 95 % CI)
  5. Track A: PCA / SVD analysis of digits
  6. Failure-case analysis (under-capacity NN on digits)
  7. Implementation sanity checks

Usage: python -m starter_pack.src.run_experiments  (from repo root)
"""

import sys, io, numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starter_pack.src.models import SoftmaxRegression, NeuralNetwork, cross_entropy_loss, one_hot, stable_softmax
from starter_pack.src.optimizers import SGD, Momentum, Adam
from starter_pack.src.train import train_model, compute_accuracy, compute_loss
from starter_pack.src.data_utils import load_synthetic, load_digits
from starter_pack.src.plotting import (
    plot_decision_boundary, plot_decision_boundary_comparison,
    plot_capacity_ablation_boundaries,
    plot_loss_curves, plot_capacity_ablation,
    plot_optimizer_comparison,
    plot_pca_scree, plot_pca_2d, plot_pca_softmax_comparison,
    get_figures_dir
)
from starter_pack.src.sanity_checks import main as run_standalone_sanity_checks

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = REPO_ROOT / "starter_pack" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Sanity Checks ─────────────────────────────────────────

def run_sanity_checks():
    print("=" * 62)
    print("SANITY CHECKS")
    print("=" * 62)
    print("Delegating to standalone sanity_checks.py...\n")
    run_standalone_sanity_checks()


# ── Experiment 1: Synthetic Tasks ─────────────────────────

def run_synthetic_experiments():
    """Train both models on linear_gaussian and moons, save figures and a summary text."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 1: SYNTHETIC TASKS")
    print("=" * 62)

    summary_lines = ["Synthetic Task Results", "=" * 50]

    for dataset_name in ['linear_gaussian', 'moons']:
        print(f"\n{'='*50}\n  Dataset: {dataset_name}")
        X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic(dataset_name)
        n_cls    = len(np.unique(y_train))
        n_feat   = X_train.shape[1]
        n_train  = len(y_train)
        X_all    = np.vstack([X_train, X_val, X_test])
        y_all    = np.concatenate([y_train, y_val, y_test])

        # Softmax Regression — full-batch SGD, lr=0.1
        print("\n  [Softmax Regression]")
        sr = SoftmaxRegression(n_feat, n_cls)
        sr.init_params(seed=42)
        hist_sr, best_sr, _ = train_model(
            sr, SGD(lr=0.1), X_train, y_train, X_val, y_val,
            n_classes=n_cls, n_epochs=300, batch_size=n_train, lam=1e-5, seed=42)
        sr.set_params(best_sr)
        acc_sr  = compute_accuracy(sr, X_test, y_test)
        loss_sr = compute_loss(sr, X_test, y_test, n_cls)
        print(f"  Test Acc: {acc_sr:.4f}, Test CE: {loss_sr:.4f}")

        # Neural Network — full-batch Adam, h=32
        print("\n  [Neural Network, h=32, Adam]")
        nn = NeuralNetwork(n_feat, 32, n_cls)
        nn.init_params(seed=42)
        hist_nn, best_nn, _ = train_model(
            nn, Adam(lr=0.01), X_train, y_train, X_val, y_val,
            n_classes=n_cls, n_epochs=300, batch_size=n_train, lam=1e-5, seed=42)
        nn.set_params(best_nn)
        acc_nn  = compute_accuracy(nn, X_test, y_test)
        loss_nn = compute_loss(nn, X_test, y_test, n_cls)
        print(f"  Test Acc: {acc_nn:.4f}, Test CE: {loss_nn:.4f}")

        summary_lines += [
            f"\nDataset: {dataset_name}",
            f"  Softmax | Test Acc: {acc_sr:.4f} | Test CE: {loss_sr:.4f}",
            f"  NN h=32 | Test Acc: {acc_nn:.4f} | Test CE: {loss_nn:.4f}",
        ]

        tag = dataset_name.replace('_', '')
        plot_decision_boundary(sr, X_all, y_all,
            title=f"Softmax Regression - {dataset_name}",
            filename=f"decision_boundary_{tag}_softmax.png")
        plot_decision_boundary(nn, X_all, y_all,
            title=f"Neural Network (h=32) - {dataset_name}",
            filename=f"decision_boundary_{tag}_nn.png")
        plot_decision_boundary_comparison(
            [sr, nn], X_all, y_all,
            titles=['Softmax Regression (Linear)', 'Neural Network (h=32, tanh)'],
            filename=f"comparison_{tag}.png")
        plot_loss_curves([hist_sr, hist_nn], ['Softmax', 'NN (h=32)'],
            title=f"Training Dynamics - {dataset_name}",
            filename=f"loss_curves_{tag}.png")

    with open(RESULTS_DIR / "synthetic_results.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("\n  Saved: results/synthetic_results.txt")


# ── Experiment 2: Digits Benchmark ────────────────────────

def run_digits_experiment():
    """Compare both models on the fixed digits split.

    NN uses Adam (lr=0.001) — the best optimizer from Experiment 4 —
    so the loss curve here is consistent with the five-seed summary table.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 2: DIGITS BENCHMARK")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_cls  = 10
    n_feat = X_train.shape[1]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    print("\n  [Softmax Regression — SGD lr=0.05]")
    sr = SoftmaxRegression(n_feat, n_cls)
    sr.init_params(seed=42)
    hist_sr, best_sr, ep_sr = train_model(
        sr, SGD(lr=0.05), X_train, y_train, X_val, y_val,
        n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42)
    sr.set_params(best_sr)
    print(f"  Test Acc: {compute_accuracy(sr, X_test, y_test):.4f}, "
          f"Test CE: {compute_loss(sr, X_test, y_test, n_cls):.4f}, Best Epoch: {ep_sr}")

    # Adam selected as final config — achieved lowest val CE in optimizer study
    print("\n  [Neural Network h=32 — Adam lr=0.001]")
    nn = NeuralNetwork(n_feat, 32, n_cls)
    nn.init_params(seed=42)
    hist_nn, best_nn, ep_nn = train_model(
        nn, Adam(lr=0.001), X_train, y_train, X_val, y_val,
        n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42)
    nn.set_params(best_nn)
    print(f"  Test Acc: {compute_accuracy(nn, X_test, y_test):.4f}, "
          f"Test CE: {compute_loss(nn, X_test, y_test, n_cls):.4f}, Best Epoch: {ep_nn}")

    plot_loss_curves([hist_sr, hist_nn], ['Softmax (SGD)', 'NN h=32 (Adam)'],
        title="Training Dynamics — Digits Benchmark (final configs, seed=42)",
        filename="loss_curves_digits.png")

    return hist_sr, hist_nn


# ── Experiment 3: Capacity Ablation (Moons) ───────────────

def run_capacity_ablation():
    """Train NNs with h = 2, 8, 32 on moons; compare boundaries and curves."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 3: CAPACITY ABLATION (MOONS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic('moons')
    n_cls, n_feat, n_train = 2, 2, len(y_train)
    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    histories, trained_models = {}, {}
    for h in [2, 8, 32]:
        print(f"\n  [NN, h={h}, Adam lr=0.01]")
        nn = NeuralNetwork(n_feat, h, n_cls)
        nn.init_params(seed=42)
        hist, best_p, best_ep = train_model(
            nn, Adam(lr=0.01), X_train, y_train, X_val, y_val,
            n_classes=n_cls, n_epochs=500, batch_size=n_train, lam=1e-5, seed=42)
        nn.set_params(best_p)
        print(f"  Test Acc: {compute_accuracy(nn, X_test, y_test):.4f}, Best Epoch: {best_ep}")
        histories[str(h)]  = hist
        trained_models[h]  = nn
        plot_decision_boundary(nn, X_all, y_all,
            title=f"NN (h={h}) - Moons",
            filename=f"decision_boundary_moons_h{h}.png")

    plot_capacity_ablation_boundaries(trained_models, X_all, y_all,
        filename="capacity_ablation_boundaries.png")
    plot_capacity_ablation(histories, filename="capacity_ablation_curves.png")


# ── Experiment 4: Optimizer Study (Digits) ────────────────

def run_optimizer_study():
    """Compare SGD, Momentum, and Adam on NN h=32 using identical hyperparameters."""
    print("\n" + "=" * 62)
    print("EXPERIMENT 4: OPTIMIZER STUDY (DIGITS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_cls, n_feat = 10, X_train.shape[1]

    opts = {
        'SGD':      SGD(lr=0.05),
        'Momentum': Momentum(lr=0.05, mu=0.9),
        'Adam':     Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8),
    }

    histories, results = {}, {}
    for name, opt in opts.items():
        print(f"\n  [{name}]")
        nn = NeuralNetwork(n_feat, 32, n_cls)
        nn.init_params(seed=42)
        hist, best_p, best_ep = train_model(
            nn, opt, X_train, y_train, X_val, y_val,
            n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42)
        nn.set_params(best_p)
        acc  = compute_accuracy(nn, X_test, y_test)
        loss = compute_loss(nn, X_test, y_test, n_cls)
        print(f"  Test Acc: {acc:.4f}, Test CE: {loss:.4f}, Best Epoch: {best_ep}")
        histories[name] = hist
        results[name]   = {'acc': acc, 'loss': loss, 'best_epoch': best_ep}

    plot_optimizer_comparison(histories, filename="optimizer_study_digits.png")

    with open(RESULTS_DIR / "optimizer_study.txt", "w") as f:
        f.write("Optimizer Study Results (NN h=32, Digits)\n" + "=" * 50 + "\n")
        for name, r in results.items():
            f.write(f"{name:10s} | Acc: {r['acc']:.4f} | Loss: {r['loss']:.4f} | Best Epoch: {r['best_epoch']}\n")

    return results


# ── Experiment 5: Repeated-Seed Evaluation ────────────────

def run_repeated_seed():
    """Run both final configs over 5 seeds and report means with 95 % CIs.

    Uses the t-distribution (t_{0.025,4} = 2.776) since n=5 is small.
    Final configs chosen by lowest val CE in the optimizer study:
      Softmax → SGD lr=0.05 | NN h=32 → Adam lr=0.001
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 5: REPEATED-SEED EVALUATION (DIGITS)")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_cls, n_feat = 10, X_train.shape[1]
    seeds  = [0, 1, 2, 3, 4]
    t_crit = 2.776   # t_{0.025, 4}

    results = {}
    for model_name in ['Softmax', 'NN']:
        accs, ces = [], []
        for s in seeds:
            if model_name == 'Softmax':
                model = SoftmaxRegression(n_feat, n_cls)
                model.init_params(seed=s)
                opt = SGD(lr=0.05)
            else:
                model = NeuralNetwork(n_feat, 32, n_cls)
                model.init_params(seed=s)
                opt = Adam(lr=0.001)

            _, best_p, _ = train_model(
                model, opt, X_train, y_train, X_val, y_val,
                n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=s, verbose=False)
            model.set_params(best_p)
            acc  = compute_accuracy(model, X_test, y_test)
            loss = compute_loss(model, X_test, y_test, n_cls)
            accs.append(acc); ces.append(loss)
            print(f"  {model_name} seed={s}: Acc={acc:.4f}, CE={loss:.4f}")

        accs, ces  = np.array(accs), np.array(ces)
        mean_acc   = accs.mean(); ci_acc = t_crit * accs.std(ddof=1) / np.sqrt(5)
        mean_ce    = ces.mean();  ci_ce  = t_crit * ces.std(ddof=1)  / np.sqrt(5)
        results[model_name] = dict(mean_acc=mean_acc, ci_acc=ci_acc,
                                   mean_ce=mean_ce,   ci_ce=ci_ce,
                                   accs=accs, ces=ces)
        print(f"\n  {model_name}: Acc {mean_acc:.4f} ± {ci_acc:.4f} | "
              f"CE {mean_ce:.4f} ± {ci_ce:.4f}")

    with open(RESULTS_DIR / "repeated_seed.txt", "w", encoding='utf-8') as f:
        f.write("Repeated-Seed Evaluation (5 seeds, Digits)\n")
        f.write("=" * 60 + "\n95% CI: mean +/- 2.776 * s / sqrt(5)\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n"
                    f"  Test Accuracy: {r['mean_acc']:.4f} ± {r['ci_acc']:.4f}\n"
                    f"  Test CE Loss:  {r['mean_ce']:.4f} ± {r['ci_ce']:.4f}\n"
                    f"  Raw accs: {r['accs']}\n  Raw CEs: {r['ces']}\n\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    names = list(results.keys())
    x = np.arange(len(names))

    for ax, metric, ylabel, title in zip(
            axes,
            [('mean_acc', 'ci_acc'), ('mean_ce', 'ci_ce')],
            ['Test Accuracy', 'Test Cross-Entropy'],
            ['Test Accuracy (5 seeds, 95% CI)', 'Test CE Loss (5 seeds, 95% CI)']):
        means = [results[n][metric[0]] for n in names]
        cis   = [results[n][metric[1]] for n in names]
        bars  = ax.bar(x, means, yerr=cis, color=['#3498DB', '#E74C3C'],
                       edgecolor='white', capsize=12, alpha=0.85, width=0.5,
                       error_kw={'linewidth': 2, 'capthick': 2})
        for bar, val, ci in zip(bars, means, cis):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ci + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')

    fig.suptitle('Repeated-Seed Evaluation — Digits', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(get_figures_dir() / "repeated_seed_digits.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: repeated_seed_digits.png")

    return results


# ── Experiment 6: Track A — PCA / SVD Analysis ────────────

def run_track_a_pca():
    """PCA / SVD on digits: scree plot, 2D projection, softmax at reduced dims.

    Centering uses only training-set mean to avoid leakage.
    Softmax is tested at m ∈ {10, 20, 40, 64} principal components.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 6: TRACK A — PCA/SVD ANALYSIS")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_cls   = 10
    X_full  = np.vstack([X_train, X_val, X_test])
    y_full  = np.concatenate([y_train, y_val, y_test])

    mean      = X_train.mean(axis=0)
    X_train_c = X_train - mean
    X_val_c   = X_val   - mean
    X_full_c  = X_full  - mean

    U, S, Vt = np.linalg.svd(X_train_c, full_matrices=False)
    evr       = (S ** 2) / np.sum(S ** 2)  # explained variance ratio per component
    cumvar    = np.cumsum(evr)

    print("\n  [Scree Plot]")
    plot_pca_scree(evr, filename="pca_scree_digits.png")
    for k in [5, 10, 20, 40]:
        print(f"    Top {k:2d} PCs: {cumvar[k-1]*100:.1f}%")

    print("\n  [2D PCA Visualization]")
    plot_pca_2d(X_full_c @ Vt[:2].T, y_full, filename="pca_2d_digits.png")

    print("\n  [Softmax at reduced PCA dimensions]")
    pca_dims = [10, 20, 40, 64]
    val_accs, val_losses = [], []
    for m in pca_dims:
        Vm       = Vt[:m] if m < X_train_c.shape[1] else None
        X_tr_pca = X_train_c @ Vm.T if Vm is not None else X_train_c
        X_va_pca = X_val_c   @ Vm.T if Vm is not None else X_val_c

        sr_pca = SoftmaxRegression(X_tr_pca.shape[1], n_cls)
        sr_pca.init_params(seed=42)
        _, best_p, _ = train_model(
            sr_pca, SGD(lr=0.05), X_tr_pca, y_train, X_va_pca, y_val,
            n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42, verbose=False)
        sr_pca.set_params(best_p)
        v_acc  = compute_accuracy(sr_pca, X_va_pca, y_val)
        v_loss = compute_loss(sr_pca, X_va_pca, y_val, n_cls)
        val_accs.append(v_acc); val_losses.append(v_loss)
        print(f"    m={m:3d} | Val Acc: {v_acc:.4f} | Val CE: {v_loss:.4f}")

    plot_pca_softmax_comparison(pca_dims, val_accs, val_losses,
                                filename="pca_softmax_comparison.png")

    with open(RESULTS_DIR / "track_a_pca.txt", "w") as f:
        f.write("Track A: PCA/SVD Analysis Results\n" + "=" * 50 + "\n")
        f.write("\nCumulative Explained Variance:\n")
        for k in [5, 10, 20, 40, 64]:
            f.write(f"  Top {k:2d} PCs: {cumvar[min(k, len(cumvar))-1]*100:.1f}%\n")
        f.write("\nSoftmax at PCA dimensions:\n")
        for m, va, vl in zip(pca_dims, val_accs, val_losses):
            f.write(f"  m={m:3d}: Val Acc={va:.4f}, Val CE={vl:.4f}\n")


# ── Experiment 7: Failure-Case Analysis ───────────────────

def run_failure_analysis():
    """Show that NN h=2 fails on 10-class digits due to a 2-D bottleneck.

    h = tanh(W1 x + b1) collapses 64-D input to 2 dimensions — not enough
    to separate 10 classes.  Compared against the adequate h=32 model.
    """
    print("\n" + "=" * 62)
    print("EXPERIMENT 7: FAILURE-CASE ANALYSIS")
    print("=" * 62)

    X_train, y_train, X_val, y_val, X_test, y_test = load_digits()
    n_cls, n_feat = 10, X_train.shape[1]

    print("\n  [NN h=2 — under-capacity]")
    nn_fail = NeuralNetwork(n_feat, 2, n_cls)
    nn_fail.init_params(seed=42)
    hist_fail, best_fail, _ = train_model(
        nn_fail, SGD(lr=0.05), X_train, y_train, X_val, y_val,
        n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42)
    nn_fail.set_params(best_fail)
    print(f"  Test Acc: {compute_accuracy(nn_fail, X_test, y_test):.4f}, "
          f"Test CE: {compute_loss(nn_fail, X_test, y_test, n_cls):.4f}")

    nn_good = NeuralNetwork(n_feat, 32, n_cls)
    nn_good.init_params(seed=42)
    hist_good, _, _ = train_model(
        nn_good, Adam(lr=0.001), X_train, y_train, X_val, y_val,
        n_classes=n_cls, n_epochs=200, batch_size=64, lam=1e-4, seed=42, verbose=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    epochs = range(1, len(hist_fail['train_loss']) + 1)

    for hist, color, tag in [(hist_fail, '#E74C3C', 'h=2 SGD'),
                              (hist_good, '#27AE60', 'h=32 Adam')]:
        axes[0].plot(epochs, hist['train_loss'], '-', color=color, alpha=0.5, linewidth=1, label=f'{tag} (train)')
        axes[0].plot(epochs, hist['val_loss'],   '-', color=color, linewidth=2.5,           label=f'{tag} (val)')
        axes[1].plot(epochs, hist['val_acc'],    '-', color=color, linewidth=2.5,           label=tag)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy Loss')
    axes[0].set_title('Loss: Under-capacity (h=2) vs Adequate (h=32)', fontweight='bold')
    axes[0].legend(fontsize=9, framealpha=0.9)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title('Accuracy: Under-capacity vs Adequate', fontweight='bold')
    axes[1].legend(fontsize=10, framealpha=0.9)

    fig.suptitle('Failure Case: Under-Capacity NN (h=2) on 10-Class Digits',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(get_figures_dir() / "failure_analysis.png",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: failure_analysis.png")

    acc_fail  = compute_accuracy(nn_fail, X_test, y_test)
    loss_fail = compute_loss(nn_fail, X_test, y_test, n_cls)
    with open(RESULTS_DIR / "failure_analysis.txt", "w") as f:
        f.write("Failure-Case Analysis: Under-Capacity NN (h=2) on 10-Class Digits\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"NN h=2  | Test Acc: {acc_fail:.4f} | Test CE: {loss_fail:.4f}\n\n")
        f.write("With only 2 hidden units the network creates a 2-D bottleneck:\n"
                "h = tanh(W1 x + b1) maps 64-D input to 2 dimensions, which is\n"
                "insufficient to separate 10 classes.  Both train and val loss\n"
                "remain high and accuracy plateaus far below what h=32 achieves.\n")


# ── Entry Point ───────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Math4AI Capstone — Track A: Full Experiment Suite")
    print("=" * 62)

    run_sanity_checks()
    run_synthetic_experiments()
    run_digits_experiment()
    run_capacity_ablation()
    run_optimizer_study()
    run_repeated_seed()
    run_track_a_pca()
    run_failure_analysis()

    print("\n" + "=" * 62)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Figures: {get_figures_dir()}")
    print(f"Results: {RESULTS_DIR}")
    print("=" * 62)


if __name__ == "__main__":
    main()