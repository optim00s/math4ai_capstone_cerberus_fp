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