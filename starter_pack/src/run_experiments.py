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