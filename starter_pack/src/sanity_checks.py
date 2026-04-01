#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation Sanity Checks for Math4AI Capstone.

This standalone script automatically verifies the 5 core implementation
checks described in Section 5.5 of the assignment and the report:

  1. Softmax probabilities sum exactly to 1 (including extreme logits).
  2. Loss decreases monotonically after the first few SGD steps.
  3. Numerical gradient verification (analytical vs finite-difference).
  4. No NaN or Inf values produced in parameters during training.
  5. Neural network can overfit a tiny dataset to near-perfect accuracy.

Usage:
    python -m starter_pack.src.sanity_checks
    (from the repository root)
"""

import sys
import io
import numpy as np
from pathlib import Path

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add the repo root to path so imports work
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starter_pack.src.models import (
    SoftmaxRegression, NeuralNetwork,
    stable_softmax, cross_entropy_loss, one_hot
)
from starter_pack.src.optimizers import SGD, Adam

RESULTS_DIR = REPO_ROOT / 'starter_pack' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class Logger:
    """Write to both console and results file simultaneously."""
    def __init__(self, filepath):
        self.console = sys.stdout
        self.file = open(filepath, 'w', encoding='utf-8')

    def write(self, msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    log_path = RESULTS_DIR / 'sanity_checks.txt'
    logger = Logger(log_path)
    sys.stdout = logger

    print("=" * 62)
    print("  MATH4AI CAPSTONE: IMPLEMENTATION SANITY CHECKS")
    print("=" * 62)

    passed_all = True

    # ----------------------------------------------------------
    # CHECK 1: Softmax probabilities sum to 1
    # ----------------------------------------------------------
    print("\n[Check 1] Softmax probabilities sum exactly to 1")
    Z = np.array([
        [1.0,  2.0,   3.0],
        [0.0,  0.0,   0.0],     # uniform logits
        [-100.0, 0.0, 100.0],   # extreme values — tests numerical stability
    ])
    P = stable_softmax(Z)
    sums = np.sum(P, axis=1)
    print(f"  Input logits Z:\n{Z}")
    print(f"  Softmax(Z):\n{np.round(P, 6)}")
    print(f"  Row sums: {sums}")

    if np.allclose(sums, 1.0, atol=1e-12):
        print("  CHECK 1 PASS: all rows sum to 1 within machine precision")
    else:
        print("  CHECK 1 FAIL: row sums deviate from 1")
        passed_all = False

    # ----------------------------------------------------------
    # CHECK 2: Loss decreases monotonically on a tiny batch
    # ----------------------------------------------------------
    print("\n[Check 2] Loss decreases monotonically after the first few SGD steps")
    rng = np.random.default_rng(99)
    X_small = rng.standard_normal((50, 4))
    y_small = rng.integers(0, 3, size=50)
    Y_small_oh = one_hot(y_small, 3)

    sr = SoftmaxRegression(n_features=4, n_classes=3)
    sr.init_params(seed=99)
    opt_sgd = SGD(lr=0.1)
    opt_sgd.init_state(sr.params)

    losses = []
    print("  Tracking loss for first 10 SGD steps (50 samples, 4 features, 3 classes):")
    for step in range(10):
        P_b, cache_b = sr.forward(X_small)
        loss = cross_entropy_loss(P_b, Y_small_oh)
        losses.append(loss)
        sr.backward(cache_b, Y_small_oh, lam=0.0)
        opt_sgd.step(sr.params, sr.grads)
        print(f"    step {step:2d} | loss = {loss:.4f}")

    # Check if the sequence is strictly monotonically decreasing
    is_decreasing = all(losses[i] < losses[i-1] for i in range(1, len(losses)))
    
    if is_decreasing:
        print(f"  CHECK 2 PASS: loss decreased monotonically from {losses[0]:.4f} → {losses[-1]:.4f}")
    else:
        print("  CHECK 2 FAIL: loss did not decrease monotonically")
        passed_all = False

    # ----------------------------------------------------------
    # CHECK 3: Numerical gradient verification
    # ----------------------------------------------------------
    print("\n[Check 3] Numerical gradient check (analytical vs finite-difference)")
    rng2 = np.random.default_rng(1)
    X_g = rng2.standard_normal((8, 4))
    y_g = rng2.integers(0, 3, size=8)
    Y_g_oh = one_hot(y_g, 3)

    nn_g = NeuralNetwork(n_features=4, n_hidden=6, n_classes=3)
    nn_g.init_params(seed=42)

    P_g, cache_g = nn_g.forward(X_g)
    nn_g.backward(cache_g, Y_g_oh, lam=0.0)

    eps = 1e-5
    max_diffs = {}
    for param_key in ['W1', 'b1', 'W2', 'b2']:
        grad_analytical = nn_g.grads[param_key]
        flat = nn_g.params[param_key].ravel()
        grad_flat_analytical = grad_analytical.ravel()
        
        n_check = min(6, len(flat))
        diffs = []
        for idx in range(n_check):
            orig = flat[idx]
            
            # f(x + eps)
            flat[idx] = orig + eps
            nn_g.params[param_key] = flat.reshape(nn_g.params[param_key].shape)
            P_p, _ = nn_g.forward(X_g)
            loss_p = cross_entropy_loss(P_p, Y_g_oh)

            # f(x - eps)
            flat[idx] = orig - eps
            nn_g.params[param_key] = flat.reshape(nn_g.params[param_key].shape)
            P_m, _ = nn_g.forward(X_g)
            loss_m = cross_entropy_loss(P_m, Y_g_oh)

            # Restore original value
            flat[idx] = orig
            nn_g.params[param_key] = flat.reshape(nn_g.params[param_key].shape)

            grad_num = (loss_p - loss_m) / (2 * eps)
            diffs.append(abs(grad_flat_analytical[idx] - grad_num))

        max_diffs[param_key] = max(diffs)
        print(f"  {param_key:4s}: max |analytical − numerical| = {max_diffs[param_key]:.2e}")

    grad_ok = all(v < 1e-4 for v in max_diffs.values())
    if grad_ok:
        print("  CHECK 3 PASS: all gradient discrepancies < 1e-4")
    else:
        print("  CHECK 3 FAIL: gradient discrepancy too large")
        passed_all = False

    # ----------------------------------------------------------
    # CHECK 4: No NaN or Inf in parameters after training
    # ----------------------------------------------------------
    print("\n[Check 4] No NaN or Inf values produced during training")
    X_nan = rng2.standard_normal((50, 4))
    y_nan = rng2.integers(0, 3, size=50)
    Y_nan_oh = one_hot(y_nan, 3)

    nn_nan = NeuralNetwork(n_features=4, n_hidden=8, n_classes=3)
    nn_nan.init_params(seed=7)
    opt_nan = SGD(lr=0.05)
    opt_nan.init_state(nn_nan.params)

    for _ in range(30):
        P_n, cache_n = nn_nan.forward(X_nan)
        nn_nan.backward(cache_n, Y_nan_oh, lam=1e-4)
        opt_nan.step(nn_nan.params, nn_nan.grads)

    has_bad = any(
        np.any(np.isnan(v)) or np.any(np.isinf(v))
        for v in nn_nan.params.values()
    )
    print(f"  NaN/Inf detected in parameters after 30 epochs: {has_bad}")
    if not has_bad:
        print("  CHECK 4 PASS: parameters are all finite")
    else:
        print("  CHECK 4 FAIL: NaN or Inf found in parameters")
        passed_all = False

    # ----------------------------------------------------------
    # CHECK 5: Overfit a tiny dataset to near-perfect accuracy
    # ----------------------------------------------------------
    print("\n[Check 5] Neural network can overfit a tiny dataset (10 samples) to 100% accuracy")
    X_tiny = np.array([
        [ 1.5,  0.8], [-1.3, -0.9], [ 1.1, -1.2], [-0.7,  1.4], [ 0.3,  0.2],
        [-1.0, -0.5], [ 0.9,  1.1], [-1.4,  0.6], [ 0.5, -1.0], [-0.2,  0.8],
    ])
    y_tiny = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    Y_tiny_oh = one_hot(y_tiny, 2)

    nn_of = NeuralNetwork(n_features=2, n_hidden=16, n_classes=2)
    nn_of.init_params(seed=0)
    opt_of = Adam(lr=0.05)
    opt_of.init_state(nn_of.params)

    for _ in range(200):
        P_of, cache_of = nn_of.forward(X_tiny)
        nn_of.backward(cache_of, Y_tiny_oh, lam=0.0)
        opt_of.step(nn_of.params, nn_of.grads)

    preds_of, _ = nn_of.predict(X_tiny)
    
    P_final, _ = nn_of.forward(X_tiny)
    final_loss = cross_entropy_loss(P_final, Y_tiny_oh)
    final_acc = np.mean(preds_of == y_tiny)
    
    print(f"  Final training loss : {final_loss:.6f}")
    print(f"  Final training accuracy: {final_acc * 100:.1f}%  (expected 100%)")

    if final_acc >= 1.0 and final_loss < 0.05:
        print("  CHECK 5 PASS: perfect overfit achieved — forward/backward passes are consistent")
    else:
        print("  CHECK 5 FAIL: could not overfit tiny dataset")
        passed_all = False

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 62)
    if passed_all:
        print("  STATUS: ALL 5 SANITY CHECKS PASSED  [OK]")
    else:
        print("  STATUS: ONE OR MORE CHECKS FAILED   [WARNING]")
    print("=" * 62)
    print(f"\nResults written to: {log_path}")

    sys.stdout = logger.console
    logger.close()


if __name__ == '__main__':
    main()