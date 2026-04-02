#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation sanity checks

Checks:
  1. Softmax rows sum to 1 (including extreme logits).
  2. CE loss decreases monotonically under full-batch SGD steps.
  3. Analytical gradients match finite-difference numerical gradients.
  4. No NaN / Inf appears in parameters after 30 training steps.
  5. NN can overfit 10 samples to 100 % training accuracy.

Usage: python -m starter_pack.src.sanity_checks  (from repo root)
"""

import sys, io, numpy as np
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starter_pack.src.models import SoftmaxRegression, NeuralNetwork, stable_softmax, cross_entropy_loss, one_hot
from starter_pack.src.optimizers import SGD, Adam

RESULTS_DIR = REPO_ROOT / 'starter_pack' / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


class Logger:
    """Tees all output to both console and a results file."""
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

    # ── Check 1: Softmax rows sum to 1 ───────────────────
    print("\n[Check 1] Softmax probabilities sum exactly to 1")
    Z = np.array([
        [1.0,  2.0,   3.0],
        [0.0,  0.0,   0.0],      # uniform logits → uniform output
        [-100.0, 0.0, 100.0],    # extreme range tests numerical stability
    ])
    P = stable_softmax(Z)
    sums = np.sum(P, axis=1)
    print(f"  Logits Z:\n{Z}")
    print(f"  Softmax(Z):\n{np.round(P, 6)}")
    print(f"  Row sums: {sums}")

    if np.allclose(sums, 1.0, atol=1e-12):
        print("  CHECK 1 PASS: all rows sum to 1 within machine precision")
    else:
        print("  CHECK 1 FAIL: row sums deviate from 1")
        passed_all = False

    # ── Check 2: Loss decreases on a tiny full-batch ─────
    print("\n[Check 2] Loss decreases monotonically after first few SGD steps")
    rng = np.random.default_rng(99)
    X_small = rng.standard_normal((50, 4))
    y_small = rng.integers(0, 3, size=50)
    Y_small_oh = one_hot(y_small, 3)

    sr = SoftmaxRegression(n_features=4, n_classes=3)
    sr.init_params(seed=99)
    opt = SGD(lr=0.1)
    opt.init_state(sr.params)

    losses = []
    print("  First 10 SGD steps (50 samples, 4 features, 3 classes):")
    for step in range(10):
        P_b, cache = sr.forward(X_small)
        loss = cross_entropy_loss(P_b, Y_small_oh)
        losses.append(loss)
        sr.backward(cache, Y_small_oh, lam=0.0)
        opt.step(sr.params, sr.grads)
        print(f"    step {step:2d} | loss = {loss:.4f}")

    if all(losses[i] < losses[i - 1] for i in range(1, len(losses))):
        print(f"  CHECK 2 PASS: {losses[0]:.4f} → {losses[-1]:.4f} (monotone ↓)")
    else:
        print("  CHECK 2 FAIL: loss is not monotonically decreasing")
        passed_all = False

    # ── Check 3: Numerical gradient check ────────────────
    print("\n[Check 3] Analytical vs finite-difference gradients")
    rng2 = np.random.default_rng(1)
    X_g  = rng2.standard_normal((8, 4))
    y_g  = rng2.integers(0, 3, size=8)
    Y_g  = one_hot(y_g, 3)

    nn  = NeuralNetwork(n_features=4, n_hidden=6, n_classes=3)
    nn.init_params(seed=42)
    P_g, cache_g = nn.forward(X_g)
    nn.backward(cache_g, Y_g, lam=0.0)

    eps = 1e-5
    max_diffs = {}
    for key in ['W1', 'b1', 'W2', 'b2']:
        flat     = nn.params[key].ravel()
        grad_a   = nn.grads[key].ravel()
        n_check  = min(6, len(flat))
        diffs    = []
        for i in range(n_check):
            orig = flat[i]
            flat[i] = orig + eps
            nn.params[key] = flat.reshape(nn.params[key].shape)
            lp = cross_entropy_loss(nn.forward(X_g)[0], Y_g)
            flat[i] = orig - eps
            nn.params[key] = flat.reshape(nn.params[key].shape)
            lm = cross_entropy_loss(nn.forward(X_g)[0], Y_g)
            flat[i] = orig  # restore
            nn.params[key] = flat.reshape(nn.params[key].shape)
            diffs.append(abs(grad_a[i] - (lp - lm) / (2 * eps)))
        max_diffs[key] = max(diffs)
        print(f"  {key:4s}: max |analytical − numerical| = {max_diffs[key]:.2e}")

    if all(v < 1e-4 for v in max_diffs.values()):
        print("  CHECK 3 PASS: all discrepancies < 1e-4")
    else:
        print("  CHECK 3 FAIL: gradient discrepancy too large")
        passed_all = False

    # ── Check 4: No NaN / Inf after training ─────────────
    print("\n[Check 4] No NaN or Inf in parameters after 30 SGD steps")
    X_nan = rng2.standard_normal((50, 4))
    y_nan = rng2.integers(0, 3, size=50)
    Y_nan = one_hot(y_nan, 3)

    nn2 = NeuralNetwork(n_features=4, n_hidden=8, n_classes=3)
    nn2.init_params(seed=7)
    opt2 = SGD(lr=0.05)
    opt2.init_state(nn2.params)
    for _ in range(30):
        Pn, cn = nn2.forward(X_nan)
        nn2.backward(cn, Y_nan, lam=1e-4)
        opt2.step(nn2.params, nn2.grads)

    has_bad = any(np.any(np.isnan(v)) or np.any(np.isinf(v))
                  for v in nn2.params.values())
    print(f"  NaN/Inf detected: {has_bad}")
    if not has_bad:
        print("  CHECK 4 PASS: all parameters are finite")
    else:
        print("  CHECK 4 FAIL: NaN or Inf found")
        passed_all = False

    # ── Check 5: Overfit tiny dataset ────────────────────
    print("\n[Check 5] NN can overfit 10 samples to 100 % training accuracy")
    X_tiny = np.array([
        [ 1.5,  0.8], [-1.3, -0.9], [ 1.1, -1.2], [-0.7,  1.4], [ 0.3,  0.2],
        [-1.0, -0.5], [ 0.9,  1.1], [-1.4,  0.6], [ 0.5, -1.0], [-0.2,  0.8],
    ])
    y_tiny = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    Y_tiny = one_hot(y_tiny, 2)

    nn3 = NeuralNetwork(n_features=2, n_hidden=16, n_classes=2)
    nn3.init_params(seed=0)
    opt3 = Adam(lr=0.05)
    opt3.init_state(nn3.params)
    for _ in range(200):
        Pof, cof = nn3.forward(X_tiny)
        nn3.backward(cof, Y_tiny, lam=0.0)
        opt3.step(nn3.params, nn3.grads)

    preds, _ = nn3.predict(X_tiny)
    Pf, _    = nn3.forward(X_tiny)
    loss_f   = cross_entropy_loss(Pf, Y_tiny)
    acc_f    = np.mean(preds == y_tiny)
    print(f"  Final loss: {loss_f:.6f} | Final accuracy: {acc_f * 100:.1f}%")

    if acc_f >= 1.0 and loss_f < 0.05:
        print("  CHECK 5 PASS: perfect overfit achieved")
    else:
        print("  CHECK 5 FAIL: could not overfit tiny dataset")
        passed_all = False

    # ── Summary ───────────────────────────────────────────
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