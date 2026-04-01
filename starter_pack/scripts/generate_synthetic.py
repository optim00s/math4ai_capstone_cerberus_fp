#!/usr/bin/env python3
"""Generate the fixed synthetic datasets for the Math4AI capstone starter pack."""

from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons


SEED = 7
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20


def stratified_split(y: np.ndarray, rng: np.random.Generator):
    train_parts = []
    val_parts = []
    test_parts = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_idx = rng.permutation(cls_idx)
        n_total = len(cls_idx)
        n_train = int(TRAIN_FRAC * n_total)
        n_val = int(VAL_FRAC * n_total)
        train_parts.append(cls_idx[:n_train])
        val_parts.append(cls_idx[n_train : n_train + n_val])
        test_parts.append(cls_idx[n_train + n_val :])

    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    return train_idx, val_idx, test_idx


def package_dataset(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    train_idx, val_idx, test_idx = stratified_split(y, rng)
    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
    }


def build_linear_gaussian(rng: np.random.Generator):
    n_per_class = 200
    mean0 = np.array([-1.2, -0.7])
    mean1 = np.array([1.1, 0.9])
    cov = np.array([[0.85, 0.18], [0.18, 0.75]])
    x0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    x1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([x0, x1]).astype(np.float64)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=np.int64), np.ones(n_per_class, dtype=np.int64)]
    )
    return X, y


def build_moons():
    X, y = make_moons(n_samples=400, noise=0.18, random_state=SEED)
    return X.astype(np.float64), y.astype(np.int64)


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data"
    rng = np.random.default_rng(SEED)

    linear_X, linear_y = build_linear_gaussian(rng)
    linear_data = package_dataset(linear_X, linear_y, rng)
    np.savez_compressed(out_dir / "linear_gaussian.npz", **linear_data)

    moons_X, moons_y = build_moons()
    moons_data = package_dataset(moons_X, moons_y, rng)
    np.savez_compressed(out_dir / "moons.npz", **moons_data)

    print("Wrote linear_gaussian.npz and moons.npz")


if __name__ == "__main__":
    main()
