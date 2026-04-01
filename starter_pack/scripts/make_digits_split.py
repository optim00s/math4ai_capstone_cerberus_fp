#!/usr/bin/env python3
"""Create the fixed digits data file and deterministic split indices."""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits


SEED = 7
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20


def stratified_indices(y: np.ndarray, rng: np.random.Generator):
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


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data"
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0
    y = digits.target.astype(np.int64)

    np.savez_compressed(out_dir / "digits_data.npz", X=X, y=y)

    rng = np.random.default_rng(SEED)
    train_idx, val_idx, test_idx = stratified_indices(y, rng)
    np.savez_compressed(
        out_dir / "digits_split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print(
        "Wrote digits_data.npz and digits_split_indices.npz "
        f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})"
    )


if __name__ == "__main__":
    main()
