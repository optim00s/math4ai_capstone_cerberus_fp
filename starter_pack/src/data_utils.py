"""
Data loading and mini-batch utilities.
"""

import numpy as np
from pathlib import Path


def get_data_dir():
    return Path(__file__).resolve().parents[1] / "data"


def load_synthetic(name):
    """Load a synthetic dataset ('linear_gaussian' or 'moons').
    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    data = np.load(get_data_dir() / f"{name}.npz")
    return (data['X_train'], data['y_train'],
            data['X_val'],   data['y_val'],
            data['X_test'],  data['y_test'])


def load_digits():
    """Load the digits dataset with the fixed split indices.
    Pixels are already normalized to [0, 1].
    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    data_dir = get_data_dir()
    digits = np.load(data_dir / "digits_data.npz")
    splits = np.load(data_dir / "digits_split_indices.npz")
    X, y = digits['X'], digits['y']
    return (X[splits['train_idx']], y[splits['train_idx']],
            X[splits['val_idx']],   y[splits['val_idx']],
            X[splits['test_idx']],  y[splits['test_idx']])


def mini_batches(X, y, batch_size, rng=None):
    """Yield randomly shuffled mini-batches — each sample appears once per epoch."""
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start:min(start + batch_size, n)]
        yield X[idx], y[idx]