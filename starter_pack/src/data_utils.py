"""
Data loading and mini-batch utilities for Math4AI Capstone.
"""

import numpy as np
from pathlib import Path


def get_data_dir():
    """Return path to the data directory."""
    return Path(__file__).resolve().parents[1] / "data"


def load_synthetic(name):
    """Load a synthetic dataset (linear_gaussian or moons).

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    data = np.load(get_data_dir() / f"{name}.npz")
    return (data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'])


def load_digits():
    """Load the digits dataset with the fixed split indices.

    Features are already scaled to [0,1] — no additional preprocessing.

    Returns
    -------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    data_dir = get_data_dir()
    digits = np.load(data_dir / "digits_data.npz")
    splits = np.load(data_dir / "digits_split_indices.npz")

    X, y = digits['X'], digits['y']
    train_idx = splits['train_idx']
    val_idx = splits['val_idx']
    test_idx = splits['test_idx']

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def mini_batches(X, y, batch_size, rng=None):
    """Generate random mini-batches.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    batch_size : int
    rng : numpy Generator or None

    Yields
    ------
    X_batch, y_batch
    """
    n = X.shape[0]
    if rng is None:
        rng = np.random.default_rng()

    indices = rng.permutation(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]