"""
Training loop with mini-batch GD and best-validation-loss checkpointing.
"""

import numpy as np
from .models import cross_entropy_loss, one_hot
from .data_utils import mini_batches


def compute_accuracy(model, X, y):
    preds, _ = model.predict(X)
    return np.mean(preds == y)


def compute_loss(model, X, y, n_classes, lam=0.0):
    """Cross-entropy loss, plus L2 weight-decay penalty if lam > 0.
    Biases are not regularized (only keys starting with 'W').
    """
    P, _ = model.forward(X)
    ce = cross_entropy_loss(P, one_hot(y, n_classes))
    if lam > 0:
        reg = sum(0.5 * lam * np.sum(v ** 2)
                  for k, v in model.params.items() if k.startswith('W'))
        ce += reg
    return ce


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                n_classes, n_epochs=200, batch_size=64, lam=1e-4,
                seed=42, verbose=True):
    """Mini-batch training loop.

    Saves a parameter snapshot whenever validation loss improves (simple
    early-stopping checkpoint, but training continues to completion).

    Returns history dict, best_params, and the epoch of best val loss.
    """
    rng = np.random.default_rng(seed)
    optimizer.init_state(model.params)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_params   = model.get_params()
    best_epoch    = 0

    for epoch in range(n_epochs):
        for X_batch, y_batch in mini_batches(X_train, y_train, batch_size, rng):
            Y_oh = one_hot(y_batch, n_classes)
            P, cache = model.forward(X_batch)
            model.backward(cache, Y_oh, lam=lam)
            optimizer.step(model.params, model.grads)

        # Evaluate without regularization so metrics are comparable across lam values
        train_loss = compute_loss(model, X_train, y_train, n_classes, lam=0)
        val_loss   = compute_loss(model, X_val,   y_val,   n_classes, lam=0)
        train_acc  = compute_accuracy(model, X_train, y_train)
        val_acc    = compute_accuracy(model, X_val,   y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params   = model.get_params()
            best_epoch    = epoch

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return history, best_params, best_epoch