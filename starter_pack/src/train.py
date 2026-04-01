"""
Training loop for Math4AI Capstone.
Mini-batch training with validation checkpointing.
"""

import numpy as np
from .models import cross_entropy_loss, one_hot
from .data_utils import mini_batches


def compute_accuracy(model, X, y):
    """Compute classification accuracy."""
    preds, _ = model.predict(X)
    return np.mean(preds == y)


def compute_loss(model, X, y, n_classes, lam=0.0):
    """Compute cross-entropy loss (with optional L2 regularization)."""
    P, _ = model.forward(X)
    Y_oh = one_hot(y, n_classes)
    ce = cross_entropy_loss(P, Y_oh)

    # L2 regularization term
    if lam > 0:
        reg = 0.0
        for key in model.params:
            if key.startswith('W'):
                reg += 0.5 * lam * np.sum(model.params[key] ** 2)
        ce += reg
    return ce


def train_model(model, optimizer, X_train, y_train, X_val, y_val,
                n_classes, n_epochs=200, batch_size=64, lam=1e-4,
                seed=42, verbose=True):
    """Train a model with mini-batch gradient descent.

    Parameters
    ----------
    model : SoftmaxRegression or NeuralNetwork
    optimizer : SGD, Momentum, or Adam
    X_train, y_train : training data
    X_val, y_val : validation data
    n_classes : int
    n_epochs : int
    batch_size : int
    lam : float
        L2 regularization coefficient.
    seed : int
    verbose : bool

    Returns
    -------
    history : dict
        Contains train_loss, val_loss, train_acc, val_acc per epoch.
    best_params : dict
        Parameters at best validation cross-entropy.
    best_epoch : int
        Epoch of best validation cross-entropy.
    """
    rng = np.random.default_rng(seed)

    # Initialize optimizer state
    optimizer.init_state(model.params)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    best_val_loss = float('inf')
    best_params = model.get_params()
    best_epoch = 0

    for epoch in range(n_epochs):
        # Mini-batch training
        for X_batch, y_batch in mini_batches(X_train, y_train, batch_size, rng):
            Y_oh = one_hot(y_batch, n_classes)
            P, cache = model.forward(X_batch)
            model.backward(cache, Y_oh, lam=lam)
            optimizer.step(model.params, model.grads)

        # End-of-epoch evaluation
        train_loss = compute_loss(model, X_train, y_train, n_classes, lam=0)
        val_loss = compute_loss(model, X_val, y_val, n_classes, lam=0)
        train_acc = compute_accuracy(model, X_train, y_train)
        val_acc = compute_accuracy(model, X_val, y_val)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Checkpoint on best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = model.get_params()
            best_epoch = epoch

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"  Epoch {epoch:3d} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if verbose:
        print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return history, best_params, best_epoch