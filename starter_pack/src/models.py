"""
Core model implementations for Math4AI Capstone.
Softmax Regression and One-Hidden-Layer Neural Network (tanh + softmax).
"""

import numpy as np


# ─────────────────────────────────────────────
# Utility functions
# ─────────────────────────────────────────────

def stable_softmax(Z):
    """Row-wise numerically stable softmax.

    Parameters
    ----------
    Z : ndarray, shape (n, k)
        Pre-activation logits.

    Returns
    -------
    P : ndarray, shape (n, k)
        Probability matrix with rows summing to 1.
    """
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss(P, Y_onehot):
    """Mean cross-entropy loss (negative log-likelihood).

    Parameters
    ----------
    P : ndarray, shape (n, k)
        Predicted probabilities.
    Y_onehot : ndarray, shape (n, k)
        One-hot encoded labels.

    Returns
    -------
    float
        Average cross-entropy over the batch.
    """
    n = P.shape[0]
    eps = 1e-12  # for numerical stability
    log_probs = -np.log(np.clip(P, eps, 1.0))
    return np.sum(Y_onehot * log_probs) / n


def one_hot(y, k):
    """Convert integer labels to one-hot encoding.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Integer class labels.
    k : int
        Number of classes.

    Returns
    -------
    Y : ndarray, shape (n, k)
        One-hot encoded matrix.
    """
    n = len(y)
    Y = np.zeros((n, k))
    Y[np.arange(n), y.astype(int)] = 1.0
    return Y


# ─────────────────────────────────────────────
# Softmax Regression (Baseline Linear Model)
# ─────────────────────────────────────────────

class SoftmaxRegression:
    """Multiclass softmax (logistic) regression.

    s(x) = Wx + b, then softmax to get probabilities.
    """

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.params = {}
        self.grads = {}

    def init_params(self, seed=None):
        """Initialize W to small random values and b to zeros."""
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (self.n_features + self.n_classes))
        self.params['W'] = rng.normal(0, scale, (self.n_classes, self.n_features))
        self.params['b'] = np.zeros((1, self.n_classes))

    def forward(self, X):
        """Forward pass: compute logits and probabilities.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Input data (row-wise).

        Returns
        -------
        P : ndarray, shape (n, k)
            Predicted probabilities.
        cache : dict
            Intermediate values for backpropagation.
        """
        S = X @ self.params['W'].T + self.params['b']  # (n, k)
        P = stable_softmax(S)
        cache = {'X': X, 'P': P}
        return P, cache

    def backward(self, cache, Y_onehot, lam=0.0):
        """Backward pass: compute parameter gradients.

        Parameters
        ----------
        cache : dict
            From forward pass.
        Y_onehot : ndarray, shape (n, k)
            One-hot labels.
        lam : float
            L2 regularization coefficient.
        """
        X = cache['X']
        P = cache['P']
        n = X.shape[0]

        dS = (P - Y_onehot) / n            # (n, k)
        dW = dS.T @ X + lam * self.params['W']  # (k, d)
        db = dS.sum(axis=0, keepdims=True)  # (1, k)

        self.grads['W'] = dW
        self.grads['b'] = db

    def predict(self, X):
        """Predict class labels.

        Returns
        -------
        predictions : ndarray, shape (n,)
        P : ndarray, shape (n, k)
        """
        P, _ = self.forward(X)
        return np.argmax(P, axis=1), P

    def get_params(self):
        """Return a deep copy of parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        """Set parameters from a dict."""
        for k, v in params.items():
            self.params[k] = v.copy()


# ─────────────────────────────────────────────
# One-Hidden-Layer Neural Network
# ─────────────────────────────────────────────

class NeuralNetwork:
    """One-hidden-layer neural network with tanh activation and softmax output.

    h = tanh(X @ W1.T + b1)
    s = h @ W2.T + b2
    P = softmax(s)
    """

    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.params = {}
        self.grads = {}

    def init_params(self, seed=None):
        """Xavier initialization for both layers."""
        rng = np.random.default_rng(seed)

        # Layer 1: input → hidden
        scale1 = np.sqrt(2.0 / (self.n_features + self.n_hidden))
        self.params['W1'] = rng.normal(0, scale1, (self.n_hidden, self.n_features))
        self.params['b1'] = np.zeros((1, self.n_hidden))

        # Layer 2: hidden → output
        scale2 = np.sqrt(2.0 / (self.n_hidden + self.n_classes))
        self.params['W2'] = rng.normal(0, scale2, (self.n_classes, self.n_hidden))
        self.params['b2'] = np.zeros((1, self.n_classes))

    def forward(self, X):
        """Forward pass through the network.

        Parameters
        ----------
        X : ndarray, shape (n, d)
            Input data (row-wise).

        Returns
        -------
        P : ndarray, shape (n, k)
            Predicted probabilities.
        cache : dict
            Intermediate values for backpropagation.
        """
        # Hidden layer
        Z1 = X @ self.params['W1'].T + self.params['b1']  # (n, h)
        H = np.tanh(Z1)                                    # (n, h)

        # Output layer
        S = H @ self.params['W2'].T + self.params['b2']    # (n, k)
        P = stable_softmax(S)                               # (n, k)

        cache = {'X': X, 'Z1': Z1, 'H': H, 'S': S, 'P': P}
        return P, cache

    def backward(self, cache, Y_onehot, lam=0.0):
        """Backward pass: compute gradients for both layers.

        Derivations (vectorized form from the assignment):
        ∂L/∂S  = (1/n)(P - Y)
        ∂L/∂W2 = (∂L/∂S)^T H
        ∂L/∂b2 = (∂L/∂S)^T 1
        ∂L/∂Z1 = (∂L/∂S) W2 ⊙ (1 - H⊙H)
        ∂L/∂W1 = (∂L/∂Z1)^T X
        ∂L/∂b1 = (∂L/∂Z1)^T 1

        Parameters
        ----------
        cache : dict
            From forward pass.
        Y_onehot : ndarray, shape (n, k)
            One-hot labels.
        lam : float
            L2 regularization coefficient.
        """
        X = cache['X']
        H = cache['H']
        P = cache['P']
        n = X.shape[0]

        # Output layer gradients
        dS = (P - Y_onehot) / n                                # (n, k)
        dW2 = dS.T @ H + lam * self.params['W2']               # (k, h)
        db2 = dS.sum(axis=0, keepdims=True)                     # (1, k)

        # Hidden layer gradients (chain rule through tanh)
        dH = dS @ self.params['W2']                             # (n, h)
        dZ1 = dH * (1 - H * H)                                 # (n, h)  tanh derivative
        dW1 = dZ1.T @ X + lam * self.params['W1']              # (h, d)
        db1 = dZ1.sum(axis=0, keepdims=True)                    # (1, h)

        self.grads['W1'] = dW1
        self.grads['b1'] = db1
        self.grads['W2'] = dW2
        self.grads['b2'] = db2

    def predict(self, X):
        """Predict class labels.

        Returns
        -------
        predictions : ndarray, shape (n,)
        P : ndarray, shape (n, k)
        """
        P, _ = self.forward(X)
        return np.argmax(P, axis=1), P

    def get_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        for k, v in params.items():
            self.params[k] = v.copy()