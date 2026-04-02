"""
Core model implementations — Softmax Regression and one-hidden-layer NN.
"""

import numpy as np


# ── Utility functions ─────────────────────────────────────

def stable_softmax(Z):
    """Row-wise softmax with max-shift to prevent overflow."""
    Z_shifted = Z - Z.max(axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / exp_Z.sum(axis=1, keepdims=True)


def cross_entropy_loss(P, Y_onehot):
    """Mean negative log-likelihood over the batch."""
    n = P.shape[0]
    eps = 1e-12  # avoid log(0)
    return np.sum(Y_onehot * -np.log(np.clip(P, eps, 1.0))) / n


def one_hot(y, k):
    """Convert integer labels to one-hot matrix of shape (n, k)."""
    n = len(y)
    Y = np.zeros((n, k))
    Y[np.arange(n), y.astype(int)] = 1.0
    return Y


# ── Softmax Regression ────────────────────────────────────

class SoftmaxRegression:
    """Linear classifier: s = Wx + b, then softmax.
    Decision boundary is always a hyperplane (linear model).
    """

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.params = {}
        self.grads = {}

    def init_params(self, seed=None):
        # Xavier scale: keeps gradient magnitudes stable at init
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (self.n_features + self.n_classes))
        self.params['W'] = rng.normal(0, scale, (self.n_classes, self.n_features))
        self.params['b'] = np.zeros((1, self.n_classes))

    def forward(self, X):
        S = X @ self.params['W'].T + self.params['b']  # linear scores (logits)
        P = stable_softmax(S)                           # convert to probabilities
        cache = {'X': X, 'P': P}                        # save for backward
        return P, cache

    def backward(self, cache, Y_onehot, lam=0.0):
        X, P = cache['X'], cache['P']
        n = X.shape[0]
        dS = (P - Y_onehot) / n                               # d(CE)/d(S) — softmax CE simplifies nicely
        self.grads['W'] = dS.T @ X + lam * self.params['W']  # + L2 regularization term
        self.grads['b'] = dS.sum(axis=0, keepdims=True)

    def predict(self, X):
        P, _ = self.forward(X)
        return np.argmax(P, axis=1), P

    def get_params(self):
        """Return a deep copy of parameters."""
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        """Set parameters from a dict."""
        for k, v in params.items():
            self.params[k] = v.copy()


# ── One-Hidden-Layer Neural Network ──────────────────────

class NeuralNetwork:
    """Single hidden layer with tanh activation and softmax output.

    The hidden layer learns a nonlinear transformation of the input,
    allowing the model to separate non-linearly separable classes.

    Forward: Z1 = XW1^T + b1, H = tanh(Z1), S = HW2^T + b2, P = softmax(S)
    """

    def __init__(self, n_features, n_hidden, n_classes):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.params = {}
        self.grads = {}

    def init_params(self, seed=None):
        # Xavier init for each layer using its own fan-in + fan-out
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
        Z1 = X @ self.params['W1'].T + self.params['b1']  # linear pre-activation, hidden layer
        H  = np.tanh(Z1)                                   # nonlinear hidden representation
        S  = H @ self.params['W2'].T + self.params['b2']  # linear pre-activation, output layer
        P  = stable_softmax(S)                             # class probabilities
        cache = {'X': X, 'Z1': Z1, 'H': H, 'S': S, 'P': P}  # save for backward
        return P, cache

    def backward(self, cache, Y_onehot, lam=0.0):
        X, H, P = cache['X'], cache['H'], cache['P']
        n = X.shape[0]

        # Output layer gradients
        dS  = (P - Y_onehot) / n                          # d(CE)/d(S) — softmax CE simplifies nicely
        dW2 = dS.T @ H + lam * self.params['W2']          # + L2 regularization term
        db2 = dS.sum(axis=0, keepdims=True)

        # Hidden layer gradients — chain rule back through tanh
        dH  = dS @ self.params['W2']                      # error signal reaching hidden units
        dZ1 = dH * (1 - H * H)                            # tanh'(z) = 1 - tanh²(z)
        dW1 = dZ1.T @ X + lam * self.params['W1']
        db1 = dZ1.sum(axis=0, keepdims=True)

        self.grads.update({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2})

    def predict(self, X):
        P, _ = self.forward(X)
        return np.argmax(P, axis=1), P

    def get_params(self):
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, params):
        for k, v in params.items():
            self.params[k] = v.copy()