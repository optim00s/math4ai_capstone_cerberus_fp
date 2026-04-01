"""
Optimizers for Math4AI Capstone
SGD, Momentum, and Adam
"""

import numpy as np


class SGD:
    """Vanilla mini-batch Stochastic Gradient Descent."""

    def __init__(self, lr=0.05):
        self.lr = lr

    def init_state(self, params):
        """No extra state needed for vanilla SGD."""
        pass

    def step(self, params, grads):
        """Update parameters in-place.

        Parameters
        ----------
        params : dict
            Model parameters (modified in-place).
        grads : dict
            Gradients with matching keys.
        """
        for key in params:
            params[key] -= self.lr * grads[key]


class Momentum:
    """SGD with momentum.

    v_t = μ * v_{t-1} + grad
    θ_t = θ_{t-1} - lr * v_t
    """

    def __init__(self, lr=0.05, mu=0.9):
        self.lr = lr
        self.mu = mu
        self.v = {}

    def init_state(self, params):
        """Initialize velocity buffers to zeros."""
        self.v = {key: np.zeros_like(val) for key, val in params.items()}

    def step(self, params, grads):
        for key in params:
            self.v[key] = self.mu * self.v[key] + grads[key]
            params[key] -= self.lr * self.v[key]


class Adam:
    """Adam optimizer.

    m_t = β1 * m_{t-1} + (1-β1) * g_t
    v_t = β2 * v_{t-1} + (1-β2) * g_t^2
    m_hat = m_t / (1 - β1^t)
    v_hat = v_t / (1 - β2^t)
    θ_t = θ_{t-1} - lr * m_hat / (sqrt(v_hat) + ε)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def init_state(self, params):
        """Initialize first and second moment estimates."""
        self.m = {key: np.zeros_like(val) for key, val in params.items()}
        self.v = {key: np.zeros_like(val) for key, val in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)