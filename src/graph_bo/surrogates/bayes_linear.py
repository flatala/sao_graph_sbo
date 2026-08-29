from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["BayesianLinearSurrogate"]


class _BLR:
    """Single-output Bayesian linear regression with evidence-maximized alpha/beta.

    Model: y = Phi w + eps, eps ~ N(0, 1/beta), w ~ N(0, 1/alpha I).
    alpha (prior precision) and beta (noise precision) are fit by maximizing the
    log marginal likelihood (Bishop, PRML 3.5).
    """

    def __init__(self, n_iter: int = 100, tol: float = 1e-3):
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.alpha = 1.0
        self.beta = 1.0
        self.m_n: np.ndarray | None = None
        self.s_n: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_std = 1.0

    # Keep alpha/beta in a sane range so the posterior precision stays well-conditioned
    # even when Phi^T Phi is rank-deficient (constant/duplicated activeness columns).
    _FLOOR = 1e-8
    _CEIL = 1e8

    def fit(self, phi: np.ndarray, y: np.ndarray) -> "_BLR":
        phi = np.asarray(phi, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, _ = phi.shape

        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        t = (y - self.y_mean) / self.y_std

        pt_t = phi.T @ t
        # Eigendecompose Phi^T Phi once; do all posterior algebra in its eigenbasis so we
        # never invert a (possibly singular) matrix directly.
        eig, vecs = np.linalg.eigh(phi.T @ phi)
        eig = np.clip(eig, 0.0, None)
        proj = vecs.T @ pt_t  # pt_t in the eigenbasis

        alpha, beta = 1.0, 1.0
        for _ in range(self.n_iter):
            denom = alpha + beta * eig  # > 0 since alpha floored above 0
            coeff = beta * proj / denom  # posterior mean in eigenbasis coords
            gamma = float(np.sum(beta * eig / denom))

            mm = float(coeff @ coeff)
            alpha_new = gamma / mm if mm > 1e-12 else alpha
            resid = t - phi @ (vecs @ coeff)
            rss = float(resid @ resid)
            n_eff = n - gamma
            beta_new = n_eff / rss if n_eff > 0.0 and rss > 1e-12 else beta

            alpha_new = float(np.clip(alpha_new, self._FLOOR, self._CEIL))
            beta_new = float(np.clip(beta_new, self._FLOOR, self._CEIL))
            if abs(alpha_new - alpha) < self.tol and abs(beta_new - beta) < self.tol:
                alpha, beta = alpha_new, beta_new
                break
            alpha, beta = alpha_new, beta_new

        self.alpha, self.beta = alpha, beta
        d_inv = 1.0 / (alpha + beta * eig)
        self.s_n = (vecs * d_inv) @ vecs.T
        self.m_n = vecs @ (beta * proj * d_inv)
        return self

    def predict(self, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.m_n is not None and self.s_n is not None
        phi = np.asarray(phi, dtype=float)
        mean = phi @ self.m_n
        var = 1.0 / self.beta + np.einsum("ij,jk,ik->i", phi, self.s_n, phi)
        mean = mean * self.y_std + self.y_mean
        var = var * (self.y_std ** 2)
        return mean, var


class BayesianLinearSurrogate:
    """Multi-output Bayesian linear regression surrogate over [1, x, is_acting].

    Fits one evidence-maximized BLR per output on the normalized design vector
    (feature vector) concatenated with the activeness vector. Requires
    ``supports['x_hierarchy']`` so sb_arch_opt forwards ``is_acting`` to both
    training and prediction.
    """

    supports = {
        "training_derivatives": False,
        "derivatives": False,
        "output_derivatives": False,
        "adjoint_api": False,
        "variances": True,
        "variance_derivatives": False,
        "x_hierarchy": True,
    }

    def __init__(self, output_names: Sequence[str] | None = None):
        self.supports = type(self).supports.copy()
        self.output_names = list(output_names or [])
        self.options = {
            "print_global": False,
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }
        self.xt: np.ndarray | None = None
        self.yt: np.ndarray | None = None
        self.at: np.ndarray | None = None
        self.models: list[_BLR] = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def _features(x: np.ndarray, is_acting) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if is_acting is None:
            a = np.ones_like(x)
        else:
            a = np.asarray(is_acting, dtype=float)
            if a.ndim == 1:
                a = a[None, :]
        bias = np.ones((x.shape[0], 1))
        return np.concatenate([bias, x, a], axis=1)

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None) -> None:
        X = np.asarray(xt, dtype=float)
        y = np.asarray(yt, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"xt and yt length mismatch: {X.shape[0]} != {y.shape[0]}")
        self.xt = X
        self.yt = y
        self.at = None if is_acting is None else np.asarray(is_acting, dtype=float)

    def train(self) -> None:
        if self.xt is None or self.yt is None:
            raise RuntimeError("Training values must be set before train().")
        phi = self._features(self.xt, self.at)
        self.models = [_BLR().fit(phi, self.yt[:, iy]) for iy in range(self.yt.shape[1])]

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        phi = self._features(x, is_acting)
        return np.column_stack([m.predict(phi)[0] for m in self.models])

    def predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        phi = self._features(x, is_acting)
        return np.column_stack([m.predict(phi)[1] for m in self.models])

    def set_log_path(self, log_path) -> None:
        pass

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "n_outputs": len(self.models) if self.models else len(self.output_names),
            "output_names": self.output_names,
            "feature_map": "[1, x, is_acting]",
        }