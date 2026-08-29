from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy.optimize import minimize
import torch


class RegressionModel(Protocol):
    training_history_: list[dict[str, float]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> RegressionModel:
        ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def get_config(self) -> dict:
        ...


class BLR:
    def __init__(
        self,
        *,
        normalize_x: bool,
        normalize_y: bool,
        include_bias: bool,
        hypers: str = "optimize",
    ):
        if hypers not in {"fixed", "optimize", "marginalize"}:
            raise ValueError(f"Unsupported hypers {hypers!r}; expected 'fixed', 'optimize' or 'marginalize'.")
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.include_bias = bool(include_bias)
        self.hypers = hypers

        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.alpha_: float = 1.0
        self.beta_: float = 1.0
        self.log_evidence_: float = np.nan
        self.posterior_mean_: np.ndarray | None = None
        self.posterior_cov_: np.ndarray | None = None
        # When marginalizing: (weight, mean, cov, beta) mixture components over the
        # (log alpha, log beta) grid; predict() moment-matches the mixture.
        self.posteriors_: list[tuple[float, np.ndarray, np.ndarray, float]] | None = None
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BLR":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}")

        Xs = self._fit_transform_x(X)
        ys = self._fit_transform_y(y)
        Phi = self._basis_matrix(Xs)
        self._fit_blr(Phi, ys)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._check_fitted()
        Xs = self._transform_x(np.asarray(X, dtype=np.float64))
        Phi = self._basis_matrix(Xs)
        if self.posteriors_ is not None:
            # Moment-matched mixture over hyperparameter grid points (law of total
            # variance: E[var_k] + Var[mean_k]).
            mean_s = np.zeros(Phi.shape[0])
            second = np.zeros(Phi.shape[0])
            for weight, mean_k, cov_k, beta_k in self.posteriors_:
                mu_k = Phi @ mean_k
                var_k = (1.0 / beta_k) + np.einsum("ij,jk,ik->i", Phi, cov_k, Phi)
                mean_s += weight * mu_k
                second += weight * (np.maximum(var_k, 0.0) + mu_k ** 2)
            var_s = np.maximum(second - mean_s ** 2, 0.0)
        else:
            mean_s = Phi @ self.posterior_mean_
            var_s = (1.0 / self.beta_) + np.einsum("ij,jk,ik->i", Phi, self.posterior_cov_, Phi)
            var_s = np.maximum(var_s, 0.0)
        mean = mean_s * self.y_std_ + self.y_mean_
        var = var_s * (self.y_std_ ** 2)
        return mean.reshape(-1, 1), var.reshape(-1, 1)

    def _fit_transform_x(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize_x:
            self.x_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.x_std_ = np.ones(X.shape[1], dtype=np.float64)
            return X
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        return self._transform_x(X)

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("BLR is not fitted.")
        return (X - self.x_mean_) / self.x_std_

    def _fit_transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.normalize_y:
            self.y_mean_ = float(y.mean())
            self.y_std_ = float(y.std())
            if self.y_std_ < 1e-12:
                self.y_std_ = 1.0
            return (y - self.y_mean_) / self.y_std_
        self.y_mean_ = 0.0
        self.y_std_ = 1.0
        return y

    def _basis_matrix(self, X: np.ndarray) -> np.ndarray:
        basis = np.asarray(X, dtype=np.float64)
        if self.include_bias:
            return np.column_stack([np.ones((basis.shape[0], 1)), basis])
        return basis

    def _fit_blr(self, Phi: np.ndarray, y: np.ndarray) -> None:
        if self.hypers == "marginalize":
            self._fit_blr_marginalized(Phi, y)
            return
        if self.hypers == "optimize":
            result = minimize(
                lambda params: self._negative_log_evidence(Phi, y, params),
                x0=np.array([0.0, 0.0]),
                method="L-BFGS-B",
                bounds=[(-10.0, 10.0), (-10.0, 10.0)],
            )
            log_alpha, log_beta = result.x if result.success else np.array([0.0, 0.0])
            self.alpha_ = float(np.exp(log_alpha))
            self.beta_ = float(np.exp(log_beta))
        else:
            self.alpha_ = 1.0
            self.beta_ = 1.0

        self.posterior_mean_, self.posterior_cov_ = self._posterior(Phi, y, self.alpha_, self.beta_)
        self.log_evidence_ = -self._negative_log_evidence(Phi, y, np.log([self.alpha_, self.beta_]))

    def _fit_blr_marginalized(self, Phi: np.ndarray, y: np.ndarray, grid_size: int = 21) -> None:
        # Grid quadrature over (log alpha, log beta) with a flat prior on the log scale:
        # the deterministic counterpart of pybnn's do_mcmc=True. Weights are the
        # normalized evidence; predict() averages over the surviving grid points.
        grid = np.linspace(-10.0, 10.0, grid_size)
        params = [(la, lb) for la in grid for lb in grid]
        log_z = np.array([
            -self._negative_log_evidence(Phi, y, np.array(p)) for p in params
        ])
        log_z[~np.isfinite(log_z)] = -np.inf
        if not np.isfinite(log_z.max()):
            raise RuntimeError("Evidence is degenerate on the whole hyperparameter grid.")
        weights = np.exp(log_z - log_z.max())
        weights /= weights.sum()

        keep = weights > 1e-6
        kept = weights[keep] / weights[keep].sum()
        self.posteriors_ = []
        for (log_alpha, log_beta), weight in zip(
            [p for p, k in zip(params, keep) if k], kept
        ):
            alpha, beta = float(np.exp(log_alpha)), float(np.exp(log_beta))
            mean, cov = self._posterior(Phi, y, alpha, beta)
            self.posteriors_.append((float(weight), mean, cov, beta))

        # Report weighted (geometric) means and the MAP component for inspection;
        # predict() uses the full mixture.
        kept_params = np.array([p for p, k in zip(params, keep) if k])
        self.alpha_ = float(np.exp(np.sum(kept * kept_params[:, 0])))
        self.beta_ = float(np.exp(np.sum(kept * kept_params[:, 1])))
        best = int(np.argmax(kept))
        self.posterior_mean_, self.posterior_cov_ = self.posteriors_[best][1], self.posteriors_[best][2]
        log_z_finite = log_z[np.isfinite(log_z)]
        self.log_evidence_ = float(
            log_z_finite.max() + np.log(np.exp(log_z_finite - log_z_finite.max()).sum()) - np.log(len(params))
        )

    @staticmethod
    def _negative_log_evidence(Phi: np.ndarray, y: np.ndarray, params: np.ndarray) -> float:
        alpha = float(np.exp(params[0]))
        beta = float(np.exp(params[1]))
        try:
            m, _, A = BLR._posterior(Phi, y, alpha, beta, return_precision=True)
            residual = y - Phi @ m
            sign, logdet = np.linalg.slogdet(A)
            if sign <= 0:
                return np.inf
            n, d = Phi.shape
            energy = 0.5 * beta * float(residual @ residual) + 0.5 * alpha * float(m @ m)
            return energy + 0.5 * logdet - 0.5 * d * np.log(alpha) - 0.5 * n * np.log(beta)
        except np.linalg.LinAlgError:
            return np.inf

    @staticmethod
    def _posterior(
        Phi: np.ndarray,
        y: np.ndarray,
        alpha: float,
        beta: float,
        return_precision: bool = False,
    ):
        d = Phi.shape[1]
        A = alpha * np.eye(d) + beta * (Phi.T @ Phi)
        A.flat[:: d + 1] += 1e-8
        cov = np.linalg.inv(A)
        mean = beta * cov @ Phi.T @ y
        if return_precision:
            return mean, cov, A
        return mean, cov

    def _check_fitted(self) -> None:
        if self.posterior_mean_ is None or self.posterior_cov_ is None:
            raise RuntimeError("BLR is not fitted.")

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "normalize_x": self.normalize_x,
            "normalize_y": self.normalize_y,
            "include_bias": self.include_bias,
            "hypers": self.hypers,
            "alpha": self.alpha_,
            "beta": self.beta_,
        }


class RandomFeatureBLR:
    def __init__(
        self,
        *,
        hidden_units: int,
        normalize_x: bool,
        normalize_y: bool,
        include_bias: bool,
        hypers: str = "optimize",
        seed: int | None = None,
    ):
        self.hidden_units = int(hidden_units)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.include_bias = bool(include_bias)
        self.hypers = hypers
        self.seed = seed

        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.proj_weight_: np.ndarray | None = None
        self.proj_bias_: np.ndarray | None = None
        self.blr_: BLR | None = None
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomFeatureBLR":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")

        Xs = self._fit_transform_x(X)
        self._init_projection(Xs.shape[1])
        Phi = self._project(Xs)
        self.blr_ = BLR(
            normalize_x=False,
            normalize_y=self.normalize_y,
            include_bias=self.include_bias,
            hypers=self.hypers,
        ).fit(Phi, y)
        self.training_history_ = self.blr_.training_history_
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.blr_ is None:
            raise RuntimeError("RandomFeatureBLR is not fitted.")
        Phi = self._project(self._transform_x(np.asarray(X, dtype=np.float64)))
        return self.blr_.predict(Phi)

    def _fit_transform_x(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize_x:
            self.x_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.x_std_ = np.ones(X.shape[1], dtype=np.float64)
            return X
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        return self._transform_x(X)

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("RandomFeatureBLR is not fitted.")
        return (X - self.x_mean_) / self.x_std_

    def _init_projection(self, in_dim: int) -> None:
        # Match nn.Linear's default init scale: U(-1/sqrt(in_dim), 1/sqrt(in_dim)).
        rng = np.random.default_rng(self.seed)
        bound = 1.0 / np.sqrt(in_dim)
        self.proj_weight_ = rng.uniform(-bound, bound, size=(self.hidden_units, in_dim))
        self.proj_bias_ = rng.uniform(-bound, bound, size=self.hidden_units)

    def _project(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.proj_weight_.T + self.proj_bias_)

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "hidden_units": self.hidden_units,
            "normalize_x": self.normalize_x,
            "normalize_y": self.normalize_y,
            "include_bias": self.include_bias,
            "hypers": self.hypers,
            "seed": self.seed,
            "blr": self.blr_.get_config() if self.blr_ is not None else None,
        }


class DNGO:
    def __init__(
        self,
        *,
        hidden_units: int,
        n_epochs: int,
        batch_size: int,
        lr: float,
        normalize_x: bool,
        normalize_y: bool,
        include_bias: bool,
        hypers: str = "optimize",
        calibrate_folds: int = 0,
        calibrate_stat: str = "mean_sq",
        seed: int | None = None,
    ):
        if calibrate_stat not in {"mean_sq", "q95"}:
            raise ValueError(f"Unsupported calibrate_stat {calibrate_stat!r}; expected 'mean_sq' or 'q95'.")
        self.hidden_units = int(hidden_units)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.include_bias = bool(include_bias)
        self.hypers = hypers
        self.calibrate_folds = int(calibrate_folds)
        self.calibrate_stat = calibrate_stat
        self.seed = seed

        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.net_: torch.nn.Sequential | None = None
        self.blr_: BLR | None = None
        self.calibration_scale_: float = 1.0
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DNGO":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}")

        Xs = self._fit_transform_x(X)
        self._train_network(Xs, y)
        Phi = self._basis(Xs)
        self.blr_ = BLR(
            normalize_x=False,
            normalize_y=self.normalize_y,
            include_bias=self.include_bias,
            hypers=self.hypers,
        ).fit(Phi, y)
        self.calibration_scale_ = 1.0
        if self.calibrate_folds > 1 and X.shape[0] > self.calibrate_folds:
            self.calibration_scale_ = self._cv_variance_scale(X, y)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.blr_ is None:
            raise RuntimeError("DNGO is not fitted.")
        Phi = self._basis(self._transform_x(np.asarray(X, dtype=np.float64)))
        mean, var = self.blr_.predict(Phi)
        return mean, var * self.calibration_scale_

    def _cv_variance_scale(self, X: np.ndarray, y: np.ndarray) -> float:
        # K-fold variance recalibration. The evidence estimates beta-hat from train
        # residuals the net has interpolated to ~0, so sigma's scale is wrong while
        # its ordering is fine. Refit on K-1 folds, collect standardized residuals
        # on the held-out fold, and scale the predictive variance by s^2 (= 1 for a
        # calibrated model). calibrate_stat picks the estimator:
        #   "mean_sq": s^2 = mean(r^2) - the Gaussian MLE / NLPD-optimal scale, but a
        #     few huge misses (heavy tails) inflate the bars for every point;
        #   "q95": s = q95(|r|)/1.96 - pins 95% coverage exactly and ignores how far
        #     the worst 5% land, so typical bars stay tight under heavy tails.
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(X.shape[0]), self.calibrate_folds)
        residuals = []
        for fold in folds:
            if fold.size == 0:
                continue
            mask = np.ones(X.shape[0], dtype=bool)
            mask[fold] = False
            sub = DNGO(
                hidden_units=self.hidden_units,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                normalize_x=self.normalize_x,
                normalize_y=self.normalize_y,
                include_bias=self.include_bias,
                hypers=self.hypers,
                seed=self.seed,
            ).fit(X[mask], y[mask])
            mu, var = sub.predict(X[fold])
            residuals.append((y[fold] - mu.ravel()) / np.sqrt(np.maximum(var.ravel(), 1e-12)))
        r = np.abs(np.concatenate(residuals))
        if self.calibrate_stat == "q95":
            return float(np.quantile(r, 0.95) / 1.959964) ** 2
        return float(np.mean(r ** 2))

    def _fit_transform_x(self, X: np.ndarray) -> np.ndarray:
        if not self.normalize_x:
            self.x_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.x_std_ = np.ones(X.shape[1], dtype=np.float64)
            return X
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ < 1e-12] = 1.0
        return self._transform_x(X)

    def _transform_x(self, X: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise RuntimeError("DNGO is not fitted.")
        return (X - self.x_mean_) / self.x_std_

    def _train_network(self, Xs: np.ndarray, y: np.ndarray) -> None:
        ys = y
        y_std = 1.0
        if self.normalize_y:
            std = y.std()
            y_std = float(std) if std > 1e-12 else 1.0
            ys = (y - y.mean()) / y_std

        if self.seed is not None:
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                net = self._build_net(Xs.shape[1])
        else:
            net = self._build_net(Xs.shape[1])

        X_t = torch.as_tensor(Xs, dtype=torch.float32)
        y_t = torch.as_tensor(ys, dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        n = X_t.shape[0]
        batch = min(self.batch_size, n)

        self.training_history_ = []
        net.train()
        for epoch in range(1, self.n_epochs + 1):
            perm = torch.randperm(n, generator=generator)
            epoch_losses = []
            for start in range(0, n, batch):
                idx = perm[start:start + batch]
                loss = (net(X_t[idx]) - y_t[idx]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            mse_scaled = float(np.mean(epoch_losses))
            self.training_history_.append({
                "epoch": float(epoch),
                "mse": mse_scaled * y_std ** 2,
                "rmse": float(np.sqrt(mse_scaled)) * y_std,
                "mse_scaled": mse_scaled,
                "rmse_scaled": float(np.sqrt(mse_scaled)),
            })
        net.eval()
        self.net_ = net

    def _build_net(self, in_dim: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, self.hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(self.hidden_units, 1),
        )

    def _basis(self, Xs: np.ndarray) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("DNGO is not fitted.")
        with torch.no_grad():
            h = self.net_[1](self.net_[0](torch.as_tensor(Xs, dtype=torch.float32)))
        return h.numpy().astype(np.float64)

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "hidden_units": self.hidden_units,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "normalize_x": self.normalize_x,
            "normalize_y": self.normalize_y,
            "include_bias": self.include_bias,
            "hypers": self.hypers,
            "calibrate_folds": self.calibrate_folds,
            "calibrate_stat": self.calibrate_stat,
            "calibration_scale": self.calibration_scale_,
            "seed": self.seed,
            "blr": self.blr_.get_config() if self.blr_ is not None else None,
        }


class RandomNNEnsembleBLR:
    def __init__(
        self,
        *,
        n_members: int,
        hidden_units: int,
        normalize_x: bool,
        normalize_y: bool,
        include_bias: bool,
        weighting: str = "uniform",
        hypers: str = "optimize",
        seed: int | None = None,
    ):
        if weighting not in {"uniform", "evidence"}:
            raise ValueError(f"Unsupported weighting {weighting!r}; expected 'uniform' or 'evidence'.")
        self.n_members = int(n_members)
        self.hidden_units = int(hidden_units)
        self.normalize_x = bool(normalize_x)
        self.normalize_y = bool(normalize_y)
        self.include_bias = bool(include_bias)
        self.weighting = weighting
        self.hypers = hypers
        self.seed = seed

        self.members_: list[RandomFeatureBLR] = []
        self.weights_: np.ndarray | None = None
        self.training_history_: list[dict[str, float]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomNNEnsembleBLR":
        self.members_ = [
            RandomFeatureBLR(
                hidden_units=self.hidden_units,
                normalize_x=self.normalize_x,
                normalize_y=self.normalize_y,
                include_bias=self.include_bias,
                hypers=self.hypers,
                seed=None if self.seed is None else self.seed + k,
            ).fit(X, y)
            for k in range(self.n_members)
        ]
        if self.weighting == "evidence":
            log_ev = np.array([member.blr_.log_evidence_ for member in self.members_])
            w = np.exp(log_ev - log_ev.max())
            self.weights_ = w / w.sum()
        else:
            self.weights_ = np.full(self.n_members, 1.0 / self.n_members)
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.members_:
            raise RuntimeError("RandomNNEnsembleBLR is not fitted.")
        predictions = [member.predict(X) for member in self.members_]
        means = np.column_stack([mean[:, 0] for mean, _ in predictions])
        variances = np.column_stack([var[:, 0] for _, var in predictions])
        mean = means @ self.weights_
        var = (variances + means ** 2) @ self.weights_ - mean ** 2
        var = np.maximum(var, 0.0)
        return mean.reshape(-1, 1), var.reshape(-1, 1)

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "n_members": self.n_members,
            "hidden_units": self.hidden_units,
            "normalize_x": self.normalize_x,
            "normalize_y": self.normalize_y,
            "include_bias": self.include_bias,
            "weighting": self.weighting,
            "hypers": self.hypers,
            "seed": self.seed,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "member_log_evidence": [
                member.blr_.log_evidence_ for member in self.members_
            ] if self.members_ else None,
        }
