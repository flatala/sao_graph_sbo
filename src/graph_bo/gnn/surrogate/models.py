from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from graph_bo.gnn.surrogate.embedders import Arch2VecEmbedder
from graph_bo.gnn.surrogate.heads import RegressionModel


class GNNMultiSurrogate:
    supports = {
        "training_derivatives": False,
        "derivatives": False,
        "output_derivatives": False,
        "adjoint_api": False,
        "variances": True,
        "variance_derivatives": False,
        "x_hierarchy": True,
    }

    def __init__(
        self,
        embedder: Arch2VecEmbedder,
        downstream: RegressionModel,
        output_names: Sequence[str] | None = None,
        reconstruction_adj_cutoff: float = 0.9,
    ):
        self.supports = type(self).supports.copy()
        self.embedder = embedder
        self.downstream = downstream
        self.output_names = list(output_names or [])
        self.reconstruction_adj_cutoff = float(reconstruction_adj_cutoff)
        self.options = {
            "print_global": False,
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }
        self.xt: np.ndarray | None = None
        self.yt: np.ndarray | None = None
        self.heads: list[RegressionModel] = []
        self._last_train_fit_metrics: dict[str, float] = {}
        self._last_recon_train_metrics: dict[str, float] = {}
        self._fit_index = 0
        self._history_path: Path | None = None

    @property
    def name(self) -> str:
        return type(self).__name__

    def set_training_values(self, xt: np.ndarray, yt: np.ndarray, name=None, is_acting=None) -> None:
        X = np.asarray(xt, dtype=float)
        y = np.asarray(yt, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"xt and yt length mismatch: {X.shape[0]} != {y.shape[0]}")
        if self.output_names and len(self.output_names) != y.shape[1]:
            raise ValueError(f"Expected {len(self.output_names)} outputs, got {y.shape[1]}")
        self.xt = X
        self.yt = y

    def train(self) -> None:
        if self.xt is None or self.yt is None:
            raise RuntimeError("Training values must be set before train().")
        Z = self.embedder.transform(self.xt)
        self.heads = [
            copy.deepcopy(self.downstream).fit(Z, self.yt[:, iy])
            for iy in range(self.yt.shape[1])
        ]
        self._fit_index += 1
        self._last_train_fit_metrics = self._train_fit_metrics(self.yt, self.predict_values(self.xt))
        self._last_recon_train_metrics = self.embedder.reconstruction_metrics(
            self.xt,
            adj_cutoff=self.reconstruction_adj_cutoff,
        )
        self._write_head_history()

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        means, _ = self._predict(x)
        return means

    def predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        _, variances = self._predict(x)
        return variances

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.heads:
            raise RuntimeError("Surrogate must be trained before prediction.")
        Z = self.embedder.transform(x)
        predictions = [head.predict(Z) for head in self.heads]
        means = np.column_stack([mean[:, 0] for mean, _ in predictions])
        variances = np.column_stack([var[:, 0] for _, var in predictions])
        return means, variances

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "n_outputs": len(self.heads) if self.heads else len(self.output_names),
            "output_names": self.output_names,
            "reconstruction_adj_cutoff": self.reconstruction_adj_cutoff,
            "embedder": self.embedder.get_config(),
            "downstream": self.downstream.get_config(),
            "head": self.heads[0].get_config() if self.heads else None,
        }

    def get_metric_names(self) -> tuple[str, ...]:
        return (
            "train_fit.mse",
            "train_fit.rmse",
            "train_fit.mae",
            "train_fit.n_points",
            "train_fit.n_valid",
            "recon_train.adj_precision",
            "recon_train.adj_recall",
            "recon_train.adj_f1",
            "recon_train.adj_edge_rate",
            "recon_train.feature_mae",
            "recon_train.feature_rmse",
            "recon_train.n_graphs",
            "recon_infill.adj_precision",
            "recon_infill.adj_recall",
            "recon_infill.adj_f1",
            "recon_infill.adj_edge_rate",
            "recon_infill.feature_mae",
            "recon_infill.feature_rmse",
            "recon_infill.n_graphs",
        )

    def get_metric_values(self, algorithm=None) -> dict[str, float]:
        values = {
            f"train_fit.{key}": value
            for key, value in self._last_train_fit_metrics.items()
        }
        values.update({
            f"recon_train.{key}": value
            for key, value in self._last_recon_train_metrics.items()
        })
        x_infill_norm = self._get_infill_x_norm(algorithm)
        if x_infill_norm is not None:
            recon = self.embedder.reconstruction_metrics(
                x_infill_norm,
                adj_cutoff=self.reconstruction_adj_cutoff,
            )
            values.update({f"recon_infill.{key}": value for key, value in recon.items()})
        return values

    def set_log_path(self, log_path) -> None:
        if log_path is None:
            self._history_path = None
            return
        log_path = Path(log_path)
        self._history_path = log_path.parent / "gnn_surrogate_head_history.csv"

    @staticmethod
    def _train_fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        err = y_pred - y_true
        finite = np.isfinite(err)
        if not np.any(finite):
            return {
                "mse": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
                "n_points": float(y_true.shape[0]),
                "n_valid": 0.0,
            }
        sq_err = err[finite] ** 2
        mse = float(np.mean(sq_err))
        return {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "mae": float(np.mean(np.abs(err[finite]))),
            "n_points": float(y_true.shape[0]),
            "n_valid": float(np.sum(finite)),
        }

    @staticmethod
    def _get_infill_x_norm(algorithm) -> np.ndarray | None:
        if algorithm is None:
            return None
        infill_obj = getattr(algorithm, "infill_obj", None)
        if infill_obj is None:
            return None
        normalization = getattr(infill_obj, "normalization", None)
        if normalization is None:
            return None

        opt_results = getattr(infill_obj, "opt_results", None)
        if not opt_results:
            return None
        opt = getattr(opt_results[-1], "opt", None)
        if opt is None:
            return None
        try:
            x_infill = opt.get("X")
        except Exception:
            return None
        if x_infill is None:
            return None

        x_infill = np.asarray(x_infill, dtype=float)
        if x_infill.ndim == 1:
            x_infill = x_infill[None, :]
        if x_infill.ndim != 2 or x_infill.shape[0] == 0:
            return None
        return normalization.forward(x_infill)

    def _write_head_history(self) -> None:
        if self._history_path is None:
            return
        output_names = self.output_names or [f"y{iy}" for iy in range(len(self.heads))]
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self._history_path.exists()
        with self._history_path.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "fit_index",
                    "output",
                    "epoch",
                    "mse",
                    "rmse",
                    "mse_scaled",
                    "rmse_scaled",
                ],
            )
            if write_header:
                writer.writeheader()
            for output_name, head in zip(output_names, self.heads):
                for row in head.training_history_:
                    writer.writerow({
                        "fit_index": self._fit_index,
                        "output": output_name,
                        "epoch": int(row["epoch"]),
                        "mse": row["mse"],
                        "rmse": row["rmse"],
                        "mse_scaled": row["mse_scaled"],
                        "rmse_scaled": row["rmse_scaled"],
                    })