from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from adsg_core.optimization.graph_processor import GraphProcessor
from pymoo.util.normalization import Normalization

from graph_bo.gnn.data.features import ADSGFeatureExtractor, ADSGNodeFeatureExtractor
from graph_bo.gnn.data.tensor_builder import ADSGTensorBuilder
from graph_bo.gnn.surrogate.heads import BLR
from graph_bo.gnn.vae.training import get_device
from graph_bo.surrogates.resolver import ArchInstanceResolver

__all__ = ["DGBOSurrogate"]


class DenseGraphConvLayer(nn.Module):
    """DGBO-style dense graph convolution with symmetric adjacency normalization."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a_norm, self.linear(h))


class DGBOFeatureNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        graph_hidden_dim: int,
        graph_layers: int,
        pool_hidden_dim: int,
        fc_layers: int,
        feature_dim: int,
        dropout: float,
    ):
        super().__init__()
        if graph_layers < 1:
            raise ValueError(f"graph_layers must be >= 1, got {graph_layers}")
        if fc_layers < 1:
            raise ValueError(f"fc_layers must be >= 1, got {fc_layers}")

        dims = [input_dim] + [graph_hidden_dim] * graph_layers
        self.graph_layers = nn.ModuleList(
            DenseGraphConvLayer(dims[i], dims[i + 1])
            for i in range(graph_layers)
        )
        self.graph_activation = nn.Tanh()
        self.dropout = nn.Dropout(float(dropout))
        self.pool_linear = nn.Linear(graph_hidden_dim, pool_hidden_dim, bias=False)
        fc_dims = [pool_hidden_dim] + [feature_dim] * fc_layers
        self.fc_layers = nn.ModuleList(
            nn.Linear(fc_dims[i], fc_dims[i + 1], bias=True)
            for i in range(fc_layers)
        )
        self.output = nn.Linear(feature_dim, 1)

    def features(self, x: torch.Tensor, a_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.graph_layers:
            h = self.graph_activation(layer(self.dropout(h), a_norm))

        pool_logits = self.pool_linear(self.dropout(h))
        pooled = torch.softmax(pool_logits, dim=-1) * mask.unsqueeze(-1)
        h_graph = pooled.sum(dim=1)
        for layer in self.fc_layers:
            h_graph = torch.tanh(layer(self.dropout(h_graph)))
        return h_graph

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(x, a_norm, mask))


@dataclass
class _GraphBatch:
    x: torch.Tensor
    a_norm: torch.Tensor
    mask: torch.Tensor

    def to(self, device: torch.device) -> "_GraphBatch":
        return _GraphBatch(
            x=self.x.to(device),
            a_norm=self.a_norm.to(device),
            mask=self.mask.to(device),
        )


class _DGBOSingleOutputModel:
    def __init__(
        self,
        *,
        input_dim: int,
        graph_hidden_dim: int,
        graph_layers: int,
        pool_hidden_dim: int,
        fc_layers: int,
        feature_dim: int,
        dropout: float,
        n_epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        blr_hypers: str,
        blr_normalize_x: bool,
        blr_normalize_y: bool,
        include_self_loops: bool,
        seed: int | None,
        device: torch.device,
    ):
        self.input_dim = int(input_dim)
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.pool_hidden_dim = int(pool_hidden_dim)
        self.fc_layers = int(fc_layers)
        self.feature_dim = int(feature_dim)
        self.dropout = float(dropout)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.blr_hypers = blr_hypers
        self.blr_normalize_x = bool(blr_normalize_x)
        self.blr_normalize_y = bool(blr_normalize_y)
        self.include_self_loops = bool(include_self_loops)
        self.seed = seed
        self.device = device

        self.net: DGBOFeatureNet | None = None
        self.blr: BLR | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.training_history_: list[dict[str, float]] = []

    def fit(self, batch: _GraphBatch, y: np.ndarray) -> "_DGBOSingleOutputModel":
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != batch.x.shape[0]:
            raise ValueError(f"X and y length mismatch: {batch.x.shape[0]} != {y.shape[0]}")

        self._set_seed()
        self.net = DGBOFeatureNet(
            input_dim=self.input_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            graph_layers=self.graph_layers,
            pool_hidden_dim=self.pool_hidden_dim,
            fc_layers=self.fc_layers,
            feature_dim=self.feature_dim,
            dropout=self.dropout,
        ).to(self.device)

        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.y_mean = float(y_t.mean().item())
        y_std = float(y_t.std(unbiased=False).item())
        self.y_std = y_std if y_std > 1e-12 else 1.0
        y_scaled = (y_t - self.y_mean) / self.y_std

        graph_batch = batch.to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        generator = torch.Generator(device="cpu")
        if self.seed is not None:
            generator.manual_seed(int(self.seed))

        n = y.shape[0]
        batch_size = min(max(1, self.batch_size), n)
        self.training_history_ = []
        for epoch in range(1, self.n_epochs + 1):
            self.net.train()
            perm = torch.randperm(n, generator=generator, device="cpu")
            losses = []
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size].to(self.device)
                pred = self.net(
                    graph_batch.x[idx],
                    graph_batch.a_norm[idx],
                    graph_batch.mask[idx],
                )
                loss = (pred - y_scaled[idx]).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            mse_scaled = float(np.mean(losses))
            self.training_history_.append({
                "epoch": float(epoch),
                "mse": mse_scaled * self.y_std ** 2,
                "rmse": float(np.sqrt(mse_scaled)) * self.y_std,
                "mse_scaled": mse_scaled,
                "rmse_scaled": float(np.sqrt(mse_scaled)),
            })

        z = self._features(batch)
        self.blr = BLR(
            normalize_x=self.blr_normalize_x,
            normalize_y=self.blr_normalize_y,
            include_bias=True,
            hypers=self.blr_hypers,
        ).fit(z, y)
        return self

    def predict(self, batch: _GraphBatch) -> tuple[np.ndarray, np.ndarray]:
        if self.blr is None:
            raise RuntimeError("DGBO model is not fitted.")
        z = self._features(batch)
        return self.blr.predict(z)

    def _features(self, batch: _GraphBatch) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("DGBO feature network is not fitted.")
        self.net.eval()
        graph_batch = batch.to(self.device)
        with torch.no_grad():
            z = self.net.features(graph_batch.x, graph_batch.a_norm, graph_batch.mask)
        return z.detach().cpu().numpy().astype(np.float64)

    def _set_seed(self) -> None:
        if self.seed is None:
            return
        random.seed(int(self.seed))
        np.random.seed(int(self.seed))
        torch.manual_seed(int(self.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.seed))

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_layers": self.graph_layers,
            "pool_hidden_dim": self.pool_hidden_dim,
            "fc_layers": self.fc_layers,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "include_self_loops": self.include_self_loops,
            "blr": self.blr.get_config() if self.blr is not None else None,
        }


class DGBOSurrogate:
    """Supervised DGBO-style graph-convolution surrogate with a BLR final layer."""

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
        graph_processor: GraphProcessor,
        normalization: Normalization,
        output_names: Sequence[str] | None = None,
        *,
        graph_hidden_dim: int = 48,
        graph_layers: int = 5,
        pool_hidden_dim: int = 50,
        fc_layers: int = 5,
        feature_dim: int = 45,
        dropout: float = 0.0,
        n_epochs: int = 200,
        batch_size: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        blr_hypers: str = "optimize",
        blr_normalize_x: bool = True,
        blr_normalize_y: bool = True,
        include_self_loops: bool = True,
        feature_extractor_cls: type[ADSGFeatureExtractor] = ADSGNodeFeatureExtractor,
        device: torch.device | str | None = None,
        seed: int | None = None,
    ):
        self.supports = type(self).supports.copy()
        self.graph_processor = graph_processor
        self.normalization = normalization
        self.output_names = list(output_names or [])
        self.graph_hidden_dim = int(graph_hidden_dim)
        self.graph_layers = int(graph_layers)
        self.pool_hidden_dim = int(pool_hidden_dim)
        self.fc_layers = int(fc_layers)
        self.feature_dim = int(feature_dim)
        self.dropout = float(dropout)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.blr_hypers = blr_hypers
        self.blr_normalize_x = bool(blr_normalize_x)
        self.blr_normalize_y = bool(blr_normalize_y)
        self.include_self_loops = bool(include_self_loops)
        self.feature_extractor_cls = feature_extractor_cls
        self.device = torch.device(device) if device is not None else get_device()
        self.seed = seed

        self.options = {
            "print_global": False,
            "print_training": False,
            "print_prediction": False,
            "print_problem": False,
            "print_solver": False,
        }
        self.tensor_builder = ADSGTensorBuilder(graph_processor, feature_extractor_cls=feature_extractor_cls)
        self.resolver = ArchInstanceResolver(graph_processor, normalization)
        self.xt: np.ndarray | None = None
        self.yt: np.ndarray | None = None
        self.models: list[_DGBOSingleOutputModel] = []
        self._graph_cache: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._history_path: Path | None = None
        self._fit_index = 0
        self._last_train_fit_metrics: dict[str, float] = {}

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
        batch = self._build_batch(self.xt)
        self.models = [
            self._new_model(output_index=iy).fit(batch, self.yt[:, iy])
            for iy in range(self.yt.shape[1])
        ]
        self._fit_index += 1
        self._last_train_fit_metrics = self._train_fit_metrics(self.yt, self.predict_values(self.xt))
        self._write_history()

    def predict_values(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        means, _ = self._predict(x)
        return means

    def predict_variances(self, x: np.ndarray, is_acting=None) -> np.ndarray:
        _, variances = self._predict(x)
        return variances

    def _predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Surrogate must be trained before prediction.")
        batch = self._build_batch(x)
        predictions = [model.predict(batch) for model in self.models]
        means = np.column_stack([mean[:, 0] for mean, _ in predictions])
        variances = np.column_stack([var[:, 0] for _, var in predictions])
        return means, variances

    def _new_model(self, output_index: int) -> _DGBOSingleOutputModel:
        seed = None if self.seed is None else int(self.seed) + output_index
        return _DGBOSingleOutputModel(
            input_dim=self.tensor_builder.feature_dim,
            graph_hidden_dim=self.graph_hidden_dim,
            graph_layers=self.graph_layers,
            pool_hidden_dim=self.pool_hidden_dim,
            fc_layers=self.fc_layers,
            feature_dim=self.feature_dim,
            dropout=self.dropout,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            weight_decay=self.weight_decay,
            blr_hypers=self.blr_hypers,
            blr_normalize_x=self.blr_normalize_x,
            blr_normalize_y=self.blr_normalize_y,
            include_self_loops=self.include_self_loops,
            seed=seed,
            device=self.device,
        )

    def _build_batch(self, x_norm: np.ndarray | Sequence[Sequence[float]]) -> _GraphBatch:
        X = np.asarray(x_norm, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"x_norm must be 1D or 2D, got shape {X.shape}")

        xs, adj, masks = [], [], []
        for row in X:
            key = self.resolver.correct_normalised_dv(row)[0]
            cached = self._graph_cache.get(key)
            if cached is None:
                cached = self._build_graph_tensors(key)
                self._graph_cache[key] = cached
            x_t, a_t, mask_t = cached
            xs.append(x_t)
            adj.append(a_t)
            masks.append(mask_t)
        return _GraphBatch(
            x=torch.stack(xs),
            a_norm=torch.stack(adj),
            mask=torch.stack(masks),
        )

    def _build_graph_tensors(self, corrected_dv: tuple) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adsg = self.resolver.get_adsg_from_corrected_dv(corrected_dv)
        _, is_active = self.resolver.correct_raw_dv(corrected_dv)
        data = self.tensor_builder.from_graph(adsg, corrected_dv, is_active)
        a = data.A_encoder.float()
        if self.include_self_loops:
            a = a + torch.eye(a.shape[0], dtype=a.dtype)
        degree = a.sum(dim=1).clamp(min=1.0)
        d_inv_sqrt = degree.pow(-0.5)
        a_norm = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
        return data.x.float(), a_norm, data.mask.float()

    def get_metric_names(self) -> tuple[str, ...]:
        return (
            "train_fit.mse",
            "train_fit.rmse",
            "train_fit.mae",
            "train_fit.n_points",
            "train_fit.n_valid",
        )

    def get_metric_values(self, algorithm=None) -> dict[str, float]:
        return {
            f"train_fit.{key}": value
            for key, value in self._last_train_fit_metrics.items()
        }

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

    def set_log_path(self, log_path) -> None:
        if log_path is None:
            self._history_path = None
            return
        log_path = Path(log_path)
        self._history_path = log_path.parent / "dgbo_surrogate_history.csv"

    def _write_history(self) -> None:
        if self._history_path is None:
            return
        output_names = self.output_names or [f"y{iy}" for iy in range(len(self.models))]
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
            for output_name, model in zip(output_names, self.models):
                for row in model.training_history_:
                    writer.writerow({
                        "fit_index": self._fit_index,
                        "output": output_name,
                        "epoch": int(row["epoch"]),
                        "mse": row["mse"],
                        "rmse": row["rmse"],
                        "mse_scaled": row["mse_scaled"],
                        "rmse_scaled": row["rmse_scaled"],
                    })

    def get_config(self) -> dict:
        return {
            "class": type(self).__name__,
            "n_outputs": len(self.models) if self.models else len(self.output_names),
            "output_names": self.output_names,
            "graph_hidden_dim": self.graph_hidden_dim,
            "graph_layers": self.graph_layers,
            "pool_hidden_dim": self.pool_hidden_dim,
            "fc_layers": self.fc_layers,
            "feature_dim": self.feature_dim,
            "dropout": self.dropout,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "blr_hypers": self.blr_hypers,
            "blr_normalize_x": self.blr_normalize_x,
            "blr_normalize_y": self.blr_normalize_y,
            "include_self_loops": self.include_self_loops,
            "feature_extractor_cls": self.feature_extractor_cls.__name__,
            "device": str(self.device),
            "seed": self.seed,
            "tensor": {
                "n_nodes": self.tensor_builder.N,
                "feature_dim": self.tensor_builder.feature_dim,
            },
            "model": self.models[0].get_config() if self.models else None,
        }
