from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.optim as optim
from adsg_core.optimization.graph_processor import GraphProcessor
from pymoo.util.normalization import Normalization
from torch.utils.data import DataLoader

from graph_bo.gnn.data.datasets import load_csv_graphs
from graph_bo.gnn.data.features import ADSGFeatureExtractor, ADSGNodeFeatureExtractor
from graph_bo.gnn.data.tensor_builder import ADSGTensorBuilder
from graph_bo.gnn.vae.losses import vae_loss_full
from graph_bo.gnn.vae.model import GIN_VAE
from graph_bo.gnn.vae.training import (
    adjacency_metrics,
    collate_dense_vae_batch,
    feature_metrics,
    get_device,
    load_vae_checkpoint,
    train_vae,
)
from graph_bo.surrogates.resolver import ArchInstanceResolver


@dataclass
class Arch2VecEmbedder:
    # arch2vec-style embedder: the VAE reconstructs the full padded graph (inactive
    # rows/cols included, every entry weighted equally) and the graph embedding is
    # the sum of all node codes. The adjacency decoder defaults to the learned map
    # on top of the inner product, which makes the padded region reconstructable.
    latent_dim: int
    hidden_dim: int
    n_gin_layers: int
    norm: str
    n_epochs: int
    batch_size: int
    lr: float
    beta: float
    adj_cutoff: float
    selection_window: int
    val_fraction: float
    model_path: Path | str | None = None
    dataset_path: Path | str | None = None
    n_samples: int | None = None
    feature_extractor_cls: type[ADSGFeatureExtractor] = ADSGNodeFeatureExtractor
    embedding_batch_size: int = 256
    seed: int | None = None
    show_progress: bool = False
    adj_decoder: str = "mlp"

    graph_processor: GraphProcessor | None = field(default=None, init=False, repr=False)
    normalization: Normalization | None = field(default=None, init=False, repr=False)
    tensor_builder: ADSGTensorBuilder | None = field(default=None, init=False, repr=False)
    model: GIN_VAE | None = field(default=None, init=False, repr=False)
    device: torch.device | None = field(default=None, init=False, repr=False)
    resolver: ArchInstanceResolver | None = field(default=None, init=False, repr=False)
    pretrain_seconds: float | None = field(default=None, init=False, repr=False)
    sample_seconds: float | None = field(default=None, init=False, repr=False)
    _embedding_cache: dict[tuple, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def embedding_dim(self) -> int:
        self._check_prepared()
        return self.model.vae_heads.fc_mu.out_features

    def prepare(
        self,
        graph_processor: GraphProcessor,
        normalization: Normalization,
        device: torch.device | str | None = None,
    ) -> "Arch2VecEmbedder":
        if self.model_path is None:
            raise ValueError("model_path is required: it is the checkpoint to load or where pretraining is saved.")

        self.graph_processor = graph_processor
        self.normalization = normalization
        self.device = torch.device(device) if device is not None else get_device()
        self.tensor_builder = ADSGTensorBuilder(graph_processor, feature_extractor_cls=self.feature_extractor_cls)
        self.resolver = ArchInstanceResolver(graph_processor, normalization)
        self._embedding_cache.clear()

        model_path = Path(self.model_path)
        if model_path.exists():
            self.model, checkpoint = load_vae_checkpoint(model_path, device=self.device)
        else:
            if self.dataset_path is None and self.n_samples is None:
                raise ValueError("Cannot pretrain: provide a dataset_path or n_samples.")
            best_path = self._pretrain(model_path.parent)
            self.model, checkpoint = load_vae_checkpoint(best_path, device=self.device)
        # Per-checkpoint timing first (epoch snapshots carry their own cumulative
        # pretrain_seconds + the seed's sample_seconds); fall back to the dir config.json.
        ckpt_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        self.pretrain_seconds = ckpt_config.get("pretrain_seconds", _read_config_seconds(model_path.parent, "pretrain_seconds"))
        self.sample_seconds = ckpt_config.get("sample_seconds", _read_config_seconds(model_path.parent, "sample_seconds"))
        self._freeze_model()
        return self

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def transform(self, x_norm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        self._check_prepared()
        X = np.asarray(x_norm, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"x_norm must be 1D or 2D, got shape {X.shape}")

        keys: list[tuple] = []
        missing: dict[tuple, tuple[tuple, tuple[bool, ...]]] = {}
        for row in X:
            dv_corr, is_active = self.resolver.correct_normalised_dv(row)
            keys.append(dv_corr)
            if dv_corr not in self._embedding_cache:
                missing[dv_corr] = (dv_corr, is_active)

        if missing:
            self._embed_missing(list(missing.values()))

        return np.vstack([self._embedding_cache[key] for key in keys])

    def reconstruction_metrics(
        self,
        x_norm: np.ndarray | Sequence[Sequence[float]],
        adj_cutoff: float = 0.9,
    ) -> dict[str, float]:
        self._check_prepared()
        X = np.asarray(x_norm, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"x_norm must be 1D or 2D, got shape {X.shape}")
        if X.shape[0] == 0:
            return _empty_reconstruction_metrics()

        corrected_items = [self.resolver.correct_normalised_dv(row) for row in X]
        data_list = [
            self.tensor_builder.from_graph(
                self.resolver.get_adsg_from_corrected_dv(dv_corr),
                dv_corr,
                is_active,
            )
            for dv_corr, is_active in corrected_items
        ]

        metric_values = {
            "adj_precision": [],
            "adj_recall": [],
            "adj_f1": [],
            "adj_edge_rate": [],
            "feature_mae": [],
            "feature_rmse": [],
        }

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(data_list), self.embedding_batch_size):
                batch = data_list[start:start + self.embedding_batch_size]
                x, A, mask, A_target, X_target = collate_dense_vae_batch(batch)
                x = x.to(self.device)
                A = A.to(self.device)
                mask = mask.to(self.device)
                A_target = A_target.to(self.device)
                X_target = X_target.to(self.device)

                outputs = self.model(x, A)
                adj_metrics = adjacency_metrics(outputs["A_logits"], A_target, mask, adj_cutoff)
                feat_metrics = feature_metrics(outputs["X_logits"], X_target, mask)

                for key in ("adj_precision", "adj_recall", "adj_f1", "adj_edge_rate"):
                    metric_values[key].extend(adj_metrics[key].cpu().tolist())
                for key in ("feature_mae", "feature_rmse"):
                    metric_values[key].extend(feat_metrics[key].cpu().tolist())

        return {
            key: float(np.mean(values)) if values else np.nan
            for key, values in metric_values.items()
        } | {"n_graphs": float(len(data_list))}

    def get_config(self) -> dict:
        values = {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "n_gin_layers": self.n_gin_layers,
            "norm": self.norm,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "beta": self.beta,
            "adj_cutoff": self.adj_cutoff,
            "selection_window": self.selection_window,
            "val_fraction": self.val_fraction,
            "model_path": str(self.model_path) if self.model_path is not None else None,
            "dataset_path": str(self.dataset_path) if self.dataset_path is not None else None,
            "n_samples": self.n_samples,
            "feature_extractor_cls": self.feature_extractor_cls.__name__,
            "embedding_batch_size": self.embedding_batch_size,
            "seed": self.seed,
            "show_progress": self.show_progress,
            "adj_decoder": self.adj_decoder,
        }
        if self.model is not None:
            values["device"] = str(self.device)
            values["embedding_dim"] = self.embedding_dim
            values["tensor"] = {
                "n_nodes": self.tensor_builder.N,
                "feature_dim": self.tensor_builder.feature_dim,
            }
        return {"class": type(self).__name__, **values}

    def _pretrain(self, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        _set_seed(self.seed)

        if self.dataset_path is not None:
            data_list = load_csv_graphs(Path(self.dataset_path), self.tensor_builder)
            source = {"kind": "csv", "path": str(self.dataset_path)}
            self.sample_seconds = 0.0
        else:
            t_sample = time.perf_counter()
            data_list = self.tensor_builder.sample(int(self.n_samples))
            self.sample_seconds = time.perf_counter() - t_sample
            source = {"kind": "sample", "sample_count": int(self.n_samples)}

        train_graphs, val_graphs = _split_train_val(data_list, self.val_fraction, self.seed)
        train_loader = _loader(train_graphs, self.batch_size, shuffle=True, seed=self.seed)
        val_loader = _loader(val_graphs, self.batch_size, shuffle=False, seed=self.seed)

        model = GIN_VAE(
            input_dim=self.tensor_builder.feature_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            n_gin_layers=self.n_gin_layers,
            norm=self.norm,
            adj_decoder=self.adj_decoder,
        ).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        run_config = {
            "source": source,
            "sample_seconds": self.sample_seconds,
            "embedder": self.get_config(),
            "tensor": {
                "n_nodes": self.tensor_builder.N,
                "feature_dim": self.tensor_builder.feature_dim,
                "n_binary_features": len(self.tensor_builder.binary_feature_indices),
                "n_continuous_features": len(self.tensor_builder.continuous_feature_indices),
            },
            "model": {
                "input_dim": self.tensor_builder.feature_dim,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "n_gin_layers": self.n_gin_layers,
                "norm": self.norm,
                "adj_decoder": self.adj_decoder,
            },
        }
        (checkpoint_dir / "pretrain_config.json").write_text(json.dumps(run_config, indent=2))

        train_vae(
            model,
            train_loader,
            val_loader,
            optimizer,
            self.device,
            n_epochs=self.n_epochs,
            beta=self.beta,
            adj_cutoff=self.adj_cutoff,
            binary_feature_indices=self.tensor_builder.binary_feature_indices,
            continuous_feature_indices=self.tensor_builder.continuous_feature_indices,
            checkpoint_dir=checkpoint_dir,
            config=run_config,
            selection_window=self.selection_window,
            show_progress=self.show_progress,
        )
        return checkpoint_dir / "best.pt"

    def _embed_missing(self, corrected_items: list[tuple[tuple, tuple[bool, ...]]]) -> None:
        data_list = []
        data_keys = []
        for dv_corr, is_active in corrected_items:
            adsg = self.resolver.get_adsg_from_corrected_dv(dv_corr)
            data_list.append(self.tensor_builder.from_graph(adsg, dv_corr, is_active))
            data_keys.append(dv_corr)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(data_list), self.embedding_batch_size):
                batch = data_list[start:start + self.embedding_batch_size]
                batch_keys = data_keys[start:start + self.embedding_batch_size]
                x, A, mask, _, _ = collate_dense_vae_batch(batch)
                x = x.to(self.device)
                A = A.to(self.device)
                mask = mask.to(self.device)
                mu, _ = self.model.encode(x, A)
                z = self.model.graph_embedding(mu).detach().cpu().numpy()
                for key, embedding in zip(batch_keys, z):
                    self._embedding_cache[key] = embedding.astype(np.float64, copy=True)

    def _freeze_model(self) -> None:
        in_features = self.model.encoder.layers[0].mlp[0].in_features
        if in_features != self.tensor_builder.feature_dim:
            raise ValueError(
                "Loaded GNN input dimension does not match tensor builder feature dimension: "
                f"{in_features} != {self.tensor_builder.feature_dim}"
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def _check_prepared(self) -> None:
        if self.model is None or self.tensor_builder is None or self.resolver is None or self.device is None:
            raise RuntimeError("Arch2VecEmbedder must be prepared before use.")


@dataclass
class RawSumEmbedder:
    feature_extractor_cls: type[ADSGFeatureExtractor] = ADSGNodeFeatureExtractor
    pooling: str = "sum"

    graph_processor: GraphProcessor | None = field(default=None, init=False, repr=False)
    normalization: Normalization | None = field(default=None, init=False, repr=False)
    tensor_builder: ADSGTensorBuilder | None = field(default=None, init=False, repr=False)
    resolver: ArchInstanceResolver | None = field(default=None, init=False, repr=False)
    _embedding_cache: dict[tuple, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    @property
    def embedding_dim(self) -> int:
        self._check_prepared()
        return self.tensor_builder.feature_dim

    def prepare(
        self,
        graph_processor: GraphProcessor,
        normalization: Normalization,
        device: torch.device | str | None = None,
    ) -> "RawSumEmbedder":
        if self.pooling not in {"sum", "mean"}:
            raise ValueError(f"Unsupported pooling {self.pooling!r}; expected 'sum' or 'mean'.")
        self.graph_processor = graph_processor
        self.normalization = normalization
        self.tensor_builder = ADSGTensorBuilder(graph_processor, feature_extractor_cls=self.feature_extractor_cls)
        self.resolver = ArchInstanceResolver(graph_processor, normalization)
        self._embedding_cache.clear()
        return self

    def clear_cache(self) -> None:
        self._embedding_cache.clear()

    def transform(self, x_norm: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        self._check_prepared()
        X = np.asarray(x_norm, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"x_norm must be 1D or 2D, got shape {X.shape}")

        keys: list[tuple] = []
        missing: dict[tuple, tuple[tuple, tuple[bool, ...]]] = {}
        for row in X:
            dv_corr, is_active = self.resolver.correct_normalised_dv(row)
            keys.append(dv_corr)
            if dv_corr not in self._embedding_cache:
                missing[dv_corr] = (dv_corr, is_active)

        for dv_corr, is_active in missing.values():
            adsg = self.resolver.get_adsg_from_corrected_dv(dv_corr)
            data = self.tensor_builder.from_graph(adsg, dv_corr, is_active)
            x_nodes = data.X_target.numpy().astype(np.float64)
            mask = data.mask.numpy().astype(np.float64)
            pooled = (x_nodes * mask[:, None]).sum(axis=0)
            if self.pooling == "mean":
                pooled = pooled / max(float(mask.sum()), 1.0)
            self._embedding_cache[dv_corr] = pooled

        return np.vstack([self._embedding_cache[key] for key in keys])

    def reconstruction_metrics(
        self,
        x_norm: np.ndarray | Sequence[Sequence[float]],
        adj_cutoff: float = 0.9,
    ) -> dict[str, float]:
        # No decoder: nothing to reconstruct.
        return _empty_reconstruction_metrics()

    def get_config(self) -> dict:
        values = {
            "class": type(self).__name__,
            "pooling": self.pooling,
            "feature_extractor_cls": self.feature_extractor_cls.__name__,
        }
        if self.tensor_builder is not None:
            values["feature_dim"] = self.tensor_builder.feature_dim
            values["embedding_dim"] = self.tensor_builder.feature_dim
        return values

    def _check_prepared(self) -> None:
        if self.tensor_builder is None or self.resolver is None:
            raise RuntimeError("RawSumEmbedder must be prepared before use.")


def _empty_reconstruction_metrics() -> dict[str, float]:
    return {
        "adj_precision": np.nan,
        "adj_recall": np.nan,
        "adj_f1": np.nan,
        "adj_edge_rate": np.nan,
        "feature_mae": np.nan,
        "feature_rmse": np.nan,
        "n_graphs": 0.0,
    }


def _read_config_seconds(checkpoint_dir: Path, key: str) -> float | None:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        return None
    seconds = json.loads(config_path.read_text()).get(key)
    return float(seconds) if seconds is not None else None


def _split_train_val(data_list, val_fraction: float, seed: int | None):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    indices = list(range(len(data_list)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    if n_val >= len(indices):
        n_val = len(indices) - 1
    val_idx = set(indices[:n_val])
    train_graphs = [data for i, data in enumerate(data_list) if i not in val_idx]
    val_graphs = [data for i, data in enumerate(data_list) if i in val_idx]
    if not train_graphs or not val_graphs:
        raise ValueError("Need at least one training and one validation graph for pretraining.")
    return train_graphs, val_graphs


def _loader(data_list, batch_size: int, shuffle: bool, seed: int | None):
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return DataLoader(
        data_list,
        batch_size=min(batch_size, len(data_list)),
        shuffle=shuffle,
        collate_fn=collate_dense_vae_batch,
        generator=generator if shuffle else None,
    )


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
