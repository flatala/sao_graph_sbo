from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch import Tensor
from torch.optim import Optimizer
from tqdm.auto import tqdm

from graph_bo.gnn.vae.losses import vae_loss_full
from graph_bo.gnn.vae.model import GIN_VAE


LOSS_KEYS = (
    "loss",
    "loss_adj",
    "loss_features",
    "loss_features_binary",
    "loss_features_continuous",
    "loss_kl",
)

METRIC_KEYS = (
    "adj_precision",
    "adj_recall",
    "adj_f1",
    "adj_edge_rate",
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_dense_vae_batch(batch):
    return (
        torch.stack([d.x for d in batch]),
        torch.stack([d.A_encoder for d in batch]),
        torch.stack([d.mask for d in batch]),
        torch.stack([d.A_target for d in batch]),
        torch.stack([d.X_target for d in batch]),
    )


def adjacency_metrics(
    A_logits: Tensor,
    A_target: Tensor,
    mask: Tensor,
    adj_cutoff: float,
) -> dict[str, Tensor]:
    A_pred = (A_logits.sigmoid() >= adj_cutoff).float()
    M = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1))
    M = M * torch.triu(torch.ones_like(M), diagonal=1)
    n_pairs = M.sum(dim=(1, 2)).clamp(min=1)

    tp = ((A_pred == 1.0) & (A_target == 1.0)).float().mul(M).sum(dim=(1, 2))
    fp = ((A_pred == 1.0) & (A_target == 0.0)).float().mul(M).sum(dim=(1, 2))
    fn = ((A_pred == 0.0) & (A_target == 1.0)).float().mul(M).sum(dim=(1, 2))
    precision = tp / (tp + fp).clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    return {
        "adj_accuracy": ((A_pred == A_target).float() * M).sum(dim=(1, 2)) / n_pairs,
        "adj_precision": precision,
        "adj_recall": recall,
        "adj_f1": 2.0 * precision * recall / (precision + recall).clamp(min=1e-8),
        "adj_edge_rate": A_target.mul(M).sum(dim=(1, 2)) / n_pairs,
    }


def feature_metrics(X_logits: Tensor, X_target: Tensor, mask: Tensor) -> dict[str, Tensor]:
    M = mask.unsqueeze(-1)
    err = (X_logits.sigmoid() - X_target) * M
    n_valid = M.sum(dim=(1, 2)).mul(X_target.shape[-1]).clamp(min=1)
    return {
        "feature_mae": err.abs().sum(dim=(1, 2)) / n_valid,
        "feature_rmse": torch.sqrt(err.square().sum(dim=(1, 2)) / n_valid),
    }


def run_vae_epoch(
    model: GIN_VAE,
    loader,
    optimizer: Optimizer | None,
    device: torch.device | str,
    beta: float,
    adj_cutoff: float,
    binary_feature_indices: Sequence[int] | Tensor,
    continuous_feature_indices: Sequence[int] | Tensor,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    totals = {key: 0.0 for key in LOSS_KEYS}
    metric_totals = {key: 0.0 for key in METRIC_KEYS}

    with torch.set_grad_enabled(train):
        for x, A, mask, A_target, X_target in loader:
            x = x.to(device)
            A = A.to(device)
            mask = mask.to(device)
            A_target = A_target.to(device)
            X_target = X_target.to(device)

            outputs = model(x, A)
            losses = vae_loss_full(
                outputs,
                A_target,
                X_target,
                beta=beta,
                binary_feature_indices=binary_feature_indices,
                continuous_feature_indices=continuous_feature_indices,
            )
            if train:
                optimizer.zero_grad()
                losses["loss"].backward()
                optimizer.step()

            batch_size = len(x)
            for key in LOSS_KEYS:
                totals[key] += losses[key].item() * batch_size

            adj_metrics = adjacency_metrics(outputs["A_logits"], A_target, mask, adj_cutoff)
            for key in METRIC_KEYS:
                metric_totals[key] += adj_metrics[key].sum().item()

    n_items = len(loader.dataset)
    epoch_metrics = {key: value / n_items for key, value in totals.items()}
    epoch_metrics.update({key: value / n_items for key, value in metric_totals.items()})
    epoch_metrics["loss_recon"] = epoch_metrics["loss_adj"] + epoch_metrics["loss_features"]
    return epoch_metrics


def train_vae(
    model: GIN_VAE,
    train_loader,
    val_loader,
    optimizer: Optimizer,
    device: torch.device | str,
    n_epochs: int,
    beta: float,
    adj_cutoff: float,
    binary_feature_indices: Sequence[int] | Tensor,
    continuous_feature_indices: Sequence[int] | Tensor,
    checkpoint_dir: Path | str | None = None,
    config: dict[str, Any] | None = None,
    selection_window: int = 5,
    show_progress: bool = True,
    snapshot_epochs: Sequence[int] | None = None,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_path is not None:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "config.json").write_text(json.dumps(config or {}, indent=2))

    # Snapshot the model state at exactly these epoch counts (e.g. {0, 10, 50, 200}),
    # so one 200-epoch pass yields the epoch-N encoders for free - the epoch-N snapshot
    # is bit-identical to a standalone N-epoch run (same seed/data/optimizer). Each
    # snapshot carries its own cumulative pretrain_seconds in the checkpoint config.
    snapshot_set = {int(e) for e in snapshot_epochs} if snapshot_epochs is not None else set()

    def _save_snapshot(epoch: int, metrics_row: dict[str, Any]) -> None:
        if checkpoint_path is None or epoch not in snapshot_set:
            return
        snap_config = {**(config or {}), "pretrain_seconds": time.perf_counter() - t_start,
                       "snapshot_epoch": int(epoch)}
        save_vae_checkpoint(checkpoint_path / f"epoch_{epoch:03d}.pt", model, optimizer,
                            int(epoch), metrics_row, snap_config)

    t_start = time.perf_counter()
    # Epoch 0 = the random-init encoder (the untrained-encoder control), zero train time.
    _save_snapshot(0, {})
    history_rows: list[dict[str, float]] = []
    best_score: tuple[float, float, int] | None = None
    best_epoch = 0

    epoch_bar = tqdm(
        range(1, n_epochs + 1),
        desc="VAE reconstruction",
        unit="epoch",
        disable=not show_progress,
    )
    for epoch in epoch_bar:
        train_metrics = run_vae_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta,
            adj_cutoff,
            binary_feature_indices,
            continuous_feature_indices,
        )
        val_metrics = run_vae_epoch(
            model,
            val_loader,
            None,
            device,
            beta,
            adj_cutoff,
            binary_feature_indices,
            continuous_feature_indices,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)

        recent_rows = history_rows[-selection_window:]
        smoothed_f1 = sum(item["val_adj_f1"] for item in recent_rows) / len(recent_rows)
        row["val_adj_f1_smooth"] = smoothed_f1
        score = (smoothed_f1, -val_metrics["loss_recon"], epoch)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            if checkpoint_path is not None:
                save_vae_checkpoint(
                    checkpoint_path / "best.pt",
                    model,
                    optimizer,
                    epoch,
                    row,
                    config or {},
                )

        _save_snapshot(epoch, row)

        if checkpoint_path is not None:
            history = pd.DataFrame(history_rows)
            history.to_csv(checkpoint_path / "history.csv", index=False)

        if show_progress:
            epoch_bar.set_postfix({
                "train_recon": f"{train_metrics['loss_recon']:.4f}",
                "val_recon": f"{val_metrics['loss_recon']:.4f}",
                "val_precision": f"{val_metrics['adj_precision']:.3f}",
                "val_recall": f"{val_metrics['adj_recall']:.3f}",
                "val_f1": f"{val_metrics['adj_f1']:.3f}",
                "val_f1_s": f"{smoothed_f1:.3f}",
                "best": best_epoch,
            })

    history = pd.DataFrame(history_rows)
    if checkpoint_path is not None:
        final_config = {**(config or {}), "pretrain_seconds": time.perf_counter() - t_start}
        (checkpoint_path / "config.json").write_text(json.dumps(final_config, indent=2))
        save_vae_checkpoint(
            checkpoint_path / "last.pt",
            model,
            optimizer,
            int(history.iloc[-1]["epoch"]),
            history.iloc[-1].to_dict(),
            final_config,
        )
        history.to_csv(checkpoint_path / "history.csv", index=False)

    return history


def save_vae_checkpoint(
    path: Path | str,
    model: GIN_VAE,
    optimizer: Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    torch.save(checkpoint, path)


def load_vae_checkpoint(
    path: Path | str,
    device: torch.device | str | None = None,
) -> tuple[GIN_VAE, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model_config = checkpoint["config"]["model"]
    model = GIN_VAE(
        input_dim=model_config["input_dim"],
        hidden_dim=model_config["hidden_dim"],
        latent_dim=model_config["latent_dim"],
        n_gin_layers=model_config["n_gin_layers"],
        norm=model_config.get("norm") or _infer_gin_norm(checkpoint["model_state_dict"]),
        adj_decoder=model_config.get("adj_decoder", "inner_product"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if device is not None:
        model = model.to(device)
    return model, checkpoint


def _infer_gin_norm(state_dict: dict[str, Tensor]) -> str:
    for key in state_dict:
        if key.endswith(".running_mean") or key.endswith(".running_var"):
            return "batch"
    return "layer"
