from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from experimenter.encoder_training.grid import (
    EncoderRunConfig,
    build_graph_processor,
    config_result_dir,
    feature_extractor_class,
    setup_result_dir,
)
from graph_bo.gnn import (
    ADSGTensorBuilder,
    collate_dense_vae_batch,
    evaluate_vae_reconstruction,
    get_device,
    load_csv_graphs,
    load_vae_checkpoint,
    split_graphs_train_val_test_by_sample_id,
    train_vae,
)
from graph_bo.gnn.vae import GIN_VAE


def run_encoder_fit(
    config: EncoderRunConfig,
    resources_root: Path,
    data_root: Path,
    results_root: Path,
    overwrite: bool = False,
) -> None:
    if str(config.params.get("source", "")) == "resample":
        _run_resample_snapshot_fit(config, resources_root, results_root, overwrite=overwrite)
        return

    result_dir = config_result_dir(results_root, config)
    run_config_path = result_dir / "run_config.json"
    if run_config_path.exists() and not overwrite:
        with run_config_path.open("r", encoding="utf-8") as fh:
            run_config = json.load(fh)
        if run_config["status"] == "completed":
            print(f"Skipping complete encoder fit: {result_dir}", flush=True)
            return

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_config_path, config, status="started")
    t0 = time.perf_counter()

    _set_seed(config.seed)
    torch_threads = int(config.params.get("torch_threads", 1))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    device = get_device()
    print(f"Device: {device}", flush=True)
    print(f"Building graph processor: {config.problem}", flush=True)
    graph_processor = build_graph_processor(config.problem, resources_root)
    feature_cls = feature_extractor_class(config.feature_extractor)
    tensor_builder = ADSGTensorBuilder(graph_processor, feature_extractor_cls=feature_cls)

    dataset_path = data_root / config.problem / f"samples_{config.n_samples}.csv"
    print(f"Loading graphs: {dataset_path}", flush=True)
    data_list = load_csv_graphs(dataset_path, tensor_builder)
    train_graphs, val_graphs, test_graphs = split_graphs_train_val_test_by_sample_id(
        data_list,
        n_folds=config.n_folds,
        val_fold=config.val_fold,
        test_fold=config.test_fold,
    )
    print(
        f"Split sizes: train={len(train_graphs)}, val={len(val_graphs)}, test={len(test_graphs)}",
        flush=True,
    )

    batch_size = int(config.params["batch_size"])
    train_loader = _loader(train_graphs, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_loader = _loader(val_graphs, batch_size=batch_size, shuffle=False, seed=config.seed)
    test_loader = _loader(test_graphs, batch_size=batch_size, shuffle=False, seed=config.seed)

    model = GIN_VAE(
        input_dim=tensor_builder.feature_dim,
        hidden_dim=int(config.params["hidden_dim"]),
        latent_dim=int(config.params["latent_dim"]),
        n_gin_layers=int(config.params["n_gin_layers"]),
        norm=str(config.params.get("norm", "layer")),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.params["lr"]))

    run_config = _model_config(config, tensor_builder, device, dataset_path)
    print(f"Training {config.problem}/{config.setup} seed={config.seed} test_fold={config.test_fold}", flush=True)
    history = train_vae(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        n_epochs=int(config.params["n_epochs"]),
        beta=float(config.params["beta"]),
        adj_cutoff=float(config.params["adj_cutoff"]),
        binary_feature_indices=tensor_builder.binary_feature_indices,
        continuous_feature_indices=tensor_builder.continuous_feature_indices,
        checkpoint_dir=result_dir,
        config=run_config,
        selection_window=int(config.params.get("selection_window", 5)),
        show_progress=False,
    )

    best_model, checkpoint = load_vae_checkpoint(result_dir / "best.pt", device=device)
    metrics = {
        "train": evaluate_vae_reconstruction(best_model, train_loader, device, float(config.params["adj_cutoff"]), False),
        "val": evaluate_vae_reconstruction(best_model, val_loader, device, float(config.params["adj_cutoff"]), False),
        "test": evaluate_vae_reconstruction(best_model, test_loader, device, float(config.params["adj_cutoff"]), False),
    }
    summary = {
        "problem": config.problem,
        "setup": config.setup,
        "seed": config.seed,
        "n_folds": config.n_folds,
        "val_fold": config.val_fold,
        "test_fold": config.test_fold,
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_test": len(test_graphs),
        "best_epoch": int(checkpoint["epoch"]),
        "elapsed_sec": time.perf_counter() - t0,
        "device": str(device),
        "metrics": metrics,
        "best_checkpoint_metrics": checkpoint["metrics"],
        "final_history_row": history.iloc[-1].to_dict(),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_run_config(run_config_path, config, status="completed")
    print(f"DONE {result_dir}", flush=True)


def aggregate_encoder_results(configs: Sequence[EncoderRunConfig], results_root: Path) -> None:
    by_setup: dict[tuple[str, int, str], list[EncoderRunConfig]] = {}
    for config in configs:
        key = (config.problem, config.n_folds, config.setup)
        by_setup.setdefault(key, []).append(config)

    for setup_configs in by_setup.values():
        rows = []
        for config in sorted(setup_configs, key=lambda item: (item.seed, item.test_fold)):
            result_dir = config_result_dir(results_root, config)
            run_config_path = result_dir / "run_config.json"
            metrics_path = result_dir / "metrics.json"
            if not run_config_path.exists() or not metrics_path.exists():
                continue
            with run_config_path.open("r", encoding="utf-8") as fh:
                run_config = json.load(fh)
            if run_config["status"] != "completed":
                continue
            with metrics_path.open("r", encoding="utf-8") as fh:
                metrics = json.load(fh)
            rows.append(_metrics_row(config, metrics))

        if len(rows) == 0:
            continue

        result_dir = setup_result_dir(results_root, setup_configs[0])
        result_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(result_dir / "metrics.csv", index=False)
        print(f"Aggregated {result_dir}", flush=True)


def _loader(data_list, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_dense_vae_batch,
        generator=generator if shuffle else None,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_config(
    config: EncoderRunConfig,
    tensor_builder: ADSGTensorBuilder,
    device: torch.device,
    dataset_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_alias": config.experiment_alias,
        "problem": config.problem,
        "setup": config.setup,
        "seed": config.seed,
        "n_samples": config.n_samples,
        "n_folds": config.n_folds,
        "val_fold": config.val_fold,
        "test_fold": config.test_fold,
        "dataset_path": str(dataset_path),
        "device": str(device),
        "feature_extractor": config.feature_extractor,
        "tensor": {
            "n_nodes": tensor_builder.N,
            "feature_dim": tensor_builder.feature_dim,
            "n_binary_features": len(tensor_builder.binary_feature_indices),
            "n_continuous_features": len(tensor_builder.continuous_feature_indices),
        },
        "model": {
            "input_dim": tensor_builder.feature_dim,
            "hidden_dim": int(config.params["hidden_dim"]),
            "latent_dim": int(config.params["latent_dim"]),
            "n_gin_layers": int(config.params["n_gin_layers"]),
        },
        "training": {
            "batch_size": int(config.params["batch_size"]),
            "n_epochs": int(config.params["n_epochs"]),
            "lr": float(config.params["lr"]),
            "beta": float(config.params["beta"]),
            "adj_cutoff": float(config.params["adj_cutoff"]),
            "selection_window": int(config.params.get("selection_window", 5)),
        },
    }


def _metrics_row(config: EncoderRunConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    if "n_test" not in metrics:
        # Resample snapshot fits (source="resample") write a reduced summary without
        # test folds; aggregate the fields they record.
        return {
            "problem": config.problem,
            "setup": config.setup,
            "seed": config.seed,
            "n_train": metrics["n_train"],
            "n_val": metrics["n_val"],
            "n_samples": metrics["n_samples"],
            "sample_seconds": metrics["sample_seconds"],
            "snapshot_epochs": metrics["snapshot_epochs"],
            "snapshots": metrics["snapshots"],
            "elapsed_sec": metrics["elapsed_sec"],
            "device": metrics["device"],
        }
    row = {
        "problem": config.problem,
        "setup": config.setup,
        "seed": config.seed,
        "n_folds": config.n_folds,
        "val_fold": config.val_fold,
        "test_fold": config.test_fold,
        "n_train": metrics["n_train"],
        "n_val": metrics["n_val"],
        "n_test": metrics["n_test"],
        "best_epoch": metrics["best_epoch"],
        "elapsed_sec": metrics["elapsed_sec"],
        "device": metrics["device"],
    }
    for split, split_metrics in metrics["metrics"].items():
        for key, value in split_metrics.items():
            row[f"{split}_{key}"] = value
    return row


def _write_run_config(run_config_path: Path, config: EncoderRunConfig, status: str) -> None:
    with run_config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": status,
                "experiment_alias": config.experiment_alias,
                "problem": config.problem,
                "setup": config.setup,
                "params": config.params,
                "problem_config": config.problem_config,
                "n_samples": config.n_samples,
                "n_folds": config.n_folds,
                "val_fold": config.val_fold,
                "test_fold": config.test_fold,
                "seed": config.seed,
            },
            fh,
            indent=2,
        )


def _run_resample_snapshot_fit(
    config: EncoderRunConfig,
    resources_root: Path,
    results_root: Path,
    overwrite: bool = False,
) -> None:
    """Phase A: resample the encoder's training graphs fresh per seed (timed), then
    train one encoder to the largest snapshot epoch, dumping epoch_{N}.pt checkpoints.
    One pass yields every epoch-N encoder (the epoch-N snapshot is bit-identical to a
    standalone N-epoch run)."""
    result_dir = config_result_dir(results_root, config)
    run_config_path = result_dir / "run_config.json"
    if run_config_path.exists() and not overwrite:
        with run_config_path.open("r", encoding="utf-8") as fh:
            if json.load(fh)["status"] == "completed":
                print(f"Skipping complete resample fit: {result_dir}", flush=True)
                return

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_config_path, config, status="started")
    t0 = time.perf_counter()

    _set_seed(config.seed)
    torch_threads = int(config.params.get("torch_threads", 1))
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)

    device = get_device()
    params = config.params
    print(f"Device: {device} | resample fit {config.problem}/{config.setup} seed={config.seed}", flush=True)
    graph_processor = build_graph_processor(config.problem, resources_root)
    feature_cls = feature_extractor_class(config.feature_extractor)
    tensor_builder = ADSGTensorBuilder(graph_processor, feature_extractor_cls=feature_cls)

    t_sample = time.perf_counter()
    data_list = tensor_builder.sample(int(config.n_samples))
    sample_seconds = time.perf_counter() - t_sample
    print(f"Resampled {len(data_list)} graphs in {sample_seconds:.1f}s", flush=True)

    val_fraction = float(params.get("val_fraction", 0.1))
    train_graphs, val_graphs = _split_sampled(data_list, val_fraction, config.seed)
    batch_size = int(params["batch_size"])
    train_loader = _loader(train_graphs, batch_size=batch_size, shuffle=True, seed=config.seed)
    val_loader = _loader(val_graphs, batch_size=batch_size, shuffle=False, seed=config.seed)

    norm = str(params.get("norm", "batch"))
    adj_decoder = str(params.get("adj_decoder", "mlp"))
    model = GIN_VAE(
        input_dim=tensor_builder.feature_dim,
        hidden_dim=int(params["hidden_dim"]),
        latent_dim=int(params["latent_dim"]),
        n_gin_layers=int(params["n_gin_layers"]),
        norm=norm,
        adj_decoder=adj_decoder,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))

    # model block must hold exactly GIN_VAE's kwargs so load_vae_checkpoint can rebuild it.
    run_config = {
        "experiment_alias": config.experiment_alias,
        "problem": config.problem,
        "setup": config.setup,
        "seed": config.seed,
        "n_samples": config.n_samples,
        "source": "resample",
        "sample_seconds": sample_seconds,
        "device": str(device),
        "feature_extractor": config.feature_extractor,
        "model": {
            "input_dim": tensor_builder.feature_dim,
            "hidden_dim": int(params["hidden_dim"]),
            "latent_dim": int(params["latent_dim"]),
            "n_gin_layers": int(params["n_gin_layers"]),
            "norm": norm,
            "adj_decoder": adj_decoder,
        },
        "training": {
            "batch_size": batch_size,
            "n_epochs": int(params["n_epochs"]),
            "lr": float(params["lr"]),
            "beta": float(params["beta"]),
            "adj_cutoff": float(params["adj_cutoff"]),
            "selection_window": int(params.get("selection_window", 5)),
            "val_fraction": val_fraction,
            "snapshot_epochs": [int(e) for e in params["snapshot_epochs"]],
        },
    }

    snapshot_epochs = [int(e) for e in params["snapshot_epochs"]]
    print(f"Training {config.setup} to {int(params['n_epochs'])} epochs, snapshots={snapshot_epochs}", flush=True)
    train_vae(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        n_epochs=int(params["n_epochs"]),
        beta=float(params["beta"]),
        adj_cutoff=float(params["adj_cutoff"]),
        binary_feature_indices=tensor_builder.binary_feature_indices,
        continuous_feature_indices=tensor_builder.continuous_feature_indices,
        checkpoint_dir=result_dir,
        config=run_config,
        selection_window=int(params.get("selection_window", 5)),
        show_progress=False,
        snapshot_epochs=snapshot_epochs,
    )

    saved = sorted(p.name for p in result_dir.glob("epoch_*.pt"))
    summary = {
        "problem": config.problem,
        "setup": config.setup,
        "seed": config.seed,
        "n_train": len(train_graphs),
        "n_val": len(val_graphs),
        "n_samples": int(config.n_samples),
        "sample_seconds": sample_seconds,
        "snapshot_epochs": snapshot_epochs,
        "snapshots": saved,
        "elapsed_sec": time.perf_counter() - t0,
        "device": str(device),
    }
    with (result_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    _write_run_config(run_config_path, config, status="completed")
    print(f"DONE {result_dir} | snapshots={saved}", flush=True)


def _split_sampled(data_list, val_fraction: float, seed: int):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    indices = list(range(len(data_list)))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(round(len(indices) * val_fraction)))
    n_val = min(n_val, len(indices) - 1)
    val_idx = set(indices[:n_val])
    train_graphs = [data for i, data in enumerate(data_list) if i not in val_idx]
    val_graphs = [data for i, data in enumerate(data_list) if i in val_idx]
    return train_graphs, val_graphs
