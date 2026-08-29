from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from experimenter.surrogate_fit.grid import (
    SurrogateBenchmarkBuilder,
    SurrogateFitConfig,
    config_result_dir,
    experiment_result_root,
    model_result_dir,
)


def run_surrogate_fit(
    builder: SurrogateBenchmarkBuilder,
    config: SurrogateFitConfig,
    results_root: Path,
    overwrite: bool = False,
) -> None:
    result_dir = config_result_dir(results_root, config)
    run_config_path = result_dir / "run_config.json"
    if run_config_path.exists() and not overwrite:
        with run_config_path.open("r", encoding="utf-8") as fh:
            run_config = json.load(fh)
        if run_config["status"] == "completed":
            print(f"Skipping complete fit: {result_dir}", flush=True)
            return

    result_dir.mkdir(parents=True, exist_ok=True)
    _write_run_config(run_config_path, config, status="started")

    dataset_path = builder.dataset_path(config)
    print(f"Loading dataset: {dataset_path}", flush=True)
    dataset = pd.read_csv(dataset_path).sort_values("sample_id").reset_index(drop=True)

    print(f"Building problem: {config.problem}", flush=True)
    problem = builder.build_problem(config)

    x_columns = tuple(col for col in dataset.columns if col.startswith("x"))
    is_active_columns = tuple(col for col in dataset.columns if col.startswith("is_active"))
    target_columns = tuple(config.problem_config["target_columns"]) if config.problem_config["target_columns"] else tuple(default_target_columns(dataset))
    if len(x_columns) != problem.n_var:
        raise ValueError(f"Dataset has {len(x_columns)} x columns, but problem has {problem.n_var} variables.")

    valid_mask = valid_benchmark_rows(dataset, target_columns)
    dataset = dataset.loc[valid_mask].reset_index(drop=True).copy()
    add_discrete_arch_id(dataset, problem, x_columns, is_active_columns)
    print(f"Using {len(dataset)} valid rows for targets={target_columns}", flush=True)

    train_pool = dataset[dataset["sample_id"] % config.n_folds == config.fold]
    test_df = dataset[dataset["sample_id"] % config.n_folds != config.fold]
    if len(train_pool) < config.budget:
        raise ValueError(
            f"Training pool for fold {config.fold} has {len(train_pool)} rows, "
            f"cannot satisfy budget {config.budget}."
        )
    train_df = select_train_df(train_pool, config)

    print(f"Building surrogate: {config.surrogate}", flush=True)
    surrogate, normalization = builder.build_surrogate(problem, config)

    result = fit_predict_score(
        config=config,
        surrogate=surrogate,
        normalization=normalization,
        train_df=train_df,
        test_df=test_df,
        x_columns=x_columns,
        is_active_columns=is_active_columns,
        target_columns=target_columns,
    )

    pd.DataFrame(result["metrics"]).to_csv(result_dir / "metrics.csv", index=False)
    pd.DataFrame(result["predictions"]).to_csv(result_dir / "predictions.csv", index=False)
    train_selection_frame(config, train_df, target_columns).to_csv(result_dir / "train_selection.csv", index=False)
    pd.DataFrame([result["timings"]]).to_csv(result_dir / "timings.csv", index=False)
    pd.DataFrame(
        result["weights"],
        columns=["problem", "model", "cv_folds", "fold", "budget", "weight_name", "weight_value"],
    ).to_csv(result_dir / "kernel_weights.csv", index=False)
    with (result_dir / "model_config.json").open("w", encoding="utf-8") as fh:
        json.dump(result["model_config"], fh, indent=2)

    _write_run_config(run_config_path, config, status="completed")
    print(f"DONE {result_dir}", flush=True)


def aggregate_surrogate_results(configs: Sequence[SurrogateFitConfig], results_root: Path) -> None:
    by_model: dict[tuple[str, int, str], list[SurrogateFitConfig]] = {}
    for config in configs:
        key = (config.problem, config.n_folds, config.surrogate)
        by_model.setdefault(key, []).append(config)

    experiment_predictions = []
    experiment_train_selection = []

    for model_configs in by_model.values():
        metrics_frames = []
        timings_frames = []
        weights_frames = []
        prediction_frames = []
        train_selection_frames = []
        model_config = None
        for config in sorted(model_configs, key=lambda item: (item.fold, item.budget)):
            result_dir = config_result_dir(results_root, config)
            run_config_path = result_dir / "run_config.json"
            if not run_config_path.exists():
                continue
            with run_config_path.open("r", encoding="utf-8") as fh:
                run_config = json.load(fh)
            if run_config["status"] != "completed":
                continue
            metrics_frames.append(pd.read_csv(result_dir / "metrics.csv"))
            timings_frames.append(pd.read_csv(result_dir / "timings.csv"))
            predictions_path = result_dir / "predictions.csv"
            if predictions_path.exists():
                prediction_frames.append(pd.read_csv(predictions_path))
            train_selection_path = result_dir / "train_selection.csv"
            if train_selection_path.exists():
                train_selection_frames.append(pd.read_csv(train_selection_path))
            kernel_weights_path = result_dir / "kernel_weights.csv"
            if kernel_weights_path.stat().st_size > 0:
                weights_frames.append(pd.read_csv(kernel_weights_path))
            with (result_dir / "model_config.json").open("r", encoding="utf-8") as fh:
                model_config = json.load(fh)

        if len(metrics_frames) == 0:
            continue

        model_dir = model_result_dir(results_root, model_configs[0])
        model_metrics = pd.concat(metrics_frames, ignore_index=True)
        model_timings = pd.concat(timings_frames, ignore_index=True)
        model_metrics.to_csv(model_dir / "metrics.csv", index=False)
        model_timings.to_csv(model_dir / "timings.csv", index=False)
        if len(prediction_frames) > 0:
            model_predictions = pd.concat(prediction_frames, ignore_index=True)
            model_predictions.to_csv(model_dir / "predictions.csv", index=False)
            experiment_predictions.append(model_predictions)
        if len(train_selection_frames) > 0:
            model_train_selection = pd.concat(train_selection_frames, ignore_index=True)
            model_train_selection.to_csv(model_dir / "train_selection.csv", index=False)
            experiment_train_selection.append(model_train_selection)
        if len(weights_frames) == 0:
            pd.DataFrame(
                columns=["problem", "model", "cv_folds", "fold", "budget", "weight_name", "weight_value"],
            ).to_csv(model_dir / "kernel_weights.csv", index=False)
        else:
            pd.concat(weights_frames, ignore_index=True).to_csv(model_dir / "kernel_weights.csv", index=False)
        with (model_dir / "model_config.json").open("w", encoding="utf-8") as fh:
            json.dump(model_config, fh, indent=2)
        print(f"Aggregated {model_dir}", flush=True)

    if len(experiment_predictions) > 0:
        experiment_dir = experiment_result_root(results_root, configs[0])
        pd.concat(experiment_predictions, ignore_index=True).to_csv(experiment_dir / "predictions.csv", index=False)
    if len(experiment_train_selection) > 0:
        experiment_dir = experiment_result_root(results_root, configs[0])
        (
            pd.concat(experiment_train_selection, ignore_index=True)
            .drop_duplicates(["problem", "cv_folds", "fold", "budget", "sample_id"])
            .to_csv(experiment_dir / "train_selection.csv", index=False)
        )


def fit_predict_score(
    config: SurrogateFitConfig,
    surrogate,
    normalization,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_columns: Sequence[str],
    is_active_columns: Sequence[str],
    target_columns: Sequence[str],
) -> dict[str, Any]:
    context = {
        "problem": config.problem,
        "model": config.surrogate,
        "cv_folds": config.n_folds,
        "fold": config.fold,
        "budget": config.budget,
    }
    context_label = f"{config.problem}/{config.surrogate} fold={config.fold} budget={config.budget}"

    x_train = train_df.loc[:, x_columns].to_numpy(dtype=float)
    x_test = test_df.loc[:, x_columns].to_numpy(dtype=float)
    y_train = train_df.loc[:, target_columns].to_numpy(dtype=float)
    y_test = test_df.loc[:, target_columns].to_numpy(dtype=float)

    x_train_norm = normalization.forward(x_train)
    x_test_norm = normalization.forward(x_test)

    train_kwargs = {}
    predict_kwargs = {}
    if surrogate.supports.get("x_hierarchy", False):
        train_kwargs["is_acting"] = train_df.loc[:, is_active_columns].to_numpy(dtype=bool)
        predict_kwargs["is_acting"] = test_df.loc[:, is_active_columns].to_numpy(dtype=bool)

    print(f"Training {context_label}", flush=True)
    t_fit0 = time.perf_counter()
    surrogate.set_training_values(x_train_norm, y_train, **train_kwargs)
    surrogate.train()
    fit_sec = time.perf_counter() - t_fit0
    print(f"Trained {context_label} in {fit_sec:.1f}s", flush=True)

    print(f"Predicting train {context_label}", flush=True)
    t_pred_train0 = time.perf_counter()
    y_pred_train = predict_values(surrogate, x_train_norm, train_kwargs)
    predict_train_sec = time.perf_counter() - t_pred_train0

    print(f"Predicting test {context_label}", flush=True)
    t_pred_test0 = time.perf_counter()
    y_pred_test = predict_values(surrogate, x_test_norm, predict_kwargs)
    predict_test_sec = time.perf_counter() - t_pred_test0
    print(f"Predicted {context_label} in {predict_train_sec + predict_test_sec:.1f}s", flush=True)

    metrics = []
    metrics.extend({**context, **row} for row in metric_rows("train", target_columns, y_train, y_pred_train))
    metrics.extend({**context, **row} for row in metric_rows("test", target_columns, y_test, y_pred_test))
    predictions = []
    predictions.extend(prediction_rows(context, "train", train_df, target_columns, y_pred_train))
    predictions.extend(prediction_rows(context, "test", test_df, target_columns, y_pred_test))

    weights = []
    get_kernel_weight_values = getattr(surrogate, "get_kernel_weight_values", None)
    if get_kernel_weight_values is not None:
        weights.extend(
            {**context, "weight_name": name, "weight_value": value}
            for name, value in get_kernel_weight_values().items()
        )

    return {
        "metrics": metrics,
        "predictions": predictions,
        "weights": weights,
        "timings": {
            **context,
            "fit_sec": fit_sec,
            "predict_train_sec": predict_train_sec,
            "predict_test_sec": predict_test_sec,
            "predict_test_sec_per_point": predict_test_sec / len(test_df),
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        "model_config": {
            "experiment_alias": config.experiment_alias,
            "problem": config.problem,
            "surrogate": config.surrogate,
            "surrogate_kind": config.surrogate_kind,
            "params": config.params,
        },
    }


def default_target_columns(dataset: pd.DataFrame) -> list[str]:
    return [
        col
        for col in dataset.columns
        if (col.startswith("f") or col.startswith("g") or col.startswith("h"))
        and dataset[col].dtype.kind in {"f", "i", "u"}
    ]


def valid_benchmark_rows(dataset: pd.DataFrame, target_columns: Sequence[str]) -> pd.Series:
    if "eval_failed" in dataset.columns:
        valid = ~dataset["eval_failed"].astype(bool)
    else:
        valid = pd.Series(True, index=dataset.index)
    target_values = dataset.loc[:, target_columns].to_numpy(dtype=float)
    valid &= np.all(np.isfinite(target_values), axis=1)
    return valid


def add_discrete_arch_id(
    dataset: pd.DataFrame,
    problem,
    x_columns: Sequence[str],
    is_active_columns: Sequence[str],
) -> None:
    discrete_mask = np.asarray(problem.is_discrete_mask, dtype=bool)
    discrete_x_columns = [x_columns[i] for i, is_discrete in enumerate(discrete_mask) if is_discrete]
    discrete_is_active_columns = [is_active_columns[i] for i, is_discrete in enumerate(discrete_mask) if is_discrete]
    discrete_key_columns = [*discrete_x_columns, *discrete_is_active_columns]
    dataset["discrete_arch_id"] = pd.factorize(
        pd.MultiIndex.from_frame(dataset.loc[:, discrete_key_columns]),
        sort=False,
    )[0]


def select_train_df(train_pool: pd.DataFrame, config: SurrogateFitConfig) -> pd.DataFrame:
    train_selection = str(config.problem_config.get("train_selection", "head"))
    if train_selection == "head":
        return train_pool.head(config.budget).copy()
    if train_selection == "unique_discrete_coverage_v1":
        unique_arch = train_pool.drop_duplicates(subset=["discrete_arch_id"], keep="first")
        if len(unique_arch) >= config.budget:
            positions = np.linspace(0, len(unique_arch) - 1, config.budget, dtype=int)
            return unique_arch.iloc[positions].copy()

        remaining = train_pool.loc[~train_pool.index.isin(unique_arch.index)]
        n_missing = config.budget - len(unique_arch)
        positions = np.linspace(0, len(remaining) - 1, min(n_missing, len(remaining)), dtype=int)
        selected = pd.concat([unique_arch, remaining.iloc[positions]], axis=0)
        if len(selected) < config.budget:
            raise ValueError(
                f"Training pool for fold {config.fold} has only {len(selected)} selectable rows "
                f"for budget {config.budget}."
            )
        return selected.head(config.budget).copy()
    raise ValueError(f"Unknown train_selection={train_selection!r}")


def train_selection_frame(
    config: SurrogateFitConfig,
    train_df: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    frame = train_df.loc[:, ["sample_id", "discrete_arch_id", *target_columns]].copy()
    frame.insert(0, "budget", config.budget)
    frame.insert(0, "fold", config.fold)
    frame.insert(0, "cv_folds", config.n_folds)
    frame.insert(0, "problem", config.problem)
    return frame


def predict_values(surrogate, x_norm: np.ndarray, kwargs: dict[str, Any]) -> np.ndarray:
    y_pred = surrogate.predict_values(x_norm, **kwargs)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]
    return y_pred


def prediction_rows(
    context: dict[str, Any],
    split: str,
    source_df: pd.DataFrame,
    target_columns: Sequence[str],
    y_pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for i_row, (_, source_row) in enumerate(source_df.iterrows()):
        row = {
            **context,
            "split": split,
            "sample_id": int(source_row["sample_id"]),
            "discrete_arch_id": int(source_row["discrete_arch_id"]),
        }
        for i_target, target in enumerate(target_columns):
            row[target] = float(source_row[target])
            row[f"{target}_pred"] = float(y_pred[i_row, i_target])
        rows.append(row)
    return rows


def metric_rows(split: str, target_columns: Sequence[str], y_true: np.ndarray, y_pred: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for i_target, target in enumerate(target_columns):
        yt = y_true[:, i_target]
        yp = y_pred[:, i_target]
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]
        err = yp - yt
        rmse = float(np.sqrt(np.mean(err**2)))
        mae = float(np.mean(np.abs(err)))
        scale = float(np.std(yt))
        ss_res = float(np.sum(err**2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        rows.append(
            {
                "split": split,
                "target": target,
                "rmse": rmse,
                "nrmse": rmse / scale if scale > 0.0 else np.nan,
                "mae": mae,
                "r2": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan,
                "n_valid": int(mask.sum()),
            }
        )
    return rows


def _write_run_config(run_config_path: Path, config: SurrogateFitConfig, status: str) -> None:
    with run_config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": status,
                "experiment_alias": config.experiment_alias,
                "problem": config.problem,
                "surrogate": config.surrogate,
                "surrogate_kind": config.surrogate_kind,
                "n_samples": config.n_samples,
                "n_folds": config.n_folds,
                "fold": config.fold,
                "budget": config.budget,
                "params": config.params,
                "problem_config": config.problem_config,
            },
            fh,
            indent=2,
        )
