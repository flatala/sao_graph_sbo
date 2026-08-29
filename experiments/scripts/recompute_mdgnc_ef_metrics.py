from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_REFERENCE_DIR = REPO_ROOT / "resources" / "reference_fronts" / "mdgnc_edge_failures"
EMPIRICAL_REFERENCE_DIR = REPO_ROOT / "resources" / "reference_fronts" / "mdgnc_edge_failures_empirical"
ORIGINAL_RESULTS_NAME = "run_results.reference_mdgnc_edge_failures.csv"
ORIGINAL_AGGREGATE_NAME = "aggregate_results.reference_mdgnc_edge_failures.csv"

FINAL_METHOD_EXPERIMENTS = {
    "Random search": REPO_ROOT
    / "experiments/kernels/final_method_baselines/mdgnc_edge_failures/random_mdgnc_edge_failures_init_30_pop_10_250_evals",
    "GA": REPO_ROOT
    / "experiments/kernels/final_method_baselines/mdgnc_edge_failures/ga_mdgnc_edge_failures_init_30_pop_10_250_evals",
    "Hierarchical BO": REPO_ROOT
    / "experiments/kernels/final_method_surrogates/mdgnc_edge_failures/bo_no_kpls_mdgnc_edge_failures_init_30_infill_10_250_evals",
    "BO-KPLS": REPO_ROOT
    / "experiments/kernels/final_method_surrogates/mdgnc_edge_failures/bo_kpls_mdgnc_edge_failures_init_30_infill_10_250_evals",
    "BLR": REPO_ROOT
    / "experiments/kernels/final_method_surrogates/mdgnc_edge_failures/blr_activeness_mdgnc_edge_failures_init_30_infill_10_250_evals",
    "ESP": REPO_ROOT
    / "experiments/kernels/final_method_surrogates/mdgnc_edge_failures/adsg_esp_semantic_type_edge_weight_imputed_interaction_mdgnc_edge_failures_init_30_infill_10_250_evals",
    "Arch2Vec": REPO_ROOT
    / "experiments/gnn/final_method_arch2vec_bo/mdgnc_edge_failures/gnn_dngo_node_d5_b0p0025_z32_e050_mdgnc_edge_failures_init_30_pop_10_250_evals",
    "DGBO": REPO_ROOT
    / "experiments/gnn/final_method_dgbo/mdgnc_edge_failures/dgbo_mdgnc_edge_failures_init_30_pop_10_250_evals",
    "MG LD-WLOA": REPO_ROOT
    / "experiments/kernels/final_method_surrogates/mdgnc_edge_failures/adsg_mg_ld_wloa_adsg_semantic_type_d8_edge_weight_imputed_interaction_mdgnc_edge_failures_init_30_infill_10_250_evals",
}

HV_COLUMNS = [
    "delta_hv.delta_hv",
    "delta_hv.hv",
    "delta_hv.true_hv",
    "delta_hv.ratio",
    "delta_hv.regret",
    "delta_hv.abs_regret",
]


def _objective_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(
        (column for column in frame.columns if column.startswith("f") and column[1:].isdigit()),
        key=lambda column: int(column[1:]),
    )


def _non_dominated(objectives: np.ndarray) -> np.ndarray:
    objectives = np.asarray(objectives, dtype=float)
    objectives = objectives[np.isfinite(objectives).all(axis=1)]
    indices = NonDominatedSorting().do(objectives, only_non_dominated_front=True)
    return objectives[indices]


def _original_results_path(run_dir: Path) -> Path:
    preserved = run_dir / ORIGINAL_RESULTS_NAME
    return preserved if preserved.exists() else run_dir / "run_results.csv"


def _original_aggregate_path(experiment_dir: Path) -> Path:
    preserved = experiment_dir / ORIGINAL_AGGREGATE_NAME
    return preserved if preserved.exists() else experiment_dir / "aggregate_results.csv"


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(data: dict, path: Path) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _build_empirical_front(original_front: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    fronts = [original_front]
    sources = []
    for method, experiment_dir in FINAL_METHOD_EXPERIMENTS.items():
        run_dirs = sorted(experiment_dir.glob("run_*"))
        if len(run_dirs) != 40:
            raise ValueError(f"Expected 40 runs for {method}, found {len(run_dirs)}")

        n_points = 0
        for run_dir in run_dirs:
            optimum = pd.read_csv(run_dir / "final_optimum.csv")
            objective_columns = _objective_columns(optimum)
            if len(objective_columns) != 2:
                raise ValueError(f"Expected two objectives in {run_dir / 'final_optimum.csv'}")
            values = optimum[objective_columns].to_numpy(dtype=float)
            fronts.append(values)
            n_points += len(values)

        sources.append(
            {
                "method": method,
                "experiment_dir": str(experiment_dir.relative_to(REPO_ROOT)),
                "n_runs": len(run_dirs),
                "n_final_optimum_points": n_points,
            }
        )

    candidates = np.unique(np.vstack(fronts), axis=0)
    empirical_front = _non_dominated(candidates)
    order = np.lexsort((empirical_front[:, 1], empirical_front[:, 0]))
    return empirical_front[order], sources


def _recalculate_run(
    run_dir: Path,
    old_true_hv: float,
    true_hv: float,
) -> tuple[pd.DataFrame, float]:
    results = pd.read_csv(_original_results_path(run_dir))
    missing = [column for column in HV_COLUMNS if column not in results]
    if missing:
        raise ValueError(f"Missing HV columns in {run_dir}: {missing}")
    if "IGD.indicator" in results:
        results = results.rename(
            columns={"IGD.indicator": "IGD.reference_mdgnc_edge_failures"}
        )

    delta0 = None
    previous_ratio = 1.0
    previous_delta = None
    previous_n_eval = None
    regret = 0.0
    absolute_regret = 0.0
    maximum_true_hv_error = 0.0

    for row_index, row in results.iterrows():
        n_eval = int(row["n_eval"])
        maximum_true_hv_error = max(
            maximum_true_hv_error,
            abs(old_true_hv - float(row["delta_hv.true_hv"])),
        )

        # The objective box is intentionally unchanged, so the recorded observed
        # hypervolume remains valid even for algorithms that did not save a
        # cumulative population archive (notably GA).
        observed_hv = float(row["delta_hv.hv"])
        delta = (true_hv - observed_hv) / true_hv
        if delta0 is None:
            delta0 = delta
        ratio = delta / delta0

        n_infill = 0 if previous_n_eval is None else n_eval - previous_n_eval
        regret += 0.5 * (ratio + previous_ratio) * n_infill
        if previous_delta is None:
            previous_delta = delta
        absolute_regret += 0.5 * (delta + previous_delta) * n_infill

        results.loc[row_index, "delta_hv.delta_hv"] = delta
        results.loc[row_index, "delta_hv.hv"] = observed_hv
        results.loc[row_index, "delta_hv.true_hv"] = true_hv
        results.loc[row_index, "delta_hv.ratio"] = ratio
        results.loc[row_index, "delta_hv.regret"] = regret
        results.loc[row_index, "delta_hv.abs_regret"] = absolute_regret

        previous_ratio = ratio
        previous_delta = delta
        previous_n_eval = n_eval

    return results, maximum_true_hv_error


def main() -> None:
    original_front = np.load(ORIGINAL_REFERENCE_DIR / "pf.npy")
    fixed_ideal = original_front.min(axis=0)
    fixed_nadir = original_front.max(axis=0)
    empirical_front, sources = _build_empirical_front(original_front)

    hypervolume = Hypervolume(
        ref_point=fixed_nadir,
        ideal=fixed_ideal,
        nadir=fixed_nadir,
        zero_to_one=True,
    )
    old_hypervolume = Hypervolume(pf=original_front, zero_to_one=True)
    old_true_hv = float(old_hypervolume.do(original_front))
    true_hv = float(hypervolume.do(empirical_front))

    corrected_runs: dict[Path, pd.DataFrame] = {}
    corrected_aggregates: dict[Path, pd.DataFrame] = {}
    maximum_true_hv_error = 0.0
    minimum_delta = np.inf

    for method, experiment_dir in FINAL_METHOD_EXPERIMENTS.items():
        run_frames = []
        for run_dir in sorted(experiment_dir.glob("run_*")):
            corrected, true_hv_error = _recalculate_run(
                run_dir,
                old_true_hv,
                true_hv,
            )
            corrected_runs[run_dir] = corrected
            run_frames.append(corrected)
            maximum_true_hv_error = max(maximum_true_hv_error, true_hv_error)
            minimum_delta = min(minimum_delta, corrected["delta_hv.delta_hv"].min())

        aggregate = pd.concat(run_frames, ignore_index=True)
        original_aggregate = pd.read_csv(_original_aggregate_path(experiment_dir))
        original_keys = original_aggregate[["run", "n_eval"]].reset_index(drop=True)
        corrected_keys = aggregate[["run", "n_eval"]].reset_index(drop=True)
        if not original_keys.equals(corrected_keys):
            raise ValueError(f"Aggregate row mismatch for {method}")
        corrected_aggregates[experiment_dir] = aggregate

    if maximum_true_hv_error > 1e-10:
        raise ValueError(f"Original true HV is inconsistent: error={maximum_true_hv_error}")
    if minimum_delta < -1e-12:
        raise ValueError(f"Corrected reference is still exceeded: minimum delta={minimum_delta}")

    EMPIRICAL_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMPIRICAL_REFERENCE_DIR / "pf.npy", empirical_front)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generation": {
            "method": "nondominated_union_of_original_reference_and_final_method_runs",
            "original_reference": str(ORIGINAL_REFERENCE_DIR.relative_to(REPO_ROOT)),
            "sources": sources,
            "n_runs": sum(source["n_runs"] for source in sources),
        },
        "problem": {
            "name": "mdgnc_edge_failures",
            "n_obj": 2,
        },
        "metric": {
            "normalization_ideal": fixed_ideal.tolist(),
            "normalization_nadir": fixed_nadir.tolist(),
            "reference_point": fixed_nadir.tolist(),
            "original_hv": old_true_hv,
            "empirical_hv": true_hv,
            "recalculated_columns": HV_COLUMNS,
            "legacy_igd_column": "IGD.reference_mdgnc_edge_failures",
        },
        "reference_front": {
            "n_points": int(len(empirical_front)),
            "pf_shape": list(empirical_front.shape),
            "ideal": empirical_front.min(axis=0).tolist(),
            "nadir": empirical_front.max(axis=0).tolist(),
        },
    }
    _write_json(metadata, EMPIRICAL_REFERENCE_DIR / "metadata.json")

    for run_dir, corrected in corrected_runs.items():
        canonical = run_dir / "run_results.csv"
        preserved = run_dir / ORIGINAL_RESULTS_NAME
        if not preserved.exists():
            canonical.replace(preserved)
        _write_csv(corrected, canonical)

    for experiment_dir, corrected in corrected_aggregates.items():
        canonical = experiment_dir / "aggregate_results.csv"
        preserved = experiment_dir / ORIGINAL_AGGREGATE_NAME
        if not preserved.exists():
            canonical.replace(preserved)
        _write_csv(corrected, canonical)

    print(f"Empirical front: {len(empirical_front)} points")
    print(f"Original fixed-box HV: {old_true_hv:.15f}")
    print(f"Empirical fixed-box HV: {true_hv:.15f}")
    print(f"Runs recalculated: {len(corrected_runs)}")
    print(f"Maximum original true-HV error: {maximum_true_hv_error:.3e}")
    print(f"Minimum corrected delta HV: {minimum_delta:.6f}")


if __name__ == "__main__":
    main()
