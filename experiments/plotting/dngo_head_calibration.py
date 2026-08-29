from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure
from experimenter.reporting import set_publication_plot_style


RESULTS_DIR = RESULTS_ROOT / "gnn" / "dngo_head_calibration"
RESULTS_PATH = RESULTS_DIR / "aggregate_results.csv"
N_TRAINS = (50, 100, 200)
COLORS = {50: "#0072B2", 100: "#E69F00", 200: "#009E73"}
XTICKS = (0, 25, 50, 75, 100, 150, 200)


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH)


def _mean_ci95(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    metrics = ["test_rmse", "sd_ratio"]
    grouped = data.groupby(group_columns, sort=True)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    count = grouped.count()
    ci95 = (1.96 * grouped.std(ddof=1) / np.sqrt(count)).add_suffix("_ci95")
    return mean.join(ci95).reset_index()


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    head = results[results["method"] == "dngo"]
    summary = _mean_ci95(head, ["n_train", "epochs"])
    return summary.rename(
        columns={
            "test_rmse_mean": "test RMSE",
            "test_rmse_ci95": "test RMSE 95% CI",
            "sd_ratio_mean": "sigma / RMSE",
            "sd_ratio_ci95": "sigma / RMSE 95% CI",
        }
    )


def make_figure(
    results: pd.DataFrame,
) -> plt.Figure:
    plt.style.use("default")
    set_publication_plot_style()
    plt.rcParams.update(
        {
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "axes.edgecolor": "#888888",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )
    head = results[results["method"] == "dngo"]
    ensemble = results[results["method"] == "random_feature_ensemble"]
    head_summary = _mean_ci95(head, ["n_train", "epochs"])
    ensemble_summary = _mean_ci95(ensemble, ["n_train"]).set_index("n_train")

    fig, (accuracy_ax, calibration_ax) = plt.subplots(
        1, 2, figsize=(7.25, 3.0), sharex=True, facecolor="white"
    )
    for n_train in N_TRAINS:
        subset = head_summary[head_summary["n_train"] == n_train]
        epochs = subset["epochs"].to_numpy(dtype=float)
        color = COLORS[n_train]
        for ax, metric in (
            (accuracy_ax, "test_rmse"),
            (calibration_ax, "sd_ratio"),
        ):
            mean = subset[f"{metric}_mean"].to_numpy(dtype=float)
            ci95 = subset[f"{metric}_ci95"].fillna(0).to_numpy(dtype=float)
            ax.plot(
                epochs,
                mean,
                "o-",
                color=color,
                markersize=3.8,
                markerfacecolor="white",
                markeredgewidth=0.9,
                label=f"n = {n_train}",
            )
            ax.fill_between(
                epochs,
                np.maximum(mean - ci95, np.finfo(float).tiny),
                mean + ci95,
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            baseline = ensemble_summary.loc[n_train, f"{metric}_mean"]
            ax.axhline(
                baseline,
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                alpha=0.55,
            )

    accuracy_ax.set_title("(a) Predictive accuracy stabilizes early")
    accuracy_ax.set_ylabel("Test RMSE")
    accuracy_ax.set_ylim(bottom=0)

    calibration_ax.axhline(1.0, color="#333333", linewidth=0.8)
    calibration_ax.text(
        XTICKS[-1],
        1.025,
        "matching uncertainty/error scale",
        fontsize=7.5,
        ha="right",
        va="bottom",
    )
    calibration_ax.set_title("(b) Predicted uncertainty collapses")
    calibration_ax.set_ylabel(r"Mean predicted $\sigma$ / test RMSE")
    calibration_ax.set_yscale("log")
    calibration_ax.set_ylim(0.3, 1.2)
    calibration_ax.yaxis.set_major_locator(FixedLocator([0.3, 0.4, 0.5, 0.7, 1.0]))
    calibration_ax.set_yticklabels(["0.3", "0.4", "0.5", "0.7", "1.0"])
    calibration_ax.yaxis.set_minor_formatter(NullFormatter())

    for ax in (accuracy_ax, calibration_ax):
        ax.set_xticks(XTICKS)
        ax.set_xlabel("DNGO head training epochs")
        ax.grid(axis="y", color="#E6E6E6", linewidth=0.6)

    handles, labels = accuracy_ax.get_legend_handles_labels()
    handles.append(
        Line2D(
            [],
            [],
            color="#666666",
            linestyle=(0, (4, 3)),
            linewidth=1.0,
        )
    )
    labels.append("Random-feature ensemble")
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.27, top=0.88, wspace=0.30)
    return fig


def main() -> None:
    figure = make_figure(load_results())
    save_figure(
        figure,
        PLOTS_ROOT / "dngo_head_calibration",
        "dngo_head_accuracy_and_calibration",
    )


if __name__ == "__main__":
    main()
