from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plotting._common import PLOTS_ROOT, RESULTS_ROOT, save_figure


DATA_PATH = RESULTS_ROOT / "gnn" / "encoder_epoch_training" / "epoch_ablation_results.csv"
EPOCHS = [0, 10, 25, 50, 100, 150]
RATIO_METRIC = {"mdgnc": "delta_hv_ratio", "engine": "best_gap_ratio"}
REGRET_METRIC = {"mdgnc": "delta_hv_regret", "engine": "best_gap_regret"}


def ci95(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    return 0 if len(values) <= 1 else 1.96 * values.std(ddof=1) / np.sqrt(len(values))


def epoch_summary(data):
    rows = []
    for (problem, epoch), group in data.groupby(["problem", "epoch"]):
        row = {"problem": problem, "epoch": int(epoch), "n": len(group)}
        for metric in (RATIO_METRIC[problem], REGRET_METRIC[problem], "bo_cutoff_adj_f1", "feature_mae"):
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_ci95"] = ci95(group[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["problem", "epoch"])


def main():
    data = pd.read_csv(DATA_PATH)
    data = data[data["checkpoint_exists"].astype(bool)].copy()
    external_n = data["external_n_samples"].dropna().astype(int).unique()
    external_seed = data["external_seed"].dropna().astype(int).unique()
    if len(external_n) != 1 or len(external_seed) != 1:
        raise ValueError("Expected one external reconstruction sample size and seed")
    summary = epoch_summary(data)

    plt.style.use("default")
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": "#222222"})
    styles = {
        "mdgnc": {"label": "MDGNC", "color": "#008B8B", "linestyle": "-", "marker": "o", "offset": -1.6},
        "engine": {"label": "Jet engine", "color": "#8E3A59", "linestyle": "--", "marker": "D", "offset": 1.6},
    }
    specs = [
        ("(a) ΔHV / Gap Ratio\n(lower is better)", "remaining / initial", lambda problem: RATIO_METRIC[problem]),
        ("(b) ΔHV / Gap Regret\n(lower is better)", "area under ratio", lambda problem: REGRET_METRIC[problem]),
        ("(c) Adjacency F1\n(higher is better)", "F1 @ 0.1", lambda _problem: "bo_cutoff_adj_f1"),
        ("(d) Feature MAE\n(lower is better)", "MAE", lambda _problem: "feature_mae"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.55))
    for ax, (title, ylabel, metric_for) in zip(axes, specs):
        for problem, style in styles.items():
            subset = summary[summary["problem"] == problem].set_index("epoch").reindex(EPOCHS)
            metric = metric_for(problem)
            x = np.asarray(EPOCHS, dtype=float) + style["offset"]
            y = subset[f"{metric}_mean"].to_numpy(dtype=float)
            error = subset[f"{metric}_ci95"].fillna(0).to_numpy(dtype=float)
            valid = np.isfinite(y)
            ax.errorbar(
                x[valid], y[valid], yerr=error[valid], label=style["label"], color=style["color"],
                linestyle=style["linestyle"], marker=style["marker"], linewidth=1.65, markersize=4,
                markerfacecolor="white", markeredgewidth=1.25, capsize=3, elinewidth=0.95,
            )
        ax.set_title(title, fontsize=8.5, pad=5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.set_xticks(EPOCHS)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlim(-7, 157)
        ax.grid(True, color="#D8D8D8", alpha=0.65, linewidth=0.55)
    axes[2].set_ylim(0, 1.02)
    axes[3].set_ylim(0, 0.53)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.03), fontsize=8.5)
    fig.supxlabel("Encoder training epochs", y=0.01, fontsize=9)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.29, top=0.77, wspace=0.42)
    save_figure(fig, PLOTS_ROOT / "encoder_epoch_ablation", "encoder_epoch_trends")


if __name__ == "__main__":
    main()
