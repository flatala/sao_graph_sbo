from pathlib import Path

import math
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experimenter.defaults import *


REPORT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#f2c94c",
    "#9467bd",
]

TIME_COLORS = {
    "pretrain": "#59A14F",
    "train": "#4E79A7",
    "infill": "#F28E2B",
    "other": "#BAB0AC",
}


def set_publication_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.8,
    })


set_publication_plot_style()


def _style_report_figure(fig):
    fig.set_facecolor("white")
    fig.set_edgecolor("white")
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor("white")
    fig.patch.set_alpha(1.0)


def _style_report_axis(ax):
    ax.set_facecolor("white")
    ax.patch.set_alpha(1.0)
    ax.tick_params(colors="#222222")
    ax.xaxis.label.set_color("#222222")
    ax.yaxis.label.set_color("#222222")
    ax.title.set_color("#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#BBBBBB")
        spine.set_linewidth(0.8)
    ax.grid(False)


def _show_report_figure(fig):
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        plt.show()
        return

    if get_ipython() is None:
        plt.show()
        return

    buffer = BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=plt.rcParams["figure.dpi"],
        facecolor="white",
        edgecolor="white",
        transparent=False,
        bbox_inches="tight",
    )
    display(Image(data=buffer.getvalue()))
    plt.close(fig)


def _load_results(results) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results

    path = Path(results)
    csv_path = path / "aggregate_results.csv" if path.is_dir() else path
    return pd.read_csv(csv_path)


def _final_per_run(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "n_eval" not in df.columns:
        raise KeyError("results must contain an 'n_eval' column")
    if "run" not in df.columns:
        df["run"] = 0
    return (
        df.sort_values(["run", "n_eval"])
          .groupby("run", as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )



def _time_breakdown_per_run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, run_df in df.groupby("run", sort=False):
        run_df = run_df.sort_values("n_eval", kind="stable")
        total = pd.to_numeric(run_df["runtime_sec"], errors="coerce").iloc[-1]
        train = pd.to_numeric(run_df["time.train"], errors="coerce").fillna(0.0).sum()
        infill = pd.to_numeric(run_df["time.infill"], errors="coerce").fillna(0.0).sum()
        # Pretraining is a one-time cost reported every step (constant), not a per-step
        # increment, so take the max rather than summing. It sits on top of runtime_sec.
        if "time.pretrain" in run_df.columns:
            pretrain = pd.to_numeric(run_df["time.pretrain"], errors="coerce").max()
            pretrain = float(pretrain) if np.isfinite(pretrain) else 0.0
        else:
            pretrain = 0.0
        rows.append(
            {
                "run": run,
                "runtime_sec": total,
                "time.pretrain": pretrain,
                "time.train": train,
                "time.infill": infill,
                "time.other": max(total - train - infill, 0.0),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics_table(
    results: pd.DataFrame,
    specs=None,
    digits: int = 4,
) -> pd.DataFrame:
    if specs is None:
        specs = DEFAULT_SPECS
    final = _final_per_run(results)

    rows = []
    for label, col, _logy in specs:
        if col not in final.columns:
            continue
        s = pd.to_numeric(final[col], errors="coerce")
        rows.append({
            "metric": label,
            "col": col,
            "mean": float(np.nanmean(s)),
            "std": float(np.nanstd(s, ddof=1)) if np.sum(~np.isnan(s)) > 1 else np.nan,
            "min": float(np.nanmin(s)),
            "max": float(np.nanmax(s)),
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # nicer print formatting
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        printable = out.copy()
        printable["mean +/- std"] = [
            f"{mean:.{digits}g} +/- {std:.{digits}g}" if pd.notna(std) else f"{mean:.{digits}g} +/- nan"
            for mean, std in zip(printable["mean"], printable["std"])
        ]
        for c in ["mean", "std"]:
            printable = printable.drop(columns=c)
        for c in ["min", "max"]:
            printable[c] = printable[c].map(lambda x: f"{x:.{digits}g}" if pd.notna(x) else "nan")
        print(printable.to_string(index=False, col_space=18))

    return out


def summarize_final_metric_averages(
    results: pd.DataFrame,
    specs=None,
    digits: int = 4,
) -> pd.DataFrame:
    if specs is None:
        specs = DEFAULT_SPECS

    specs = [spec for spec in specs if spec[1] in results.columns]
    if "experiment" in results.columns:
        final = (
            results.sort_values(["run", "n_eval"], kind="stable")
            .groupby(["experiment", "run"], as_index=False, sort=False)
            .tail(1)
            .reset_index(drop=True)
        )
        grouped = final.groupby("experiment", sort=False)
    else:
        final = _final_per_run(results)
        grouped = [("final average", final)]

    rows = []
    for experiment, df in grouped:
        row = {"experiment": experiment}
        for label, col, _logy in specs:
            values = pd.to_numeric(df[col], errors="coerce")
            row[label] = values.mean()
            row[f"{label} std"] = values.std(ddof=1)
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        printable = pd.DataFrame({"experiment": out["experiment"]})
        for label, _col, _logy in specs:
            mean_col = label
            std_col = f"{label} std"
            if mean_col not in out.columns:
                continue
            printable[label] = [
                f"{mean:.{digits}g} +/- {std:.{digits}g}" if pd.notna(std) else f"{mean:.{digits}g} +/- nan"
                for mean, std in zip(out[mean_col], out[std_col])
            ]
        print(printable.to_string(index=False, col_space=18))

    return out


def plot_metrics_grid(
    results: pd.DataFrame,
    specs=None,
    ncols: int = 4,
    figsize_per_ax=(4.2, 3.2),
    band: str = "minmax",  # "minmax", "std", or "p75"
    color=REPORT_COLORS[0],
):
    """
    For time-series metrics: plots mean with a band (min-max or +-std) over n_eval.
    For runtime_sec: bar plot per run with mean line.
    """
    if specs is None:
        specs = DEFAULT_SPECS
    if results.empty:
        raise ValueError("Empty results DataFrame.")

    specs = [s for s in specs if s[1] in results.columns]
    if not specs:
        raise ValueError("None of the requested metric columns exist in results.")

    n = len(specs)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))

    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), facecolor="white")
    _style_report_figure(fig)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (label, col, logy) in enumerate(specs):
        ax = axes[i]
        _style_report_axis(ax)

        if col == "runtime_sec":
            timing = _time_breakdown_per_run(results)
            x = timing["run"].values
            pretrain = timing["time.pretrain"].to_numpy()
            train = timing["time.train"].to_numpy()
            infill = timing["time.infill"].to_numpy()
            other = timing["time.other"].to_numpy()
            has_pretrain = np.nansum(pretrain) > 0
            bottom = pretrain if has_pretrain else np.zeros_like(train)
            total = timing["runtime_sec"].to_numpy() + (pretrain if has_pretrain else 0.0)
            if has_pretrain:
                ax.bar(x, pretrain, label="pretrain", color=TIME_COLORS["pretrain"])
            ax.bar(x, train, bottom=bottom, label="train", color=TIME_COLORS["train"])
            ax.bar(x, infill, bottom=bottom + train, label="infill", color=TIME_COLORS["infill"])
            ax.bar(x, other, bottom=bottom + train + infill, label="other", color=TIME_COLORS["other"])
            ax.axhline(np.nanmean(total), linestyle="--", color="#222222", linewidth=1.0)
            ax.set_title(label)
            ax.set_xlabel("run")
            ax.set_ylabel("time [s]")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4 if has_pretrain else 3, frameon=False)
            if logy:
                ax.set_yscale("log")
            continue

        # time-series: aggregate by n_eval across runs
        tmp = results[["n_eval", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        g = tmp.groupby("n_eval")[col]
        mean = g.mean()
        if band == "std":
            std = g.std(ddof=1)
            lo = mean - std
            hi = mean + std
        elif band == "p75":
            lo = g.quantile(0.125)
            hi = g.quantile(0.875)
        else:
            lo = g.min()
            hi = g.max()

        x = mean.index.to_numpy()
        y = mean.to_numpy()
        ax.plot(x, y, color=color)
        ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), color=color, alpha=0.2)

        ax.set_title(label)
        ax.set_xlabel("n_eval")
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("log")

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    _show_report_figure(fig)


def report_experiment(
    results,
    specs=None,
    ncols: int = 4,
    digits: int = 4,
    band: str = "minmax",
    color=REPORT_COLORS[0],
):
    results = _load_results(results)
    summary = summarize_metrics_table(results, specs=specs, digits=digits)
    plot_metrics_grid(results, specs=specs, ncols=ncols, band=band, color=color)
    return summary


def report_experiments(
    paths,
    labels=None,
    specs=None,
    ncols: int = 4,
    digits: int = 4,
    band: str = "minmax",
    colors=None,
    show_plot: bool = True,
):
    if specs is None:
        specs = DEFAULT_SPECS
    if labels is None:
        labels = [str(path).rstrip("/").split("/")[-1] for path in paths]
    if len(labels) != len(paths):
        raise ValueError("labels must match paths")

    frames = []
    for path, label in zip(paths, labels):
        path_str = str(path)
        if path_str.startswith(("http://", "https://")):
            csv_path = path_str if path_str.endswith(".csv") else f"{path_str.rstrip('/')}/aggregate_results.csv"
        else:
            path = Path(path)
            csv_path = path / "aggregate_results.csv" if path.is_dir() else path
        df = pd.read_csv(csv_path)
        df = df.copy()
        df["experiment"] = label
        frames.append(df)
    results = pd.concat(frames, ignore_index=True)

    specs = [spec for spec in specs if spec[1] in results.columns]
    final_summary = summarize_final_metric_averages(results, specs=specs, digits=digits)
    n = len(specs)
    ncols = min(ncols, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), facecolor="white")
    _style_report_figure(fig)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    if colors is None:
        colors = [REPORT_COLORS[i % len(REPORT_COLORS)] for i in range(len(labels))]
    for i, (title, col, logy) in enumerate(specs):
        ax = axes[i]
        _style_report_axis(ax)
        if col == "runtime_sec":
            pretrain_means = []
            train_means = []
            infill_means = []
            other_means = []
            for label in labels:
                timing = _time_breakdown_per_run(results[results["experiment"] == label])
                pretrain_means.append(timing["time.pretrain"].mean())
                train_means.append(timing["time.train"].mean())
                infill_means.append(timing["time.infill"].mean())
                other_means.append(timing["time.other"].mean())
            pretrain_means = np.asarray(pretrain_means, dtype=float)
            train_means = np.asarray(train_means, dtype=float)
            infill_means = np.asarray(infill_means, dtype=float)
            other_means = np.asarray(other_means, dtype=float)
            has_pretrain = np.nansum(pretrain_means) > 0
            left = pretrain_means if has_pretrain else np.zeros_like(train_means)
            y = np.arange(len(labels))
            if has_pretrain:
                ax.barh(y, pretrain_means, label="pretrain", color=TIME_COLORS["pretrain"])
            ax.barh(y, train_means, left=left, label="train", color=TIME_COLORS["train"])
            ax.barh(y, infill_means, left=left + train_means, label="infill", color=TIME_COLORS["infill"])
            ax.barh(
                y,
                other_means,
                left=left + train_means + infill_means,
                label="other",
                color=TIME_COLORS["other"],
            )
            ax.set_title("time")
            ax.set_xlabel("time [s]")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            totals = left + train_means + infill_means + other_means
            x_max = max(totals.max() if totals.size else 1.0, 1.0)
            ax.set_xlim(0, x_max * 1.25)
            runtime_legend = ax.legend(loc="upper right", frameon=True)
            runtime_legend.get_frame().set_facecolor("white")
            runtime_legend.get_frame().set_edgecolor("#DDDDDD")
            runtime_legend.get_frame().set_alpha(1.0)
            for text in runtime_legend.get_texts():
                text.set_color("#222222")
            if logy:
                ax.set_xscale("log")
            continue

        for label, color in zip(labels, colors):
            df = results[results["experiment"] == label][["n_eval", col]].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            grouped = df.groupby("n_eval")[col]
            mean = grouped.mean()
            if band == "std":
                std = grouped.std(ddof=1)
                lo, hi = mean - std, mean + std
            elif band == "p75":
                lo, hi = grouped.quantile(0.125), grouped.quantile(0.875)
            else:
                lo, hi = grouped.min(), grouped.max()
            ax.plot(mean.index, mean.values, label=label, color=color)
            ax.fill_between(mean.index, lo.values, hi.values, color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("n_eval")
        ax.set_ylabel(col)
        if logy:
            ax.set_yscale("log")

    for ax in axes[n:]:
        ax.axis("off")

    handles, legend_labels = [], []
    for ax in axes[:n]:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        if any(label in labels for label in axis_labels):
            handles, legend_labels = axis_handles, axis_labels
            break
    if len(handles) == 0:
        handles, legend_labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(legend_labels), 4),
        frameon=True,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#DDDDDD")
    legend.get_frame().set_alpha(1.0)
    for text in legend.get_texts():
        text.set_color("#222222")
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.95))
    if show_plot:
        _show_report_figure(fig)
    else:
        plt.close(fig)
    return results, final_summary
