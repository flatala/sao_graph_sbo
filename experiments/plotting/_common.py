from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RESULTS_ROOT = REPO_ROOT / "experiments"
PLOTS_ROOT = RESULTS_ROOT / "plots"


def save_figure(fig, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "bbox_inches": "tight",
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    for suffix in ("pdf", "svg"):
        fig.savefig(output_dir / f"{filename}.{suffix}", **kwargs)
    plt.close(fig)
