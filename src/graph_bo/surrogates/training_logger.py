from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


class TrainingLogger:
    """Sink for theta-log lines and per-phase timing summaries.

    `label_provider` is called lazily to obtain the context string included in
    each log entry, so callers don't have to thread the label through every call.

    log_path=None prints to stdout; otherwise lines are appended to the file.
    Timing recording is a no-op unless enable_timing=True.
    """

    def __init__(
        self,
        label_provider: Callable[[], str],
        log_path: Path | None = None,
        enable_timing: bool = False,
    ):
        self.label_provider = label_provider
        self.log_path = log_path
        self.enable_timing = bool(enable_timing)
        self._timings: dict[str, dict[str, list[float]]] = {
            "train": defaultdict(lambda: [0.0, 0]),
            "predict": defaultdict(lambda: [0.0, 0]),
        }
        self._theta_logged = False

    ### ---------- output ----------

    def set_log_path(self, log_path: Path | None) -> None:
        self.log_path = log_path

    def emit(self, lines: str | Iterable[str]) -> None:
        message = lines if isinstance(lines, str) else "\n".join(lines)
        if self.log_path is None:
            print(message)
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(message)
            fh.write("\n")

    ### ---------- timing ----------

    @contextmanager
    def time(self, name: str, phase: str):
        if not self.enable_timing:
            yield
            return
        t_start = time.perf_counter()
        try:
            yield
        finally:
            bucket = self._timings[phase][name]
            bucket[0] += time.perf_counter() - t_start
            bucket[1] += 1

    def reset_timings(self, phase: str) -> None:
        self._timings[phase] = defaultdict(lambda: [0.0, 0])

    ### ---------- structured log entries ----------

    def reset_theta_logged(self) -> None:
        self._theta_logged = False

    def log_theta(
        self,
        composition: str,
        branch_weights: np.ndarray | None,
        branch_theta: dict[str, np.ndarray],
        *,
        weight_names: Sequence[str] = (),
    ) -> None:
        if self._theta_logged:
            return

        lines = [
            "",
            f"### ADSGKriging OPTIMIZED THETA | {self.label_provider()} | composition={composition}",
        ]
        if branch_weights is not None:
            lines.append(f"  branch_weights : {branch_weights.tolist()}")
            if len(weight_names) == len(branch_weights):
                lines.append(f"  weight_names   : {list(weight_names)}")
        for branch_name, theta_branch in branch_theta.items():
            lines.append(f"  {branch_name:<16}: {theta_branch.tolist()}")
        self.emit(lines)
        self._theta_logged = True

    def log_timing_summary(self, phase: str) -> None:
        timings = self._timings[phase]
        if not timings:
            return

        title = "TRAINING" if phase == "train" else "PREDICTION"
        lines = [
            "",
            f"### ADSGKriging {title} TIMINGS | {self.label_provider()}",
            f"{'name':<40} {'total_s':>9} {'count':>10} {'mean_s':>11}",
            "-" * 73,
        ]
        for name in sorted(timings):
            total, count = timings[name]
            mean = total / count if count > 0 else 0.0
            lines.append(f"{name:<40} {total:>9.6f} {count:>10d} {mean:>11.6f}")
        self.emit(lines)

    def flush_phase(self, phase: str) -> None:
        self.log_timing_summary(phase)
        self.reset_timings(phase)
