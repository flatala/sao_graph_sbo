from __future__ import annotations

from typing import Literal, NamedTuple


class ThetaParamSpec(NamedTuple):
    """Bounds, optimizer space, and initial value for one theta parameter."""
    lb: float
    ub: float
    space: Literal["log", "linear"]
    init: float
