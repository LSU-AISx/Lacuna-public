"""High-level definitions for the standard executable synapse families."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .errors import ResolutionError
from .graph import GraphSynapse


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{name} must be a finite number")
    return result


class StandardSynapse(ABC):
    """Authoring-only description lowered to an ordinary graph synapse."""

    name: str

    @property
    @abstractmethod
    def source(self) -> str | None:
        """Intrinsic synapse source, or ``None`` for an instantaneous delta."""

    @property
    @abstractmethod
    def output_name(self) -> str | None:
        """Output expression deposited into a compatible receptor."""

    @property
    @abstractmethod
    def initial(self) -> float | tuple[float, ...]:
        """Initial edge-local state in model declaration order."""

    @property
    def model(self) -> GraphSynapse | None:
        """Return the graph model record or ``None`` for a delta synapse."""

        return None if self.source is None else GraphSynapse(self.name, self.source)


@dataclass(frozen=True)
class Delta(StandardSynapse):
    """Instantaneous additive deposit into the postsynaptic readout state."""

    name: str = "delta"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ResolutionError("delta synapse name must be nonempty")

    @property
    def source(self) -> None:
        return None

    @property
    def output_name(self) -> None:
        return None

    @property
    def initial(self) -> float:
        return 0.0


@dataclass(frozen=True)
class ExponentialCurrent(StandardSynapse):
    """One-state additive current with ``ds/dt = -s/tau``."""

    tau: float
    name: str = "exponential_current"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ResolutionError("exponential synapse name must be nonempty")
        if _finite(self.tau, "tau") <= 0.0:
            raise ResolutionError("tau must be positive")

    @property
    def source(self) -> str:
        return f"""synapse StandardExponentialCurrent {{
    params {{ tau : positive = {float(self.tau)!r} }}
    state {{ s }}
    dynamics {{ ds/dt = -s/tau }}
    on_spike {{ s <- s + w }}
    output {{ current = s }}
}}
"""

    @property
    def output_name(self) -> str:
        return "current"

    @property
    def initial(self) -> float:
        return 0.0


@dataclass(frozen=True)
class AlphaCurrent(StandardSynapse):
    """Two-state unit-area alpha current with time constant ``tau``."""

    tau: float
    name: str = "alpha_current"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ResolutionError("alpha synapse name must be nonempty")
        if _finite(self.tau, "tau") <= 0.0:
            raise ResolutionError("tau must be positive")

    @property
    def source(self) -> str:
        return f"""synapse StandardAlphaCurrent {{
    params {{ tau : positive = {float(self.tau)!r} }}
    state {{ s; z }}
    dynamics {{
        ds/dt = -s/tau + z
        dz/dt = -z/tau
    }}
    on_spike {{ z <- z + w/tau^2 }}
    output {{ current = s }}
}}
"""

    @property
    def output_name(self) -> str:
        return "current"

    @property
    def initial(self) -> tuple[float, float]:
        return (0.0, 0.0)


__all__ = [
    "StandardSynapse",
    "Delta",
    "ExponentialCurrent",
    "AlphaCurrent",
]
