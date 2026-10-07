"""Post-run spike analyses that never participate in simulation causality."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .errors import ResolutionError
from .simulation import NetworkSpike, SpikeSeries


def _events(spikes: SpikeSeries | Iterable[NetworkSpike]) -> tuple[NetworkSpike, ...]:
    values = tuple(spikes.events if isinstance(spikes, SpikeSeries) else spikes)
    if any(not isinstance(item, NetworkSpike) for item in values):
        raise ResolutionError("spike analysis requires NetworkSpike records")
    return values


def spike_count(
    spikes: SpikeSeries | Iterable[NetworkSpike],
) -> dict[int, int]:
    """Count spikes independently for every observed node."""

    result: dict[int, int] = {}
    for item in _events(spikes):
        result[item.node] = result.get(item.node, 0) + 1
    return result


def firing_rate(
    spikes: SpikeSeries | Iterable[NetworkSpike],
    *,
    t_start: float,
    t_end: float,
    nodes: Iterable[int] | None = None,
) -> dict[int, float]:
    """Per-node spike count divided by one half-open observation duration."""

    start = float(t_start)
    end = float(t_end)
    if not math.isfinite(start) or not math.isfinite(end) or not start < end:
        raise ResolutionError("firing-rate window requires finite start < end")
    values = tuple(item for item in _events(spikes) if start <= item.t < end)
    selected = (
        tuple(dict.fromkeys(int(node) for node in nodes))
        if nodes is not None
        else tuple(sorted({item.node for item in values}))
    )
    counts = {node: 0 for node in selected}
    for item in values:
        if item.node in counts:
            counts[item.node] += 1
    duration = end - start
    return {node: count / duration for node, count in counts.items()}


def interspike_intervals(
    spikes: SpikeSeries | Iterable[NetworkSpike],
) -> dict[int, tuple[float, ...]]:
    """Chronological adjacent-spike intervals for each node."""

    grouped: dict[int, list[float]] = {}
    for item in sorted(_events(spikes), key=lambda value: (value.node, value.t)):
        grouped.setdefault(item.node, []).append(item.t)
    return {
        node: tuple(right - left for left, right in zip(times, times[1:]))
        for node, times in grouped.items()
    }


def coefficient_of_variation(
    spikes: SpikeSeries | Iterable[NetworkSpike],
) -> dict[int, float | None]:
    """Population-independent ISI standard deviation divided by mean.

    Nodes with fewer than two intervals return ``None``.
    """

    result: dict[int, float | None] = {}
    for node, intervals in interspike_intervals(spikes).items():
        if len(intervals) < 2:
            result[node] = None
            continue
        mean = sum(intervals) / len(intervals)
        variance = sum((value - mean) ** 2 for value in intervals) / len(intervals)
        result[node] = math.sqrt(variance) / mean if mean > 0.0 else None
    return result


@dataclass(frozen=True)
class PopulationRate:
    """Half-open bin starts and rates in spikes per time unit per node."""

    times: tuple[float, ...]
    values: tuple[float, ...]
    bin_width: float
    node_count: int

    def to_numpy(self):
        """Return rate samples as a NumPy array."""

        try:
            import numpy
        except ImportError as exc:
            raise RuntimeError("NumPy is required for to_numpy()") from exc
        return numpy.asarray((self.times, self.values), dtype=float).T


def population_rate(
    spikes: SpikeSeries | Iterable[NetworkSpike],
    *,
    t_start: float,
    t_end: float,
    bin_width: float,
    nodes: Iterable[int] | None = None,
) -> PopulationRate:
    """Bin spikes and normalize by bin duration and selected node count."""

    start = float(t_start)
    end = float(t_end)
    width = float(bin_width)
    if (
        not math.isfinite(start)
        or not math.isfinite(end)
        or not math.isfinite(width)
        or not start < end
        or width <= 0.0
    ):
        raise ResolutionError(
            "population rate requires finite start < end and positive bin_width"
        )
    values = _events(spikes)
    selected = (
        frozenset(int(node) for node in nodes)
        if nodes is not None
        else frozenset(item.node for item in values)
    )
    if not selected:
        raise ResolutionError("population rate requires at least one selected node")
    count = int(math.ceil((end - start) / width))
    bins = [0] * count
    for item in values:
        if item.node in selected and start <= item.t < end:
            index = min(int((item.t - start) / width), count - 1)
            bins[index] += 1
    times = tuple(start + index * width for index in range(count))
    rates = tuple(
        value
        / (min(width, end - times[index]) * len(selected))
        for index, value in enumerate(bins)
    )
    return PopulationRate(times, rates, width, len(selected))


__all__ = [
    "spike_count",
    "firing_rate",
    "interspike_intervals",
    "coefficient_of_variation",
    "PopulationRate",
    "population_rate",
]
