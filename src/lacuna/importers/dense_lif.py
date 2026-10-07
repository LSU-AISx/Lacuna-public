"""Framework-neutral construction of static dense LIF networks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..errors import ResolutionError
from ..ir import NeuronPolarity
from ..models import LIF
from ..network import Network, NetworkBuilder
from ..synapses import Delta


def _items(value: object, label: str) -> tuple:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not (
        hasattr(value, "__len__") and hasattr(value, "__getitem__")
    ):
        raise ResolutionError(f"{label} must be a nonempty sequence")
    try:
        result = tuple(value[index] for index in range(len(value)))
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise ResolutionError(f"{label} must be a nonempty sequence") from exc
    if not result:
        raise ResolutionError(f"{label} must be a nonempty sequence")
    return result


def _number(value: object, label: str) -> float:
    if isinstance(value, (bool, str, bytes, bytearray)) or hasattr(
        value, "__len__"
    ):
        raise ResolutionError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class DenseLIFLayer:
    """Static weights in [output, input] order and continuous LIF parameters.

    Rest, reset, drive, and refractory duration are zero. Weights are signed
    delta deposits. Threshold equality emits a spike in the Lacuna runtime.
    ``delay`` is the incoming synaptic delay in the same units as ``tau_m``.
    """

    weights: tuple[tuple[float, ...], ...]
    tau_m: float
    threshold: float
    delay: float = 0.0

    def __post_init__(self) -> None:
        rows = _items(self.weights, "weights")
        normalized = tuple(
            tuple(
                _number(value, f"weight [{output}, {source}]")
                for source, value in enumerate(_items(row, "weight row"))
            )
            for output, row in enumerate(rows)
        )
        if any(len(row) != len(normalized[0]) for row in normalized):
            raise ResolutionError("weight rows must have the same width")
        object.__setattr__(self, "weights", normalized)
        for field in ("tau_m", "threshold", "delay"):
            value = _number(getattr(self, field), field)
            if field == "delay":
                if value < 0.0:
                    raise ResolutionError("delay must be nonnegative")
            elif value <= 0.0:
                raise ResolutionError(f"{field} must be positive")
            object.__setattr__(self, field, value)


@dataclass(frozen=True)
class DenseLIFDeployment:
    """An ordinary network with input ports and ordered layer identifiers.

    Supply unit-valued input spikes, with at most one spike per input port at
    each timestamp. The relay neurons reproduce those binary input events.
    """

    network: Network
    input_ports: tuple[str, ...]
    layer_nodes: tuple[tuple[int, ...], ...]


def build_dense_lif(
    layers: Sequence[DenseLIFLayer],
    name: str = "imported-dense-lif",
) -> DenseLIFDeployment:
    """Build a signed feedforward LIF network without training dependencies."""

    selected = _items(layers, "layers")
    if any(not isinstance(layer, DenseLIFLayer) for layer in selected):
        raise ResolutionError("layers must contain DenseLIFLayer instances")
    for index, layer in enumerate(selected[1:], start=1):
        if len(layer.weights[0]) != len(selected[index - 1].weights):
            raise ResolutionError(
                f"layer {index} input width must match "
                "the previous output width"
            )

    builder = NetworkBuilder(name)
    previous = builder.population(
        "input",
        len(selected[0].weights[0]),
        LIF(
            name="input_relay_lif",
            tau_m=1.0,
            v_rest=0.0,
            v_reset=0.0,
            drive=0.0,
            v_threshold=1.0,
            refractory=0.0,
        ),
        polarity=NeuronPolarity.MIXED,
    )
    input_ports = tuple(builder.inputs("input_spikes", previous))
    layer_nodes = []
    for index, layer in enumerate(selected):
        current = builder.population(
            f"layer_{index}",
            len(layer.weights),
            LIF(
                name=f"layer_{index}_lif",
                tau_m=layer.tau_m,
                v_rest=0.0,
                v_reset=0.0,
                drive=0.0,
                v_threshold=layer.threshold,
                refractory=0.0,
            ),
            polarity=NeuronPolarity.MIXED,
        )
        # Lacuna expands dense edges in source-major order.
        weights = tuple(
            row[source]
            for source in range(len(previous))
            for row in layer.weights
        )
        builder.connect(
            previous,
            current,
            synapse=Delta(),
            weight=weights,
            delay=layer.delay,
        )
        builder.outputs(f"layer_{index}_spikes", current)
        layer_nodes.append(current.node_ids)
        previous = current

    return DenseLIFDeployment(
        network=builder.build(),
        input_ports=input_ports,
        layer_nodes=tuple(layer_nodes),
    )
