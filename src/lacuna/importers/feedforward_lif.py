"""Framework-neutral deployment of static convolutional and dense LIF layers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from ..errors import ResolutionError
from ..ir import NeuronPolarity
from ..models import LIF
from ..network import ExplicitConnections, Network, NetworkBuilder
from ..synapses import Delta
from .dense_lif import DenseLIFLayer, _items, _number


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ResolutionError(f"{label} must be an integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ResolutionError(f"{label} must be {qualifier}")
    return int(value)


def _pair(value: object, label: str, minimum: int = 1) -> tuple[int, int]:
    items = _items(value, label)
    if len(items) != 2:
        raise ResolutionError(f"{label} must contain two integers")
    return tuple(_integer(item, label, minimum) for item in items)


def normalize_input_shape(input_shape: object) -> tuple[int, ...]:
    """Validate a feature vector or channel-first spatial shape."""

    shape = _items(input_shape, "input_shape")
    if len(shape) not in (1, 3):
        raise ResolutionError("input_shape must be (features,) or (C, H, W)")
    return tuple(_integer(value, "input_shape dimension") for value in shape)


@dataclass(frozen=True)
class Conv2dLIFLayer:
    """A static cross-correlation kernel followed by ordinary LIF neurons.

    Weights have [output, input within group, kernel row, kernel column]
    order. Spatial populations use channel, row, column order. Padding is
    zero padding, with no connections outside the input field. Rest, reset,
    drive, and refractory duration are zero, as for ``DenseLIFLayer``.
    """

    weights: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...]
    tau_m: float
    threshold: float
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    groups: int = 1
    delay: float = 0.0

    def __post_init__(self) -> None:
        weights = tuple(
            tuple(
                tuple(
                    tuple(
                        _number(value, "kernel weight")
                        for value in _items(row, "kernel row")
                    )
                    for row in _items(channel, "kernel channel")
                )
                for channel in _items(kernel, "output kernel")
            )
            for kernel in _items(self.weights, "weights")
        )
        channels = len(weights[0])
        rows = len(weights[0][0])
        columns = len(weights[0][0][0])
        if any(
            len(kernel) != channels
            or any(
                len(channel) != rows or any(len(row) != columns for row in channel)
                for channel in kernel
            )
            for kernel in weights
        ):
            raise ResolutionError(
                "weights must form a rectangular four-dimensional kernel"
            )
        object.__setattr__(self, "weights", weights)
        for field in ("tau_m", "threshold", "delay"):
            value = _number(getattr(self, field), field)
            if field == "delay":
                if value < 0:
                    raise ResolutionError("delay must be nonnegative")
            elif value <= 0:
                raise ResolutionError(f"{field} must be positive")
            object.__setattr__(self, field, value)
        for field in ("stride", "padding", "dilation"):
            object.__setattr__(
                self,
                field,
                _pair(getattr(self, field), field, 0 if field == "padding" else 1),
            )
        groups = _integer(self.groups, "groups")
        if len(weights) % groups:
            raise ResolutionError("output channels must be divisible by groups")
        object.__setattr__(self, "groups", groups)

    def output_shape(self, input_shape: object) -> tuple[int, int, int]:
        """Return the channel-first output shape and validate kernel geometry."""

        shape = normalize_input_shape(input_shape)
        if len(shape) != 3:
            raise ResolutionError(
                "convolution requires a spatial (C, H, W) input_shape"
            )
        channels, rows, columns = shape
        if channels != len(self.weights[0]) * self.groups:
            raise ResolutionError(
                "input channels must match kernel channels times groups"
            )
        kernel_shape = (len(self.weights[0][0]), len(self.weights[0][0][0]))
        output = tuple(
            (size + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1
            for size, pad, dilation, kernel, stride in zip(
                (rows, columns), self.padding, self.dilation, kernel_shape, self.stride
            )
        )
        if min(output) <= 0:
            raise ResolutionError("convolution kernel does not fit the padded input")
        return len(self.weights), output[0], output[1]


@dataclass(frozen=True)
class FeedforwardLIFDeployment:
    """An ordinary graph and the flattened spatial layout of each layer.

    Supply binary input spikes in channel, row, column order, with at most
    one spike per input at each timestamp. Dense layers flatten this order.
    The input relays reproduce the supplied events without preprocessing.
    """

    network: Network
    input_ports: tuple[str, ...]
    layer_nodes: tuple[tuple[int, ...], ...]
    input_shape: tuple[int, ...]
    layer_shapes: tuple[tuple[int, ...], ...]


def _convolution_edges(layer, input_shape, output_shape):
    _, input_rows, input_columns = input_shape
    output_channels, output_rows, output_columns = output_shape
    channels_per_group = len(layer.weights[0])
    outputs_per_group = output_channels // layer.groups
    for output_channel, kernel in enumerate(layer.weights):
        channel_offset = (output_channel // outputs_per_group) * channels_per_group
        for output_row in range(output_rows):
            for output_column in range(output_columns):
                target = (
                    output_channel * output_rows + output_row
                ) * output_columns + output_column
                for local_channel, channel in enumerate(kernel):
                    input_channel = channel_offset + local_channel
                    for kernel_row, row in enumerate(channel):
                        input_row = (
                            output_row * layer.stride[0]
                            - layer.padding[0]
                            + kernel_row * layer.dilation[0]
                        )
                        if not 0 <= input_row < input_rows:
                            continue
                        for kernel_column, weight in enumerate(row):
                            input_column = (
                                output_column * layer.stride[1]
                                - layer.padding[1]
                                + kernel_column * layer.dilation[1]
                            )
                            if weight != 0 and 0 <= input_column < input_columns:
                                source = (
                                    input_channel * input_rows + input_row
                                ) * input_columns + input_column
                                yield source, target, weight


def build_feedforward_lif(
    layers: Sequence[Conv2dLIFLayer | DenseLIFLayer],
    input_shape: tuple[int, ...],
    name: str = "imported-feedforward-lif",
) -> FeedforwardLIFDeployment:
    """Expand trained kernels into signed static edges without a dense matrix."""

    selected = _items(layers, "layers")
    shape = normalize_input_shape(input_shape)
    normalized_input_shape = shape
    shapes = []
    for index, layer in enumerate(selected):
        if isinstance(layer, Conv2dLIFLayer):
            shape = layer.output_shape(shape)
        elif isinstance(layer, DenseLIFLayer):
            if len(layer.weights[0]) != math.prod(shape):
                raise ResolutionError(
                    f"layer {index} input width must match the flattened previous output width"
                )
            shape = (len(layer.weights),)
        else:
            raise ResolutionError(
                "layers must contain Conv2dLIFLayer or DenseLIFLayer instances"
            )
        shapes.append(shape)

    builder = NetworkBuilder(name)
    previous = builder.population(
        "input",
        math.prod(normalized_input_shape),
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
    shape = normalized_input_shape
    for index, (layer, output_shape) in enumerate(zip(selected, shapes)):
        current = builder.population(
            f"layer_{index}",
            math.prod(output_shape),
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
        if isinstance(layer, Conv2dLIFLayer):
            edges = _convolution_edges(layer, shape, output_shape)
        else:
            edges = (
                (source, target, row[source])
                for source in range(len(previous))
                for target, row in enumerate(layer.weights)
                if row[source] != 0
            )
        pairs = []
        weights = []
        for source, target, weight in edges:
            pairs.append((source, target))
            weights.append(weight)
        builder.connect(
            previous,
            current,
            synapse=Delta(),
            pattern=ExplicitConnections(tuple(pairs)),
            weight=tuple(weights),
            delay=layer.delay,
        )
        builder.outputs(f"layer_{index}_spikes", current)
        layer_nodes.append(current.node_ids)
        previous = current
        shape = output_shape

    return FeedforwardLIFDeployment(
        network=builder.build(),
        input_ports=input_ports,
        layer_nodes=tuple(layer_nodes),
        input_shape=normalized_input_shape,
        layer_shapes=tuple(shapes),
    )
