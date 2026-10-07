"""High-level construction and persistence API over Lacuna's flat graph IR."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .codec import Decoder, Encoder, HeldCurrentEncoder, NativeEventEncoder
from .errors import ResolutionError
from .graph import (
    Graph,
    GraphEdge,
    GraphNode,
    InputMode,
    InputPort,
    ModulatorPort,
    NeuronPolarity,
    OutputPort,
)
from .ir import ResolvedScalarLIF
from .models import LIF, StandardNeuron
from .synapses import Delta, StandardSynapse
from .plasticity import (
    ModulatedSTDP,
    PairSTDP,
    PlasticityRule,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
)

NETWORK_SCHEMA_VERSION = 1


def _finite(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ResolutionError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ResolutionError(f"{context} must be a finite number") from exc
    if not math.isfinite(result):
        raise ResolutionError(f"{context} must be a finite number")
    return result


def _sequence(value: object) -> bool:
    return (
        not isinstance(value, (str, bytes, bytearray, Mapping))
        and not isinstance(value, (int, float, complex, bool))
        and hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
    )


def _polarity(value: object, context: str) -> NeuronPolarity:
    if isinstance(value, NeuronPolarity):
        return value
    if isinstance(value, str):
        try:
            return NeuronPolarity[value.strip().upper()]
        except KeyError:
            pass
    raise ResolutionError(f"{context} must be EXCITATORY, INHIBITORY, or MIXED")


def _expanded_polarities(
    value: object, count: int, context: str = "polarity"
) -> tuple[NeuronPolarity, ...]:
    if _sequence(value):
        if len(value) != count:
            raise ResolutionError(f"{context} sequence must contain {count} values")
        return tuple(
            _polarity(value[index], f"{context}[{index}]") for index in range(count)
        )
    item = _polarity(value, context)
    return (item,) * count


@dataclass(frozen=True)
class Neuron:
    """Stable reference to one graph neuron."""

    name: str
    id: int
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class NodeSelection:
    """Ordered immutable selection of graph neuron identifiers."""

    name: str
    node_ids: tuple[int, ...]
    _owner: object = field(repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.node_ids)

    def __iter__(self):
        for index, node in enumerate(self.node_ids):
            yield Neuron(f"{self.name}[{index}]", node, self._owner)

    def __getitem__(self, item: int | slice) -> Neuron | "NodeSelection":
        if isinstance(item, slice):
            return NodeSelection(
                self.name,
                self.node_ids[item],
                self._owner,
            )
        node = self.node_ids[item]
        index = item if item >= 0 else len(self.node_ids) + item
        return Neuron(f"{self.name}[{index}]", node, self._owner)


Population = NodeSelection


@dataclass(frozen=True)
class Reservoir:
    """A recurrent population and the explicit edges created inside it."""

    name: str
    neurons: Population
    recurrent_edge_ids: tuple[int, ...]

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Return neuron identifiers in reservoir order."""

        return self.neurons.node_ids

    def __len__(self) -> int:
        return len(self.neurons)

    def __iter__(self):
        return iter(self.neurons)

    def __getitem__(self, item: int | slice):
        return self.neurons[item]


@dataclass(frozen=True)
class PortSelection:
    """Ordered immutable selection of input or output port names."""

    name: str
    port_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.port_ids)

    def __iter__(self):
        return iter(self.port_ids)

    def __getitem__(self, item: int | slice):
        return self.port_ids[item]


class ConnectionPattern:
    """Authoring-only expansion of two node selections into explicit pairs."""

    def pairs(
        self, pre: tuple[int, ...], post: tuple[int, ...]
    ) -> tuple[tuple[int, int], ...]:
        """Expand selected nodes into ordered source and target pairs."""

        raise NotImplementedError

    def weight_group_indices(
        self, pre: tuple[int, ...], post: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Return pair-aligned local sharing groups, or None for independent weights."""

        return None


@dataclass(frozen=True)
class AllToAll(ConnectionPattern):
    """Connect every source to every eligible target."""

    exclude_self: bool = False

    def pairs(self, pre, post):
        """Return every eligible source and target pair."""

        return tuple(
            (source, target)
            for source in pre
            for target in post
            if not self.exclude_self or source != target
        )


@dataclass(frozen=True)
class OneToOne(ConnectionPattern):
    """Connect selections pairwise in their current order."""

    def pairs(self, pre, post):
        """Return position-aligned source and target pairs."""

        if len(pre) != len(post):
            raise ResolutionError("one-to-one connection requires equal selection sizes")
        return tuple(zip(pre, post))


@dataclass(frozen=True)
class LocallyConnected(ConnectionPattern):
    """Connect aligned one-dimensional selections within an index radius.

    Equal-sized selections define the shared local coordinate.  Offset zero is
    included for distinct source and target populations. On a recurrent selection
    it is included only when ``include_self`` is true.  Non-periodic boundaries
    have fewer neighbors, while ``wrap=True`` makes the coordinate a ring.
    """

    radius: int
    include_self: bool = False
    wrap: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.radius, int)
            or isinstance(self.radius, bool)
            or self.radius < 0
        ):
            raise ResolutionError("local connection radius must be a nonnegative integer")
        if not isinstance(self.include_self, bool) or not isinstance(self.wrap, bool):
            raise ResolutionError("local include_self and wrap must be boolean")

    def pairs(self, pre, post):
        """Return neighboring pairs in stable source and offset order."""

        if len(pre) != len(post):
            raise ResolutionError(
                "locally connected selections must have equal sizes"
            )
        count = len(post)
        if count == 0:
            return ()
        result = []
        for source_index, source in enumerate(pre):
            seen: set[int] = set()
            for offset in range(-self.radius, self.radius + 1):
                target_index = source_index + offset
                if self.wrap:
                    target_index %= count
                elif not 0 <= target_index < count:
                    continue
                target = post[target_index]
                if target in seen or (source == target and not self.include_self):
                    continue
                seen.add(target)
                result.append((source, target))
        return tuple(result)


@dataclass(frozen=True)
class FixedProbability(ConnectionPattern):
    """Sample each eligible connection with a fixed probability."""

    probability: float
    seed: int = 0
    exclude_self: bool = False

    def __post_init__(self) -> None:
        probability = _finite(self.probability, "connection probability")
        if not 0.0 <= probability <= 1.0:
            raise ResolutionError("connection probability must lie in [0, 1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ResolutionError("connection seed must be an integer")

    def pairs(self, pre, post):
        """Sample eligible pairs with a deterministic local generator."""

        generator = random.Random(self.seed)
        return tuple(
            (source, target)
            for source in pre
            for target in post
            if (not self.exclude_self or source != target)
            and generator.random() < self.probability
        )


@dataclass(frozen=True)
class FixedOutDegree(ConnectionPattern):
    """Choose exactly ``degree`` postsynaptic targets for every source."""

    degree: int
    seed: int = 0
    exclude_self: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.degree, int)
            or isinstance(self.degree, bool)
            or self.degree < 0
        ):
            raise ResolutionError("fixed out-degree must be a nonnegative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ResolutionError("fixed out-degree seed must be an integer")
        if not isinstance(self.exclude_self, bool):
            raise ResolutionError("fixed out-degree exclude_self must be boolean")

    def pairs(self, pre, post):
        """Expand the pattern into stable source and target identifiers."""

        generator = random.Random(self.seed)
        position = {node: index for index, node in enumerate(post)}
        result = []
        for source in pre:
            eligible = [
                target
                for target in post
                if not self.exclude_self or source != target
            ]
            if self.degree > len(eligible):
                raise ResolutionError(
                    "fixed out-degree exceeds the eligible postsynaptic selection"
                )
            selected = sorted(
                generator.sample(eligible, self.degree), key=position.__getitem__
            )
            result.extend((source, target) for target in selected)
        return tuple(result)


@dataclass(frozen=True)
class FixedInDegree(ConnectionPattern):
    """Choose exactly ``degree`` presynaptic sources for every target."""

    degree: int
    seed: int = 0
    exclude_self: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.degree, int)
            or isinstance(self.degree, bool)
            or self.degree < 0
        ):
            raise ResolutionError("fixed in-degree must be a nonnegative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ResolutionError("fixed in-degree seed must be an integer")
        if not isinstance(self.exclude_self, bool):
            raise ResolutionError("fixed in-degree exclude_self must be boolean")

    def pairs(self, pre, post):
        """Expand the pattern into stable source and target identifiers."""

        generator = random.Random(self.seed)
        source_position = {node: index for index, node in enumerate(pre)}
        result = []
        for target in post:
            eligible = [
                source
                for source in pre
                if not self.exclude_self or source != target
            ]
            if self.degree > len(eligible):
                raise ResolutionError(
                    "fixed in-degree exceeds the eligible presynaptic selection"
                )
            selected = sorted(
                generator.sample(eligible, self.degree),
                key=source_position.__getitem__,
            )
            result.extend((source, target) for source in selected)
        return tuple(sorted(result))


@dataclass(frozen=True)
class ExplicitConnections(ConnectionPattern):
    """Pairs of zero-based local indices into the two selected node lists."""

    indices: tuple[tuple[int, int], ...]

    def pairs(self, pre, post):
        """Map authored local indices onto selected node identifiers."""

        result = []
        for index, pair in enumerate(self.indices):
            if (
                not isinstance(pair, (tuple, list))
                or len(pair) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in pair)
            ):
                raise ResolutionError(
                    f"explicit connection {index} must contain two integer indices"
                )
            source, target = pair
            if not 0 <= source < len(pre) or not 0 <= target < len(post):
                raise ResolutionError(
                    f"explicit connection {index} references a node outside its selection"
                )
            result.append((pre[source], post[target]))
        return tuple(result)


def _spatial_pair(value: int | tuple[int, int], context: str) -> tuple[int, int]:
    values = (value, value) if isinstance(value, int) and not isinstance(value, bool) else value
    if (
        not isinstance(values, (tuple, list))
        or len(values) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
            for item in values
        )
    ):
        raise ResolutionError(f"{context} must be a positive integer or pair")
    return int(values[0]), int(values[1])


@dataclass(frozen=True)
class Convolution2D(ConnectionPattern):
    """NHWC local projection with trainable kernel coefficients shared in space.

    The source and target populations are flattened in row, column, channel
    order.  Each output channel owns one kernel over every input channel.
    Padding is implicit zero padding: connections outside the source field are
    omitted.  Shared plasticity uses mean reduction across spatial copies.
    """

    input_shape: tuple[int, int, int]
    output_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_shape, (tuple, list))
            or len(self.input_shape) != 3
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item <= 0
                for item in self.input_shape
            )
        ):
            raise ResolutionError("convolution input_shape must be three positive integers")
        if (
            not isinstance(self.output_channels, int)
            or isinstance(self.output_channels, bool)
            or self.output_channels <= 0
        ):
            raise ResolutionError("convolution output_channels must be positive")
        _spatial_pair(self.kernel_size, "convolution kernel_size")
        _spatial_pair(self.stride, "convolution stride")
        padding = (
            (self.padding, self.padding)
            if isinstance(self.padding, int) and not isinstance(self.padding, bool)
            else self.padding
        )
        if (
            not isinstance(padding, (tuple, list))
            or len(padding) != 2
            or any(
                not isinstance(item, int) or isinstance(item, bool) or item < 0
                for item in padding
            )
        ):
            raise ResolutionError("convolution padding must be a nonnegative integer or pair")
        if min(self.output_shape[:2]) <= 0:
            raise ResolutionError("convolution kernel does not fit the padded input")

    @property
    def output_shape(self) -> tuple[int, int, int]:
        """Return the valid output field shape for this projection."""

        input_rows, input_columns, _ = self.input_shape
        kernel_rows, kernel_columns = _spatial_pair(
            self.kernel_size, "convolution kernel_size"
        )
        stride_rows, stride_columns = _spatial_pair(
            self.stride, "convolution stride"
        )
        padding_rows, padding_columns = (
            (self.padding, self.padding)
            if isinstance(self.padding, int)
            else self.padding
        )
        return (
            (input_rows + 2 * padding_rows - kernel_rows) // stride_rows + 1,
            (input_columns + 2 * padding_columns - kernel_columns) // stride_columns + 1,
            self.output_channels,
        )

    @property
    def kernel_parameter_count(self) -> int:
        """Return the number of independently trained kernel weights."""

        kernel_rows, kernel_columns = _spatial_pair(
            self.kernel_size, "convolution kernel_size"
        )
        return (
            self.output_channels
            * kernel_rows
            * kernel_columns
            * self.input_shape[2]
        )

    def _expanded(self, pre, post):
        input_rows, input_columns, input_channels = self.input_shape
        output_rows, output_columns, output_channels = self.output_shape
        kernel_rows, kernel_columns = _spatial_pair(
            self.kernel_size, "convolution kernel_size"
        )
        stride_rows, stride_columns = _spatial_pair(
            self.stride, "convolution stride"
        )
        padding_rows, padding_columns = (
            (self.padding, self.padding)
            if isinstance(self.padding, int)
            else self.padding
        )
        if len(pre) != input_rows * input_columns * input_channels:
            raise ResolutionError("convolution source size does not match input_shape")
        if len(post) != output_rows * output_columns * output_channels:
            raise ResolutionError("convolution target size does not match output_shape")
        pairs = []
        groups = []
        for output_row in range(output_rows):
            for output_column in range(output_columns):
                for output_channel in range(output_channels):
                    target_index = (
                        (output_row * output_columns + output_column)
                        * output_channels
                        + output_channel
                    )
                    for kernel_row in range(kernel_rows):
                        input_row = output_row * stride_rows + kernel_row - padding_rows
                        if not 0 <= input_row < input_rows:
                            continue
                        for kernel_column in range(kernel_columns):
                            input_column = (
                                output_column * stride_columns
                                + kernel_column
                                - padding_columns
                            )
                            if not 0 <= input_column < input_columns:
                                continue
                            for input_channel in range(input_channels):
                                source_index = (
                                    (input_row * input_columns + input_column)
                                    * input_channels
                                    + input_channel
                                )
                                group = (
                                    ((output_channel * kernel_rows + kernel_row)
                                     * kernel_columns + kernel_column)
                                    * input_channels
                                    + input_channel
                                )
                                pairs.append((pre[source_index], post[target_index]))
                                groups.append(group)
        return tuple(pairs), tuple(groups)

    def pairs(self, pre, post):
        """Return all in-bounds spatial connection pairs."""

        return self._expanded(pre, post)[0]

    def weight_group_indices(self, pre, post):
        """Return the shared kernel parameter for every expanded pair."""

        return self._expanded(pre, post)[1]


class ValueDistribution:
    """Deterministic expansion of a scalar parameter across connections."""

    def values(self, count: int) -> tuple[float, ...]:
        """Generate exactly ``count`` finite values."""

        raise NotImplementedError


@dataclass(frozen=True)
class Uniform(ValueDistribution):
    """Uniform value distribution with a local random seed."""

    low: float
    high: float
    seed: int = 0

    def values(self, count: int) -> tuple[float, ...]:
        """Draw deterministic uniform values with the configured seed."""

        low = _finite(self.low, "uniform low")
        high = _finite(self.high, "uniform high")
        if high < low:
            raise ResolutionError("uniform high must be at least low")
        generator = random.Random(self.seed)
        return tuple(generator.uniform(low, high) for _ in range(count))


@dataclass(frozen=True)
class Normal(ValueDistribution):
    """Normal value distribution with a local random seed."""

    mean: float
    standard_deviation: float
    seed: int = 0

    def values(self, count: int) -> tuple[float, ...]:
        """Draw deterministic normal values with the configured seed."""

        mean = _finite(self.mean, "normal mean")
        deviation = _finite(self.standard_deviation, "normal standard deviation")
        if deviation < 0.0:
            raise ResolutionError("normal standard deviation must be nonnegative")
        generator = random.Random(self.seed)
        return tuple(generator.gauss(mean, deviation) for _ in range(count))


def _expanded_values(value: object, count: int, context: str) -> tuple[float, ...]:
    if isinstance(value, ValueDistribution):
        result = value.values(count)
    elif _sequence(value):
        if len(value) != count:
            raise ResolutionError(f"{context} sequence must contain {count} values")
        result = tuple(_finite(value[index], f"{context}[{index}]") for index in range(count))
    else:
        scalar = _finite(value, context)
        result = (scalar,) * count
    if len(result) != count or any(not math.isfinite(item) for item in result):
        raise ResolutionError(f"{context} expansion produced invalid values")
    return result


@dataclass(frozen=True)
class PopulationRecord:
    """Serializable population membership record."""

    name: str
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReservoirRecord:
    """Serializable reservoir membership and recurrent edge record."""

    name: str
    node_ids: tuple[int, ...]
    recurrent_edge_ids: tuple[int, ...]


@dataclass(frozen=True)
class NodeCapability:
    """Resolved execution capability for one network node."""

    node: int
    model: str
    polarity: NeuronPolarity
    dispatch: str
    state_count: int
    synapse_tier: str


@dataclass(frozen=True)
class NetworkReport:
    """Resolved size, execution capabilities, and validation warnings."""

    semantic_sha256: str
    node_count: int
    edge_count: int
    weight_parameter_count: int
    shared_weight_group_count: int
    state_count: int
    dispatch_counts: Mapping[str, int]
    synapse_tier_counts: Mapping[str, int]
    nodes: tuple[NodeCapability, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class Network:
    """Immutable high-level network backed by one canonical flat :class:`Graph`."""

    graph: Graph
    name: str = "network"
    population_records: tuple[PopulationRecord, ...] = ()
    reservoir_records: tuple[ReservoirRecord, ...] = ()
    neuron_labels: Mapping[str, int] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)
    _owner: object = field(default_factory=object, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ResolutionError("network name must be nonempty")
        if len({item.name for item in self.population_records}) != len(
            self.population_records
        ):
            raise ResolutionError("population names must be unique")
        if len({item.name for item in self.reservoir_records}) != len(
            self.reservoir_records
        ):
            raise ResolutionError("reservoir names must be unique")
        known = {node.id for node in self.graph.nodes}
        known_edges = {edge.id for edge in self.graph.edges}
        if any(node not in known for item in self.population_records for node in item.node_ids):
            raise ResolutionError("population metadata references an unknown node")
        if any(
            node not in known
            for item in self.reservoir_records
            for node in item.node_ids
        ) or any(
            edge not in known_edges
            for item in self.reservoir_records
            for edge in item.recurrent_edge_ids
        ):
            raise ResolutionError("reservoir metadata references an unknown node or edge")
        if any(node not in known for node in self.neuron_labels.values()):
            raise ResolutionError("neuron label metadata references an unknown node")
        self.graph.resolve()
        _canonical_json(dict(self.metadata))

    @property
    def semantic_sha256(self) -> str:
        """Return the content hash of the executable graph."""

        return hashlib.sha256(self.graph.to_text().encode("utf-8")).hexdigest()

    @property
    def populations(self) -> Mapping[str, Population]:
        """Return immutable population views keyed by authoring name."""

        return MappingProxyType(
            {
                item.name: Population(item.name, item.node_ids, self._owner)
                for item in self.population_records
            }
        )

    def population(self, name: str) -> Population:
        """Return one named population or raise a resolution error."""

        try:
            return self.populations[name]
        except KeyError as exc:
            raise ResolutionError(f"unknown population '{name}'") from exc

    @property
    def reservoirs(self) -> Mapping[str, Reservoir]:
        """Return immutable reservoir views keyed by authoring name."""

        return MappingProxyType(
            {
                item.name: Reservoir(
                    item.name,
                    Population(item.name, item.node_ids, self._owner),
                    item.recurrent_edge_ids,
                )
                for item in self.reservoir_records
            }
        )

    def reservoir(self, name: str) -> Reservoir:
        """Return one named reservoir or raise a resolution error."""

        try:
            return self.reservoirs[name]
        except KeyError as exc:
            raise ResolutionError(f"unknown reservoir '{name}'") from exc

    def neuron(self, value: str | int) -> Neuron:
        """Resolve a neuron label or identifier into a stable reference."""

        if isinstance(value, str):
            try:
                node = self.neuron_labels[value]
            except KeyError as exc:
                raise ResolutionError(f"unknown neuron label '{value}'") from exc
            return Neuron(value, node, self._owner)
        if not isinstance(value, int) or isinstance(value, bool) or value not in {
            item.id for item in self.graph.nodes
        }:
            raise ResolutionError(f"unknown neuron id {value!r}")
        return Neuron(str(value), value, self._owner)

    def node_ids(self, target: object | None = None) -> tuple[int, ...]:
        """Normalize a high-level selection into authored node identifiers."""

        known = tuple(node.id for node in self.graph.nodes)
        if target is None:
            return known
        if isinstance(target, str):
            if target in self.populations:
                return self.population(target).node_ids
            return (self.neuron(target).id,)
        if isinstance(target, int) and not isinstance(target, bool):
            if target not in known:
                raise ResolutionError(f"unknown neuron id {target}")
            return (target,)
        if isinstance(target, Neuron):
            if target._owner is not self._owner or target.id not in known:
                raise ResolutionError("neuron reference belongs to another network")
            return (target.id,)
        if isinstance(target, Reservoir):
            if target.neurons._owner is not self._owner:
                raise ResolutionError("reservoir reference belongs to another network")
            return target.node_ids
        if isinstance(target, NodeSelection):
            if target._owner is not self._owner or any(node not in known for node in target.node_ids):
                raise ResolutionError("node selection belongs to another network")
            return target.node_ids
        if _sequence(target):
            result: list[int] = []
            for item in target:
                result.extend(self.node_ids(item))
            return tuple(result)
        raise ResolutionError(f"unsupported node selection {target!r}")

    def validate(self) -> NetworkReport:
        """Resolve the graph and summarize its selected execution capabilities."""

        resolved = self.graph.resolve()
        graph_nodes = {node.id: node for node in self.graph.nodes}
        capabilities = []
        dispatch_counts: dict[str, int] = {}
        tier_counts: dict[str, int] = {}
        total_states = 0
        for node_id, model in zip(resolved.node_ids, resolved.models):
            dispatch = model.dispatch.value
            tier = getattr(getattr(model, "tier", None), "value", "DELTA")
            count = 1 if isinstance(model, ResolvedScalarLIF) else len(model.state_names)
            total_states += count
            dispatch_counts[dispatch] = dispatch_counts.get(dispatch, 0) + 1
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            capabilities.append(
                NodeCapability(
                    node=node_id,
                    model=graph_nodes[node_id].model,
                    polarity=graph_nodes[node_id].polarity,
                    dispatch=dispatch,
                    state_count=count,
                    synapse_tier=tier,
                )
            )
        warnings = []
        stepped = dispatch_counts.get("STEPPED", 0)
        if stepped:
            warnings.append(
                f"{stepped} node(s) require adaptive numerical stepping"
            )
        shared_groups = {
            edge.weight_group
            for edge in self.graph.edges
            if edge.weight_group is not None
        }
        independent_weights = sum(
            edge.weight_group is None for edge in self.graph.edges
        )
        return NetworkReport(
            semantic_sha256=self.semantic_sha256,
            node_count=len(self.graph.nodes),
            edge_count=len(self.graph.edges),
            weight_parameter_count=independent_weights + len(shared_groups),
            shared_weight_group_count=len(shared_groups),
            state_count=total_states,
            dispatch_counts=MappingProxyType(dict(sorted(dispatch_counts.items()))),
            synapse_tier_counts=MappingProxyType(dict(sorted(tier_counts.items()))),
            nodes=tuple(capabilities),
            warnings=tuple(warnings),
        )

    def to_text(self) -> str:
        """Serialize executable graph data and authoring metadata."""

        graph_document = json.loads(self.graph.to_text())
        authoring = {
            "name": self.name,
            "populations": [
                {"name": item.name, "nodes": list(item.node_ids)}
                for item in self.population_records
            ],
            "reservoirs": [
                {
                    "name": item.name,
                    "nodes": list(item.node_ids),
                    "recurrent_edges": list(item.recurrent_edge_ids),
                }
                for item in self.reservoir_records
            ],
            "neuron_labels": dict(sorted(self.neuron_labels.items())),
            "metadata": dict(self.metadata),
        }
        authoring_bytes = _canonical_json(authoring)
        document = {
            "network_schema": NETWORK_SCHEMA_VERSION,
            "graph_sha256": self.semantic_sha256,
            "authoring_sha256": hashlib.sha256(authoring_bytes).hexdigest(),
            "graph": graph_document,
            "authoring": authoring,
        }
        return json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        ) + "\n"

    @classmethod
    def from_text(cls, text: str) -> "Network":
        """Validate and load a network or compatible bare graph document."""

        if not isinstance(text, str):
            raise ResolutionError("network document must be text")
        try:
            document = json.loads(text)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ResolutionError("invalid network JSON document") from exc
        if not isinstance(document, dict):
            raise ResolutionError("network JSON document must be an object")
        if "network_schema" not in document:
            graph = Graph.from_text(text)
            return cls(graph=graph)
        if document.get("network_schema") != NETWORK_SCHEMA_VERSION:
            raise ResolutionError(
                f"unsupported network schema; expected {NETWORK_SCHEMA_VERSION}"
            )
        try:
            graph_document = document["graph"]
            authoring = document["authoring"]
            graph_sha256 = document["graph_sha256"]
            authoring_sha256 = document["authoring_sha256"]
            if not isinstance(graph_document, dict) or not isinstance(authoring, dict):
                raise TypeError
            graph_text = json.dumps(
                graph_document,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            ) + "\n"
            if hashlib.sha256(graph_text.encode("utf-8")).hexdigest() != graph_sha256:
                raise ResolutionError("network graph hash mismatch")
            if hashlib.sha256(_canonical_json(authoring)).hexdigest() != authoring_sha256:
                raise ResolutionError("network authoring metadata hash mismatch")
            graph = Graph.from_text(graph_text)
            populations = tuple(
                PopulationRecord(str(item["name"]), tuple(item["nodes"]))
                for item in authoring.get("populations", [])
            )
            reservoirs = tuple(
                ReservoirRecord(
                    str(item["name"]),
                    tuple(item["nodes"]),
                    tuple(item["recurrent_edges"]),
                )
                for item in authoring.get("reservoirs", [])
            )
            labels = authoring.get("neuron_labels", {})
            metadata = authoring.get("metadata", {})
            if not isinstance(labels, dict) or not isinstance(metadata, dict):
                raise TypeError
            return cls(
                graph=graph,
                name=str(authoring.get("name", "network")),
                population_records=populations,
                reservoir_records=reservoirs,
                neuron_labels={str(key): int(value) for key, value in labels.items()},
                metadata=metadata,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ResolutionError):
                raise
            raise ResolutionError("invalid network JSON document") from exc

    def save(self, path: str | os.PathLike[str]) -> None:
        """Persist the network with an atomic file replacement."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(self.to_text())
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, destination)
        except Exception:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "Network":
        """Load a network document from disk."""

        return cls.from_text(Path(path).read_text(encoding="utf-8"))


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ResolutionError("network metadata must be finite JSON data") from exc


class NetworkBuilder:
    """Mutable construction surface that freezes into one immutable Network."""

    def __init__(
        self,
        name: str = "network",
        *,
        time_unit: str = "ms",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ResolutionError("network name must be nonempty")
        self.name = name
        self.time_unit = time_unit
        self.metadata = dict(metadata or {})
        _canonical_json(self.metadata)
        self._owner = object()
        self._models: dict[str, object] = {}
        self._synapses: dict[str, object] = {}
        self._nodes: list[GraphNode] = []
        self._node_models: dict[int, StandardNeuron] = {}
        self._edges: list[GraphEdge] = []
        self._input_ports: list[InputPort] = []
        self._output_ports: list[OutputPort] = []
        self._modulator_ports: list[ModulatorPort] = []
        self._populations: list[PopulationRecord] = []
        self._reservoirs: list[ReservoirRecord] = []
        self._labels: dict[str, int] = {}
        self._names: set[str] = set()
        self._port_names: set[str] = set()
        self._next_weight_group = 0
        self._sealed = False

    def _open(self) -> None:
        if self._sealed:
            raise RuntimeError("network builder is sealed after build()")

    def _register_model(self, model: StandardNeuron) -> None:
        if not isinstance(model, StandardNeuron):
            raise ResolutionError("model must be a StandardNeuron")
        existing = self._models.get(model.name)
        graph_model = model.model
        if existing is not None and existing != graph_model:
            raise ResolutionError(
                f"model name '{model.name}' is already bound to a different definition"
            )
        self._models[model.name] = graph_model

    def _parameter_rows(
        self,
        size: int,
        parameters: Mapping[str, object] | None,
    ) -> tuple[dict[str, float], ...]:
        rows = [dict() for _ in range(size)]
        for name, value in (parameters or {}).items():
            values = _expanded_values(value, size, f"parameter '{name}'")
            for index, item in enumerate(values):
                rows[index][name] = item
        return tuple(rows)

    def _initial_rows(
        self,
        model: StandardNeuron,
        size: int,
        initial: object | None,
    ) -> tuple[object | None, ...]:
        if initial is None or not _sequence(initial):
            return (initial,) * size
        state_count = len(model.state_names)
        if state_count > 1 and len(initial) == state_count and all(
            not _sequence(item) for item in initial
        ):
            vector = tuple(initial)
            return (vector,) * size
        if len(initial) != size:
            raise ResolutionError(f"initial state sequence must contain {size} values")
        return tuple(initial[index] for index in range(size))

    def population(
        self,
        name: str,
        size: int,
        model: StandardNeuron,
        *,
        parameters: Mapping[str, object] | None = None,
        initial: object | None = None,
        polarity: object = NeuronPolarity.EXCITATORY,
    ) -> Population:
        """Create a named population with per-neuron parameter expansion."""

        self._open()
        if not isinstance(name, str) or not name or name in self._names:
            raise ResolutionError(f"population name '{name}' is empty or already used")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ResolutionError("population size must be a positive integer")
        self._register_model(model)
        rows = self._parameter_rows(size, parameters)
        initials = self._initial_rows(model, size, initial)
        polarities = _expanded_polarities(polarity, size)
        ids = []
        for index in range(size):
            node_id = len(self._nodes)
            node = model.node(
                node_id,
                initial=initials[index],
                bindings=rows[index],
                polarity=polarities[index],
            )
            self._nodes.append(node)
            self._node_models[node_id] = model
            label = f"{name}[{index}]"
            self._labels[label] = node_id
            ids.append(node_id)
        record = PopulationRecord(name, tuple(ids))
        self._populations.append(record)
        self._names.add(name)
        return Population(name, record.node_ids, self._owner)

    def neuron(
        self,
        name: str,
        model: StandardNeuron,
        *,
        initial: object | None = None,
        polarity: NeuronPolarity | str = NeuronPolarity.EXCITATORY,
        **parameters: float,
    ) -> Neuron:
        """Create one named neuron and return its stable reference."""

        self._open()
        if not isinstance(name, str) or not name or name in self._names:
            raise ResolutionError(f"neuron name '{name}' is empty or already used")
        self._register_model(model)
        node_id = len(self._nodes)
        node = model.node(
            node_id,
            initial=initial,
            bindings=parameters,
            polarity=_polarity(polarity, "polarity"),
        )
        self._nodes.append(node)
        self._node_models[node_id] = model
        self._labels[name] = node_id
        self._names.add(name)
        return Neuron(name, node_id, self._owner)

    def reservoir(
        self,
        name: str,
        size: int,
        model: StandardNeuron,
        *,
        connectivity: ConnectionPattern | str | None = None,
        synapse: StandardSynapse | None = None,
        plasticity: PlasticityRule | None = None,
        weight: object = 1.0,
        delay: object = 1.0,
        parameters: Mapping[str, object] | None = None,
        initial: object | None = None,
        seed: int = 0,
        polarity: object | None = None,
        inhibitory_fraction: float = 0.2,
    ) -> Reservoir:
        """Construct one population plus its explicit recurrent projection.

        The default is a deterministic 10% random graph without self-connections.
        Passing a pattern such as ``LocallyConnected`` or ``FixedOutDegree`` makes
        the reservoir topology explicit without changing the runtime representation.
        """

        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ResolutionError("reservoir size must be a positive integer")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ResolutionError("reservoir seed must be an integer")
        fraction = _finite(inhibitory_fraction, "inhibitory fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ResolutionError("inhibitory fraction must lie in [0, 1]")
        if polarity is None:
            inhibitory_count = int(math.floor(size * fraction + 0.5))
            generator = random.Random(seed ^ 0xDA1E)
            inhibitory = frozenset(generator.sample(range(size), inhibitory_count))
            polarity = tuple(
                NeuronPolarity.INHIBITORY
                if index in inhibitory
                else NeuronPolarity.EXCITATORY
                for index in range(size)
            )
        if connectivity is None:
            connectivity = FixedProbability(0.1, seed=seed, exclude_self=True)
        neurons = self.population(
            name,
            size,
            model,
            parameters=parameters,
            initial=initial,
            polarity=polarity,
        )
        edges = self.connect(
            neurons,
            neurons,
            synapse=synapse,
            plasticity=plasticity,
            pattern=connectivity,
            weight=weight,
            delay=delay,
        )
        record = ReservoirRecord(name, neurons.node_ids, edges)
        self._reservoirs.append(record)
        return Reservoir(name, neurons, edges)

    def _target_ids(self, target: object) -> tuple[int, ...]:
        if isinstance(target, int) and not isinstance(target, bool):
            ids = (target,)
        elif isinstance(target, Neuron):
            if target._owner is not self._owner:
                raise ResolutionError("neuron reference belongs to another builder")
            ids = (target.id,)
        elif isinstance(target, Reservoir):
            if target.neurons._owner is not self._owner:
                raise ResolutionError("reservoir reference belongs to another builder")
            ids = target.node_ids
        elif isinstance(target, NodeSelection):
            if target._owner is not self._owner:
                raise ResolutionError("node selection belongs to another builder")
            ids = target.node_ids
        else:
            raise ResolutionError("target must be a neuron or node selection")
        if any(node not in self._node_models for node in ids):
            raise ResolutionError("target references an unknown neuron")
        return ids

    def _pattern(self, pattern: ConnectionPattern | str | None) -> ConnectionPattern:
        if pattern is None or pattern == "all_to_all":
            return AllToAll()
        if pattern == "one_to_one":
            return OneToOne()
        if not isinstance(pattern, ConnectionPattern):
            raise ResolutionError("unsupported connection pattern")
        return pattern

    def connect(
        self,
        pre: object,
        post: object,
        *,
        synapse: StandardSynapse | None = None,
        pattern: ConnectionPattern | str | None = None,
        weight: object = 1.0,
        delay: object = 0.0,
        receptor: str | None = None,
        plasticity: PlasticityRule | None = None,
    ) -> tuple[int, ...]:
        """Expand a connection pattern into canonical graph edges."""

        self._open()
        pre_ids = self._target_ids(pre)
        post_ids = self._target_ids(post)
        selected_pattern = self._pattern(pattern)
        pairs = selected_pattern.pairs(pre_ids, post_ids)
        local_groups = selected_pattern.weight_group_indices(pre_ids, post_ids)
        if local_groups is not None:
            # Expand one trainable kernel value across all spatial copies.
            if len(local_groups) != len(pairs):
                raise ResolutionError("connection pattern returned a misaligned weight-group layout")
            group_count = max(local_groups, default=-1) + 1
            kernel_weights = _expanded_values(weight, group_count, "kernel weight")
            weights = tuple(kernel_weights[group] for group in local_groups)
            group_base = self._next_weight_group
            weight_groups = tuple(group_base + group for group in local_groups)
            self._next_weight_group += group_count
        else:
            weights = _expanded_values(weight, len(pairs), "weight")
            weight_groups = (None,) * len(pairs)
        delays = _expanded_values(delay, len(pairs), "delay")
        if any(
            value < 0.0
            and self._nodes[source].polarity is not NeuronPolarity.MIXED
            for (source, _), value in zip(pairs, weights)
        ):
            raise ResolutionError(
                "connection weights must be nonnegative magnitudes; "
                "signed weights require a MIXED presynaptic neuron"
            )
        if plasticity is not None and any(
            self._nodes[source].polarity is NeuronPolarity.MIXED
            for source, _ in pairs
        ):
            raise ResolutionError(
                "online plasticity is not supported for "
                "MIXED presynaptic neurons"
            )
        if any(value < 0.0 for value in delays):
            raise ResolutionError("connection delays must be nonnegative")
        selected_synapse = synapse or Delta()
        if not isinstance(selected_synapse, StandardSynapse):
            raise ResolutionError("synapse must be a StandardSynapse")
        if plasticity is not None and not isinstance(
            plasticity,
            (
                PairSTDP,
                TripletSTDP,
                ModulatedSTDP,
                VoltageModulatedSTDP,
                SoftExcursionModulated,
            ),
        ):
            raise ResolutionError(
                "plasticity must be PairSTDP, TripletSTDP, ModulatedSTDP, "
                "VoltageModulatedSTDP, or SoftExcursionModulated"
            )
        graph_synapse = selected_synapse.model
        if graph_synapse is not None:
            existing = self._synapses.get(graph_synapse.id)
            if existing is not None and existing != graph_synapse:
                raise ResolutionError(
                    f"synapse name '{graph_synapse.id}' is already bound to a different definition"
                )
            self._synapses[graph_synapse.id] = graph_synapse
        edge_ids = []
        for index, ((source, target), edge_weight, edge_delay, weight_group) in enumerate(
            zip(pairs, weights, delays, weight_groups)
        ):
            mapped_receptor = receptor
            if graph_synapse is not None and mapped_receptor is None:
                mapped_receptor = getattr(
                    self._node_models[target], "receptor_name", None
                )
            if graph_synapse is not None and mapped_receptor is None:
                raise ResolutionError(
                    f"postsynaptic node {target} has no current receptor; "
                    "use LIF(synaptic_input=True) or provide a compatible model"
                )
            edge_id = len(self._edges)
            edge = GraphEdge(
                id=edge_id,
                pre=source,
                post=target,
                weight=edge_weight,
                delay=edge_delay,
                synapse=None if graph_synapse is None else graph_synapse.id,
                receptor=mapped_receptor,
                output=selected_synapse.output_name,
                initial=selected_synapse.initial,
                plasticity=plasticity,
                weight_group=weight_group,
            )
            self._edges.append(edge)
            edge_ids.append(edge_id)
        return tuple(edge_ids)

    def modulator(self, name: str, *, targets: object) -> str:
        """Create a named third-factor input for selected modulated edges."""

        self._open()
        if not isinstance(name, str) or not name or name in self._port_names:
            raise ResolutionError(f"port name '{name}' is empty or already used")
        if isinstance(targets, Reservoir):
            edge_ids = targets.recurrent_edge_ids
        elif isinstance(targets, int) and not isinstance(targets, bool):
            edge_ids = (targets,)
        elif _sequence(targets):
            edge_ids = tuple(targets)
        else:
            raise ResolutionError("modulator targets must be edge identifiers or a projection")
        if not edge_ids or any(
            not isinstance(edge, int)
            or isinstance(edge, bool)
            or not 0 <= edge < len(self._edges)
            for edge in edge_ids
        ):
            raise ResolutionError("modulator targets contain an unknown edge")
        if len(edge_ids) != len(set(edge_ids)):
            raise ResolutionError("modulator targets repeat an edge")
        if any(
            not isinstance(
                self._edges[edge].plasticity,
                (
                    ModulatedSTDP,
                    VoltageModulatedSTDP,
                    SoftExcursionModulated,
                ),
            )
            for edge in edge_ids
        ):
            raise ResolutionError("modulator targets must all use modulated plasticity")
        already_targeted = {
            edge for port in self._modulator_ports for edge in port.edges
        }
        overlap = already_targeted.intersection(edge_ids)
        if overlap:
            raise ResolutionError(
                "modulated edge is already assigned to another port: "
                + ", ".join(str(edge) for edge in sorted(overlap))
            )
        self._modulator_ports.append(ModulatorPort(name, tuple(edge_ids)))
        self._port_names.add(name)
        return name

    def _broadcast(self, value: object, count: int, context: str) -> tuple[object, ...]:
        if _sequence(value):
            if len(value) != count:
                raise ResolutionError(f"{context} must contain {count} entries")
            return tuple(value[index] for index in range(count))
        return (value,) * count

    def inputs(
        self,
        name: str,
        target: object,
        *,
        encoder: Encoder | Sequence[Encoder] = NativeEventEncoder(),
        parameter: str | Sequence[str | None] | None = None,
        mode: InputMode | None = None,
    ) -> PortSelection:
        """Create one input port per selected neuron."""

        self._open()
        nodes = self._target_ids(target)
        encoders = self._broadcast(encoder, len(nodes), "encoders")
        parameters = self._broadcast(parameter, len(nodes), "input parameters")
        ids = []
        for index, (node, item_encoder, item_parameter) in enumerate(
            zip(nodes, encoders, parameters)
        ):
            port_id = name if len(nodes) == 1 else f"{name}[{index}]"
            if port_id in self._port_names:
                raise ResolutionError(f"port name '{port_id}' is already used")
            inferred = mode
            if inferred is None:
                inferred = (
                    InputMode.DRIVE
                    if item_parameter is not None or isinstance(item_encoder, HeldCurrentEncoder)
                    else InputMode.SPIKE
                )
            self._input_ports.append(
                InputPort(
                    port_id,
                    node,
                    inferred,
                    item_parameter,
                    item_encoder,
                )
            )
            self._port_names.add(port_id)
            ids.append(port_id)
        return PortSelection(name, tuple(ids))

    def input(self, name: str, target: object, **kwargs) -> str:
        """Create one input port for a single target neuron."""

        ports = self.inputs(name, target, **kwargs)
        if len(ports) != 1:
            raise ResolutionError("input() requires exactly one target neuron")
        return ports[0]

    def outputs(
        self,
        name: str,
        target: object,
        *,
        decoder: Decoder | Sequence[Decoder | None] | None = None,
    ) -> PortSelection:
        """Create one output port per selected neuron."""

        self._open()
        nodes = self._target_ids(target)
        decoders = self._broadcast(decoder, len(nodes), "decoders")
        ids = []
        for index, (node, item_decoder) in enumerate(zip(nodes, decoders)):
            port_id = name if len(nodes) == 1 else f"{name}[{index}]"
            if port_id in self._port_names:
                raise ResolutionError(f"port name '{port_id}' is already used")
            self._output_ports.append(OutputPort(port_id, node, item_decoder))
            self._port_names.add(port_id)
            ids.append(port_id)
        return PortSelection(name, tuple(ids))

    def output(self, name: str, target: object, **kwargs) -> str:
        """Create one output port for a single target neuron."""

        ports = self.outputs(name, target, **kwargs)
        if len(ports) != 1:
            raise ResolutionError("output() requires exactly one target neuron")
        return ports[0]

    def build(self) -> Network:
        """Freeze the builder into an immutable validated network."""

        self._open()
        if not self._nodes:
            raise ResolutionError("network must contain at least one neuron")
        incoming_filtered = {
            edge.post for edge in self._edges if edge.synapse is not None
        }
        lowered_nodes = list(self._nodes)
        plain_variants: dict[str, LIF] = {}
        # Remove unused current receptors so delta-only nodes take the compact path.
        for index, node in enumerate(lowered_nodes):
            family = self._node_models[node.id]
            if (
                isinstance(family, LIF)
                and family.synaptic_input
                and node.id not in incoming_filtered
            ):
                plain = plain_variants.get(family.name)
                if plain is None:
                    candidate = f"{family.name}__delta_only"
                    suffix = 2
                    while candidate in self._models:
                        candidate = f"{family.name}__delta_only_{suffix}"
                        suffix += 1
                    plain = replace(
                        family,
                        name=candidate,
                        synaptic_input=False,
                    )
                    self._register_model(plain)
                    plain_variants[family.name] = plain
                lowered_nodes[index] = replace(node, model=plain.name)
        graph = Graph(
            models=tuple(self._models[name] for name in sorted(self._models)),
            nodes=tuple(lowered_nodes),
            edges=tuple(self._edges),
            input_ports=tuple(self._input_ports),
            output_ports=tuple(self._output_ports),
            time_unit=self.time_unit,
            synapses=tuple(self._synapses[name] for name in sorted(self._synapses)),
            modulator_ports=tuple(self._modulator_ports),
        )
        graph.resolve()
        self._sealed = True
        return Network(
            graph=graph,
            name=self.name,
            population_records=tuple(self._populations),
            reservoir_records=tuple(self._reservoirs),
            neuron_labels=dict(self._labels),
            metadata=dict(self.metadata),
            _owner=self._owner,
        )


__all__ = [
    "NETWORK_SCHEMA_VERSION",
    "Neuron",
    "NodeSelection",
    "Population",
    "Reservoir",
    "PortSelection",
    "ConnectionPattern",
    "Convolution2D",
    "AllToAll",
    "OneToOne",
    "LocallyConnected",
    "FixedOutDegree",
    "FixedInDegree",
    "FixedProbability",
    "ExplicitConnections",
    "ValueDistribution",
    "Uniform",
    "Normal",
    "NodeCapability",
    "ReservoirRecord",
    "NetworkReport",
    "Network",
    "NetworkBuilder",
]
