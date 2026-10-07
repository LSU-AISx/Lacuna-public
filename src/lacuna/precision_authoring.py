"""Detached target-representation inputs for precision preflight.

This stage rounds authored constants before symbolic resolution. It does not
evaluate neuronal trajectories or create an executable reduced-precision graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from decimal import Decimal, InvalidOperation
from io import StringIO
import re
import tokenize

from .codec import (
    BurstEncoder,
    HeldCurrentEncoder,
    LatencyBurstEncoder,
    NativeEventEncoder,
    PoissonRateEncoder,
    RateDecoder,
    RegularRateEncoder,
    TemporalWeightDecoder,
    TTFSEncoder,
    TTFSDecoder,
)
from .dsl import _definition_blocks, _statements, parse_neuron, parse_synapse
from .errors import PrecisionResolutionError, ResolutionError
from .graph import (
    Graph,
    GraphEdge,
    GraphModel,
    GraphNode,
    GraphSynapse,
    InputPort,
    ModulatorPort,
    OutputPort,
)
from .ir import NeuronModel, SynapseModel
from .plasticity import (
    ModulatedSTDP,
    PairSTDP,
    SoftExcursionModulated,
    TripletSTDP,
    VoltageModulatedSTDP,
)
from .precision import PrecisionProfile, normalize_precision


RecordValue = Callable[[float, str, bool], float]
_DECIMAL = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_RULE_TYPES = (
    PairSTDP,
    TripletSTDP,
    ModulatedSTDP,
    VoltageModulatedSTDP,
    SoftExcursionModulated,
)
_RULE_FLAGS = frozenset((
    "consume_on_modulation", "adaptive_baseline", "use_upward_excursion",
    "use_proximity",
))
_CODEC_FIELDS = {
    NativeEventEncoder: ((), ()),
    RegularRateEncoder: (("min_rate", "max_rate", "amplitude"), ()),
    PoissonRateEncoder: (("min_rate", "max_rate", "amplitude"), ()),
    TTFSEncoder: (
        ("amplitude", "silence_threshold"), ("min_latency", "max_latency"),
    ),
    BurstEncoder: (("min_rate", "max_rate", "amplitude"), ("duration",)),
    LatencyBurstEncoder: (
        ("rate", "amplitude", "silence_threshold"),
        ("min_latency", "max_latency", "duration"),
    ),
    HeldCurrentEncoder: (("gain", "offset", "baseline"), ()),
    RateDecoder: ((), ("width", "origin")),
    TTFSDecoder: ((), ()),
    TemporalWeightDecoder: (("tau",), ()),
}


def _known_type(value: object, expected: type, path: str) -> None:
    if type(value) is not expected:
        raise PrecisionResolutionError(
            f"{path} has unsupported precision-preflight type "
            f"{type(value).__name__}"
        )


def _source_literal_integrity(source: str, kind: str, path: str) -> None:
    """Reject source values whose nonzero magnitude was lost by the parser.

    This host-loss check includes unused defaults because their original value
    cannot be recovered from parsed IR. Target-range checks still apply only to
    effective defaults and bindings, so an unused large default remains allowed.
    """
    _, blocks = _definition_blocks(
        source,
        kind,
        frozenset({"params"}),
        frozenset({
            "state", "dynamics", "reset", "threshold", "hazard", "refractory",
            "reactive", "on_spike", "output",
        }),
    )
    literals = []
    for statement in _statements(blocks["params"]):
        declaration, literal = statement.split("=", 1)
        name = declaration.split(":", 1)[0].strip()
        literals.append((literal.strip(), f"{path}.defaults.{name}"))
    if "refractory" in blocks:
        literal = blocks["refractory"].split("=", 1)[-1].strip()
        literals.append((literal, f"{path}.refractory"))
    for literal, label in literals:
        unsigned = literal.lstrip("+-")
        if unsigned.lower().startswith("0x"):
            significand = unsigned[2:].lower().split("p", 1)[0]
            nonzero = any(character not in "0." for character in significand)
            host_value = float.fromhex(literal)
        else:
            significand = unsigned.lower().split("e", 1)[0]
            nonzero = any(character not in "0." for character in significand)
            host_value = float(literal)
        if nonzero and host_value == 0.0:
            raise PrecisionResolutionError(
                f"{label} underflows to zero in the source parser"
            )


def _expression(text: str, path: str, record: RecordValue) -> str:
    """Replace numeric tokens without changing identifiers or integer powers."""
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    changes = []
    try:
        tokens = tokenize.generate_tokens(StringIO(text).readline)
        for index, token in enumerate(tokens):
            if token.type != tokenize.NUMBER:
                continue
            if not _DECIMAL.fullmatch(token.string):
                raise PrecisionResolutionError(
                    f"{path} contains unsupported numeric literal {token.string!r}"
                )
            label = f"{path}.literal[{index}]"
            try:
                exact = Decimal(token.string)
                value = float(token.string)
            except (InvalidOperation, ValueError, OverflowError) as exc:
                raise PrecisionResolutionError(
                    f"{label} cannot be represented by the host parser"
                ) from exc
            if not exact.is_zero() and value == 0.0:
                raise PrecisionResolutionError(
                    f"{label} underflows to zero before target rounding"
                )
            target = record(value, label, False)
            # Integer tokens can define polynomial structure, so never round them.
            unchanged = target == value
            if token.string.isdigit():
                if not target.is_integer() or Decimal(target) != exact:
                    raise PrecisionResolutionError(
                        f"{label} integer token is not exactly representable "
                        "in the target precision"
                    )
                unchanged = True
            if not unchanged:
                start = offsets[token.start[0] - 1] + token.start[1]
                end = offsets[token.end[0] - 1] + token.end[1]
                changes.append((start, end, repr(target)))
    except (tokenize.TokenError, IndentationError) as exc:
        raise PrecisionResolutionError(f"{path} has invalid expression syntax") from exc
    for start, end, replacement in reversed(changes):
        text = text[:start] + replacement + text[end:]
    return text


def _model(model, path: str, used_defaults: set[str], record: RecordValue):
    parameters = tuple(
        replace(
            item,
            default=record(item.default, f"{path}.defaults.{item.name}", False),
        ) if item.name in used_defaults else replace(item)
        for item in model.parameters
    )
    dynamics = {
        name: _expression(value, f"{path}.dynamics.{name}", record)
        for name, value in model.dynamics.items()
    }
    if type(model) is SynapseModel:
        return replace(
            model,
            parameters=parameters,
            dynamics=dynamics,
            spike_update=_expression(model.spike_update, f"{path}.on_spike", record),
            outputs={
                name: _expression(value, f"{path}.outputs.{name}", record)
                for name, value in model.outputs.items()
            },
        )
    return replace(
        model,
        parameters=parameters,
        states=tuple(replace(state) for state in model.states),
        dynamics=dynamics,
        threshold=None if model.threshold is None else replace(
            model.threshold,
            level=_expression(model.threshold.level, f"{path}.threshold", record),
        ),
        reset={
            name: _expression(value, f"{path}.reset.{name}", record)
            for name, value in model.reset.items()
        },
        hazard=None if model.hazard is None else replace(
            model.hazard,
            rate=_expression(model.hazard.rate, f"{path}.hazard", record),
        ),
        refractory=None if model.refractory is None else replace(
            model.refractory,
            duration=record(model.refractory.duration, f"{path}.refractory", True),
        ),
    )


def _bindings(model, supplied, path, defaults, record):
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, Mapping):
        raise PrecisionResolutionError(f"{path} must be a parameter mapping")
    names = {item.name for item in model.parameters}
    if set(supplied) - names:
        raise PrecisionResolutionError(f"{path} contains unknown parameters")
    values = {}
    for item in model.parameters:
        if item.name not in supplied:
            defaults.add(item.name)
        values[item.name] = record(
            supplied.get(item.name, item.default), f"{path}.{item.name}", False,
        )
    return values


def _initial(value, path, record):
    if isinstance(value, (tuple, list)):
        return tuple(
            record(item, f"{path}[{index}]", False)
            for index, item in enumerate(value)
        )
    return record(value, path, False)


def _plasticity(rule, path, record):
    if rule is None:
        return None
    if type(rule) not in _RULE_TYPES:
        raise PrecisionResolutionError(
            f"{path} has unsupported plasticity type {type(rule).__name__}"
        )
    values = {}
    for field in fields(rule):
        value = getattr(rule, field.name)
        label = f"{path}.{field.name}"
        if field.name == "bounds":
            values[field.name] = tuple(
                record(item, f"{label}[{index}]", False)
                for index, item in enumerate(value)
            )
        elif field.name in _RULE_FLAGS:
            if type(value) is not bool:
                raise PrecisionResolutionError(f"{label} must be boolean")
            values[field.name] = value
        else:
            values[field.name] = record(value, label, False)
    try:
        return replace(rule, **values)
    except (ResolutionError, TypeError, ValueError) as exc:
        raise PrecisionResolutionError(
            f"{path} is invalid after target rounding: {exc}"
        ) from exc


def _codec(codec, path, record):
    if codec is None:
        return None
    role_fields = _CODEC_FIELDS.get(type(codec))
    if role_fields is None:
        raise PrecisionResolutionError(
            f"{path} has unsupported codec type {type(codec).__name__}"
        )
    values = {}
    for is_time, names in enumerate(role_fields):
        for name in names:
            value = getattr(codec, name)
            if value is not None:
                values[name] = record(value, f"{path}.{name}", bool(is_time))
    return replace(codec, **values)


def prepare_precision_graph(
    graph: Graph,
    profile: PrecisionProfile,
    *,
    record: RecordValue,
) -> tuple[Graph, dict[str, NeuronModel], dict[str, SynapseModel]]:
    """Prepare target inputs without mutating the authored graph or its models.

    ``record`` rounds a value in the selected representation and records its path.
    A true final argument selects timestamps, while false selects model scalars.
    Defaults overridden by every instance are never consumed or target-validated.
    The returned parsed models must accompany the graph during private preflight
    resolution because its source strings remain the original authoring text.
    """
    _known_type(graph, Graph, "graph")
    normalize_precision(profile)
    parsed = {}
    synapses = {}
    for entries, expected, parser, destination, label in (
        (graph.models, GraphModel, parse_neuron, parsed, "models"),
        (graph.synapses, GraphSynapse, parse_synapse, synapses, "synapses"),
    ):
        sources = {}
        for entry in entries:
            _known_type(entry, expected, label)
            if entry.id in destination:
                raise PrecisionResolutionError(f"duplicate {label} id {entry.id!r}")
            if entry.source not in sources:
                sources[entry.source] = parser(entry.source)
                _source_literal_integrity(
                    entry.source,
                    "neuron" if expected is GraphModel else "synapse",
                    f"{label}[{entry.id}]",
                )
            destination[entry.id] = sources[entry.source]
    neuron_defaults = {key: set() for key in parsed}
    synapse_defaults = {key: set() for key in synapses}

    def synapse_bindings(item, path):
        if item.synapse is None:
            if item.synapse_bindings:
                raise PrecisionResolutionError(f"{path} has bindings without a synapse")
            return None
        if item.synapse not in synapses:
            raise PrecisionResolutionError(f"{path} references an unknown synapse")
        return _bindings(
            synapses[item.synapse], item.synapse_bindings,
            f"{path}.synapse_bindings", synapse_defaults[item.synapse], record,
        )

    nodes = []
    for node in graph.nodes:
        _known_type(node, GraphNode, "nodes")
        path = f"nodes[{node.id}]"
        if node.model not in parsed:
            raise PrecisionResolutionError(f"{path} references an unknown model")
        nodes.append(replace(
            node,
            initial=_initial(node.initial, f"{path}.initial", record),
            bindings=_bindings(
                parsed[node.model], node.bindings, f"{path}.bindings",
                neuron_defaults[node.model], record,
            ),
            synapse_bindings=synapse_bindings(node, path),
        ))
    edges = []
    for edge in graph.edges:
        _known_type(edge, GraphEdge, "edges")
        path = f"edges[{edge.id}]"
        edges.append(replace(
            edge,
            initial=_initial(edge.initial, f"{path}.initial", record),
            weight=record(edge.weight, f"{path}.weight", False),
            delay=record(edge.delay, f"{path}.delay", True),
            synapse_bindings=synapse_bindings(edge, path),
            plasticity=_plasticity(edge.plasticity, f"{path}.plasticity", record),
        ))
    inputs = []
    for port in graph.input_ports:
        _known_type(port, InputPort, "input_ports")
        inputs.append(replace(
            port, encoder=_codec(port.encoder, f"inputs[{port.id}].encoder", record),
        ))
    outputs = []
    for port in graph.output_ports:
        _known_type(port, OutputPort, "output_ports")
        outputs.append(replace(
            port, decoder=_codec(port.decoder, f"outputs[{port.id}].decoder", record),
        ))
    for port in graph.modulator_ports:
        _known_type(port, ModulatorPort, "modulator_ports")
    prepared = replace(
        graph,
        models=tuple(replace(item) for item in graph.models),
        synapses=tuple(replace(item) for item in graph.synapses),
        nodes=tuple(nodes),
        edges=tuple(edges),
        input_ports=tuple(inputs),
        output_ports=tuple(outputs),
        modulator_ports=tuple(
            replace(item, edges=tuple(item.edges)) for item in graph.modulator_ports
        ),
    )
    return prepared, {
        key: _model(value, f"models[{key}]", neuron_defaults[key], record)
        for key, value in parsed.items()
    }, {
        key: _model(value, f"synapses[{key}]", synapse_defaults[key], record)
        for key, value in synapses.items()
    }
