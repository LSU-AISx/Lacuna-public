"""Non-executable target-representation checks before numerical certification."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from .errors import PrecisionResolutionError, ResolutionError
from .graph import Graph, _resolve_graph
from .ir import ResolvedReactiveIF
from .learning import resolve_learning
from .network import Network
from .precision import PrecisionProfile, normalize_precision, require_time_progress
from .precision_authoring import prepare_precision_graph
from .precision_checks import validate_precision_dag, validate_resolved_precision
from .resolution_cache import _cached_resolution


_PREFLIGHT_REVISION = 1
_OUTSTANDING_CHECKS = (
    "Native target expression binding and crossing validation",
    "State-dependent singularities and intermediate dynamic range",
    "Runtime input values, clock progress, and event ordering",
    "Selected native library and target-toolchain validation",
)


@dataclass(frozen=True)
class PrecisionValue:
    """One checked value, including its target storage role and rounding."""

    path: str
    role: str
    bits: int
    source_value: float
    target_value: float

    @property
    def changed(self) -> bool:
        """Compare retained binary values, including the sign of zero."""

        return self.source_value.hex() != self.target_value.hex()


@dataclass(frozen=True)
class PrecisionNodeAnalysis:
    """Host capability proposals, not certified target execution strategies."""

    node: int
    original_dispatch: str
    rounded_input_dispatch: str
    state_count: int


@dataclass(frozen=True)
class TargetPrecisionAnalysis:
    """Successful representation checks that cannot be compiled or executed."""

    precision: PrecisionProfile
    source_hash: str
    analysis_key: str
    values: tuple[PrecisionValue, ...]
    nodes: tuple[PrecisionNodeAnalysis, ...]
    time_horizon: float | None
    validation_revision: int = _PREFLIGHT_REVISION
    outstanding_checks: tuple[str, ...] = _OUTSTANDING_CHECKS
    executable: bool = field(default=False, init=False)

    @property
    def changed_values(self) -> tuple[PrecisionValue, ...]:
        """Return only values changed by the requested representation."""

        return tuple(value for value in self.values if value.changed)

    def to_document(self) -> dict[str, object]:
        """Export inspection metadata without exporting a runnable graph."""

        return {
            "schema": "lacuna-target-precision-analysis-v1",
            "precision": self.precision.to_record(),
            "source_hash": self.source_hash,
            "analysis_key": self.analysis_key,
            "validation_revision": self.validation_revision,
            "executable": False,
            "outstanding_checks": list(self.outstanding_checks),
            "time_horizon": (
                None if self.time_horizon is None else self.time_horizon.hex()
            ),
            "nodes": [
                {
                    "node": node.node,
                    "original_dispatch": node.original_dispatch,
                    "rounded_input_dispatch": node.rounded_input_dispatch,
                    "state_count": node.state_count,
                }
                for node in self.nodes
            ],
            "values": [
                {
                    "path": value.path,
                    "role": value.role,
                    "bits": value.bits,
                    "source": value.source_value.hex(),
                    "target": value.target_value.hex(),
                }
                for value in self.values
            ],
        }


def analyze_precision(
    network: Network | Graph,
    precision: PrecisionProfile | str,
    *,
    time_precision: str | None = None,
    time_horizon: float | None = None,
) -> TargetPrecisionAnalysis:
    """Inspect rounded authoring values and derived representation constraints.

    This does not train, quantize a compiled network, or approve target
    execution. The optional horizon conservatively checks fixed delays and
    refractory intervals at that clock value. Runtime checks remain necessary.
    """

    profile = normalize_precision(precision, time_precision)
    graph = network.graph if isinstance(network, Network) else network
    if not isinstance(graph, Graph):
        raise TypeError("precision analysis requires an authored Network or Graph")
    if time_horizon is not None:
        try:
            horizon = PrecisionProfile.FLOAT64.round_time(
                time_horizon, name="time horizon"
            )
        except (TypeError, ValueError) as exc:
            raise PrecisionResolutionError(str(exc)) from exc
        if horizon < 0.0:
            raise PrecisionResolutionError("time horizon must be nonnegative")
    else:
        horizon = None
    # The canonical source captures effective bindings without changing the model.
    return _analyze_document(
        graph.to_text(), profile.cache_key, horizon, _PREFLIGHT_REVISION
    )


@_cached_resolution
def _analyze_document(
    source: str,
    profile_key: tuple[str, int, int, int],
    horizon: float | None,
    revision: int,
) -> TargetPrecisionAnalysis:
    profile = PrecisionProfile(profile_key[0])
    values: list[PrecisionValue] = []

    def record(value: float, path: str, is_time: bool) -> float:
        try:
            convert = profile.round_time if is_time else profile.round_real
            rounded = convert(value, name=path)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PrecisionResolutionError(f"{profile.value}: {exc}") from exc
        values.append(PrecisionValue(
            path, "time" if is_time else "model",
            profile.time_bits if is_time else profile.real_bits,
            float(value), rounded,
        ))
        return rounded

    graph = Graph.from_text(source)
    original = graph.resolve()
    prepared, models, synapses = prepare_precision_graph(graph, profile, record=record)
    try:
        candidate = _resolve_graph(
            prepared, parsed_models=models, parsed_synapse_models=synapses
        )
    except ResolutionError as exc:
        raise PrecisionResolutionError(
            f"{profile.value}: rounded inputs cannot be resolved: {exc}"
        ) from exc
    nodes: list[PrecisionNodeAnalysis] = []
    for index, (node_id, model, initial) in enumerate(zip(
        candidate.node_ids, candidate.models, candidate.initial_values
    )):
        context = f"node {node_id} derived"
        if type(model) is not type(original.models[index]):
            raise PrecisionResolutionError(
                f"{profile.value}: {context} rounding changes the resolved model "
                "family and requires target-aware capability derivation"
            )
        validate_resolved_precision(model, profile, context=context, record=record)
        components = initial if isinstance(initial, tuple) else (initial,)
        for state, value in enumerate(components):
            record(value, f"{context} initial[{state}]", False)
        if (
            not isinstance(model, ResolvedReactiveIF)
            and getattr(model, "hazard", None) is None
            and profile.round_real(components[model.readout_index])
            >= profile.round_real(model.threshold)
        ):
            raise PrecisionResolutionError(
                f"{profile.value}: {context} initial value must remain below threshold"
            )
        nodes.append(PrecisionNodeAnalysis(
            node_id, original.models[index].dispatch.value, model.dispatch.value,
            len(components),
        ))
    for edge in candidate.edges:
        record(edge.weight, f"edge {edge.pre}->{edge.post} derived weight", False)
        record(edge.deposit_scale, f"edge {edge.pre}->{edge.post} derived scale", False)

    for edge in prepared.edges:
        if edge.plasticity is None:
            continue
        learning = resolve_learning(edge.plasticity)
        bindings = dict(zip(learning.program.parameter_names, learning.parameter_values))
        context = f"edge {edge.id} learning"
        for name, value in bindings.items():
            record(value, f"{context} parameter {name}", False)
        for event in learning.program.events:
            validate_precision_dag(
                event.expressions, bindings, profile,
                context=f"{context} {event.event.value}", record=record,
            )
        if learning.program.observer is not None:
            validate_precision_dag(
                learning.program.observer.expressions, bindings, profile,
                context=f"{context} observer", record=record,
            )

    target_horizon = None if horizon is None else record(horizon, "time horizon", True)
    if target_horizon is not None:
        intervals = [
            (f"edge {edge.id} delay", edge.delay) for edge in prepared.edges
        ]
        intervals += [
            (f"node {node_id} refractory", model.refractory)
            for node_id, model in zip(candidate.node_ids, candidate.models)
        ]
        for path, interval in intervals:
            if interval > 0.0:
                try:
                    require_time_progress(target_horizon, interval, profile)
                except ValueError as exc:
                    raise PrecisionResolutionError(
                        f"{profile.value}: {path} fails conservative horizon check: {exc}"
                    ) from exc

    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    identity = json.dumps({
        "source_hash": source_hash,
        "profile": profile_key,
        "validation_revision": revision,
        "time_horizon": None if horizon is None else horizon.hex(),
    }, sort_keys=True, separators=(",", ":"))
    return TargetPrecisionAnalysis(
        profile, source_hash, hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        tuple(values), tuple(nodes), target_horizon, revision,
    )
