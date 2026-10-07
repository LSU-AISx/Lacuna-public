"""Versioned cross-release behavioral fixtures for the Lacuna C evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .codec import BurstEncoder, RateDecoder, TTFSDecoder
from .ffi import CoreEvaluator, RecordingConfig
from .graph import (
    DriveInput,
    Graph,
    GraphModel,
    GraphNode,
    InputMode,
    InputPort,
    OutputPort,
    ScalarInput,
    SpikeInput,
)
from .validation import (
    VALIDATION_LIF,
    NetworkCase,
    adaptive_population_case,
    alpha_population_case,
    chain_case,
    drive_boundary_case,
    mixed_chain_case,
)

REFERENCE_FIXTURE_SCHEMA = 1
DEFAULT_ABSOLUTE_TOLERANCE = 1e-10


class ReferenceFixtureError(RuntimeError):
    """A checked-in behavioral reference does not match the live evaluator."""


@dataclass(frozen=True)
class ReferenceCase:
    """Expected scalar trajectory and spike behavior for one fixture."""

    name: str
    graph: Graph
    spike_inputs: tuple[SpikeInput, ...]
    drive_inputs: tuple[DriveInput, ...]
    scalar_inputs: tuple[ScalarInput, ...]
    t_end: float


def _from_network(case: NetworkCase) -> ReferenceCase:
    return ReferenceCase(
        case.name,
        case.graph,
        case.spike_inputs,
        case.drive_inputs,
        (),
        case.t_end,
    )


def _scalar_codec_case() -> ReferenceCase:
    graph = Graph(
        models=(GraphModel("lif", VALIDATION_LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "burst",
                0,
                InputMode.SPIKE,
                encoder=BurstEncoder(1.0, 1.0, duration=3.0, amplitude=20.0),
            ),
        ),
        output_ports=(
            OutputPort("rate", 0, RateDecoder()),
            OutputPort("first", 0, TTFSDecoder()),
        ),
    )
    return ReferenceCase(
        "streaming_scalar_codec",
        graph,
        (),
        (),
        (ScalarInput(0.0, 4.0, "burst", 1.0),),
        4.0,
    )


def reference_cases() -> tuple[ReferenceCase, ...]:
    """Return the bounded, intentionally diverse compatibility corpus."""

    return (
        _from_network(chain_case(8)),
        _from_network(drive_boundary_case(4)),
        _from_network(alpha_population_case(4)),
        _from_network(adaptive_population_case(4)),
        _from_network(mixed_chain_case(6)),
        _scalar_codec_case(),
    )


def _hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _case_inputs(case: ReferenceCase) -> dict[str, object]:
    return {
        "spikes": [
            {"t": _hex(item.t), "port": item.port, "value": _hex(item.value)}
            for item in case.spike_inputs
        ],
        "drives": [
            {"t": _hex(item.t), "port": item.port, "value": _hex(item.value)}
            for item in case.drive_inputs
        ],
        "scalars": [
            {
                "t_start": _hex(item.t_start),
                "t_end": _hex(item.t_end),
                "port": item.port,
                "value": _hex(item.value),
            }
            for item in case.scalar_inputs
        ],
    }


def _case_identity(case: ReferenceCase) -> tuple[str, str]:
    graph_text = case.graph.to_text()
    graph_sha256 = hashlib.sha256(graph_text.encode("utf-8")).hexdigest()
    definition = {
        "graph_sha256": graph_sha256,
        "inputs": _case_inputs(case),
        "t_end": _hex(case.t_end),
    }
    canonical = json.dumps(definition, sort_keys=True, separators=(",", ":"))
    return graph_sha256, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_case(core: CoreEvaluator, case: ReferenceCase):
    trace_counts: Counter[str] = Counter()

    def consume(record) -> None:
        """Count one trace record by causal event kind."""

        trace_counts[record.kind.name] += 1

    resolved = case.graph.resolve()
    result = resolved.run(
        core,
        spike_inputs=case.spike_inputs,
        drive_inputs=case.drive_inputs,
        scalar_inputs=case.scalar_inputs,
        t_end=case.t_end,
        queue_capacity=4096,
        output_capacity=4096,
        same_time_cascade_limit=4096,
        recording=RecordingConfig(
            capture_state=False,
            capacity=0,
            consumer=consume,
        ),
    )
    return resolved, result, dict(sorted(trace_counts.items()))


def _expected_result(core: CoreEvaluator, case: ReferenceCase) -> dict[str, object]:
    resolved, result, trace_counts = _run_case(core, case)
    return {
        "spikes": [
            {"t": _hex(item.t), "node": resolved.node_ids[item.node]}
            for item in result.core.spikes
        ],
        "states": [
            {
                "node": resolved.node_ids[index],
                "t": _hex(item.t_last),
                "values": [_hex(value) for value in item.values],
            }
            for index, item in enumerate(result.core.states)
        ],
        "stats": asdict(result.core.stats),
        "trace_kind_counts": trace_counts,
        "decoded": [
            {
                "port": item.port,
                "node": item.node,
                "window": item.window,
                "valid": item.valid,
                "count": item.count,
                "value": _hex(item.value),
                "first_spike": _hex(item.first_spike),
                "window_start": _hex(item.window_start),
                "window_end": _hex(item.window_end),
            }
            for item in result.decoded
        ],
        "decoded_events": [
            {
                "port": item.port,
                "node": item.node,
                "window": item.window,
                "kind": item.kind.name,
                "valid": item.valid,
                "emitted_at": _hex(item.emitted_at),
                "source_spike_time": _hex(item.source_spike_time),
                "observed_through": _hex(item.observed_through),
                "count": item.count,
                "value": _hex(item.value),
                "first_spike": _hex(item.first_spike),
            }
            for item in result.decoded_events
        ],
    }


def build_reference_document(core: CoreEvaluator) -> dict[str, object]:
    """Evaluate and materialize a deterministic fixture document."""

    cases = []
    for case in reference_cases():
        graph_sha256, case_sha256 = _case_identity(case)
        cases.append(
            {
                "name": case.name,
                "graph_sha256": graph_sha256,
                "case_sha256": case_sha256,
                "t_end": _hex(case.t_end),
                "absolute_tolerance": DEFAULT_ABSOLUTE_TOLERANCE.hex(),
                "expected": _expected_result(core, case),
            }
        )
    return {
        "schema": REFERENCE_FIXTURE_SCHEMA,
        "fixture_set": "lacuna-core-slice-1",
        "cases": cases,
    }


def write_reference_document(
    core: CoreEvaluator,
    path: str | Path,
) -> None:
    """Explicitly replace a reference document after intentional review."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_reference_document(core), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_reference_document(path: str | Path) -> Mapping[str, object]:
    """Load and validate a versioned reference fixture document."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferenceFixtureError(f"cannot read reference fixtures: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != REFERENCE_FIXTURE_SCHEMA:
        raise ReferenceFixtureError(
            f"unsupported reference fixture schema; expected {REFERENCE_FIXTURE_SCHEMA}"
        )
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ReferenceFixtureError("reference fixture cases must be a list")
    return document


def _is_hex_float(value: object) -> bool:
    return isinstance(value, str) and value.lower().lstrip("+-").startswith("0x")


def _compare(
    expected: object,
    actual: object,
    *,
    tolerance: float,
    path: str,
) -> None:
    if _is_hex_float(expected) and _is_hex_float(actual):
        left = float.fromhex(expected)
        right = float.fromhex(actual)
        if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance:
            raise ReferenceFixtureError(
                f"{path}: expected {left!r}, received {right!r}, tolerance={tolerance}"
            )
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ReferenceFixtureError(f"{path}: object keys differ")
        for key in sorted(expected):
            _compare(
                expected[key], actual[key], tolerance=tolerance,
                path=f"{path}.{key}",
            )
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ReferenceFixtureError(
                f"{path}: expected {len(expected)} entries, received "
                f"{len(actual) if isinstance(actual, list) else 'non-list'}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, tolerance=tolerance, path=f"{path}[{index}]")
        return
    if expected != actual:
        raise ReferenceFixtureError(
            f"{path}: expected {expected!r}, received {actual!r}"
        )


def _verify_incremental(core: CoreEvaluator, case: ReferenceCase) -> None:
    resolved = case.graph.resolve()
    expected = resolved.run(
        core,
        spike_inputs=case.spike_inputs,
        drive_inputs=case.drive_inputs,
        scalar_inputs=case.scalar_inputs,
        t_end=case.t_end,
        queue_capacity=4096,
        output_capacity=4096,
        same_time_cascade_limit=4096,
    )
    boundaries = (case.t_end / 3.0, 2.0 * case.t_end / 3.0)
    chunks = []
    frontier = 0.0
    with resolved.compile(core) as compiled:
        with compiled.create_incremental_run(
            t_end=case.t_end,
            queue_capacity=4096,
            output_capacity=4096,
            same_time_cascade_limit=4096,
        ) as run:
            for boundary in boundaries:
                chunks.append(
                    run.advance_until(
                        boundary,
                        spike_inputs=tuple(
                            item
                            for item in case.spike_inputs
                            if frontier <= item.t < boundary
                        ),
                        drive_inputs=tuple(
                            item
                            for item in case.drive_inputs
                            if frontier <= item.t < boundary
                        ),
                        scalar_inputs=tuple(
                            item
                            for item in case.scalar_inputs
                            if frontier <= item.t_start < boundary
                        ),
                    )
                )
                frontier = boundary
            chunks.append(
                run.finish(
                    spike_inputs=tuple(
                        item
                        for item in case.spike_inputs
                        if frontier <= item.t <= case.t_end
                    ),
                    drive_inputs=tuple(
                        item
                        for item in case.drive_inputs
                        if frontier <= item.t <= case.t_end
                    ),
                    scalar_inputs=tuple(
                        item
                        for item in case.scalar_inputs
                        if frontier <= item.t_start <= case.t_end
                    ),
                )
            )
    split_spikes = tuple(
        spike for chunk in chunks for spike in chunk.core.spikes
    )
    if len(split_spikes) != len(expected.core.spikes):
        raise ReferenceFixtureError(
            f"{case.name}: incremental spike count differs from one-shot"
        )
    for index, (actual, reference) in enumerate(
        zip(split_spikes, expected.core.spikes)
    ):
        if actual.node != reference.node or abs(actual.t - reference.t) > DEFAULT_ABSOLUTE_TOLERANCE:
            raise ReferenceFixtureError(
                f"{case.name}: incremental spike {index} differs from one-shot"
            )
    final = chunks[-1]
    for index, (actual, reference) in enumerate(
        zip(final.core.states, expected.core.states)
    ):
        if actual.t_last != reference.t_last or any(
            abs(left - right) > DEFAULT_ABSOLUTE_TOLERANCE
            for left, right in zip(actual.values, reference.values)
        ):
            raise ReferenceFixtureError(
                f"{case.name}: incremental final state {index} differs from one-shot"
            )
    if final.decoded != expected.decoded:
        raise ReferenceFixtureError(
            f"{case.name}: incremental decoded values differ from one-shot"
        )


def check_reference_document(
    core: CoreEvaluator,
    document: Mapping[str, object],
    *,
    verify_incremental: bool = True,
) -> dict[str, object]:
    """Compare the live core against a loaded versioned reference document."""

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise ReferenceFixtureError("reference fixture cases must be a list")
    expected_by_name: dict[str, Mapping[str, object]] = {}
    for item in raw_cases:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ReferenceFixtureError("every reference case requires a string name")
        if item["name"] in expected_by_name:
            raise ReferenceFixtureError(f"duplicate reference case '{item['name']}'")
        expected_by_name[item["name"]] = item
    live_cases = {case.name: case for case in reference_cases()}
    if set(expected_by_name) != set(live_cases):
        raise ReferenceFixtureError("reference fixture case registry differs from the document")

    checked = []
    for name in sorted(live_cases):
        case = live_cases[name]
        fixture = expected_by_name[name]
        graph_sha256, case_sha256 = _case_identity(case)
        if fixture.get("graph_sha256") != graph_sha256:
            raise ReferenceFixtureError(f"{name}: graph identity changed")
        if fixture.get("case_sha256") != case_sha256:
            raise ReferenceFixtureError(f"{name}: input or horizon identity changed")
        try:
            tolerance = float.fromhex(str(fixture["absolute_tolerance"]))
            expected = fixture["expected"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ReferenceFixtureError(f"{name}: malformed fixture metadata") from exc
        actual = _expected_result(core, case)
        _compare(expected, actual, tolerance=tolerance, path=name)
        if verify_incremental:
            _verify_incremental(core, case)
        checked.append(name)
    return {
        "schema": REFERENCE_FIXTURE_SCHEMA,
        "fixture_set": document.get("fixture_set"),
        "cases_checked": len(checked),
        "incremental_equivalence_checked": verify_incremental,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run reference generation or verification from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, help="path to the C evaluator")
    parser.add_argument("--fixtures", required=True, help="reference JSON path")
    parser.add_argument(
        "--update",
        action="store_true",
        help="explicitly replace fixtures with current behavior",
    )
    parser.add_argument(
        "--one-shot-only",
        action="store_true",
        help="skip incremental equivalence checks",
    )
    arguments = parser.parse_args(argv)
    core = CoreEvaluator(arguments.library)
    if arguments.update:
        write_reference_document(core, arguments.fixtures)
        summary = {"updated": True, "path": str(arguments.fixtures)}
    else:
        summary = check_reference_document(
            core,
            load_reference_document(arguments.fixtures),
            verify_incremental=not arguments.one_shot_only,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
