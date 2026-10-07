from __future__ import annotations

from dataclasses import replace

import pytest

from lacuna import (
    DriveInput,
    Graph,
    GraphModel,
    GraphNode,
    RecordingConfig,
    ReconstructionSource,
    ReconstructionSupport,
    ReconstructionUnavailableError,
    SpikeInput,
    StateInspectionRequest,
    TraceArtifactIdentityError,
    TraceArtifactMetadata,
    TraceArtifactReader,
    TraceArtifactWriter,
    TraceKind,
    TraceReconstructor,
)
from lacuna.ffi import CoreEvaluator
from lacuna.validation import adaptive_population_case, alpha_population_case

from .test_graph import _graph
from .test_stepped import QIF


def _stream_trace(
    core: CoreEvaluator,
    resolved,
    path,
    *,
    t_end: float,
    recording: RecordingConfig,
    inspections=(),
    spike_inputs=(),
    drive_inputs=(),
):
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved, recording)
    with TraceArtifactWriter(path, metadata, chunk_records=3) as writer:
        result = resolved.run(
            core,
            spike_inputs=spike_inputs,
            drive_inputs=drive_inputs,
            t_end=t_end,
            recording=replace(recording, capacity=0, consumer=writer),
            inspections=inspections,
        )
        writer.set_summary({"t_end": t_end})
    return result, metadata


def test_scalar_reconstruction_matches_in_run_inspection_across_refractory(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    requests = (
        StateInspectionRequest(3.5, 20, (0,)),
        StateInspectionRequest(0.5, 20, (0,)),
        StateInspectionRequest(4.0, 20, (0,)),
        StateInspectionRequest(1.0, 20, (0,)),
        StateInspectionRequest(1.5, 20, (0,)),
        StateInspectionRequest(3.0, 20, (0,)),
    )
    path = tmp_path / "scalar.lctrace"
    run, metadata = _stream_trace(
        core,
        resolved,
        path,
        t_end=4.0,
        recording=RecordingConfig(capacity=0),
        inspections=requests,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
    )

    with TraceArtifactReader(
        path, expected_graph_sha256=metadata.graph_sha256
    ) as reader:
        reconstructor = reader.reconstructor(core, resolved)
        diagnostic = reconstructor.diagnostic(20, (0,))
        reconstructed = reconstructor.reconstruct(requests)
        with pytest.raises(ValueError, match="exceeds artifact end time"):
            reconstructor.reconstruct_one(4.1, 20, (0,))

    assert diagnostic.support is ReconstructionSupport.EXACT
    assert diagnostic.missing_state_names == ()
    assert [item.values for item in reconstructed] == [
        item.values for item in run.inspections
    ]
    assert [item.clamped for item in reconstructed] == [
        item.clamped for item in run.inspections
    ]
    assert reconstructed[0].source is ReconstructionSource.ANALYTICAL_PROPAGATION
    assert reconstructed[1].source is ReconstructionSource.ANALYTICAL_PROPAGATION
    assert reconstructed[2].source is ReconstructionSource.RECORDED_EVENT
    assert reconstructed[3].source is ReconstructionSource.RECORDED_EVENT
    assert reconstructed[4].source is ReconstructionSource.ANALYTICAL_PROPAGATION
    assert reconstructed[5].source is ReconstructionSource.RECORDED_EVENT


def test_drive_parameter_replay_matches_inspection(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    requests = (
        StateInspectionRequest(4.0, 10, (0,)),
        StateInspectionRequest(7.0, 20, (0,)),
        StateInspectionRequest(5.0, 10, (0,)),
        StateInspectionRequest(10.0, 10, (0,)),
        StateInspectionRequest(15.0, 10, (0,)),
    )
    path = tmp_path / "drive.lctrace"
    run, _ = _stream_trace(
        core,
        resolved,
        path,
        t_end=15.0,
        recording=RecordingConfig(capacity=0),
        inspections=requests,
        drive_inputs=(DriveInput(5.0, "bias", 20.0),),
    )

    with TraceArtifactReader(path) as reader:
        reconstructed = TraceReconstructor(core, resolved, reader).reconstruct(
            requests
        )
    assert [item.values for item in reconstructed] == [
        item.values for item in run.inspections
    ]
    assert reconstructed[3].anchor_time == 5.0
    assert reconstructed[3].source is ReconstructionSource.ANALYTICAL_PROPAGATION


def test_stepped_reconstruction_replays_numerically_in_the_c_core(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = Graph(
        models=(GraphModel("qif", QIF),),
        nodes=(GraphNode(0, "qif", 0.0, {}),),
    ).resolve()
    requests = (
        StateInspectionRequest(0.2, 0, (0,)),
        StateInspectionRequest(0.5, 0, (0,)),
        StateInspectionRequest(0.6, 0, (0,)),
    )
    path = tmp_path / "stepped.lctrace"
    run, _ = _stream_trace(
        core,
        resolved,
        path,
        t_end=0.6,
        recording=RecordingConfig(capacity=0),
        inspections=requests,
    )
    with TraceArtifactReader(path) as reader:
        reconstructor = TraceReconstructor(core, resolved, reader)
        diagnostic = reconstructor.diagnostic(0, (0,))
        reconstructed = reconstructor.reconstruct(requests)
    assert diagnostic.support is ReconstructionSupport.EXACT
    assert [item.values for item in reconstructed] == pytest.approx(
        [item.values for item in run.inspections], abs=2e-10
    )
    assert reconstructed[0].source is ReconstructionSource.NUMERICAL_PROPAGATION
    assert reconstructed[1].source is ReconstructionSource.NUMERICAL_PROPAGATION
    assert reconstructed[2].source is ReconstructionSource.RECORDED_EVENT


@pytest.mark.parametrize("case_factory", [alpha_population_case, adaptive_population_case])
def test_vector_state_reconstruction_matches_inspection(
    core: CoreEvaluator,
    tmp_path,
    case_factory,
) -> None:
    case = case_factory(1)
    resolved = case.graph.resolve()
    times = (0.5, 1.0, 3.0, case.t_end)
    state_count = len(resolved.models[0].state_names)
    requests = tuple(
        StateInspectionRequest(t, 0, tuple(range(state_count))) for t in times
    )
    path = tmp_path / f"{case.name}.lctrace"
    run, _ = _stream_trace(
        core,
        resolved,
        path,
        t_end=case.t_end,
        recording=RecordingConfig(capacity=0),
        inspections=requests,
        spike_inputs=case.spike_inputs,
        drive_inputs=case.drive_inputs,
    )

    with TraceArtifactReader(path) as reader:
        reconstructor = TraceReconstructor(core, resolved, reader)
        assert reconstructor.diagnostic(0).support is ReconstructionSupport.EXACT
        reconstructed = reconstructor.reconstruct(requests)
    assert [item.values for item in reconstructed] == [
        item.values for item in run.inspections
    ]
    assert [item.clamped for item in reconstructed] == [
        item.clamped for item in run.inspections
    ]


def test_partial_vector_trace_is_event_only_and_names_missing_dependencies(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    case = alpha_population_case(1)
    resolved = case.graph.resolve()
    path = tmp_path / "partial-alpha.lctrace"
    _stream_trace(
        core,
        resolved,
        path,
        t_end=case.t_end,
        recording=RecordingConfig(state_indices=(0,), capacity=0),
        spike_inputs=case.spike_inputs,
    )

    with TraceArtifactReader(path) as reader:
        reconstructor = TraceReconstructor(core, resolved, reader)
        membrane = reconstructor.diagnostic(0, (0,))
        synapse = reconstructor.diagnostic(0, (1,))
        at_deposit = reconstructor.reconstruct_one(1.0, 0, (0,))
        with pytest.raises(ReconstructionUnavailableError, match="event-time"):
            reconstructor.reconstruct_one(2.0, 0, (0,))

    assert membrane.support is ReconstructionSupport.EVENT_ONLY
    assert membrane.missing_state_names == (
        "ValidationAlpha.s",
        "ValidationAlpha.z",
    )
    assert synapse.support is ReconstructionSupport.UNAVAILABLE
    assert at_deposit.source is ReconstructionSource.RECORDED_EVENT


@pytest.mark.parametrize(
    ("case_factory", "recorded", "selected", "t", "required_names"),
    [
        (
            alpha_population_case,
            (2,),
            (2,),
            2.0,
            ("ValidationAlpha.z",),
        ),
        (
            alpha_population_case,
            (1, 2),
            (1,),
            2.0,
            ("ValidationAlpha.s", "ValidationAlpha.z"),
        ),
        (
            adaptive_population_case,
            (1,),
            (1,),
            20.0,
            ("w",),
        ),
    ],
)
def test_dependency_closed_partial_trace_reconstructs_between_events(
    core: CoreEvaluator,
    tmp_path,
    case_factory,
    recorded,
    selected,
    t,
    required_names,
) -> None:
    case = case_factory(1)
    resolved = case.graph.resolve()
    request = StateInspectionRequest(t, 0, selected)
    path = tmp_path / (
        f"dependency-{case.name}-{'-'.join(map(str, recorded))}.lctrace"
    )
    run, _ = _stream_trace(
        core,
        resolved,
        path,
        t_end=case.t_end,
        recording=RecordingConfig(state_indices=recorded, capacity=0),
        inspections=(request,),
        spike_inputs=case.spike_inputs,
        drive_inputs=case.drive_inputs,
    )

    with TraceArtifactReader(path) as reader:
        reconstructor = TraceReconstructor(core, resolved, reader)
        diagnostic = reconstructor.diagnostic(0, selected)
        reconstructed = reconstructor.reconstruct_one(t, 0, selected)

    assert diagnostic.support is ReconstructionSupport.EXACT
    assert diagnostic.required_state_names == required_names
    assert diagnostic.missing_state_names == ()
    assert reconstructed.values == run.inspections[0].values
    assert reconstructed.clamped == run.inspections[0].clamped
    assert reconstructed.source is ReconstructionSource.ANALYTICAL_PROPAGATION


def test_missing_transition_kind_is_reported_before_query(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    kinds = frozenset(kind for kind in TraceKind if kind is not TraceKind.RESET)
    path = tmp_path / "missing-reset.lctrace"
    _stream_trace(
        core,
        resolved,
        path,
        t_end=2.0,
        recording=RecordingConfig(kinds=kinds, capacity=0),
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
    )

    with TraceArtifactReader(path) as reader:
        reconstructor = TraceReconstructor(core, resolved, reader)
        diagnostic = reconstructor.diagnostic(20)
        with pytest.raises(ReconstructionUnavailableError, match="RESET"):
            reconstructor.reconstruct_one(1.5, 20)
    assert diagnostic.support is ReconstructionSupport.UNAVAILABLE
    assert diagnostic.missing_trace_kinds == (TraceKind.RESET,)


def test_reconstruction_rejects_incomplete_or_different_graph(
    core: CoreEvaluator,
    tmp_path,
) -> None:
    resolved = _graph().resolve()
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved)
    incomplete_path = tmp_path / "incomplete.lctrace"
    writer = TraceArtifactWriter(incomplete_path, metadata)
    writer.abort()
    with TraceArtifactReader(incomplete_path, allow_incomplete=True) as reader:
        with pytest.raises(ReconstructionUnavailableError, match="complete"):
            TraceReconstructor(core, resolved, reader)

    complete_path = tmp_path / "complete.lctrace"
    _stream_trace(
        core,
        resolved,
        complete_path,
        t_end=1.0,
        recording=RecordingConfig(capacity=0),
    )
    other = alpha_population_case(1).graph.resolve()
    with TraceArtifactReader(complete_path) as reader:
        with pytest.raises(TraceArtifactIdentityError, match="graph hash"):
            TraceReconstructor(core, other, reader)
