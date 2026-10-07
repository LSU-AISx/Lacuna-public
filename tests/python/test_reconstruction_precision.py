"""Persisted target traces must reconstruct through the same native profile."""

from dataclasses import replace

import pytest

from lacuna import (
    CoreEvaluator, PrecisionProfile, RecordingConfig, SpikeInput,
    StateInspectionRequest, TraceArtifactMetadata, TraceArtifactReader,
    TraceArtifactWriter, TraceArtifactIdentityError,
)
from lacuna.tracefile import TraceArtifactError
from lacuna.target_lowering import resolve_target_graph

from .test_graph import _graph
from .test_reconstruction import _stream_trace


@pytest.fixture(params=("float32", "float32-time64"))
def target_core(request):
    return CoreEvaluator(precision=request.param)


def test_target_trace_reconstruction_and_precision_guards(target_core, core, tmp_path):
    graph = _graph()
    resolved = resolve_target_graph(graph, target_core)
    requests = (
        StateInspectionRequest(0.5, 20, (0,)),
        StateInspectionRequest(1.5, 20, (0,)),
        StateInspectionRequest(3.5, 20, (0,)),
    )
    path = tmp_path / "target.lctrace"
    run, metadata = _stream_trace(
        target_core, resolved, path, t_end=4.0,
        recording=RecordingConfig(capacity=0), inspections=requests,
        spike_inputs=(SpikeInput(1.0, "stimulus", 20.0),),
    )
    assert metadata.precision is target_core.precision
    with TraceArtifactReader(path) as reader:
        assert reader.metadata.precision is target_core.precision
        for source in (graph, resolved):
            values = reader.reconstructor(target_core, source).reconstruct(requests)
            assert [item.values for item in values] == [item.values for item in run.inspections]
        with pytest.raises(TraceArtifactIdentityError, match="C evaluator"):
            reader.reconstructor(core, resolved)
        with pytest.raises(TraceArtifactIdentityError, match="resolved graph"):
            reader.reconstructor(target_core, graph.resolve())


def test_trace_metadata_rejects_contradictory_precision(target_core):
    resolved = resolve_target_graph(_graph(), target_core)
    with pytest.raises(ValueError, match="precision"):
        TraceArtifactMetadata.from_resolved_graph(resolved, RecordingConfig(), extra={
            "lacuna_precision": PrecisionProfile.FLOAT64.to_record(),
        })


def test_trace_writer_rejects_values_wider_than_declared_profile(target_core, tmp_path):
    resolved = resolve_target_graph(_graph(), target_core)
    metadata = TraceArtifactMetadata.from_resolved_graph(resolved, RecordingConfig())
    with resolved.compile(target_core) as compiled:
        result = compiled.run(t_end=1.0, recording=RecordingConfig(capacity=32))
    record = result.trace[0]
    with TraceArtifactWriter(tmp_path / "invalid.lctrace", metadata) as writer:
        with pytest.raises(TraceArtifactError, match="precision"):
            writer(replace(record, value=0.1))
        writer(record)
