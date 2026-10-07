from __future__ import annotations

import json
import math

import pytest

from lacuna import (
    BurstEncoder,
    DecodeEventKind,
    DecodeQuery,
    DecodeWindow,
    DecoderBinding,
    DecoderQueryBinding,
    DecoderWindowBinding,
    EmissionPolicy,
    EncodedBatch,
    Graph,
    GraphModel,
    GraphNode,
    HeldCurrentEncoder,
    InputMode,
    InputPort,
    LatencyBurstEncoder,
    MixedInputSpike,
    OutputPort,
    PoissonRateEncoder,
    Presentation,
    RateDecoder,
    RateMode,
    RecordingConfig,
    RegularRateEncoder,
    ScalarInput,
    Spike,
    StateInspectionRequest,
    TemporalSpikeMode,
    TemporalWeightDecoder,
    TTFSEncoder,
    TTFSDecoder,
    TraceKind,
    parse_neuron,
    resolve_scalar_lif,
)
from lacuna.errors import CoreError, ResolutionError
from lacuna.ffi import CoreEvaluator

from .test_dsl_resolver import LIF


def test_regular_rate_preserves_phase_across_presentations(
    core: CoreEvaluator,
) -> None:
    encoded = core.encode_presentations(
        (RegularRateEncoder(1.0, 1.0),),
        (Presentation(0.0, 0.6, 0, 1.0), Presentation(0.6, 1.2, 0, 1.0)),
    )
    assert [item.t for item in encoded.spikes] == pytest.approx([1.0])


def test_regular_rate_defers_an_end_boundary_spike_to_the_next_window(
    core: CoreEvaluator,
) -> None:
    encoded = core.encode_presentations(
        (RegularRateEncoder(1.0, 1.0),),
        (Presentation(0.0, 1.0, 0, 1.0), Presentation(1.0, 2.0, 0, 1.0)),
    )
    assert [item.t for item in encoded.spikes] == pytest.approx([1.0])


def test_poisson_streams_are_seeded_replayable_and_port_independent(
    core: CoreEvaluator,
) -> None:
    encoders = (PoissonRateEncoder(2.0, 2.0), PoissonRateEncoder(2.0, 2.0))
    presentations = (
        Presentation(0.0, 10.0, 0, 1.0),
        Presentation(0.0, 10.0, 1, 1.0),
    )
    first = core.encode_presentations(encoders, presentations, seed=91)
    replay = core.encode_presentations(encoders, presentations, seed=91)
    changed = core.encode_presentations(encoders, presentations, seed=92)
    assert replay == first
    assert changed != first
    assert [item.t for item in first.spikes if item.encoder == 0] != [
        item.t for item in first.spikes if item.encoder == 1
    ]


def test_ttfs_burst_latency_burst_and_held_current(core: CoreEvaluator) -> None:
    encoders = (
        TTFSEncoder(1.0, 5.0, amplitude=3.0),
        BurstEncoder(0.0, 2.0, duration=2.0, amplitude=4.0),
        LatencyBurstEncoder(1.0, 5.0, rate=2.0, duration=1.1, amplitude=5.0),
        HeldCurrentEncoder(gain=2.0, offset=-1.0, baseline=-3.0),
    )
    encoded = core.encode_presentations(
        encoders,
        tuple(Presentation(10.0, 20.0, index, 1.0) for index in range(4)),
    )
    assert [(item.encoder, item.t, item.value) for item in encoded.spikes] == [
        (0, 11.0, 3.0),
        (1, 10.0, 4.0),
        (1, 10.5, 4.0),
        (1, 11.0, 4.0),
        (1, 11.5, 4.0),
        (2, 11.0, 5.0),
        (2, 11.5, 5.0),
        (2, 12.0, 5.0),
    ]
    assert [(item.t, item.value) for item in encoded.drives] == [
        (10.0, 1.0),
        (20.0, -3.0),
    ]


def test_new_presentation_cancels_unelapsed_prior_encoding(
    core: CoreEvaluator,
) -> None:
    encoded = core.encode_presentations(
        (RegularRateEncoder(1.0, 1.0),),
        (Presentation(0.0, 2.0, 0, 1.0), Presentation(1.0, 3.0, 0, 1.0)),
    )
    assert [item.t for item in encoded.spikes] == pytest.approx([1.0, 2.0])


def test_duplicate_presentation_start_is_rejected(core: CoreEvaluator) -> None:
    with pytest.raises(CoreError, match="invalid argument"):
        core.encode_presentations(
            (RegularRateEncoder(1.0, 1.0),),
            (Presentation(0.0, 2.0, 0, 1.0), Presentation(0.0, 3.0, 0, 1.0)),
        )


def test_incremental_encoder_session_matches_one_shot_across_chunks(
    core: CoreEvaluator,
) -> None:
    encoders = (
        RegularRateEncoder(0.5, 2.0, amplitude=2.0),
        PoissonRateEncoder(1.0, 3.0, amplitude=3.0),
        TTFSEncoder(0.1, 1.2, amplitude=4.0),
        BurstEncoder(0.5, 2.0, duration=1.5, amplitude=5.0),
        LatencyBurstEncoder(
            0.1, 1.0, rate=3.0, duration=1.2, amplitude=6.0
        ),
        HeldCurrentEncoder(gain=2.0, offset=-0.5, baseline=-2.0),
    )
    first = tuple(Presentation(0.0, 4.0, index, 0.8) for index in range(6))
    second = tuple(Presentation(1.0, 3.0, index, 0.4) for index in range(6))
    third = tuple(Presentation(3.5, 5.0, index, 1.0) for index in range(6))
    presentations = first + second + third
    expected = core.encode_presentations(
        encoders, presentations, seed=817, spike_capacity=256
    )

    session = core.create_encoder_session(
        encoders, seed=817, spike_capacity=256
    )
    session.submit(first)
    empty = session.advance(0.5)
    session.submit(second)
    early = session.advance(1.0)
    middle = session.advance(3.0)
    session.submit(third)
    late = session.advance(3.5)
    final = session.finish()

    assert empty == EncodedBatch((), ())
    assert EncodedBatch(
        early.spikes + middle.spikes + late.spikes + final.spikes,
        early.drives + middle.drives + late.drives + final.drives,
    ) == expected
    assert session.finished
    with pytest.raises(RuntimeError, match="finished"):
        session.submit(())


def test_encoder_session_watermarks_and_ordering_are_explicit(
    core: CoreEvaluator,
) -> None:
    session = core.create_encoder_session((RegularRateEncoder(1.0, 1.0),))
    session.submit((Presentation(0.0, 2.0, 0, 1.0),))
    assert session.advance(1.0) == EncodedBatch((), ())
    with pytest.raises(ValueError, match="committed watermark"):
        session.submit((Presentation(0.5, 3.0, 0, 1.0),))
    with pytest.raises(ValueError, match="nondecreasing"):
        session.advance(0.5)
    session.submit((Presentation(1.0, 3.0, 0, 1.0),))
    encoded = session.advance(1.0)
    assert encoded == EncodedBatch((), ())
    assert [spike.t for spike in session.finish().spikes] == pytest.approx(
        [1.0, 2.0]
    )


def test_encoder_session_is_poisoned_after_partial_c_failure(
    core: CoreEvaluator,
) -> None:
    session = core.create_encoder_session(
        (RegularRateEncoder(100.0, 100.0),), spike_capacity=1
    )
    session.submit((Presentation(0.0, 1.0, 0, 1.0),))
    with pytest.raises(CoreError, match="output spike capacity exceeded"):
        session.advance(1.0)
    with pytest.raises(RuntimeError, match="failed C encoding call"):
        session.finish()


@pytest.mark.parametrize(
    "encoder",
    (
        RegularRateEncoder(1.0, 1.0, amplitude=2.0),
        PoissonRateEncoder(2.0, 2.0, amplitude=3.0),
        TTFSEncoder(1.0, 1.0, amplitude=4.0),
        BurstEncoder(2.0, 2.0, duration=4.0, amplitude=5.0),
        LatencyBurstEncoder(
            1.0, 1.0, rate=2.0, duration=2.0, amplitude=6.0
        ),
        HeldCurrentEncoder(gain=2.0, offset=-0.5, baseline=-2.0),
    ),
)
def test_streaming_encoder_run_matches_one_shot_with_frontier_replacement(
    core: CoreEvaluator,
    encoder,
) -> None:
    presentations = (
        Presentation(0.0, 5.0, 0, 0.8),
        Presentation(2.5, 4.5, 0, 0.4),
    )
    expected = core.encode_presentations(
        (encoder,), presentations, seed=817, spike_capacity=256
    )
    with core.create_streaming_encoder_run(
        (encoder,), seed=817, spike_capacity=256
    ) as run:
        first = run.advance_until(1.25, presentations[:1])
        second = run.advance_until(2.5)
        third = run.advance_until(3.5, presentations[1:])
        final = run.finish(5.0)

    spikes = first.spikes + second.spikes + third.spikes + final.spikes
    drives = first.drives + second.drives + third.drives + final.drives
    assert [(item.encoder, item.value) for item in spikes] == [
        (item.encoder, item.value) for item in expected.spikes
    ]
    assert [item.t for item in spikes] == pytest.approx(
        [item.t for item in expected.spikes], abs=1e-14
    )
    assert [(item.encoder, item.value) for item in drives] == [
        (item.encoder, item.value) for item in expected.drives
    ]
    assert [item.t for item in drives] == pytest.approx(
        [item.t for item in expected.drives], abs=1e-14
    )


def test_streaming_encoder_keeps_pause_open_and_includes_final_horizon(
    core: CoreEvaluator,
) -> None:
    encoder = BurstEncoder(1.0, 1.0, duration=10.0)
    presentation = Presentation(0.0, 10.0, 0, 1.0)
    with core.create_streaming_encoder_run((encoder,)) as run:
        first = run.advance_until(1.0, (presentation,))
        second = run.advance_until(2.0)
        final = run.finish(3.0)

    assert [item.t for item in first.spikes] == [0.0]
    assert [item.t for item in second.spikes] == [1.0]
    assert [item.t for item in final.spikes] == [2.0, 3.0]


def test_streaming_encoder_episode_reset_clears_regular_phase(
    core: CoreEvaluator,
) -> None:
    encoder = RegularRateEncoder(0.3, 0.3)
    with core.create_streaming_encoder_run((encoder,)) as run:
        first = run.advance_until(2.0, (Presentation(0.0, 2.0, 0, 1.0),))
        run.reset_episode()
        second = run.advance_until(4.0, (Presentation(2.0, 4.0, 0, 1.0),))

    assert first.spikes == ()
    assert second.spikes == ()


def test_streaming_held_current_restores_only_at_true_end(
    core: CoreEvaluator,
) -> None:
    encoder = HeldCurrentEncoder(gain=2.0, offset=-1.0, baseline=-3.0)
    presentation = Presentation(0.0, 2.0, 0, 1.0)
    with core.create_streaming_encoder_run((encoder,)) as run:
        first = run.advance_until(1.0, (presentation,))
        boundary = run.advance_until(2.0)
        final = run.finish(3.0)

    assert [(item.t, item.value) for item in first.drives] == [(0.0, 1.0)]
    assert boundary.drives == ()
    assert [(item.t, item.value) for item in final.drives] == [(2.0, -3.0)]


def test_streaming_encoder_capacity_failure_is_transactional(
    core: CoreEvaluator,
) -> None:
    presentation = Presentation(0.0, 1.0, 0, 1.0)
    with core.create_streaming_encoder_run(
        (RegularRateEncoder(100.0, 100.0),), spike_capacity=1
    ) as run:
        with pytest.raises(CoreError, match="output spike capacity exceeded"):
            run.advance_until(1.0, (presentation,))
        assert run.frontier == 0.0
        retry = run.advance_until(0.01, (presentation,))
    assert retry.spikes == ()


def test_decoders_cover_all_initial_modes(core: CoreEvaluator) -> None:
    spikes = (Spike(0.5, 0), Spike(1.5, 0), Spike(4.0, 0))
    decoded = core.decode_spikes(
        (
            DecoderBinding(0, RateDecoder()),
            DecoderBinding(0, RateDecoder(RateMode.SLIDING, width=2.0)),
            DecoderBinding(0, RateDecoder(RateMode.CUMULATIVE, origin=-1.0)),
            DecoderBinding(0, TTFSDecoder()),
            DecoderBinding(1, TTFSDecoder(normalize=True)),
            DecoderBinding(0, TemporalWeightDecoder(2.0)),
            DecoderBinding(
                0,
                TemporalWeightDecoder(
                    2.0, spikes=TemporalSpikeMode.FIRST, normalize=True
                ),
            ),
        ),
        spikes,
        t_start=0.0,
        t_end=5.0,
    )
    assert decoded[0].count == 3 and decoded[0].value == pytest.approx(0.6)
    assert decoded[1].count == 1 and decoded[1].value == pytest.approx(0.5)
    assert decoded[2].count == 3 and decoded[2].value == pytest.approx(0.5)
    assert decoded[3].value == pytest.approx(0.5)
    assert not decoded[4].valid and decoded[4].value is None
    assert decoded[5].value == pytest.approx(
        math.exp(-0.25) + math.exp(-0.75) + math.exp(-2.0)
    )
    assert decoded[6].count == 1 and decoded[6].value == pytest.approx(
        math.exp(-0.25)
    )


def test_streaming_decoder_bank_matches_batch_and_resets(core: CoreEvaluator) -> None:
    bindings = (
        DecoderBinding(0, RateDecoder()),
        DecoderBinding(0, RateDecoder(RateMode.SLIDING, width=2.0)),
        DecoderBinding(0, RateDecoder(RateMode.CUMULATIVE, origin=-1.0)),
        DecoderBinding(0, TTFSDecoder(normalize=True)),
        DecoderBinding(0, TemporalWeightDecoder(2.0)),
        DecoderBinding(
            0, TemporalWeightDecoder(2.0, spikes=TemporalSpikeMode.FIRST)
        ),
    )
    # The spike at the right boundary is excluded from every decoder.
    spikes = (Spike(0.0, 0), Spike(1.5, 0), Spike(4.0, 0), Spike(5.0, 0))
    expected = core.decode_spikes(bindings, spikes, t_start=0.0, t_end=5.0)
    with core.compile_decoders(bindings, node_count=1) as compiled:
        with compiled.create_run(t_start=0.0, t_end=5.0) as run:
            run.consume(spikes)
            first = run.finalize()
            first_events = run.events()
            run.reset(t_start=0.0, t_end=5.0)
            run.consume(spikes)
            replay = run.finalize()
            replay_events = run.events()
    assert first == expected
    assert replay == first
    assert replay_events == first_events
    assert first_events[0].kind is DecodeEventKind.FINAL
    assert first_events[0].emitted_at == 0.0
    assert first_events[0].source_spike_time == 0.0


def _codec_graph() -> Graph:
    return Graph(
        models=(GraphModel("lif", LIF),),
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
            OutputPort("weighted", 0, TemporalWeightDecoder(2.0)),
        ),
    )


def test_graph_uses_per_port_codecs_and_round_trips_them(core: CoreEvaluator) -> None:
    graph = _codec_graph()
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
    )
    # The model's two-unit refractory clamp rejects the middle burst spike.
    assert [item.t for item in result.core.spikes] == pytest.approx([0.0, 2.0])
    assert {item.port: item.value for item in result.decoded} == pytest.approx(
        {
            "rate": 0.5,
            "first": 0.0,
            "weighted": 1.0 + math.exp(-1.0),
        }
    )
    text = graph.to_text()
    assert Graph.from_text(text).to_text() == text
    document = json.loads(text)
    assert document["schema"] == 10
    assert len(document["input_ports"][0]["encoder_hash"]) == 64
    assert len(document["output_ports"][0]["decoder_hash"]) == 64
    first_port = next(
        item for item in document["output_ports"] if item["id"] == "first"
    )
    assert first_port["decoder"]["emission"] == "ON_EVENT"


def test_incremental_graph_streams_scalar_presentations_without_restart(
    core: CoreEvaluator,
) -> None:
    resolved = _codec_graph().resolve()
    presentation = ScalarInput(0.0, 4.0, "burst", 1.0)
    with resolved.compile(core) as compiled:
        expected = compiled.run(
            scalar_inputs=(presentation,),
            t_end=4.0,
        )
        with compiled.create_incremental_run(t_end=4.0) as run:
            first = run.advance_until(1.0, scalar_inputs=(presentation,))
            second = run.advance_until(2.0)
            final = run.finish()

    assert first.core.spikes + second.core.spikes + final.core.spikes == (
        expected.core.spikes
    )
    assert final.core.states == expected.core.states
    assert final.decoded == expected.decoded
    assert first.outputs + second.outputs + final.outputs == expected.outputs


def test_incremental_graph_rejects_scalar_input_behind_frontier(
    core: CoreEvaluator,
) -> None:
    with _codec_graph().resolve().compile(core) as compiled:
        with compiled.create_incremental_run(t_end=4.0) as run:
            run.advance_until(2.0)
            with pytest.raises(ValueError, match="start must be"):
                run.advance_until(
                    3.0,
                    scalar_inputs=(ScalarInput(1.0, 3.0, "burst", 1.0),),
                )


def test_graph_emits_exact_time_decoder_updates_and_deadline_finals(
    core: CoreEvaluator,
) -> None:
    base = _codec_graph()
    graph = Graph(
        models=base.models,
        nodes=base.nodes,
        input_ports=base.input_ports,
        output_ports=(
            OutputPort("rate", 0, RateDecoder()),
            OutputPort("first", 0, TTFSDecoder()),
            OutputPort(
                "weighted",
                0,
                TemporalWeightDecoder(
                    2.0,
                    emission=EmissionPolicy.ON_EVENT_AND_WINDOW_CLOSE,
                ),
            ),
        ),
    )
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
        output_capacity=0,
    )

    assert [
        (event.port, event.kind, event.emitted_at, event.source_spike_time)
        for event in result.decoded_events
    ] == [
        ("first", DecodeEventKind.FINAL, 0.0, 0.0),
        ("weighted", DecodeEventKind.UPDATE, 0.0, 0.0),
        ("weighted", DecodeEventKind.UPDATE, 2.0, 2.0),
        ("rate", DecodeEventKind.FINAL, 4.0, None),
        ("weighted", DecodeEventKind.FINAL, 4.0, None),
    ]
    assert result.decoded_events[2].value == pytest.approx(1.0 + math.exp(-1.0))
    assert result.decoded_events[2].observed_through == 2.0
    assert result.decoded_events[-1].observed_through == 4.0


def test_ttfs_no_spike_is_emitted_at_the_deadline(core: CoreEvaluator) -> None:
    result = _codec_graph().resolve().run(
        core,
        t_end=4.0,
        output_capacity=0,
    )
    event = next(item for item in result.decoded_events if item.port == "first")
    assert event.kind is DecodeEventKind.NO_SPIKE
    assert event.emitted_at == 4.0
    assert event.observed_through == 4.0
    assert event.source_spike_time is None
    assert event.value is None
    assert not event.valid


def test_ttfs_can_delay_a_valid_result_until_window_close(
    core: CoreEvaluator,
) -> None:
    base = _codec_graph()
    graph = Graph(
        models=base.models,
        nodes=base.nodes,
        input_ports=base.input_ports,
        output_ports=(
            OutputPort(
                "first",
                0,
                TTFSDecoder(emission=EmissionPolicy.ON_WINDOW_CLOSE),
            ),
        ),
    )
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
        output_capacity=0,
    )
    assert len(result.decoded_events) == 1
    event = result.decoded_events[0]
    assert event.kind is DecodeEventKind.FINAL
    assert event.emitted_at == 4.0
    assert event.source_spike_time is None
    assert event.first_spike == 0.0
    assert event.value == 0.0


def test_decoder_event_capacity_is_explicit_and_can_be_disabled(
    core: CoreEvaluator,
) -> None:
    base = _codec_graph()
    graph = Graph(
        models=base.models,
        nodes=base.nodes,
        input_ports=base.input_ports,
        output_ports=(
            OutputPort("first", 0, TTFSDecoder()),
            OutputPort(
                "weighted",
                0,
                TemporalWeightDecoder(2.0, emission=EmissionPolicy.ON_EVENT),
            ),
        ),
    ).resolve()
    arguments = dict(
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
        output_capacity=0,
    )
    with pytest.raises(CoreError, match="decoded event capacity exceeded") as captured:
        graph.run(core, decoder_event_capacity=1, **arguments)
    diagnostic = captured.value.diagnostic
    assert captured.value.status == 11
    assert diagnostic is not None
    assert diagnostic.resource == "decoder_output"
    assert (diagnostic.capacity, diagnostic.occupancy, diagnostic.peak) == (1, 0, 0)
    assert diagnostic.event_kind == "OUTPUT_SPIKE"
    assert diagnostic.event_phase == "PREDICTION"
    assert diagnostic.event_index == 0
    assert diagnostic.node == 0
    assert diagnostic.t == 0.0

    result = graph.run(core, decoder_event_capacity=0, **arguments)
    assert result.decoded_events == ()
    assert {item.port: item.value for item in result.decoded} == pytest.approx(
        {"first": 0.0, "weighted": 1.0 + math.exp(-1.0)}
    )


def test_graph_closes_multiple_windows_before_same_time_boundary_spikes(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "burst",
                0,
                InputMode.SPIKE,
                encoder=BurstEncoder(1.0, 1.0, duration=5.0, amplitude=20.0),
            ),
        ),
        output_ports=(
            OutputPort("rate", 0, RateDecoder()),
            OutputPort("first", 0, TTFSDecoder()),
        ),
    )
    windows = (
        DecodeWindow(0.0, 2.0),
        DecodeWindow(2.0, 4.0),
        DecodeWindow(4.0, 6.0),
    )
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 6.0, "burst", 1.0),),
        t_end=6.0,
        decode_windows=windows,
        output_capacity=0,
    )
    assert [
        (item.window, item.port, item.count, item.value) for item in result.decoded
    ] == [
        (0, "first", 1, 0.0),
        (0, "rate", 1, 0.5),
        (1, "first", 1, 0.0),
        (1, "rate", 1, 0.5),
        (2, "first", 1, 0.0),
        (2, "rate", 1, 0.5),
    ]
    assert [
        (item.window, item.port, item.emitted_at, item.source_spike_time)
        for item in result.decoded_events
    ] == [
        (0, "first", 0.0, 0.0),
        (0, "rate", 2.0, None),
        (1, "first", 2.0, 2.0),
        (1, "rate", 4.0, None),
        (2, "first", 4.0, 4.0),
        (2, "rate", 6.0, None),
    ]
    with graph.resolve().compile(core) as prepared:
        replay = prepared.run(
            scalar_inputs=(ScalarInput(0.0, 6.0, "burst", 1.0),),
            t_end=6.0,
            decode_windows=windows,
            output_capacity=0,
        )
    assert replay.decoded == result.decoded
    assert replay.decoded_events == result.decoded_events


def test_each_output_port_can_have_its_own_sparse_window_schedule(
    core: CoreEvaluator,
) -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "burst",
                0,
                InputMode.SPIKE,
                encoder=BurstEncoder(1.0, 1.0, duration=5.0, amplitude=20.0),
            ),
        ),
        output_ports=(
            OutputPort("rate", 0, RateDecoder()),
            OutputPort("first", 0, TTFSDecoder()),
        ),
    )
    schedule = {
        "rate": (
            DecodeWindow(0.0, 2.0),
            DecodeWindow(2.0, 4.0),
            DecodeWindow(4.0, 6.0),
        ),
        "first": (DecodeWindow(0.0, 6.0),),
    }
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 6.0, "burst", 1.0),),
        t_end=6.0,
        decode_windows=schedule,
        output_capacity=0,
    )
    assert [(item.port, item.window, item.count, item.value) for item in result.decoded] == [
        ("first", 0, 1, 0.0),
        ("rate", 0, 1, 0.5),
        ("rate", 1, 1, 0.5),
        ("rate", 2, 1, 0.5),
    ]
    assert [
        (item.port, item.window, item.emitted_at)
        for item in result.decoded_events
    ] == [
        ("first", 0, 0.0),
        ("rate", 0, 2.0),
        ("rate", 1, 4.0),
        ("rate", 2, 6.0),
    ]
    with graph.resolve().compile(core) as prepared:
        replay = prepared.run(
            scalar_inputs=(ScalarInput(0.0, 6.0, "burst", 1.0),),
            t_end=6.0,
            decode_windows=schedule,
            output_capacity=0,
        )
    assert replay.decoded == result.decoded
    assert replay.decoded_events == result.decoded_events

    omitted = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 6.0, "burst", 1.0),),
        t_end=6.0,
        decode_windows={"first": (DecodeWindow(0.0, 6.0),)},
        output_capacity=0,
    )
    assert [(item.port, item.window) for item in omitted.decoded] == [
        ("first", 0)
    ]


def test_streaming_decoder_schedule_supports_overlapping_windows(
    core: CoreEvaluator,
) -> None:
    bindings = (DecoderBinding(0, RateDecoder()),)
    windows = (DecodeWindow(0.0, 3.0), DecodeWindow(1.0, 4.0))
    with core.compile_decoders(bindings, node_count=1) as compiled:
        with compiled.create_run(windows=windows) as run:
            run.consume((Spike(0.0, 0), Spike(2.0, 0), Spike(3.0, 0)))
            values = run.finalize()
            events = run.events()
    assert [(item.window, item.count, item.value) for item in values] == [
        (0, 2, pytest.approx(2.0 / 3.0)),
        (1, 2, pytest.approx(2.0 / 3.0)),
    ]
    assert [(item.window, item.emitted_at, item.count) for item in events] == [
        (0, 3.0, 2),
        (1, 4.0, 2),
    ]


def test_streaming_decoder_can_expose_an_intermediate_closed_window(
    core: CoreEvaluator,
) -> None:
    windows = (DecodeWindow(0.0, 2.0), DecodeWindow(2.0, 4.0))
    with core.compile_decoders(
        (DecoderBinding(0, RateDecoder()),), node_count=1
    ) as compiled:
        with compiled.create_run(windows=windows) as run:
            run.consume((Spike(0.0, 0),))
            run.advance(2.0)
            intermediate = run.events()
            run.consume((Spike(2.0, 0),))
            values = run.finalize()
    assert [(item.window, item.emitted_at, item.count) for item in intermediate] == [
        (0, 2.0, 1)
    ]
    assert [(item.window, item.count) for item in values] == [(0, 1), (1, 1)]


def test_on_query_snapshots_precede_same_time_spikes_and_window_close(
    core: CoreEvaluator,
) -> None:
    bindings = (
        DecoderBinding(0, RateDecoder(emission=EmissionPolicy.ON_QUERY)),
        DecoderBinding(0, TTFSDecoder(emission=EmissionPolicy.ON_QUERY)),
        DecoderBinding(
            0, TemporalWeightDecoder(2.0, emission=EmissionPolicy.ON_QUERY)
        ),
    )
    queries = (
        DecoderQueryBinding(2, 0, 4.0),
        DecoderQueryBinding(1, 0, 0.0),
        DecoderQueryBinding(0, 0, 2.0),
        DecoderQueryBinding(2, 0, 0.0),
        DecoderQueryBinding(1, 0, 2.0),
        DecoderQueryBinding(2, 0, 2.0),
        DecoderQueryBinding(0, 0, 4.0),
    )
    with core.compile_decoders(bindings, node_count=1) as compiled:
        with compiled.create_run(
            t_start=0.0, t_end=4.0, queries=queries
        ) as run:
            run.consume((Spike(0.0, 0), Spike(2.0, 0), Spike(4.0, 0)))
            values = run.finalize()
            events = run.events()

    assert [
        (event.decoder, event.emitted_at, event.count, event.valid)
        for event in events
    ] == [
        (1, 0.0, 0, False),
        (2, 0.0, 0, True),
        (0, 2.0, 1, True),
        (1, 2.0, 1, True),
        (2, 2.0, 1, True),
        (0, 4.0, 2, True),
        (2, 4.0, 2, True),
    ]
    assert all(event.kind is DecodeEventKind.QUERY for event in events)
    assert events[0].value is None
    assert events[2].value == pytest.approx(0.5)
    assert events[3].value == pytest.approx(0.0)
    assert events[4].value == pytest.approx(1.0)
    assert events[5].value == pytest.approx(0.5)
    assert events[6].value == pytest.approx(1.0 + math.exp(-1.0))
    assert [(value.decoder, value.count) for value in values] == [
        (0, 2),
        (1, 1),
        (2, 2),
    ]


def test_delayed_decoder_advance_merges_queries_and_closes_chronologically(
    core: CoreEvaluator,
) -> None:
    bindings = (
        DecoderBinding(0, RateDecoder()),
        DecoderBinding(
            0, TemporalWeightDecoder(1.0, emission=EmissionPolicy.ON_QUERY)
        ),
    )
    schedule = (
        DecoderWindowBinding(0, 7, 0.0, 1.0),
        DecoderWindowBinding(1, 9, 0.0, 4.0),
    )
    queries = (DecoderQueryBinding(1, 9, 2.0),)
    with core.compile_decoders(bindings, node_count=1) as compiled:
        with compiled.create_run(schedule=schedule, queries=queries) as run:
            run.advance(4.0)
            events = run.events()
    assert [
        (event.decoder, event.window, event.kind, event.emitted_at)
        for event in events
    ] == [
        (0, 7, DecodeEventKind.FINAL, 1.0),
        (1, 9, DecodeEventKind.QUERY, 2.0),
    ]


def test_incremental_network_preserves_open_decoder_query_boundaries(
    core: CoreEvaluator,
) -> None:
    model = resolve_scalar_lif(parse_neuron(LIF), {"drive": 0.0})
    decoder_bindings = (
        DecoderBinding(0, RateDecoder(emission=EmissionPolicy.ON_QUERY)),
    )
    queries = (
        DecoderQueryBinding(0, 0, 2.0),
        DecoderQueryBinding(0, 0, 4.0),
    )
    with core.compile_mixed((model,)) as compiled:
        with core.compile_decoders(decoder_bindings, node_count=1) as bank:
            with bank.create_run(
                t_start=0.0, t_end=4.0, queries=queries
            ) as decoder_run:
                with compiled.create_incremental_run(
                    (-65.0,),
                    t_end=4.0,
                    output_capacity=0,
                    decoder_run=decoder_run,
                ) as run:
                    before = run.advance_until(2.0)
                    middle = run.advance_until(
                        4.0,
                        inputs=(MixedInputSpike(2.0, 0, 20.0),),
                    )
                    final = run.finish()

    assert before.decoded_events == ()
    assert [
        (event.kind, event.emitted_at, event.count, event.value)
        for event in middle.decoded_events
    ] == [(DecodeEventKind.QUERY, 2.0, 0, pytest.approx(0.0))]
    assert [
        (event.kind, event.emitted_at, event.count, event.value)
        for event in final.decoded_events
    ] == [(DecodeEventKind.QUERY, 4.0, 1, pytest.approx(0.25))]
    assert final.decoded[0].count == 1
    assert final.decoded[0].value == pytest.approx(0.25)


def test_graph_maps_exact_decoder_queries_per_output_port(
    core: CoreEvaluator,
) -> None:
    base = _codec_graph()
    graph = Graph(
        models=base.models,
        nodes=base.nodes,
        input_ports=base.input_ports,
        output_ports=(
            OutputPort(
                "rate", 0, RateDecoder(emission=EmissionPolicy.ON_QUERY)
            ),
            OutputPort(
                "first", 0, TTFSDecoder(emission=EmissionPolicy.ON_QUERY)
            ),
        ),
    ).resolve()
    arguments = dict(
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
        decode_queries={
            "rate": (DecodeQuery(0, 2.0), DecodeQuery(0, 4.0)),
            "first": (DecodeQuery(0, 0.0), DecodeQuery(0, 2.0)),
        },
        output_capacity=0,
    )
    result = graph.run(core, **arguments)
    assert [
        (event.port, event.emitted_at, event.count, event.valid, event.value)
        for event in result.decoded_events
    ] == [
        ("first", 0.0, 0, False, None),
        ("first", 2.0, 1, True, pytest.approx(0.0)),
        ("rate", 2.0, 1, True, pytest.approx(0.5)),
        ("rate", 4.0, 2, True, pytest.approx(0.5)),
    ]
    with graph.compile(core) as compiled:
        assert compiled.run(**arguments) == result


def test_sliding_rate_query_is_explicitly_reserved() -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        output_ports=(
            OutputPort(
                "rate",
                0,
                RateDecoder(
                    RateMode.SLIDING,
                    width=2.0,
                    emission=EmissionPolicy.ON_QUERY,
                ),
            ),
        ),
    )
    with pytest.raises(ResolutionError, match="timestamp retention"):
        graph.resolve()


def test_graph_streams_decoding_with_raw_output_disabled(core: CoreEvaluator) -> None:
    resolved = _codec_graph().resolve()
    arguments = dict(
        scalar_inputs=(ScalarInput(0.0, 4.0, "burst", 1.0),),
        t_end=4.0,
        output_capacity=0,
        decoder_event_capacity=0,
        recording=RecordingConfig(
            kinds=frozenset({TraceKind.SPIKE}),
            capture_state=False,
            capacity=4,
        ),
        inspections=(StateInspectionRequest(1.5, 0),),
    )
    result = resolved.run(core, **arguments)
    assert result.core.spikes == ()
    assert result.outputs == ()
    assert result.core.stats.output_spikes == 2
    assert [record.kind for record in result.trace] == [
        TraceKind.SPIKE,
        TraceKind.SPIKE,
    ]
    assert result.decoded_events == ()
    assert result.inspections[0].values == (-65.0,)
    assert result.inspections[0].clamped is True
    assert {item.port: item.value for item in result.decoded} == pytest.approx(
        {"rate": 0.5, "first": 0.0, "weighted": 1.0 + math.exp(-1.0)}
    )

    with resolved.compile(core) as compiled:
        first = compiled.run(**arguments)
        replay = compiled.run(**arguments)
    assert replay == first
    assert first.decoded == result.decoded


def test_held_current_runs_as_exact_drive_boundaries(core: CoreEvaluator) -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "current",
                0,
                InputMode.DRIVE,
                "drive",
                HeldCurrentEncoder(gain=20.0, baseline=0.0),
            ),
        ),
        output_ports=(OutputPort("spikes", 0),),
    )
    result = graph.resolve().run(
        core,
        scalar_inputs=(ScalarInput(0.0, 14.0, "current", 1.0),),
        t_end=15.0,
    )
    assert [item.t for item in result.outputs] == pytest.approx([10.0 * math.log(4.0)])


def test_raw_event_cannot_bypass_a_configured_scalar_encoder(
    core: CoreEvaluator,
) -> None:
    from lacuna import SpikeInput

    with pytest.raises(ResolutionError, match="requires a scalar presentation"):
        _codec_graph().resolve().run(
            core,
            spike_inputs=(SpikeInput(0.0, "burst", 20.0),),
            t_end=1.0,
        )


def test_drive_ports_cannot_share_a_last_writer_parameter() -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "first",
                0,
                InputMode.DRIVE,
                "drive",
                HeldCurrentEncoder(),
            ),
            InputPort(
                "second",
                0,
                InputMode.DRIVE,
                "drive",
                HeldCurrentEncoder(),
            ),
        ),
    )
    with pytest.raises(ResolutionError, match="duplicates the node/parameter"):
        graph.resolve()


def test_codec_hashes_are_verified_independently() -> None:
    document = json.loads(_codec_graph().to_text())
    document["input_ports"][0]["encoder"]["duration"] = float(4.0).hex()
    with pytest.raises(ResolutionError, match="encoder hash mismatch"):
        Graph.from_text(json.dumps(document))


def test_codec_port_mode_mismatch_fails_during_resolution() -> None:
    graph = Graph(
        models=(GraphModel("lif", LIF),),
        nodes=(GraphNode(0, "lif", -65.0, {"drive": 0.0}),),
        input_ports=(
            InputPort(
                "bad", 0, InputMode.DRIVE, "drive", RegularRateEncoder(0.0, 1.0)
            ),
        ),
    )
    with pytest.raises(ResolutionError, match="requires a SPIKE port"):
        graph.resolve()
