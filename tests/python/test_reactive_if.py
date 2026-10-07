from __future__ import annotations

from pathlib import Path

import pytest

from lacuna import (
    CoreEvaluator,
    Graph,
    GraphEdge,
    InputMode,
    InputPort,
    IntegrateAndFire,
    OutputPort,
    ReactiveMode,
    ResolvedReactiveIF,
    SpikeInput,
)


def _core() -> CoreEvaluator:
    return CoreEvaluator(next(Path("build").glob("liblacuna_core.*")))


def _single(model: IntegrateAndFire) -> Graph:
    return Graph(
        models=(model.model,),
        nodes=(model.node(0),),
        input_ports=(InputPort("input", 0, InputMode.SPIKE),),
        output_ports=(OutputPort("output", 0),),
    )


def test_standard_if_resolves_to_reactive_capability() -> None:
    held = IntegrateAndFire().resolve()
    leaked = IntegrateAndFire(leak=True).resolve()

    assert isinstance(held, ResolvedReactiveIF)
    assert held.reactive_mode is ReactiveMode.HOLD
    assert leaked.reactive_mode is ReactiveMode.RESET_BEFORE_DEPOSIT
    assert held.dispatch.value == "REACTIVE"


def test_held_charge_accumulates_across_event_timestamps() -> None:
    result = _single(IntegrateAndFire()).resolve().run(
        _core(),
        spike_inputs=(
            SpikeInput(1.0, "input", 0.6),
            SpikeInput(2.0, "input", 0.6),
        ),
        t_end=3.0,
    )

    assert [spike.t for spike in result.outputs] == [2.0]
    assert result.core.states[0].values == (0.0,)


def test_timestamp_leak_resets_before_each_distinct_event_batch() -> None:
    result = _single(IntegrateAndFire(leak=True)).resolve().run(
        _core(),
        spike_inputs=(
            SpikeInput(1.0, "input", 0.6),
            SpikeInput(2.0, "input", 0.6),
        ),
        t_end=3.0,
    )

    assert result.outputs == ()
    assert result.core.states[0].values == pytest.approx((0.6,))


def test_timestamp_leak_sums_all_deposits_at_the_same_time() -> None:
    result = _single(IntegrateAndFire(leak=True)).resolve().run(
        _core(),
        spike_inputs=(
            SpikeInput(1.0, "input", 0.6),
            SpikeInput(1.0, "input", 0.6),
        ),
        t_end=2.0,
    )

    assert [spike.t for spike in result.outputs] == [1.0]

def test_reactive_if_propagates_spikes_with_edge_delay() -> None:
    model = IntegrateAndFire()
    graph = Graph(
        models=(model.model,),
        nodes=(model.node(0), model.node(1)),
        edges=(GraphEdge(0, 0, 1, 1.0, 2.0),),
        input_ports=(InputPort("input", 0, InputMode.SPIKE),),
        output_ports=(OutputPort("output", 1),),
    )
    result = graph.resolve().run(
        _core(),
        spike_inputs=(SpikeInput(1.0, "input", 1.0),),
        t_end=4.0,
    )

    assert [spike.t for spike in result.outputs] == [3.0]


def test_reactive_if_allows_risp_style_negative_threshold() -> None:
    model = IntegrateAndFire(threshold=-1.0, reset=0.0)
    result = _single(model).resolve().run(
        _core(),
        spike_inputs=(SpikeInput(1.0, "input", 0.0),),
        t_end=2.0,
    )

    assert [spike.t for spike in result.outputs] == [1.0]
