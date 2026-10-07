"""Half-precision graph execution, replay, and representability limits."""

import math
from pathlib import Path

import pytest

from lacuna import AdEx, AugmentedState, CoreEvaluator, Graph, QIF
from lacuna.errors import CoreError, PrecisionResolutionError
from lacuna.target_lowering import resolve_target_graph
from tests.python.test_execution_plan_matrix import PLAN_CASES, _matrix_inputs


@pytest.fixture(scope="module")
def half_core():
    if not list(Path("build-float16").glob("liblacuna_half_host.*")):
        pytest.skip("native half runtime is not built")
    return CoreEvaluator(precision="float16")


@pytest.mark.parametrize("name,factory", PLAN_CASES[:-1],
                         ids=[name for name, _ in PLAN_CASES[:-1]])
def test_diverse_half_graphs_and_image_replay(half_core, name, factory):
    target = resolve_target_graph(factory(), half_core)
    plan = target.execution_plan()
    inputs = _matrix_inputs(target)
    with half_core.compile_execution_plan(plan) as graph:
        if name == "folded-alpha-program-deposit":
            # The existing extremum guard cannot certify this cancellation in half.
            with pytest.raises(CoreError, match="root finder did not converge"):
                graph.run(target.initial_values, inputs=inputs, t_end=8.0)
            return
        expected = graph.run(target.initial_values, inputs=inputs, t_end=8.0)
        assert expected.spikes
        assert all(left.t <= right.t for left, right in
                   zip(expected.spikes, expected.spikes[1:]))
        assert all(half_core.precision.round_time(event.t) == event.t
                   for event in expected.spikes)
        assert graph.run(target.initial_values, inputs=inputs, t_end=8.0) == expected
        image = graph.to_bytes()
    with half_core.load_compiled_graph_image(image) as loaded:
        assert loaded.to_bytes() == image
        assert loaded.run(target.initial_values, inputs=inputs, t_end=8.0) == expected


def test_half_rejects_unrepresentable_learning_parameters(half_core):
    factory = dict(PLAN_CASES)["mixed-static-pair-triplet-modulated"]
    with pytest.raises(PrecisionResolutionError, match="a2_plus underflows"):
        resolve_target_graph(factory(), half_core)


@pytest.mark.parametrize("start", (0.0, 0.67529296875, 4.0, 16.0))
def test_half_qif_crossing_against_closed_form(start, half_core):
    model = QIF()
    target = resolve_target_graph(
        Graph(models=(model.model,), nodes=(model.node(0),)), half_core,
    )
    bound = target.models[0]
    prediction = half_core.predict_stepped(
        bound, AugmentedState((0.0,), start), start + 2.0,
    )
    reference = start + math.pi / 4.0
    # This fixture has a closed-form crossing, not a global integrator guarantee.
    spacing = 2.0 ** (math.floor(math.log2(reference)) - 10)
    assert prediction.t_spike == pytest.approx(reference, abs=max(0.002, 2 * spacing))
    assert prediction.diagnostics.last_step <= bound.numerical.maximum_step
    advanced, diagnostics = half_core.advance_stepped(
        bound, AugmentedState((0.0,), start), prediction.t_spike,
    )
    assert advanced.t_last == prediction.t_spike
    assert advanced.values[0] == pytest.approx(1.0, abs=0.04)
    assert diagnostics.last_step <= bound.numerical.maximum_step


def test_half_stepper_does_not_increase_user_maximum_step_to_fit_clock(half_core):
    model = QIF()
    target = resolve_target_graph(
        Graph(models=(model.model,), nodes=(model.node(0),)), half_core,
    )
    bound = target.models[0]
    assert bound.numerical.maximum_step == 0.25
    with pytest.raises(CoreError):
        half_core.predict_stepped(bound, AugmentedState((0.0,), 1024.0), 1026.0)


def test_half_adex_fires_with_native_adaptation(half_core):
    model = AdEx(drive=480.0)
    graph = Graph(models=(model.model,), nodes=(model.node(0),))
    target = resolve_target_graph(graph, half_core)
    with half_core.compile_execution_plan(target.execution_plan()) as compiled:
        result = compiled.run(target.initial_values, t_end=30.0)
    reference_core = CoreEvaluator()
    reference = graph.resolve()
    with reference_core.compile_execution_plan(reference.execution_plan()) as compiled:
        expected = compiled.run(reference.initial_values, t_end=30.0)
    assert len(result.spikes) == len(expected.spikes) == 2
    assert [event.t for event in result.spikes] == pytest.approx(
        [event.t for event in expected.spikes], abs=0.5,
    )
