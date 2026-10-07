from __future__ import annotations

import math

import pytest

from lacuna import (
    AdEx,
    AdaptiveLIF,
    Graph,
    GraphEdge,
    GraphNode,
    GraphSynapse,
    LIF,
    NumericalConfig,
    OutputPort,
    QIF,
)
from lacuna.errors import CapabilityError, ResolutionError
from lacuna.ffi import CoreEvaluator
from lacuna.ir import (
    DispatchForm,
    ResolvedAdaptiveLIF,
    ResolvedPerEdgeLIF,
    ResolvedScalarLIF,
    ResolvedSteppedNeuron,
)


EXPONENTIAL_CURRENT = """
synapse ExponentialCurrent {
    params { tau_syn : positive = 5.0 }
    state { s }
    dynamics { ds/dt = -s/tau_syn }
    on_spike { s <- s + w }
    output { current = s }
}
"""


def test_lif_builds_graph_records_and_resolves_analytically() -> None:
    family = LIF(name="excitatory", drive=20.0)
    node = family.node(7, v_rest=-60.0)

    assert family.model.id == "excitatory"
    assert node == GraphNode(7, "excitatory", -60.0, {"v_rest": -60.0})
    assert family.authored_model.name == "StandardLIF"

    resolved = family.resolve()
    assert isinstance(resolved, ResolvedScalarLIF)
    assert resolved.dispatch is DispatchForm.CLOSED_FORM
    assert resolved.a == pytest.approx(-1.0 / 20.0)
    assert resolved.asymptote == pytest.approx(-45.0)


def test_standard_qif_runs_through_graph_without_authored_equations(
    core: CoreEvaluator,
) -> None:
    family = QIF(name="quadratic")
    graph = Graph(
        models=(family.model,),
        nodes=(family.node(0),),
        output_ports=(OutputPort("spikes", 0),),
    )

    resolved = graph.resolve()
    assert isinstance(resolved.models[0], ResolvedSteppedNeuron)
    result = resolved.run(core, t_end=1.7)
    assert [item.t for item in result.core.spikes] == pytest.approx(
        [math.pi / 4.0, math.pi / 2.0], abs=3e-8
    )
    assert [item.port for item in result.outputs] == ["spikes", "spikes"]


def test_adaptive_lif_selects_exact_adaptation_capability() -> None:
    family = AdaptiveLIF(drive=25.0, adaptation_increment=2.0)
    resolved = family.resolve()

    assert family.default_initial == (-65.0, 0.0)
    assert isinstance(resolved, ResolvedAdaptiveLIF)
    assert resolved.dispatch is DispatchForm.ROOT_FIND
    assert resolved.coupling == pytest.approx(-1.0 / 20.0)
    assert resolved.adaptation_jump == 2.0


def test_adex_supports_direct_numerical_configuration_and_graph_execution(
    core: CoreEvaluator,
) -> None:
    family = AdEx(drive=500.0)
    numerical = NumericalConfig(maximum_step=0.1)
    direct = family.resolve(numerical=numerical)

    assert isinstance(direct, ResolvedSteppedNeuron)
    assert direct.numerical == numerical
    graph = Graph(models=(family.model,), nodes=(family.node(0),))
    result = graph.resolve().run(core, t_end=15.0)
    assert [item.t for item in result.core.spikes] == pytest.approx(
        [14.0920641071], abs=5e-7
    )


def test_receptor_enabled_lif_preserves_per_edge_synapse_path() -> None:
    source = LIF(name="source")
    target = LIF(name="target", synaptic_input=True)
    graph = Graph(
        models=(source.model, target.model),
        synapses=(GraphSynapse("exp", EXPONENTIAL_CURRENT),),
        nodes=(source.node(0), target.node(1)),
        edges=(
            GraphEdge(
                0,
                0,
                1,
                25.0,
                synapse="exp",
                receptor=target.receptor_name,
                output="current",
            ),
        ),
    )

    resolved = graph.resolve()
    assert isinstance(resolved.models[0], ResolvedScalarLIF)
    assert isinstance(resolved.models[1], ResolvedPerEdgeLIF)
    assert resolved.models[1].state_names[0] == "v"
    assert Graph.from_text(graph.to_text()).to_text() == graph.to_text()

    with pytest.raises(CapabilityError, match="must be resolved in a Graph"):
        target.resolve()


@pytest.mark.parametrize(
    ("construction", "message"),
    [
        (lambda: LIF(tau_m=0.0), "tau_m must be positive"),
        (
            lambda: AdaptiveLIF(tau_adaptation=20.0),
            "tau_adaptation must differ",
        ),
        (lambda: AdEx(v_spike=-60.0), "v_reset must be strictly below"),
        (lambda: QIF(v_reset=1.0), "v_reset must be strictly below"),
    ],
)
def test_invalid_standard_model_defaults_fail_at_construction(
    construction, message: str
) -> None:
    with pytest.raises(ResolutionError, match=message):
        construction()


def test_node_overrides_are_validated_and_remain_sparse() -> None:
    family = LIF()
    node = family.node(3, bindings={"drive": 10}, v_rest=-62)
    assert node.bindings == {"drive": 10.0, "v_rest": -62.0}
    assert node.initial == -62.0

    with pytest.raises(ResolutionError, match="supplied twice"):
        family.node(3, bindings={"drive": 10}, drive=12)
    with pytest.raises(ResolutionError, match="unknown standard neuron"):
        family.node(3, typo=1)
    with pytest.raises(ResolutionError, match="tau_m must be positive"):
        family.node(3, tau_m=0)
