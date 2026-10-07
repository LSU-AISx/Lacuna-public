from dataclasses import replace
import math

import pytest

from lacuna import LIF, ModulatedSTDP, NetworkBuilder, NeuronPolarity

from optimization.mnist_neuron_search import (
    _add_winner_specific_inhibition,
    _project_kernel_sum,
    _shared_pool_reward,
    base_candidates,
    build_candidate,
    normalize_shared_pool_kernels,
)


def test_winner_specific_inhibition_excludes_the_winning_channel() -> None:
    builder = NetworkBuilder("winner-specific-competition")
    features = builder.population("features", 3, LIF())

    inhibitor_nodes = _add_winner_specific_inhibition(
        builder,
        features,
        side=1,
        channels=3,
        magnitude=10.0,
        trigger=12.0,
        delay=0.0,
    )
    network = builder.build()
    graph = network.graph

    assert len(inhibitor_nodes) == 3
    assert all(
        graph.nodes[node].polarity is NeuronPolarity.INHIBITORY
        for node in inhibitor_nodes
    )

    trigger_edges = graph.edges[:3]
    assert {(edge.pre, edge.post) for edge in trigger_edges} == {
        (features.node_ids[channel], inhibitor_nodes[channel])
        for channel in range(3)
    }

    feedback_edges = graph.edges[3:]
    assert len(feedback_edges) == 6
    assert {(edge.pre, edge.post) for edge in feedback_edges} == {
        (inhibitor_nodes[winner], features.node_ids[competitor])
        for winner in range(3)
        for competitor in range(3)
        if competitor != winner
    }
    assert all(
        edge.post != features.node_ids[edge.pre - inhibitor_nodes[0]]
        for edge in feedback_edges
    )


def test_bounded_kernel_projection_preserves_exact_sum() -> None:
    projected = _project_kernel_sum(
        (0.001, 0.1, 100.0), 1.2, lower=0.1, upper=0.7
    )

    assert sum(projected) == pytest.approx(1.2)
    assert min(projected) >= 0.1
    assert max(projected) <= 0.7


def test_larger_shared_kernels_are_normalized_per_output_channel() -> None:
    config = replace(
        base_candidates(19)[0],
        architecture="pool2",
        pool_kernel=6,
        pool_side=12,
        shared_pool=True,
        shared_pool_stride=2,
        shared_pool_channels=2,
        shared_pool_plasticity="pair",
        shared_pool_normalize_initial_kernels=True,
        shared_pool_kernel_sum=12.0,
        shared_pool_normalization_interval=10,
        shared_pool_normalization_sum=9.0,
    )
    candidate = build_candidate(config)
    initial = tuple(edge.weight for edge in candidate.network.graph.edges)
    normalized = normalize_shared_pool_kernels(candidate, config, initial)
    groups: dict[int, list[int]] = {}
    for edge_id in candidate.shared_pool_edges:
        group = candidate.network.graph.edges[edge_id].weight_group
        assert group is not None
        groups.setdefault(group, []).append(edge_id)
    ordered = sorted(groups)

    assert len(candidate.feature_nodes) == 12 * 12 * 2
    assert len(ordered) == 6 * 6 * 2
    for channel in range(2):
        channel_groups = ordered[channel * 36 : (channel + 1) * 36]
        assert sum(normalized[groups[group][0]] for group in channel_groups) == (
            pytest.approx(9.0)
        )
    for members in groups.values():
        assert len({normalized[edge_id] for edge_id in members}) == 1


def test_shared_pool_modulated_stdp_has_one_scoped_reward_port() -> None:
    config = replace(
        base_candidates(23)[0],
        architecture="pool2",
        pool_side=14,
        shared_pool=True,
        shared_pool_channels=2,
        shared_pool_plasticity="modulated",
        shared_pool_trace_tau=5.0,
        shared_pool_eligibility_tau_plus=7_210.0,
        shared_pool_eligibility_tau_minus=3_610.0,
    )
    candidate = build_candidate(config)
    port = next(
        item
        for item in candidate.network.graph.modulator_ports
        if item.id == candidate.shared_pool_reward_port
    )

    assert port.edges == candidate.shared_pool_edges
    assert candidate.shared_pool_reward_port == "shared_pool_reward"
    for edge_id in candidate.shared_pool_edges:
        rule = candidate.network.graph.edges[edge_id].plasticity
        assert isinstance(rule, ModulatedSTDP)
        assert rule.tau_pre == 5.0
        assert rule.tau_post == 5.0
        assert rule.tau_eligibility_plus == 7_210.0
        assert rule.tau_eligibility_minus == 3_610.0
        assert rule.consume_on_modulation


def test_shared_pool_reward_supports_correctness_and_signed_margin() -> None:
    base = replace(
        base_candidates(29)[0],
        shared_pool_reward_mode="correctness",
    )
    evidence = (0.0, 0.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    assert _shared_pool_reward(base, evidence, 2, 2) == 1.0
    assert _shared_pool_reward(base, evidence, 3, 2) == -1.0
    centered = replace(
        base,
        shared_pool_correct_reward=0.9,
        shared_pool_incorrect_reward=-0.1,
    )
    assert _shared_pool_reward(centered, evidence, 2, 2) == 0.9
    assert _shared_pool_reward(centered, evidence, 3, 2) == -0.1
    margin = replace(
        base,
        shared_pool_reward_mode="signed_margin",
        shared_pool_reward_scale=2.0,
    )
    assert _shared_pool_reward(margin, evidence, 2, 2) == pytest.approx(
        math.tanh(1.5)
    )
    assert _shared_pool_reward(margin, evidence, 3, 2) < 0.0
