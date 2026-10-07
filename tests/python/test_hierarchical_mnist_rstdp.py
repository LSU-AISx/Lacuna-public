from lacuna import Engine, ModulatedSTDP, PairSTDP
from lacuna.experiments import (
    HierarchicalMNISTConfig,
    SpatialShape,
    build_hierarchical_mnist_classifier,
    freeze_hidden_plasticity,
    hierarchy_shapes,
    local_many_to_one_pairs,
    pretrain_hidden_stages,
    train_classifier,
)
from lacuna.ffi import CoreEvaluator
from lacuna.graph import NeuronPolarity


def _small_config() -> HierarchicalMNISTConfig:
    return HierarchicalMNISTConfig(
        feature_count=12 * 12,
        class_count=3,
        input_rows=12,
        input_columns=12,
        hidden_channels=(2, 3, 4),
        kernels=(3, 3, 2),
        strides=(2, 2, 1),
        hidden_weight_ranges=((0.5, 0.6), (0.5, 0.6), (0.5, 0.6)),
        hidden_thresholds=(-55.0, -60.0, -60.0),
        reservoir_excitatory=6,
        reservoir_inhibitory=2,
        reservoir_probability=0.25,
        presentation_duration=10.0,
        probe_delay=3.0,
        reward_delay=1.0,
        sample_duration=30.0,
        pixel_max_rate=1.0,
    )


def test_local_many_to_one_mapping_covers_full_receptive_field_per_channel() -> None:
    source = SpatialShape(4, 4, 2)
    target = SpatialShape(2, 2, 3)
    pairs = local_many_to_one_pairs(source, target, kernel=2, stride=2)

    assert len(pairs) == 2 * 2 * 3 * 2 * 2 * 2
    first_target_sources = tuple(source for source, target in pairs if target == 0)
    assert first_target_sources == (0, 1, 2, 3, 8, 9, 10, 11)
    assert {target for _, target in pairs} == set(range(target.size))


def test_hierarchy_is_delta_lif_competitive_and_dimension_reducing() -> None:
    config = _small_config()
    classifier = build_hierarchical_mnist_classifier(config, seed=4)
    graph = classifier.network.graph
    nodes = {node.id: node for node in graph.nodes}

    assert hierarchy_shapes(config) == (
        SpatialShape(12, 12, 1),
        SpatialShape(5, 5, 2),
        SpatialShape(2, 2, 3),
        SpatialShape(1, 1, 4),
    )
    assert all(edge.synapse is None for edge in graph.edges)
    assert all(model.source.startswith("neuron StandardLIF") for model in graph.models)
    assert all(
        isinstance(graph.edges[edge].plasticity, PairSTDP)
        for stage in classifier.stage_edge_ids
        for edge in stage
    )
    first_target_delays = {
        graph.edges[edge].delay
        for edge in classifier.stage_edge_ids[0]
        if graph.edges[edge].post == graph.edges[classifier.stage_edge_ids[0][0]].post
    }
    assert len(first_target_delays) > 1
    for lateral in classifier.inhibition_edge_ids:
        midpoint = len(lateral) // 2
        assert all(
            nodes[graph.edges[edge].pre].polarity is NeuronPolarity.EXCITATORY
            for edge in lateral[:midpoint]
        )
        assert all(
            nodes[graph.edges[edge].pre].polarity is NeuronPolarity.INHIBITORY
            for edge in lateral[midpoint:]
        )
    assert all(
        isinstance(graph.edges[edge].plasticity, ModulatedSTDP)
        for projection in classifier.class_edges
        for edge in projection
    )
    assert tuple(len(port.edges) for port in graph.modulator_ports) == tuple(
        len(edges) for edges in classifier.class_edges
    )


def test_small_hierarchy_executes_hidden_and_readout_learning(core: CoreEvaluator) -> None:
    classifier = build_hierarchical_mnist_classifier(_small_config(), seed=6)
    before = tuple(edge.weight for edge in classifier.network.graph.edges)
    training = train_classifier(
        classifier,
        (tuple(255 for _ in range(12 * 12)),),
        (0,),
        engine=Engine(core._lib._name),
        seed=8,
    )
    after = tuple(edge.weight for edge in training.classifier.network.graph.edges)

    assert len(after) == len(before)
    first_stage = classifier.stage_edge_ids[0]
    assert any(after[edge] != before[edge] for edge in first_stage)
    assert training.metrics.samples == 1


def test_hidden_stages_can_be_pretrained_sequentially_then_frozen(
    core: CoreEvaluator,
) -> None:
    classifier = build_hierarchical_mnist_classifier(_small_config(), seed=9)
    before = tuple(edge.weight for edge in classifier.network.graph.edges)
    pretrained = pretrain_hidden_stages(
        classifier,
        (tuple(255 for _ in range(12 * 12)),),
        engine=Engine(core._lib._name),
        seed=10,
        stage_indices=(0,),
    )
    after = tuple(edge.weight for edge in pretrained.network.graph.edges)

    assert any(after[edge] != before[edge] for edge in classifier.stage_edge_ids[0])
    assert all(
        after[edge] == before[edge]
        for stage in classifier.stage_edge_ids[1:]
        for edge in stage
    )
    assert all(
        isinstance(pretrained.network.graph.edges[edge].plasticity, PairSTDP)
        for stage in pretrained.stage_edge_ids
        for edge in stage
    )
    frozen = freeze_hidden_plasticity(pretrained)
    assert all(
        frozen.network.graph.edges[edge].plasticity is None
        for stage in frozen.stage_edge_ids
        for edge in stage
    )
    assert frozen.network.graph.modulator_ports
