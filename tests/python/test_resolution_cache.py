from __future__ import annotations

from collections import UserDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import multiprocessing as mp
from threading import Event, Thread

import pytest

from lacuna import (
    AdaptiveLIF, AdEx, AlphaCurrent, CoreEvaluator, Delta, ExponentialCurrent,
    Graph, GraphEdge, GraphModel, GraphNode, IntegrateAndFire, LIF,
    MixedInputSpike, NetworkBuilder, NumericalConfig, PairSTDP,
    PerEdgeSynapseInstance, clear_resolution_cache, configure_resolution_cache,
    parse_neuron, parse_synapse, resolution_cache_disabled, resolution_cache_info,
    resolve_per_edge_lif, resolve_scalar_lif, resolve_stepped_neuron,
)
from lacuna.errors import ResolutionError
from lacuna.resolution_cache import _CACHE, _cached_resolution


@pytest.fixture(autouse=True)
def isolated_cache():
    original = resolution_cache_info()
    configure_resolution_cache(enabled=True, max_entries=4096, max_bytes=64 * 1024 * 1024)
    yield
    configure_resolution_cache(enabled=original.enabled, max_entries=original.max_entries,
                               max_bytes=original.max_bytes)


def test_cache_returns_independent_parsed_and_resolved_records():
    model = parse_neuron(LIF().source)
    expected = resolve_scalar_lif(model, {"drive": 18.0})
    hit = resolve_scalar_lif(model, {"drive": 18.0})
    assert hit == expected and hit is not expected
    hit.bindings["drive"] = 999.0
    hit.binding_dag.roots["a"] = 999
    assert resolve_scalar_lif(model, {"drive": 18.0}) == expected
    model.dynamics["v"] = "0.0"
    assert parse_neuron(LIF().source).dynamics["v"] != "0.0"
    assert resolution_cache_info().hits >= 3


def test_changed_bindings_and_model_defaults_cannot_share_numeric_resolution():
    model = parse_neuron(LIF().source)
    bindings = {"drive": 18.0}
    original = resolve_scalar_lif(model, bindings)
    bindings["drive"] = 19.0
    changed = resolve_scalar_lif(model, bindings)
    assert changed.resolution_key == original.resolution_key
    assert changed.b != original.b
    changed_default = replace(model, parameters=tuple(
        replace(p, default=19.0) if p.name == "drive" else p for p in model.parameters
    ))
    assert resolve_scalar_lif(changed_default).b == changed.b
    assert resolve_scalar_lif(model).b != changed.b


def test_float_bits_and_binding_order_are_preserved():
    model = parse_neuron(LIF().source)
    positive_zero = resolve_scalar_lif(model, {"drive": 0.0})
    negative_zero = resolve_scalar_lif(model, {"drive": -0.0})
    assert positive_zero.bindings["drive"].hex() == "0x0.0p+0"
    assert negative_zero.bindings["drive"].hex() == "-0x0.0p+0"
    first = resolve_scalar_lif(model, {"drive": 18., "tau_m": 20.})
    second = resolve_scalar_lif(model, {"tau_m": 20., "drive": 18.})
    assert tuple(first.bindings)[:2] == ("drive", "tau_m")
    assert tuple(second.bindings)[:2] == ("tau_m", "drive")


def test_numerical_configuration_is_part_of_the_key():
    model = parse_neuron(AdEx().source)
    first = resolve_stepped_neuron(model, numerical=NumericalConfig(maximum_step=.25))
    second = resolve_stepped_neuron(model, numerical=NumericalConfig(maximum_step=.125))
    assert first.numerical.maximum_step == .25
    assert second.numerical.maximum_step == .125
    with resolution_cache_disabled():
        assert second == resolve_stepped_neuron(model, numerical=second.numerical)


def test_per_edge_key_covers_kernel_bindings_identifiers_and_initial_state():
    neuron = parse_neuron(LIF(synaptic_input=True).source)
    instance = PerEdgeSynapseInstance(
        7, parse_synapse(AlphaCurrent(5.).source), "i_syn", "current", {"tau": 5.},
    )
    first = resolve_per_edge_lif(neuron, (instance,))
    # Moving to the equal-decay regime must recompute the kernel.
    instance.bindings["tau"] = 20.
    equal_decay = resolve_per_edge_lif(neuron, (instance,))
    assert first.synaptic_decays != equal_decay.synaptic_decays
    changed = replace(instance, edge_id=11, initial=(1., 2.))
    actual = resolve_per_edge_lif(neuron, (changed,))
    with resolution_cache_disabled():
        expected = resolve_per_edge_lif(neuron, (changed,))
    assert actual == expected
    assert tuple(actual.edge_deposits) == (11,)
    assert actual.group_initials != equal_decay.group_initials
    actual.edge_deposits[11] = (999, 999.)
    assert resolve_per_edge_lif(neuron, (changed,)) == expected


def test_graph_validation_still_runs_after_a_resolution_cache_hit():
    graph = Graph(models=(GraphModel("lif", LIF().source),),
                  nodes=(GraphNode(0, "lif", -65., {"drive": 18.}),))
    expected = graph.resolve()
    assert graph.resolve() == expected
    with pytest.raises(ResolutionError, match="below threshold"):
        replace(graph, nodes=(replace(graph.nodes[0], initial=0.),)).resolve()
    with pytest.raises(ResolutionError, match="invalid node"):
        replace(graph, edges=(GraphEdge(0, 0, 99, 1.),)).resolve()
    graph.nodes[0].bindings["tau_m"] = -1.
    with pytest.raises(ResolutionError, match="positive"):
        graph.resolve()


def test_failures_are_not_cached_and_custom_mappings_bypass():
    model = parse_neuron(LIF().source)
    count = resolution_cache_info().entries
    for _ in range(2):
        with pytest.raises(ResolutionError, match="finite"):
            resolve_scalar_lif(model, {"drive": float("nan")})
    assert resolution_cache_info().entries == count
    resolve_scalar_lif(model, UserDict({"drive": 18.}))
    assert resolution_cache_info().entries == count


def test_entry_and_memory_limits_and_eviction_preserve_values():
    model = parse_neuron(LIF().source)
    clear_resolution_cache()
    expected = resolve_scalar_lif(model, {"drive": 18.})
    budget = resolution_cache_info().estimated_bytes
    assert budget > 0
    configure_resolution_cache(max_entries=1, max_bytes=budget * 2)
    for value in (18., 19., 20., 18.):
        result = resolve_scalar_lif(model, {"drive": value})
        assert resolution_cache_info().entries == 1
        assert resolution_cache_info().estimated_bytes <= budget * 2
    assert result == expected
    configure_resolution_cache(max_entries=100, max_bytes=budget)
    for value in range(18, 28):
        resolve_scalar_lif(model, {"drive": float(value)})
        assert resolution_cache_info().estimated_bytes <= budget
    configure_resolution_cache(max_bytes=1)
    assert resolve_scalar_lif(model, {"drive": 18.}) == expected
    assert resolution_cache_info().entries == 0


def test_disabled_cache_and_nested_bypass_leave_cache_untouched():
    model = parse_neuron(LIF().source)
    expected = resolve_scalar_lif(model, {"drive": 18.})
    before = resolution_cache_info()
    with resolution_cache_disabled():
        with resolution_cache_disabled():
            assert resolve_scalar_lif(model, {"drive": 18.}) == expected
        assert resolve_scalar_lif(model, {"drive": 18.}) == expected
    assert resolution_cache_info() == before
    resolve_scalar_lif(model, {"drive": 18.})
    assert resolution_cache_info().hits == before.hits + 1
    configure_resolution_cache(enabled=False)
    assert resolve_scalar_lif(model, {"drive": 18.}) == expected
    assert resolution_cache_info().entries == resolution_cache_info().misses == 0


@pytest.mark.parametrize("kwargs", ({"enabled": 1}, {"max_entries": True},
                                  {"max_entries": -1}, {"max_bytes": 1.5}))
def test_bad_configuration_is_rejected_without_clearing(kwargs):
    LIF().resolve()
    before = resolution_cache_info()
    with pytest.raises(ValueError):
        configure_resolution_cache(**kwargs)
    assert resolution_cache_info() == before


def test_threads_get_independent_values_and_context_local_bypass():
    model = parse_neuron(LIF().source)
    with resolution_cache_disabled():
        expected = resolve_scalar_lif(model, {"drive": 18.})
        before = resolution_cache_info()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = tuple(pool.map(lambda _: resolve_scalar_lif(model, {"drive": 18.}), range(16)))
        assert all(result == expected for result in results)
        assert len({id(result) for result in results}) == len(results)
        assert resolution_cache_info().entries > before.entries
    results[0].bindings["drive"] = 999.
    assert all(result.bindings["drive"] == 18. for result in results[1:])


def test_clear_during_a_miss_does_not_repopulate_cache():
    entered, release = Event(), Event()

    @_cached_resolution
    def operation(value):
        entered.set()
        assert release.wait(10.)
        return {"value": value}

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(operation, 18.)
        try:
            assert entered.wait(10.)
            clear_resolution_cache()
        finally:
            release.set()
        assert pending.result(timeout=10.) == {"value": 18.}
    assert resolution_cache_info().entries == 0


def _fork_probe(connection):
    before = resolution_cache_info()
    resolved = LIF(drive=18.).resolve()
    connection.send((before.entries, before.hits, resolved.bindings["drive"]))
    connection.close()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires POSIX fork")
def test_fork_resets_entries_and_replaces_a_lock_held_by_another_thread():
    LIF().resolve()
    before = resolution_cache_info()
    entered, release = Event(), Event()

    def hold_lock():
        with _CACHE.lock:
            entered.set()
            release.wait(10.)

    thread = Thread(target=hold_lock)
    thread.start()
    assert entered.wait(10.)
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_fork_probe, args=(sender,))
    try:
        child.start()
        sender.close()
        release.set()
        thread.join(timeout=10.)
        assert receiver.poll(10.), "child cache deadlocked after fork"
        assert receiver.recv() == (0, 0, 18.)
        child.join(timeout=10.)
        assert child.exitcode == 0
        assert resolution_cache_info() == before
    finally:
        release.set()
        thread.join(timeout=10.)
        if child.is_alive():
            child.terminate()
            child.join(timeout=10.)
        receiver.close()
        sender.close()


def _family_network(kind):
    builder = NetworkBuilder(kind)
    source = builder.neuron("source", LIF(name="source"), drive=20.)
    if kind == "adaptive":
        model = AdaptiveLIF(drive=18.)
    elif kind == "adex":
        model = AdEx(drive=400.)
    elif kind == "reactive_if":
        model = IntegrateAndFire(threshold=.5)
    else:
        model = LIF(name="target", drive=18.,
                    synaptic_input=kind in ("alpha", "equal_alpha", "exponential"))
    target = builder.neuron("target", model)
    synapse = (AlphaCurrent(20. if kind == "equal_alpha" else 5.)
               if kind in ("alpha", "equal_alpha") else
               ExponentialCurrent(5.) if kind == "exponential" else Delta())
    builder.connect(source, target, synapse=synapse, delay=.75,
                    weight=.5 if kind == "plastic" else 20.,
                    plasticity=PairSTDP() if kind == "plastic" else None)
    return builder.build()


@pytest.mark.parametrize("kind", ("lif", "adaptive", "adex", "reactive_if",
                                  "alpha", "equal_alpha", "exponential", "plastic"))
def test_cached_compilation_has_identical_images_and_loader_bypasses_cache(
    kind, core: CoreEvaluator, monkeypatch,
):
    inputs = (MixedInputSpike(.5, 0, 20.), MixedInputSpike(2., 1, 20.))
    with resolution_cache_disabled():
        reference = _family_network(kind)
        resolved = reference.graph.resolve()
        plan = resolved.execution_plan()
        reference_text = reference.to_text()
        with core.compile_execution_plan(plan) as compiled:
            expected_image = compiled.to_bytes()
            expected_result = compiled.run(resolved.initial_values, inputs=inputs, t_end=40.)

    actual = _family_network(kind)
    assert actual.graph.resolve() == resolved
    assert actual.to_text() == reference_text
    with core.compile_execution_plan(actual.graph.resolve().execution_plan()) as compiled:
        assert compiled.to_bytes() == expected_image
        assert compiled.run(resolved.initial_values, inputs=inputs, t_end=40.) == expected_result
    assert resolution_cache_info().hits > 0
    if kind == "plastic":
        assert expected_result.weights[0] != .5

    def forbidden(*args, **kwargs):
        raise AssertionError("binary image loader used authoring/compilation/cache")

    # Block every decorated parser/resolver, cache calls, graph resolution, and
    # the Python compiler. Loading/execution must still succeed from bytes alone.
    monkeypatch.setattr(_CACHE, "call", forbidden)
    monkeypatch.setattr(Graph, "resolve", forbidden)
    monkeypatch.setattr(core, "compile_execution_plan", forbidden)
    with core.load_compiled_graph_image(expected_image) as loaded:
        assert loaded.to_bytes() == expected_image
        assert loaded.run(resolved.initial_values, inputs=inputs, t_end=40.) == expected_result
