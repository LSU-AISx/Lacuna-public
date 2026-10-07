# Equation-resolution cache

Lacuna reuses successful parsing and neuron/synapse resolution results in a
bounded, process-local Python cache. This reduces network construction and
compilation work, including repeated construction of evolutionary candidates.
Lacuna-EONS benefits through its existing adapter without an EONS API change.

It does not accelerate the native simulation of an already compiled graph.
Fitness evaluation, simulation state, plastic weights, compiled C handles,
execution plans, and binary image loading are not cached here.

## Correctness contract

Cache keys include the operation and its complete argument contents. This covers
model definitions and defaults, explicit bindings, numerical solver settings,
and, for per-edge resolution, synapse definitions, edge identifiers, receptor and
output mappings, and initial states. Mapping/sequence order and exact binary64
bits are preserved; values are never rounded. The structural `resolution_key`
alone is not sufficient because different numeric bindings can share it.

Graph wiring, initial-state, port, and learning configuration checks still run
on every graph resolution. Only repeated successful model work is reused. A
parameter dictionary edited between calls produces a different key. Failed
resolutions are never cached. Custom argument types outside the built-in value
records and ordinary containers use the original uncached path.

Cached values are private copies. Each hit returns another independent copy,
including nested bindings and expression-root dictionaries, so editing a result
cannot corrupt another call. As before, callers must not mutate arguments
concurrently while a resolver is reading them.

The cache uses an LRU policy, with defaults of **4,096 entries and 64 MiB of
estimated retained Python storage per process**. Both limits apply. Accounting
includes keys and copied results; it is an estimate, not a process RSS limit,
and may count shared objects in separate entries more than once. Oversized
results are returned normally without retention. Temporary resolution/copying
allocations are outside this retained-storage budget.

Cache access and reconfiguration are protected by a lock. Symbolic work and
copies run outside the lock; concurrent misses may compute the same pure result
independently. Clearing or reconfiguring prevents earlier in-flight misses from
repopulating the cleared cache. Forked workers keep the configured limits but
start with empty caches and fresh locks, including when a different parent
thread held a cache lock at fork. No entries are saved in checkpoints or on disk.

## Controls

Caching is enabled by default. Applications can inspect and adjust it:

```python
from lacuna import (
    clear_resolution_cache,
    configure_resolution_cache,
    resolution_cache_info,
    resolution_cache_disabled,
)

configure_resolution_cache(max_entries=2048, max_bytes=32 * 1024 * 1024)
print(resolution_cache_info())
clear_resolution_cache()
```

Reconfiguration clears entries and counters. Unspecified settings retain their
current values. `enabled=False`, `max_entries=0`, or `max_bytes=0` bypasses the
cache process-wide. `clear_resolution_cache()` keeps the current configuration.
For multiprocessing, the budget applies to each worker separately.

A context-local bypass supports direct reference comparisons:

```python
with resolution_cache_disabled():
    reference = graph.resolve()

actual = graph.resolve()
assert actual == reference
```

This context bypasses both reads and writes, including nested parser/resolver
calls, and leaves existing entries and counters untouched. It is local to the
current thread/task context; other threads can continue using the cache.
`resolution_cache_info().enabled` describes the process-wide setting.

## Binary deployment images

The cache does not change the compiled-image format, C ABI, loader, or runtime.
`CoreEvaluator.load_compiled_graph_image()` and the native C deserializer
continue to reconstruct a graph directly from bytes, without using the parser,
resolver, cache, or Python graph compiler. Regression tests require byte-identical
images from cached/uncached compilation and identical loaded-image execution,
including plasticity. They also block authoring/cache/compiler calls while
loading and executing an image to check this separation explicitly.

## Validation

`tests/python/test_resolution_cache.py` checks exact configuration separation,
mutable input/output isolation, unchanged validation, bounded retention,
eviction, bypass/configuration, concurrent access, clearing during an active
miss, and fork safety. It compares resolved records, JSON, binary images, and
runtime results for LIF, adaptive LIF, AdEx, reactive IF, exponential current,
alpha current, equal-decay alpha, and plastic connections.

## Development measurements, 2026-09-09

Controlled serial checks compared this implementation with
`configure_resolution_cache(enabled=False)` in the same working tree, using
Python 3.10.16, SymPy 1.14.0, and arm64. Fixed synthetic LIF/delta candidates
were measured over five alternating trials, with SymPy warmed and the
resolution cache empty at the start of each candidate. Timings include
materialization and compilation, excluding simulation and resource cleanup.

| Candidate | Cache disabled | Cache enabled |
|---|---:|---:|
| 32 neurons, 228 edges | 264.284 ms | 115.285 ms |
| 128 neurons, 990 edges | 1,276.357 ms | 548.193 ms |

A separate short Lacuna-EONS check used seed 1729, six candidates, four evaluated
generations (three evolutionary steps), initially 32 neurons and 128 attempted
edges, and three 40-time-unit simulations per candidate. Fitness was the sum of
output spike counts. This single run took 6.726 seconds uncached and 1.278
seconds cached. Genomes, lineage, fitness, RNG states/counters, and simulation
result records matched at every generation. The cache retained 193 entries
with 4,986,899 estimated bytes and recorded 2,198 hits and 193 misses.

These are synthetic setup/short-run observations, not a prediction of Wine or
MNIST training speed. They do not measure parallel throughput or an improvement
to pure native inference. The Lacuna-EONS compatibility suite also passed all
19 tests with the cache enabled.
