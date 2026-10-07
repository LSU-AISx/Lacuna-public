# Equal-delay delivery grouping

## Purpose

Compiled graphs now combine the deliveries produced by one presynaptic spike
when those edges have exactly equal delays. The scheduler stores one physical
heap event for the group, while the deposit phase still processes every edge
individually in canonical edge order.

This is a compiler-proven optimization. It does not merge events merely because
they happen to arrive at the same time, and it does not quantize or bucket
continuous timestamps. Edges from different presynaptic spikes remain distinct.

Groups of one use the original direct-delivery path. If a compiled graph has no
repeated outgoing delays at all, it uses a graph-wide direct fast path and does
not consult group descriptors during delivery scheduling.

## Preserved semantics

- Delivery traces, plasticity updates, deposits, and delivery statistics remain
  per edge.
- Queue capacity, peak occupancy, and overflow diagnostics remain expressed in
  logical per-edge events. Grouping therefore cannot make a previously
  overflowing run succeed merely by reducing physical heap entries.
- An overflow inside a group reports the exact first edge that the ungrouped
  scheduler would have rejected.
- Equal-delay edges execute in canonical edge order, preserving deterministic
  same-time behavior.
- Delay equality is exact floating-point equality. There is no tolerance-based
  merging and no loss of temporal precision.

## Validation

The native C suite and complete Python suite pass (one expected skip). Targeted
tests compare compiled grouping against the noncompiled compatibility evaluator
and cover per-edge results, run statistics, delivery-trace order, and structured
queue-overflow diagnostics.

An independent differential harness compared the new library with the
previously committed `d34bd75` library. It required exact equality of final
states, spikes, statistics, traces, weights, plasticity state, and overflow
diagnostics. Coverage included:

- 15 validation trace cases containing 2,199 spikes, 10,972 logical events,
  22,573 trace records, and 44,540 captured state values;
- 32 randomized recurrent networks containing 20,704 spikes and 120,975
  logical events; and
- static, pair-STDP, triplet-STDP, reward-modulated-STDP, and alpha-current
  plasticity cases.

The full validation campaign also matched the prior non-timing report exactly
across 15 validation and 37 scaling cases, up to 1,025 nodes and 49,152 edges.

## Benchmark

The Apple M4 Max benchmark compared the committed `d34bd75` library with the
grouped implementation through the same compiled-network, reusable-run, and
decoder path. It used 16 inputs, four outputs, out-degree eight, 96 pre-generated
input spikes, and a 50-unit horizon. Repeated and uniform regimes used five
warmups and seven trials of 50 inferences. The event-heavy distinct-delay
control used five warmups and five trials of five inferences. Values below are
trial medians.

| Delay pattern | Nodes | Edges | Edges/group | Baseline inf/s | Grouped inf/s | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| Five repeated delays | 100 | 772 | 1.911 | 600.89 | 994.65 | 1.655x |
| Five repeated delays | 1,000 | 7,972 | 1.929 | 71.91 | 112.39 | 1.563x |
| Five repeated delays | 2,000 | 15,972 | 1.906 | 42.72 | 65.25 | 1.527x |
| One delay per source | 100 | 772 | 8.042 | 714.79 | 2,032.47 | 2.843x |
| One delay per source | 1,000 | 7,972 | 8.004 | 62.02 | 215.76 | 3.479x |
| One delay per source | 2,000 | 15,972 | 8.002 | 29.53 | 101.78 | 3.447x |
| All delays distinct per source | 100 | 772 | 1.000 | 59.73 | 59.47 | 0.996x |
| All delays distinct per source | 1,000 | 7,972 | 1.000 | 6.29 | 6.25 | 0.995x |
| All delays distinct per source | 2,000 | 15,972 | 1.000 | 3.07 | 3.06 | 0.996x |

The repeated-delay workloads improve by about 1.5–1.7x, and the uniform-delay
workloads by about 2.8–3.5x. The deliberately ungroupable control is effectively
neutral (about 0.4–0.5% below the baseline), confirming that the optimization
does not impose a material cost on arbitrary real-delay networks.
