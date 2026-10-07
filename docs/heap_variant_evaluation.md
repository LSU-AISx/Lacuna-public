# Heap variant evaluation

## Purpose

This experiment compares Lacuna's original swap-based binary event heap with
two exact continuous-time alternatives:

1. a hole-based binary heap; and
2. a hole-based 4-ary heap.

All variants retain the complete `(time, phase, kind-order, sequence)` key and
therefore have identical temporal precision and deterministic tie ordering.
After validation, the hole-based binary heap replaced the original
implementation. The slower implementations and experimental compile-time
selection switches were removed.

## Validation

Each build passed the native C suite. Both hole-based builds also passed the
complete Python suite (one expected skip). Every measured inference case
required exact Lacuna/RISP output-count parity before timing.

The original and hole-based binary builds were additionally subjected to a
cross-library differential campaign. After removing timing-only fields, all
reported fields matched exactly across 15 validation cases and 37 scaling
cases. The campaign covered up to 1,025 nodes and 49,152 edges and audited:

- 2,199 validation spikes and 10,972 popped validation events;
- 22,573 validation causal-trace records and 253,955 scaling trace records;
- deterministic replay, exact peak-queue replay, and one-below-peak overflow;
- chronological output, input and delivery accounting, refractory spacing,
  finite final states, and same-time cascade depth.

A second, direct comparison retained the complete low-level results instead of
only campaign summaries. The builds produced exactly equal final states,
spikes, statistics, traces, weights, and plasticity state for all 15 validation
cases. This included 22,573 trace records containing 44,540 captured
before/after state values. Another 32 independently seeded 64-node sparse
recurrent graphs matched exactly over 20,704 spikes and 120,975 popped events.
Targeted learning cases matched for static edges, pair STDP, triplet STDP,
reward-modulated STDP, and pair STDP on an alpha-current edge. Finally, the
structured queue-overflow status, resource, capacity, occupancy, peak, event
kind and phase, event index, node, timestamp, and error text were all equal.

## Benchmark

The Apple M4 Max baseline used the corrected reusable-session, output-decoder
benchmark with 96 pre-generated input spikes, a 50-timestamp horizon, five
warmups, and seven trials of 50 inferences. Values are trial medians.

| Nodes | Edges | Original inf/s | Hole binary inf/s | Change | Hole 4-ary inf/s | Change |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 772 | 508.53 | 618.04 | +21.5% | 506.44 | -0.4% |
| 1,000 | 7,972 | 61.37 | 71.78 | +16.9% | 63.81 | +4.0% |
| 2,000 | 15,972 | 36.91 | 42.87 | +16.2% | 38.48 | +4.3% |

A reverse-order 1,000-neuron replication produced 61.37, 71.69, and 64.60
inferences/s for the original, hole-binary, and hole-4 variants respectively,
confirming that the ordering is not a thermal or execution-order artifact.

## Conclusion

The hole-based binary heap is the clear winner for this pop-heavy event
distribution. It removes repeated 56-byte event swaps while retaining the same
tree shape and comparison count. The 4-ary heap reduces depth but spends more
time selecting among four children; that tradeoff is unfavorable here.

The measured speedup applies to the reactive-IF benchmark. The implementation
is model-independent and should help any heap-dominated workload, but the
magnitude must still be measured on analytical LIF, multi-state synapses, and
stepped nonlinear models before making a broader performance claim.

The hole-based binary heap is now Lacuna's sole event-heap implementation.
