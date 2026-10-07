# Mixed analytical-network validation report

This report covers scalar stable-LIF/delta networks, folded and edge-scoped alpha
kernels, edge-scoped scalar exponential currents, equal-rate repeated-real
crossings, and one-mode spike-triggered adaptive-current LIF populations. It is
not evidence for STEPPED, conductance-based synapses, hardware, or
cross-simulator paths.

The machine-readable results are in `network_report.json`. Regenerate them with
the command documented in the project README.

## Outcome

All fifteen validation networks passed every enabled invariant and deterministic
replay check. Thirty-seven additional scaling cases completed without queue, output,
numeric, or cascade failures. Every case also passed the independent complete
causal-trace audit.

- Validation networks: 15
- Scaling networks: 37
- Validation spikes: 2,199
- Validation events popped: 10,972
- Validation causal records audited: 22,573
- Scaling causal records audited: 253,955
- Largest network: 1,025 nodes
- Largest topology: 49,152 edges
- Deterministic replay: passed
- Exact observed queue-capacity replay: passed
- One-entry-below-peak queue failure: passed
- Complete causal-trace audit: passed
- Explicit-time inspection causality and replay checks: passed
- Native AddressSanitizer and UndefinedBehaviorSanitizer run: passed
- Six-case versioned reference fixture replay, including incremental equivalence:
  passed
- Structured queue/output/decoder/trace/inspection overflow diagnostics: passed
- DSL/graph mutation and malformed-document campaigns: passed
- ABI version, error-layout, and null-safe lifecycle checks: passed

## Validation corpus

| Case | Nodes | Edges | Spikes | Events | Peak queue | Median run |
|---|---:|---:|---:|---:|---:|---:|
| Feed-forward chain | 32 | 31 | 32 | 61 | 9 | 0.231 ms |
| Fan-out/fan-in | 34 | 64 | 34 | 99 | 65 | 0.254 ms |
| Recurrent ring | 32 | 32 | 99 | 194 | 5 | 0.326 ms |
| Bipartite burst | 128 | 3,072 | 128 | 3,232 | 3,104 | 1.200 ms |
| Sparse recurrent | 128 | 512 | 987 | 4,804 | 211 | 2.650 ms |
| Driven population | 64 | 0 | 192 | 384 | 64 | 0.451 ms |
| Drive boundaries | 64 | 0 | 64 | 320 | 128 | 0.701 ms |
| Folded-alpha population | 64 | 0 | 64 | 128 | 64 | 4.486 ms |
| Per-edge shared exponential fan-in | 65 | 64 | 65 | 194 | 128 | 0.462 ms |
| Per-edge distinct exponential fanout | 99 | 288 | 99 | 486 | 291 | 2.146 ms |
| Per-edge equal-rate alpha fanout | 65 | 64 | 129 | 322 | 65 | 2.509 ms |
| Per-edge multi-alpha fanout | 66 | 128 | 66 | 260 | 130 | 3.063 ms |
| Adaptive-current population | 64 | 0 | 192 | 384 | 64 | 4.810 ms |
| Mixed scalar/alpha chain | 16 | 15 | 16 | 40 | 3 | 1.148 ms |
| Folded-alpha drive boundaries | 32 | 0 | 32 | 64 | 32 | 1.924 ms |

Timings are descriptive measurements from one host and are not pass/fail
thresholds.

## Invariants checked

- Spike timestamps are chronological.
- Each case reaches its intended nontrivial firing regime.
- Runtime output-spike accounting equals the emitted trace.
- Every in-horizon input spike and drive update is processed exactly once.
- Every in-horizon delivery scheduled by an output spike is processed exactly
  once.
- No node fires again before its fixed refractory duration has elapsed.
- Every final state is finite and advanced to the requested final time.
- Repeated runs produce identical states, spikes, and runtime statistics.
- A queue sized to the measured peak reproduces the result exactly.
- A queue one entry smaller than the measured peak fails explicitly.
- Complete trace records have contiguous sequence numbers, chronological times,
  and the correct scheduler phase and payload shape for their kind.
- Full state snapshots are finite and preserve same-timestamp before/after
  continuity across deposits, predictions, spikes, resets, clamps, and finalization.
- Every trace spike matches a returned output and is immediately followed by one
  atomic reset; refractory entry and release form a legal state machine and no
  node fires while clamped.
- Every input or delivery batch is applied before same-time firing, and the
  topology implies exactly the observed in-horizon edge deliveries.
- Confirmed predictions fire at the same generation, while stale predictions do
  not fire or confirm at that generation.
- Trace-derived event counts equal runtime statistics, and one canonical final
  snapshot per node exactly equals the returned final state.

The causal auditor does not propagate state or solve crossings. It consumes the
immutable topology, complete trace, and returned result, so it checks scheduler
behavior independently without becoming a second numerical evaluator. Eleven
adversarial corruptions change one fact in each invariant family and confirm that
the corresponding audit fails.

The generated campaign streams that complete trace into a temporary persisted
artifact and audits its checksum-validated iterator in one pass. Artifact
identity/completeness is checked before the eleven scheduler families. Each chunk
is decoded once; the auditor retains current-timestamp facts, compact per-node and
per-edge state, and expected in-flight deliveries rather than the full record log.

## Explicit-time inspection checks

Focused native and Python tests establish the adjacent state-inspection contract:

- A request sharing a timestamp with input, delivery, firing, reset, or a
  zero-delay cascade observes the settled post-cascade state.
- A request between events analytically advances a private state copy, including
  decay of adaptive state while the membrane is refractory-clamped.
- Removing inspection results from an inspected run leaves states, spikes, and
  runtime statistics exactly equal to an uninspected replay.
- Scalar, folded-alpha, and adaptive-current vector states support per-request
  local state selection; graph-level results restore public node ids and names.
- Inspection composes with causal tracing and streaming decoding, including when
  raw spike retention is disabled.
- Invalid times, nodes, and state selections fail before execution. The native
  ABI also rejects nonchronological requests and reports an explicit capacity
  overflow without consuming the run session.

## Persisted trace artifact checks

Focused Python tests establish the version-one host artifact contract:

- Synthetic records round-trip exact binary64 timestamp, scalar, and state bits,
  including adjacent representable values and signed zero. Repeating the same
  write with the same options produces byte-identical files.
- A graph run streams the C recorder directly into bounded chunks while retaining
  no trace tuple in the run result. Reading the artifact reproduces a separately
  buffered graph trace exactly, including public node ids and state names.
- Footer identity checks reject a mismatched canonical graph digest. Mutating a
  stored payload is detected by its raw-data checksum.
- Files without a valid completion trailer are refused by default. Explicit
  recovery returns only consecutive checksum-valid complete chunks; an
  interrupted final payload is excluded and the result remains marked incomplete.
- Indexed time-range, node, and kind filtering uses a one-chunk-at-a-time reader.
  The separate convenience reader deliberately materializes all records and is
  reserved for artifacts known to fit in memory.
- Persisted auditing accepts only complete all-kind/all-node/full-state artifacts,
  maps public node identifiers back through the verified resolved graph, and
  establishes the same scheduler checks as the in-memory API. Tests verify that
  every small chunk is read exactly once and that validly encoded causal
  corruption is still classified by the responsible invariant.

The persisted-trace stress pass covers all 15 validation graphs and all 37
scaling graphs. The validation corpus round-trips 22,573 records exactly against
separate buffered replays across scalar, folded-alpha, adaptive-current,
per-edge exponential/alpha, recurrent, driven, and mixed networks. The scaling
corpus writes and incrementally validates 253,955 records; traced run results remain equal to untraced replays once
the intentionally external trace is removed. Large-case reads stay bounded by a
1,024-record chunk rather than the complete artifact size.

## Offline reconstruction checks

Focused tests compare offline results with explicit-time inspections captured by
the live C scheduler:

- Scalar queries cover initial propagation, settled input/reset boundaries,
  clamped refractory propagation, exact refractory release, normal propagation
  after release, final state, and caller-order preservation.
- Drive updates replay the recorded propagation-DAG binding and reproduce live
  state before, at, and after the parameter boundary.
- Folded-alpha and adaptive-current vectors reproduce live inspection values and
  clamp state across event boundaries and between-event intervals.
- A membrane-only alpha recording is classified `EVENT_ONLY`, names both missing
  synaptic variables, returns the settled membrane value at an exact deposit
  timestamp, and refuses an intermediate extrapolation. Requesting an unrecorded
  variable is `UNAVAILABLE`.
- Dependency-closed partial traces are `EXACT` between events: alpha `z` is
  reconstructed from `z`, alpha `s` from `s+z`, and adaptive `w` from `w`.
  Each result exactly matches a simultaneous live C inspection.
- Omitting `RESET` is diagnosed before querying, a mismatched graph is rejected
  by identity, an incomplete recovered artifact is refused, and requests beyond
  the final-state horizon fail explicitly.

The generated reconstruction pass covers all 15 validation and all 37 scaling
graphs. For the first, middle, and last node of each graph, four times spanning
the run are reconstructed from the persisted trace. All 624 reconstructed vectors
and refractory flags exactly equal live C inspection results, including scalar,
folded-alpha, adaptive-current, recurrent, driven, and mixed networks.

## Scaling observations

The compiler builds a canonical compressed-sparse-row (CSR) outgoing-edge index
once per prepared graph. Firing a node visits only its outgoing edges, in
canonical edge order, instead of scanning the complete edge array. The campaign
reuses each prepared graph for deterministic replays, queue-boundary checks, and
timed repetitions. A regression case also covers canonical edges interleaved by
source node.

The 1,024-node chain processed 2,045 events in 5.561 ms. The 1,024-node
independently driven population processed 6,144 events in 5.650 ms. A 1,024-node,
degree-four sparse recurrent network processed 41,477 events in 24.300 ms.
Bipartite burst scaling reached 512 nodes, 49,152 edges, and 49,792 processed
events in 17.323 ms.

The 1,024-node folded-alpha population processed 1,024 certified ROOT_FIND
crossings and 2,048 queue events in 68.680 ms. The 1,024-node adaptive-current
population produced 3,072 spikes and processed 6,144 events in 73.985 ms using
the certified two-real-exponential crossing capability. A 64-node alternating
scalar/alpha chain processed its full 63-edge causal path with a peak queue of
three entries in 3.409 ms. The per-edge campaign reached a 1,025-node shared
exponential fan-in and 256-target distinct exponential, multi-alpha, and
equal-rate alpha fanouts. The two alpha paths processed 1,028 and 1,282 events in
11.180 ms and 10.236 ms respectively.

Burst queue demand is dominated by deliveries simultaneously in flight. The
128-node burst required 3,104 entries for 3,072 edges, the 256-node burst
required 12,352 entries for 12,288 edges, and the 512-node burst required 49,280
entries for 49,152 edges. This confirms that topology alone does not provide a
small queue bound and that reporting the measured high-water mark is necessary.

The first campaign run also exposed drive-event preparation re-entering SymPy for
every update. Binding expressions are now lowered once and evaluated by the C DAG
evaluator. The 64-node drive-boundary median fell from roughly 233 ms to under 2 ms,
while retaining the same trace and invariants.

## Remaining risks

- The one-shot compatibility API still prepares a temporary graph. Repeated
  workloads must retain a compiled graph to receive the setup-cost benefit.
- The largest tested case is still not a production-scale stress test.
- Edge-scoped current execution currently recognizes scalar exponential and
  unit-area alpha kernels only. Pair, triplet, and split-eligibility modulated
  weight plasticity execute independently of that kinetic tier; broader
  multi-state kinetics, composed/custom learning rules, and structural plasticity
  remain future work.
- Adaptive-current execution currently accepts one independent decay mode with a
  rate distinct from the membrane decay. Moving-threshold crossing remains
  unimplemented; repeated-real mode is implemented for the supported current kernels.
- Cross-platform reproducibility, cross-simulator convergence, and hardware
  numerical equivalence remain untested.
- The causal audit validates scheduler consistency, not the analytical equations
  themselves; closed-form, high-precision, semigroup, and limit tests remain the
  independent numerical evidence.
- Compact trace files, incremental persisted auditing, offline analytical
  reconstruction, and dependency-closed partial-state reconstruction are
  implemented. The audit still requires the returned C run result as its
  independent source of spikes, statistics, and final state; a trace artifact by
  itself can establish internal consistency but not agreement with an omitted
  result.
