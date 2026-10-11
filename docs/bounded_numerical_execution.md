# Bounded numerical execution and trajectory reuse

Float64 numerical network nodes use bounded prediction windows and reuse
accepted integration work. The policy applies to both mixed analytical/numerical
networks and fully numerical networks. It is equation-derived: there is no
model-specific AdEx implementation or approximate-math shortcut.

## Prediction and reuse

A numerical node predicts at most its configured `maximum_step` beyond its
current state, clipped to the simulation end. Dormand-Prince 5(4) still adapts
and rejects integration steps using the configured error tolerances. The
prediction-window bound does not replace adaptive error control.

If a window contains no threshold crossing, a generation-tagged continuation
event resumes prediction at its end. Continuations run in the prediction phase,
after same-time boundaries and deposits, and are not neuronal spikes. Finding
no crossing within one window therefore does not suppress later spikes.

Each numerical node lazily allocates a run-owned cache of up to eight accepted
steps. Dense-output coefficients and accepted endpoints allow intervening state
updates and observations to reuse the trajectory. A full cache creates another
continuation boundary rather than discarding the remaining simulation interval.
Uninterrupted continuations also retain the adaptive step-size suggestion and
the last-stage derivative for first-same-as-last (FSAL) reuse.

Inputs, resets and drive changes invalidate trajectory and derivative reuse.
An input inside a prediction window consumes its valid prefix and invalidates
the rest; the implementation does not scan the event heap to find the next input.
Observations do not move the integration boundaries. Refractory/clamped advances
continue to use the ordinary integrator.

## Equation evaluation and storage

An internal derivative plan evaluates only expressions needed by the right-hand
side. Parameter-only expressions are evaluated once with the checked expression
evaluator, preserving arithmetic order. Parameter snapshots are checked before
reuse; changed bindings rebuild the plan and invalidate cached derivatives.
Structurally validated plans avoid repeating binding and reference checks at
every integration stage. Value-dependent finite/domain checks, adaptive error
control, work limits and threshold-crossing checks remain active.

Expressions used only by resets or other operations are evaluated when those
operations execute, not speculatively during derivative evaluation. Invalid
unused expressions can consequently fail at a different time than with the
former full-expression evaluator.

Trajectory storage is sized to the node's state count, with the same eight-step
capacity for all supported state counts. Run and episode resets retain allocated
buffers and plans but invalidate all trajectory state. Destruction releases
their storage. These caches are not serialized in compiled graph images.

## Precision and compatibility

Float32/time64, Float32 and Float16 retain their existing numerical execution
policies. Direct `advance_stepped` and `predict_stepped` calls retain the ordinary
uncached path. No solver tolerance is relaxed. Changed integration boundaries
mean Float64 network trajectories are not promised to be bit-identical to the
previous network algorithm.

`NUMERICAL_CONTINUATION` is a distinct trace record. Causal accounting includes
these events without counting them as confirmed spikes. Existing enum values
and compiled-image layouts are unchanged; use matching Python/native versions
when producing or reading the new trace records.

New Float64 stepped-graph metadata created by `from_resolved_graph` carries
`extra.lacuna_numerical_execution = "bounded_dense_v2"`. Offline reconstruction
uses the trace's final time as the prediction horizon and replays bounded dense
output so that requested sample times do not change integration boundaries.
Explicit `bounded_dense_v1` metadata retains restart-at-each-window replay;
untagged traces retain legacy direct reconstruction. When deliberately recording
with an older native producer, pass
`extra={"lacuna_numerical_execution": "legacy"}` rather than mislabelling the
producer's policy. Unknown policy tags are rejected.

## Regression checks

`tests/c/test_step_plan.c` checks derivative plans, expression operations,
parameter changes, domain failures, dense coefficients, compact storage and
reset/reuse behavior. `tests/python/test_numerical_continuation.py` checks
continuations, autonomous crossings, intervening and coincident inputs,
observations, incremental execution, resets, causal traces and reconstruction.
Run these along with the existing native and Python suites:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 4
ctest --test-dir build --output-on-failure
python3 -m pytest
```

The full Python suite also requires the reduced-precision builds described in
[numerical precision](numerical_precision.md).
