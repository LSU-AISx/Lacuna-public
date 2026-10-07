# Numerical precision implementation plan

Update, 2026-09-14: the requested end state is now exclusively uniform float16,
float32 and float64 execution. The implementation and historical validation
below predate that decision. See `docs/uniform_precision_plan.md` for the strict
contract and migration plan. The subsequent strict FP16 implementation is
described in `docs/float16.md`. Mixed-profile retirement and float64-guard review
remain outstanding and are not implied by FP16 support.

Status: all three native profiles are implemented on
`codex/numerical-precision`, including target-aware compilation, native learning,
recording, codecs, hazards and deployment images. Float64 remains the default.
See `docs/numerical_precision.md` for current usage and numerical limits.
Physical MCU validation remains a separate hardware acceptance step and is not
claimed by this desktop implementation. Earlier dated sections below record
historical gates and do not describe the current availability of float32.

Baseline: `origin/main` at `13ba3ca`, reviewed on 2026-09-12. The existing
float64 implementation remains the reference for compatibility. Experimental
work outside this baseline is not included in precision validation.

## Required behavior

Precision is selected before initialization and training. The same profile is
used for training, validation, graph export, and execution on the target.
Converting a float64-trained network to float32 is not the normal workflow.
Persist the chosen profile with training configuration, checkpoints, and export
provenance so that resuming a run cannot silently change its arithmetic.

Support these profiles without changing the default:

| Profile | Model and learning arithmetic | Event time |
| --- | --- | --- |
| Existing default | float64 | float64 |
| Optional mixed precision | float32 | float64 |
| Strict float32 | float32 | float32 |

Strict float32 must not depend on float64 or long-double arithmetic in the C
execution path. This includes propagation, threshold prediction, numerical
integration, learning, encoders, decoders, and recording. Profiling and diagnostic
code must also be audited so that enabling it does not introduce an undeclared
double-precision runtime dependency on an MCU.

Precision is a native build and compiled-graph property. The Python engine may
select the matching native library, but individual arithmetic operations must
not branch on precision. Initially, every component in a graph uses the same
declared profile. Per-neuron and per-edge precision mixtures are out of scope.

## Compatibility contract

The default profile must retain existing model coverage, execution strategies,
event ordering, refractory behavior, plasticity semantics, recording, and
deployment behavior. Do not relax the existing float64 tests to accommodate a
precision refactor. Do not replace equation-derived execution with model-specific
float32 handlers.

For the initial type abstraction, preserve the existing float64 C layouts,
function signatures as compiled, numerical defaults, expression order, and image
bytes. Later ABI or image-format changes require explicit versioning and a
tested compatibility policy. Known legacy float64 images should remain readable
where their semantics can be preserved. Otherwise require a documented migration,
not reinterpretation or silent conversion. Binary compatibility is a separate
gate from numerical equivalence.

Float32 is a different numerical execution contract, not a promise of identical
float64 spike trains or learning outcomes. Analytical propagation remains
analytical where supported, but its evaluation has finite-precision error.
Float64 timestamps do not recover accuracy already lost in float32 state or
crossing calculations.

A float32 limitation must not remove an existing float64 capability. Report an
unreliable or unrepresentable reduced-precision configuration explicitly. Do not
silently widen it, substitute a different model, or report an unresolved crossing
as proof that no crossing exists. Reduced-precision profiles remain experimental
until their advertised capability matrix passes.

## Implementation slices and gates

### 1. Freeze the float64 baseline

Build and test a clean snapshot of the committed source, not the current dirty
working tree. Record source identity, compiler, build flags, Python dependencies,
native library identity, and any unavailable fixtures.

Capture deterministic fixtures for states, timestamps, spikes, learned weights,
traces, inspections, and compiled images. Include stateless and reusable runs,
episode reset, split execution, frozen and plastic shared weights, and both
cached and uncached resolution. Runtime duration is not an equality fixture.

Gate: establish the actual baseline before modifying numerical code. Existing
failures or missing artifacts must be reported separately from new regressions.

### 2. Centralize types without enabling float32

Introduce role-specific numeric types, math wrappers, and constants while keeping
their float64 implementation unchanged. Retain `lc_time_t` and introduce a model
scalar type such as `lc_real_t`. Explicitly classify clock intervals, error
tolerances, accumulators, expression inputs, learning state, and telemetry.

Audit public descriptors, private graph/run arrays, allocation sizes, copies,
callbacks, and all C modules. A storage-only change is insufficient. Existing
double literals and mixed time/state operations can promote float operations
back to double. Do not change expression order, enable fast-math, or introduce
new approximation formulas in this slice.

Gate: the unchanged float64 test suite passes. Under the same toolchain and flags,
deterministic fixtures and deployment bytes match the baseline. Verify public
structure sizes and field offsets, and exercise the new default library through
unmodified old bindings. Keep this slice independently reviewable.

### 3. Define and validate target precision

Add explicit profile metadata to the runtime and precision-dependent compilation
results. Include the profile and precision-dependent numerical policy in relevant
cache keys and artifact identities. The pure symbolic frontend may continue to
use exact symbolic work or host precision, but its emitted target model must be
checked under the selected representation before target specialization is trusted.

Validate both bound parameters and derived constants after target conversion.
Cover overflow, underflow, rates rounding to zero, distinct rates becoming equal,
and reset/threshold separation disappearing. Re-derive a valid supported form or
reject the configuration. Casting an already-specialized float64 graph at load
time is not a safe substitute for target-aware compilation.

Time variables inside user-authored equations need an explicit policy too.
Keeping the scheduler clock in float64 while silently narrowing an absolute time
input to a float32 expression does not preserve that equation's time resolution.

Gate: float64 behavior remains unchanged and target-invalid configurations fail
before execution or return a precise runtime diagnostic where validation cannot
decide in advance.

### 4. Implement float32 numerical execution

Use the selected types for model state, weights, learning traces and updates,
expression workspaces, and ordinary arithmetic. Use matching math functions and
constants. Audit analytical propagation, adaptive Dormand--Prince integration,
integrated hazards, codecs, and threshold prediction separately.

Review root isolation and near-zero classification algorithm by algorithm.
Current long-double guards cannot simply be replaced with float arithmetic, and
changing every `DBL_EPSILON` to `FLT_EPSILON` is not a correctness argument.
Use stable formulations, justified precision-specific error bounds, and explicit
unresolved outcomes. Any required new numerical method needs its own tests and
review. Retaining a wider safeguard is permissible only in a separately declared
profile, never in strict float32.

Preserve existing float64 tolerances. Introduce justified float32 defaults and
report effective tolerances. Do not silently loosen user-requested accuracy or
choose arbitrary tolerances just to make tests pass.

Check representable clock progress, delayed delivery, refractory release, and
integration step advancement. A positive interval that rounds away must not
silently become a zero-delay event or an infinite loop. Check the requested time
horizon where possible and continue validating dynamically as the run advances.
Time tolerance must not become a blanket permission to merge distinct queue
events. Long-running clock rebasing or integer-time profiles are separate future
work, not hidden behavior of float32.

Gate: each advertised numerical capability passes precision-specific tests. The
strict build must be audited for unintended double/long-double operations using
compiler diagnostics and target binary inspection, including linked support
functions. Symbol checks alone do not detect all hardware double operations.

### 5. Extend the Python and deployment boundaries

Query precision through fixed-width integer metadata before any precision-bearing
FFI call. Use evaluator-owned ctypes layouts and buffers so that loading one
profile cannot corrupt bindings for another. Native C consumers also need a
checked precision contract, not only an unchecked header build flag.

Version the deployment representation where needed. Distinguish wire scalars
from wire timestamps and validate profile-dependent record sizes, references,
checksums, and allocation bounds. Loading a mismatched profile fails explicitly.
The normal exporter preserves values already trained in the selected profile.
Loading still reconstructs the existing graph abstraction without running the
symbolic compiler on the target.

Initial neuronal state and codecs are not currently included in the graph image.
Their external deployment interfaces must follow the same profile, as must
recording callbacks and readback buffers. An archival trace may explicitly widen
float32 values for storage without changing execution, but that archival format
must not force double arithmetic into the MCU runtime.

Gate: native and Python round trips pass for all profiles. Wrong-library and
wrong-image combinations fail before unsafe memory access. Loaded execution
matches host execution with the same numerical profile, within the declared
cross-platform numerical contract.

### 6. Validate native-precision learning and MCU deployment

Train small networks from initialization in each profile. Cover standard pair,
triplet, and modulated STDP and generic learning programs.
Include frozen and learned shared convolution weights, accepted/discarded
refractory inputs, tiny updates, cancellation, clipping, and episode reset.

Training helpers that calculate losses, credit transport, rewards, or modulation
signals must explicitly implement their chosen numerical policy. Performing those
operations in Python float64 and casting only the final signal does not establish
strict float32 training. Initialization and codec-generated inputs must also be
converted and validated before they influence a training run.

Compare a trained profile's exported/reloaded values with that same training run.
Do not require a float32 training trajectory to duplicate a float64 trajectory.
Do not infer unchanged training quality from a single seed or an isolated neuron.

Run representative graphs on actual MCU toolchains and boards. Record math-library
behavior, contraction/fast-math settings, and treatment of subnormal values.
The same nominal float type does not guarantee bitwise agreement across machines.
Measure RAM, image size, kernel time, and numerical disagreement separately.
Numeric arrays may shrink, but indexes, pointers, timestamps in the mixed profile,
and structure padding prevent a blanket claim of half the memory.

Gate: measured target evidence supports the advertised capabilities and costs.
Do not change defaults or publish performance claims based only on desktop tests.

## Regression coverage

The existing execution-plan matrix is the backbone, not the whole test strategy.
Retain independent analytical references and targeted suites for:

- Reactive IF, closed-form LIF, adaptive/root-found trajectories, repeated and
  nearly repeated real modes, alpha/exponential kernels, and stepped custom ODEs.
- Per-edge and aggregated synaptic state, signed deposits, mixed-polarity static
  graphs, simultaneous events, delays, refractory boundaries, and sparse scheduling.
- Stochastic escape models and seeded encoders. Separate random-bit stream
  reproducibility from precision-dependent transformations and spike times.
- Plasticity, learning observers, shared weights, drive updates, codecs, exact
  event-time records, state inspection, trace artifacts, and episode reset.
- Compiled-image round trips, malformed inputs, incompatible profiles, cache
  separation, direct-C use, and interleaved Python evaluators of different profiles.

The concrete starting suites are `test_execution_plan_matrix.py`, `test_expr.py`,
`test_alpha.py`, `test_adaptive.py`, `test_per_edge.py`, `test_stepped.py`,
`test_escape.py`, `test_compiled_graph.py`, `test_learning_program.py`,
`test_plasticity.py`, `test_weight_sharing.py`,
`test_codecs.py`, `test_tracefile.py`, and `test_resolution_cache.py` under
`tests/python`, together with the native C suite. Full default-profile testing
must still cover the remaining public-API, validation, and application suites.

The current native fixture selects the first library in `build/liblacuna_core.*`,
and some Python tests instantiate an engine directly. Separate profile builds
need explicit library selection throughout the tests, not only in one fixture.
Do not regenerate existing reference artifacts to make changed results pass.

For unchanged float64, use existing assertions and same-toolchain differential
fixtures. For float32, use independent target-precision arithmetic checks and
higher-precision references with justified error bounds. Include adversarial
near-threshold and near-tangency cases rather than only well-separated spikes.
Long recurrent runs can amplify small rounding differences, so trajectory
disagreement alone neither proves nor disproves correct reduced-precision execution.

## Current acceptance boundary

The software implementation now includes all six slices' host/runtime work.
Native C and Python validation covers the supported execution and learning
matrix, deployment round trips and float64 compatibility. The remaining part of
slice 6 is physical-target acceptance: build with each selected MCU toolchain,
verify its math behavior and clock range, then measure RAM, image size, timing
and numerical disagreement. Do not infer those measurements from desktop runs.

## Initial baseline observation, 2026-09-12

This planning pass did not modify runtime code or enable a new precision profile.
A fresh Release build used an isolated archive of commit
`13ba3ca666a5062e9b2636ec826db50f52e7473e`, Apple Clang 17.0.0
(`clang-1700.0.13.5`, arm64 macOS), CMake 3.31.5, and Python 3.10.16 with
pytest 9.1.1, NumPy 2.2.6, and SymPy 1.14.0. Release flags were `-O3 -DNDEBUG`,
with the project's warning flags and its existing `-UNDEBUG` for native tests.

The native CTest executable passed, 1/1. The source-only Python run had 666
passed, 1 failed, and 1 skipped. The failure was a missing external archived
fixture, not a numerical failure:
`artifacts/optimization/mnist_pairwise_pool2_10000_disjoint.json`.

The existing local fixture was copied unchanged into the isolated snapshot.
Its SHA-256 was verified before and after copying:
`b5645f9bd2c6c185ad5187f43c214d8e18d3d286cf283cead4bc6a1752845165`.
The full rerun passed with **667 passed and 1 skipped**. The skipped test was
`test_visualizer_serves_tokenized_local_application` because local socket binding
is unavailable in this sandbox.

The tested native library SHA-256 was
`bc59d16ee2e76dbeb97512aadaaf3e60f6b2e5f9179ada7142f6c2cf29754aca`.
Local reports are retained under
`artifacts/validation/precision_baseline_20260912/` as
`python_source_only.xml`, `python_with_archived_fixture.xml`, and
`native_ctest.log`. The isolated source/build remains at
`/private/tmp/lacuna-precision-baseline.SNFGa1` for this session.

These results establish an initial current-source test baseline, not completion
of every gate in slice 1. Cross-build numerical fixtures, old-client/layout checks,
Debug/sanitizer coverage, and MCU verification remain to be performed. No reduced-
precision accuracy, learning, memory, or performance result is claimed here.

## Float64 foundation implementation, 2026-09-12

The implementation lives in an isolated `codex/numerical-precision` worktree
based on the baseline commit above. It excludes unrelated uncommitted research
experiments from the original working tree.

`c/include/lacuna_numeric.h` centralizes `lc_real_t`, `lc_time_t`, `lc_wide_t`,
numeric limits, literal helpers, and direct libm aliases. Public descriptors,
private state/parameter buffers, expression evaluation, learning, codecs, and
root/integration clock variables now use these roles. All aliases retain their
original underlying types and functions. Host wall-clock statistics and version-1
wire values deliberately remain explicit `double`. Codec finite checks use the
type-generic `isfinite` directly so a future mixed profile does not narrow clock
values before checking them.

This slice changes neither ABI 17 nor image version 1. It does not implement
selectable profiles, convert trained models, change default tolerances, or alter
existing solver formulas. Wider guards remain exactly as before. Literal helpers
are available, but auditing every arithmetic literal and mixed clock/state
expression is still required before enabling float32. The type aliases alone
are not a reduced-precision implementation.

Verification on the same host and compiler as the baseline:

- Direct comparison found the rebuilt Release library byte-identical to the
  baseline library, with SHA-256
  `bc59d16ee2e76dbeb97512aadaaf3e60f6b2e5f9179ada7142f6c2cf29754aca`.
- Release and Debug native tests passed, 2/2 in each build. The additional native
  test checks default numeric types, constants, and original math-function aliases.
- AddressSanitizer and UndefinedBehaviorSanitizer native tests passed, 2/2.
  Leak detection was disabled. The Python process was not sanitizer-instrumented.
- The complete Python suite passed, 676 passed and 1 skipped, including nine new
  tests for the differential validation utility. The skip remains the sandbox's
  unavailable local socket binding. The unchanged archived fixture described
  above was supplied to the isolated worktree.
- Strict C99 and C++11 public-header smoke checks passed.
- The old/new differential gate passed for all 11 execution-plan matrix fixtures.
  It compares complete records using exact binary64 bits, excluding only the
  `kernel_seconds` wall-clock field. This includes neuron state, spike times,
  learned weights, plasticity state, integer statistics, traces, and inspections.
- All 49 public C structures retained their size and alignment, and all 367
  members retained their offsets and sizes. The old, unchanged Python bindings
  successfully exercised the new native library.
- Compiled images were byte-identical for every fixture. Each runtime loaded and
  executed the other runtime's images with identical observed results.
- Reusable reset replay, five incremental segments per fixture, and preservation
  of learned weights across episode reset are included in the differential gate.

The 11 fixtures cover scalar, adaptive, reactive, stepped AdEx/QIF, mixed neuron
families, per-edge alpha, distinct/equal-rate exponentials, folded deposits, and
mixed static/pair/triplet/modulated learning. Standalone codec execution,
stochastic models and shared convolution training are covered by existing
tests but not by this new 11-case cross-build comparison. Do not describe that
comparison as exhaustive cross-build coverage of those features.

The reusable command is `scripts/validate_float64_compatibility.py`, with
`--baseline-root`, `--candidate-root`, and a new or empty `--output` directory.
Both roots must already contain matching-toolchain builds under `build/`.
The utility keeps complete result records, image bytes, ABI probes, source
inventories, and library hashes rather than only a pass/fail summary.

Local evidence is under `artifacts/validation/precision_float64_slice1/`:
`python-final.xml`, `native-release.log`, `native-debug.log`,
`native-sanitize.log`, and `differential-final/report.json` with its accompanying
fixtures and images. Earlier intermediate reports are retained separately.

No MCU build, float32 numerical run, training-quality study, or performance
comparison was performed in this slice. These passing gates establish the tested
float64 compatibility boundary, not a universal proof for every network or host.

## Precision metadata and validation foundation, 2026-09-12

This increment starts implementation slice 3 without enabling either reduced-
precision profile. `PrecisionProfile` defines `float64`, `float32-time64`, and
`float32` identities, with explicit model and time widths and an arithmetic
revision. Its versioned metadata record rejects contradictory widths, unknown
fields, and unsupported schema or arithmetic revisions. A profile's cache key
is its name, both widths, and arithmetic revision. This is the precision portion
of a future target cache key, not a substitute for model and policy identity.

`Engine` and `CoreEvaluator` accept keyword-only `precision` and `time_precision`
arguments. The only runnable setting is still float64. For example:

```python
from lacuna import Engine, PrecisionProfile

engine = Engine(precision=PrecisionProfile.FLOAT64)
print(engine.core.precision_info)
```

`float32` alone names strict float32, while `float32` with `time_precision="float64"`
names the mixed profile. Both currently raise `CapabilityError` before native
library loading. No graph initialization, training, quantization, or silent
fallback occurs. CMake settings `LACUNA_REAL_BITS` and `LACUNA_TIME_BITS` likewise
reject anything except 64. Direct C compilation has matching header guards.

The new C metadata interface uses only fixed-width integer arguments and return
values. `lc_numeric_property` reports the actual compiled representations and
`lc_numeric_profile_check` validates an expected profile, widths, and arithmetic
revision. Python checks the original ABI first, then the numeric metadata before
binding any precision-bearing call. A missing or contradictory half-handshake
is an error. Original ABI-17 libraries lacking both new symbols remain accepted
under their fixed float64 legacy contract. `precision_info.metadata_source` labels
that inference explicitly and leaves their unknown wider arithmetic fields unset.

Execution plans carry a frozen precision field and reject unsupported profiles
during construction, lowering, and at native compile/load boundaries. Compiled
graphs and high-level compiled networks expose the validated profile. Existing
version-1 deployment bytes remain unchanged, with image size/write/read checks
requiring binary64 runtime and wire representations. ABI17 remains unchanged
because this increment adds integer query functions without changing existing
data layouts or call signatures.

Pure authoring helpers now test finite range, signed zero, subnormals, nonzero
underflow, strict ordering after conversion, and clock progress after rounding
both operands and their sum. Tests include nearby thresholds and distinct rates
that collapse in float32, and small intervals that cannot advance a large
float32 timestamp. These helpers use host scalar packing and do not execute
neuronal equations or establish a target math library's numerical behavior.

Verification on the baseline host:

- Python: **853 passed, 1 skipped**. The skip is the same unavailable sandbox
  socket binding. This includes 177 new profile, build-guard, and integration
  cases. The original archived fixture was supplied unchanged.
- Native Release and Debug: **2/2 passed** each.
- Native AddressSanitizer/UndefinedBehaviorSanitizer Debug: **2/2 passed** with
  leak detection disabled. No Python sanitizer or MCU execution is claimed.
- All 11 baseline matrix fixture records and image bytes match exactly. Public
  sizes, alignment, offsets, and member sizes match for 49 structures and 367
  members. Both cross-image loading directions pass.
- The full 11-fixture comparison additionally passes with the new Python
  bindings against each of the legacy and new native libraries. The matrix
  source file is identical in both trees. This extends the unchanged-old-binding
  comparison to cover all four binding/library combinations.

Evidence is retained under `artifacts/validation/precision_metadata_slice2/`:
`python.xml`, `native-release.log`, `native-debug.log`, `native-sanitize.log`,
`differential/report.json`, and complete records and images in
`new_bindings_legacy/` and `new_bindings_new/`. The library now contains additional
metadata functions and is not claimed to be byte-identical to the old library.
The numerical records and graph deployment images are byte-identical in the
tested matrix.

Remaining work in slice 3 is target-aware parameter binding and derived-constant
validation before analytical capability selection. Precision and numerical
policy must then enter every affected resolution cache and persisted training
artifact. No graph-schema precision field or checkpoint migration is implemented
yet. Existing symbolic caches are still float64-only. Solver tolerances, root
certification, literal promotion, mixed clock/state arithmetic, bindings for32,
and reduced-precision deployment images remain later gates. In particular, this
increment does not replace long-double safeguards with float32 or cast a
float64-trained graph.

## Target-representation preflight, 2026-09-12

`analyze_precision` now prepares rounded authoring inputs before host equation
resolution and checks derived representation constraints afterward. It accepts
an authored `Network` or `Graph`, not a compiled graph or simulation result:

```python
from lacuna import analyze_precision

report = analyze_precision(network, "float32", time_horizon=1000.0)
print(report.executable)  # Always False for this inspection report
print(report.changed_values)
```

The mixed profile can be requested as `"float32-time64"` or with
`precision="float32", time_precision="float64"`. This analysis never creates a
native run, changes the source network, trains weights, or casts a compiled
float64 graph into an executable float32 graph. Normal compilation still rejects
both reduced-precision profiles. Even a float64 analysis report is an inspection
artifact rather than an executable object.

Authored preparation covers effective parameter defaults and instance overrides,
numeric equation literals, node and edge initial values, weights and learning
rules, codec settings, delays, and refractory durations. Parameters with time
units still use model precision when that is their declared runtime storage
role. The original source and cached parser records are not modified. Prepared
IR is passed privately into the shared graph resolver and is not exposed in the
report as a compilable graph.

Derived checks cover finite range and nonzero underflow, stable rates,
threshold/reset and initial-state separation, grouped synaptic initial values,
deposit scales, and parameter-only algebraic fragments in model and learning
DAGs. Integer equation tokens must remain exactly representable, so rounding
cannot silently change a polynomial degree. Original source literals that were
already lost to zero by host parsing are rejected, including decimal and hex
defaults or refractory durations. That source-integrity check deliberately also
covers unused defaults. Otherwise, unused defaults are not target-range checked
when every instance overrides them. All authored model equation definitions are
inspected, including currently unreferenced definitions.

Rate equality is checked after derived values are rounded. A regression uses
the already-representable float32 constants `24.00001335144043` and
`24.000015258789062`. Their distinct host negative reciprocals both round to
`-0.04166664183139801` in float32. Such a configuration is rejected before a
distinct-rate crossing formula can be reused. A rounded model that changes
resolved family is also rejected, including analytical adaptation that would
otherwise fall through the existing capability exception handler to stepped
execution. `PrecisionResolutionError` is intentionally separate from
`CapabilityError` to prevent that fallback from consuming a precision failure.

The optional horizon checks fixed delays and refractory intervals conservatively
at the requested clock value. It does not establish clock progress for all
future dynamic events. At a float32 timestamp of `2**24`, a delay of `0.1` cannot
advance the clock and is rejected. The same fixed-delay check passes with float64
timestamps, without implying that mixed clock/state arithmetic is certified.

Reports retain checked source/target values, storage roles, host capability
proposals, and explicit outstanding checks. Numeric values export as hexadecimal
strings through `to_document()` so the audit does not lose retained bits.
The bounded resolution cache stores detached report copies under the canonical
graph source, complete precision identity, arithmetic revision, preflight policy
revision, and horizon. Pure host-resolution cache entries remain host-only and
may be shared across preflights with identical rounded inputs. No target runtime
specialization is stored in those entries.

This preflight is deliberately not a target numerical evaluator. It does not
evaluate neuron state trajectories or learning updates in Python. Exponential,
logarithmic, and trigonometric function evaluation is left to the runtime, though
known constant domain violations such as a nonpositive logarithm argument are
rejected. State-dependent singularities, root certification, numerical tolerance
adequacy, dynamic input ranges, and event ordering remain uncertified. Reduced-
precision hazard sentinels and mixed-profile absolute-time DAG dependencies are
explicitly rejected pending their respective numerical contracts.

Verification:

- **1,017 Python tests passed, 1 skipped**, including 164 new authoring,
  derived-check, and public preflight tests. The skip remains sandbox local-socket
  binding. The archived optimization fixture remains unchanged.
- All 11 execution-plan matrix families pass representation preflight under each
  of the three profile identities. These 33 checks are not float32 simulations.
- Release, Debug, and native ASan/UBSan tests passed **2/2** each. These rerun the
  existing native builds because this increment changes Python only. Leak
  detection was disabled, and Python was not sanitizer-instrumented.
- The 11 old/new differential fixture records and graph images match exactly,
  including bidirectional image loading and all 49 public structure layouts with
  367 members. Additional workers using the new Python frontend with the legacy
  and current C libraries also match every baseline fixture and image.

Evidence is under `artifacts/validation/precision_preflight_slice3/` in
`python.xml`, the three native logs, `differential/report.json`, and the complete
`new_bindings_legacy/` and `new_bindings_new/` worker outputs.

The next gate is certified target lowering and numerical policy, not enabling
float32 by changing typedefs. Target-rounded constants must agree with the
selected execution programs, equal-rate cases need valid target derivations,
and float32 tolerances and root guards require explicit numerical justification.
No native float32 execution, training-quality, memory, performance, or MCU result
is claimed by this increment.

## Implementation record: host target binding and numerical-policy contract

The compiler now has an optional host-side primitive binder in
`c/src/target_bind.c`, exposed through `TargetBindingEvaluator`. It is built as
`lacuna_target_bind`, not linked into `lacuna_core`. An embedded/runtime-only
build can disable it with `-DLACUNA_BUILD_TARGET_BINDER=OFF`. Normal simulation,
compilation, and deployment loading do not call or load this companion.

The companion uses a separate fixed-wire ABI with integer version, size, and
field-offset checks before Python binds its numerical function. Binary64 wire
buffers carry input values and exact widened binary32 results for the host
frontend. Selected binary32 primitive operations use float operands and stored
float results. Binary64 uses a separate loop. These host transport buffers do
not establish or require double arithmetic on a future strict-float32 MCU.

Only constants and immutable parameter closures can bind. The supported
arithmetic is negation, addition, subtraction, multiplication, division, and
maximum. The existing lowered DAG order is preserved without reassociation or
contraction. This does not promise the textual evaluation order of the authored
equation, which has already passed through symbolic lowering. Unsupported
powers, transcendental functions, and `PHI1` variants remain explicit in the
source program. Variables and declared mutable parameter closures remain
dynamic, including expressions affected by drive updates.

The host checks binary32/binary64 representations, round-to-nearest mode, and
subnormal preservation. It rejects nonfinite constants or results, nonzero
underflow to zero, zero denominators in bound divisions, and malformed DAG
references. It does not evaluate neuronal states, runtime timestamps, or
dynamic domains. A dynamic division by a known zero is still an unresolved
expression, not a certified valid operation. `MAX` follows host `fmax`/`fmaxf`
behavior, including platform-specific signed-zero ties. Bound result bits are
included in identity, so differing retained results cannot share an artifact
key. No cross-MCU bitwise libm guarantee is made.

`TargetBoundProgram` retains a detached immutable DAG snapshot, binding kinds,
exact result values, fixed bindings, mutable names, and policy metadata. Its
identity includes all of these numerical inputs and results together with the
precision profile, arithmetic revision, binding revision, companion ABI, and
supported operation set. Mutable parameter values do not enter the binding key
because none of their dependent expressions is evaluated. Initial mutable
values are still checked for representability at this boundary.

These artifacts are always non-executable. They do not replace an
`ExecutionPlan`, specialize crossing descriptors, or freeze parameters in the
current runtime. Native mutation enforcement remains a future prerequisite for
using fixed-parameter specialization in executable graphs. In particular, the
current scalar resolver lowers reciprocals as `POW`. Such coefficient roots
remain unresolved here, and the binder does not substitute a cast of a host
coefficient. Supporting these roots requires an explicit target arithmetic
decision and consistency checks between propagation and crossing programs.

`numerical_policy(profile)` adds a versioned immutable policy record. Float64
entries describe existing implementations and retain their existing settings.
All reduced-precision propagation, crossing, root finding, adaptive integration,
hazard, and clock-comparison entries remain uncertified. The record contains no
new tolerance defaults and cannot authorize execution. Both reduced-precision
Engine selections and native runtime build selections remain disabled.

For example, this inspects target-width constants without running a neuron:

```python
from lacuna import ExprDAG, ExprNode, ExprOp, TargetBindingEvaluator

program = ExprDAG(
    nodes=(
        ExprNode(ExprOp.CONST, value=16777216.0),
        ExprNode(ExprOp.CONST, value=1.0),
        ExprNode(ExprOp.ADD, lhs=0, rhs=1),
        ExprNode(ExprOp.SUB, lhs=2, rhs=0),
    ),
    parameters=(), variables=(), roots={"result": 3},
)
bound = TargetBindingEvaluator(precision="float32").bind(
    program, {}, mutable_parameters=(),
)
assert bound.root_value("result") == 0.0
assert not bound.executable
```

The same lowered operations yield 1.0 under binary64. This is a primitive
rounding test, not a float32 network simulation. Precision-dependent behavior
must eventually be validated in full propagation, crossing, learning, and
event-ordering tests before target execution can be enabled.

Validation on the Apple Clang 17 arm64 host:

- **1,126 Python tests passed, 1 skipped**, including 84 target-binding tests
  and 25 numerical-policy tests. The skip remains sandbox local-socket binding.
- All 11 matrix model families have their top-level expression DAGs inspected
  under all three profile identities. The tests retain variable and mutable
  parameter closures. These are binding tests, not reduced-precision runs.
- Release, Debug, and native ASan/UBSan suites each passed **3/3**. This includes
  the new companion suite. Leak detection was disabled, and Python was not
  sanitizer-instrumented. AArch64 flush-to-zero rejection was exercised on
  this host. Conditional x86 FTZ/DAZ tests were not executed here.
- A separate Release build with the companion disabled passed its **2/2**
  native runtime tests and contains no companion build target.
- The 11 old/new float64 fixtures remain bit-identical, with all 49 public
  layouts and 367 members unchanged. Deployment images remain byte-identical
  and load in both directions. New Python bindings with the old and current
  libraries also reproduce every baseline fixture and image.

Evidence is under `artifacts/validation/precision_binding_slice4/`. It contains
`python.xml`, `target-binding.xml`, native build test logs, the runtime-only
build and test log, `differential/report.json`, and the complete old/new binding
worker outputs. The three comparison `fixtures.json` files share SHA256
`c012b2f97612f9b5e8b5877686087edfd04da9025edd7a0d83ae077ad60f88cd`.

The next gate is consistent target coefficient derivation from reviewed
arithmetic, beginning with reciprocal/power handling, then target-aware
crossing and integration policies. Native parameter mutability enforcement is
also required before any bound program can become executable. No float32
network, training, performance, or MCU result is claimed here.

## Native execution completion, 2026-09-12

All three profiles now execute on the development host. The implementation
rounds authored values before resolution, evaluates derived constant roots in
the selected C evaluator, retains runtime and mutable expressions, and rejects
unsafe target-dependent analytical regimes. It does not use the optional
inspection-only primitive binder as an executable graph or silently cast an
already-specialized float64 network.

Model state, learning updates and traces, codecs, analytical trajectories,
threshold prediction, numerical integration and stochastic hazards follow the
declared profile. In the mixed profile the clock and numerical guards may use
binary64, while scalar model calculations remain binary32. Strict float32 uses
binary32 throughout the execution runtime and disables wall-clock profiling.
User-facing details and local numerical targets are documented in
`docs/numerical_precision.md`.

Version-1/ABI-17 float64 image compatibility is retained. The reduced profiles
use version-2/ABI-18 images with explicit scalar and clock widths. Python layouts
are evaluator-owned. Compiled plans, learned network snapshots, decoder handles
and trace reconstruction enforce profile consistency. Binary64 trace-file
transport preserves native values without changing execution arithmetic.

Final testing found and corrected three specific reduced-precision failures.
An overflowing intermediate AdEx trial is now rejected and retried at a smaller
step rather than aborting an otherwise valid trajectory. A rounded scalar
equilibrium no longer certifies a constant hazard trajectory. Incremental
requests are validated against their target-rounded timestamps before any
modulation can be queued. Each has a regression test. No compensated clock,
global timestep, widened strict solver or polynomial approximation to an
analytical trajectory was introduced.

Final desktop evidence on Apple Clang 17, arm64 macOS:

- Python: **1,330 passed, 1 skipped**. The skip is the existing local-socket
  sandbox restriction. Both reduced libraries were built and available, so the
  precision execution suites were not skipped.
- Native Release: **7/7** for float64 and **4/4** for each reduced profile.
  The optional host binder was built only in the float64 Release directory.
- Native Debug and ASan/UBSan: **7/7** float64 and **5/5** for each reduced
  profile in each build type. These isolated builds included the host binder.
  No sanitizer findings were reported. Leak detection was disabled, and Python
  was not sanitizer-instrumented.
- All **11** old/new float64 matrix fixtures match exactly, excluding only
  wall-clock timing. The **49** public structure layouts and **367** member
  offsets/sizes remain unchanged. Deployment images are byte-identical and load
  in both directions. The current bindings also reproduce the same fixture
  records with the original and current float64 libraries.
- The final strict-float32 LLVM audit passed for all **five** runtime
  translation units at both **O0 and O3**, with no wider floating arithmetic,
  comparisons, conversions or calls in the inspected IR. This audit excludes
  the optional host binder and third-party math-library internals.
- Native training tests exercise pair, triplet, and modulated rules from
  initialization, including learned-weight preservation across episode
  reset. Shared-weight and image tests check same-profile execution and reload.
  These are numerical/functional tests, not training-quality benchmarks.
- The two-neuron training example completed in all three profiles with six
  spikes per run. Its learned weight was `0.5066894029199218` in float64 and
  `0.5066894292831421` in each reduced profile, starting from `0.5`. Matching-profile
  JSON reload and image execution retained those weights exactly. Replay states,
  spikes, learning state, integer counters and all 44 trace records matched,
  excluding wall-clock timing. Artifacts and actual library hashes are retained
  in `training/validation.json` and its accompanying files. This is a smoke test,
  not evidence of equivalent training quality.

The retained evidence is under `artifacts/validation/precision_native`:
`python-final.xml`, `native-release.log`, the six profile-specific Debug and
sanitizer logs, `differential-final/report.json`, and
`strict-ir-final/report.json` with compiler commands, hashes and IR files.
Historical intermediate logs are retained separately and are not final results.

Software implementation is complete at this tested boundary. Actual MCU builds,
device math-library behavior, long-horizon suitability, memory use, inference
timing and learning quality remain unmeasured here. They require experiments on
the intended devices and must not be inferred from these desktop passes.
