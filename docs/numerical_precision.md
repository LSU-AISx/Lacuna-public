# Native numerical precision

The implementation below describes the current code. The subsequent requirement
is three uniform profiles, float16, float32 and float64, with no mixed or wider
runtime arithmetic. Strict float16 is now implemented as described in the
[FP16 guide](float16.md). See the
[uniform-precision plan](uniform_precision_plan.md) for the new contract and the
required migration of existing mixed profiles and wider float64 safeguards.

Lacuna supports the following native execution profiles. Select one before initializing
and training a network, then keep that profile for evaluation and deployment.
This is not a utility for quantizing a float64-trained network.

| Profile | State, weights, learning and expression arithmetic | Event timestamps | Numerical safeguards |
| --- | --- | --- | --- |
| `float64` (default) | binary64 | binary64 | Existing `long double` guards |
| `float32-time64` | binary32 | binary64 | binary64 guards |
| `float32` | binary32 | binary32 | binary32 guards |
| `float16` | binary16 | binary16 | binary16 guards |

The mixed profile remains for compatibility, not as an intended uniform profile.
Its retirement and the review of existing float64 guards remain separate work.

All neurons and connections in one compiled graph use the same profile. Each
profile uses the same equation-derived execution plans and sparse C scheduler.
The reduced profiles do not replace analytical trajectories with polynomial
approximations. Analytical evaluation still has floating-point rounding error,
and numerical threshold searches have a finite tolerance.

## Build and select a profile

From the repository root, using the Python environment already installed for
Lacuna, build and test the usual float64 and float32 libraries:

```sh
python scripts/build_precision_profiles.py --jobs 4
```

Use `--profiles float16` to build the native-half backend on a supported target.
The legacy mixed build can still be requested explicitly with
`--profiles float32-time64`. Libraries are placed in their matching `build-*`
directories, with float64 in `build`.
The engine selects the matching directory automatically:

```python
from lacuna import Engine

engine = Engine(precision="float32")
with engine.compile(network) as simulation:
    result = simulation.run(100.0)
print(result.precision)
```

Use `precision="float32-time64"` for float32 model values with a float64 clock.
`Engine(precision="float32", time_precision="float64")` is an equivalent spelling.
`Engine()` retains the original float64 behavior. An explicit library path may
be supplied as the first argument. Missing or mismatched libraries cause an
error, never a fallback to a different profile. Independent engines may load
different profiles within one Python process.

For a single strict-float32 native build:

```sh
cmake -S . -B build-float32 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLACUNA_REAL_BITS=32 -DLACUNA_TIME_BITS=32 \
  -DLACUNA_ENABLE_PROFILING=OFF -DLACUNA_BUILD_TARGET_BINDER=OFF
cmake --build build-float32 --parallel 4
ctest --test-dir build-float32 --output-on-failure
```

Native applications must compile against headers with the same
`LACUNA_REAL_BITS`, `LACUNA_TIME_BITS`, and profiling definitions as their library.
CMake propagates these definitions when linking its `lacuna_core` target. The
integer-only `lc_numeric_profile_check` and `lc_numeric_property` calls can check
the library contract before calling functions with precision-bearing arguments.

## Training and saving

Authored parameters, literals, initial state, weights, codec settings and learning
parameters are prepared in the target representation before model resolution.
The compiler evaluates derived constant expressions using the selected C
evaluator, not by casting an already-specialized float64 descriptor. Mutable
drives and runtime expressions remain runtime expressions. Unsafe rate
collisions, lost threshold/reset separation and nonrepresentable values fail
explicitly. The original authored network is not mutated.

Pair, triplet and modulated learning, including supported learning programs and
shared weights, execute in the selected runtime. Host-provided reward values are
validated and rounded at the boundary. This does not make an arbitrary external
Python reward or optimization algorithm float32. If such a calculation is part
of an experiment's precision contract, its arithmetic must be specified and
validated separately.

Run the small native-training example:

```sh
PYTHONPATH=src python examples/native_precision_training.py --precision float32
```

It trains a two-neuron Pair-STDP network and writes a learned JSON network and
compiled deployment image under `artifacts/native-precision`. This demonstrates
native learning and export, not classification accuracy or MCU performance.

To save and resume learned weights in an application:

```python
from lacuna import Engine, Network

engine = Engine(precision="float32")
with engine.compile(network) as simulation:
    result = simulation.run(100.0, inputs=inputs)
result.save_learned_network("learned.json")

learned = Network.load("learned.json")
with engine.compile(learned) as simulation:
    simulation.save_compiled_graph_image("learned.lcg")
    next_result = simulation.run(100.0, inputs=next_inputs)
```

The learned snapshot stores the profile under reserved network metadata
`lacuna_precision`. A conflicting engine profile is rejected. It preserves
learned weights, not a live event queue, current neuronal state or eligibility
traces. It is a new-episode initialization, not a mid-run checkpoint. An authored
network without precision metadata may be compiled for any supported profile.

Export the learned snapshot rather than the graph compiled before training.
Compiled graphs are immutable and retain their initial weights. Changing weights
are owned by the run, so serializing the original compiled graph does not capture
the run's learned weights.

## Numerical and clock limits

Float64 tolerances and formulas retain their existing behavior. Float32-profile
Dormand--Prince integration defaults to a relative local-error target of
`64 * 2**-23`, an absolute target of `1e-10`, and an event-time target of
`8 * 2**-23` in the network's time units. Analytical root hints use a relative
target of `8 * 2**-23`. These are local numerical targets, not guarantees of a
global state or spike-time error. Event localization may terminate at adjacent
representable timestamps when a tighter clock interval does not exist.

Overflow in an intermediate reduced-precision integration trial rejects that
trial and retries from the accepted state with a smaller step. An invalid
accepted state still fails. Retry, subdivision and root-search budgets remain
bounded. Ambiguous near-tangencies return nonconvergence rather than inventing
a spike or declaring that none exists.

Clock precision limits long runs. For example, binary32 timestamps around
`2**24` have spacing 2 in the selected time units. A positive delay of 1 cannot
always advance that clock. The runtime rejects collapsed positive intervals
rather than converting them into zero-delay events or silently shifting them.
It does not implement clock rebasing or integer timestamps. A float64 clock
extends the representable time range, but does not recover information lost in
float32 state calculations. Elapsed intervals entering model arithmetic must
also be representable as `lc_real_t`, even in the mixed profile.

An authored equation's `Time` variable is a model scalar. In the mixed profile
the scheduler retains binary64 timestamps, while `Time` is explicitly converted
to binary32 before equation evaluation. Out-of-range conversions fail. A model
that requires float64 absolute-time sensitivity should use the float64 profile.

`analyze_precision(network, "float32", time_horizon=10000.0)` remains an optional,
conservative, non-executable preflight report. It does not replace native
compilation or validate every dynamic singularity. Numerical-policy metadata
likewise describes the implementation and defaults, not a safety certificate.

## Deployment, recording and verification

The default profile retains ABI 17 and byte-compatible version-1 images. Reduced
float32 profiles use ABI 18 and float16 uses ABI 19, with version-2 images,
independent scalar/time widths and
an arithmetic revision. Loaders reject a different profile before reconstructing
the graph. Image loading performs no quantization or symbolic compilation.
Initial state and codecs still travel separately from the graph image. See
[compiled graph images](compiled_graph_images.md) for the complete format.

Native trace callbacks and state inspection use the selected scalar and clock
types. Host trace artifacts use binary64 transport to store these values
losslessly and record the execution profile separately. Reconstruction requires
a matching evaluator and target-resolved graph. Wider file fields do not change
the arithmetic that produced the recorded values.

Strict float32 and float16 disable the built-in wall-clock kernel profiler so it does not
introduce a double-precision dependency. Its `kernel_seconds` field is zero
because timing is disabled, not because execution took zero time. Integer event
and queue statistics remain available. Use an external target timer for MCU
measurements.

Reduced builds disable fast-math and floating-point contraction. Strict builds
also reject implicit double promotions. The retained LLVM audit checks all five
runtime source files at both `-O0` and `-O3` for wider arithmetic, comparisons,
conversions and calls:

```sh
python scripts/audit_strict_float32.py --output artifacts/strict-float32-audit
```

The audit requires Clang and a new output directory. It establishes the property
for the inspected compiler output, not for untested MCU toolchains or the
internals of their math libraries. The optional host constant-binding companion
is separate and never linked into the execution runtime.

Desktop tests cover analytical, stepped, stochastic and mixed-model execution,
plasticity, codecs, images, callbacks and profile mismatches. They do not establish
cross-platform bitwise identity, learning quality, MCU timing or memory savings.
Before reporting embedded results, build with the actual target toolchain and
validate its math library, subnormal handling, clock horizon and memory capacity.
