# Uniform FP16 execution

The `float16` profile uses IEEE binary16 for state, parameters, weights, learning
updates, timestamps, numerical integration, and crossing and hazard safeguards.
It is not FP16 storage with FP32 computation. Select the profile before training.
The existing float32 and float64 profiles are unchanged by this addition.

## Build and use

The current backend requires native `_Float16` arithmetic and an audited GCC or
Clang configuration. Builds require `__ARM_FEATURE_FP16_SCALAR_ARITHMETIC` or
`__AVX512FP16__`. A compiler accepting `_Float16` without native half arithmetic
is not sufficient. No software-arithmetic fallback is implemented.

```sh
python scripts/build_precision_profiles.py --profiles float16 --jobs 4
PYTHONPATH=src python examples/native_precision_training.py --precision float16
```

The first command builds `build-float16` and runs the native tests. The second
trains a small Pair-STDP network in native half precision and saves its learned
JSON network and deployment image under `artifacts/native-precision`.

```python
from lacuna import Engine

engine = Engine(precision="float16")
with engine.compile(network) as simulation:
    result = simulation.run(30.0, inputs=inputs)
learned = result.save_learned_network("learned-half.json")
with engine.compile(learned) as simulation:
    simulation.save_compiled_graph_image("learned-half.lcg")
```

Image loading requires a matching half runtime. Profile identifier 4, ABI 19,
image version 2, scalar/time widths 16/16, and arithmetic revision 1 are checked
before graph reconstruction. Floating wire fields occupy two bytes and preserve
their bits. Images contain the compiled graph, not a live run checkpoint.
Initial state and codec configurations remain separately supplied as documented
in [compiled graph images](compiled_graph_images.md).

Native applications build with `LACUNA_REAL_BITS=16`, `LACUNA_TIME_BITS=16`, and
`LACUNA_ENABLE_PROFILING=0`. Use `-ffp-eval-method=source` for Clang or
`-fexcess-precision=16` for GCC, with `-fno-fast-math -ffp-contract=off`. CMake
propagates these requirements to consumers of its `lacuna_core` target. Set
`LACUNA_BUILD_HALF_HOST=OFF` and `LACUNA_BUILD_TARGET_BINDER=OFF` on embedded
builds. Neither host companion is required to load or execute a graph image.

Python uses two-byte scalar and structure fields plus a separate integer-bit
transport shim for by-value half arguments. The shim calls the exact selected
core function address, so different precision libraries may coexist in one
process without sharing arithmetic or relying on dynamic symbol precedence.
Host JSON and trace transport may use wider fields to preserve half values
losslessly. That transport is not the arithmetic used for simulation or learning.

## Math backend and storage cost

Native half instructions perform basic arithmetic. Elementary functions use
integer-indexed, nearest-even binary16 lookup tables. There is no runtime libm
dependency or widening to float32 to evaluate these functions. The tables provide
`exp`, `expm1`, `log`, `log1p`, `sqrt`, `sin`, `cos`, `tanh`, and the removable
singularity helpers `phi1` and its derivative. They evaluate elementary functions,
not replacement neuron trajectories. Analytical model formulas are retained.

The first implementation has **1,113,472 bytes of read-only math table data**.
It shares repeated table blocks but currently links the whole table set, without
per-network pruning. This fixed flash cost can outweigh the savings from
two-byte network state on small MCUs. FP16 support alone is not evidence that a
network fits on a board or runs faster. No MCU measurements are claimed here.

General `pow` is bounded. Supported fixed exponents are `-2`, `-1`, `0`, `0.5`,
integers `1` through `8`, and the half-rounded values of `-0.2` and `0.2`.
Adaptive error control uses `-0.2`. The compiler rejects other exponents and exponents that
depend on state or mutable drives before graph allocation. Direct low-level
math calls return NaN for unsupported powers, which expression evaluation
reports as a numerical failure. Extending this set
requires additional validated math support, not repeated multiplication passed
off as a correctly rounded general power function.

Table generation is offline. The checked-in generator evaluates functions with
mpmath interval arithmetic and requires both interval endpoints to round to the
same binary16 value. All 1,376,256 entries were generated, with finite non-special
cases resolved using 160-bit intervals. Zeros, infinities, NaNs, and exact special
cases are assigned explicitly. Regeneration with the same generator at 240 bits
produced identical table bytes.
Runtime tests compare every entry against the generated row hashes, exercise
special values, independently check representative results with Decimal, and
exhaustively compare square and reciprocal tables with native half arithmetic.
The helpers do not implement libm `errno` or floating-exception side effects.

Execution requires round-to-nearest, ties-to-even and gradual underflow. Native
entry points check this using half arithmetic probes, including subnormal inputs
and outputs. They reject incompatible rounding or flush-to-zero environments.
The caller must retain the validated floating-point environment throughout each
call, including callbacks. The runtime does not silently change processor modes.

## Numerical limits

Binary16 has 11 significant binary digits, largest finite magnitude 65504,
smallest normal positive value `2**-14`, and smallest positive subnormal
`2**-24`. Nonzero authored values that underflow, overflowing parameters, merged
threshold/reset values, and unsafe decay-rate collisions are rejected. A
representable learning update may still round away when added to a larger
weight. It is not saved in a wider hidden accumulator. For example, adding
`0.0001` to `0.5` in half arithmetic leaves `0.5`.

Time uses the same precision. Timestamps near 1000 are spaced 0.5 apart. If the
time unit is milliseconds, that is 0.5 ms. A positive synaptic delay, refractory
period, or predicted crossing that collapses onto the current time is an error.
Automatic clock rebasing and integer timestamps are not implemented.

The half adaptive integrator retains Dormand--Prince 5(4). Constant rational
coefficients are rounded directly to binary16 at compilation, rather than
overflowing their individual numerators or denominators. Defaults are relative
local-error target `8 * 2**-10`, absolute target and minimum step `2**-24`,
event-time target `2**-10`, initial step `0.001`, and maximum step `0.25` in model
units. Analytical root hints use relative target `2 * 2**-10`. These are local
targets, not a global error bound or certification of all half-precision models.
If the preferred initial step is smaller than the next clock increment, the
integrator starts with that increment, provided it respects the configured
maximum. It integrates the interval represented by the chosen endpoint and
still rejects trials that fail error control. It never moves a physical event
forward merely to make it representable.

Root and hazard checks remain conservative. An ambiguous sign or threshold
tangency fails rather than silently changing the model, using a wider guard, or
inventing a spike. For example, the existing folded-alpha cancellation fixture
cannot be certified in half and reports root nonconvergence. The visual-cortex
triplet preset's `a2_plus=5e-10` is not representable and is rejected. Other
representable pair, triplet, and modulated configurations execute and learn in
half precision. A configuration that works in float32 need not work in float16.

## Validation and remaining deployment work

Desktop tests cover reactive and scalar LIF, adaptation, normalized QIF, AdEx,
mixed analytical/stepped graphs, exponential and alpha per-edge currents,
equal-decay dynamics, plasticity, shared weights, codecs, recording, image
replay, and explicit representation failures. QIF crossings are checked against
the independently known `pi/4` solution. These are bounded test cases, not
learning-quality benchmarks or a universal model-accuracy claim.

```sh
python scripts/audit_strict_float16.py --output artifacts/strict-float16-audit
```

The audit checks all six runtime translation units at `-O0` and `-O3` in LLVM IR
and final assembly, then checks linked external symbols. The development-host
Apple Clang arm64 audit found no wider floating arithmetic or math helpers.
This result does not certify a different target, compiler, or build flags.
Numerical-policy metadata therefore retains the half capabilities as
`uncertified`. Before deploying, audit the actual firmware and measure memory,
clock resolution, numerical behavior, and performance on the selected MCU.

The 2026-09-14 development-host validation completed with 1,484 Python tests
passing and one existing local-socket test skipped by the sandbox. Native suites
passed 7/7 for float64, 4/4 for each float32 profile, and 5/5 for float16. The five
FP16 native suites also passed AddressSanitizer and UndefinedBehaviorSanitizer
with leak detection disabled. The float64 compatibility gate retained bit-exact
results for all 11 fixtures, unchanged layouts for 49 public structures and 367
members, and byte-identical deployment images with bidirectional loading.

Detailed local evidence is retained under `artifacts/validation/precision_native`:
`final-fp16-pytest.xml`, `final-float16/report.json`,
`half-sanitizers-final.log`, and `final-fp16-float64-differential/report.json`.
