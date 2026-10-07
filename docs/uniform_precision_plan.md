# Uniform runtime precision

Requirement updated 2026-09-14 following user review. This document records the
implementation contract and its remaining migration work.

## Required profiles

The intended public choices are `float16`, `float32`, and `float64`. Each choice
applies uniformly to model values, weights, learning state and updates, expression
arithmetic, event timestamps, numerical integration, root and hazard safeguards,
encoders, decoders and runtime floating-point diagnostics. Select the profile
before initialization and training and retain it through export and deployment.

Mixed float16/float32 arithmetic and the existing `float32-time64` profile are
not part of the intended public contract. Float16 storage backed by float32
calculation must not be presented as uniform float16 execution. The same rule
applies to float32 and float64. Wider floating-point accumulators, clock values,
solver workspaces or mathematical-function internals are not permitted in a
uniform profile. Ordinary integer indexes, counters and bit manipulation are
not floating-point widening.

This contract concerns compiled runtime execution. Host-side symbolic analysis,
reference calculations and lossless file transport remain separate. No trained
network is silently converted to another profile, and no unsupported image is
relabelled to bypass compatibility checks.

## Current implementation versus the target

The pushed `float32-support` baseline supports strict float32, the default
float64 profile and a mixed float32/time64 profile. It does not implement
float16. The subsequent `codex/float16-support` work implements strict binary16
including elementary functions, guards, timestamps, learning and deployment.
See `docs/float16.md` for its tested scope and target limitations.

The existing float64 safeguards still use `long double`, which may be wider than
binary64 depending on the target. The earlier float64 compatibility gate
preserved that behavior intentionally. Meeting the new uniform contract requires
reviewing those guards, rather than merely renaming the profile.

Strict float32 passed the local runtime arithmetic audit, but that audit did not
inspect third-party math-library internals or certify an MCU toolchain. Uniform
precision claims must identify the actual linked implementation and target.

## Implementation order

1. Establish an auditable binary16 arithmetic and math-function backend. Require
   IEEE binary16 semantics and controlled rounding. Reject compiler configurations
   that promote half expressions or emit wider floating-point helper calls.
   Function names and half-typed LLVM IR alone are insufficient evidence. Audit
   final machine code and linked helpers as well. Use independent high-precision
   host references for validation, never as part of target execution.
2. Separate arithmetic width from platform implementation so uniform widths apply
   consistently to state, clocks and numerical guards. Remove mixed-profile
   selection with an explicit compatibility diagnostic. Keep its old identifiers
   reserved and reject its artifacts rather than interpreting them as another
   profile. Review float64 guards and version any changed arithmetic contract.
3. Rework numerical constants and safeguards for representability and error
   control at each width. Existing Dormand--Prince constants include `92097` and
   `339200`, which cannot be represented individually in binary16 even though
   their ratio is finite. The existing absolute tolerance `1e-10` rounds to zero.
   Derive validated constants and policies rather than applying a blanket cast
   or scaling every epsilon-based rule. Preserve analytical trajectory families.
4. Extend target-aware compilation, native interfaces, Python transport, images,
   learned snapshots and recording with explicit uniform-profile identities.
   The Python boundary needs a checked half-value transport because the current
   ctypes interface only handles native float and double arguments directly.
   A host transport shim must remain outside the embedded execution runtime.
5. Validate model, synapse, learning and event behavior at each width, including
   representability failures, small updates, near-threshold and near-tangent
   trajectories, rate collisions, clock collapse, resets, image round trips and
   mixed-model graphs. Keep existing float32 and float64 regression evidence and
   report intentional differences rather than replacing reference results.
6. Validate the selected MCU compiler, linked math implementation and final
   firmware. Measure numerical behavior, memory and timing independently. Native
   half type support is not evidence of hardware half arithmetic or a speedup.

## Failure and accuracy contract

Uniform float16 has limited range and resolution. With milliseconds as the time
unit, binary16 timestamps near `1000` are spaced `0.5` milliseconds apart. A
binary16 weight of `0.5` remains `0.5` after adding `0.0001` and rounding. These
are representation properties, not measured network-performance results.

Small representable updates may round away under the declared arithmetic. Do
not retain them in a hidden wider accumulator. Positive time intervals that
cannot advance the clock, invalid target coefficients and unresolved crossings
must produce explicit diagnostics. Do not silently widen, shift an event or
substitute another neuron model. Some configurations may therefore be supported
in float32 or float64 but rejected in float16.

Analytical propagation refers to the model's trajectory formula, not infinite
precision. Numerical implementations of elementary functions need separate
accuracy validation. Any new approximation must be documented and tested, not
introduced implicitly as a substitute for the analytical model.
