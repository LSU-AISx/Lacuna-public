# Reduced-precision codecs and intrinsic hazards

These rules supplement the numerical-profile contract. Model values, rates,
phase accumulators, decoder outputs, and modal voltages use `lc_real_t`.
Timestamp differences, latency interpolation, and phase/rate intervals use
`lc_time_t`. A clock difference that is passed into model arithmetic is checked
after conversion: overflow or loss of a positive interval to zero returns
`LC_NUMERIC_ERROR`. Float64 timestamps do not make float32 state calculations
double precision. In particular, a finite binary64 elapsed interval larger than
`FLT_MAX` still fails when it is required by a binary32 model calculation.

## Encoders and decoders

Regular and Poisson encoders reject a positive interval that cannot advance the
clock, including two later events that would round to the same timestamp. Burst
encoders perform the same check between successive events. A zero phase at an
open presentation boundary remains valid and is emitted once when that boundary
is included. Insufficient output capacity remains a separate overflow status.

Poisson streams are deterministic for a given profile, seed, stream identifier,
and call sequence. Float32 draws use 23-bit half-bin uniform values strictly
inside `(0, 1)`, followed by the profile's logarithm. They are not expected to
reproduce the float64 stream's transformed timestamps. Native-event encoders do
not turn scalar presentations into events; their already-encoded events enter
the network directly.

Decoder normalization, rates, and temporal weighting are evaluated in model
precision after checked conversion of elapsed time. Nonfinite decoded values
produce `LC_NUMERIC_ERROR`, including in retained streaming events; they are not
reported as valid results. The event timestamp fields retain clock precision.

## Float32 integrated-hazard safeguards

A scalar constant-rate shortcut is permitted only for zero state and zero
affine drive. Equality with the rounded value of `-b/a` is insufficient because
the actual scalar propagation formula can move that state by an ULP. Reduced
scalar hazard evaluation uses the same propagation routine as ordinary scalar
advancement. A modal shortcut requires all-zero coefficients and a limit equal
to the current readout. Neither an epsilon-sized distance to equilibrium nor an
unchanged short rounded probe proves a constant trajectory. Both float32
profiles evaluate modal voltage with float32 products and sums. The declared
wider type may still be used by quadrature/search guards in the mixed profile.
Strict float32 uses float32 guards.

Float32 rate evaluation preserves representable subnormal `expf` results.
An overflowing exponential is a numerical failure, not a capped finite rate.
Waiting-time division uses clock precision. A positive predicted wait that does
not advance the clock is a numerical failure; reduced profiles do not move the
event to the next representable timestamp.

For reduced profiles, `lc_hazard_config.maximum_root_iterations` bounds both
the iterations of an individual crossing inversion and the number of forward
search segments (each no larger than 512 simulation-time units). Exhausting
either budget returns `LC_ROOT_NONCONVERGENCE`, not `LC_NO_CROSSING`. Collapsed
quadrature subdivisions likewise report nonconvergence. These explicit limits
prevent a very long, nonconstant search from doing unbounded practical work.
Increase the configured budget deliberately when a validated problem needs a
longer search; doing so does not repair unrepresentable time or state values.

The original float64 hazard shortcuts, rate cutoffs, and boundary handling remain
unchanged for compatibility. Native regression tests cover constant and varying
hazards, near-equilibrium gain amplification, rounded short-probe ambiguity,
search exhaustion, clock collapse, and same-profile deployment-image execution.
These desktop tests do not establish MCU performance or cross-platform bitwise
identity.

## Binary16 environment and failure boundaries

Binary16 execution requires round-to-nearest, ties-to-even arithmetic and gradual
subnormal input/output behavior. The runtime probes the current floating-point
environment on each numerical graph, run, codec, and image operation; an
incompatible environment returns `LC_NUMERIC_ERROR`. The checks are not cached,
and the runtime does not change the caller's rounding or flush modes. Callers,
including callbacks, must retain the validated environment for the entire call.
Pure copy, information, and destruction operations do not require this check.

Binary16 RNG transforms use ten-bit half-bin uniforms strictly inside `(0, 1)`.
Hazard quadrature uses the authoritative trajectory DAG when a modal-cancellation
bound has not been established for binary16. Overflowing quadrature intermediates
fail explicitly instead of being saturated into a successful integral. A positive
numerical crossing that collapses to the current timestamp is a numerical error.
Derived nonzero root coefficients that underflow to zero report nonconvergence;
they cannot justify a claim that no crossing exists. These checks do not provide
a global trajectory or learning-error bound.
