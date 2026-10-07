# Intrinsic escape-neuron capability

Lacuna supports intrinsically stochastic `EscapeLIF` and
`AdaptiveEscapeLIF` neurons. These are neuron models, not input encoders. Their
conditional spike intensity is

\[
\lambda(v)=\lambda_0\exp\left(\frac{v-v_{escape}}{\Delta_v}\right).
\]

The intrinsic DSL expresses this as a `hazard` instead of a hard threshold:

```text
hazard { rate = escape_rate*exp((v-v_escape)/delta_v) }
```

The first hazard capability accepts algebraically equivalent expressions when
the resolver can prove that the log-rate is affine in the single membrane
state, has a positive prefactor, and increases with voltage. A neuron definition
must contain exactly one `threshold` or `hazard` block.

## Execution contract

For each neuron and each prospective spike, the runtime draws one unit-rate
exponential variate `E` and solves

\[
\int_{t_0}^{t_*}\lambda(v(s))\,ds=E.
\]

The state trajectory inside the integral is the exact analytical trajectory
already selected for LIF or adaptive LIF. Lacuna does not poll a Bernoulli
probability at a timestep. Integration and inversion use explicit relative,
absolute, and time tolerances stored in the resolved hazard descriptor.

An event that reaches a neuron before its prospective spike consumes the hazard
accumulated through that event, applies the event, and reschedules with the same
remaining exponential variate. It does not redraw. This makes the stochastic
process invariant, to the declared numerical tolerance, to harmless event
interruptions and queue invalidation.

After a real spike, the authored reset map is applied and exactly one new
exponential variate is drawn. During fixed refractory clamping, intrinsic hazard
is paused; the new variate remains pending while non-membrane state such as
adaptation continues its authored clamped evolution.

Random draws use a counter-based stream keyed by the run seed, compiled neuron
index, and that neuron's draw index. A fixed graph, seed, and inputs therefore
reproduce the same spike train independently of stale prediction events.

## Equation-derived execution

Repeated hazard integration does not need to reinterpret the full propagation
expression at every quadrature sample. When the resolver proves that the
membrane trajectory has the bounded form

\[
v(t)=v_\infty+\sum_k c_k e^{r_k t},\qquad r_k<0,
\]

it emits roots for the limit, coefficients, and rates. The runtime evaluates
those roots once for the current state and parameter bindings, then samples the
same analytical trajectory with direct exponential arithmetic throughout
quadrature and root inversion. This is a capability of the resolved equations;
it is not selected by a standard-model name.

Modal coefficients become poorly conditioned when distinct rates approach one
another. The compiler therefore withholds this lowering for structurally close
rates, and the runtime also rejects it dynamically if the coefficient
cancellation is too large for the requested accuracy. Both cases use the
existing stable propagation expression (including its `phi1` limit) without
changing the hazard process or relaxing any tolerance.

## Current scope

- `EscapeLIF`: exact stable scalar LIF propagation plus exponential voltage
  hazard.
- `AdaptiveEscapeLIF`: exact two-real-exponential adaptive-LIF propagation plus
  the same hazard family.
- Delta connections, mixed deterministic/stochastic populations, drive updates,
  recording, refractory behavior, and graph persistence are supported.
- Folded and per-edge filtered-current synapses on hazard neurons are rejected at
  resolution time for now. Extending them requires no change to the stochastic
  process, but their analytical trajectory/hazard integration combinations need
  their own validation matrix.

At the high-level API, `CompiledNetwork.run(..., seed=N)` supplies the intrinsic
seed. At the graph/core layer the corresponding argument is
`stochastic_seed=N`.
