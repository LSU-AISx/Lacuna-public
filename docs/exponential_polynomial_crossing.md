# Stable real exponential-polynomial crossing contract

This note defines the mathematical capability used by bounded current-based
multi-state synapses. It is a resolver and evaluator contract, not a general
root finder for arbitrary functions.

## Admissible trajectory

The normalized threshold gap must have the finite form

\[
g(t)=L+\sum_{j=1}^{m}P_j(t)e^{r_jt},\qquad r_j<0,
\]

where the real rates are distinct after exact grouping, each polynomial has
finite degree, and the total number of polynomial coefficients is bounded by
the analytical state limit. Coefficients and rates are evaluated from the
lowered expression program at the prediction boundary and must be finite.

This form follows from a stable affine membrane driven additively by homogeneous
linear current kernels with real Jordan modes. A Jordan block of length `d + 1`
contributes a polynomial of degree at most `d`. Coupling a kernel whose rate is
exactly equal to the membrane rate extends the corresponding polynomial rather
than evaluating a distinct-rate quotient at its singular boundary.

For a scalar current `s' = q s`, equality `q = a` gives

\[
v(t)=v_\infty+(v_0-v_\infty+c s_0t)e^{at}.
\]

For an alpha block `z' = qz`, `s' = qs+z`, equality `q = a` gives

\[
v(t)=v_\infty+
\left(v_0-v_\infty+c s_0t+\tfrac12 c z_0t^2\right)e^{at}.
\]

## Finite horizon

When `L != 0`, doubling starts no earlier than every monomial peak
`degree / abs(rate)` (and the slowest time constant) until

\[
\sum_j\sum_k |c_{jk}|t^k e^{r_jt}<|L|.
\]

Beyond that horizon each absolute monomial envelope is nonincreasing and the
sign is fixed by `L`. When `L = 0`, the block with the largest rate dominates
asymptotically; within that block its highest nonzero polynomial coefficient
dominates. The initial horizon is also placed beyond every faster-block ratio
peak. Doubling continues until the dominant leading monomial exceeds its own
lower-order absolute bound and that remaining margin exceeds the normalized
envelope of every faster block. Those ratios can only decrease afterward.
Failure to establish either bound within the fixed doubling limit is
nonconvergence, never an assumed absence of a crossing.

## Complete exact-arithmetic isolation

For one exponential block, roots are precisely the roots of its polynomial and
are isolated recursively using derivative roots as monotone partitions.

For multiple blocks, choose a block with rate `r` and polynomial degree `d`.
The operator

\[
\mathcal L_r=(D-r)
\]

satisfies

\[
\mathcal L_r(P(t)e^{rt})=P'(t)e^{rt}.
\]

Applying it `d + 1` times eliminates the chosen block while preserving the
exponential-polynomial class for every other block. Recursively isolate the
reduced function, then recover the roots of each preceding function in reverse
order. If `H = (D-r)F`, then

\[
\frac{d}{dt}(e^{-rt}F(t))=e^{-rt}H(t).
\]

Consequently the roots of `H` partition `e^{-rt}F`, and therefore `F`, into
monotone intervals. Endpoint zeros and sign-changing intervals recover all
roots, including repeated roots in exact arithmetic. Each elimination reduces
the total basis dimension, so recursion terminates. The real exponential-
polynomial basis is an extended complete Chebyshev system; a nonzero function
with `N` coefficients has at most `N - 1` real zeros counting multiplicity.

## Floating-point policy

Evaluation uses a positive exponential normalization, long-double accumulation,
and explicit magnitude-based rounding bounds. Isolation roots use a machine-
scale tolerance; scheduled spike time uses the configured simulation tolerance.
A root is a spike only when signs on the adjacent root-free intervals change
from positive gap to negative gap. Tangencies are not spikes. Numerically
indistinguishable roots, unresolved signs, exhausted iteration budgets, or a
failed horizon return nonconvergence rather than inventing or dropping a spike.

The existing scalar-log, one-alpha, two-real-exponential, and pure
multi-exponential capabilities remain separate fast paths. This capability is
selected only when repeated real modes or multiple polynomial-exponential
blocks actually require it.
