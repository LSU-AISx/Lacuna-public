# Lacuna: Authoritative Design Specification

An equation-derived, event-driven spiking neural network engine.

> **Status:** This is the authoritative current-design document for Lacuna. It
> describes behavior implemented in the repository, the invariants that new
> work must preserve, and explicitly marked deferred extensions. It was
> reconciled against the implementation and tests on 2026-08-20. When this
> document, the README, an experiment note, and historical discussion disagree,
> this document governs the architecture; executable code and tests govern
> behavior until the discrepancy is corrected here.

Numerical profiles, target-aware compilation and related persistence contracts
were updated and validated on 2026-09-12.

### Current compatibility identifiers

| Artifact or boundary | Current value | Compatibility rule |
|---|---:|---:|
| Python distribution | `lacuna-snn` `0.1.0.dev0` | Python 3.10 or newer |
| Canonical graph schema | 10 typed, 11 mixed | loader accepts 9, 10 and 11 |
| High-level network schema | 1 | loader accepts 1 |
| Persisted trace schema | 1 | reader accepts 1 |
| C shared-library ABI | 17 for float64, 18 for reduced profiles | exact ABI and numerical-profile match required |
| Compiled graph image | 1 for float64, 2 for reduced profiles | exact image, ABI and numerical-profile match required |
| Symbolic resolver | SymPy 1.14.0 | version pinned |

The graph writer preserves schema 10 for Dale-typed networks and emits schema
11 when any neuron is mixed. Older schemas cannot encode mixed polarity.
Compatibility acceptance is not a
promise to preserve every development schema indefinitely; changing an accepted
version requires an explicit loader decision and tests. Capability-changing
pull requests must update this document in the same change. Detailed algorithm
documents may refine a named subsystem, notably
[`docs/intrinsic_escape_neurons.md`](docs/intrinsic_escape_neurons.md), but may
not silently broaden the central capability boundary recorded here.

## Purpose and scope

This document specifies Lacuna's architecture, intrinsic model DSL, intermediate
representation (IR), equation resolver, execution plan, C runtime, graph and
trace formats, high-level Python API, learning subsystem, codecs, recording, and
validation contract. Lacuna accepts arbitrary flat directed graphs and allows
different supported neuron, synapse, codec, and plasticity configurations to
coexist in one compiled network.

The current analytical engine accepts stable affine LIF-family trajectories
with certified first-event prediction: deterministic hard thresholds, reactive
deposit-triggered integrate-and-fire, one spike-triggered adaptive-current mode,
bounded exponential and alpha-current input modes, repeated real rates, and an
intrinsic exponential voltage-hazard form for `EscapeLIF` and
`AdaptiveEscapeLIF`. The current deterministic `STEPPED` capability
accepts autonomous ODE systems of one through eight states, a fixed rising
threshold on one designated readout, simultaneous resets, piecewise-constant
parameter updates, delta deposits, and an optional fixed refractory clamp. It
uses adaptive Dormand--Prince 5(4) propagation and complete rising-root
isolation on each accepted step's quartic dense interpolant. AdEx and QIF are
standard nonlinear acceptance families.

Current synapse kinetics are delta, stable one-state exponential current, and
unit-area two-state alpha current. Equal compatible kernels are folded into
postsynaptic state; distinct supported kernels allocate bounded per-edge-derived
state, with at most eight total analytical state components per postsynaptic
cluster. Each edge is either static or binds exactly one learning rule. Pair
STDP, Pfister--Gerstner triplet STDP, split-eligibility modulated STDP,
voltage-modulated STDP, and soft-excursion modulation are implemented;
different edges may use different rules in the same graph.

Moving thresholds, arbitrary stochastic equations and SDEs, algebraic or delay
equations, host callbacks, conductance-coupled synapses, generic stateful
synapse merging into nonlinear nodes, structural plasticity, and differentiated
training evaluators remain outside the current boundary. These are explicit
resolver or runtime capability limits, not permission to approximate silently.

Optional host-side LIF importers accept static externally trained dense and
convolutional weights and parameters. The official Lava-DL SLAYER adapter also
accepts fixed nonoverlapping pooling on evenly divisible spatial dimensions,
lowered to independent per-channel connectivity and ordinary LIF neurons.
It supports a restricted zero-drive LIF/delta configuration and requires explicit
acknowledgment of source fixed-point versus target binary64 state differences.
Both training and source validation use the unmodified external framework.
The imported network uses the existing C engine and persisted formats, with no
SLAYER training code or sampled SRM executor in Lacuna. See
`docs/imported_networks.md` for the public `lacuna.importers` API and deployment
workflow, and `docs/official_slayer_deployment.md` for the source contract and
finite-input validation. Importing this public API does not load Torch or
Lava-DL. Those dependencies are required only for source conversion and
cross-framework validation, not for the framework-neutral builders or C runtime.

The name refers to the gaps between events, which is where the engine does its distinctive work: rather than discretizing the interval from one event to the next, Lacuna evaluates its resolved analytical trajectory and jumps. The PyPI distribution is `lacuna-snn`, since `lacuna` is already taken there, but the import name is `lacuna`.

Lacuna's job is to study networks as dynamical objects. The target hardware is a consumer of the engine, not the definition of it. The first deployment target is a basic microcontroller SNN event processor, but that target must not reach back into the dynamics representation. This separation is the central discipline of the whole design.

The scope is point neurons. There is no spatial state and no axial coupling
between compartments. This is a deliberate boundary, and it keeps each
compiled state program within the current eight-state limit. If
multicompartment models enter scope, they will change the state-size,
propagation, crossing, and memory assumptions together; they cannot be treated
as ordinary point-neuron instances with more labels.

### Current system at a glance

```
intrinsic DSL + explicit graph + ports
                 |
                 v
        Python parse and resolve
                 |
                 v
   equation-derived ExecutionPlan
                 |
                 v
       compiled C graph and codecs
                 |
                 v
 one sparse event scheduler, with each node using
 its proved analytical, reactive, hazard, or STEPPED operations
                 |
                 v
 spikes + decoded events + state inspection + causal trace + learned weights
```

| Area | Executable now | Current boundary |
|---|---|---|
| Graphs | Arbitrary explicit flat directed graphs; supported models, synapses, codecs, and plasticity rules may be mixed | Point neurons only; no runtime hierarchy or multicompartment state |
| Deterministic neurons | `IntegrateAndFire`, `LIF`, `AdaptiveLIF`, `AdEx`, `QIF`, and equivalent accepted custom DSL | Exact execution only for recognized stable real LIF families; other accepted fixed-threshold autonomous ODEs use STEPPED |
| Intrinsic stochastic neurons | `EscapeLIF` and `AdaptiveEscapeLIF` through integrated cumulative hazard | Supported analytical trajectory and delta synapses only; no general stochastic equations or SDEs |
| Synapses | Delta for all current neuron families; stable exponential current and unit-area alpha current for receptor-enabled scalar LIF; compatible modes fold and distinct supported modes use bounded derived state | Filtered current does not yet compose with adaptive, escape, AdEx, or QIF nodes; no conductance coupling; eight analytical state components per postsynaptic cluster |
| Plasticity | Pair, triplet, modulated, voltage-modulated, and soft-excursion; one rule per edge; different rules may coexist | No structural plasticity or public custom learning-program authoring; static shared-weight compilation has the gap in section 12.1 |
| Input codecs | Native events, regular rate, Poisson rate, TTFS/latency, burst, latency burst, and held current | Scalar presentations are normalized; native event amplitudes are arbitrary finite values |
| Output codecs | Finite/sliding/cumulative rate, TTFS, and fixed temporal weighting; exact query events where supported | `ON_QUERY` for sliding-rate, population, and learned decoders are deferred |
| Observation | Raw spikes, explicit-time state inspection, causal traces, persisted trace artifacts, reconstruction, final learning state, and basic spike analysis | Inspection is read-only; learning traces are not neuronal state-recording variables |
| Runtime | The default `ExecutionPlan`/`CompiledNetwork` path uses one sparse C scheduler, analytical gaps where admitted, adaptive numerical gaps otherwise, and deterministic same-time phases | Binary64 remains the default, with binary32 and strict binary16 profiles plus legacy mixed compatibility; no bit-exact cross-platform guarantee |
| Persistence | Canonical self-contained JSON graphs/networks, versioned compiled-graph deployment images, and chunked binary trace artifacts | JSON remains the editable model authority; graph images are ABI-bound derived artifacts and currently exclude run initialization and codecs |
| Deployment | A host can compile once and serialize `lc_compiled_graph`; the C runtime reconstructs it with a matching numerical profile without Python, SymPy, the DSL parser, or the compiler | No production microcontroller build target, full deployment bundle, or hardware-noise profile yet |

## 1. Design invariants

These are the commitments that everything else depends on. They are stated first because the architecture, the schema, and the pipeline are all consequences of them.

The authored dynamics are target-independent. A separate numerical profile
determines how their values and derived expressions are represented and
evaluated. Target-aware compilation checks that the selected representation
preserves an admitted analytical family or reports an explicit limitation.
It does not inject a different neuron equation or hardware-noise model.

Persisted source and executable source are distinct and singular. The persisted
source of truth is the embedded intrinsic DSL plus explicit instance bindings.
After resolution, the executable source of truth is the canonical lowered
equation program: exact trajectory roots for accepted analytical nodes or a
right-hand-side DAG for `STEPPED`, plus reset and either threshold or hazard.
Recognized affine coefficients, singular-safe trajectory formulas, crossing
coefficients, modal hazard trajectories, operation closures, and specialized
arithmetic records are resolver-derived artifacts. A backend consumes those
artifacts and may not re-interpret the authored model or invent a different
approximation.

In the primary compiled path, execution capabilities are peers over one IR and
one scheduler. Analytical
propagation and adaptive numerical propagation are selected per node from its
resolved equations and may coexist in one run. The default
`ExecutionPlan`/`CompiledNetwork` path does not select separate fast versus
general network runners. Public compatibility entry points remain:
`lc_delta_network_run`, `lc_mixed_network_run`, Python `run_delta`, and the
float64 one-shot `ResolvedGraph.run()` path through `compile_mixed`. They are not
automatically selected by the default high-level compiler and must remain
differentially tested while public. The one-shot `ResolvedGraph.run()`
float64 compatibility path accepts pair, triplet, and modulated STDP only; reduced
profiles use the canonical target-compiled path even for a one-shot run. The canonical
`ExecutionPlan`/`CompiledResolvedGraph` path is required for the complete
six-rule learning surface. "Analytical" means that state propagation between
events is evaluated from a closed expression. It does not mean exact real
arithmetic: the implementation uses profile-selected floating-point functions,
ROOT_FIND has a solver tolerance, integrated hazard has quadrature and inversion
tolerances, and `STEPPED` has integration and event-localization tolerances.

Parameters and synaptic magnitudes are explicit numeric bindings rather than
numbers hidden in model source. The resolver validates domains and selects a
structural formula regime, including equal-rate versus distinct-rate forms.
Within an active run, only approved DRIVE parameters may change at an event
boundary, and plasticity may mutate compiled nonnegative weights. All other
parameter, polarity, model, synapse, or topology changes currently require a
new graph resolution and compilation. Structural keys and within-plan program
deduplication exist, along with bounded process-local memoization of successful
parsing and model resolution (section 6.7). There is no public arbitrary-rebinding
API or persistent on-disk resolver cache.

Dale typing is the default graph and evaluator policy, not a mandatory property
of every network. Each neuron instance has an explicit `EXCITATORY`,
`INHIBITORY`, or `MIXED` polarity independent of its equation family. Excitatory
and inhibitory sources retain nonnegative edge magnitudes and supply a fixed
positive or negative sign. An explicitly mixed source instead carries finite
signed edge weights, so different outgoing connections can have different
signs. Ordinary neurons and populations still default to excitatory. All three
types can coexist in the same graph and use the same event scheduler and
equation-derived execution strategies. Native external spike and drive ports
remain exogenous boundaries rather than neuron outputs.

Mixed-sign support does not change neuron equations or numerical tolerances.
Existing online STDP and modulatory learning rules remain restricted to
Dale-typed sources because their current
updates and bounds describe magnitudes. Attaching such a rule to a mixed source
is rejected, rather than silently choosing signed plasticity semantics. See
`docs/mixed_sign.md` for the current API and compatibility contract.

The user writes mathematics, and the resolver recovers the structure it can
prove: affine versus nonlinear dynamics, accepted standard trajectory form,
synapse merge/tier, formula regime, spike-prediction method, deposit operation,
and arithmetic specialization. The DSL still requires semantic state roles such
as membrane, receptor, and adaptation. Generic automatic core/observer
partitioning is not implemented; learning observers are explicitly isolated in
their own compiled arenas.

SymPy lives only at build time. Symbolic parsing, coefficient extraction,
equivalence checks, singular-limit construction, and crossing analysis happen
in the Python resolver and lower to self-contained expression DAGs and numeric
operation descriptors before the C evaluator sees them. No SymPy object and no
Python interpreter participates in runtime equation evaluation. The current
resolver constructs proved trajectory families directly; it does not expose a
general symbolic matrix-exponential pipeline.

There are three synapse tiers and four node dispatch forms. DELTA,
FOLDED_SHARED, and a bounded analytical subset of PER_EDGE are executable;
REACTIVE, CLOSED_FORM, ROOT_FIND, and STEPPED describe state/crossing execution.
Integrated hazard is a crossing strategy on an accepted analytical trajectory,
not a fifth biological node kind. Dispatch, arithmetic, crossing, deposit, and
learning operations are derived from equations and bindings, never requested as
manual performance hints. Equal compatible edge modes are folded; distinct
modes allocate state only when the graph requires them.

Auxiliary learning state cannot enter intrinsic spike prediction directly.
Edge-local learning traces are stored outside neuronal state and have no path
into evolution, threshold, hazard, or reset
programs. Learned weights do affect later predictions through the magnitudes of
ordinary future deliveries. Consequently attaching supported plasticity does
not change a node's intrinsic execution capability. A future general state
dependency partition must preserve this invariant, but the current resolver
does not infer arbitrary observer variables.

## 2. System architecture

### 2.1 The C core and the host/device partition

Runtime equation and event evaluation lives in C. The primary compiled path has
one sparse mixed-capability scheduler, while the compatibility entry points
listed in section 1 retain older C loops. The current build produces a host
shared library and tests; a production microcontroller target is not yet
implemented. The embedded runtime is intended to be a build of this same C
evaluator rather than a second implementation. SymPy and the resolver remain
host-side in Python, and Python is a driver rather than a second evaluator: it
builds and lowers models, calls C, and reads outputs.

The host can serialize an already compiled `lc_compiled_graph` into a versioned
deployment image. The image is a field-wise little-endian encoding rather than
a dump of C structure memory. It contains node and edge descriptors, expression
programs, parameters, sparse connectivity indexes, expression-evaluation plans,
and the complete immutable learning configuration and derived work indexes.
Process-local pointers are represented as checked references and relocated when
`lc_compiled_graph_deserialize` reconstructs an ordinary graph object. The
loader validates the format version, C ABI, numerical profile, declared
length, checksum, expression references, graph layouts, sparse indexes, and
learning maps before returning the graph. It invokes neither the Python
resolver nor any C graph-compilation entry point.

This image is deliberately a compiled-graph artifact rather than a second
canonical network format. The JSON network remains the editable and archival
model representation from which a graph can be recompiled for a different
runtime release. A graph image currently carries exactly the data owned by
`lc_compiled_graph`. Initial neuronal values and timestamps are supplied when
creating `lc_mixed_run`, while named ports, encoder sessions, and decoder banks
remain separate runtime objects. A future deployment bundle may package those
records together, but the embedded target does not need the model compiler to
load or execute the graph itself.

To preserve that path, module boundaries are drawn along the future
host-versus-device line now. This is an architectural dependency rule, not a
claim that the embedded target already builds.

Device-eligible modules, which must not depend on anything host-only, are the
event queue, expression evaluation, analytical and numerical state advance,
threshold and integrated-hazard prediction, delta and programmed deposits,
bounded event-driven learning programs, and the compiled codec state machines.
Host-only modules include DSL parsing and symbolic resolution, JSON
persistence, high-level construction, trace-file storage, visualization, and
run orchestration. The current shared library contains the evaluator and codec
runtime; Python owns codec authoring and translation but not their live state
machines. The dependency direction must permit a later embedded extraction
without pulling Python, SymPy, filesystem, browser, or unbounded host services
into the device core.

### 2.2 The Python wrapper

The wrapper is a thin `ctypes` foreign-function interface against the C core
compiled as a shared library. It is not a Python extension module. Keeping it a
thin binding preserves the C core as a first-class standalone artifact that is
also callable from Python. The wrapper must not become load-bearing: live
simulation logic stays in C, while Python owns authoring, resolution, lowering,
orchestration, projection, and offline analysis.

### 2.3 Standard-model construction API

Python exposes typed high-level constructors for the standard executable neuron
families: deposit-triggered `IntegrateAndFire`, deterministic `LIF` and
`AdaptiveLIF`, stochastic `EscapeLIF` and `AdaptiveEscapeLIF`, and stepped
`AdEx` and `QIF`. A constructed family owns a graph model id and shared
parameter defaults. Its `model` property produces the ordinary embedded
`GraphModel`; its `node(id, ...)` method produces a `GraphNode` with validated
per-node overrides and a sensible default initial state; `authored_model`
exposes the same typed equation IR accepted by the low-level resolver; and
`resolve(...)` supports direct evaluator-oriented use. This layer is authoring
sugar only. It generates the canonical intrinsic DSL, enters the same
structural resolver, and never propagates state or evaluates equations in
Python.

The standard LIF equation is `tau_m*dv/dt = -(v-v_rest) +
resistance*drive`. An opt-in `synaptic_input` form adds the declared current
receptor `i_syn` and the term `resistance*i_syn`; this is the form used by the
node-scoped and PER_EDGE filtered-current merge. Keeping the receptor opt-in
preserves the one-state delta LIF and allocates filtered-current state only for
graph nodes whose synapse mappings require it. In the current resolver,
exponential and alpha current synapses compose only with this receptor-enabled
scalar LIF form; adaptive, escape, AdEx, and QIF families currently accept
delta deposits only. `AdaptiveLIF` adds one
independent exponentially decaying spike-triggered current and selects the
exact two-real-exponential capability. The escape variants replace the hard
threshold with a supported intrinsic hazard while retaining the same exact
deterministic trajectory. AdEx and normalized QIF select STEPPED. Constructor
defaults are shared model defaults, while node keyword arguments remain
node-local bindings. Serialization writes complete bindings, so later default
changes cannot alter a saved graph.

The network-level authoring layer is `NetworkBuilder`. It creates named
populations or individual neurons from those standard families, accepts scalar
or per-node parameter, initial-state, and polarity values, assigns deterministic
integer node ids, and returns stable selections. Polarity is stored on each
graph node, not inferred from a model name or metadata. `Delta`,
`ExponentialCurrent`, and unit-area `AlphaCurrent` are the high-level synapse
families. `AllToAll`, `OneToOne`, seeded `FixedProbability`, exact seeded
`FixedOutDegree` and `FixedInDegree`, one-dimensional `LocallyConnected`,
two-dimensional convolutional `Convolution2D` with shared weights, and explicit
connection pairs are construction-time patterns. Seeded uniform and normal
distributions materialize edge magnitudes and delays. `build()` expands all
topology to explicit edges and seals the builder; negative weights are rejected.
The immutable `Network` is backed by exactly one flat `Graph`; populations,
kernels, reservoirs, and labels remain authoring metadata rather than runtime
hierarchy.

`NetworkBuilder.reservoir()` is the high-level constructor for a named recurrent population. It accepts any ordinary neuron family, synapse, connection pattern, weight/delay values or distributions, parameter/initial-state expansion, explicit per-neuron polarity, and a seed. When polarity is omitted it deterministically assigns approximately eighty percent excitatory and twenty percent inhibitory neurons from the seed. Its default recurrent topology is a deterministic seeded ten-percent `FixedProbability` graph without self-connections, but local, fixed-degree, all-to-all, or explicit topology can be selected without changing the runtime representation. It returns a `Reservoir` selection that behaves like its underlying population and additionally identifies the recurrent edges. Reservoir name, membership, and recurrent edge ids are independently hashed authoring metadata retained across save/load; polarity is semantic node data in the graph itself. Reservoir metadata does not alter graph semantics or create a special reservoir executor: compilation still receives only explicit ordinary nodes and edges, so capability-driven execution and dependency pruning apply unchanged.

Each input or output population binding expands to one explicit port per neuron,
so encoders and decoders remain independently selectable at neuron granularity.
A named modulator port targets an explicit set of compatible third-factor edges;
each modulated edge belongs to exactly one such port. `Network.validate()`
resolves the graph and reports each node's dispatch form, synapse tier, and
executable state count before compilation. `Engine.compile(network)` owns the C
compilation boundary and returns a reusable `CompiledNetwork`. Named spike
trains, drive series, scalar presentations, modulation series, and a run seed
are translated into typed runtime records. The high-level seed is passed to
both the encoder bank and intrinsic-hazard scheduler; lower-level graph APIs
expose those seeds separately. The C evaluator remains the only runtime
evaluator. `CompiledNetwork.start_run()` wraps the
resumable scheduler: open advances exclude their right boundary, input may be
supplied exactly at the previous frontier, encoder/decoder/learning/hazard state
stays in C, and `finish()` processes the final boundary. A selected
persisted-trace writer stays open across advances and receives one contiguous
causal sequence.

The low-level C/graph incremental run additionally supports episode reset. It
restores authored neuronal state, clears queued work, encoder phase, edge
learning traces, and neuron-local learning observers, and preserves learned
weights. It preserves each intrinsic-hazard draw counter and consumes the next
unused variate instead of replaying old noise. This operation is not yet exposed
by `IncrementalSimulationRun`, and an attached streaming decoder must currently
be reset or recreated separately.

High-level recording is a plan over three independent products: retained output spikes, explicit-time state inspection, and the causal trace. `Every` and `AtTimes` expand to inspection requests and do not introduce an integration timestep. A trace selection either remains in the bounded run result or streams directly to the existing chunked binary trace artifact. Structured run results expose graph-id spikes, sampled and final state, final edge weights, compact plasticity state, decoded values and events, statistics, and optional interoperability with NumPy or pandas. `learned_network()` freezes the final weights into a new immutable graph, and `save_learned_network()` persists that snapshot through the normal canonical network format. Offline spike-count, finite-window firing-rate, inter-spike-interval, coefficient-of-variation, and binned population-rate helpers consume returned spikes only and cannot feed data back into the active scheduler.

### 2.4 Interactive compiled-network visualization

`CompiledNetwork.visualize()` opens an interactive browser client over one fresh incremental C-backed run. The server binds only to a loopback interface, assigns an unguessable session path, serves no external resources, and borrows rather than owns the compiled network. The visualizer has no neuron or synapse evaluator of its own. Advancing, encoding, state inspection, decoding, and injection all pass through the existing incremental Python-to-C orchestration path. The browser close action waits for an acknowledged synchronous server shutdown before closing its page; failure remains visible to the user instead of silently orphaning a blocking `viewer.wait()` process.

Initial node positions use a deterministic role-aware layout derived from
explicit nodes, edges, and input/output membership. Input-only nodes begin on
the left, output-only nodes on the right, dual-role nodes in the center, and
other nodes in the central graph. A browser-side force pass refines topology
and applies a distinct reservoir anchor; population metadata is descriptive and
does not affect layout. Input and output neurons carry persistent directional
markers, so the distinction does not depend on position or color alone. The
user can pan, zoom, re-layout, fit the view, and drag individual neurons, with a
dragged neuron pinned until the next requested re-layout. Every explicit
connection is rendered as a directed arrow. Effective signed weight chooses
excitatory versus inhibitory color, while normalized absolute magnitude
affects opacity and width. Mixed neuron bodies use a neutral color and expose
their explicit type when selected.

An emitted spike creates particles on the source's outgoing edges. The payload
uses the recorded emission time plus the actual edge delay, but rendering
enforces a visible minimum travel duration of `max(0.2, display_step/4)`. Thus a
short-delay particle may arrive visually later than the true evaluator
delivery; it is an animation, not a causal event or timing measurement.

Clicking a neuron changes the live inspection selection. The next incremental
advance requests the chunk frontier plus interval-spaced exact-time samples for
that node and every exposed state variable; the sealed end is also sampled.
Because cadence restarts at each evaluator chunk, it is globally uniform only
when the display step is an integer multiple of the sampling interval. Every
state variable is rendered in an aligned small-multiple lane with its own
vertical scale and a shared time axis. A spike-raster lane on that axis uses
exact graph spike times, and a derived inter-spike-rate lane places
`1/(t_n-t_{n-1})` at each retained spike after the first. These are
presentation-only diagnostics and do not feed back into execution.

Browser history is bounded to the most recent 10,000 state points per variable,
20,000 spike times per neuron, and 30,000 active/recent particles. The C results
and optional causal trace remain the durable complete records.
`IncrementalSimulationRun.advance()` and `finish()` accept additional
inspection requests alongside a predeclared `RecordingPlan`, and the evaluator
preserves post-cascade, read-only inspection semantics.

Interactive injection is constrained by the compiled graph's declared input ports. Native SPIKE ports accept timestamped single spikes or finite regular spike trains. Scalar-encoded ports accept normalized presentations and retain their authored regular-rate, Poisson-rate, TTFS, burst, latency-burst, or held-current encoder semantics and live C encoder state. Native DRIVE ports accept direct piecewise-constant drive updates and optional return-to-baseline pulses. Selecting a neuron selects a matching port when one exists; the client never fabricates an undeclared receptor or bypasses port mode and encoder validation.

The client can export a PNG screenshot and records canvas activity through
`MediaRecorder`, requesting WebM/VP9 when the browser supports it and otherwise
using the browser's default recording format. Live playback requests evaluator
chunks ahead of the visible frontier while retaining a bounded lead. During
recording, buffer exhaustion pauses `MediaRecorder`, freezes the visible
frontier, computes a larger lead, and resumes both together, omitting evaluator
wait time from the saved media. Playback reaching completion automatically
stops and saves an active recording; manual completion while paused still
requires the user to stop recording.

Exact state plots, spike flashes, particles, and raster marks are gated by the
display frontier, and user-visible stepping consumes buffered time before
requesting another chunk. The aggregate activity counter is updated when a
prefetched chunk arrives and can therefore include future buffered spikes.
Screenshots and recordings are presentation artifacts; persisted causal traces
remain the exact audit and reconstruction source.

### 2.5 Time representation

Time is continuous and unsnapped in the idealized engine. Spike times are computed by a closed-form evaluation or a root-find and are never intentionally quantized to a grid, since snapping to a grid would reintroduce the fixed step the event-driven design exists to avoid. On the host, time is a double. CLOSED_FORM times are accurate to the implementation's floating-point and transcendental behavior; ROOT_FIND times are additionally limited by solver tolerance.

The core references time through a typedef rather than hardcoding `double` everywhere. This costs nothing now and preserves the option for a device backend to use wide fixed-point deterministic timing without a floating-point unit. Time quantization, if it appears, is a hardware-faithfulness concern that enters only in that backend, not in the idealized core.

The queue has a total storage order on time, phase, limited same-phase kind
precedence, and insertion sequence. At equal time, BOUNDARY precedes DEPOSIT,
which precedes PREDICTION. DRIVE updates precede other boundary kinds; external
input spikes precede internal deliveries; insertion sequence orders the
remaining equal category. A global sequence counter makes this deterministic
without allowing insertion history to move a deposit behind prediction. NaN or
past times are rejected, and sequence/generation exhaustion is a hard error.

## 3. The intermediate representation

### 3.1 Graph container

The top-level object. Synapse types are a shared registry rather than per-edge definitions, which is what makes the folded-shared tier possible. The container is what the canonical JSON format serializes and reloads, per the round-tripping requirement in section 4.

```
Graph:
    time_unit:       str                         # one global time unit, e.g. "ms"
    models:          List<GraphModel>            # embedded intrinsic neuron DSL
    synapses:        List<GraphSynapse>          # embedded intrinsic synapse DSL
    nodes:           List<GraphNode>
    edges:           List<GraphEdge>
    input_ports:     List<InputPort>             # host-visible injection boundaries
    output_ports:    List<OutputPort>            # observation/decoder selections
    modulator_ports: List<ModulatorPort>         # named third-factor edge scopes

InputPort:  { id: InputPortId,
              node_id: NodeId,
              mode: {SPIKE, DRIVE},
              parameter: Optional<str>,           # required only for DRIVE
              encoder: EncoderSpec }              # instance configuration; not part of the neuron model
OutputPort: { id: OutputPortId,
              node_id: NodeId,
              decoder: Optional<DecoderSpec> }    # absent means raw observation only
ModulatorPort: { id: str, edges: List<EdgeId> }
```

The public `Graph` is the persisted, authored form above. Resolution expands it
into canonical node order, numeric binding arenas, merged state layouts,
compiled edge operations, stable indexes, and an equation-derived execution
plan. Those resolved objects are caches and are not serialized as authority.
SPIKE ports deposit into the node's resolved spike-input target; DRIVE ports bind
one named parameter proven safe for event-boundary updates.

Codec configuration belongs to a port instance rather than to a neuron model.
Two nodes sharing intrinsic dynamics may therefore use different codecs without
changing those dynamics. One output port carries at most one decoder, but
several ports may select the same node. Immutable specifications live with the
compiled graph; mutable phase, random position, presentation arming, counts, and
first-spike state live per session. Fresh sessions reproduce their configured
seed. Low-level episode reset clears encoder presentation/phase state while
preserving the next unused random draw; decoder state is reset separately.

### 3.2 Node record

The following is the normative logical content of a resolved node, not a literal
public Python dataclass or serialized record. Current code uses bounded
capability-specific records such as `ResolvedScalarLIF`, `ResolvedAdaptiveLIF`,
`ResolvedAlphaLIF`, `ResolvedPerEdgeLIF`, `ResolvedReactiveIF`, and
`ResolvedSteppedNeuron`, then lowers them into common execution-plan and C
descriptors. It does not expose one `Node` class containing every field below.
The state shown is the post-merge internal layout, not authored neuron syntax.

```
Node:
    id:             NodeId
    polarity:       {EXCITATORY, INHIBITORY, MIXED} # fixed sign or signed edges
    dispatch_form:  {REACTIVE, CLOSED_FORM, ROOT_FIND, STEPPED}  # SET BY RESOLUTION

    # state layout: the augmented vector x, AFTER folded-shared synapses merge in
    state_vars:     List<StateVar>
        StateVar: { name: str,
                    role: {MEMBRANE, RECEPTOR, ADAPTATION, OBSERVER, AUX},  # AUTHORED; capability-validated
                    init: Number }

    # conceptual resolved dynamics analysis
    dynamics:
        A:    Optional<Matrix[n,n]>  # affine analysis form; may be lowered/discarded
        b:    Optional<Vector[n]>
        rhs:  Optional<ExprVec[n]>   # lowered for STEPPED

    parameters:     Map<str, ParameterBinding>    # node-local runtime bindings
        ParameterBinding: {
            value: Number,
            domain: ParameterDomain,              # e.g. finite and tau > 0
            regime: StructuralRegimeId            # selects a valid formula variant
        }
    threshold:      Optional< Expr >              # normalized hard crossing expression
    hazard:         Optional< ResolvedHazard >    # mutually exclusive with threshold
    reset:          Optional< Map<StateVar, Expr> >  # simultaneous own-spike map over the state
    refractory:     Optional< FixedRefractory >   # simple clamped non-excitable period (see 3.11)

    # derived analytical cache: COMPUTED by the resolver, READ by the event core
    analytical:     Optional< AnalyticalBlock >   # present for REACTIVE / CLOSED_FORM / ROOT_FIND

```

Every current `NeuronModel` has exactly one hard threshold or intrinsic hazard.
There is no public `spiking` flag or per-node precision/noise field. Numerical
precision belongs to the whole execution plan and native library. Learned
network metadata and deployment-image headers also retain that profile.

### 3.3 The analytical block

This block is likewise a conceptual normalization of the information distributed
across current resolved records, root hints, expression DAGs, and C descriptors.
There is no literal `AnalyticalBlock` object and no retained general matrix
propagator or eigenstructure table. Capability-specific resolvers emit only the
numeric coefficients, expression roots, and proof hints their accepted family
needs.

```
AnalyticalBlock:
    state_count:       uint32
    readout_index:     uint32
    normal_state_roots:  List<DagRootId>
    clamped_state_roots: List<DagRootId>
    reset_state_roots:   List<DagRootId>
    arithmetic:        {EXPR_DAG, SCALAR_AFFINE}

    crossing:
        mode:  {REACTIVE, CLOSED_FORM, ROOT_FIND, INTEGRATED_HAZARD}
        family: {SCALAR_LOG, TWO_REAL_EXP, MULTI_REAL_EXP,
                 REPEATED_REAL_MODE, ALPHA_REAL, MULTI_EXP_POLY,
                 EXPONENTIAL_VOLTAGE_HAZARD}
                # capability-specific first-crossing proof; only the families
                # listed in sections 7 and 8 are executable

        # REACTIVE: no autonomous crossing is scheduled under the current
        #   binding. Scalar LIF still retains SCALAR_LOG roots so a later drive
        #   update can make it autonomous; event-batched IF has no solution.

        # CLOSED_FORM: next-spike time has an algebraic solution.
        solution:      Optional< Expr >   # t*(x): the next-spike time as a function of state

        # ROOT_FIND: crossing exists but has no symbolic inverse.
        g:             Optional< Expr >   # g(Delta) = threshold.expr evaluated at x(Delta)
        bracket_hint:  Optional< BracketHint >  # solver contract; see section 7

        # INTEGRATED_HAZARD: deterministic state follows the analytical program;
        #   the scheduler integrates lambda(x(Delta)) and inverts cumulative
        #   hazard against one retained exponential variate.
        hazard:        Optional< HazardConfig >

    deposit_roots:     Optional<List<DagRootId>>
```

Reset expressions are evaluated from the complete pre-reset state and committed together. They are not sequential assignments, because a reset such as `{v <- v_reset; w <- w + beta}` must not depend on map iteration order. The current C descriptor bounds analytical state programs to eight roots. That is an explicit embedded-memory bound of the implementation, not a mathematical claim that the IR supports only eight states; a node exceeding it fails lowering with a capacity diagnostic.

The executable C descriptor is capability-driven rather than neuron-model-driven. It contains the common normal, clamped, and reset root lists; a readout-state index; runtime parameter bindings; a certified crossing-family or integrated-hazard descriptor; and, when required, a bounded deposit program. The scheduler does not carry or switch on a biological neuron kind. It evaluates the common state and reset programs for every analytical node and dispatches only to the arithmetic, crossing, hazard, and deposit operations named by the resolved capabilities. This does not imply one heuristic root finder: `SCALAR_LOG`, `ALPHA_REAL`, `TWO_REAL_EXP`, `MULTI_REAL_EXP`, and `MULTI_EXP_POLY` retain separate admissibility proofs and certified first-crossing algorithms, while integrated hazard has its own numerical contract.

A note on REACTIVE. For scalar delta LIF, the resolver records REACTIVE when
the initial binding gives $v_\infty\le v_{th}$. The scalar program nevertheless
retains its logarithmic crossing roots, and the runtime checks the current
asymptote whenever it reschedules. A DRIVE update can therefore turn autonomous
prediction on or off without changing the compiled biological model. Under a
subthreshold-asymptote binding, the node schedules no autonomous event and is
woken only by a deposit or boundary event. This is the cheapest scheduling
regime and the common case for sparse delta-driven networks.

REACTIVE also contains an explicit deposit-triggered integrate-and-fire capability, but it must not be confused with the scalar-LIF scheduling regime above. Its authored dynamics are exactly $dv/dt=0$ and it has no autonomous crossing program by definition. `HOLD` retains subthreshold charge between event timestamps. `RESET_BEFORE_DEPOSIT` applies the simultaneous reset map once before the first deposit batch at a timestamp, then sums every already-scheduled same-time deposit before testing threshold. Both modes reset after a spike in the ordinary firing phase. Threshold is never tested merely because the initial or reset state is at or above the configured level; it is tested only when an event batch affects the node. This permits RISP-equivalent negative thresholds and timestamp leak without inventing a small exponential decay or a timestep loop. Zero-delay causal cascades may create later deposit batches at the same timestamp, but the timestamp reset is applied at most once for that node and timestamp.

### 3.4 Resolved synapse use: the three tiers

An authored `GraphSynapse` contains only an id and intrinsic DSL source. Tier is
a property of its resolved use in one postsynaptic merge context, not a field on
the reusable synapse type. The same source may fold on one node and require
PER_EDGE-derived state on another because bindings, mappings, or the set of
incoming modes differ. Tier determines where kinetic state lives and how memory
scales.

```
ResolvedSynapseUse:
    id:    SynTypeId
    tier:  {DELTA, FOLDED_SHARED, PER_EDGE}
    parameters: Map<str, ParameterBinding>
    parameter_scope: {SHARED_PER_POST_NODE_TYPE, PER_EDGE}
    outputs: Map<SynapseOutputPortId, Expr>       # named values mapped to neuron receptors

    # DELTA: no state. A spike deposits weight w into the edge-mapped neuron receptor.
    #   Memory: none per edge. Preserves the cheapest node dispatch.
    delta_deposit: { kind: POST_RECEPTOR_ADD }

    # FOLDED_SHARED: a shared kernel merged into the post-node's augmented state.
    #   Memory scales with NODES, not edges. Every incoming spike is still a delta
    #   deposit into the shared receptor accumulator.
    kernel: {
        adds_state: List<StateVar>   # extra dims contributed to the post-node
        A_block:    Matrix           # linear dynamics, merged into the post-node A
        deposit_target: StateVarRef  # internal kernel accumulator that receives the deposit
    }

    # PER_EDGE: executable for bounded additive one-state stable exponential and
    #   unit-area two-state alpha current synapses. The resolver augments the
    #   postsynaptic analytical state with distinct incoming kernel blocks so
    #   propagation and crossing see the same continuous contribution.
    #   Structurally identical blocks at exactly equal rates aggregate.
    #   General per-edge state can scale with EDGES (O(N^2) on dense graphs), so
    #   reserve it for genuinely distinct kinetic or dynamical synapse state.
    #   Plasticity traces live outside this state and do not select the tier.
    edge_dynamics: {
        state_vars: List<StateVar>
        A_edge:     Matrix
        b_edge:     Vector
        coupling:   Expr             # how this edge's state drives the post-node
    }
```

The design rule is to fold linear shared-kernel synapses into the node by
default and reserve per-edge dynamical state only for genuinely distinct
per-synapse kinetics. Plasticity uses separate compact trace arenas and does not
select a kinetic tier. If only DELTA and PER_EDGE existed, users would reach for
PER_EDGE for any synaptic filtering at all and blow the memory budget doing what
FOLDED_SHARED handles cheaply. The alpha synapse is the canonical
FOLDED_SHARED case, since its two state variables $s$ and $z$ fold into the
postsynaptic node and every spike is a delta deposit into $z$.

Current PER_EDGE execution uses the augmented-postsynaptic-state option. A filtered edge state continuously drives its postsynaptic membrane, so every distinct active kernel block that can affect a crossing is included in the node's normal, clamped, reset, and crossing programs. Incoming instances with the same resolved rate, kernel structure, direct current output, receptor mapping, and additive deposit semantics are summed into one block and therefore retain FOLDED_SHARED memory and evaluation cost. Distinct supported blocks receive distinct local state components and select PER_EDGE. The analytical descriptor admits at most eight total state components and requires all rates to be stable and real. Equality between a synaptic rate and the membrane rate selects a singular-safe repeated-real formula containing $t e^{at}$ for a scalar exponential or $t^2e^{at}$ for alpha. The accepted multi-state edge capability is currently the two-state unit-area alpha structure; broader multi-state, nonlinear, and conductance-coupled kinetic blocks remain represented future capabilities rather than implicit numerical fallbacks. For a learning rule admitted on a given kinetic configuration, its traces observe arrival and postsynaptic-spike events but never drive the membrane, so adding the rule retains the static configuration's crossing capability. The public rules compose with supported filtered currents.

### 3.5 Edge record

An edge carries a weight and a delay. For excitatory and inhibitory sources the
weight is a nonnegative magnitude, with the sign fixed by the source. For mixed
sources the weight is signed. The effective delivery coefficient is
`pre.polarity.sign * weight`, where `sign` is a stored-weight multiplier of -1
only for inhibitory sources and +1 otherwise. For a mixed source this is an
identity operation, not a declaration that the source is excitatory. Per-edge
state exists only for the PER_EDGE tier. Delays live here and nowhere else.

```
Edge:
    id:        EdgeId
    pre:       NodeId
    post:      NodeId
    syn_type:  SynTypeId
    syn_output: SynapseOutputPortId  # output port selected from the synapse type
    receptor:  ReceptorPortRef       # explicitly mapped post-node receptor
    weight:    Number                # signed only when pre.polarity is MIXED
    delay:     Number                # deliver at t_spike + delay
    bindings:  Map<str, Number>      # edge-local kinetic parameters when required
    state:     Optional<Vector>      # edge-authored initial state for PER_EDGE
    plasticity: Optional<PlasticityRule>  # at most one learning rule per edge
    weight_group: Optional<uint32>   # shared mutable coefficient identity
```

Delays are pure scheduling. They shift delivery time and touch neither dynamics
nor prediction equations. One source spike creates one logical delivery per
outgoing edge. The compiler may coalesce all equal-delay fan-out from that
source into one heap entry, but processing still visits every logical edge in
canonical order, applies its learning event, and counts its delivery. Different
delays remain separate future timestamps. The compiled edge retains authored
weight separately from any deposit scale such as
$1/\tau_s^2$ for a unit-area alpha state. At arrival, the scheduler applies
deposit scale and presynaptic polarity exactly once.

The output-to-receptor mapping is explicit on the edge because it is topology, not intrinsic synapse structure. For FOLDED_SHARED reuse, all compatible edges entering the same postsynaptic node must select the same output/receptor mapping and resolve to the same kinetic mode. When edge-local bindings produce distinct supported decay modes, the resolver preserves them as PER_EDGE state instead of silently combining incompatible kernels. Unsupported mappings or kinetics are rejected before compilation.

Online plasticity mutates the nonnegative edge magnitude and cannot override
Dale polarity. It is rejected on mixed sources in the current implementation.
The runtime stores mutable weight and compact trace state only for
plastic edges; static edges perform no trace decay or learning computation.
Exactly one rule may be attached to each edge, while static and differently
plastic edges may coexist anywhere in one graph. A fresh run begins from
authored weights. Low-level incremental episode reset restores authored neuronal
state, clears queues, encoder presentation state, edge traces, and neuron-local learning observers,
but deliberately preserves weights learned in earlier episodes. A completed
result exposes one final magnitude per edge and can be frozen into a newly
authored network.

An optional `weight_group` makes several explicit edges views of one mutable
coefficient, as used by `Convolution2D`. Group members keep edge-local traces,
and each local update is scaled by the reciprocal of group size so accumulated
learning is a mean over spatial copies. Every member must agree on initial
magnitude, presynaptic polarity, synapse and receptor/deposit behavior,
plasticity program and parameters, bounds, and modulator scope. Current compiled
shared-weight execution requires a supported learning program; static grouped
edges persist and validate but are a known runtime conformance gap recorded in
section 12.

The compiled representation is model-neutral. The five public rule classes lower
into `LearningProgram`: a bounded table of at most six edge-local traces, an
exact lazy-decay parameter per trace, five optional event DAGs--presynaptic
arrival, postsynaptic spike, learning observation, positive modulation, and
negative modulation--and an optional postsynaptic `LearningObserverProgram`.
Each event declares which traces to advance and simultaneous roots for the
normalized weight and updated traces. Numeric parameters and bounds bind per
edge; reusable program identity is structural and excludes rule names and
numeric values. Stable program-homogeneous batches reduce dispatch overhead
without merging independent state. The C executor interprets these descriptors
and does not switch on pair, triplet, or modulated model names.

Learning programs may request an `input_accepted` base variable. On presynaptic
arrival it is zero if a hard refractory clamp discards the delivery's membrane
readout deposit, and one otherwise; on other event types it is zero. Deposits
into non-clamped state remain accepted. This context is derived from the
executed deposit target and clamp policy, not model-name dispatch.

The causal presynaptic learning event is arrival at `t_spike + delay`, not source
emission. The delivery reads its current magnitude, then its presynaptic program
updates traces or weight for future deliveries. Modulation is a boundary event,
so a same-time modulation cannot consume eligibility created by the later
arrival phase. Deposits aggregate before firing resolution. Postsynaptic and
learning-observation programs read the settled pre-reset state, and zero-delay
cascades re-enter the same timestamp deterministically.

All learning equations operate on normalized magnitude $u=(w-w_{min})/(w_{max}-w_{min})$ and clip $u$ to $[0,1]$ after an update. Traces are all-to-all exponential traces advanced exactly and lazily as $x(t)=x(t_0)e^{-(t-t_0)/\tau}$ only when an event needs them.

Pair STDP uses fast pre- and postsynaptic traces $x$ and $y$. On presynaptic arrival it applies $u \leftarrow u-\eta A_-u y$ and then increments $x$. On a postsynaptic spike it applies $u \leftarrow u+\eta A_+(1-u)x$ and then increments $y$. This retains the old implementation's multiplicative soft bounds while removing timestep-dependent Euler decay.

Triplet STDP implements the all-to-all Pfister--Gerstner rule with fast traces $r_1,o_1$ and slow traces $r_2,o_2$. On presynaptic arrival it applies $u \leftarrow u-\eta o_1(A_2^-+A_3^-r_2)$ before incrementing both presynaptic traces. On a postsynaptic spike it applies $u \leftarrow u+\eta r_1(A_2^++A_3^+o_2)$ before incrementing both postsynaptic traces. The standard visual-cortex and hippocampus parameter families are exposed as constructors; all coefficients remain configurable.

Modulated STDP retains separate causal and anti-causal eligibility traces $e_+$ and $e_-$. A postsynaptic spike adds the current pre trace to $e_+$; a presynaptic arrival adds the current post trace to $e_-$. A named third-factor event of value $R$ applies $u \leftarrow u+\eta |R|(c_+(\operatorname{sign}R)e_+ + c_-(\operatorname{sign}R)e_-)$ to every edge assigned to that modulator port. Positive and negative signals have independent causal/anti-causal coefficients. The default is reward-positive causal potentiation and anti-causal depression, with the signs inverted for negative reward. Eligibility may be consumed after modulation, which is the executable default, or retained to accumulate effects across multiple modulator events.

`VoltageModulatedSTDP` and `SoftExcursionModulated` are implemented
voltage-dependent learning rules. The former combines spike eligibility with an
explicit smooth voltage-derived gain; the latter uses an explicit smooth
near-threshold/excursion gate. They are useful experimental controls but must
not be described as learning a firing derivative from local traces.

### 3.6 Observer variables and eligibility traces

Conceptually, an observer is driven by neuronal activity but has no path back
into evolution, reset, threshold, or hazard. Synaptic current is the opposite:
it drives neuronal state and therefore can change intrinsic prediction.

Edge traces live in a compact plastic-edge arena and decay exactly and lazily
on learning events. They are separate from neuronal state, so attaching a
supported learning rule does not change the neuron's intrinsic execution tier.
Connections may bind different programs, parameters, or no learning at all;
exactly one program binds to an edge.

The intrinsic DSL parser also accepts `observer` and `aux` state roles, but the
current resolver does not infer or exploit a general observer partition. A
custom neuron with an extra one-way observer state is not guaranteed to retain
the analytical capability of its membrane subsystem; it may route to STEPPED or
be rejected. Automatic dependency reachability and lazy propagation of general
authored observers remain deferred. Public custom or composed learning-program
authoring is likewise deferred even though the C learning ABI is structurally
program-based.

### 3.7 The lowered expression IR

The resolver produces SymPy expressions and lowers them into this
self-contained expression DAG, which every backend walks directly. Repeated
subexpressions such as the shared $\exp(-\Delta/\tau_s)$ evaluated in several
places are represented once. Common-subexpression elimination maps onto this
DAG directly.

```
ExprDAG:
    nodes:   List<ExprNode>          # topologically ordered; later nodes reference earlier
    roots:   Map<str, ExprNodeId>    # named outputs (e.g. "v_next", "g", "t_star")

ExprNode:
    op:      {CONST, PARAM, VAR, NEG, ADD, SUB, MUL, DIV, POW,
              EXP, LOG, PHI1, PHI1_DERIV, SIN, COS, TANH, MAX}
    lhs:     ExprNodeId              # used by unary/binary operations
    rhs:     ExprNodeId              # used by binary operations
    binding: uint32                  # parameter or runtime-variable slot
    value:   Number                  # CONST payload
```

There is no separate `Shared` opcode. Common subexpressions are shared because
several later nodes or roots reference the same earlier node index.

Parameters referenced by an expression are `Param` nodes evaluated from the
compiled binding arena rather than folded indiscriminately into `Const` nodes.
For Dale-typed sources presynaptic polarity supplies sign outside the authored
magnitude. Mixed-source weights already carry their sign. This supports
approved event-boundary drive updates and mutable learned weights without
rewriting a DAG. It does not expose general arbitrary parameter mutation: other
binding changes currently rebuild and re-resolve the graph.

Because this DAG is the single lowered representation, the current C evaluator
consumes the same operation graph that Python resolved; Python does not
independently walk it. Future backends should preserve operation order where
practical, but the portability contract is numerical equivalence under
recorded tolerances, not unconditional bit identity across different
transcendental libraries or arithmetic formats.

The default general compiler derives a separate dependency closure for each runtime use of an accepted expression program: normal propagation, clamped propagation, own-spike reset, autonomous crossing, and program-defined deposit. This specialization is structural, not algebraic. The strict C implementation preserves the source program's length and node numbering, copies every node in the selected closure byte-for-byte, and replaces only unreachable nodes with inert constants. Consequently every reachable primitive retains its original operands, topological position, evaluation order, profile-selected intermediate storage, and library operation. Crossing hints and root indexes require no remapping. Closure specialization does not change the selected profile's first-crossing routine.

This specialization must not expand, factor, cancel, reassociate, reorder, or approximate an expression. In particular it must not introduce Horner forms, fused multiply-add, fast-math identities, polynomial approximations, epsilon-based term deletion, or special equal-rate formulas that were not already selected by the capability resolver. It may eliminate only a node with no dependency path to any root used by that operation. STEPPED and numerical-crossing programs currently retain the unspecialized evaluator; a structurally incompatible shared program also falls back per operation. "Equation-derived" therefore means that optimization eligibility and dependency closure come from the lowered equations, not that arbitrary equations bypass the analytical capability and first-crossing proofs.

Within one build, the compatibility compile entry point remains an unspecialized differential oracle. The strict specialization target is bit-identical states, spike times, learned weights, trace state, and event order against that oracle on the same platform. The regression matrix enforces float-bit equality across scalar, adaptive, reactive, equal- and distinct-rate exponential, alpha, folded-deposit, mixed-plasticity, and stepped-fallback networks. This stronger same-build check does not change the cross-platform portability contract above, because different math libraries and arithmetic environments may still differ within the declared numerical tolerances.

### 3.8 Numerical precision and deferred hardware noise

Native precision profiles execute through the same model-neutral plans and sparse
scheduler. `Engine()` retains `float64` model arithmetic and time. The optional
`float32-time64` profile uses binary32 model and learning values with a binary64
clock, and `float32` uses binary32 for both. The new `float16` profile uses
binary16 for all runtime floating arithmetic, including clocks and safeguards.
The mixed profile and existing wider float64 safeguards remain compatibility
behavior pending the migration in `docs/uniform_precision_plan.md`.
Precision is selected before
initialization and training. Neither image loading nor engine selection silently
converts a trained network or falls back to a different profile.

`lacuna_numeric.h` defines `lc_real_t` for model and learning values, `lc_time_t`
for timestamps and clock intervals, and `lc_wide_t` for numerical safeguards.
Their types are respectively double/double/long double in the default profile,
float/double/double in the mixed profile, float/float/float in strict float32,
and `_Float16` throughout in strict float16.
Each library is built once for its profile. There is no per-operation precision
dispatch or per-node precision mixture. Matching libm functions and typed
constants prevent accidental widening in strict float32. Reduced builds disable
fast-math and floating-point contraction. Strict float32 disables the native
wall-clock profiler, so `kernel_seconds=0` means timing is unavailable, not a
measured zero duration. Float16 also disables this profiler. Integer execution
counters remain available.

Float16 requires native half arithmetic plus strict compiler evaluation rules.
Elementary functions use integer-indexed, nearest-even binary16 tables rather
than wider libm calculations. These tables evaluate math functions, not
substitute neuronal trajectories. The current shared table set consumes
1,113,472 read-only bytes and is not pruned per graph. General powers are limited
to validated fixed exponents. Unsupported or state-dependent exponents are
rejected. Runtime entry points check nearest-even and gradual-underflow behavior,
and callers must preserve the floating-point environment during execution.
No software arithmetic fallback or MCU speedup is implied.

The compiler prepares detached target-representable authored parameters,
literals, initial state, weights, learning and codec settings before resolution.
It then evaluates derived constant roots in the selected C evaluator and checks
native decay-rate and threshold assumptions. Runtime state expressions and
mutable drive dependencies remain expressions. It does not cast an already
specialized float64 graph. Supported repeated-real formulas can be re-derived
from target bindings. A remaining distinct-rate collision, lost reset/threshold
separation or precision-induced analytical-to-stepped fallback is rejected with
`PrecisionResolutionError`, outside the ordinary capability fallback.

Analytical propagation retains its supported exponential and repeated-real
forms. Float32 evaluation is not exact real arithmetic and does not promise the
same trajectory as float64. Float32-profile adaptive integration uses relative
local-error target `64 * 2**-23`, absolute target `1e-10`, and event-time target
`8 * 2**-23` in model time units. Analytical root hints use relative target
`8 * 2**-23`. These are local targets, not global accuracy guarantees. Crossing
localization may stop at adjacent representable timestamps. Unresolved
near-tangencies report nonconvergence. An overflowing intermediate numerical
trial is retried from the accepted state with a smaller step under the existing
iteration and minimum-step limits. Invalid accepted states still fail.

Float16 uses relative local-error target `8 * 2**-10`, absolute target and minimum
step `2**-24`, event target `2**-10`, and analytical root relative target
`2 * 2**-10`. Rational solver constants are rounded to half without overflowing
their separate numerators. Preferred initial numerical steps are adjusted to
the next representable clock increment only if that respects the configured
maximum. Actual accepted intervals follow representable endpoints. Error
control remains active and unrepresentable physical event intervals fail.
Root and hazard safeguards can reject configurations supported at wider
precision, including cancellations and derivative coefficients that cannot be
represented reliably. The half policy remains explicitly uncertified.

Clock boundaries remain explicit. Positive delays, refractory periods and
generated spike intervals must advance the represented clock. Rounding them to
the current time is an error, not a zero-delay event or an implicit shift to the
next timestamp. In the mixed profile, elapsed intervals entering model
calculations and the equation variable `Time` are checked and converted to
binary32. A binary64 scheduler does not give a binary32 equation binary64 time
sensitivity. Integer timestamps and automatic clock rebasing remain future work.

The integer-only `lc_numeric_property` and `lc_numeric_profile_check` calls
identify profile, scalar/clock/guard representations and arithmetic revision
before Python binds precision-bearing signatures. Evaluator-owned layouts allow
multiple profiles in one process without global ctypes changes. Default ABI 17
and version-1 deployment bytes remain compatible. Float32 profiles use ABI 18
and version-2 images with independent real/time widths and revision metadata.
Float16 uses ABI 19, image version 2, profile identifier 4, and two-byte scalar
and time fields. A separate host-only shim transports half arguments as integer
bits and dispatches to the selected core's exact function addresses. The shim
does not depend on the embedded runtime or introduce wider floating arithmetic.
Wrong-profile plans, opaque decoder handles and images are rejected rather than
reinterpreted. Original ABI-17 libraries without metadata remain accepted under
their known fixed64 contract, explicitly labeled `legacy-abi17`.

Results expose their execution profile. Learned JSON snapshots retain it under
reserved `lacuna_precision` network metadata, which is checked on recompilation.
The graph schema and hexadecimal value transport remain unchanged. A snapshot
contains learned weights and authored initialization, not a live event queue or
eligibility checkpoint. Exporting a learned deployment image requires compiling
that snapshot because an immutable compiled graph retains its initial weights.
Host trace artifacts store native values losslessly in binary64 wire fields and
retain profile metadata. State reconstruction requires a matching evaluator and
target-resolved graph. The MCU runtime does not depend on that host file format.

`analyze_precision(network_or_graph, precision, time_horizon=None)` remains an
optional conservative preflight. Its report is non-executable and includes
source/target values, candidate dispatch, identity and outstanding checks. It
cannot validate every dynamic singularity or certify a target toolchain.
`numerical_policy(profile)` exposes implementation status and precision-dependent
defaults, not independent authority to run a graph. Profile, arithmetic and
policy revisions participate in target plan identities.

`TargetBindingEvaluator` is a separate optional host companion for inspecting
bounded constant-only primitive fragments. Its binary64 transport is not part
of strict-float32 execution. The runtime compiler uses the selected core's
expression evaluator for supported derived roots, including transcendental
operations. The companion is not linked into `lacuna_core` and is not required
for deployment-image loading. `LACUNA_BUILD_TARGET_BINDER=OFF` excludes it.

Desktop validation covers analytical, stepped, stochastic and mixed graphs,
learning, codecs, recording, native image round trips and default compatibility.
The strict float32 Clang audit checks all five runtime translation units at `-O0` and
`-O3` for wider floating operations. This is not a certificate for untested MCU
compilers, device math libraries or long-run learning quality. Target-toolchain,
clock-horizon, memory and hardware performance measurements remain required.
Usage is documented in `docs/numerical_precision.md`, with implementation history
and evidence in `docs/numerical_precision_plan.md`. The float16 audit additionally
checks its sixth math translation unit, final machine code and linked helpers.
Half table generation uses independently checked high-precision host intervals,
not high-precision runtime execution. Its tested scope and remaining deployment
constraints are documented in `docs/float16.md`.

Hardware-noise configuration is also not implemented and is absent from the
graph schema.

The target-independence invariant reserves a place for a future external hardware profile;
the following is a design sketch, not a current IR record:

```
HardwareSpec:
    number_format:   { kind: {FIXED_POINT, ...}, total_bits: int, frac_bits: int }
    rounding_mode:   {TRUNCATE, NEAREST_EVEN, ...}
    saturation:      {WRAP, CLAMP}
    noise_model:     Optional< { kind: ..., params: Map<str, Number> } >
    update_boundary: {PER_EVENT, PER_STEP}   # where quantize/noise applies
```

When co-design is built, such a profile may drive a differentiated forward path
or a hardware backend. Its `update_boundary` would determine whether event-only
quantization is faithful or a target clock requires per-step effects. Adding it
requires an explicit schema and ABI change, target-specific numerical
equivalence tests, and a concrete hardware semantics; current idealized graph
values must remain unchanged.

### 3.9 Runtime structures

These wrap the static IR at simulation time without modifying it. The static IR supplies symbolic expressions. The runtime supplies state and bookkeeping.

```
NodeRuntime:
    x:                Vector          # current augmented state
    t_last:           Time            # time x was last advanced to (typedef, see 2.5)
    generation:       uint64          # bumped on reschedule; stamps the scheduled spike
    refractory_generation: uint64     # stamps the current fixed clamp release
    clamped:          bool             # release time lives in the queued boundary event
    affine_disabled:   bool            # runtime rebind invalidated scalar arithmetic cache
    hazard_remaining: Number           # residual unit-exponential target
    hazard_draw_index: uint64          # per-neuron counter-based stream position
    hazard_initialized: bool

RunConfig:
    stochastic_seed: uint64           # intrinsic-hazard seed; codec seed is separate

GlobalRuntime:
    seq_counter:      uint64           # monotonic; exhaustion is a hard error (see 2.3)
```

The C implementation stores state and `t_last` in packed arenas beside the
compact per-node bookkeeping shown above; the combined record is conceptual.
Refractory-until time is represented by a queued, generation-stamped release,
and the reset readout already stored in `x` is the held value. At a timestamp
boundary, the loop advances each affected state through its selected analytical,
reactive, or stepped evolution operation, then applies same-time phases,
aggregates deposits, invalidates predictions, and schedules at most one current
autonomous prediction per affected node.

The compiled scheduler maintains bounded sparse worklists for affected and firing nodes. A timestamp with one arrival therefore clears and visits only that arrival's target rather than scanning state-sized buffers for the whole network. Simultaneous targets are sorted into canonical node order before state mutation and firing, so this sparse discovery changes cost but not deterministic same-time semantics. Deposit accumulators and membership flags are cleared only for worklist members after each cascade; timestamp-reset flags are likewise cleared only for nodes that used the reset-before-deposit capability. The worklists are preallocated to the compiled node count and cannot grow during a run. Chronologically ordered external spike streams are merged with the next internal event by a run-local cursor rather than copied into the event heap; unsorted low-level input retains a heap fallback. This keeps heap occupancy and insertion cost proportional to internally scheduled work without changing input-before-delivery ordering at equal timestamps.

The scalar-LIF reactive-versus-driven guard lives here. At each reschedule it
evaluates the sign of $v_\infty-v_{th}$ under the current drive. If it is
nonpositive, no autonomous spike is scheduled and threshold is checked at the
next deposit; otherwise the scalar-log prediction is scheduled. The initial
REACTIVE/CLOSED_FORM dispatch label therefore does not freeze later DRIVE
behavior.

An intrinsic-hazard node instead retains one positive unit-exponential target.
Advancing to an earlier delivery or drive boundary subtracts cumulative hazard
along the exact deterministic state trajectory, then reschedules against the
same residual target. Only an actual spike consumes a new counter-based draw.
The draw stream is keyed by run seed, canonical compiled node index, and draw
index, so stale queue predictions do not perturb randomness.

### 3.10 Events and the queue

#### Facts, predictions, and boundary events

Every queued event is at or after the current timestamp. The most important distinction is whether an event is a fact, a prediction, or a state-boundary marker.

A delivery is a fact about the future. The presynaptic node has already fired, the spike is in flight, and the edge delay fixes its arrival exactly. Nothing downstream can un-fire the source, so a delivery is always valid when popped and can never go stale.

An input spike is also a fact. It names an input port whose graph record
resolves the target and carries a finite deposit amplitude, whether supplied
natively or emitted by a compiled encoder. A drive update is a boundary event:
the old drive governs propagation up to its timestamp and the new
piecewise-constant value governs the following open interval. A refractory
release is a boundary event scheduled when a fixed clamp begins; it wakes the
node even if no delivery arrives.

An autonomous spike is a prediction about the future. For a deterministic node
it asserts the first hard-threshold crossing. For an intrinsic-hazard node it
asserts the time at which cumulative hazard reaches the retained random target.
A state or drive event landing first invalidates the queued time; deterministic
crossing is recomputed, while stochastic hazard already consumed through the
event is subtracted and the same residual target is inverted from the new state.

Autonomous spikes exist because in a pure event-driven simulator time advances only by popping events. A tonically driven node with no input arriving would otherwise never be woken at its crossing time and would never fire, and the only alternative to a self-scheduled wakeup is periodically polling every node, which is a clock and is what this design exists to avoid. This is also exactly why REACTIVE nodes schedule no autonomous spikes: their asymptote lies at or below threshold, so they cannot fire without input, the delivery is always the wakeup, and threshold is checked there. That is what makes REACTIVE the cheapest tier.

When an autonomous prediction reaches the firing phase and its generation still matches, no separate numerical threshold re-check is needed. A matching generation at that phase means no delivery or boundary update at or before that timestamp invalidated the prediction, so the crossing is certain within the tier's numerical contract. A stale prediction is discarded, with its replacement already queued or computed during the current batch.

#### Queue, input, and output records

```
QueueEvent:
    t:           Time      # scheduled time (typedef, see 2.5)
    phase:       EventPhase
    seq:         uint64    # insertion order inside one same-time phase
    payload: one of
        AutonomousSpike { node_id: NodeId, generation: uint64 }
        Delivery        { edge_id: EdgeId }
        InputSpike      { input_port: InputPortId, value: Number }
        DriveUpdate     { input_port: InputPortId, value: Number }
        Modulation      { modulator_port: ModulatorPortId, value: Number }
        RefractoryRelease { node_id: NodeId, refractory_generation: uint64 }

EventPhase: one of
    BOUNDARY    # RefractoryRelease, DriveUpdate, Modulation, decoder query/close
    DEPOSIT     # Delivery, InputSpike
    PREDICTION  # AutonomousSpike

OutputEvent:
    t:       Time
    payload: one of
        Spike { node_id: NodeId }

DecodedEvent:
    decoder_id:       OutputPortId
    window_id:        uint32
    kind:             UPDATE | FINAL | NO_SPIKE | QUERY
    emitted_at:       Time
    source_spike_time: Optional<Time>
    window_start:     Time
    window_end:       Time
    observed_through: Time
    value:            Number
    count:            uint64
    first_spike:      Optional<Time>
    valid:            bool
```

`QueueEvent` is a discriminated union because inputs and boundary events have genuinely different payloads and invariants. Ordering uses `(t, phase, limited kind precedence, seq)`: drive updates lead other boundary events, and external input deposits lead internal deliveries. `OutputEvent` is not fed back into the simulation queue. It is emitted through a C output sink that fans each spike to the bounded raw-output buffer when retention is enabled and to decoder schedules assigned through output ports. The low-level raw buffer sees every emitted node spike; high-level `spike_targets` may select the returned subset. Decoding therefore cannot alter simulation causality. Backpressure and raw-output overflow are explicit run errors, but a run with at least one streaming consumer may disable raw retention entirely.

The current decoder set consumes spikes only. `DecodedEvent` is a separate C
output stream derived from `OutputEvent`; it is not fed back into the simulation
queue. `emitted_at` says when the decoder made the result available, while
`source_spike_time` identifies the exact causal spike when one exists.
`observed_through` makes the evidence horizon explicit, so an early TTFS answer
cannot be mistaken for a window-close answer. Optional times use presence flags
in the ABI rather than NaN sentinels. A later one-way output extension may add
state/readout samples, but no such output kind is implemented now.

A delivery carries the edge id rather than resolved deposit data. The handler dereferences the compiled edge's weight, the presynaptic node's polarity, and the synapse records for output-to-receptor mapping, tier, and either the post receptor or folded-kernel deposit target. It derives the signed effective weight at that point, negating inhibitory magnitudes and retaining signed mixed-source weights. This keeps the event narrow, which matters because the struct sits in device-eligible code and its width multiplies across the whole queue capacity. The cost is an indirection into resident, stably indexed node, edge, and synapse arrays; topology and neuron polarity are fixed for the life of a run and mutate only between evolutionary candidates.

Carrying the edge id also supplies identity for spike-driven observers. A delivery's presynaptic node is recoverable by dereferencing its edge, an input spike names its input port, and an emitted output spike names its node. No generic untyped identity field is needed.

#### Same-time phases

Insertion sequence is not allowed to determine physical causality at an equal timestamp. The event loop processes a timestamp as a batch with these phases.

1. Advance every affected node to time $t$ under the dynamics and drive valid on the open interval before $t$. A clamped membrane remains at reset during this advance, while its non-membrane state advances normally.
2. Emit due decoder queries and then close due decoder windows before observing
   same-time spikes. Apply drive updates ahead of other queued boundary events;
   refractory releases and third-factor modulation then follow deterministic
   insertion order. The
   new drive applies after the boundary. A release makes the node excitable
   before same-time deposits. Modulation updates weights before those deposits,
   while an already scheduled delivery still uses its defined read-before-update
   arrival semantics.
3. Apply all deliveries and input-spike deposits at $t$, aggregating additive deposits per target before testing threshold. Delta deposits aimed directly at a membrane that remains clamped are ignored; receptor-state deposits are retained and evolve normally.
4. Invalidate affected autonomous predictions. Confirm still-current threshold
   or integrated-hazard predictions and evaluate deposit-time hard thresholds
   once per affected excitable node against the post-deposit state. When a
   trace-derived learning observer is present, evaluate any qualifying
   post-deposit subthreshold excursion using the resolved analytical trajectory.
5. For all nodes that fire, update pre-reset postsynaptic learning observations
   and ordinary post-spike programs, emit one `OutputEvent::Spike`, apply the
   simultaneous neuronal reset, draw the next intrinsic-hazard target when
   applicable, enter refractory state when configured, and schedule outgoing
   deliveries.

New zero-delay deliveries at the same timestamp re-enter phases 3 through 5. The loop continues until the timestamp is quiescent. A configurable same-time cascade limit turns a zero-delay algebraic loop or runaway cascade into a diagnostic error rather than an infinite loop. Nodes firing in the same phase are treated simultaneously: their outgoing events cannot suppress another firing already established in that phase.

#### Lazy invalidation

Stale predictions are handled by lazy invalidation rather than decrease-key on the queue. Each scheduled autonomous spike is stamped with the node's generation at scheduling time. When one is popped, its stamp is compared against the node's current generation, and if they disagree the node has since rescheduled and the entry is discarded. The queue stays push and pop only.

#### Queue capacity and overflow

The queue must hold live entries plus stale ones not yet evicted, so its occupancy depends on runtime activity rather than on topology. Live internal entries include at most one current autonomous prediction and one current refractory release per node, deliveries in flight, and configured future input and drive events. Stale predictions depend on how often nodes reschedule between pops, which is driven by fan-in and input rate and does not admit a tight a priori bound.

Capacity is therefore a user-specified parameter rather than something the
engine derives. This matches how the constraint actually arises on a device,
where a fixed RAM budget dictates the allocation and the question is whether a
network runs inside it. The queue is the only internal scheduler structure
whose required capacity depends on activity rather than topology. Output,
encoder-spike, encoder-drive, decoded-event, causal-trace, and same-time-cascade
resources also have explicit run budgets; node and edge arrays follow from the
compiled graph.

Overflow is a hard error and halts the run. Dropping events was considered and rejected. Dropping the incoming push discards an event that may be earlier than entries already queued; dropping the farthest-future entry requires finding a maximum in a min-heap; and dropping a delivery silently breaks the invariant that a presynaptic spike reaches all its targets, making a fan-out partial in a way nothing downstream can detect. Since fidelity under overflow is not recoverable, halting is the honest behavior and the condition means the buffer is undersized.

Overflow was also considered as a noise source and rejected. Its rate depends on buffer capacity and instantaneous activity rather than on a specified distribution, so two runs of the same model with different capacities would see different noise, which is a confound rather than a research parameter. It is additionally biased, since overflow occurs during bursts and falls hardest on high-fan-out nodes, making it structured interference correlated with the dynamics under study. Treating it as noise would also destroy its diagnostic value, since a nonzero count would no longer distinguish a correctly sized run from a broken one.

The error message must be actionable, reporting the configured capacity, the peak occupancy, and the event that overflowed, because "queue full" alone does not tell a user whether to enlarge the buffer or fix a runaway network. A high-water mark tracks peak occupancy throughout a run, costing one comparison per push, and reports headroom against the configured capacity so a run that peaked near its limit is visible before a different input exceeds it.

### 3.11 Refractory periods

Refractoriness is two distinct model families that happen to share a name, and they land in different places.

The current scheduler-level refractory case is a fixed clamp:

```
FixedRefractory:
    duration: Number
    mode:     {CLAMP_RESET}   # the only current scheduler mode
```

After a spike, the simultaneous reset expression is committed to live state.
The designated membrane/readout is held at that reset value and is
non-excitable until a generation-stamped `REFRACTORY_RELEASE` event. The runtime
does not carry a separate clamp value or refractory-until field. Non-membrane
state continues to evolve under the resolved clamped state program, and
filtered receptor deposits continue to accumulate. Intrinsic hazard is paused
rather than integrated during the clamp; the freshly drawn target remains
pending. Delta deposits aimed directly at the clamped membrane are ignored. At
release, prediction resumes from the complete analytically or numerically
advanced state.

The model-level case is recovery behavior written in the equations, such as a recovery variable, undershoot, or dynamic threshold. The membrane can be below reset yet remain excitable because the behavior is part of the state trajectory rather than a scheduler lockout. This is a dynamics change and routes through the normal resolver machinery. A model that remains inside a supported affine family uses an analytical tier; nonlinear recovery routes to STEPPED; a moving threshold remains outside the first stepped capability. Scheduler-level clamp and model-level recovery are separate semantics. They may coexist only when both are explicitly authored: during the clamp the designated readout is held at reset and is non-excitable, while every other model state continues under its own equation.

## 4. The model DSL

The intrinsic DSL is the human-readable text embedded in model and synapse
records. A whole graph is serialized as canonical JSON containing that source,
bindings, topology, ports, and metadata; topology is never DSL text. The user
writes mathematics and declares only the semantic roles the resolver cannot
recover, while the resolver classifies every supported structural capability it
can prove.

### 4.1 Statement forms

The grammar keeps statement kinds syntactically unmistakable. Parameter
assignment uses `=`. A differential equation uses `dx/dt =`. A reset uses the
arrow `<-`. A hard threshold uses a comparison. An intrinsic hazard contains
the distinguished equation `rate = <expression>`. These forms let the parser
tag intent without guessing and reject undeclared symbols. Every currently
executable neuron contains exactly one `threshold` or `hazard` block, never
both.

### 4.2 Neuron models are intrinsic-only

A neuron model should declare its intrinsic dynamics plus receptor targets that
synapses may deposit into. Reusable synapse kinetics belong in separate synapse
definitions and are merged into a node at resolution; this enables sharing and
the folded-state memory model. The augmented neuron shown in the IR and worked
examples is the post-merge internal view. This separation is an authoring and
composition contract, not a provenance check: the parser cannot infer that an
otherwise valid authored state was intended to be synaptic, and an unmatched
ODE may route to STEPPED or fail another capability guard instead.

A neuron model has blocks for `params`, `state`, `dynamics`, exactly one of
`threshold` or `hazard`, `reset`, and optionally `refractory` or the specialized
`reactive` declaration. In `state`, the user declares the membrane readout and
named receptor ports. The parser also accepts authored `observer` and `aux`
roles, although current exact resolvers do not infer or exploit a general
observer partition. A receptor belongs to the neuron namespace and a synapse
state to the synapse namespace. Explicit output-to-receptor mapping connects
them during merge; name collision is never wiring. A representative threshold
neuron whose receptor `i_exc` is supplied by a separate alpha synapse follows:

```
neuron LIF {
    params {
        tau_m   : positive = 10.0
        v_rest  = -65.0
        v_th    = -50.0
        v_reset = -65.0
    }
    state {
        v : membrane
        i_exc : receptor      # a synapse output is explicitly mapped here
    }
    dynamics {
        dv/dt = -(v - v_rest)/tau_m + i_exc
    }
    threshold  { v > v_th }
    reset      { v <- v_reset }
    refractory { 2.0 }
}
```

### 4.3 Threshold normalization

A threshold is written as a friendly comparison such as `v > v_th`. The parser records the designated membrane readout `v`, the level expression `v_th`, and a rising-edge direction, and also derives the normalized crossing expression `v - v_th`. A fixed threshold is one whose level expression is free of all state variables. The membrane symbol in the normalized crossing expression does not make the threshold moving. A moving threshold is one whose level itself depends on dynamic state. Moving thresholds are representable in the IR but outside the current analytical and stepped capabilities. Discontinuous delta deposits are handled explicitly in the delivery phase: a post-deposit value above a fixed threshold fires even though no continuous zero of the crossing expression occurred.

### 4.3.1 Intrinsic hazard normalization

An intrinsic stochastic neuron writes a conditional intensity, for example:

```
hazard { rate = escape_rate*exp((v-v_escape)/delta_v) }
```

The first hazard resolver accepts algebraically equivalent forms only when it
can prove that the log rate is affine in the single membrane readout, the
prefactor is strictly positive, and voltage gain is strictly positive. It
normalizes the expression to

$$
\lambda(v)=\exp(\ell_0+\gamma v),\qquad \gamma>0,
$$

and records numerical quadrature, inversion, and time tolerances. A hazard is
not a threshold with random jitter and is not a Bernoulli test performed at a
hidden timestep. Current hazard neurons require a supported exact analytical
LIF or adaptive-LIF trajectory and delta synapses.

### 4.3.2 Parameter domains and structural regimes

Parameters that affect analytical structure declare a domain. Current
analytical capabilities require finite values, positive time constants, and
stable real decay rates. The resolver selects formula regimes, most notably
equal versus distinct time constants. Equality uses a separately valid
repeated-rate expression; it is never obtained by substituting into a singular
distinct-rate formula. The alpha kernel's internal repeated rate and equality
between membrane and synapse decay are executable and produce $t e^{at}$ or
$t^2e^{at}$ terms. Validation occurs during construction, loading, resolution,
and approved drive updates.

### 4.4 Synapse types

A synapse type is registered once and shared. It declares its own kinetics as equations, a deposit rule for how an arriving spike injects its effective signed amplitude `w`, and how it couples into the postsynaptic node. Canonical and compiled edges store nonnegative magnitudes for Dale-typed sources and signed weights for mixed sources. The scheduler applies the source's stored-weight multiplier before `w` reaches the deposit program. As with neurons, the user writes the kinetics and parameter scope, and the resolver decides the executable tier. A strawman alpha synapse supplying a current receptor:

```
synapse alpha_exc {
    params { tau_s : positive = 5.0 }
    state  { s ; z }
    dynamics {
        ds/dt = -s/tau_s + z
        dz/dt = -z/tau_s
    }
    on_spike { z <- z + w/tau_s^2 }  # unit-area alpha kernel; integral of s is w
    output   { current = s }     # named synapse output port
}
```

For example, connecting `alpha_exc.output.current` to `LIF.receptor.i_exc`
causes the resolver to substitute the folded synaptic state `s` for that
receptor port in the augmented node system. The current exact edge-scoped
capability accepts one-state synapses of the form $\dot s=q s$, $q<0$, with
direct current output and an additive $s\leftarrow s+w$ deposit, plus the
two-state unit-area alpha structure shown above. An alpha delivery resolves
once to the numeric update $z\mathrel{+}=q^2w$; the runtime delivery remains one
state addition. Structurally identical equal-rate instances share one
postsynaptic block, while distinct supported blocks select PER_EDGE and enter
the complete crossing expression. Broader kernels fail capability resolution
rather than silently falling back to sampled integration.

### 4.5 Graph construction and round-tripping

Graph topology is built in the Python API, not written by hand in the DSL, because the evolutionary loop generates and mutates topology programmatically and the resolution key is keyed to model structure rather than topology. Canonical JSON is the on-disk representation. A graph built in Python is saved to one self-contained JSON object and loaded back to reconstruct the same graph. JSON is primarily a generated interchange and persistence format rather than a second hand-maintained authoring surface.

Round-tripping means the schema represents everything the Python API
constructs: embedded `neuron` and `synapse` DSL sources, node instances with
explicit polarity, edges with source-constrained weights and delays,
synapse mappings and initial state, optional per-edge plasticity and shared
weight-group identity, input/output codec bindings, and edge-scoped modulator
ports. Model sources remain readable strings, while topology arrays are
machine-written and machine-read. Canonical key and record ordering,
indentation, and one trailing newline make unchanged saves byte-identical.

Structure and values need not be split into parallel serialized sections. A
node's structural key excludes bindings that do not alter equations or formula
regimes, while merge structure includes incoming synapse kinetics and receptor
mappings. The current `resolve_graph()` nevertheless parses and resolves the
complete graph on each call; it does not perform an incremental topology patch.
Equivalent already-resolved equation and learning programs are deduplicated
within one execution plan. Weight, delay, polarity, topology, or model edits are
made by constructing a new immutable graph and recompiling it.

### 4.6 Serialized graph format

A saved graph is a single self-contained JSON artifact. One file fully determines one graph, with no dependence on an external model library that could drift. The object carries the graph schema version, model and synapse definitions with content hashes, explicit topology, ports, codec specifications, and codec hashes.

The high-level `Network` artifact wraps that canonical graph object without changing it. It adds a separate network-schema version, the canonical graph SHA-256, and an independently hashed `authoring` object containing the network name, population membership, optional reservoir membership and recurrent-edge identities, standalone neuron labels, and finite JSON metadata. Population and reservoir labels do not affect graph semantics. `Network.load` also accepts a bare graph-schema JSON document for compatibility with the lower-level API.

A representative abbreviated document is:

```
{
  "network_schema": 1,
  "graph_sha256": "...",
  "authoring_sha256": "...",
  "graph": {
    "schema": 10,
    "time_unit": "ms",
    "models": [{"id": "LIF", "model_hash": "...", "source": "neuron ..."}],
    "nodes": [{
      "id": 0,
      "model": "LIF",
      "polarity": "EXCITATORY",
      "bindings": {"tau_m": "0x1.4p+4", "v_rest": "-0x1.04p+6"},
      "initial": "-0x1.04p+6"
    }],
    "edges": [{
      "id": 0,
      "pre": 0,
      "post": 1,
      "weight": "0x1.0p-1",
      "delay": "0x1.0p+0",
      "synapse": "alpha_exc",
      "output": "current",
      "receptor": "i_syn",
      "weight_group": 0,
      "plasticity": {
        "kind": "MODULATED_STDP",
        "parameters": {"learning_rate": "0x1.999999999999ap-5", "...": "..."}
      }
    }],
    "input_ports": [],
    "output_ports": [],
    "modulator_ports": [{"id": "reward", "edges": [0]}]
  },
  "authoring": {
    "name": "example",
    "populations": [{"name": "hidden", "nodes": [0, 1]}],
    "reservoirs": [{"name": "hidden", "nodes": [0, 1], "recurrent_edges": [0]}],
    "neuron_labels": {},
    "metadata": {}
  }
}
```

The rules the format follows.

Floating-point numbers are hex float. Every floating-point value is written in
C99 `%a` form, which round-trips the stored double bit pattern without
ambiguity. Schema versions, identifiers, indexes, and counts remain JSON
integers. This is required so save and reload do not introduce an additional
numeric difference; it does not imply that different runtime arithmetic
backends produce bit-identical trajectories. The readability cost falls on the
machine-written topology section, where readability matters least.

Authored values use binary64 host transport, while learned snapshots may contain
native binary32 values represented losslessly in that transport. Their reserved
network metadata records the training profile, and a conflicting engine profile
is rejected. The edge object carries its magnitude, or a signed weight for an
explicit mixed-sign source. Polarity is stored on the presynaptic node. Precision
is not a per-edge graph-schema field, and hardware noise remains separate future
work. A binary sidecar for very large value payloads is also deferred.

Every effective instance binding is stated explicitly. Embedded DSL source may
contain declaration defaults, and the construction API also supplies defaults,
but loading never depends on whatever those defaults become in a later API
release. A saved graph therefore reproduces independently of future
construction-default changes. A node's model reference supplies its structure,
equations, threshold, reset, declarations, and source hash; its complete
effective numeric bindings are serialized on the instance.

Node ids are plain integers, because the common producer is the evolutionary loop rather than a human. Optional labels live in independently hashed authoring metadata and change nothing structural. JSON comments are not supported; generated artifacts carry descriptive metadata as ordinary fields instead.

Object keys are emitted in canonical sorted order. Models, synapses, nodes, edges, and ports are sorted by their identifiers before serialization. Re-saving an unchanged network therefore produces a byte-identical file. The writer rejects NaN, infinity, and non-JSON metadata rather than relying on implementation-specific JSON extensions.

Connectivity is always explicitly enumerated. Generative shorthand, meaning population-to-population all-to-all, local-neighborhood, exact-degree, random-with-seed, or reservoir forms, is authoring sugar in the Python API that expands to explicit edges before saving, and the graph serializer never emits it. Optional reservoir membership in the independently hashed authoring object merely names a selection of those already explicit nodes and edges. Explicit enumeration is the one canonical graph format because it can represent any graph including an arbitrary evolved one, whereas a generated form cannot and a mutated generated graph has to flatten anyway.

### 4.7 Versioning, hashes, and load-time validation

Schema versions and content hashes answer different questions. The graph writer
emits schema 10 for Dale-typed graphs and schema 11 for graphs with mixed
neurons. The current loader accepts schemas 9, 10 and 11, but rejects mixed
polarity in a document labeled schema 9 or 10. The
high-level network wrapper emits and accepts network schema 1. Every other
schema value is a hard refusal with a message naming the unsupported boundary;
there is no automatic migration.

Persisted `model_hash` and `synapse_hash` values are SHA-256 over the normalized
embedded source text, defined as `source.strip() + "\n"` encoded as UTF-8. They
ignore leading and trailing whitespace but detect changes to interior source
content, including interior whitespace, comments, and defaults. On load the
hash is recomputed before parsing. These persisted source hashes must not be
confused with resolver keys.

The resolver separately computes canonical structural hashes from parsed
symbols and expressions. A node resolution key additionally covers merge
context such as incoming synapse structure, receptor mappings, state ordering,
formula regime, threshold or hazard capability, and reset/refractory behavior.
Learning and learning-observer identities are separate plan-level structures.
Numeric bindings that stay inside one structural regime may be excluded so
equivalent programs can be deduplicated within a compilation; this does not
create a public general rebinding path. State, receptor, and parameter names
remain significant because bindings are named and lowered evaluator inputs are
positional.

Encoder and decoder specifications have separate content hashes stored on their
port records. Their hashes cover codec behavioral configuration: kind,
numerical mapping, window mode, spike-selection policy, and emission policy.
Port target, SPIKE-versus-DRIVE ownership, and a DRIVE parameter binding are
graph fields covered by graph identity rather than `_codec_hash`. Codec hashes
are intentionally independent of neuron and synapse hashes: changing a decoder
window or when decoded values are emitted must invalidate a prepared IO artifact
but must not pretend that the simulated neuron dynamics changed. Codec numbers
are serialized as hexadecimal floats under the same rule as other graph values.

Internal resolver keys use deterministic structural encodings, including SymPy
representations where appropriate. They are audit and within-plan
deduplication identities, not persisted compatibility promises or a persistent
cache. SymPy is pinned to 1.14.0 so a library upgrade is an explicit resolver
change with regenerated tests and, if it changes persisted behavior, a
deliberate schema decision.

On load, the loader recomputes each referenced model's hash from the embedded definition and compares against the recorded value. A mismatch means the dynamics changed under a graph whose parameters were saved against the old dynamics, so the load is refused rather than silently binding the old parameter set to new equations.

The schema value remains separate from every content hash. Supporting graph
schema 9 is an explicit compatibility path in the loader, not a general
migration framework; old versions remain identifiable for a future deliberate
migrator.

Embedded model definitions are DSL source, not resolved form, and every JSON
load validates them through the current parser/resolver. Exact successful
results from the same process may be reused under section 6.7; cache contents
are not serialized. Resolver fixes therefore apply in the new process, and
derived artifacts never become authority. A future persistent cache would need
validation against source hashes, structural keys, resolver version, and
capability version. Binary deployment images retain their separate compiler-free
loader and do not consult this cache.

Evolutionary candidate-population bundle serialization is deferred. A saved
collection of many candidate networks sharing a handful of models would
duplicate the embedded definitions once per candidate, so a bundle save is a
distinct future operation that may factor shared models out internally. This
does not conflict with the implemented authoring metadata for named neuron
populations and reservoirs inside one saved network.

### 4.8 Subnetworks

Subnetworks are not implemented. The reserved design contract is that they will
be Python authoring and optional metadata only: instantiation must flatten to
fresh node ids and explicit edges before resolution, serialization, or runtime.
The scheduler will remain one flat graph and one global event queue. If editable
module identity is later persisted, it belongs in independently hashed authoring
metadata over the canonical flat graph, never in evaluator semantics.

## 5. Input, output, and recording

### 5.1 Codec framework and ownership

Input and output go through modular encoding and decoding state machines in the
C evaluator. Python constructs and serializes codec specifications, submits
normalized scalar presentations or finite native events, and reads results; it
does not implement a second simulator. Encoders produce typed
`InputSpike` or `DriveUpdate` records. The core emits `OutputEvent::Spike` to
decoders and recorders. Decoders are output-only and cannot enqueue simulation
events. A host may implement explicit feedback by observing an incremental
result and submitting a later input, but there is no automatic closed-loop
adapter in the current API.

An encoder is configured per input port, so every input neuron may have its own
code and parameters. A decoder is configured per output port, so every output
neuron may have its own readout; multiple ports may select the same node. Codec
state is per session rather than global. Creating a fresh seeded session
reproduces regular-rate phase, Poisson state, TTFS/burst arming, rate-window
accumulators, and first-spike state. Low-level episode reset instead clears
active encoder presentations while preserving random-stream progress; attached
decoder state must be reset or recreated independently.

Seed ownership is explicit below the high-level convenience API.
`ResolvedGraph.run()` and compiled graph runs accept `encoder_seed` and
`stochastic_seed` separately; C `RunConfig.stochastic_seed` keys only intrinsic
hazard. `CompiledNetwork.run(seed=...)` and `start_run(seed=...)` intentionally
pass one user seed to both subsystems without merging their counter streams.

Scalar encoder input is a normalized value $x \in [0,1]$ presented over a half-open interval $[t_0,t_1)$. Presentations for one port are ordered by start time. When a new presentation begins before the prior one ends, it replaces the prior value and cancels the unelapsed part of its spike train or held drive. Two presentations for one port may not have the same start time. Encoder-generated spikes and decoder windows use half-open boundaries throughout: a spike exactly at an end boundary belongs to the following window. Rate and Poisson phase is carried through value changes rather than restarted.

The first input primitive is a spike train, which enters as scheduled
`InputSpike` events. It names an input port and carries a finite deposit
amplitude; unlike scalar presentations, native amplitudes are not restricted to
$[0,1]$. The graph's port registry resolves the target. It does not pretend to
have a presynaptic graph edge.

The second primitive is a piecewise-constant held current, changed by
`DriveUpdate` boundary events. It stays inside the accepted trajectory because
the selected drive binding is constant over each open gap. A DRIVE port binds a
named neuron parameter that the resolver proves safe for the current exact or
stepped program. Genuinely varying within-gap forcing such as ramps, sinusoids,
or interpolation requires an additional analytical forcing solution or a
stepped source and an event-versus-stepped synchronization boundary; it is
deferred rather than approximated silently.

### 5.2 Current encoders

The current encoder set is deliberately bounded. Learned,
rank-order/population, and level-crossing encoders are deferred.

`NATIVE_EVENT` is exact passthrough for an already timestamped external event. It performs no rate conversion or retiming. Native spike input remains available even when higher-level scalar presentations are used elsewhere in the same graph.

`REGULAR_RATE` maps the normalized value linearly to

$$
r(x)=r_{min}+x(r_{max}-r_{min}).
$$

It integrates phase, $\phi(t)=\int r(u)\,du$, and emits on integer crossings. A rate change therefore preserves partial progress toward the next spike. Computing each timestamp from the phase anchor rather than repeatedly adding an inter-spike interval avoids cumulative timestamp drift.

`POISSON_RATE` uses the same linear rate mapping and integrates hazard across rate changes. Each input port has an independent counter-based random stream derived from the encoder seed, the port's stable canonical index, and its draw counter. A fresh session recreates the stream exactly; episode reset continues with the next unused draw. Partitioning a continuous presentation into adjacent constant-value segments does not itself restart the random process.

`TTFS` and input latency coding are one encoder rather than redundant names. A normalized value maps inversely to latency,

$$
t_{spike}=t_0+L_{max}-x(L_{max}-L_{min}),
$$

and at most one spike is emitted during the presentation. A configurable silence threshold suppresses low-valued inputs. TTFS describes the one-spike representation; latency describes the value-to-delay mapping used to obtain it.

`BURST` begins at $t_0$, maps value linearly to burst rate, and emits for a fixed configured duration $D$, clipped by the presentation boundary:

$$
t_k=t_0+\frac{k}{r(x)},\qquad
t_k < \min(t_0+D,t_1).
$$

The first spike is at burst onset when $r(x)>0$. Ordinary burst encoding is distinct from regular-rate encoding because activity stops after $D$ even when the presentation remains active.

`LATENCY_BURST` maps value to the TTFS latency above and then emits a fixed-rate burst for fixed duration $D$:

$$
t_{onset}=t_0+L(x),\qquad
t_k=t_{onset}+\frac{k}{r_b},\qquad
t_k < \min(t_{onset}+D,t_1).
$$

It combines temporal onset information with a more robust multi-spike observation. Rate, duration, amplitude, and presentation bounds are validated explicitly; ordinary run/input capacities provide the separate resource bound.

`HELD_CURRENT` applies the affine mapping $u=offset+gain\,x$ at the presentation
start and restores a configured baseline at its effective end. This is
zero-order hold, not an approximation to an arbitrary continuous waveform. It
emits `DriveUpdate` boundary events and is accepted only on an approved DRIVE
binding. Analytical families require a parameter that enters only their affine
constant term; STEPPED permits an RHS-only parameter, including nonlinear RHS
use, provided it does not enter the threshold or readout reset.

The two burst modes expose complementary value mappings: ordinary `BURST`
encodes value in rate with immediate onset, while `LATENCY_BURST` encodes value
in onset with fixed post-onset rate and duration. The internal specification
keeps latency, rate, duration, and amplitude as separate fields so later fixed
count, value-to-duration, or jointly mapped burst variants do not require
replacing the port or event contracts.

### 5.3 Current decoders

The current decoder set consumes output spikes and produces both a final typed summary and, when requested, a chronological decoded-event stream. A result contains the decoded value, contributing spike count, optional first-spike time, and a validity flag. A no-spike TTFS result is invalid rather than being confused with zero latency. Learned and population decoders are deferred, although multiple per-node ports already provide the ownership structure a later population aggregator can consume.

Observation time and emission time are separate concepts. A decoder observes
exact spike timestamps within its half-open window, but its emission policy
determines when an external consumer receives a value. The ABI supports
`ON_EVENT`, `ON_WINDOW_CLOSE`, `ON_EVENT_AND_WINDOW_CLOSE`, and `ON_QUERY`.
Rate accepts window close and query; TTFS accepts event, window close, and query;
temporal weighting accepts all four policies. `ON_QUERY` is implemented for
finite and cumulative rate, TTFS, and temporal weighting. Sliding-rate query is
rejected because it would require bounded timestamp retention not yet provided.
Unsupported combinations fail at graph validation.

`RATE` supports three explicit window policies. FINITE counts spikes in the requested $[t_0,t_1)$ interval. SLIDING counts in $[\max(t_0,t_1-W),t_1)$ for width $W$. CUMULATIVE counts in $[t_{origin},t_1)$. All report both count and

$$
r=\frac{N}{\text{effective window duration}}.
$$

Keeping the policies under one rate decoder prevents three subtly inconsistent count implementations. Online execution may report a provisional value before a finite window closes and a finalized value at the boundary; the batch interface returns the finalized value.

`TTFS` records the first spike in $[t_0,t_1)$ and ignores later spikes until reset or rearm. It returns raw latency $t_{first}-t_0$ or, when requested, latency normalized by window duration. Under `ON_EVENT`, the first spike emits a `FINAL` event at that exact spike time; because later spikes cannot change TTFS, no redundant final event is emitted at window close. If no spike arrives, the decoder emits `NO_SPIKE` at the window deadline. Under `ON_WINDOW_CLOSE`, either the valid result or `NO_SPIKE` is emitted at the deadline. Deadline, no-spike validity, end-boundary exclusion, and rearming are explicit. Cross-neuron winner selection is a future population decoder rather than hidden behavior in a per-neuron TTFS decoder.

`TEMPORAL_WEIGHT` implements a fixed early-spike exponential score,

$$
y=\sum_k \exp\left(-\frac{t_k-t_0}{\tau_w}\right),
$$

over the finite decode window. Configuration selects all spikes or first spike only and may divide by the number of contributing spikes. `ON_EVENT` emits an `UPDATE` after each contributing spike at that spike's exact timestamp. `ON_EVENT_AND_WINDOW_CLOSE` additionally emits a `FINAL` event at the deadline; `ON_WINDOW_CLOSE` emits only that final value. This is intentionally different from a causal recency trace $\sum_k\exp(-(t-t_k)/\tau)$, which favors recent spikes at query time. Learned temporal weights and learned neural decoders are deferred; the fixed exponential form provides a useful sparse temporal readout without introducing training into the current decoder.

The implementation compiles decoder specifications once, creates independent mutable decoder state per run, and consumes each spike directly at the C output sink. The raw chronological spike buffer is an optional second consumer rather than the source used for decoding. A run carries an explicit sparse schedule of `(decoder, local window id, start, end)` assignments and allocates one state record only for each requested assignment. A common-window convenience call expands to the corresponding full decoder-by-window schedule, preserving the simpler API when every port shares boundaries. Windows may overlap. Results retain deterministic schedule order, and every result and decoded event carries its decoder and decoder-local window id.

The scheduler maintains a deadline-sorted index across all sparse assignments and advances decoder observation time before processing simulation work at the same timestamp and once more at the run end. Every window whose deadline is due is therefore closed before a spike at its right boundary is observed, including when no spike or other output occurs at the deadline. A second run-owned node-to-assignment index makes spike consumption proportional to the number of scheduled decoder windows observing the firing node rather than to every schedule entry. Sequential windows rearm TTFS naturally, while overlapping windows observe the same source spike into independent state. The graph API accepts either one common window sequence or a mapping from output-port id to its own window sequence; omitted decoder ports allocate no state and produce no result for that run.

For `ON_QUERY`, the run carries explicit `(decoder, local window id, time)`
bindings. A due query emits a `QUERY` snapshot at that exact observation time
before a same-time source spike and before the window-close action. A query at
the right boundary is legal and therefore reports the half-open window without
including a boundary spike. Queries are read-only decoder observations; they do
not enqueue neuronal work or alter simulation causality.

A bounded decoded-event buffer makes retention and overflow explicit; capacity zero disables event retention while final summaries continue to work. Explicit sealing is idempotent, consuming after sealing is invalid, and reset clears all window states and retained events. Advancing across one or several deadlines reserves every required close event before changing any window to closed, avoiding partial boundary processing on overflow.

### 5.4 Recording

High-level recording has three independent products. Raw output spikes may be
retained or streamed. Explicit-time state inspection samples selected neuronal
state without mutating the run. The causal trace records typed scheduler facts
and optional before/after neuronal state. Runtime statistics and final state are
returned separately. Learning state and final weights have their own compact
result records; edge learning traces are not neuronal `StateRecording`
variables.

A causal trace may remain in a bounded in-memory buffer or stream to a persisted
artifact. Persisted artifacts encode every causal record kind uniformly. They
preserve binary64 timestamp and payload bits, use lossless timestamp-bit XOR and
sequence-delta encoding, optionally compress chunks with zlib, protect each
chunk with a CRC, and finish with an indexed trailer and identity metadata.
Human readability is not a format goal; deterministic integrity-checked replay
is.

#### State inspection and reconstruction

`AtTimes` and `Every` expand into explicit inspection timestamps. At an event
timestamp, inspection observes the settled post-cascade state. Between events,
the C evaluator propagates a private copy through the node's resolved analytical
or numerical operation. This does not add an integration timestep, change live
state, invalidate a prediction, consume hazard, or alter statistics. A dense
regular inspection schedule does cost additional read-only evaluations and is
therefore paid only when requested.

Causal trace state snapshots use the event record's defined before/after
semantics. A post-update event anchor is the initial condition for the following
gap. Offline reconstruction combines such anchors, the embedded graph identity,
and replayed drive/refractory control state. Analytical nodes evaluate their
exact resolved trajectory; STEPPED nodes re-integrate with the same C numerical
contract. Results identify whether they came from an exact analytical
trajectory, numerical propagation, a recorded event, or an event-only partial
selection.

Reconstruction is available for both analytical and STEPPED nodes. Analytical nodes evaluate only the dependency closure of the selected state-transition roots. A STEPPED node re-integrates from the most recent complete recorded anchor through the same C numerical evaluator and therefore requires every coupled state at that anchor, even when the query selects only one state. Partial stepped-state recordings remain valid at recorded event times but are `EVENT_ONLY` between anchors. Reconstruction records whether its source was analytical propagation, numerical propagation, or a recorded event.

#### Selection

Recording is selective per neuronal state variable, so a node carrying a
membrane variable and a folded alpha's two synaptic dimensions can be recorded
for membrane potential alone. Edge-local plasticity traces live in a separate
learning arena and are returned through the learning-state interface rather
than selected as neuronal state variables.

Selective recording and exact reconstruction are in tension, and the recorder
is explicit about which is being obtained. Reconstruction between events uses
the complete coupled trajectory program, so recording $v$ while discarding the
synaptic state that drove it can make $v$ unreconstructable at intermediate
times. Recording a partial core still yields correct values at event times,
which is sufficient for plotting and spike-adjacent analysis. The current
diagnostic follows dependencies in the resolved state-transition program; it
does not apply a generic authored-observer shortcut.

Recording configuration validates requested node and state names against the
resolved layout. Partial selections are accepted without a pre-run dependency
warning. After a trace exists, `TraceReconstructor.diagnostic()` computes the
required dependency closure and reports whether an intermediate-time query is
analytically reconstructable, numerically reconstructable, or event-only, with
the missing state components when applicable.

#### Causal-trace contract

The recorder exposes typed native-precision causal records directly from the C scheduler. The supported kinds are external input, edge delivery, drive update, third-factor modulation, refractory release, stale and confirmed prediction, aggregated deposit application, emitted spike, simultaneous reset, refractory entry, and final state. Every retained record carries an exact timestamp, deterministic trace sequence, scheduler phase, node, optional edge/input/binding/modulator subject, prediction or refractory generation, and relevant scalar value. A modulation record identifies the named modulator index and signal value; its target edge set and rule descriptors come from the hashed graph, while learning updates are derivable from the corresponding arrival, observation, postsynaptic-spike, and modulation records plus the compiled program. Mutation records may carry bounded before/after neuron-state arrays. Both arrays denote the named event timestamp: for example, `DEPOSIT_APPLY` captures state after analytical propagation to the timestamp immediately before the aggregate deposit and again immediately afterward, while `RESET` captures the complete pre-reset and atomically committed post-reset vectors.

Configuration selects record kinds, graph nodes, whether state is captured, and which local state indices are returned. The low-level evaluator uses resolved node indices; the graph API accepts and returns public graph node ids and annotates selected state positions with their model state names. Selections are validated before execution. A run chooses exactly one sink: a bounded caller-owned buffer or a callback invoked synchronously as records occur. The buffer never drops records and returns `TRACE_OVERFLOW` when full; the callback path retains no trace in the run result and propagates consumer failure out of the run. Recording disabled is the zero-overhead semantic path apart from fixed instrumentation branches, and tracing does not enqueue events or otherwise participate in simulation causality.

The recording path includes the in-memory inspection ABI, compact chunked
persisted trace artifacts with identity metadata and integrity checks,
post-run dependency diagnostics, offline analytical and stepped
reconstruction, and read-only queries at explicitly requested non-event times.
Explicit-time inspection is non-mutating: it advances a copied state with the
node's resolved analytical or numerical evaluator and cannot invalidate
predictions or change scheduler state.

#### Implemented trace-backed invariant audit

The validation harness can request an unfiltered, all-node, full-state causal trace and pass it to an independent host-side auditor. The auditor does not propagate neuron equations, solve a crossing, or reproduce deposit arithmetic. It derives scheduler facts from the immutable topology, the emitted causal record, and the result returned by the C evaluator. This keeps it independent enough to detect scheduler faults without creating a second simulator whose agreement could merely duplicate the same numerical mistake.

The audit establishes chronological contiguous trace sequencing; the phase and payload contract of every record kind; complete native-precision state snapshots; same-timestamp before/after state continuity; one atomic reset per spike; legal refractory entry, release, and exclusion of firing while clamped; application of every same-time input and delivery batch before firing; exactly one in-horizon delivery per emitted spike and outgoing edge; consistency of stale and confirmed prediction generations; agreement between trace-derived event counts and runtime statistics; and one canonical final-state record per node equal to the returned state. Autonomous ROOT_FIND threshold checks use the resolved capability tolerance, while copied state, ordering, generation, and accounting facts are compared exactly.

Filtered traces are intentionally rejected by this audit because absence from a filtered record is not evidence that an event did not occur. Selective tracing remains the analysis interface; the full audit profile is a validation mode. The generated network campaign applies it to both the curated validation corpus and the scaling corpus, and adversarial tests independently corrupt each invariant class to verify that the auditor rejects the resulting trace. The machine-readable report retains audit check names and record counts rather than the potentially large trace payload.

## 6. The resolver pipeline

The resolver is a build-time pipeline of symbolic analyses that routes each node
to supported operations and produces their artifacts. Its intelligence is in
the routing logic, not one solver call. SymPy provides build-time symbolic work.
Resolution is node-structured and produces stable structural identities, while
the current public graph path resolves the complete immutable graph each time.

### 6.1 Parse against a fixed namespace

Parse DSL statements into symbolic expression trees using `sympy.sympify`
against a pre-declared table containing exactly the allowed state variables,
parameters, and reserved names. Reject any unknown symbol so a typo cannot
silently become a free variable. The distinct statement forms from section 4.1
let the parser tag dynamics, reset, and the mutually exclusive threshold or
hazard. Hard thresholds are normalized to rising-edge crossing expressions;
hazards are normalized to a conditional-rate expression for capability
analysis.

### 6.2 Recognize an executable dynamics capability

The current resolver does not implement a generic constant-Jacobian matrix
pipeline. It recognizes bounded equation families by symbolic coefficient and
equivalence checks over authored equations and semantic state roles. In order,
graph resolution selects deposit-triggered reactive IF, scalar or adaptive
escape, independent-current adaptive LIF, scalar LIF, or the supported folded
and per-edge filtered-current variants. A valid fixed-threshold ODE that does not
match an exact family is offered to the bounded STEPPED resolver. Failure of
both paths is a capability error with evidence; the engine never assumes that
an arbitrary affine system has an implemented analytical crossing solver.

Standard Python model classes enter this same equation inspection. Their names
do not select C handlers. The resolver verifies their emitted equations just as
it verifies equivalent user-authored DSL.

### 6.3 Validate the bounded state layout

Exact resolvers require the state roles and equation structure of their proved
families: one membrane for scalar LIF, membrane plus one independent adaptation
state for adaptive LIF, or a bounded merged layout for exponential/alpha
currents. STEPPED accepts one through eight authored states under its fixed
readout/threshold/reset contract. The parser accepts `OBSERVER` and `AUX` roles,
but there is no generic dependency-reachability partition today; adding an
otherwise unsupported state can change resolution to STEPPED or rejection.
Learning traces avoid this issue by living in a non-neuronal arena.

### 6.4 Build the accepted trajectory program

For each exact family, the resolver constructs its singular-safe closed
trajectory directly from the recognized coefficients. Scalar affine terms use
exponentials and `phi1` limits. Adaptive, exponential-current, alpha-current,
and repeated-rate families use their proved one-, two-, or
polynomial-exponential formulas. Equal-rate expressions are emitted explicitly;
a distinct-rate formula is never evaluated across its removable singularity.

Normal state, clamped state, simultaneous reset, and analytical crossing roots
share the propagation DAG when that family needs them. Integrated hazard stores
its normalized log-rate parameters in a numeric descriptor and may reference
trajectory roots in the propagation DAG. A stateful programmed deposit uses a
separate deposit DAG with the same parameter layout; a delta deposit lowers
directly to `ADD_STATE` and has no deposit DAG. These specialized programs are
mathematically equivalent to the admitted affine trajectories, but the
implementation does not store or exponentiate a general symbolic $A$ matrix. A
future general affine resolver would need constant-Jacobian analysis,
singular-safe input response, automatic dependency partitioning, and a
certified first-crossing module before it could broaden this boundary.

### 6.5 Classify the crossing and dispatch tier

Classify intrinsic spike generation after validating the coupled core from stage
6.3. Hard-threshold analytical families require a stable affine LIF core with
real negative decay rates and a fixed scalar membrane threshold. They include
delta input, a folded alpha kernel, one independently decaying spike-triggered
adaptation current, and bounded edge-scoped scalar exponential or alpha-current
blocks whose complete crossing is a stable real exponential polynomial.
Complex or non-decaying modes and other trajectories outside a certified family
are not admitted to analytical ROOT_FIND; an otherwise valid autonomous
fixed-threshold ODE is then offered to STEPPED. Moving thresholds and other
forms outside the stepped contract are rejected rather than passed to a
heuristic solver.

If the authored node has a hazard, validate the exponential-voltage form in
section 4.3.1 and first resolve its deterministic state trajectory through the
ordinary scalar or adaptive analytical capability. The current graph resolver
admits delta synapses only. It emits `INTEGRATED_HAZARD`, the normalized log
rate, quadrature/inversion tolerances, and, when numerically well conditioned,
equation-derived roots for a bounded exponential modal voltage trajectory.
Near-equal rates or ill-conditioned modal coefficients retain the ordinary
expression evaluator as an exact-trajectory fallback; this changes arithmetic
cost, not stochastic semantics or tolerances.

For a hard threshold, if the level excluding the designated readout contains
dynamic state, it is moving and remains unsupported. A delta-only single-mode
trajectory is CLOSED_FORM and refines to REACTIVE when its asymptote is provably
at or below threshold. One distinct alpha block uses `ALPHA_REAL`; one adaptive
current uses `TWO_REAL_EXP`; a bounded distinct scalar sum uses
`MULTI_REAL_EXP`; equal rates use `REPEATED_REAL_MODE`; and mixed scalar/alpha
blocks use `MULTI_EXP_POLY`. Specialized families remain preferred because
their complete first-root proofs are cheaper than the general bounded
exponential-polynomial isolator.

Structure decides the capability; symbolic solving only fills its proved
artifact. CLOSED_FORM stores $t^*(x)$. ROOT_FIND stores
$g(\Delta)=v_{th}-v(\Delta)$, its derivative, family coefficients or extremum
expressions, and tolerances as roots or coefficient arrays appropriate to the
family. Event-batched REACTIVE IF stores no continuous crossing solution;
scalar LIF in a reactive binding retains its scalar-log program for later drive
changes. INTEGRATED_HAZARD stores a conditional-rate descriptor over the
resolved deterministic trajectory rather than a threshold function.

### 6.6 Lower to the expression DAG

Walk the sympy trees once and emit them into the lowered expression DAG, so that no sympy object survives into the C evaluator or any future backend. Emit parameter references as `Param` nodes bound at runtime, not as `Const` nodes. Weights normally enter through event deposits rather than the autonomous propagator DAG. Run common-subexpression elimination with `sympy.cse` and map its output onto shared DAG nodes, so repeated exponential evaluations are computed once. Python constructs and serializes this DAG, then calls the C evaluator; it does not independently evaluate it.

### 6.7 Structural identity and rebuild behavior

Each resolved node carries a key distinct from its persisted source hash. It
covers equation forms, threshold or hazard definition, reset and refractory
semantics, incoming synapse structure and receptor mappings, parameter domains,
formula regime, spike-prediction capability, and state ordering. Learning and
learning-observer identities are compiled at the plan/connection layer rather
than folded into the intrinsic node-resolution key. Numeric bindings that do
not change those structural facts may be excluded from the identity.

The structural key supports auditability and deduplication inside a compilation.
It is not a sufficient key for reusing a numerically bound model. Successful
parsing and model resolution are additionally memoized in a bounded process-local
cache keyed by the exact operation and complete argument contents: source/model
definitions and defaults, explicit parameter values, numerical configuration,
and all supplied synapse merge context, identifiers, mappings, and initial states.
Numeric keys preserve binary64 bits; there is no rounding or approximate reuse.

`resolve_graph()` continues checking wiring, initial values, ports, and learning
configuration and lowering edges on each call. Unchanged model configurations
may reuse their successful resolution. A graph edit can change dispatch--for
example adding alpha input can move a delta LIF from CLOSED_FORM/REACTIVE to
ROOT_FIND--and the new resolver arguments select a distinct cache entry. Failed
resolutions are never cached. Custom argument types outside supported value
records/containers bypass caching.

The default LRU limits are 4,096 entries and 64 MiB of estimated retained Python
storage per process. Cached records and returned hits are independent copies
because frozen authoring/IR dataclasses contain mutable nested dictionaries.
Thread-safe controls expose statistics, clearing, configurable limits, and
process-wide or context-local bypass. Forked workers retain settings but reset
entries and locks. Clearing/configuration cannot be undone by an earlier
in-flight miss. The cache contains no compiled handles or simulation state and
does not participate in binary image loading. Cached and uncached compilation
must produce identical image bytes and execution results. API details are in
`docs/resolution_cache.md`.

### 6.8 Equation-derived execution plan

After every node and edge resolves, Lacuna lowers one backend-neutral
`ExecutionPlan`. The plan describes physical operations rather than biological
model names:

- state slots and evolution programs, including exact expressions,
  event-batched reactive state, or adaptive numerical RHS evaluation;
- arithmetic operations, currently a proved scalar-affine direct operation or
  the unchanged expression DAG;
- spike prediction by reactive deposit test, closed form, certified analytical
  root isolation, numerical dense-output isolation, or integrated hazard;
- state-add or bounded expression-program deposits;
- explicit connections with enough delay data for C lowering to derive
  equal-delay delivery groups, drive bindings, input and output indexes, and
  modulator scopes; and
- bounded learning programs, observer programs, numeric bindings, and shared
  weight groups.

The plan chooses the least-cost operation whose preconditions the resolver can
prove. A narrower arithmetic operation is an optimization of the same resolved
equation, not a model-specific handler. If a runtime drive rebind invalidates
the cached scalar-affine coefficients, that node falls back to its ordinary DAG
after first advancing under the old coefficients. One compiled sparse scheduler
executes every combination, including mixed analytical, stepped, deterministic,
stochastic, static, and plastic nodes. Operation-specific dependency closures
remove only unreachable DAG nodes and preserve every reachable operation and
evaluation order; they do not perform algebraic reassociation or approximation.

## 7. Certified crossing solvers

ROOT_FIND nodes need the first hard-threshold crossing located numerically.
Lacuna has shared safeguarded primitives and capability-specific isolators.
Sections 7.1--7.9 describe the analytical hard-threshold families, section
7.10 the numerical ODE capability, and section 7.11 intrinsic integrated
hazard. None is a general solver for arbitrary systems. Explicit admission
boundaries ensure that every accepted family has the root-count, horizon, or
numerical-budget contract its predictor requires.

### 7.1 The alpha-family function and why it is not monotone

For a folded-alpha ROOT_FIND node with distinct membrane and synapse decay
rates, the membrane trajectory from an arbitrary current state has the form

$$
v(\Delta) = c_1 e^{-\Delta/\tau_m} + (c_2 + c_3\Delta)e^{-\Delta/\tau_s} + v_\infty,
$$

with the coefficients set by the current state. This is not monotone. A synaptic input makes the membrane rise, peak, and decay back toward $v_\infty$, so $g(\Delta) = v_{th} - v(\Delta)$ starts positive, may dip negative, and may return positive.

When the membrane and alpha rates are equal, the exact trajectory instead has
the repeated-mode form

$$
v(\Delta)=v_\infty+(c_0+c_1\Delta+c_2\Delta^2)e^{r\Delta},
$$

and selects `REPEATED_REAL_MODE`/the exponential-polynomial isolator. It is not
obtained by substituting equal rates into the distinct-rate alpha coefficients.

For the distinct-rate alpha trajectory above, the number of extrema is bounded and known. Setting $v'(\Delta) = 0$ and dividing through by $e^{-\Delta/\tau_s}$ reduces to an exponential equal to a linear function in $\Delta$, which has at most two solutions. So $v$ has at most two extrema, admitting the shape rise to a peak, fall to a trough, then rise toward $v_\infty$, and therefore at most two up-crossings of threshold. The solver wants the first. This bound must not be generalized to the adaptive-current family, equal-rate alpha, nodes with additional folded kernels, complex modes, or higher Jordan structure; each needs its own proof.

### 7.2 Brackets come from structure, not from sampling

Stepping forward in coarse $\Delta$ until $g$ changes sign is incorrect, not merely slow. A narrow peak that rises above threshold and falls back entirely within one coarse step leaves both endpoints below threshold, no sign change is observed, and the spike is silently lost. This is a wrong answer rather than a slow one, and it is the failure mode most likely to survive casual testing, since it appears only when the synaptic time constant is fast relative to the step.

For the alpha capability, the bracket is therefore derived from the extrema. The resolver computes $v'(\Delta)$ symbolically, which it has for free. At runtime the procedure is to find the extrema of $v$ on $[0, \Delta_{horizon}]$ by solving $v' = 0$, use them to partition the interval into at most three subintervals on each of which $v$ is monotone, and walk the subintervals in order. Within a monotone subinterval a sign change at the endpoints is necessary and sufficient for a crossing, so endpoint evaluation is conclusive and the bracket is guaranteed clean and unique. The first subinterval showing a sign change contains the first crossing. No crossing can hide. The two-real-exponential capability applies the same principle with its own tighter one-extremum bound.

For the two-distinct-exponential case the extremum has the closed form $\Delta_{peak} = \ln(-c_2\tau_m / c_1\tau_s) / (1/\tau_s - 1/\tau_m)$. For the alpha case, with its $\Delta e^{-\Delta/\tau_s}$ term, locating the extrema is itself a transcendental solve of exponential against linear. It has at most two roots with known structure, so the implementation partitions a certified horizon and applies the same bounded safeguarded solve to those extremum brackets before solving the crossing.

The horizon is certified from the transient coefficients and the threshold margin, not chosen as an informal number of time constants. For the accepted trajectory family, the runtime or resolver derives a monotonically decreasing envelope $B(\Delta)$ satisfying $|v(\Delta)-v_\infty| \le B(\Delta)$. When $v_\infty < v_{th}$, choose a horizon for which $B(\Delta_{horizon}) < v_{th}-v_\infty$ for all later times; then a late crossing is impossible. The equality case $v_\infty = v_{th}$ is handled separately as asymptotic approach and does not count as a finite rising-edge crossing. When $v_\infty > v_{th}$, a crossing exists after the transients settle, though the extremum partition is still used to locate the first one. Failure to construct a certified envelope is a resolver capability failure, never permission to use a heuristic cutoff.

### 7.3 Family-specific bracket refinement

`ALPHA_REAL` and `TWO_REAL_EXP` refine their certified brackets with
safeguarded Newton. Newton uses the emitted derivative and bisection supplies
the guaranteed floor. `MULTI_REAL_EXP` and the repeated/mixed
exponential-polynomial families instead use bounded bisection after their
recursive root isolators have produced unique brackets. STEPPED dense-output
crossings also bisect monotone polynomial intervals. Integrated hazard uses its
own safeguarded Newton/bisection inversion of cumulative hazard.

For the safeguarded analytical primitive, bisection replaces a Newton step that
would leave the bracket, has a derivative below the numeric floor, or would fail
to cut the bracket width by at least half. The slow-progress guard matters near
a flattening of the trajectory, where a legal Newton step may otherwise consume
the iteration budget without providing bisection-quality contraction.

These solvers are implemented in C rather than delegated to a platform root
library. Python does not run a second solver. A future backend should use the
same family-specific isolation, bracket updates, and termination tests where
practical. Cross-platform results are compared under the recorded numerical
tolerances unless both builds intentionally share a deterministic math and
floating-point environment.

### 7.4 Tolerance

The tolerance is relative and scaled to the accepted family's fastest resolved
time constant, $\min(\tau_m,\tau_s,\ldots)$. Fastest rather than slowest is used
because the fastest mode sets how quickly the trajectory can pass threshold.
The resolver records the resulting per-node tolerance in its capability-specific
root hint; no general eigenstructure summary is retained.

A floor is still required. A relative tolerance alone can demand a bracket narrower than the floating-point spacing of the time value, which never converges and always hits the cap. The effective test is therefore the relative tolerance or a machine-epsilon-scaled absolute floor, whichever is larger.

### 7.5 Iteration cap and non-convergence

All numerical predictors have finite loops or descriptor budgets. The base
analytical root primitive uses `LC_ROOT_MAX_ITERATIONS=192`; multi-mode and
integrated-hazard limits are carried in resolved descriptors. Analytical
crossing and hazard quadrature/inversion exhaustion return hard errors. The
network error record includes detailed analytical-root diagnostics, but does
not yet provide equivalent structured root context for every stepped or hazard
failure.

The current STEPPED dense-output extremum and crossing bisections stop at the
shared iteration cap and return their midpoint without separately proving that
the requested event-time tolerance was met. This is a known numerical
conformance gap, recorded in section 12.1. There is no general
`continue_on_nonconvergence` option; callers cannot opt into a knowingly
inaccurate analytical or hazard result.

A non-converged crossing yields a wrong spike time whose effects may propagate downstream and diverge trajectories. Zero non-convergences is therefore a precondition of every reference or cross-backend comparison.

### 7.6 Capability-specific root hints

There is no single runtime `BracketHint`. Python emits distinct
`RootFindHint`, `TwoExpRootHint`, `MultiExpRootHint`, and `ExpPolyRootHint`
records. Alpha and two-exponential hints name DAG roots for $g$, $g'$, limits,
coefficients, rates, and extrema as required; C lowering converts those names
to root indexes. Multi-exponential and exponential-polynomial hints instead
carry limit/rate/coefficient arrays from which the C isolator evaluates its
recursive functions, so they do not require generic $g$ and $g'$ roots.

Roots that do coexist in a propagation DAG share common subexpressions. In the
alpha and adaptive families this avoids recomputing common exponentials during
an iteration. The family descriptor selects a proved isolator; it never asks a
generic primitive to discover an arbitrary trajectory's shape.

### 7.7 The accuracy asymmetry

CLOSED_FORM and REACTIVE nodes use a fixed sequence of profile-selected operations and
transcendentals with no grid discretization. ROOT_FIND adds a root tolerance.
Integrated hazard adds deterministic quadrature/inversion tolerances plus seeded
sampling. STEPPED adds adaptive ODE error and dense-event tolerances. Lacuna's
analytical timing removes a fixed global grid where the analytical capability
applies; it never implies exact real arithmetic.

### 7.8 Implemented two-real-exponential adaptation capability

The first adaptive-current family has one membrane state and one independently decaying adaptation state,

$$
\dot v = a v + c w + b, \qquad \dot w = q w,
$$

with $a<0$, $q<0$, $c<0$, and an own-spike update $w\leftarrow w+\beta$ for $\beta\ge 0$. The affine term $b$ may be rebound by piecewise-constant drive events. The adaptation variable is zero-centered and independently decaying: the current capability does not admit voltage feedback into $\dot w$. For distinct rates, its fixed-threshold crossing function has the form

$$
g(\Delta)=g_\infty+d_1e^{a\Delta}+d_2e^{q\Delta}.
$$

Its derivative is a sum of two exponentials and therefore has at most one finite extremum. When the two derivative terms have opposite signs, the extremum time is obtained directly from their coefficient ratio. The extremum partitions the certified horizon into at most two monotone intervals, which are walked in chronological order and solved with the same safeguarded Newton primitive as the alpha capability. A tail envelope

$$
|g(\Delta)-g_\infty|\le |d_1|e^{a\Delta}+|d_2|e^{q\Delta}
$$

certifies a horizon after which the sign of $g$ cannot change. The
$g_\infty=0$ case is handled separately as asymptotic approach plus, when the
two transient coefficients oppose, the one possible finite logarithmic zero.
Exact equality $a=q$ is outside this exact adaptive capability. The standard
`AdaptiveLIF` constructor rejects it; an equivalent custom fixed-threshold ODE
may fall back to STEPPED if it satisfies that capability. The resolver never
substitutes equality into the singular distinct-rate formulas.

The own-spike update is a simultaneous reset map over the complete state, not scheduler-specific adaptation logic. For a fixed clamp, $v$ is held at reset and is non-excitable while $w$ continues its exact decay. Inputs and drive changes received during the clamp follow the ordinary refractory policy, and prediction resumes from the analytically advanced state at release. This reset-program interface is also the path later adaptive-threshold and other affine adaptive capabilities use.

This model provides spike-frequency adaptation: recent spikes transiently suppress subsequent firing. It is not by itself target-rate homeostasis, because it has no measured-rate state, set point, or feedback law that drives the neuron toward a prescribed firing rate. That distinction remains explicit in the future plan in section 11.

### 7.9 Implemented bounded exponential-polynomial capability

The repeated-rate and multi-alpha edge families reduce the fixed-threshold crossing to

$$
g(t)=L+\sum_{j=1}^{m}P_j(t)e^{r_jt},\qquad r_j<0,
$$

where the $r_j$ are distinct real rates after equal-rate blocks are combined and each $P_j$ is a real polynomial. A scalar exponential block has degree zero unless its rate equals the membrane rate, in which case it contributes a degree-one term. An alpha block has degree one at a distinct rate and contributes through degree two when its rate equals the membrane rate. The current descriptor limits the complete analytical node to eight state components, and the flattened transient coefficient count is bounded by the same limit.

Root isolation uses a finite generalized-Rolle recursion rather than time sampling. For any rate $r$, define $H=(D-r)F$. Because

$$
\frac{d}{dt}\left(e^{-rt}F(t)\right)=e^{-rt}H(t),
$$

the ordered roots of $H$ partition the domain into intervals on which $e^{-rt}F$ is monotone, and hence each interval contains at most one root of $F$. Choosing $r$ from one of $F$'s blocks reduces that block's polynomial degree by one. Repeating the operation therefore reduces the total coefficient count at every recursion and terminates. Equivalently, these functions form an extended complete Chebyshev system on every finite interval, so a nonzero function with $N$ flattened coefficients has at most $N-1$ finite roots counted with multiplicity. Here $N$ includes the nonzero constant block when $L\ne0$. This is the loop and storage bound used by the C isolator.

The horizon begins beyond every polynomial-exponential monomial peak,
$d/|r|$, and every positive normalized cross-block ratio peak of the form
$(d_{fast}-d_{slow})/(r_{slow}-r_{fast})$. It then doubles until a
polynomial-weighted transient envelope is strictly dominated by the nonzero
limit margin. For zero limit, the slowest nonzero block supplies the eventual
sign after the remaining normalized envelope is dominated. Every isolated zero
is checked for direction from the signs on its adjacent root-free intervals.
Only positive-to-negative $g$ is a rising voltage crossing; a same-sign
threshold touch is a tangency and emits no spike. Floating-point ambiguity or
exhaustion of the fixed iteration budget is a hard non-convergence error.

This capability does not replace cheaper solvers. Scalar-log, one distinct alpha, two-real-exponential, and pure distinct scalar multi-exponential nodes keep their specialized paths. `REPEATED_REAL_MODE` uses this representation when all transient terms share one rate; `MULTI_EXP_POLY` uses it for multiple polynomial-rate blocks.

### 7.10 Implemented generic stepped ODE capability

The current nonlinear fallback is a bounded vector Dormand--Prince 5(4)
integrator implemented in the C evaluator. Python parses, validates, resolves,
and lowers equations but never evaluates a right-hand side. A stepped program
uses the canonical variable layout `[Time, x0, ..., xn]`; authored equations are
currently autonomous, so `Time` is reserved for ABI compatibility but is not an
author-visible symbol. The safe intrinsic set is explicit: arithmetic, real
powers where defined, `exp`, `log`, `sqrt`, `sin`, `cos`, and `tanh`. Unknown
functions, host callbacks, and expressions that produce a non-finite real value
fail rather than escaping into Python.

Each attempted step evaluates the ordinary seven Dormand-Prince stages, forms the fifth-order update and embedded fourth-order error estimate, and accepts under a componentwise scale

$$
s_i = atol + rtol\max(|x_i|,|x_i^{new}|).
$$

The maximum normalized component error must not exceed one. Step growth and shrinkage are bounded, the minimum and maximum step are explicit, and both attempted-step and right-hand-side-evaluation budgets are hard errors. The default contract is `rtol=1e-8`, `atol=1e-10`, initial step `1e-3`, minimum step `1e-12`, maximum step `0.25`, event-time tolerance `1e-9`, at most one million attempted steps, and at most seven million right-hand-side evaluations. Units remain those of the graph; users whose state scales make these defaults inappropriate will receive a future graph-level numerical configuration rather than an implicit rescaling.

Threshold detection is not endpoint sampling. On every accepted step, the seven stage derivatives define the Dormand-Prince quartic dense interpolant for the designated readout. Its derivative is cubic. The evaluator isolates every real derivative root in the unit step interval by partitioning the cubic at the roots of its quadratic derivative, then uses those extrema to partition the readout polynomial into monotone intervals. It walks those intervals chronologically and bisects the first negative-to-nonnegative threshold gap whose direction is rising. A narrow above-threshold excursion whose step endpoints are both below threshold is therefore still found; a tangent touch does not emit a spike. The returned prediction is generation stamped and enters the same scheduler queue as analytical predictions.

Advancement, prediction, inspection, refractory propagation, drive-boundary invalidation, reset, recording, and reconstruction all call this same C evaluator. During a fixed clamp the readout derivative is forced to zero and the readout remains at reset, while every other derivative is evaluated normally against the clamped readout. A matching prediction generation fires without a second threshold test, exactly as in the analytical tiers. Step and event diagnostics are available on the direct evaluator API; aggregate per-network step statistics are a later observability extension.

The current admission boundary is deliberate. There must be one through eight declared states, exactly one membrane role, one equation and one simultaneous reset expression per state, a fixed rising threshold on a declared readout, and a state-independent readout reset strictly below threshold. Other reset components may depend on pre-reset state, which permits the AdEx update `w <- w+b`. Moving thresholds, state-dependent readout resets, SDEs, algebraic equations, discontinuous callbacks, delayed state inside the ODE, arbitrary continuous external waveforms, and automatic merging of stateful edge kernels into nonlinear nodes remain future capabilities.

### 7.11 Implemented intrinsic integrated-hazard capability

`EscapeLIF` and `AdaptiveEscapeLIF` use the conditional intensity

$$
\lambda(v)=\lambda_0\exp\!\left(\frac{v-v_{escape}}{\Delta_v}\right)
=\exp(\ell_0+\gamma v),\qquad \lambda_0,\Delta_v,\gamma>0.
$$

For each prospective spike, the runtime obtains a counter-based uniform draw and
stores $E=-\log U$, a unit-rate exponential target. It predicts the first
$\Delta$ within the run horizon satisfying

$$
\int_0^\Delta \lambda(v(s))\,ds=E,
$$

where $v(s)$ is the already resolved exact scalar- or adaptive-LIF trajectory.
The integral uses bounded adaptive Simpson quadrature. Once a segment contains
the target, safeguarded Newton uses $\lambda(v(\Delta))$ as the derivative and
falls back to bisection inside the cumulative-hazard bracket. Relative,
absolute, time, quadrature-depth, and inversion-iteration limits are part of the
resolved hazard descriptor. Budget exhaustion is a hard numerical error.

When the resolver can express voltage as a well-conditioned bounded modal sum
$v_\infty+\sum_k c_ke^{r_kt}$, the runtime evaluates its coefficients once and
uses direct exponential arithmetic at quadrature samples. Near-equal rates or
large cancellation disable that specialization and evaluate the unchanged
exact propagation DAG instead. Neither path polls spike probability on a time
grid.

If a delivery or drive boundary precedes the predicted spike, the runtime
integrates and subtracts hazard through that event, applies the event, and
reschedules against the same residual $E$. Only a confirmed spike applies the
reset and draws a new target. A fixed refractory clamp pauses hazard and retains
the pending post-spike target while non-readout state continues evolving. Draws
are keyed by the run seed, canonical compiled neuron index, and per-neuron draw
index, making fixed-seed one-shot and incremental runs reproducible and immune
to stale prediction count.

This is a bounded conditional point-process capability, not support for SDEs or
arbitrary stochastic equations. It currently permits delta synapses and mixed
deterministic/stochastic populations. Filtered-current hazard neurons are
rejected until their trajectory/integral combinations have their own validation
matrix.

## 8. Dispatch tier decision, consolidated

A single statement of how a node lands in each form, since the logic is
distributed across the pipeline above. Learning traces live outside neuronal
state, so attaching supported plasticity does not
change intrinsic dispatch. Generic authored observer partitioning is not yet a
resolver capability.

STEPPED. No narrower analytical capability accepts an otherwise valid
autonomous fixed-threshold ODE node. This commonly includes nonlinear dynamics,
but the criterion is capability admission rather than a generic Jacobian test.
The stepped descriptor carries one DAG containing the vector right-hand side
and simultaneous reset roots, a designated readout, a fixed threshold, and a
bounded numerical configuration. State propagation uses adaptive
Dormand--Prince 5(4); threshold prediction uses its quartic dense output and
isolates all extrema of that interpolant before selecting the first rising
root. AdEx and quadratic integrate-and-fire are canonical acceptance families.
Analytical nodes retain priority whenever a cheaper certified capability
applies.

ROOT_FIND. The resolver recognizes one of the stable real trajectory families
for which Lacuna has both an exact state program and a complete first-root
isolator, but the crossing has no supported closed inverse. Implemented
families are LIF plus alpha current, LIF plus one independently decaying
spike-triggered adaptation current, pure bounded sums of stable scalar
exponential current, and bounded stable real exponential polynomials produced
by repeated rates or multiple alpha edge blocks. The crossing time is computed
once per reschedule. Other multimode systems remain future analytical
ROOT_FIND candidates and may use STEPPED when they satisfy its admission
contract. Moving thresholds remain unsupported by both paths and are rejected
rather than handled heuristically.

CLOSED_FORM, driven. A recognized scalar LIF trajectory has a logarithmic
crossing and its asymptote can sit above threshold. The next-spike time is a
formula evaluated once per reschedule. A tonically driven LIF neuron lives
here.

REACTIVE. There are two subfamilies. A scalar delta LIF whose current binding
has an asymptote at or below threshold propagates its exact decay but schedules
no autonomous spike under that binding. Its retained scalar-log program and
runtime guard allow a later DRIVE update to enable an interior crossing. A
deposit-triggered integrate-and-fire node instead has zero between-event
dynamics, an explicit `HOLD` or `RESET_BEFORE_DEPOSIT` policy, and can fire only
after deposits. They share sparse scheduling cost while retaining different
propagation and prediction contracts.

INTEGRATED_HAZARD is a spike-prediction method over an accepted analytical
trajectory, not a fifth dispatch form. Scalar `EscapeLIF` retains exact scalar
evolution; adaptive escape retains exact two-mode evolution. Their next event is
sampled by cumulative-hazard inversion rather than hard-threshold crossing.

The delta case is the $\tau_s \to 0$ limit of the filtered synapses. As the synaptic time constant shrinks, the interior crossing migrates toward the delivery instant, and in the limit the crossing collapses onto the delivery time. This is why delta needs no interior crossing search while alpha does.

## 9. Worked examples

These show the post-merge internal view. The user writes an intrinsic-only neuron plus a separately defined synapse type, per section 4, and the resolver merges them to produce the augmented forms below.

### 9.1 Current-based LIF with delta synapses

One state variable, $x = [v]$. Writing $I=R\,\text{drive}$ as an effective
voltage, the standard dynamics are
$dv/dt = -\frac{1}{\tau_m} v + \frac{v_{rest}}{\tau_m} + \frac{I}{\tau_m}$.

The resolver extracts $A = [-\frac{1}{\tau_m}]$ and $b = [\frac{v_{rest} + I}{\tau_m}]$. The propagator is the scalar $\exp(A\Delta) = e^{-\Delta/\tau_m}$. The state solution is $v(t+\Delta) = v_\infty + (v(t) - v_\infty) e^{-\Delta/\tau_m}$ with $v_\infty = v_{rest} + I$. The threshold $v - v_{th}$ is state-independent and the trajectory is a single exponential, so the crossing inverts to $\Delta^* = \tau_m \ln\frac{v(t) - v_\infty}{v_{th} - v_\infty}$.

The initial dispatch is REACTIVE when the current binding gives
$v_\infty\le v_{th}$ and CLOSED_FORM otherwise. The scalar-log crossing roots
remain available in either case, and the runtime asymptote guard responds to
DRIVE updates. Synapse tier is DELTA, depositing weight directly into $v$ when
the membrane is not clamped.

### 9.1.1 Deposit-triggered integrate-and-fire

One state variable, $x=[v]$, with $dv/dt=0$. The normal and clamped propagation roots are the identity and the own-spike reset root is the authored fixed reset. `HOLD` computes

$$
v(t_k^+) = v(t_{k-1}^+) + \sum_j d_j(t_k),
$$

while `RESET_BEFORE_DEPOSIT` computes

$$
v(t_k^+) = v_{reset} + \sum_j d_j(t_k).
$$

The sums include every deposit already queued for that node and timestamp. Threshold is checked once after the sum. A spike emits the ordinary output event, schedules delayed outgoing deliveries, and applies the same reset map used by all other node capabilities. No prediction event exists between $t_{k-1}$ and $t_k$. The standard Python constructor is `IntegrateAndFire`; its `leak=True` option selects timestamp reset rather than continuous exponential leak.

### 9.2 LIF with an alpha synapse

Post-merge, three state variables, $x = [v, s, z]$. The user wrote a one-variable intrinsic neuron with receptor `i_exc` and mapped a separate alpha synapse output `s` into it. The alpha response $t\,e^{-t/\tau_s}$ requires two synaptic variables cascaded at the same rate, since a single first-order equation cannot produce a $\Delta$-times-exponential term. A spike deposits $w/\tau_s^2$ into $z$, which makes the integral of the resulting $s$ kernel equal to $w$. This normalization gives `w` the same integrated impulse meaning across alpha time constants and makes the $\tau_s \to 0$ delta-limit validation well-defined.

```
dv/dt = -(1/tau_m) v + (v_rest / tau_m) + s     # i_exc mapped to synapse output s
ds/dt = -(1/tau_s) s + z                        # from the merged alpha synapse
dz/dt = -(1/tau_s) z                            # from the merged alpha synapse
```

The recognized equations correspond mathematically to the block-triangular

$$
A = \begin{bmatrix} -\tfrac{1}{\tau_m} & 1 & 0 \\ 0 & -\tfrac{1}{\tau_s} & 1 \\ 0 & 0 & -\tfrac{1}{\tau_s} \end{bmatrix}, \qquad b = \begin{bmatrix} \tfrac{v_{rest}}{\tau_m} \\ 0 \\ 0 \end{bmatrix}.
$$

The resolver recognizes the alpha block's repeated synaptic rate
$-1/\tau_s$, which produces the $\Delta e^{-\Delta/\tau_s}$ term, and the
membrane rate $-1/\tau_m$. The emitted trajectory mixes
$e^{-\Delta/\tau_m}$, $e^{-\Delta/\tau_s}$, and
$\Delta e^{-\Delta/\tau_s}$. Setting it equal to a fixed threshold has no
symbolic inverse; no general eigenstructure table is stored.

Tier is ROOT_FIND. State propagation uses the exact three-state trajectory
formula constructed for this block. The crossing is found numerically on
$g(\Delta) = v_{th} - v(\Delta)$. Synapse tier is FOLDED_SHARED, with $s$ and
$z$ folded into the node's augmented state and memory scaling with nodes.

### 9.3 LIF with a spike-triggered adaptation current

Two intrinsic states, $x=[v,w]$, evolve as

```
dv/dt = -(v - v_rest)/tau_m + resistance*(drive - w)/tau_m
dw/dt = -w/tau_w
```

and the own-spike reset is `{ v <- v_reset; w <- w + beta }`. Each spike therefore increases an inhibitory current which decays between spikes. For sustained drive this lengthens subsequent inter-spike intervals and approaches an adapted firing regime.

The authored standard DSL labels $v$ as `MEMBRANE` and $w$ as `ADAPTATION`;
the resolver validates those roles, $\tau_m,\tau_w>0$, inhibitory coupling,
$\beta\ge0$, a fixed threshold, and a reset below it. For distinct rates the
exact trajectory is
$v_\infty+c_m e^{-\Delta/\tau_m}+c_w e^{-\Delta/\tau_w}$ and uses
`TWO_REAL_EXP`. Equal membrane/adaptation rates remain outside this particular
adaptive-neuron capability even though repeated-rate current-kernel crossings
exist elsewhere; singular distinct-rate coefficients are never used.

During a fixed refractory clamp, the membrane root is held at $v_{reset}$ while
the adaptation root continues as $w e^{-\Delta/\tau_w}$. Delta synapses still
map to the membrane but their readout-targeted deposits are ignored while it is
clamped; deposits to other retained state would follow that state's resolved
policy. Piecewise-constant current injection rebinds the affine membrane drive.
The C evaluator advances both roots and applies the vector reset atomically.
Python only supplies the resolved program, bindings, inputs, and orchestration;
it does not evaluate an independent adaptive trajectory.

## 10. Validation strategy

Correctness is existential for a simulator, since a wrong propagator produces a wrong answer rather than a crash. The strategy has four layers, ordered here by strength of the guarantee each provides.

### 10.1 Resolver-level symbolic checks

Resolver tests compare every admitted standard family with hand-derived
trajectories or independent numerical references, verify dispatch and crossing
methods, exercise equal/distinct-rate guards, and reject unsupported equation,
role, hazard, synapse, and reset forms. Adaptive-current tests cover the exact
two-state trajectory, simultaneous reset, authored `ADAPTATION` role, stable
real/inhibitory guards, and the rejected equal membrane/adaptation rate. Generic
observer-partition tests are future work because that partition is not yet
implemented.

### 10.2 Analytical runtime checks

The strongest and cheapest runtime layer. REACTIVE and CLOSED_FORM nodes have independently known analytical answers, so the C evaluator is compared against high-precision offline expectations under a small, explicit floating-point tolerance. A tonically driven LIF has an analytically known firing period. A single delta input at a known state has an analytically known response. These tests catch propagator errors, sign errors, reset bugs, and transcendental-domain mistakes immediately without claiming exact real arithmetic.

The more valuable variant validates the ROOT_FIND path against closed form by exploiting that the unit-area normalized alpha kernel becomes a delta distribution as $\tau_s \to 0$. Running a ROOT_FIND node with decreasing positive synaptic time constants inside supported numerical regimes and confirming convergence toward the analytically known delta result validates the root-finder and the extrema partition against a known limit.

For adaptive-current nodes, exact propagation is compared with the hand solution and checked for the semigroup property. Predicted crossings are compared with a high-precision offline root calculation, including trajectories with one extremum, no finite crossing, an asymptote exactly at threshold, and distinct rates close enough to expose cancellation. Reset and refractory tests verify that $w$ jumps exactly once on a spike and decays analytically while $v$ is clamped. The exact resolver and standard constructor reject equal-rate or unstable adaptive regimes explicitly; equivalent custom equal-rate ODEs are separately eligible for STEPPED rather than silently using singular exact coefficients.

### 10.3 Property-based invariant checks

Invariants must hold regardless of model, and they catch the queue and bookkeeping failures that analytical unit tests cannot, since those tests use one or two nodes while the interesting failures are in the invalidation logic under load. The invariants include that no node fires during a fixed clamp, that every autonomous spike which fires has a matching generation in the firing phase, that every presynaptic spike produces exactly one delivery per outgoing edge, that timestamp batches execute in nondecreasing order, that state advanced across two consecutive gaps agrees with state advanced across their sum under tolerance, that vector resets commit atomically, and that the same C build, configuration, seed, and inputs produce byte-identical spike output. For adaptive nodes, the event record must also show one adaptation increment per emitted spike and exact continued decay through a clamp.

These are written as separate tests rather than as assertions compiled into the engine for debug builds. Keeping the engine free of test scaffolding matters on the device side, where conditional compilation risks debug and release behavior drifting apart. The cost is that violations are caught only on corpus topologies rather than on any network run in a debug build, which is mitigated by growing the corpus whenever a failure is found.

Because the checks run after a simulation rather than during it, they need enough recorded information to reconstruct what happened, since invariants like refractory violation and pop ordering are not answerable from a spike trace alone. That event log is the recording facility of section 5.4 configured to capture events, not a separate mechanism.

The implemented `audit_causal_trace` profile requires every record kind, every node, and every local state position. It validates eleven invariant families covering the record contract, same-time state continuity, spike/reset and refractory state machines, deposit causality, delivery conservation, prediction generations, runtime accounting, and final-state agreement. The auditor consumes only topology and trace/result facts; analytical propagation and crossing evaluation remain solely in C. Focused adversarial tests mutate one fact in each invariant family and require a specifically classified audit failure, preventing a checker that merely passes known-good runs from being mistaken for useful validation.

### 10.4 The test corpus

Property-based checks run against a curated set of topologies known to sit in a useful firing regime, rather than against uniformly sampled random graphs, which mostly produce networks that never spike or that saturate and therefore exercise nothing. A curated set is reproducible and debuggable, since a failure yields a fixed artifact to inspect rather than a seed to reconstruct. The tradeoff is that it exercises only what was thought to include, which is acceptable because each case added after a real failure becomes a permanent regression guard, and coverage grows toward the failure modes actually encountered rather than uniformly over a space of no interest.

The generated whole-network campaign currently contains 15 curated validation
cases and 37 scaling cases. It exercises deterministic analytical delta,
exponential, alpha, repeated-rate, adaptive, reactive, driven, recurrent,
fan-out/fan-in, mixed-tier, and drive-boundary behavior over the canonical
outgoing-edge CSR index, with sizes through 1,024 nodes. These generators do
not currently cover STEPPED, intrinsic hazard, plasticity, or convolutional
weight sharing; dedicated suites cover those capabilities instead.

The focused suites exercise AdEx and QIF stepped execution, mixed
analytical/stepped networks, all six plasticity rules, modulation and learning
observers, shared weights, codec boundary ordering, persisted traces,
reconstruction, execution-plan equivalence, and schema/reference behavior. The
intrinsic-hazard matrix covers constant and varying hazard, fixed-seed replay
and divergence, interruption invariance, refractory pause, one-shot versus
incremental execution, modal specialization versus expression fallback, mixed
deterministic/stochastic populations, and rejection of unsupported hazard or
filtered-synapse forms.

The generated corpus is produced by deterministic checked-in generators and
each report records the canonical graph hash. Deterministic replays compare
complete returned states, spike sequences, and runtime statistics. A complete
causal trace is audited in memory for every generated case; only its record
count and established invariant names are retained in the JSON report. Separate
versioned reference fixtures cover selected one-shot and incremental behavior.

Trace comparison is tolerance-aware, following section 7.7. Spike order and spike count are compared exactly when trajectories remain corresponding. Spike times use a recorded floating-point tolerance for REACTIVE and CLOSED_FORM nodes and that tolerance plus the solver tolerance where ROOT_FIND nodes are involved. Because recurrent spiking trajectories can diverge after a tiny accepted timing difference, longer cross-platform tests also compare bounded-window causal prefixes and aggregate invariants rather than pretending that every later spike must remain pairwise matchable. The comparator obtains each node's tier and tolerance from the graph and recording header.

### 10.5 Cross-backend numerical equivalence

The current engine has no independent Python evaluator: Python invokes the C core. When a device or other evaluator is added, cross-backend tests compare event order, counts, causal prefixes, states, and spike times under declared arithmetic and solver tolerances, with zero non-convergences as a precondition. Builds that intentionally share a deterministic math implementation, compiler floating-point policy, rounding environment, and arithmetic format may additionally enable bit-identical comparison as a stronger optional check.

Equivalence does not prove correctness, since two implementations of the same wrong propagator can agree. It is a consistency guard protecting deployment and eventual co-design contracts, not a validation of the underlying dynamics.

### 10.6 Cross-simulator comparison

The separate benchmark project compares Lacuna with other CPU-capable SNN
simulators under declared model, timing, API-boundary, thread-count, and
recording contracts. These comparisons are useful for performance and
interoperability evidence, but they are not the primary proof of Lacuna's
correctness. Reference frameworks differ in event precision, threshold
detection, solver choice, delay quantization, setup boundaries, and exposed
native APIs. Accuracy therefore uses convergence or model-specific tolerances
rather than assuming exact trace equality.

Closed-form checks, focused numerical references, causal-trace invariants, and
versioned fixtures remain stronger conformance evidence. Recurrent networks can
amplify tiny accepted timing differences, so external comparisons emphasize
short corresponding prefixes, spike/count error, state error where exposed,
and aggregate behavior rather than forcing long trajectories into false
pairwise agreement.

## 11. Deferred items

These are outside the currently executable capability set and are named so the deferral stays explicit. Exact state propagation is necessary but not sufficient for admission to the analytical engine: every spiking family also needs a complete first-crossing isolator, parameter-regime guards, reset and refractory semantics, and validation at network scale.

### 11.1 Adaptive-model roadmap

The implemented stopping point is the one-mode spike-triggered adaptive current in section 7.8. Further adaptive model work is paused while implementation returns to the core engine. When this thread resumes, capabilities should be added in the following order; later entries must not be enabled merely because a generic matrix exponential can propagate their state.

Repeated real mode. The first extension closes the $\tau_m=\tau_w$ gap in the existing adaptive-current model. Its voltage trajectory has the form

$$
v(\Delta)=v_\infty+(d_0+d_1\Delta)e^{-\Delta/\tau},
$$

so it selects `REPEATED_REAL_MODE` rather than `TWO_REAL_EXP`. The implementation should first provide a certified monotone partition, horizon, and safeguarded solve. A Lambert-$W$ closed form is an optional later optimization only if real-branch selection and all degenerate coefficient cases are proved; it is not required for correctness.

One-component adaptive threshold. Introduce a dynamic threshold state

$$
\dot\theta=-\frac{\theta-\theta_0}{\tau_\theta}, \qquad
\theta\leftarrow\theta+\Delta\theta\ \text{on a spike},
$$

and use the crossing expression $v-\theta$. The threshold state is part of the coupled core even though it does not drive $v$, because it directly determines firing. With distinct stable membrane and threshold rates, the crossing is a two-real-exponential problem and can reuse the mathematical shape proof behind `TWO_REAL_EXP` after the resolver and runtime accept moving thresholds. When the rates are equal, the two transient terms combine and the crossing reduces to the scalar-log family, subject to ordinary reachability and direction guards. This is the recommended next neuron-model capability after core-engine work.

Voltage-coupled subthreshold adaptation. A linearized adaptive model of the form

$$
\tau_m\dot v=-(v-E_L)-R w+R I, \qquad
\tau_w\dot w=a(v-E_L)-w
$$

is a genuine two-dimensional affine core. The first accepted regime should require two distinct stable real eigenvalues, giving a `TWO_REAL_EXP` voltage crossing. Repeated eigenvalues select `REPEATED_REAL_MODE`; complex-conjugate eigenvalues select a future oscillatory capability. Stability, eigenvalue type, reset semantics, and the sign convention for adaptation must be explicit resolver guards rather than inferred from a nominal model name.

Combined and multi-timescale adaptation. Combining an adaptive current with an adaptive threshold, or adding several after-spike currents or threshold components, produces three or more transient modes. These MAT-style or GLIF-style model families remain representable as bounded affine state and vector resets, but require a bounded multi-exponential first-crossing capability. They must not be admitted by repeatedly applying the two-mode solver, because the number and ordering of extrema are properties of the complete crossing function.

Complex stable modes. A real two-state affine system may have complex-conjugate eigenvalues and an exact damped-oscillatory trajectory. Such a trajectory can have many extrema within the prediction horizon, so the alpha-family at-most-two-extrema bound and the adaptive-current at-most-one-extremum bound do not apply. A future `DAMPED_OSCILLATORY` capability must derive a certified horizon and enumerate every relevant extremum from the damping rate, angular frequency, phase, and horizon before invoking the shared safeguarded root primitive.

Target-rate homeostasis. Spike-frequency adaptation is not the same as controlling firing toward a prescribed rate. A true homeostatic model needs at least a filtered spike-rate estimate, a target, and a slow feedback law that changes excitability, drive, threshold, or gain. Some additive affine versions remain exactly propagatable between spike jumps, but they add modes and may introduce an integral or zero eigenmode that falls outside the current stable-negative-rate contract. This is a separate research capability to specify and validate; it is not an implicit extension of the current adaptation variable.

Nonlinear and multiplicative adaptive models. AdEx now executes through the generic stepped path and is its standard acceptance family. Other smooth Izhikevich-type voltage dynamics and multiplicative gain-control equations can use the same path when they satisfy its fixed-threshold and reset contract, but they require their own validation families before being advertised as supported models. Conductance-based synapses become bilinear when the decaying conductance state multiplies membrane voltage through $g(E_{rev}-v)$; automatically merging those edge states into a stepped postsynaptic node remains future work. Current-based synapses retain the cheaper analytical path when eligible.

### 11.2 Other deferred capabilities

Broader STEPPED contracts. The autonomous fixed-threshold vector backend is executable. Remaining extensions include author-visible time dependence, moving or expression-valued thresholds, state-dependent readout resets with a post-reset safety contract, non-spiking stepped nodes, smooth external source coupling, larger state vectors where justified, stateful edge-kernel merging, stochastic differential equations, and differentiated/training evaluators. Each extension must keep Python out of runtime equation evaluation and must state its discontinuity and event-localization rules.

Broader non-adaptive analytical crossing capabilities. The IR can represent additional folded kernels, double-exponential kernels, and larger stable affine systems. Each new accepted family must provide a complete first-crossing isolation contract and certified horizon; unsupported families fail resolution rather than falling through to sampling.

Broader PER_EDGE synapse execution. Additive one-state stable exponential and
unit-area two-state alpha current synapses now execute by augmenting
postsynaptic analytical state with distinct resolved blocks; equal
membrane/synapse rates use the repeated-real capability. Pair, triplet,
modulated, voltage-modulated, and soft-excursion rules are orthogonal to those
current kinetics and lower into the generic learning-event program table.
Remaining work includes broader multi-state kinetic kernels, public
custom/composed learning-rule authoring, learning programs beyond the current
six-trace bound, structural plasticity, and other coupling forms. Conductance
coupling $g(E_{rev}-v)$ is bilinear and requires STEPPED execution or a
separately proven specialized capability.

Generic dependency partitioning. Learning observers are explicitly isolated,
but arbitrary authored observer/core reachability is not inferred. Automatic
partitioning, lazy observer propagation, and proof that an observer cannot
affect evolution, reset, threshold, or hazard remain future resolver work.

Incremental resolution and rebinding. Structural identities, per-plan
deduplication, and exact process-local parser/resolver memoization exist. There
is no persistent on-disk resolver cache or general parameter-mutation API.
Expanding beyond approved drive updates and plastic weights requires
domain/regime guards, exact old-to-new boundary semantics, and cache invalidation
rules; the current cache does not provide runtime rebinding.

Continuous within-gap inputs. The engine accepts spike inputs and piecewise-constant drive updates for both analytical and stepped nodes. Smooth external waveforms require stepped source nodes and an explicit event/step synchronization boundary; they are not approximated by hidden zero-order holds.

The non-spiking node type. A future node with neither threshold nor hazard could
advance state without firing. Current DSL, graph schema, IR, and C descriptors
do not expose a `spiking` flag, so this requires an explicit format and runtime
extension.

Co-design training. A future hardware profile may impose quantization and noise
on a differentiated forward model. It requires new schema/ABI records plus a
hardware-specific numerical-equivalence contract. No current
`precision_noise` layer or differentiated evaluator exists.

Training methods. Surrogate-gradient backpropagation through time binds to the stepped differentiable path. EventProp binds to the analytically tractable event path. Both attach to the target-independent dynamics layer, which is why keeping hardware concerns out of that layer now preserves them as options.

Stochastic transmission failure. Probabilistic loss belongs as an explicit
per-edge or synapse-type mechanism with its own seed, independent of queue
capacity and intrinsic neuron hazard. It is not implemented.

The microcontroller backend. Compiled graph images now provide the compiler-free
handoff from a host to the C runtime. A future hardware-faithful consumer must
still define its memory provisioning, supported native floating-point profile,
timing limits, and cross-backend numerical-equivalence contract. No production
microcontroller build target exists today.

Analysis. Basic spike count, firing rate, interspike intervals, coefficient of
variation, and population-rate helpers are implemented as offline consumers.
Spike-train distances, broader synchrony measures, reservoir/edge-of-chaos
analysis, energy proxies, and connectome analysis remain deferred.

## 12. Known conformance gaps and open decisions

### 12.1 Known conformance gaps

These are mismatches between an intended current contract and an implementation
path. They are recorded here so a permissive path is not mistaken for a
supported capability.

Static convolutional sharing. `Convolution2D` creates and persists valid
`weight_group` metadata, but C lowering currently requires a learning program
for a shared group. Static grouped projections therefore fail compilation even
though immutable sharing should be legal. Until the lowering guard is fixed,
static convolutional sharing is not an executable configuration.

High-level episode reset. The low-level graph/C run supports episode reset with
preserved learned weights and random-stream progress, but
`IncrementalSimulationRun` does not yet expose it and decoder reset is a
separate operation.

STEPPED event-tolerance enforcement. Dense-output extremum and crossing
bisections have a fixed iteration cap, but the current implementation returns
the final midpoint without an explicit post-cap tolerance check. Analytical
root failures carry detailed root context; equivalent structured context is not
yet populated for every STEPPED or integrated-hazard error. Publication-grade
use should keep the focused convergence tests in the release gate until this
contract is hardened.

### 12.2 Open design decisions

These come due only when their deferred feature is implemented.

The remaining intrinsic DSL lexical grammar. Section 4 settles the neuron and synapse statement forms, canonical JSON graph format, versioning, and hash procedures. The JSON container needs no custom topology lexer. What remains is a formal token specification for intrinsic DSL identifiers, numeric literals, comments, and block delimiters; this is isolated from saved topology records and carries less compatibility risk than a second whole-network grammar.

The port-wiring syntax for subnetworks. Deferred with the subnetwork layer itself, per section 4.8, since the first version is Python authoring sugar that flattens before saving and needs no text syntax at all.

## 13. Current baseline and next implementation order

The current compatibility baseline is graph schema 10 for Dale-typed networks
or 11 for mixed-sign networks, network schema 1, trace schema 1, and C ABI 16.
It includes intrinsic threshold and hazard DSL source,
canonical graph/network persistence, capability-derived analytical and STEPPED
execution, DELTA/FOLDED_SHARED/bounded-PER_EDGE current kinetics, default Dale
typing with explicit mixed-sign support,
the five-event generic learning ABI with six standard public rule classes,
shared plastic weights, compiled codecs, explicit-time inspection, causal trace
persistence and reconstruction, low-level episode reset, and equation-derived
arithmetic/dependency specialization. Historical schema and ABI milestones are
not part of this current contract; compatibility is exactly the table at the
top of this document.

Implementation proceeds through four stable layers. The Python DSL/resolver proves
capabilities and emits canonical programs. `ExecutionPlan` expresses only the
operations the graph needs. The primary sparse C scheduler executes those programs and
owns its live simulation, codec, stochastic, and learning state. The high-level
Python API adds construction, persistence, orchestration, recording, analysis,
and visualization without becoming an evaluator. New work must enter through
these same boundaries rather than introduce a model-name handler or another
automatically selected network executor. Existing compatibility runners are
maintenance surfaces, not templates for new capability work.

The next implementation order is:

1. Close the known conformance gaps in section 12.1 and add regression tests for
   each correction.
2. Keep the compatibility table, capability matrix, generated corpus, focused
   suites, and reference fixtures synchronized as one release gate.
3. Expand a deferred capability only with its resolver proof, runtime resource
   contract, persistence/ABI decision, mixed-network tests, and recording
   semantics in the same change.
4. Add a hardware backend only as a consumer of the same resolved semantics;
   define precision, timing, and noise in an explicit target profile instead of
   changing idealized model equations.

This ordering preserves future objectives without claiming they already work.
The extension points are capability descriptors, backend-neutral expression
DAGs, explicit parameter regimes, typed input/output events, and a single
equation-derived execution plan. Reserved enum space or representability in the
IR is never by itself evidence of executable support.
