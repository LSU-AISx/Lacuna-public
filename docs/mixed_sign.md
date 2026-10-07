# Mixed-sign neurons

Lacuna keeps Dale typing as the default and supports explicitly mixed-sign
sources. This is a connection-weight policy, independent of the neuron's
equations and execution strategy.

| Source polarity | Stored outgoing weight | Effective coefficient |
| --- | --- | --- |
| `EXCITATORY` | Finite nonnegative magnitude | `weight` |
| `INHIBITORY` | Finite nonnegative magnitude | `-weight` |
| `MIXED` | Finite signed value | `weight` |

Ordinary neurons and populations still default to excitatory. Reservoirs retain
their existing seeded excitatory/inhibitory assignment when polarity is omitted.
An explicit polarity list can mix all three types in one population. Negative
weights on an excitatory or inhibitory source remain errors in Python and C.

```python
from lacuna import LIF, NetworkBuilder, NeuronPolarity

builder = NetworkBuilder("signed-fanout")
source = builder.neuron("source", LIF(), polarity=NeuronPolarity.MIXED)
targets = builder.population("targets", 2, LIF())
builder.connect(source, targets, weight=(4.0, -3.0), delay=1.0)
network = builder.build()
```

The string `"mixed"` is also accepted by the high-level builder. Both targets
receive spikes from the same source neuron. The positive and negative effects
come from their respective edges, without duplicating the source's state.

## Execution

The compiler still derives propagation and crossing methods from equations.
Mixed signs require no new neuron handler, event type or synchronization grid.
Delta deposits and supported exponential and alpha currents apply the same
effective signed amplitude already used for typed inhibitory sources. External
input ports keep their existing meaning and are not implicitly changed.

Compiled graphs remain immutable. Static weights are supplied when constructing
the graph. Shared parameters must retain one source polarity, so mixed sources
may share signed coefficients with other mixed sources, but not with excitatory
or inhibitory sources.

The `NeuronPolarity.sign` property is a stored-weight multiplier, not a test
for whether a neuron is biologically excitatory. It is -1 only for inhibitory
sources and +1 for mixed and excitatory sources. C descriptors use the explicit
codes 0, 1 and 2 for excitatory, inhibitory and mixed respectively.

## Learning boundaries

Existing online pair, triplet and modulatory STDP remain magnitude-based and
are rejected on edges from mixed sources. These rules continue to work on
typed edges elsewhere in the same network. This implementation does not invent
a signed STDP update or decide when an online synapse should change sign.

Core mixed-sign support does not add SLAYER training, sampled-response models,
or an external weight-update API. The optional
[trained-network importer](imported_networks.md) uses mixed sources to preserve
signed static weights from compatible external models without changing the
execution engine.

## Persistence and visualization

Typed graphs continue to serialize with graph schema 10. Graphs containing
mixed neurons use schema 11. The loader still accepts schemas 9 and 10 with
their previous typed semantics, and rejects a mixed value mislabeled as one of
those schemas. The high-level network wrapper remains schema 1.

Native graph images retain their existing layout, image version 1 and C ABI 16.
The loader validates the new polarity value and signed weights. Older runtimes
reject unknown mixed polarity instead of treating it as inhibition. Saved typed
networks keep their prior interpretation.

The visualizer uses a neutral body color for mixed neurons. Each connection
retains a color based on its effective sign and a width based on its absolute
weight. The selected-neuron inspector reports `MIXED` explicitly.
