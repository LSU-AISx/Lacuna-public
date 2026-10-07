# Compiled graph deployment images

Lacuna compiled graph images move an already lowered network from a development
host to the C runtime without running the model parser, symbolic resolver, or
graph compiler on the target. They are derived deployment artifacts. The
canonical JSON network remains the editable source of truth.

## Scope

An image contains the complete immutable `lc_compiled_graph` payload. This
includes neuron and edge descriptors, parameters, expression programs,
specialized evaluation plans, delivery indexes, learning programs, plastic-edge
maps, shared-weight metadata, and learning work batches. Loading reconstructs
the same `lc_compiled_graph` type returned by the compiler, so the scheduler has
no image-specific execution path.

The image does not currently contain initial neuronal state, initial timestamps,
named ports, encoder state, or decoder banks. Initial state is provided to
`lc_mixed_run_create`. Codecs remain separately compiled runtime objects. A
future deployment bundle may package these records with the graph image.

## Host API

The C API measures and writes an image without exposing the private graph
layout:

```c
uint64_t image_size = 0;
uint64_t written = 0;
uint8_t *image;

lc_compiled_graph_image_size(graph, &image_size);
image = malloc((size_t)image_size);
lc_compiled_graph_serialize(graph, image, image_size, &written);
```

The Python API exposes the same operation:

```python
with Engine().compile(network) as simulation:
    simulation.save_compiled_graph_image("network.lcg")
```

Low-level prepared graphs also provide `CompiledGraph.to_bytes()` and
`CompiledGraph.save_image()`.

## Target API

The target reads the image into memory and reconstructs an ordinary graph:

```c
lc_compiled_graph *graph = NULL;
lc_mixed_run *run = NULL;

lc_compiled_graph_deserialize(image, image_size, &graph);
lc_mixed_run_create(
    graph, initial_state, state_count, initial_times, node_count, &run
);
lc_mixed_graph_destroy(graph);
```

The run retains the graph, so the caller may release its graph reference after
creating the run. The input image buffer may also be released immediately after
deserialization because the loader owns independent storage.

## Format contract

Images use fixed-width little-endian fields. They are not raw memory dumps and
contain no process addresses, structure padding, or platform `size_t` values.
The existing float64/time64 profile continues to write version 1, ABI 17 images
with exactly the existing binary64 representation and header layout. Reduced
profiles write version 2 images with independent scalar and time widths.
Float32 profiles require ABI 18 and float16 requires ABI 19:

| Native profile | Image version | Numeric identifier | Model scalars | Time fields |
| --- | --- | --- | --- | --- |
| float64/time64 | 1 | 1 | IEEE 754 binary64 | IEEE 754 binary64 |
| float32/time64 | 2 | 2 | IEEE 754 binary32 | IEEE 754 binary64 |
| float32/time32 | 2 | 3 | IEEE 754 binary32 | IEEE 754 binary32 |
| float16/time16 | 2 | 4 | IEEE 754 binary16 | IEEE 754 binary16 |

An image records values already compiled under its profile. Export and import
copy their bits; neither operation quantizes, widens, or re-specializes a graph.
Train, compile, serialize, and load with the same profile. Loading a float64
image into a float32 runtime, or exchanging images between the two float32
profiles, returns `LC_IMAGE_INCOMPATIBLE` before graph allocation. Select the
matching native library on the development host as well as on the target.

The common header is 40 bytes; version 2 appends 16 bytes of profile metadata.
The eight-byte `LCGIMG01` magic remains the container signature in both versions;
the explicit version field, not the magic's trailing digits, selects the schema.

| Byte offset | Field | Width |
| --- | --- | --- |
| 0 | Container magic | 8 bytes |
| 8 | Image version | uint32 |
| 12 | Required native ABI version | uint32 |
| 16 | Numeric profile identifier | uint32 |
| 20 | Header flags, currently zero | uint32 |
| 24 | Total image byte length | uint64 |
| 32 | Payload CRC32 | uint32 |
| 36 | Reserved, must be zero | uint32 |
| 40, version 2 only | Model scalar width in bits | uint32 |
| 44, version 2 only | Time width in bits | uint32 |
| 48, version 2 only | Arithmetic revision | uint32 |
| 52, version 2 only | Reserved, must be zero | uint32 |

CRC32 covers the payload beginning at byte 40 for version 1 or byte 56 for
version 2. Header compatibility and reserved fields are validated separately;
the checksum is corruption detection, not authentication. Version 2 requires
an exact arithmetic-revision match in addition to ABI, profile, and both widths.

Scalar fields follow `lc_real_t`; time fields follow `lc_time_t` in the public
descriptors. For example, an edge's weight and deposit scale use the scalar
width, whereas its delay uses the time width. Expression records contain four
uint32 fields followed by one scalar (24 bytes for float64, 20 for float32,
18 for float16).
Integer offsets and indexes retain their original widths. Every encoded record
size is computed from wire fields, never from native structure size. The strict
float32 image codec contains no binary64 conversion path.

The loader checks the exact encoded length before allocating graph storage. It
then validates expression references, node state and parameter layouts, edge
and delivery indexes, plastic slot maps, observer maps, shared weights,
modulator indexes, and learning batches. Invalid, incompatible, and
checksum-damaged images return distinct status codes.

The compatibility rule requires the image schema, C ABI, and numerical profile
to match the target runtime. Existing version-1 float64 images remain readable
by the float64 runtime. Recompile the canonical network with the intended
profile when targeting a different precision or incompatible ABI; do not relabel
an existing image's header. This conservative rule prevents an image from
silently acquiring changed runtime semantics.

Mixed-sign neurons use polarity value 2 in the existing node descriptor and
signed edge weights in the selected representation. Polarity alone does not
change the image version. The loader validates negative weights against each source's
polarity. Older runtimes that only understand values 0 and 1 reject mixed
images during graph validation even if their image and ABI versions match.
Deploy mixed graphs only to a runtime with mixed-sign support. Existing typed
images retain their previous interpretation.
