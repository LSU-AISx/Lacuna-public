# Official SLAYER to Lacuna MNIST pilot

This was a bounded transfer test of one fixed dense LIF/delta configuration,
not an accuracy search or a full-MNIST training run. The official Lava-DL
SLAYER implementation trained both weight layers of a 784→64→10 network.
The resulting static mixed-sign network used ordinary Lacuna execution.

## Protocol

Verified local MNIST files supplied the canonical 60,000 training and 10,000
test images. The pilot selected 10,000 training and 1,000 disjoint validation
images from the canonical training pool, plus 1,000 images from the separate
test pool. Sampling was balanced by digit, without replacement, with split
seed 137. The actual indices and file hashes are in `protocol.json`.

Each row-major 28×28 image used one input neuron per pixel and 32 time bins.
The spike probability was `0.4 * pixel / 255` per bin, with no background drive.
Training images received fresh spike realizations each epoch. Validation and
test spike tensors used fixed independent seeds and were replayed unchanged
in both engines. No test predictions were used for model selection.

Training used model seed 0, ten epochs, minibatches of 128, Adam at learning
rate 0.003, and official SpikeRate loss with true/false target rates 0.2/0.02.
Both layers used the established import-compatible profile: current decay 1,
voltage decay 512/4096, threshold 1, constructor `scale=4096` (internal state
scale 262,144), no bias or normalization,
zero reset/refractory, and no layer delay. Checkpoints were ranked after every
epoch by validation accuracy, then lower validation loss and earlier epoch.

## Measured results

Epoch 9 was selected at 843/1,000 validation accuracy (84.3%). Official SLAYER
and Lacuna each classified 861/1,000 test images correctly (86.1%). Every test
classification agreed between the two engines. Validation classifications also
agreed for all 1,000 images.

Spike trains were not identical. On the test batch, SLAYER emitted 121,397
hidden spikes and 10,514 output spikes. Lacuna emitted 121,400 and 10,512.
There were 11 differing hidden-layer spike bins and 28 differing output-layer
spike bins. Validation had 35 differing hidden bins and 10 differing output
bins. A displaced spike can count as two differing bins, so these are not
counts of distinct displaced spikes. The comparison checks every layer and
does not equate matching classifications with exact spike equivalence.

The known source/target arithmetic difference remains: official SLAYER uses
float32 weighted sums and fixed-point state operations, while Lacuna executes
the corresponding analytical LIF dynamics in binary64. No attempt was made
to tune away spike disagreement using test examples.

The ten-epoch training and validation-selection loop took 12.80 seconds on
one local CPU thread. Export, cross-engine comparisons of both 1,000-image
batches, and binary-image generation took 41.31 seconds. These are local
workflow timings, not a controlled simulator benchmark. Selected weight
displacements from initialization had L2 norms 1.97237 and 3.57551. Lacuna used
the existing `compiled_sparse` path.

This result does not estimate the attainable accuracy of larger models or
full-dataset training. It establishes that this trained network transfers with
matching class predictions on the evaluated batches, despite some spike-level
differences.

## Artifacts and reproducibility

The completed run is
`artifacts/official-slayer-mnist/20260912T140703852778Z/`.
It contains the selected official checkpoint, history, split/encoding protocol,
saved validation/test spikes and labels, separate cross-engine comparison
reports, canonical `network.json`, and deployable `network.lcbin`.

The report SHA-256 is
`1699099ea5c1288016ec00f0f42caf90862efe847873d8486d6f661b0878d630`.
It records source and artifact hashes. The official source was clean at
`d825ac506eb9bb91ea2b945116fb45a40e775451`, using Torch 2.9.0.

An earlier identical training attempt in
`artifacts/official-slayer-mnist/20260912T140535217179Z/` stopped during validation
replay because the helper still used a 4,096-event queue. It did not reach test
classification. That directory is retained and marked incomplete. The replay
helper now sizes buffers from graph and input dimensions. The successful
repeat produced the exact same exported network JSON, with no change to
training settings, selected weights, or C runtime code. A wide-fanout test
covers the buffer correction.

After the isolated setup in [official SLAYER deployment](official_slayer_deployment.md),
run from the Lacuna checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
python examples/train_official_slayer_mnist.py \
  --data-dir artifacts/mnist-data \
  --train-samples 10000 --validation-samples 1000 --test-samples 1000 \
  --hidden 64 --bins 32 --epochs 10 --batch-size 128 \
  --learning-rate 0.003 --seed 0 --split-seed 137
```

The script verifies the existing MNIST cache and does not download data or
overwrite another run directory. This run used the dense-only importer.
The subsequent [convolutional pilot](official_slayer_conv_mnist.md) tests the
spatial importer. Recurrent source models remain unsupported.
