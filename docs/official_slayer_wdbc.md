# WDBC official SLAYER deployment pilot

This pilot trained a 60→32→2 LIF/delta network with unmodified official
Lava-DL SLAYER, then exported its weights and parameters into ordinary Lacuna.
Both dense weight layers trained. No C runtime changes were needed.

The verified UCI WDBC dataset contained 569 cases and 30 features. Split seed
137 produced 343 training, 113 validation, and 113 test cases. Feature scaling
used only training minima and maxima, clipping held-out values to [0,1]. Each
feature became adjacent complementary rate channels (x,1-x), giving 60 input
neurons. Each presentation used 64 bins with Bernoulli spike probability
`0.02 + 0.43 * channel_value`. Training inputs were resampled each epoch,
while validation and test used separate fixed encoding seeds.

The pilot used model seed 0, 200 full-batch Adam epochs at learning rate 0.01,
and official SpikeRate loss with target rates 0.25 and 0.02. Checkpoints were
ranked every epoch by validation balanced accuracy, then accuracy, lower loss,
and earlier epoch. Epoch 24 was selected. The test set did not participate in
checkpoint selection. This was one fixed pilot, not a parameter search.

## Measured results

Selected validation accuracy was 111/113 (98.23%). Official SLAYER and Lacuna
both classified 109/113 test cases correctly (96.46%). Test balanced accuracy
was 95.72%, malignant sensitivity was 39/42 (92.86%), and benign specificity
was 70/71 (98.59%). The confusion matrix was `[[70,1],[3,39]]`, with rows true
and columns predicted in benign/malignant order.

All 18,721 hidden-layer and 2,191 output-layer test spikes matched. Validation
had one hidden spike displaced from bin 49 in SLAYER to bin 50 in Lacuna, with
no change to any output spike. Thus the test batch passed exact spike
comparison, but the combined validation and test batches did not.

The validation discrepancy was inspected without retraining. For validation
sample 99, hidden channel 27, SLAYER's pre-reset voltage at bin 49 was exactly
1.0 after fixed-point operations. Lacuna's voltage was 0.999994177276676,
just below the shared threshold 1.0. SLAYER fired then, while Lacuna fired at
the next bin. The source state quantum was 0.000003814697265625. This is the
explicitly acknowledged source rounding difference, not a changed Lacuna
threshold or integration method.

The training and validation-selection loop took 6.73 seconds locally on one CPU
thread, excluding export and cross-engine replay. Selected weight displacements
from initialization had L2 norms 1.84198 and 0.43017 for the hidden and output
layers. Lacuna used `compiled_sparse` execution. These single-seed results on
one small split establish a workflow pilot, not clinical validation or a robust
estimate across data splits.

## Artifacts and reproduction

The run is in `artifacts/official-slayer-wdbc/20260912T135317389482Z/`.
It contains `report.json`, `protocol.json`, `official_checkpoint.pt`,
`validation_inputs.pt`, `network.json`, and `network.lcbin`. The protocol stores
every split index, scaler value, and encoding seed. The report stores actual
source, library, script, and output artifact hashes. No result file was
replaced during the discrepancy investigation.

Report SHA-256:
`fe31d16968b4cca2689c92cd9100cfc3d2db4d15ce7827b134a93ae0836ddda8`.
Dataset SHA-256:
`d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a`.
Official source revision:
`d825ac506eb9bb91ea2b945116fb45a40e775451`, clean, with Torch 2.9.0.

After the isolated setup in [official SLAYER deployment](official_slayer_deployment.md),
run from the Lacuna checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD:$PWD/artifacts/lava-dl-official/src"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
python examples/train_official_slayer_wdbc.py \
  --data artifacts/datasets/wdbc.data \
  --epochs 200 --seed 0 --split-seed 137
```

Point `--data` at an existing verified file if it is stored elsewhere. The pilot
used `../../artifacts/datasets/wdbc.data` relative to this isolated worktree.
The script does not download data or overwrite prior run directories. A later
input-validation fix rejects fractional class labels before casting and does
not alter this run's integer labels or predictions. The report retains the
exact script hash used for the measurement.
