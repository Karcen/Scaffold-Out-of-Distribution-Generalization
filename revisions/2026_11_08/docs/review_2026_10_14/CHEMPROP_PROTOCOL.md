# Fixed Chemprop D-MPNN protocol

Frozen before the first model fit on 14 October 2026. This adds a standard directed message-passing baseline to the six-endpoint evaluation. It uses an isolated environment and new outputs, preserving all prior fitted results and the existing reproduction environment.

## Environment and molecular inputs

Use Chemprop 2.3.1, PyTorch 2.12.0, Lightning 2.6.6 and Python 3.14.2 in `<environment>/admet-chemprop-2026-10-14`. The Chemprop dependency `cuik_molmaker_pin` requires RDKit 2026.03.6 in this environment; prior classical/common-pipeline results retain RDKit 2026.03.2. Freeze all installed package versions. Use the conventional RDKit-based `SimpleMoleculeMolGraphFeaturizer`, not cuik-accelerated featurisation. Save the Chemprop wheel identity and graph/source metadata.

Copy the exact six harmonised dataset files and scaffold/descriptor-frontier/Morgan-frontier target-20% split manifests, seeds 0–4, from `results/revision/experiments/`. Use their unchanged molecular rows, connectivity IDs, SMILES, labels and group memberships. Do not recanonicalise, standardise, deduplicate, filter or resplit the data. Record parsing/graph failures with row IDs and stop that fit; never silently drop a molecule. Graph atom/bond inputs use Chemprop's built-in fixed featuriser and no added molecular, atom or bond descriptors. Graphs may be cached before fitting; caching uses no labels.

## Fixed architecture and fitting budget

- Directed `BondMessagePassing`: hidden dimension 300, depth 3, bias=False, undirected=False, ReLU, dropout 0.
- Mean atom aggregation, with batch normalisation disabled.
- Scalar task head: Chemprop predictor `n_layers=2`, meaning two 300-unit hidden FFN layers plus the output layer; ReLU and dropout 0. Binary classification uses its standard sigmoid/BCE head; regression uses its standard linear/MSE head.
- Adam with Chemprop's Noam-like schedule: warmup 2 epochs, initial learning rate 1e-4, peak 1e-3, final 1e-4; no added weight decay or gradient clipping.
- Maximum 50 epochs, batch size 64, fitting/allocation seeds 0–4, CPU accelerator, one device, two intra-op threads, one inter-op thread, dataloader workers 0. `drop_last=False` retains every training molecule. Training shuffles with the fitting seed; validation/test loaders retain manifest order. No class balancing or resampling.
- Validation-only checkpoint selection and early stopping with patience 10, min_delta=0, consistently across the three regimes. Select maximum validation AUROC for BBB/HIA/DILI, maximum validation average precision for CYP2D6, and minimum validation MAE in original units for PPBR/Solubility. A callback calculates the same primary metric implementation as the project. Chemprop's built-in trapezoidal PR area is not substituted for average precision.
- If a classification validation partition has only one class, ranking-based validation selection is unavailable. Use validation BCE as a prespecified fallback, minimising it and recording the reason in configuration. Do not insert AUROC=0.5 or use test labels. Nonfinite primary scores with otherwise valid partitions are logged as errors rather than silently selecting another rule.
- Classification labels remain unchanged. Regression target `StandardScaler` is fitted only on training labels. Its parameters transform training, validation and test dataset targets; model predictions are inverse-transformed before validation MAE selection, final metrics and per-molecule output. The standardised MSE loss and original-unit MAE are recorded separately.

This gives 90 formal fits: six endpoints × three regimes × five seeds. The first is DILI descriptor frontier, seed 0, and serves as the timing pilot. It is a formal retained result, not a disposable tuning experiment. Hyperparameters, validation rule and budget remain fixed after observing that fit; its completed output is reused in the full resumable run.

## Recording and validation

Code: `codes/review_2026_10_14/chemprop_baseline.py`. Output registry: `results/review_2026_10_14/chemprop/`. Each fit records its complete architecture/training configuration, seed, dataset and split copy/hash, code/protocol/environment snapshots, target-transform parameters, model parameter count, graph metadata, per-epoch train loss, validation loss and original-unit primary score, checkpoint selection, elapsed time, warnings/errors, per-molecule test predictions and best/last checkpoints. Predictions retain `row_index`, `connectivity_key`, `smiles`, `group`, `y_true`, and `prediction`; regression outputs remain in the original endpoint units.

Completed fits are skipped only after their input/config/snapshot identities are checked. Interrupted matching fits resume from the saved last checkpoint, while histories identify the continued epoch sequence. A failed fit never contributes a fabricated score. The SHA-256 manifest and aggregate run registry update after each completed fit. Validate row order, labels, group integrity, predictions, reported metrics, code/protocol hashes and checkpoint hashes before adding uncertainty estimates or manuscript text. Statistical intervals remain conditional on fitted models and realised partitions.

Official implementation references checked before use: [Chemprop 2.3.1 on PyPI](https://pypi.org/project/chemprop/2.3.1/), [installation](https://chemprop.readthedocs.io/en/main/installation.html), and [training documentation](https://chemprop.readthedocs.io/en/main/training.html). The versioned wheel was verified against PyPI's SHA-256 checksum before inspecting its model, predictor, loader and scheduler source.
