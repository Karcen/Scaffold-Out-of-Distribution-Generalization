# Submission-revision experiment protocol

Status: specified before any new model fitting. Date: 2026-10-12.

## Scope and frozen choices

Use local harmonised TDC input tables and the existing label definitions. No external data, hyperparameter search, test-driven model selection, paid compute or prospective biological conclusions. Existing results stay intact. New outputs live in `results/revision/experiments/` and scripts in `codes/repro/`.

Seeds: 0, 1, 2, 3, 4 for allocation and model fitting. Geometry SVD seed: 1729, fixed across model seeds. Compound-bootstrap seed: 20261012; 2,000 replicates. Record software versions and source hashes.

## A. Symmetric Morgan frontier and difficulty gradient

Use the benchmark's canonicalisation, connectivity deduplication, label aggregation and human-only PPBR filter. Compute Murcko/acyclic grouping once from the existing descriptor grouping rule; retain exactly these units in both geometries. Keep group medians, median-centre radius, mean distance to ten nearest group centres, stable ordinal ranks, equal score weights, seeded tie resolution and whole-group allocation unchanged.

Descriptor geometry: the existing twelve descriptors, full-unlabelled-table RobustScaler (10th–90th percentiles), clipped to [-20,20]. Morgan geometry: radius-2, 1,024-bit fingerprints reduced to 32 components by TruncatedSVD (random_state=1729), followed by the same robust scaling and clipping rule. Both use the full unlabelled collection to define geometry. The SVD basis is fixed per endpoint and is not fitted on labels. This is a sensitivity analysis of collection-relative support, not a prospective split.

Evaluate BBB, HIA, CYP2D6, PPBR, Solubility and DILI. Target test fractions are 10%, 20%, 30% and 40%; validation stays at 10%, and training receives the remainder. Whole groups can cause deviations. At every fraction include a size-aware scaffold control with the same grouping and target fractions. Save realised counts and train–test distances so increasing held-out size is not presented as an isolated causal effect of distance.

Use the existing benchmark Morgan MLP: raw bits; one 64-unit ReLU layer, dropout 0.1, AdamW learning rate 0.002, weight decay 1e-4, batch at most 1,024, maximum 30 epochs, validation patience 6, gradient clipping 5. Reuse its loss and metric implementation. This gives 360 fits. At 20%, compare rank ordering with physicochemical LR and Morgan LR on DILI and BBB. These LR fits use training-fitted StandardScaler, L2 C=1 and at most 2,000 iterations; record convergence warnings.

## B. DILI common-pipeline grid and architecture

Use the same harmonised DILI order, labels and manifest registry as A. Add stratified random allocation at 70/10/20. At the 20% target, fit LR and a common-protocol MLP on physchem, MACCS, Morgan and combined features under random, scaffold, descriptor-frontier and Morgan-frontier allocation. Fingerprints stay binary; continuous descriptors use training-fitted RobustScaler (10th–90th), clipping [-20,20]. The MLP uses the benchmark recipe from A. LR uses the same input matrix and C=1. This deliberately replaces the historical grid's StandardScaler/sklearn-MLP recipe; it is a new, explicitly named analysis, not a correction that makes the historical values identical.

Fit All-MLP and MolTriNet on these same DILI records and manifests, using the existing architecture-specific optimiser and stopping schedule (AdamW 1e-3, weight decay 1e-4, label smoothing 0.1, batch 64, maximum 100 epochs, patience 12). Apply the same training-fitted descriptor transformation and leave MACCS bits unchanged for both models. Tokenisation uses canonical SMILES. Architecture-specific training remains explicit; common curation and partitions do not isolate parameter count from all other model differences. Save predictions and configuration per fit; do not overwrite the earlier architecture comparison.

## C. Compound uncertainty and diagnostics

For a fixed frontier, resample test-compound IDs within class, jointly across seeds and model predictions. Report mean-seed AUROC/AUPRC and paired differences from the same resamples. For random/scaffold runs with varying membership, use class-stratified resampling over the union of test-compound IDs and pass the resulting common molecule weights into each seed's metric. This estimates uncertainty conditional on fitted models and the realised collection/splits; it does not create independent frontier-domain replicates. Preserve seed SD as a secondary optimisation diagnostic.

Analyse BBB frontier prevalence, model scores and within-stratum/cross-stratum ranking across score quintiles. Compute nearest-training-neighbour Morgan Tanimoto similarity and label agreement. Repeat relevant checks for cyclic-only DILI. Interpret associations as diagnostics; do not claim that they causally identify distribution shift or sparsity. Recompute cyclic-only performance from recoverable predictions and label the protocol used.

Use the existing All-MLP teacher runs to quantify the frontier-minus-scaffold contrast, with the same uncertainty qualification. It is a computational-label negative control, not a biological endpoint or a mechanistic isolation of label provenance.

## Output and integrity contract

Each fit: JSON configuration, seed, split-manifest path/hash, per-molecule test predictions keyed by stable connectivity ID, metrics, realised class counts, training history/stopping epoch when supported, runtime and status. Each endpoint: harmonised dataset, grouping/geometry metadata and manifests. Each stage: aggregate CSVs plus an updated SHA-256 manifest. Record failed or single-class fits explicitly; never replace missing scores with 0.5. Verify no group or connectivity leakage. Check the descriptor 20% manifests against the original benchmark. Figure/claim updates follow successful output validation.

## Feasibility and stop conditions

Run locally on CPU with bounded thread counts. Persist each completed fit so interruption is recoverable. Architecture fits may take longer; keep their completion status separate. Unavailable original run bundles cannot be reconstructed exactly from rounded tables. If any planned analysis cannot finish, identify the missing artefact and list it in TODO without adding a manuscript conclusion.
