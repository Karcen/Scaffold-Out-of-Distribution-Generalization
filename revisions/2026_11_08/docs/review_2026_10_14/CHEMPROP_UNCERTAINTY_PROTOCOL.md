# Conditional Chemprop uncertainty

Specified before reading the completed Chemprop scores. No fitting, tuning,
checkpoint selection or change to existing benchmark outputs is performed here.

Require the complete frozen grid: six endpoints, scaffold/descriptor/Morgan
allocations and seeds 0–4, giving 90 complete fits. Verify each copied dataset
and split against the original common-pipeline file, configuration and source
snapshots, graph parsing audit, prediction identity/labels/group membership,
saved metrics, history and selected checkpoint hash.

Reuse the exact 2,000 compound and group weight matrices saved by
`benchmark_uncertainty.py` for each endpoint. Their seed is 20261014. The
resampling universe contains the union of test compounds/groups across the
original benchmark fits. Chemprop uses these same manifests, so it introduces
no new resampling unit. Check this explicitly. Group intervals are primary;
class-stratified compound intervals are sensitivity results. Reuse the separate
cyclic-subset weights for BBB and DILI, excluding only acyclic test molecules
while retaining the complete training partitions and original fits.

Calculate mean five-seed AUROC and average precision for classification. For
regression calculate mean five-seed MAE, training-median MAE and per-fit skill
`1-model_MAE/median_MAE`. Each baseline median uses only the complete training
partition. Keep endpoint units separate. An empty weighted test, missing class
or zero baseline denominator is retained as invalid where relevant; average
precision is defined for positive-only draws but requires at least one positive.
Require all five constituent fit scores to be finite for a mean. Record every
invalid category per fit and scheme.

Save conditional 95% percentile intervals, shared-weight frontier-minus-scaffold
differences and all within-regime Chemprop-minus-existing-model contrasts for
AUROC/AP or MAE/skill. The latter are the complete newly added exploratory
contrast family, unadjusted and without hypothesis p-values. Existing benchmark
draws and summaries remain unchanged. All intervals condition on fitted models
and realised evaluation populations; they do not resample model training or
external chemical domains.

Code: `codes/review_2026_10_14/chemprop_uncertainty.py`. Outputs:
`results/review_2026_10_14/chemprop_uncertainty/`. Save the protocol/code hashes,
all input hashes, identity/metric checks, invalid-draw counts, exact score draws
and an output SHA-256 manifest. The model label is `chemprop:dmpnn`.
