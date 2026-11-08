# Statistical post-processing plan (14 October 2026)

This plan uses existing fitted models and per-molecule predictions. It does not refit models, change molecular curation or recreate partitions. All outputs will be written to `results/review_2026_10_14/statistics/`, retaining the October 12 results.

## Available evidence

The common DILI grid has 160 completed fits: eight representation–model combinations, four regimes and five seeds. The common DILI architecture comparison has 40 fits. Their prediction files, molecular labels and stable grouping identifiers are available. All six endpoints have complete harmonised `dataset.csv.gz` tables with `y` and split manifests with `partition`, `group` and `y`.

Consequently, classification prevalence, regression target mean/SD/range, training-median regression baselines and CYP2D6 AUROC can be calculated without refitting. These composition analyses are separate from the ranking analysis below.

The two Table 2 cells displaying 0.454 ± 0.030 are from different endpoints and metrics: HIA Morgan-frontier MLP AUROC and CYP2D6 descriptor-frontier GNN AUPRC. Their five independent run records and saved predictions are available and will be recalculated before deciding whether the common display is a transcription error.

## Frozen ranking and uncertainty procedure

- Use the eight common-pipeline DILI models under random, scaffold control, descriptor frontier and Morgan frontier. Retain the mean of five per-seed AUROCs as the estimand; do not average predictions or treat seeds as independently sampled test domains.
- Use 2,000 draws and NumPy seed 20261014. Every draw assigns shared weights across all four regimes, models and seeds. The resampling universe is the union of their observed test compounds.
- Compound sensitivity: resample compound IDs separately within their observed binary-label strata, retaining common weights for any shared ID.
- Group sensitivity: resample fixed grouping IDs from the observed union, without label stratification. Apply each group's multiplicity to all its observed compounds. Label-stratifying individual compounds would break clusters, and mixed-label groups prevent ordinary class stratification of whole groups. The indivisible DILI acyclic group is retained as one unit.
- A fit with an empty weighted test set or only one weighted class has undefined AUROC. Record empty and single-class counts explicitly. A model's draw-level mean requires all five fit AUROCs; do not silently average fewer seeds. Report the usable draw count for every interval, rank comparison and multiplicity family.
- Save draw-level model means. Obtain model percentile intervals, Kendall tau-b between every pair of regimes, and top-1 frequencies within each regime. Exact/numerical ties within 1e-12 split top-1 credit evenly.
- Retain all 28 pairwise model contrasts per regime. Report point estimates, percentile intervals, and bootstrap maximum-standardised-centred-deviation simultaneous intervals. Use both the full 28-contrast family within each regime and the full 112-contrast grid across four regimes. These are exploratory conditional uncertainty summaries; no contrasts are selected on the basis of observed ranking. Do not attach causal or deployment-domain interpretations.
- Apply the same joint group draws to All-MLP and MolTriNet, using common curation and groups. Report compound and cluster percentile intervals plus the complete four-contrast architecture family across regimes.

## Reproducibility and interpretation

Store the configuration, source and script SHA-256 hashes, tested fit IDs, resampling units, group draw multiplicities, AUROC draw outputs, valid/invalid counts, top-1 frequencies, Kendall tau-b intervals and pairwise/simultaneous intervals. Independently check point AUROCs against saved run summaries and verify the weighted AUROC implementation against sklearn, including ties.

The analyses remain conditional on fitted models and realised partitions. Group bootstrap can reveal sensitivity to the observed scaffold clusters but cannot turn a deterministic frontier into an independent domain replicate. Bootstrap ranking intervals quantify sensitivity to observed test units, not a null distribution of arbitrary split geometries. Source-table tests with missing original fit records retain their provenance limitation.
