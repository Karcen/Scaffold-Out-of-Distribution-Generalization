# Allocation controls: frozen protocol

Written before fitting on 14 October 2026. This protocol tests whether frontier
effects persist relative to label-blind whole-group allocations with exactly
the same test and validation molecule counts, and checks the low-score frontier.
It does not tune the allocation or model recipe after examining outcomes.

## Inputs and grouping

Use the six harmonised molecular tables, raw cached Morgan features and cached
group labels from `results/revision/experiments/`. Reuse the nine physicochemical
descriptors from `codes/repro/grid_common.py`; derive those same descriptors from
the unchanged harmonised SMILES when an endpoint lacks a cached matrix. Save
that matrix and its hash in the new output directory. Keep the original
Murcko/descriptor-derived acyclic cells intact. Use the saved descriptor and
Morgan group-score files, with unchanged score definitions and geometry.

## Exact-count random controls

For each endpoint and each geometry, the original high-frontier seed-0 split
at target 70/10/20 supplies the **realised** test and validation molecule counts.
Generate 50 allocations satisfying both counts exactly. All decisions use group
sizes and a seeded random priority only. No labels, prevalence, model output or
chemical coordinates are inspected by the sampler.

Select test groups using an exact bitset subset-sum calculation over a random
group order, stopping at the first prefix that reaches the exact target.
Backtrack with preference for earlier priority groups. Repeat for validation
using the remaining groups. If validation has no feasible exact subset, discard
that label-blind proposal and retry with fresh random priorities, up to 500
attempts. The original partition proves that the joint target counts are
feasible; failure of this sequential sampler is reported as failure to find an
allocation, **not** as proof of infeasibility. Do not substitute an approximate
count or the original frontier partition for a failed random allocation.

This sampler is not uniform over feasible whole-group partitions: random
priorities, first-reachable-prefix selection, size restrictions and conditional
rejection determine its distribution. Report it as a constrained allocation
sensitivity distribution. There is no inferential claim about all possible
group partitions. The base seed is 2026101400; endpoint offsets are multiples of
10000, geometry offsets are multiples of 1000, and replicate offsets are 0--49.

## Reversed frontier

Sort groups from the lowest score upward with the original 1e-9 seeded tie jitter.
Use the original whole-group allocation: fill test to nominal 20%, validation
to nominal 10%, then assign the rest to training. Overshoot is retained and
reported. Use allocation seeds 0--4. These are allocation/tie seeds, not five
independent optimisation fits. All deterministic linear fits are computed once
per distinct realised partition; identical allocations reuse predictions with
an explicit canonical-run reference. This control is not asserted to be exactly
size matched to the high-score frontier.

## Models and same-model frontier references

For every allocation, fit physicochemical and Morgan linear controls. Also fit
both models on each original seed-0 high frontier. Compare controls only with
these same-model, same-recipe references; do not subtract new LR/Ridge results
from historical neural-network scores.

- Classification: LogisticRegression, L2 penalty, C=1, L-BFGS, maximum 2000
  iterations, no class weighting or resampling, deterministic fitting seed 0.
- All six endpoints, including BBB: training-fitted median imputation and
  RobustScaler(10th,90th percentile) with clipping to [-20,20] for the nine
  physicochemical descriptors; Morgan fingerprints remain binary.
- Regression: Ridge, alpha=1, LSQR solver, tolerance1e-8 and the shared helper's
  default iteration limit. Standardise targets with training mean/std only and convert test
  predictions back to original units. Feature rules are as above.

The new common BBB recipe intentionally differs from the older standard-scaled
BBB LR sensitivity control. The older outputs are not overwritten. Reuse
`baselines.transform`, `baselines.estimator` and `baselines.target_transform` so
these controls and the new classical baselines have identical transformations,
estimator settings and regression inverse transformations. Preserve the common
helper's source hash and snapshot. Ridge uses the shared helper's default
iteration limit, with the recorded LSQR tolerance fixed at 1e-8.

The two geometries use identical model recipes within each endpoint and
representation. Validation labels are not used for model selection. Single-class
training partitions are retained as allocation outcomes and recorded as failed
fits; allocation is never retried based on labels. Record convergence warnings
without suppressing them. AUROC for a single-class test is undefined and remains
missing, not replaced with a guessed score.

## Outputs and analysis boundary

Save protocol and code hashes before fitting; retain an exact code snapshot.
For every allocation, save a per-molecule split manifest, allocation metadata,
counts, attempts, seeds and group-score/input hashes. Each fitted model retains
its full configuration, input transformations, deterministic fitting seed,
coefficients, per-molecule test predictions and metric summary. Record aliases
for identical reversed partitions. Outputs reside in
`results/review_2026_10_14/allocation_controls/`, with endpoint run registries and
a SHA-256 file manifest. The comparison distribution measures allocation
variation; it is distinct from compound-bootstrap uncertainty or fitting-seed SD.

No manuscript result or diagnostic conclusion is added before outputs exist and
the allocation integrity and metric recomputation checks pass.
