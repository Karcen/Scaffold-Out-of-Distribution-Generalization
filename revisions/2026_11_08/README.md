# ADMET frontier evaluation revision

Code companion for **Model-dependent ADMET performance under descriptor and fingerprint frontier evaluation**. This dated directory extends the earlier scaffold-frontier study without replacing its original files. [中文说明](README_ZH.md).

The analysis asks how descriptor and fingerprint frontiers change ADMET evaluation populations, model performance and model ordering. It covers BBB, HIA, CYP2D6, DILI, plasma-protein binding and solubility. The geometry changes share grouping units and allocation rules. Different geometries can select different molecules and target distributions, so score differences do not isolate a causal geometry effect on a fixed test population. Bootstrap intervals condition on fitted models and realised test groups; fitting seeds are not independent chemical domains.

## Release contents

| Location | Purpose |
| --- | --- |
| `codes/scaffold/src/` | Chemistry, grouping, metrics, objectives, neural and graph primitives |
| `codes/repro/` | Common input preparation, symmetric descriptor/Morgan allocations, benchmark, DILI grid and architecture recipes |
| `codes/review_2026_10_14/` | Classical and Chemprop baselines, allocation controls, composition, neighbour diagnostics, joint uncertainty and sensitivities |
| `codes/figures_2026_11_08/` | The ten revised scientific figure definitions; no model fitting |
| `docs/` | Frozen experiment and uncertainty protocols |
| `results/` | Selected aggregate score/uncertainty tables, environment/configuration records and executed-source snapshots; **not** the full fitted-result archive |
| `data/README.md` | Source access, schemas and missing molecular-level prerequisites |
| `publication_manifest.json` | SHA-256 identities of the published files and documented portability edits |

This release does not contain manuscripts, source-paper archives, raw molecular datasets, per-molecule labels or predictions, model weights, private templates or administrative documents. Some shared modules contain earlier helper functions; teacher/marine-candidate branches are not part of the revised paper's active evidence.

## Environments

The recorded common-pipeline fits use Python 3.14.2, NumPy 2.3.5, pandas 2.3.3, SciPy 1.16.3, scikit-learn 1.7.2, RDKit 2026.03.2 and PyTorch 2.12.0. See the retained environment JSON records before reproducing a particular batch. The earlier `codes/scaffold/requirements-lock.txt` describes that component's recorded core environment; use `requirements-core.txt` for the currently verified analysis and plotting package versions. Historical fitting versions remain in the corresponding batch records.

Chemprop runs use a separate Python environment, Chemprop 2.3.1 and RDKit 2026.03.6. Its full package freeze is retained in `results/review_2026_10_14/chemprop/requirements-lock.txt`. Do not silently replace the common-pipeline RDKit version with the Chemprop dependency version. See `docs/review_2026_10_14/CHEMPROP_PROTOCOL.md`.

From this dated directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-core.txt
```

The figure scripts additionally use an external `nature-figure` QA utility containing `scripts/audit_panel_alignment.py`, `validate_figure.py`, `audit_pdf_text.py` and `audit_figure_collisions.py`. Set `NATURE_FIGURE_SKILL` to that local directory. No private installation path is required or included. Figure exports use editable PDF/SVG and PNG. The library's own licence governs that external dependency; its source is not redistributed here.

## Redraw from the included aggregate results

```sh
export NATURE_FIGURE_SKILL=/path/to/nature-figure
python codes/plot_available_figures.py
```

This redraws seven figures: benchmark comparisons, allocation-reference distributions, DILI ordering, curation, geometry sensitivity and the two gradient views. Outputs go to `work/figure_redesign_2026_11_08/`. Full target-distribution ECDFs, atom-pair support ECDFs and BBB group diagnostics need the molecular/group diagnostic records described in `data/README.md`; those records are not supplied by this code release. The full figure definitions remain available, with explicit prerequisites.

## Rebuild experiments and analyses

First obtain the unchanged public TDC tables and the original split reference from the preceding repository. The helper copies existing public inputs when available, or downloads missing TDC files only with `--download`:

```sh
python codes/prepare_public_inputs.py --download --reference-root ../..
python codes/repro/run_submission_revision.py --mode benchmark
python codes/repro/run_submission_revision.py --mode grid --tasks DILI
python codes/repro/run_submission_revision.py --mode architecture --tasks DILI
python codes/review_2026_10_14/baselines.py
python codes/review_2026_10_14/baselines.py --morgan-linear-only
python codes/review_2026_10_14/allocation_controls.py --workers 2
python codes/review_2026_10_14/sensitivities.py --mode all
```

These are computationally substantial fitting commands; none ran as part of preparing this publication. Preserve the selected environment, input hashes and frozen recipes. Core scripts write split manifests, configurations, stopping histories and per-molecule predictions for their own runs. The original graph-control fits are generated by the preceding repository's `scripts/run_gnn_control.py`; their source remains at the repository root, rather than being represented as common-pipeline MLP fits.

After the required fitting records exist:

```sh
python codes/review_2026_10_14/composition.py
python codes/review_2026_10_14/statistics.py
python codes/review_2026_10_14/cyclic_grid_uncertainty.py
python codes/review_2026_10_14/benchmark_uncertainty.py
python codes/review_2026_10_14/knn_diagnostics_1024.py
python codes/review_2026_10_14/knn_diagnostics.py
python codes/repro/analyze_submission_revision.py --mode distances
```

Use the isolated Chemprop environment for `chemprop_baseline.py`, `chemprop_uncertainty.py` and its validators. The main drivers are the portable fresh-run entrypoints. Retained worker and source snapshots document recorded execution variants and can enforce the earlier source/protocol hashes; do not use a modified publication copy to resume an earlier frozen run. Recorded hashes identify the versions actually executed, rather than declaring that every current file generated every older result.

The included aggregate tables can be replaced by newly generated outputs only after preserving their original identities. Reproducing exact saved scores requires the complete corresponding curated inputs, manifests, predictions and code/configuration versions; the code-only release should not be described as the entire numerical archive.

## Licence and attribution

The repository's existing [MIT License](../../LICENSE), copyright 2026 Jiacheng Zheng, is retained without change. This does not relicense third-party packages or molecular datasets. The [existing data notice](../../DATA_LICENSES.md) describes the preceding source collections. This dated addition uses the public TDC access route and supplies no new copy of raw TDC or CMNPD data. No CMNPD input is needed for the active revised study.
