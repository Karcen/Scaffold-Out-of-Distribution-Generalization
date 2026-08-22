# Figure quality-assurance record

Final QA date: 2026-07-11

## Export contract

All four figures were generated exclusively with Python/matplotlib from recorded
source-data tables. Each has an editable SVG, vector PDF, opaque RGB TIFF at 600 dpi,
and PNG preview at 300 dpi. Final page width is exactly 518.4 pt (7.20 in / 182.9 mm).

| Figure | PDF page size | TIFF pixels | TIFF mode | SVG text nodes |
|---|---:|---:|---:|---:|
| `ood_generalization_curve` | 518.4 x 345.6 pt | 4320 x 2880 | RGB | 65+ |
| `embedding_shift_pca` | 518.4 x 360.0 pt | 4320 x 3000 | RGB | 100+ |
| `teacher_agreement_by_novelty` | 518.4 x 360.0 pt | 4320 x 3000 | RGB | 65 |
| `method_ablation` | 518.4 x 241.2 pt | 4320 x 2010 | RGB | 110+ |

## Mechanical checks

- `pdfinfo` reports one valid page for every PDF.
- `pdftotext` extracts human-readable panel titles, axes, ticks, and annotations.
- Every SVG contains `<text>` nodes; labels were not converted to paths.
- Heatmap cells remain vector geometry in PDF/SVG rather than low-resolution
  embedded raster panels.
- TIFF metadata reports 600 x 600 dpi and RGB mode.
- PNG metadata reports approximately 300 x 300 dpi.
- All source-data paths, transformation script hash, claims, and limitations are
  recorded in `figure_manifest.json`.

## Visual render checks

Each final PDF was rendered with Poppler at 180 dpi and inspected at original render
resolution.

- No clipped panel letters, titles, colorbars, tick labels, legends, or footnotes.
- No overlapping annotations or unreadable error bars.
- Figure 1 uses a zero-reference/error-ratio axis and explicitly labels normalization.
- Figure 2 uses a descriptive PCA only as the hero panel; quantitative view-specific
  heatmaps retain the evidential claim.
- Figure 3 defines seed error bars, adds the BBB compound-bootstrap ribbon, exposes
  teacher/filter coverage, and repeats the teacher-agreement limitation on the page.
- Figure 4 uses zero-centered diverging scales; color does not imply significance.
- Method colors are consistent; no rainbow scale or red/green-only distinction is
  used.

## Repairs made during QA

1. Repaired a Python escape error in the rho math label.
2. Converted RGBA TIFF exports to opaque RGB.
3. Replaced tight auto-cropping with exact 183-mm page width.
4. Added explicit page gutters after detecting edge-clipped colorbar ticks.
5. Moved the Figure 3 panel label away from the vertical rho label.
6. Replaced Figure 1's long vertical colorbar label with a compact rho title.
7. Rebuilt all heatmaps as vector cells and added 95% covariance ellipses to the
   descriptive PCA panel.
8. Expanded Figure 3 to a 2 x 2 coverage-aware audit and renamed Figure 4's fixed
   penalties so they cannot be mistaken for canonical implementations.

Final status: **PASS - zero visible rendering defects**.
