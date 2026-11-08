# Molecular input access and schemas

Raw inputs are the six public Therapeutics Data Commons tables associated with Harvard Dataverse DOI [10.7910/DVN/21LKWG](https://doi.org/10.7910/DVN/21LKWG). The existing repository's `DATA_LICENSES.md` records source licence checks. This addition does not redistribute those raw tables. `codes/prepare_public_inputs.py` records actual downloaded/copied file hashes; a future download is not automatically asserted to match an earlier input.

| Endpoint | File | Dataverse file ID |
| --- | --- | --- |
| BBB | BBB_Martins.tab | 4259566 |
| HIA | HIA_Hou.tab | 4259591 |
| CYP2D6 | CYP2D6_Veith.tab | 4259580 |
| PPBR | PPBR_AZ.tab | 6413140 |
| Solubility | Solubility_AqSolDB.tab | 4259610 |
| DILI | DILI.tab | 4259585 |

Raw TSV fields include `Drug_ID`, `Drug` (SMILES), and `Y`; PPBR also uses `Species` and retains human records. The common pipeline creates `results/revision/experiments/<endpoint>/dataset.csv.gz` with `row_index`, `source_id`, `smiles`, `connectivity_key`, `y`, and `replicate_n`, plus feature arrays and grouping identifiers. It records training/validation/test membership in `splits/<regime>_test<fraction>_seed<seed>.csv.gz`. Neural fit folders retain predictions and histories. This prepared molecular information is not included here.

The included CSV allowlist contains aggregate fit metrics, allocation-reference scores, conditional intervals, top-rank frequencies, composition summaries, sensitivity scores and difficulty summaries. It contains no molecular structures, molecule identifiers or per-molecule labels/predictions. Synthetic run IDs identify model recipes and seeds.

Full target ECDF and atom-pair ECDF figures additionally require `results/review_2026_10_14/composition/independent_support_molecules.csv.gz`, which the composition script generates from completed fits. BBB group diagnostics additionally require the full neighbour diagnostic outputs, including per-group error burdens. These are prerequisites, not implied public files. Refer to the fitting and analysis scripts to regenerate them from allowed source inputs, or obtain the exact archived inputs and outputs from the authors.
