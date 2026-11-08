# Included aggregates and recipe metadata

The CSV allowlist contains aggregate scores and intervals, allocation records, target-composition summaries, sensitivity records and difficulty summaries. `publication_manifest.json` inventories every included file. It excludes molecular structures, molecule identifiers, per-molecule labels and predictions.

Configuration/environment JSON and code/protocol snapshots describe recorded recipes. They do not establish that the complete fitted runs or every historical input is published. Machine-specific environment paths in prose or metadata are replaced by neutral placeholders; the manifest records original and public hashes. Scientific algorithms and fitting settings stay unchanged. Use the main drivers for fresh runs and preserve new hashes; worker snapshots can intentionally enforce recorded identities when resuming archived runs.
