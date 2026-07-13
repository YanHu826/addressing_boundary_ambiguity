BRACE KBS reproducibility artifacts

This folder contains lightweight provenance files for the recorded BRACE paper
run used by the manuscript tables.

Files:
- paper_run_manifest.txt: recorded run tag, source branch/commit, seed, common
  training arguments, batch size, worker count, learning rate, and GAN epoch.
- split_sha256.txt: SHA-256 hashes of the split files used by the recorded run.
  These hashes make the BUSI, PSFHS, HC18, and TN3K partition files auditable.
- case_metrics/: per-case evaluation CSV files exported by the recorded run.
  These files include case identifiers, spacing fields, prediction/ground-truth
  areas, and finite/NaN distance metric values where applicable.

Important interpretation note:
These artifacts document the reported run under the public reference protocols
used in the manuscript. The split hashes make the benchmark partitions auditable:
TN3K has distinct validation and test files, while BUSI, PSFHS, and HC18 follow
the corresponding single-evaluation or image-level reference partitions used by
the cited Shape Prior and BiPCC settings. The manuscript therefore reports the
results as reference-protocol evaluations and provides split hashes plus per-case
metric files to make the protocol scope transparent.
