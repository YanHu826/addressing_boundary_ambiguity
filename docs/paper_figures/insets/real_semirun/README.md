# Real semi-run image/GT/prediction insets

This directory contains real examples extracted from the completed semi-supervised BRACE paper run. These files are intended as small inset materials for redrawing architecture/module figures.

For each selected sample, the following files are provided:

- `*_image.png`: normalized original ultrasound image.
- `*_gt.png`: ground-truth mask rendered as a white foreground on black background.
- `*_pred.png`: semi-run prediction mask rendered as a white foreground on black background.
- `*_overlay.png`: image overlay with GT-only in green, prediction-only in red, and overlap in yellow.
- `*_crop_*`: 256x256 crop centered on the target, recommended for compact module/architecture insets.

Source run:

`outputs/paper_runs/brace_paper_freshgan5000_infra8_06161206b/semi/results/`

The original absolute data paths are intentionally omitted from the public
repository. The inset filenames encode the dataset and case identifiers used for
the figure crops.
