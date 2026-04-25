# Two-stage CNN checkpoints

Trained PyTorch checkpoints for the MS3 two-stage baseline. Two files belong
here:

- `best_box_regressor.pt` (~130 MB) — best epoch of the ResNet18 box regressor
  (Stage 1 of the two-stage CNN).
- `best_label_classifier.pt` (~130 MB) — best epoch of the ResNet18 follicle
  classifier (Stage 2).

These files are gitignored because they exceed GitHub's 100 MB per-file limit.

## How to obtain them

**Option 1: regenerate by running the notebook.** Open `MS3/ms3.ipynb`, set
`USE_EXISTING_CHECKPOINTS = False` in the config cell, and run all cells. The
training cells write fresh checkpoints to this directory. On an H100 the full
training takes about 10 minutes; on M4 MPS expect 45–90 minutes.

**Option 2: drop in pre-trained checkpoints.** If a teammate already has the
checkpoints, copy them into this directory. The notebook detects them and
skips retraining when `USE_EXISTING_CHECKPOINTS = True`.

Per-epoch metric histories live in `../../history/` and are committed.
