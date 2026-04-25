# Trained model checkpoints

This directory holds the trained PyTorch checkpoints for the MS3 two-stage
baseline. The checkpoints themselves are **not committed to git** because they
exceed GitHub's 100 MB per-file limit.

Two files belong here:

- `best_box_regressor.pt` (~130 MB) — best epoch of the ResNet18 box regressor
- `best_label_classifier.pt` (~130 MB) — best epoch of the ResNet18 follicle
  classifier

## How to obtain them

**Option 1: regenerate by running the notebook.** Open `MS3/ms3.ipynb`, set
`USE_EXISTING_CHECKPOINTS = False` in the config cell, and run all cells. The
training cells will write fresh checkpoints to this directory. On an H100 the
full training takes about 10 minutes; on M4 MPS expect 45–90 minutes.

**Option 2: drop in pre-trained checkpoints.** If a teammate already has the
checkpoints from a previous run, copy them into this directory. The notebook
will detect them and skip retraining when `USE_EXISTING_CHECKPOINTS = True`.

The corresponding training-history pickles (which carry the per-epoch metrics)
live in `../history/` and are committed.
