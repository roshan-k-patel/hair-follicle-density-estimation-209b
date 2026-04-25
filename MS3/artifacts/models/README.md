# Trained model checkpoints

This directory holds trained PyTorch checkpoints, organized by architecture.

- `cnn/` — Two-stage CNN baseline (Yafan's MS3 work). Contains the box
  regressor and the follicle classifier. Used by the density head in MS3.
- `detr/` — DETR baseline (placeholder for MS4). Currently empty. Will hold
  the fine-tuned `facebook/detr-resnet-50` checkpoint when the work on the
  `feature/DERT` branch is integrated into `main`.

Checkpoint files (`*.pt`) are not committed to git because they exceed
GitHub's 100 MB per-file limit. See the README in each subdirectory for how
to obtain or regenerate them.
