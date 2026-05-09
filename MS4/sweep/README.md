# MS4 DETR — W&B Bayesian Hyperparameter Sweep

Bayesian optimization over a 9-dimensional hyperparameter space using W&B Sweeps with Hyperband early-stopping. Designed to find the best DETR fine-tune config for the hair-follicle dataset given what we already learned today (`num_queries=100` is critical, box_scale must match, etc.).

## Files

- `train_detr_sweep.py` — the trial script. The W&B agent invokes this once per trial with hyperparameters from `wandb.config`. Self-contained: parses XMLs, builds dataset, builds augmentation from sweep params, trains DETR for 5 epochs, reports `eval_loss` per epoch.
- `sweep_config.yaml` — the Bayesian search space and Hyperband config.

## What's pinned vs searched

**Pinned** (we verified these in earlier experiments — searching them is a waste):
- `num_queries=100`
- train+val `box_scale=1.2`
- batch=2, grad_accum=4
- no WeightedRandomSampler

**Searched** (9 dims, grouped by what they physically control):

| Group | Hyperparameter | Range |
|-------|----------------|-------|
| Optimization | `learning_rate` | log [3e-6, 3e-5] |
| Optimization | `weight_decay` | log [1e-5, 1e-2] |
| Optimization | `warmup_steps` | int [100, 1500] |
| Loss balance | `bbox_loss_coefficient` | [3, 12] |
| Loss balance | `giou_loss_coefficient` | [1, 6] |
| Loss balance | `eos_coefficient` | [0.05, 0.5] |
| Loss balance | `alpha_cls` | [0, 2] |
| Augmentation | `p_vflip` | [0, 0.5] |
| Augmentation | `aug_strength` | [0.5, 1.5] |

## Setup (one-time)

```bash
# 1) Install wandb in the same env you've been using
/path/to/hw3/venv/bin/pip install wandb

# 2) Get a free W&B account at https://wandb.ai (academic accounts available)
# 3) Get your API key from https://wandb.ai/authorize
# 4) Login from the command line — this writes ~/.netrc with your key
wandb login
# When prompted, paste the API key from step 3.
```

## Smoke test (recommended before launching the real sweep)

Runs ONE trial locally with `WANDB_MODE=disabled`, no W&B logging — just verifies the script trains end-to-end without errors. Takes ~25 minutes.

```bash
cd MS4/sweep/
WANDB_MODE=disabled /path/to/hw3/venv/bin/python train_detr_sweep.py
```

Expected output: 5 epochs of training, eval_loss reported each epoch, ends with "Final eval metrics: {...}". Compare to today's Yosef-clone result (eval_loss ≈ 3.29 at epoch 5) to confirm reasonableness.

## Launching the sweep

```bash
cd MS4/sweep/
wandb sweep sweep_config.yaml
# Returns a sweep ID like "your-username/ms4-detr-sweep/abc1d2e3"
# Copy that.

wandb agent your-username/ms4-detr-sweep/abc1d2e3
```

The agent runs trials sequentially, one at a time. Each trial:
- Picks a hyperparameter combination via Bayesian optimization
- Trains for up to 5 epochs (Hyperband may early-stop at epoch 2 if losing)
- Reports `eval_loss` to W&B
- W&B's optimizer uses that to choose the next trial's hyperparameters

For 30 trials on M4 Pro MPS:
- Without Hyperband: ~12.5 hours
- With Hyperband (configured): ~9 hours

## Monitoring

Open `https://wandb.ai/<your-username>/ms4-detr-sweep` in a browser. Live plots:
- Loss curves per trial
- Sweep progress (best-so-far over trials)
- Parallel-coordinates plot (which hyperparameters interact)
- Parameter-importance bar chart (which mattered most)

## After the sweep — final 30-epoch run

1. Open the W&B run with the lowest `eval_loss`. Note its hyperparameters.
2. Edit `MS4/MS4_DETR.ipynb` cell 25 to those values.
3. Set `cfg.num_train_epochs = 30` and remove any sweep `max_steps` overrides.
4. Run the notebook end-to-end → final aug4 model + Hungarian + torchmetrics mAP cells.

## Known gotchas

- **MPS DataLoader fork issues:** the script uses `dataloader_num_workers=0` to avoid macOS-specific fork segfaults. Don't change this on M4.
- **fp16 disabled:** transformers DETR + MPS doesn't play nice with fp16. Stays at fp32.
- **The `MS4/checkpoints/sweep_outputs/` directory** is the trainer's `output_dir`. Already covered by `MS4/checkpoints/` in `.gitignore`.
- **Stopping mid-sweep:** `Ctrl-C` the agent. Sweep is resumable: rerun `wandb agent <ID>` and it continues from where it stopped.

## Reserved for future: copy-paste augmentation

`p_copy_paste` is currently pinned at 0 (no-op). If you implement class-aware copy-paste later (paste cropped abnormal/undersize instances into other training images), unpin it in `sweep_config.yaml` to add it to the search.
