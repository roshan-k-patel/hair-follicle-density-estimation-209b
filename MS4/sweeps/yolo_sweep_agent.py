"""
W&B Bayesian sweep agent for MS4 YOLO11m.

Runs ONE trial. The sweep agent invokes this script repeatedly with different
hyperparameter values from wandb.config. Each trial:

1. Reads hyperparameters from wandb.config
2. Ensures a YOLO-format dataset exists on disk (built once, reused)
3. Trains YOLO11m for num_train_epochs (default 20)
4. Ultralytics' built-in W&B integration auto-logs metrics each epoch:
   metrics/mAP50(B), metrics/mAP50-95(B), metrics/precision(B), metrics/recall(B),
   train/box_loss, train/cls_loss, train/dfl_loss
5. The W&B optimizer reads metrics/mAP50(B) to choose the next trial.

The sweep optimizes mAP50 directly, fixing the loss-vs-mAP decoupling issue
we hit on the DETR sweep.

For local smoke-testing without W&B:
    WANDB_MODE=disabled python MS4/sweep/train_yolo_sweep.py

For the real sweep:
    cd MS4/sweep
    wandb sweep sweep_config_yolo.yaml
    wandb agent --count 25 <SWEEP_ID>
"""
import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import wandb
from ultralytics import YOLO


# --------------------------------------------------------------------------
# Paths -- walk up from CWD to find repo root
# --------------------------------------------------------------------------
def find_repo_root() -> Path:
    p = Path(__file__).resolve().parent
    while p != p.parent and not (p / ".git").is_dir():
        p = p.parent
    if not (p / ".git").is_dir():
        raise RuntimeError(f"Could not find repo root from {Path(__file__).resolve().parent}")
    return p


REPO_ROOT = find_repo_root()
DATA_ROOT = REPO_ROOT / "data"
SPLIT_DIR = DATA_ROOT / "ImageSets" / "Main"
ANNOT_DIR = DATA_ROOT / "Annotations"
IMAGE_DIR = DATA_ROOT / "Images"
YOLO_DATASET_DIR = REPO_ROOT / "MS4" / "yolo_dataset_sweep"  # built once and cached
CHECKPOINT_ROOT = REPO_ROOT / "MS4" / "checkpoints"
CHECKPOINT_ROOT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# Sweep defaults (used when WANDB_MODE=disabled for smoke testing)
# --------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # Searched
    lr0=3e-4,
    weight_decay=5e-4,
    box=9,
    cls=1.2,
    dfl=2.0,
    mosaic=0.1,
    copy_paste=0.0,
    close_mosaic=20,
    # Pinned
    imgsz=1024,
    batch=8,
    epochs=20,
    optimizer="AdamW",
    patience=15,
    seed=42,
)


CLASS_NAMES = ["premium", "single", "undersize", "abnormal"]
LABEL2ID = {n: i for i, n in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------
# VOC -> YOLO format conversion (run once, cached)
# --------------------------------------------------------------------------
def voc_xml_to_yolo_lines(xml_path: Path) -> list[str]:
    """Convert one VOC XML to YOLO-format lines: 'class cx cy w h' normalized."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or obj.findtext("class") or "").strip().lower()
        if name not in LABEL2ID:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = float(bb.findtext("xmin"))
            ymin = float(bb.findtext("ymin"))
            xmax = float(bb.findtext("xmax"))
            ymax = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted([xmin, xmax])
        y1, y2 = sorted([ymin, ymax])
        x1 = max(0.0, min(x1, img_w - 1))
        y1 = max(0.0, min(y1, img_h - 1))
        x2 = max(0.0, min(x2, img_w - 1))
        y2 = max(0.0, min(y2, img_h - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{LABEL2ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def find_image(stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = IMAGE_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_yolo_dataset_if_needed():
    """Convert VOC to YOLO dataset structure. Idempotent -- skips if already built."""
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    if yaml_path.exists() and (YOLO_DATASET_DIR / "labels" / "train").exists():
        print(f"YOLO dataset already built at {YOLO_DATASET_DIR}, skipping rebuild.")
        return yaml_path

    print(f"Building YOLO-format dataset at {YOLO_DATASET_DIR}...")
    (YOLO_DATASET_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "images" / "val").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "labels" / "val").mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        with open(SPLIT_DIR / f"{split}.txt") as f:
            stems = [l.strip() for l in f if l.strip()]
        n_ok = 0
        for stem in stems:
            img = find_image(stem)
            xml = ANNOT_DIR / f"{stem}.xml"
            if img is None or not xml.exists():
                continue
            # Symlink image (faster than copying for ~1300 large images)
            dst_img = YOLO_DATASET_DIR / "images" / split / img.name
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img.resolve())
                except OSError:
                    shutil.copy2(img, dst_img)
            # Write label file
            lines = voc_xml_to_yolo_lines(xml)
            (YOLO_DATASET_DIR / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            n_ok += 1
        print(f"  {split}: {n_ok} image/label pairs")

    # Write data.yaml
    yaml_content = f"""# Auto-generated by train_yolo_sweep.py
path: {YOLO_DATASET_DIR}
train: images/train
val: images/val

nc: {len(CLASS_NAMES)}
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    yaml_path.write_text(yaml_content)
    print(f"Wrote {yaml_path}")
    return yaml_path


# --------------------------------------------------------------------------
# Main: one trial
# --------------------------------------------------------------------------
def main():
    # Argument parsing for direct CLI invocation (sweep agents pass kwargs via wandb.config,
    # but the agent also reads command-line args that wandb sends -- argparse swallows them
    # gracefully and they show up in wandb.config too)
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr0", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--box", type=float)
    parser.add_argument("--cls", type=float)
    parser.add_argument("--dfl", type=float)
    parser.add_argument("--mosaic", type=float)
    parser.add_argument("--copy_paste", type=float)
    parser.add_argument("--close_mosaic", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    args, _ = parser.parse_known_args()
    cli_overrides = {k: v for k, v in vars(args).items() if v is not None}

    # Initialize wandb. Project comes from sweep agent env vars; with
    # WANDB_MODE=disabled this is a no-op and DEFAULT_CONFIG is used.
    wandb.init(config={**DEFAULT_CONFIG, **cli_overrides})
    cfg = dict(wandb.config)

    print("=" * 70)
    print("Trial config:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print("=" * 70, flush=True)

    # Build dataset (once, cached for subsequent trials in same agent)
    yaml_path = build_yolo_dataset_if_needed()

    # Run name -- unique per trial via wandb run id
    run_name = f"yolo_sweep_{wandb.run.id if wandb.run is not None else 'smoke'}"

    # Set output dir under CHECKPOINT_ROOT so trial weights are organized
    project_dir = str(CHECKPOINT_ROOT / "yolo_sweep_outputs")

    # Train
    print(f"\nLoading yolo11m.pt...", flush=True)
    model = YOLO("yolo11m.pt")

    train_kwargs = dict(
        data=str(yaml_path),
        imgsz=int(cfg["imgsz"]),
        epochs=int(cfg["epochs"]),
        batch=int(cfg["batch"]),
        optimizer=str(cfg["optimizer"]),
        lr0=float(cfg["lr0"]),
        weight_decay=float(cfg["weight_decay"]),
        box=float(cfg["box"]),
        cls=float(cfg["cls"]),
        dfl=float(cfg["dfl"]),
        mosaic=float(cfg["mosaic"]),
        copy_paste=float(cfg["copy_paste"]),
        close_mosaic=int(cfg["close_mosaic"]),
        patience=int(cfg["patience"]),
        seed=int(cfg["seed"]),
        # Reproducibility / output
        project=project_dir,
        name=run_name,
        exist_ok=True,
        # Ultralytics' W&B integration auto-fires when wandb is installed + WANDB_PROJECT is set
        # plot=True, save=True are the defaults; explicit for clarity
        save=True,
    )

    print(f"Training {run_name}...", flush=True)
    results = model.train(**train_kwargs)

    # Ultralytics already logs metrics to W&B per epoch via its built-in callback.
    # We add a final summary log so the sweep optimizer always has a clean
    # final number to compare.
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        final_metrics = {}
        for k, v in rd.items():
            if isinstance(v, (int, float)):
                final_metrics[f"final/{k}"] = float(v)
        if final_metrics:
            wandb.log(final_metrics)
            print(f"Final metrics: {final_metrics}", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
