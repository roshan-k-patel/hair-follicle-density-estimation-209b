"""
W&B Bayesian HPO sweep agent for MS4 Conditional DETR.

Runs ONE trial. The sweep agent invokes this script repeatedly with different
hyperparameter values from `wandb.config`. Each trial:

1. Reads hyperparameters from wandb.config
2. Builds the augmentation pipeline using sampled values
3. Trains a fresh Conditional DETR for num_train_epochs (default 5)
4. Computes mAP via torchmetrics + per-class NMS at the end of each epoch
   (via MAPLoggerCallback - bypasses HF Trainer's eval path because
   Conditional DETR has a cross-attention shape bug there on transformers 5.6.2)
5. Logs eval/map_50, eval/map, eval/map_75, and per-class mAP to W&B
6. The W&B optimizer reads eval/map_50 to choose the next trial

Critical difference vs train_detr_sweep.py: this targets eval/map_50 (the
deployment metric) instead of eval_loss (the training surrogate). The DETR
sweep's loss-vs-mAP decoupling motivated this fix.

For local smoke-testing without W&B: set WANDB_MODE=disabled. The script
then uses a fixed default config and logs nothing externally.

Run:
    WANDB_MODE=disabled python MS4/sweep/train_conditional_detr_sweep.py
    wandb agent <SWEEP_ID>

Pinned settings (from prior 5-epoch DETR experiments; same architecture family):
    num_queries=100              # 50 collapses the model
    train/val box_scale=1.2/1.2  # mismatch hurts
    batch=2, grad_accum=4        # MPS/GPU memory-constrained
"""
import os
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import batched_nms

import albumentations as A
import cv2

import wandb
from transformers import (
    ConditionalDetrConfig,
    ConditionalDetrForObjectDetection,
    ConditionalDetrImageProcessor,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from torchmetrics.detection import MeanAveragePrecision


# --------------------------------------------------------------------------
# Paths -- walk up from CWD until we find the repo root.
# --------------------------------------------------------------------------
def find_repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != "/" and not os.path.isdir(os.path.join(p, ".git")):
        p = os.path.dirname(p)
    if not os.path.isdir(os.path.join(p, ".git")):
        raise RuntimeError(
            f"Could not find repo root (no .git found) starting from {os.path.dirname(__file__)}"
        )
    return p


REPO_ROOT = find_repo_root()
DATA_ROOT = os.path.join(REPO_ROOT, "data")
CHECKPOINT_ROOT = os.path.join(REPO_ROOT, "MS4", "checkpoints")
SPLIT_DIR = os.path.join(DATA_ROOT, "ImageSets", "Main")
ANNOT_DIR = os.path.join(DATA_ROOT, "Annotations")
IMAGE_DIR = os.path.join(DATA_ROOT, "Images")
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)


# --------------------------------------------------------------------------
# Defaults (used when WANDB_MODE=disabled for smoke testing)
# --------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    learning_rate=1e-5,
    weight_decay=1e-4,
    warmup_steps=500,
    bbox_loss_coefficient=8,
    giou_loss_coefficient=4,
    focal_alpha=0.25,
    cls_loss_coefficient=2,
    p_vflip=0.2,
    aug_strength=1.0,
    # pinned (not in search space)
    num_queries=100,
    train_box_scale=1.2,
    val_box_scale=1.2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
)


CLASS_NAMES = ["premium", "single", "undersize", "abnormal"]
LABEL2ID = {n: i for i, n in enumerate(CLASS_NAMES)}
ID2LABEL = {i: n for i, n in enumerate(CLASS_NAMES)}
PRETRAINED = "microsoft/conditional-detr-resnet-50"


# --------------------------------------------------------------------------
# XML parser & box utilities (lifted from train_detr_sweep.py)
# --------------------------------------------------------------------------
def clip_box_xyxy(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), w - 1))
    y1 = max(0.0, min(float(y1), h - 1))
    x2 = max(0.0, min(float(x2), w - 1))
    y2 = max(0.0, min(float(y2), h - 1))
    return [x1, y1, x2, y2]


def is_valid_xyxy(box):
    x1, y1, x2, y2 = box
    return (x2 > x1) and (y2 > y1)


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def scale_box_xyxy(box, scale, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    return clip_box_xyxy(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], img_w, img_h
    )


def parse_voc_xml(xml_path: str, label2id: Dict[str, int]) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.findtext("filename")
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name") or obj.findtext("class")
        if name is None:
            continue
        name = name.strip().lower()
        if name not in label2id:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = float(bb.findtext("xmin"))
        ymin = float(bb.findtext("ymin"))
        xmax = float(bb.findtext("xmax"))
        ymax = float(bb.findtext("ymax"))
        x1, x2 = sorted([xmin, xmax])
        y1, y2 = sorted([ymin, ymax])
        x1 = max(0.0, min(x1, width - 1))
        y1 = max(0.0, min(y1, height - 1))
        x2 = max(0.0, min(x2, width - 1))
        y2 = max(0.0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append({"name": name, "class_id": label2id[name], "bbox_xyxy": [x1, y1, x2, y2]})
    return {"filename": filename, "width": width, "height": height, "objects": objects}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class PascalVOCDataset(Dataset):
    def __init__(self, image_dir, annot_dir, processor, label2id, split, transform=None, box_scale=1.0):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.processor = processor
        self.label2id = label2id
        self.split = set(split)
        self.transform = transform
        self.box_scale = box_scale
        self.xml_files = sorted([
            f for f in os.listdir(annot_dir)
            if f.endswith(".xml") and os.path.splitext(f)[0] in self.split
        ])

    def __len__(self):
        return len(self.xml_files)

    def __getitem__(self, idx):
        xml_file = self.xml_files[idx]
        parsed = parse_voc_xml(os.path.join(self.annot_dir, xml_file), self.label2id)
        image_path = os.path.join(self.image_dir, parsed["filename"])
        if not os.path.exists(image_path):
            stem = os.path.splitext(xml_file)[0]
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                alt = os.path.join(self.image_dir, stem + ext)
                if os.path.exists(alt):
                    image_path = alt
                    break
        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        boxes_xyxy, class_labels = [], []
        for obj in parsed["objects"]:
            box = clip_box_xyxy(obj["bbox_xyxy"], W, H)
            if not is_valid_xyxy(box):
                continue
            if self.box_scale != 1.0:
                box = scale_box_xyxy(box, self.box_scale, W, H)
            if not is_valid_xyxy(box):
                continue
            boxes_xyxy.append([float(c) for c in box])
            class_labels.append(int(obj["class_id"]))

        image_np = np.array(image)
        if self.transform is not None and len(boxes_xyxy) > 0:
            t = self.transform(image=image_np, bboxes=boxes_xyxy, class_labels=class_labels)
            image_np = t["image"]
            boxes_xyxy = list(t["bboxes"])
            class_labels = list(t["class_labels"])

        H_new, W_new = image_np.shape[:2]
        annotations, ann_id = [], 0
        for box, cls in zip(boxes_xyxy, class_labels):
            box = clip_box_xyxy(box, W_new, H_new)
            if not is_valid_xyxy(box):
                continue
            x, y, bw, bh = xyxy_to_xywh(box)
            if bw <= 1 or bh <= 1:
                continue
            annotations.append({
                "id": ann_id, "image_id": idx, "category_id": int(cls),
                "bbox": [float(x), float(y), float(bw), float(bh)],
                "area": float(bw * bh), "iscrowd": 0,
            })
            ann_id += 1

        target = {"image_id": idx, "annotations": annotations}
        encoding = self.processor(images=image_np, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


# --------------------------------------------------------------------------
# Augmentation builder
# --------------------------------------------------------------------------
def build_train_transform(p_vflip: float, aug_strength: float):
    s = float(np.clip(aug_strength, 0.0, 2.0))
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=float(np.clip(p_vflip, 0.0, 1.0))),
            A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=min(0.4 * s, 1.0)),
            A.RandomBrightnessContrast(
                brightness_limit=0.15, contrast_limit=0.15, p=min(0.4 * s, 1.0)
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=min(0.3 * s, 1.0)),
            A.Sharpen(alpha=(0.1, 0.25), lightness=(0.8, 1.2), p=min(0.25 * s, 1.0)),
            A.GaussNoise(p=min(0.15 * s, 1.0)),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc", label_fields=["class_labels"], min_visibility=0.3
        ),
    )


def collate_fn_factory(processor):
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        max_h = max(pv.shape[-2] for pv in pixel_values)
        max_w = max(pv.shape[-1] for pv in pixel_values)
        padded, masks = [], []
        for pv in pixel_values:
            c, h, w = pv.shape
            out = torch.zeros((c, max_h, max_w), dtype=pv.dtype)
            out[:, :h, :w] = pv
            padded.append(out)
            mask = torch.zeros((max_h, max_w), dtype=torch.long)
            mask[:h, :w] = 1
            masks.append(mask)
        return {
            "pixel_values": torch.stack(padded),
            "pixel_mask": torch.stack(masks),
            "labels": labels,
        }
    return collate_fn


# --------------------------------------------------------------------------
# mAP eval (called from MAPLoggerCallback during training, and once at end)
# --------------------------------------------------------------------------
@torch.no_grad()
def compute_map_torchmetrics(model, val_dataloader, processor, device,
                              score_threshold=0.05, nms_iou_threshold=0.5):
    """COCO-style mAP with per-class NMS. Returns the dict from
    MeanAveragePrecision.compute() with map_50, map, map_75, map_per_class."""
    metric = MeanAveragePrecision(
        box_format="xyxy", iou_type="bbox",
        iou_thresholds=None, class_metrics=True,
    )
    model.eval()
    for batch in val_dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch.get("pixel_mask")
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        target_sizes = torch.stack([t["orig_size"] for t in labels])
        results = processor.post_process_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes,
        )
        preds, tgts = [], []
        for r, t in zip(results, labels):
            if r["boxes"].numel() > 0:
                keep = batched_nms(
                    r["boxes"], r["scores"], r["labels"],
                    iou_threshold=nms_iou_threshold,
                )
                preds.append({
                    "boxes":  r["boxes"][keep].cpu(),
                    "scores": r["scores"][keep].cpu(),
                    "labels": r["labels"][keep].cpu(),
                })
            else:
                preds.append({
                    "boxes":  torch.zeros((0, 4)),
                    "scores": torch.zeros(0),
                    "labels": torch.zeros(0, dtype=torch.long),
                })
            h, w = t["orig_size"]
            cx, cy, bw, bh = t["boxes"].unbind(1)
            gt_xyxy = torch.stack([
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
            ], dim=1)
            tgts.append({
                "boxes":  gt_xyxy.cpu(),
                "labels": t["class_labels"].cpu(),
            })
        metric.update(preds, tgts)
    return metric.compute()


class MAPLoggerCallback(TrainerCallback):
    """End-of-epoch mAP eval + W&B logging.

    Bypasses Trainer.evaluate() (broken on Conditional DETR in transformers
    5.6.2). The sweep optimizer reads eval/map_50 from W&B to guide search.
    """
    def __init__(self, val_dataset, processor, collate_fn, num_workers=0):
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=2, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn,
        )
        self.processor = processor

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        device = next(model.parameters()).device
        results = compute_map_torchmetrics(
            model, self.val_dataloader, self.processor, device,
        )
        metrics = {
            "eval/map_50": results["map_50"].item(),
            "eval/map":    results["map"].item(),
            "eval/map_75": results["map_75"].item(),
        }
        for i, name in enumerate(CLASS_NAMES):
            metrics[f"eval/map_{name}"] = results["map_per_class"][i].item()
        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)
        ep = int(state.epoch) if state.epoch is not None else 0
        print(
            f"  [mAP] epoch={ep:>2d}  map_50={metrics['eval/map_50']:.4f}  "
            f"map={metrics['eval/map']:.4f}  abnormal={metrics['eval/map_abnormal']:.4f}",
            flush=True,
        )
        model.train()


# --------------------------------------------------------------------------
# Main: one trial
# --------------------------------------------------------------------------
def main():
    # Initialize wandb. project comes from sweep agent env vars; with
    # WANDB_MODE=disabled this is a no-op and we use DEFAULT_CONFIG.
    wandb.init(config=DEFAULT_CONFIG)
    cfg = dict(wandb.config)

    print("=" * 70)
    print("Trial config:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print("=" * 70, flush=True)

    # --- Splits ---
    with open(os.path.join(SPLIT_DIR, "train.txt")) as f:
        train_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SPLIT_DIR, "val.txt")) as f:
        val_files = [l.strip() for l in f if l.strip()]
    print(f"Train: {len(train_files)} images. Val: {len(val_files)}.", flush=True)

    # --- Processor + augmentation ---
    processor = ConditionalDetrImageProcessor.from_pretrained(
        PRETRAINED,
        size={"shortest_edge": 800, "longest_edge": 1333},
    )
    train_transform = build_train_transform(
        p_vflip=float(cfg["p_vflip"]), aug_strength=float(cfg["aug_strength"])
    )

    # --- Datasets ---
    train_dataset = PascalVOCDataset(
        IMAGE_DIR, ANNOT_DIR, processor, LABEL2ID, train_files,
        transform=train_transform, box_scale=float(cfg["train_box_scale"]),
    )
    val_dataset = PascalVOCDataset(
        IMAGE_DIR, ANNOT_DIR, processor, LABEL2ID, val_files,
        transform=None, box_scale=float(cfg["val_box_scale"]),
    )
    print(f"train_dataset: {len(train_dataset)}; val_dataset: {len(val_dataset)}", flush=True)

    # --- Model config ---
    # Conditional DETR has focal loss built in for classification. The
    # corresponding hyperparameters are focal_alpha and cls_loss_coefficient.
    # eos_coefficient (vanilla DETR's "no object" weight) does not apply.
    model_config = ConditionalDetrConfig.from_pretrained(PRETRAINED)
    model_config.num_queries = int(cfg["num_queries"])
    model_config.num_labels = len(CLASS_NAMES)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.bbox_loss_coefficient = int(cfg["bbox_loss_coefficient"])
    model_config.giou_loss_coefficient = int(cfg["giou_loss_coefficient"])
    model_config.focal_alpha = float(cfg["focal_alpha"])
    model_config.cls_loss_coefficient = int(cfg["cls_loss_coefficient"])

    # attn_implementation="eager" sidesteps a transformers 5.6.2 bug where the
    # SDPA attention dispatch in ConditionalDetr cross-attention crashes during
    # inference once the model has been touched by the Trainer's training loop
    # (linear shape mismatch on o_proj: 100x512 vs 256x256). Eager attention
    # does the math directly and avoids the buggy dispatch.
    model = ConditionalDetrForObjectDetection.from_pretrained(
        PRETRAINED,
        config=model_config,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )

    # --- Training args ---
    # eval_strategy="no" because of the cross-attention shape bug in HF
    # Trainer's eval path on Conditional DETR (transformers 5.6.2). The
    # MAPLoggerCallback below runs mAP directly via model inference instead.
    training_args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_ROOT, "conditional_sweep_outputs"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_steps=int(cfg["warmup_steps"]),
        lr_scheduler_type="cosine",
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=20,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=False,
        max_grad_norm=1.0,
        report_to=["wandb"],
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_factory(processor),
        processing_class=processor,
        callbacks=[MAPLoggerCallback(
            val_dataset=val_dataset,
            processor=processor,
            collate_fn=collate_fn_factory(processor),
            num_workers=0,
        )],
    )

    trainer.train()

    # Final mAP -- guaranteed log so the sweep sees a final number even if
    # something went wrong with the per-epoch callback.
    val_dataloader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=0, collate_fn=collate_fn_factory(processor),
    )
    device = next(model.parameters()).device
    final_results = compute_map_torchmetrics(
        model, val_dataloader, processor, device,
    )
    final_map_50 = float(final_results["map_50"].item())
    print(f"Final eval/map_50: {final_map_50:.4f}", flush=True)
    wandb.log({"final_eval_map_50": final_map_50})

    wandb.finish()


if __name__ == "__main__":
    main()
