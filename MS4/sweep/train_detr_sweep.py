"""
W&B Bayesian HPO sweep agent for MS4 DETR.

Runs ONE trial. The sweep agent invokes this script repeatedly with different
hyperparameter values from `wandb.config`. Each trial:

1. Reads hyperparameters from wandb.config
2. Builds the augmentation pipeline using sampled values
3. Trains a fresh DETR for num_train_epochs (default 5)
4. Reports `eval_loss` per epoch via HF Trainer's wandb integration
5. The W&B optimizer uses these metrics to choose the next trial

For local smoke-testing without W&B: set WANDB_MODE=disabled. The script
then uses a fixed default config.

Run:
    WANDB_MODE=disabled python MS4/sweep/train_detr_sweep.py    # smoke test
    wandb agent <SWEEP_ID>                                      # real sweep

Pinned settings (from prior 5-epoch experiments):
    num_queries=100         # 50 collapses the model
    train/val box_scale=1.2/1.2  # mismatch hurts
    sample_weights=None     # WeightedRandomSampler caused class collapse
    batch=2, grad_accum=4   # MPS memory-constrained
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
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_convert, generalized_box_iou

import albumentations as A
import cv2

import wandb
from transformers import (
    DetrConfig,
    DetrForObjectDetection,
    DetrImageProcessor,
    Trainer,
    TrainingArguments,
)


# --------------------------------------------------------------------------
# Paths — walk up from CWD until we find the repo root.
# --------------------------------------------------------------------------
def find_repo_root():
    p = os.path.abspath(os.path.dirname(__file__))
    while p != "/" and not os.path.isdir(os.path.join(p, ".git")):
        p = os.path.dirname(p)
    if not os.path.isdir(os.path.join(p, ".git")):
        raise RuntimeError(f"Could not find repo root (no .git found) starting from {os.path.dirname(__file__)}")
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
    eos_coefficient=0.1,
    alpha_cls=1.0,
    p_vflip=0.2,
    aug_strength=1.0,
    p_copy_paste=0.0,             # not implemented yet; reserved for Group 4
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


# --------------------------------------------------------------------------
# XML parser & box utilities (lifted from the notebook)
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
        # XML uses <class>, parser also accepts <name>
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
        # repair flipped coordinates
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
# Augmentation builder — driven by sweep hyperparameters
# --------------------------------------------------------------------------
def build_train_transform(p_vflip: float, aug_strength: float):
    """Build train_transform with sweep-controlled probabilities.

    aug_strength multiplies the probabilities of all variable transforms (rotate,
    brightness, CLAHE, sharpen, gauss noise) by the same factor.
    p_vflip is searched independently because of anatomical-orientation concern.
    """
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
    """Build a collate fn closure capturing the processor instance."""
    def collate_fn(batch):
        # transformers >=5.6 dropped processor.pad(images_list, return_tensors=...);
        # pad manually (same logic as the notebook).
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
# Class-weighted matched CE add-on (for alpha_cls > 0)
# --------------------------------------------------------------------------
def weighted_matched_class_loss(outputs, labels, class_weights, device):
    logits = outputs.logits
    pred_boxes = outputs.pred_boxes
    batch_losses = []
    for b, target in enumerate(labels):
        tgt_labels = target["class_labels"].to(device)
        tgt_boxes = target["boxes"].to(device)
        if len(tgt_labels) == 0:
            continue
        out_prob = logits[b].softmax(-1)
        out_prob_obj = out_prob[:, :-1]
        out_bbox = pred_boxes[b]
        cost_class = -out_prob_obj[:, tgt_labels]
        cost_bbox = torch.cdist(out_bbox, tgt_boxes, p=1)
        out_bbox_xyxy = box_convert(out_bbox, in_fmt="cxcywh", out_fmt="xyxy")
        tgt_bbox_xyxy = box_convert(tgt_boxes, in_fmt="cxcywh", out_fmt="xyxy")
        cost_giou = -generalized_box_iou(out_bbox_xyxy, tgt_bbox_xyxy)
        C = 1.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
        C = C.detach().cpu().numpy()
        pred_idx, tgt_idx = linear_sum_assignment(C)
        pred_idx = torch.as_tensor(pred_idx, dtype=torch.long, device=device)
        tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long, device=device)
        matched_logits = logits[b, pred_idx, :-1]
        matched_labels = tgt_labels[tgt_idx]
        loss = F.cross_entropy(
            matched_logits, matched_labels,
            weight=class_weights.to(matched_logits.device),
        )
        batch_losses.append(loss)
    if not batch_losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(batch_losses).mean()


# --------------------------------------------------------------------------
# Custom Trainer with the matched-CE add-on (alpha_cls)
# --------------------------------------------------------------------------
class ClassWeightedDetrTrainer(Trainer):
    def __init__(self, *args, class_weights=None, alpha_cls=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.alpha_cls = alpha_cls

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        labels_on_device = [
            {k: v.to(model.device) for k, v in t.items()} for t in labels
        ]
        outputs = model(
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs.get("pixel_mask", None),
            labels=labels_on_device,
        )
        loss = outputs.loss
        if self.alpha_cls > 0:
            extra = weighted_matched_class_loss(
                outputs, labels_on_device, self.class_weights, model.device
            )
            loss = loss + self.alpha_cls * extra
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------
# Main: one trial
# --------------------------------------------------------------------------
def main():
    # Initialize wandb. With WANDB_MODE=disabled, this is a no-op and we
    # use DEFAULT_CONFIG. With a real sweep, wandb.config carries the trial's
    # hyperparameters.
    wandb.init(config=DEFAULT_CONFIG)  # project comes from sweep agent env vars
    cfg = dict(wandb.config)  # actual trial values (or DEFAULT_CONFIG when disabled)

    print("=" * 70)
    print("Trial config:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print("=" * 70)

    # --- Splits ---
    with open(os.path.join(SPLIT_DIR, "train.txt")) as f:
        train_files = [l.strip() for l in f if l.strip()]
    with open(os.path.join(SPLIT_DIR, "val.txt")) as f:
        val_files = [l.strip() for l in f if l.strip()]
    print(f"Train: {len(train_files)} images. Val: {len(val_files)}.")

    # --- Processor + augmentation ---
    processor = DetrImageProcessor.from_pretrained(
        "facebook/detr-resnet-50",
        size={"shortest_edge": 800, "longest_edge": 1333},
    )
    train_transform = build_train_transform(
        p_vflip=cfg["p_vflip"], aug_strength=cfg["aug_strength"]
    )

    # --- Datasets ---
    train_dataset = PascalVOCDataset(
        IMAGE_DIR, ANNOT_DIR, processor, LABEL2ID, train_files,
        transform=train_transform, box_scale=cfg["train_box_scale"],
    )
    val_dataset = PascalVOCDataset(
        IMAGE_DIR, ANNOT_DIR, processor, LABEL2ID, val_files,
        transform=None, box_scale=cfg["val_box_scale"],
    )
    print(f"train_dataset: {len(train_dataset)}; val_dataset: {len(val_dataset)}")

    # --- Class weights (inverse frequency, normalized to mean 1) ---
    class_counts = Counter({0: 8063, 1: 2815, 2: 917, 3: 422})
    cw = torch.tensor(
        [1.0 / class_counts[i] for i in sorted(class_counts.keys())], dtype=torch.float
    )
    class_weights = cw / cw.mean()

    # --- Model config ---
    model_config = DetrConfig.from_pretrained("facebook/detr-resnet-50")
    model_config.num_queries = int(cfg["num_queries"])
    model_config.num_labels = len(CLASS_NAMES)
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID
    model_config.bbox_loss_coefficient = int(cfg["bbox_loss_coefficient"])
    model_config.giou_loss_coefficient = int(cfg["giou_loss_coefficient"])
    model_config.eos_coefficient = float(cfg["eos_coefficient"])

    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        config=model_config,
        ignore_mismatched_sizes=True,
    )

    # --- Training args ---
    training_args = TrainingArguments(
        output_dir=os.path.join(CHECKPOINT_ROOT, "sweep_outputs"),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(cfg["num_train_epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
        warmup_steps=int(cfg["warmup_steps"]),
        lr_scheduler_type="cosine",
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=20,
        remove_unused_columns=False,
        dataloader_num_workers=0,            # macOS DataLoader fork issue
        fp16=False,                          # MPS doesn't like fp16 for DETR
        max_grad_norm=1.0,
        report_to=["wandb"],                 # critical for sweep optimizer
        disable_tqdm=False,
    )

    trainer = ClassWeightedDetrTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_factory(processor),
        processing_class=processor,
        class_weights=class_weights,
        alpha_cls=float(cfg["alpha_cls"]),
    )

    trainer.train()

    # Final eval — guaranteed call so the sweep optimizer sees a final number.
    metrics = trainer.evaluate()
    print(f"Final eval metrics: {metrics}")
    wandb.log({"final_eval_loss": float(metrics.get("eval_loss", float("inf")))})

    wandb.finish()


if __name__ == "__main__":
    main()
