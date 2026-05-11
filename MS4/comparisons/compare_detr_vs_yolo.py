"""
Compare DETR (aug4, weakest) vs YOLO (sweep-winner, best) on val images.

Outputs:
- per_image_detr_vs_yolo.csv
- examples/{both_good, both_bad, yolo_wins}/{stem}_cmp.png
- detr_confusion_matrix.png, yolo_confusion_matrix.png
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.ops import box_iou, nms
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForObjectDetection

ROOT = "/home/ubuntu/hair-follicle-density-estimation-209b"
YOLO_PT = f"{ROOT}/MS4/yolo_runs/tuned_basic/weights/best.pt"
DETR_DIR = "/home/ubuntu/detr_aug4"
VAL_IMG_DIR = "/tmp/yolo_follicle_dataset_baseline/images/val"
VAL_LBL_DIR = "/tmp/yolo_follicle_dataset_baseline/labels/val"
OUT_DIR = f"{ROOT}/MS4/comparisons/detr_vs_yolo"

CLASS_NAMES = ["premium", "single", "undersize", "abnormal"]

os.makedirs(OUT_DIR, exist_ok=True)
for sub in ["both_good", "both_bad", "yolo_wins"]:
    os.makedirs(f"{OUT_DIR}/{sub}", exist_ok=True)


def yolo_to_xyxy(box_norm, W, H):
    cx, cy, bw, bh = box_norm
    return [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]


def load_gt(image_path, label_path):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    boxes, labels = [], []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5: continue
                labels.append(int(float(p[0])))
                boxes.append(yolo_to_xyxy(list(map(float, p[1:])), W, H))
    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64), (W, H)


def predict_yolo(model, img_path, conf=0.25, imgsz=1024, iou=0.7, max_det=1000):
    r = model.predict(img_path, conf=conf, imgsz=imgsz, iou=iou, max_det=max_det, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0,4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(np.int64),
            r.boxes.conf.cpu().numpy().astype(np.float32))


def predict_detr(model, processor, img_path, device, conf_thr=0.25, nms_iou=0.5, max_det=300):
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([[H, W]], device=device)
    results = processor.post_process_object_detection(outputs, threshold=conf_thr, target_sizes=target_sizes)[0]
    boxes = results["boxes"].cpu().numpy().astype(np.float32)
    scores = results["scores"].cpu().numpy().astype(np.float32)
    labels = results["labels"].cpu().numpy().astype(np.int64)
    if len(boxes) == 0:
        return boxes, labels, scores
    # NMS
    keep = nms(torch.tensor(boxes), torch.tensor(scores), nms_iou).numpy()
    if len(keep) > max_det: keep = keep[:max_det]
    return boxes[keep], labels[keep], scores[keep]


def hungarian_metrics(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thr=0.5):
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return {"n_gt": 0, "n_pred": 0, "n_matched": 0, "n_class_correct": 0,
                "mean_iou_matched": 0, "matched_acc": 0, "f1": 0}
    if len(pred_boxes) == 0:
        return {"n_gt": len(gt_boxes), "n_pred": 0, "n_matched": 0, "n_class_correct": 0,
                "mean_iou_matched": 0, "matched_acc": 0, "f1": 0}
    if len(gt_boxes) == 0:
        return {"n_gt": 0, "n_pred": len(pred_boxes), "n_matched": 0, "n_class_correct": 0,
                "mean_iou_matched": 0, "matched_acc": 0, "f1": 0}
    pb = torch.tensor(pred_boxes); gb = torch.tensor(gt_boxes)
    ious = box_iou(pb, gb).numpy()
    p_idx, g_idx = linear_sum_assignment(-ious)
    matches = [(p, g, ious[p, g]) for p, g in zip(p_idx, g_idx) if ious[p, g] >= iou_thr]
    n_matched = len(matches)
    if n_matched == 0:
        return {"n_gt": len(gt_boxes), "n_pred": len(pred_boxes), "n_matched": 0,
                "n_class_correct": 0, "mean_iou_matched": 0, "matched_acc": 0, "f1": 0}
    n_class_correct = sum(1 for p, g, _ in matches if pred_labels[p] == gt_labels[g])
    mean_iou = float(np.mean([m[2] for m in matches]))
    precision = n_class_correct / len(pred_boxes)
    recall = n_class_correct / len(gt_boxes)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"n_gt": len(gt_boxes), "n_pred": len(pred_boxes), "n_matched": n_matched,
            "n_class_correct": n_class_correct, "mean_iou_matched": mean_iou,
            "matched_acc": n_class_correct/n_matched, "f1": f1,
            "_matches": matches, "_pred_labels": pred_labels, "_gt_labels": gt_labels}


def render_side_by_side(img_path, gt_boxes, gt_labels, d_metrics, y_metrics,
                        d_pred, y_pred, out_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = ['#22c55e', '#3b82f6', '#f97316', '#ef4444']

    panels = [
        (axes[0], f"Ground truth ({len(gt_boxes)} objects)", gt_boxes, gt_labels, None),
        (axes[1], f"DETR (aug4 baseline): matched {d_metrics['n_matched']}/{d_metrics['n_gt']}, IoU={d_metrics['mean_iou_matched']:.2f}, F1={d_metrics['f1']:.2f}",
         d_pred[0], d_pred[1], d_pred[2]),
        (axes[2], f"YOLO (tuned, best): matched {y_metrics['n_matched']}/{y_metrics['n_gt']}, IoU={y_metrics['mean_iou_matched']:.2f}, F1={y_metrics['f1']:.2f}",
         y_pred[0], y_pred[1], y_pred[2]),
    ]
    for ax, title, boxes, labels, scores in panels:
        ax.imshow(img); ax.set_title(title, fontsize=11); ax.axis('off')
        for i, b in enumerate(boxes):
            x1, y1, x2, y2 = b
            c = colors[int(labels[i]) % 4]
            ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=c, linewidth=1.5))
            lbl = CLASS_NAMES[int(labels[i])]
            if scores is not None: lbl = f"{lbl} {scores[i]:.2f}"
            ax.text(x1, max(y1 - 3, 8), lbl, color=c, fontsize=7, fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.6, pad=1, edgecolor='none'))
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading YOLO...")
    yolo = YOLO(YOLO_PT)

    print("Loading DETR...")
    processor = AutoImageProcessor.from_pretrained(DETR_DIR)
    detr = AutoModelForObjectDetection.from_pretrained(DETR_DIR).to(device)
    detr.eval()

    # Determine label mapping for DETR (config has id2label)
    detr_id2label = detr.config.id2label  # {0: "premium", ...}
    print(f"DETR id2label: {detr_id2label}")

    val_images = sorted(Path(VAL_IMG_DIR).glob("*.jpg"))
    print(f"Found {len(val_images)} val images")

    rows = []
    cache = {}
    for i, img_path in enumerate(val_images):
        if i % 25 == 0: print(f"  {i}/{len(val_images)}")
        stem = img_path.stem
        lbl_path = os.path.join(VAL_LBL_DIR, stem + ".txt")
        gt_boxes, gt_labels, _ = load_gt(str(img_path), lbl_path)

        yp, yl, ys = predict_yolo(yolo, str(img_path))
        dp, dl, ds = predict_detr(detr, processor, str(img_path), device)

        ym = hungarian_metrics(yp, yl, gt_boxes, gt_labels)
        dm = hungarian_metrics(dp, dl, gt_boxes, gt_labels)

        rows.append({
            "image": img_path.name, "n_gt": ym["n_gt"],
            "d_matched": dm["n_matched"], "d_iou": dm["mean_iou_matched"], "d_f1": dm["f1"],
            "y_matched": ym["n_matched"], "y_iou": ym["mean_iou_matched"], "y_f1": ym["f1"],
        })
        cache[img_path.name] = {"gt_boxes": gt_boxes, "gt_labels": gt_labels,
                                "d_metrics": dm, "y_metrics": ym,
                                "d_pred": (dp, dl, ds), "y_pred": (yp, yl, ys),
                                "img_path": str(img_path)}

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT_DIR}/per_image_detr_vs_yolo.csv", index=False)
    print(f"\nSaved per_image_detr_vs_yolo.csv ({len(df)} images)\n")

    df = df[df["n_gt"] >= 5].copy()
    df["delta"] = df["y_f1"] - df["d_f1"]

    both_good = df[(df["d_f1"] > 0.55) & (df["y_f1"] > 0.65)].sort_values(
        "y_f1", ascending=False).head(3)
    both_bad = df[(df["d_f1"] < 0.4) & (df["y_f1"] < 0.45)].sort_values(
        "d_f1", ascending=True).head(3)
    yolo_wins = df[(df["y_f1"] > 0.6) & (df["delta"] > 0.2)].sort_values(
        "delta", ascending=False).head(3)

    print(f"Both good: {len(both_good)} | Both bad: {len(both_bad)} | YOLO wins: {len(yolo_wins)}\n")

    for category, top_df in [("both_good", both_good), ("both_bad", both_bad), ("yolo_wins", yolo_wins)]:
        print(f"=== {category} ===")
        for _, r in top_df.iterrows():
            d = cache[r["image"]]
            out = f"{OUT_DIR}/{category}/{Path(r['image']).stem}_cmp.png"
            render_side_by_side(d["img_path"], d["gt_boxes"], d["gt_labels"],
                                d["d_metrics"], d["y_metrics"], d["d_pred"], d["y_pred"], out)
            print(f"  {r['image']}: d_f1={r['d_f1']:.2f}, y_f1={r['y_f1']:.2f}, n_gt={r['n_gt']}")

    # Confusion matrices
    print("\n=== Confusion matrices ===")
    for label, cache_key in [("detr_aug4", "d_metrics"), ("yolo_tuned", "y_metrics")]:
        true, pred = [], []
        for _, d in cache.items():
            m = d[cache_key]
            if "_matches" in m:
                for p, g, _ in m["_matches"]:
                    true.append(int(m["_gt_labels"][g]))
                    pred.append(int(m["_pred_labels"][p]))
        if not true: continue
        cm = confusion_matrix(true, pred, labels=list(range(len(CLASS_NAMES))))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASS_NAMES))); ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
        ax.set_title(f"{label}: row-normalized confusion matrix")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                        ha='center', va='center',
                        color='black' if cm_norm[i,j] < 0.5 else 'white', fontsize=9)
        plt.colorbar(im, ax=ax); plt.tight_layout()
        out = f"{OUT_DIR}/{label}_confusion_matrix.png"
        plt.savefig(out, dpi=130, bbox_inches='tight'); plt.close()
        print(f"  Saved {out}")

    print(f"\nDone. Artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
