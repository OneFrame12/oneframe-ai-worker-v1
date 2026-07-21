#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


EXPECTED_DATASET_HASH = "0d26f09f6c48733efd65d5401193504235f6530acb05213900a17211a8a8a4ff"
SPLITS = ("train", "valid", "test")
AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_WORKER_ROOT / "src"))

from dataset_hashing import compute_training_payload_hash, sha256_file  # noqa: E402


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_to_rfdetr_dataset(source_dir: Path, out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    summary: dict[str, Any] = {"splits": {}}
    for split in SPLITS:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True)
        coco = read_json(source_dir / "annotations" / f"instances_{split}.json")
        converted = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 0, "name": "ball", "supercategory": "ball"}],
        }
        image_id_by_original_id = {img["id"]: img for img in coco["images"]}
        for img in coco["images"]:
            src = source_dir / "images" / img["file_name"]
            dst = split_dir / Path(img["file_name"]).name
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")
            shutil.copy2(src, dst)
            converted_img = dict(img)
            converted_img["file_name"] = dst.name
            converted["images"].append(converted_img)
        for ann in coco["annotations"]:
            if ann["image_id"] not in image_id_by_original_id:
                raise ValueError(f"Annotation image_id not found: {ann}")
            converted_ann = dict(ann)
            converted_ann["category_id"] = 0
            converted["annotations"].append(converted_ann)
        write_json(split_dir / "_annotations.coco.json", converted)
        summary["splits"][split] = {
            "images": len(converted["images"]),
            "annotations": len(converted["annotations"]),
            "annotation_file": str(split_dir / "_annotations.coco.json"),
        }
    return summary


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def parse_predictions(pred: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    xyxy = getattr(pred, "xyxy", None)
    conf = getattr(pred, "confidence", None)
    cls = getattr(pred, "class_id", None)
    if xyxy is None:
        return rows
    for i, box in enumerate(np.asarray(xyxy).tolist()):
        class_id = int(np.asarray(cls).tolist()[i]) if cls is not None and len(cls) > i else None
        confidence = float(np.asarray(conf).tolist()[i]) if conf is not None and len(conf) > i else None
        rows.append({"bbox_xyxy": [float(v) for v in box], "confidence": confidence, "class_id": class_id})
    rows.sort(key=lambda r: r["confidence"] if r["confidence"] is not None else -1.0, reverse=True)
    return rows


def draw_boxes(image_path: Path, gt_boxes: list[list[float]], predictions: list[dict[str, Any]]) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(img, "GT", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    for idx, pred in enumerate(predictions[:3]):
        x1, y1, x2, y2 = [int(round(v)) for v in pred["bbox_xyxy"]]
        color = (0, 180, 255) if idx == 0 else (0, 120, 255)
        label = f"P{idx + 1} {pred.get('confidence') or 0:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, min(img.shape[0] - 4, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def evaluate_split(model: Any, dataset_dir: Path, rfdetr_dir: Path, split: str, out_dir: Path, threshold: float) -> dict[str, Any]:
    source_coco = read_json(dataset_dir / "annotations" / f"instances_{split}.json")
    anns_by_image: dict[int, list[list[float]]] = {}
    for ann in source_coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(xywh_to_xyxy(ann["bbox"]))
    rows = []
    frames = []
    matched = 0
    false_positive_images = 0
    missed_positive_images = 0
    negative_images_with_prediction = 0
    for img in source_coco["images"]:
        image_path = rfdetr_dir / split / Path(img["file_name"]).name
        pil_img = Image.open(image_path).convert("RGB")
        pred = model.predict(pil_img, threshold=threshold)
        predictions = parse_predictions(pred)
        gt_boxes = anns_by_image.get(int(img["id"]), [])
        best_iou = 0.0
        if predictions and gt_boxes:
            best_iou = max(iou_xyxy(predictions[0]["bbox_xyxy"], gt) for gt in gt_boxes)
        is_match = bool(predictions and gt_boxes and best_iou >= 0.5)
        if is_match:
            matched += 1
        if predictions and not gt_boxes:
            false_positive_images += 1
            negative_images_with_prediction += 1
        if gt_boxes and not is_match:
            missed_positive_images += 1
        rows.append(
            {
                "image_id": img["id"],
                "file_name": img["file_name"],
                "has_gt": bool(gt_boxes),
                "prediction_count": len(predictions),
                "top_confidence": predictions[0]["confidence"] if predictions else None,
                "top_bbox_xyxy": predictions[0]["bbox_xyxy"] if predictions else None,
                "best_iou": best_iou,
                "matched_iou_050": is_match,
            }
        )
        frames.append(draw_boxes(image_path, gt_boxes, predictions))
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"{split}_predictions.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (512, 512))
    for frame in frames:
        writer.write(frame)
    writer.release()
    metrics = {
        "split": split,
        "images": len(source_coco["images"]),
        "positive_images": len(anns_by_image),
        "negative_images": len(source_coco["images"]) - len(anns_by_image),
        "matched_positive_iou_050": matched,
        "missed_positive_images": missed_positive_images,
        "false_positive_images": false_positive_images,
        "negative_images_with_prediction": negative_images_with_prediction,
        "threshold": threshold,
        "video_path": str(video_path),
    }
    write_json(out_dir / "evaluation" / f"{split}_predictions.json", rows)
    write_json(out_dir / "evaluation" / f"{split}_metrics.json", metrics)
    return metrics


def find_checkpoints(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*.pth")):
        rows.append({"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dataset_dir = Path(args.dataset_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    train_output_dir = run_dir / "rfdetr_output"
    rfdetr_dataset_dir = run_dir / "rfdetr_dataset"

    payload_hash = compute_training_payload_hash(dataset_dir)
    dataset_hash = payload_hash["training_payload_hash"]
    expected_v1_path = dataset_dir / "training_payload_hash_v1.txt"
    expected_training_hash = expected_v1_path.read_text(encoding="utf-8").strip() if expected_v1_path.exists() else ""
    if not expected_training_hash or dataset_hash != expected_training_hash:
        raise SystemExit(f"Training payload hash mismatch: {dataset_hash} != {expected_training_hash or 'missing training_payload_hash_v1.txt'}")
    adapter_summary = convert_to_rfdetr_dataset(dataset_dir, rfdetr_dataset_dir)

    import torch
    import rfdetr
    from rfdetr import RFDETRSmall

    environment = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "rfdetr_version": getattr(rfdetr, "__version__", "unknown"),
    }
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; refusing to train")

    model = RFDETRSmall()
    smoke_img = Image.open(next((rfdetr_dataset_dir / "train").glob("*.jpg"))).convert("RGB")
    smoke_pred = model.predict(smoke_img, threshold=args.threshold)
    smoke = {"prediction_count": len(parse_predictions(smoke_pred))}
    write_json(run_dir / "smoke_cuda.json", {"environment": environment, "smoke": smoke})

    train_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    model.train(
        dataset_dir=str(rfdetr_dataset_dir),
        output_dir=str(train_output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        num_workers=2,
        tensorboard=False,
        wandb=False,
        early_stopping=True,
        early_stopping_patience=12,
        run_test=True,
        class_names=["ball"],
    )
    train_finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    metrics = {
        split: evaluate_split(model, dataset_dir, rfdetr_dataset_dir, split, run_dir, args.threshold)
        for split in ("valid", "test")
    }
    checkpoints = find_checkpoints(train_output_dir)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = {
        "phase": "PE-0A4-T1",
        "model": "RF-DETR-S OneFrame Ball Specialist v0",
        "started_at": started_at,
        "train_started_at": train_started_at,
        "train_finished_at": train_finished_at,
        "finished_at": finished_at,
        "dataset_dir": str(dataset_dir),
        "dataset_hash": dataset_hash,
        "legacy_dataset_hash": EXPECTED_DATASET_HASH,
        "expected_training_payload_hash_v1": expected_training_hash,
        "rfdetr_dataset_adapter": adapter_summary,
        "environment": environment,
        "training": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "output_dir": str(train_output_dir),
        },
        "evaluation": metrics,
        "checkpoints": checkpoints,
        "videos": {
            "valid": str(run_dir / "videos" / "valid_predictions.mp4"),
            "test": str(run_dir / "videos" / "test_predictions.mp4"),
        },
    }
    write_json(run_dir / "training_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
