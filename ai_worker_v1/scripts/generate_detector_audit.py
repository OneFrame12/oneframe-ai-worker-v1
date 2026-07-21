#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from config import VisionConfig
from detectors import RFDETRDetectorAdapter, YOLODetectorAdapter
from engine import BallDetector, Detection


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def build_r2_client():
    endpoint = os.getenv("AI_WORKER_V1_R2_ENDPOINT") or os.getenv("R2_ENDPOINT")
    access_key = os.getenv("AI_WORKER_V1_R2_READ_ACCESS_KEY_ID")
    secret_key = os.getenv("AI_WORKER_V1_R2_READ_SECRET_ACCESS_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "Faltan credenciales R2 read-only: AI_WORKER_V1_R2_ENDPOINT, "
            "AI_WORKER_V1_R2_READ_ACCESS_KEY_ID, AI_WORKER_V1_R2_READ_SECRET_ACCESS_KEY."
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def list_frame_keys(client, bucket: str, prefix: str, limit: int) -> List[str]:
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item.get("Key", "")
            if key.lower().endswith(IMAGE_EXTENSIONS):
                keys.append(key)
                if limit and len(keys) >= limit:
                    return keys
    return keys


def read_frame(client, bucket: str, key: str):
    obj = client.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo decodificar imagen: {key}")
    return frame


def image_hash(frame) -> str:
    digest = hashlib.sha256()
    digest.update(str(frame.shape).encode("utf-8"))
    digest.update(frame.tobytes())
    return digest.hexdigest()


def dedupe_frame_keys(client, bucket: str, keys: List[str], mode: str) -> tuple[List[str], List[Dict[str, str]]]:
    if mode == "none":
        return keys, []

    selected = []
    seen: Dict[str, str] = {}
    duplicates = []
    for key in keys:
        if mode == "basename":
            identity = Path(key).name
        elif mode == "image_hash":
            identity = image_hash(read_frame(client, bucket, key))
        else:
            raise ValueError(f"dedupe mode no soportado: {mode}")

        if identity in seen:
            duplicates.append(
                {
                    "duplicate_key": key,
                    "kept_key": seen[identity],
                    "dedupe_identity": identity,
                }
            )
            continue

        seen[identity] = key
        selected.append(key)
    return selected, duplicates


def resolve_worker_model_path(config: VisionConfig) -> None:
    raw_path = str(config.yolo.model_path or "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return
    for candidate in (SRC_DIR / raw_path, SCRIPT_DIR.parent / raw_path, Path.cwd() / raw_path):
        if candidate.exists():
            config.yolo.model_path = str(candidate)
            return


def safe_name(key: str) -> str:
    path = Path(key)
    stem = f"{path.parent.name}_{path.stem}" if path.parent.name else path.stem
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)


def detection_to_dict(det: Any) -> Dict[str, Any]:
    return {
        "x": round(float(det.x), 3),
        "y": round(float(det.y), 3),
        "w": round(float(det.w), 3),
        "h": round(float(det.h), 3),
        "confidence": round(float(det.confidence), 6),
        "class_id": int(det.class_id),
        "class_name": str(getattr(det, "class_name", "") or ""),
    }


def parse_rfdetr_detections(
    detector: RFDETRDetectorAdapter,
    raw_detections: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    threshold: float,
) -> List[Any]:
    parsed = []
    for raw in raw_detections:
        if float(raw.get("confidence", 0.0)) < threshold:
            continue
        det = detector._prediction_to_detection(raw, frame_w=frame_w, frame_h=frame_h)
        if det is not None:
            parsed.append(det)
    parsed.sort(key=lambda item: item.confidence, reverse=True)
    return parsed


def top_conf(detections: List[Any]) -> float:
    return round(float(detections[0].confidence), 6) if detections else 0.0


def draw_detection(frame, det: Any, label: str, color: tuple, rank: int = 0) -> None:
    x1 = int(round(float(det.x) - float(det.w) / 2.0))
    y1 = int(round(float(det.y) - float(det.h) / 2.0))
    x2 = int(round(float(det.x) + float(det.w) / 2.0))
    y2 = int(round(float(det.y) + float(det.h) / 2.0))
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text = f"{label} {float(det.confidence):.3f}"
    y_text = max(22, y1 - 8 - (rank * 24))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 8, y_text + 4), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(
    frame,
    frame_key: str,
    yolo: List[Any],
    rfdetr_025: List[Any],
    rfdetr_050: List[Any],
):
    overlay = frame.copy()
    if yolo:
        draw_detection(overlay, yolo[0], "YOLO", (0, 220, 255), rank=0)
    if rfdetr_025:
        draw_detection(overlay, rfdetr_025[0], "RFDETR-025", (60, 220, 60), rank=1)
    if rfdetr_050:
        draw_detection(overlay, rfdetr_050[0], "RFDETR-050", (255, 80, 80), rank=2)

    title = frame_key[-110:]
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera overlays comparativos YOLO vs RF-DETR.")
    parser.add_argument("--bucket", default=os.getenv("AI_WORKER_V1_R2_INPUT_BUCKET", "one-frame"))
    parser.add_argument("--prefix", default="training_frames/")
    parser.add_argument("--limit", type=int, default=146)
    parser.add_argument("--output-root", default=str(SCRIPT_DIR.parent / "benchmark_outputs"))
    parser.add_argument("--dedupe", choices=("none", "basename", "image_hash"), default="none")
    args = parser.parse_args()

    config = VisionConfig()
    config.detector_mode = "rfdetr_only"
    config.rfdetr.conf_threshold = 0.25
    resolve_worker_model_path(config)

    client = build_r2_client()
    raw_keys = list_frame_keys(client, args.bucket, args.prefix, args.limit)
    if not raw_keys:
        raise RuntimeError(f"No se encontraron frames en bucket={args.bucket} prefix={args.prefix}")
    keys, duplicates = dedupe_frame_keys(client, args.bucket, raw_keys, args.dedupe)
    if not keys:
        raise RuntimeError(
            f"No quedaron frames despues de dedupe={args.dedupe} "
            f"bucket={args.bucket} prefix={args.prefix}"
        )

    yolo_detector = YOLODetectorAdapter(BallDetector(config), inference_only=True)
    rfdetr_detector = RFDETRDetectorAdapter(detection_factory=Detection, config=config.rfdetr)
    if not yolo_detector.is_available:
        raise RuntimeError("YOLO no disponible para auditoría.")
    if rfdetr_detector.detector_status != "available":
        raise RuntimeError(
            f"RF-DETR no disponible para auditoría: "
            f"{rfdetr_detector.unavailable_reason or rfdetr_detector.load_error}"
        )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = "" if args.dedupe == "none" else f"_dedupe_{args.dedupe}"
    output_dir = Path(args.output_root) / f"audit_{timestamp}{suffix}"
    overlays_dir = output_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    all_detections: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": {
            "bucket": args.bucket,
            "prefix": args.prefix,
            "limit": args.limit,
            "sample_level": "object" if args.dedupe == "none" else "frame",
            "dedupe_mode": args.dedupe,
            "raw_objects_seen": len(raw_keys),
            "unique_frames_after_dedupe": len(keys),
            "duplicates_removed": len(duplicates),
            "yolo_threshold": config.yolo.confidence,
            "rfdetr_thresholds": [0.25, 0.50],
            "rfdetr_sports_ball_class_id": rfdetr_detector.sports_ball_class_id,
            "rfdetr_sports_ball_class_name": rfdetr_detector.class_names.get(
                rfdetr_detector.sports_ball_class_id,
                "",
            ),
            "preprocessing": "cv2 BGR frames are converted to RGB inside RFDETRDetectorAdapter.",
        },
        "duplicates_removed": duplicates,
        "frames": [],
    }

    for index, key in enumerate(keys, start=1):
        frame = read_frame(client, args.bucket, key)
        h, w = frame.shape[:2]
        yolo_detections = yolo_detector.detect_ball(frame)
        yolo_detections.sort(key=lambda det: det.confidence, reverse=True)

        raw_rfdetr = rfdetr_detector.raw_detections(frame, threshold=0.25)
        rfdetr_025 = parse_rfdetr_detections(rfdetr_detector, raw_rfdetr, w, h, threshold=0.25)
        rfdetr_050 = parse_rfdetr_detections(rfdetr_detector, raw_rfdetr, w, h, threshold=0.50)

        overlay = draw_overlay(frame, key, yolo_detections, rfdetr_025, rfdetr_050)
        overlay_name = f"{index:03d}_{safe_name(key)}.jpg"
        cv2.imwrite(str(overlays_dir / overlay_name), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

        csv_rows.append(
            {
                "frame_key": key,
                "yolo_detected": bool(yolo_detections),
                "yolo_conf": top_conf(yolo_detections),
                "yolo_count": len(yolo_detections),
                "rfdetr_025_detected": bool(rfdetr_025),
                "rfdetr_025_conf": top_conf(rfdetr_025),
                "rfdetr_025_count": len(rfdetr_025),
                "rfdetr_050_detected": bool(rfdetr_050),
                "rfdetr_050_conf": top_conf(rfdetr_050),
                "rfdetr_050_count": len(rfdetr_050),
                "manual_label_yolo": "",
                "manual_label_rfdetr_025": "",
                "manual_label_rfdetr_050": "",
                "ball_visible_manual": "",
                "best_detector_manual": "",
                "needs_training_sample": "",
                "review_notes": "",
                "notes": "",
            }
        )
        all_detections["frames"].append(
            {
                "frame_key": key,
                "overlay": f"overlays/{overlay_name}",
                "yolo": [detection_to_dict(det) for det in yolo_detections],
                "rfdetr_025": [detection_to_dict(det) for det in rfdetr_025],
                "rfdetr_050": [detection_to_dict(det) for det in rfdetr_050],
            }
        )

    csv_path = output_dir / "manual_review.csv"
    fieldnames = [
        "frame_key",
        "yolo_detected",
        "yolo_conf",
        "yolo_count",
        "rfdetr_025_detected",
        "rfdetr_025_conf",
        "rfdetr_025_count",
        "rfdetr_050_detected",
        "rfdetr_050_conf",
        "rfdetr_050_count",
        "manual_label_yolo",
        "manual_label_rfdetr_025",
        "manual_label_rfdetr_050",
        "ball_visible_manual",
        "best_detector_manual",
        "needs_training_sample",
        "review_notes",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file_stream:
        writer = csv.DictWriter(file_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    json_path = output_dir / "all_detections.json"
    json_path.write_text(json.dumps(all_detections, indent=2), encoding="utf-8")

    yolo_unique_frames_detected = sum(1 for row in csv_rows if row["yolo_detected"])
    rfdetr_025_unique_frames_detected = sum(1 for row in csv_rows if row["rfdetr_025_detected"])
    rfdetr_050_unique_frames_detected = sum(1 for row in csv_rows if row["rfdetr_050_detected"])
    yolo_only_unique = sum(
        1 for row in csv_rows if row["yolo_detected"] and not row["rfdetr_025_detected"]
    )
    rfdetr_025_only_unique = sum(
        1 for row in csv_rows if row["rfdetr_025_detected"] and not row["yolo_detected"]
    )
    both_unique = sum(
        1 for row in csv_rows if row["yolo_detected"] and row["rfdetr_025_detected"]
    )
    summary = {
        "output_dir": str(output_dir),
        "overlays_dir": str(overlays_dir),
        "overlays": len(list(overlays_dir.glob("*.jpg"))),
        "manual_review_csv": str(csv_path),
        "all_detections_json": str(json_path),
        "sample_level": "object" if args.dedupe == "none" else "frame",
        "dedupe_mode": args.dedupe,
        "raw_objects_seen": len(raw_keys),
        "unique_frames_after_dedupe": len(keys),
        "duplicates_removed": len(duplicates),
        "frames_audited": len(keys),
        "yolo_unique_frames_detected": yolo_unique_frames_detected,
        "rfdetr_025_unique_frames_detected": rfdetr_025_unique_frames_detected,
        "rfdetr_050_unique_frames_detected": rfdetr_050_unique_frames_detected,
        "yolo_only_unique": yolo_only_unique,
        "rfdetr_025_only_unique": rfdetr_025_only_unique,
        "both_unique": both_unique,
    }
    summary_path = output_dir / "summary.json"
    summary["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
