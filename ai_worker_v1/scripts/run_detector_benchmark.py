#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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
DETECTOR_MODES = ("yolo_primary", "rfdetr_primary", "rfdetr_only", "dual_compare", "ensemble")


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
    buffer = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo decodificar imagen: {key}")
    return frame


def safe_token(value: str) -> str:
    raw = str(value or "").strip().strip("/")
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw) or "root"


def resolve_worker_model_path(config: VisionConfig) -> None:
    raw_path = str(config.yolo.model_path or "").strip()
    if not raw_path:
        return

    path = Path(raw_path)
    if path.is_absolute() and path.exists():
        return

    candidates = [
        SRC_DIR / raw_path,
        SCRIPT_DIR.parent / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            config.yolo.model_path = str(candidate)
            return


def detector_status(detector: Any) -> str:
    return str(getattr(detector, "detector_status", "available" if detector.is_available else "unavailable"))


def detector_message(detector: Any) -> str:
    return str(getattr(detector, "load_error", "") or getattr(detector, "unavailable_reason", "") or "")


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_detector: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        detector = row["detector"]
        stats = by_detector.setdefault(
            detector,
            {
                "total_frames": 0,
                "frames_with_ball": 0,
                "ball_detections": 0,
                "confidence_sum": 0.0,
                "latency_ms_sum": 0.0,
                "frames_without_detection": 0,
            },
        )
        stats["total_frames"] += 1
        stats["ball_detections"] += row["detection_count"]
        stats["latency_ms_sum"] += row["latency_ms"]
        if row["detector_status"] == "available" and row["detection_count"] > 0:
            stats["frames_with_ball"] += 1
            stats["confidence_sum"] += row["top_confidence"]
        else:
            stats["frames_without_detection"] += 1

    for stats in by_detector.values():
        frames = max(stats["total_frames"], 1)
        with_ball = max(stats["frames_with_ball"], 1)
        stats["avg_latency_ms"] = round(stats.pop("latency_ms_sum") / frames, 3)
        stats["avg_confidence_when_detected"] = round(stats.pop("confidence_sum") / with_ball, 4)
    return by_detector


def write_reports(
    rows: List[Dict[str, Any]],
    output_dir: Path,
    report_prefix: str,
    metadata: Dict[str, Any],
    detector_statuses: Dict[str, Dict[str, str]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_prefix}.json"
    csv_path = output_dir / f"{report_prefix}.csv"
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
        "detector_statuses": detector_statuses,
        "summary_by_detector": summarize(rows),
        "frames": rows,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file_stream:
        fieldnames = [
            "detector_mode",
            "frame_key",
            "detector",
            "detector_status",
            "detector_error",
            "detection_count",
            "top_confidence",
            "latency_ms",
            "top_bbox",
        ]
        writer = csv.DictWriter(file_stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"json": str(json_path), "csv": str(csv_path)}


def write_json_csv(
    output_dir: Path,
    report_prefix: str,
    payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
    csv_fieldnames: Sequence[str],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_prefix}.json"
    csv_path = output_dir / f"{report_prefix}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file_stream:
        writer = csv.DictWriter(file_stream, fieldnames=list(csv_fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return {"json": str(json_path), "csv": str(csv_path)}


def run_detector(detector, frame) -> Dict[str, Any]:
    status = detector_status(detector)
    if status != "available":
        return {
            "detector_status": status,
            "detector_error": detector_message(detector),
            "detection_count": 0,
            "top_confidence": 0.0,
            "latency_ms": 0.0,
            "top_bbox": [],
        }

    started = time.perf_counter()
    detections = detector.detect_ball(frame)
    latency_ms = (time.perf_counter() - started) * 1000.0
    top = detections[0] if detections else None
    return {
        "detector_status": status,
        "detector_error": detector_message(detector),
        "detection_count": len(detections),
        "top_confidence": round(float(top.confidence), 4) if top else 0.0,
        "latency_ms": round(latency_ms, 3),
        "top_bbox": (
            [round(float(top.x), 2), round(float(top.y), 2), round(float(top.w), 2), round(float(top.h), 2)]
            if top
            else []
        ),
    }


def parse_thresholds(raw: str) -> List[float]:
    thresholds = []
    for item in str(raw or "").split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return sorted(set(thresholds))


def rfdetr_raw_name(detector: RFDETRDetectorAdapter, class_id: int) -> str:
    return str(getattr(detector, "class_names", {}).get(class_id, ""))


def bbox_from_raw(det: Dict[str, Any]) -> List[float]:
    return [
        round(float(det.get("xmin", 0.0)), 2),
        round(float(det.get("ymin", 0.0)), 2),
        round(float(det.get("xmax", 0.0)), 2),
        round(float(det.get("ymax", 0.0)), 2),
    ]


def run_rfdetr_diagnostics(
    detector: RFDETRDetectorAdapter,
    client,
    bucket: str,
    keys: List[str],
    output_dir: Path,
    prefix: str,
    limit: int,
    thresholds: List[float],
    raw_top_n: int,
) -> Dict[str, Any]:
    if detector_status(detector) != "available":
        raise RuntimeError(f"RF-DETR no disponible: {detector_message(detector)}")

    min_threshold = min(thresholds) if thresholds else detector.conf_threshold
    raw_frames = []
    raw_rows = []
    sweep_stats = {
        threshold: {
            "threshold": threshold,
            "total_frames": len(keys),
            "frames_with_ball": 0,
            "ball_detections": 0,
            "frames_without_detection": 0,
            "avg_confidence_when_detected": 0.0,
            "confidence_sum": 0.0,
            "raw_sports_ball_frames": 0,
            "raw_sports_ball_detections": 0,
        }
        for threshold in thresholds
    }
    class_counts: Dict[str, Dict[str, Any]] = {}

    for key in keys:
        frame = read_frame(client, bucket, key)
        h, w = frame.shape[:2]
        started = time.perf_counter()
        raw_detections = detector.raw_detections(frame, threshold=min_threshold)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)

        top = raw_detections[:raw_top_n]
        frame_payload = {
            "frame_key": key,
            "latency_ms": latency_ms,
            "raw_detection_count": len(raw_detections),
            "top_detections": [],
        }
        for det in top:
            class_id = int(det.get("class_id", -1))
            class_name = str(det.get("class_name") or rfdetr_raw_name(detector, class_id))
            confidence = round(float(det.get("confidence", 0.0)), 6)
            bbox = bbox_from_raw(det)
            frame_payload["top_detections"].append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
            raw_rows.append(
                {
                    "frame_key": key,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )
            class_key = f"{class_id}:{class_name}"
            bucket_stats = class_counts.setdefault(
                class_key,
                {"class_id": class_id, "class_name": class_name, "detections": 0, "max_confidence": 0.0},
            )
            bucket_stats["detections"] += 1
            bucket_stats["max_confidence"] = max(bucket_stats["max_confidence"], confidence)
        raw_frames.append(frame_payload)

        for threshold in thresholds:
            raw_above = [det for det in raw_detections if float(det.get("confidence", 0.0)) >= threshold]
            raw_sports_ball = [
                det for det in raw_above if int(det.get("class_id", -1)) == detector.sports_ball_class_id
            ]
            parsed_ball = []
            for det in raw_above:
                parsed = detector._prediction_to_detection(det, frame_w=w, frame_h=h)
                if parsed is not None:
                    parsed_ball.append(parsed)

            stats = sweep_stats[threshold]
            stats["raw_sports_ball_detections"] += len(raw_sports_ball)
            if raw_sports_ball:
                stats["raw_sports_ball_frames"] += 1
            stats["ball_detections"] += len(parsed_ball)
            if parsed_ball:
                stats["frames_with_ball"] += 1
                stats["confidence_sum"] += max(float(det.confidence) for det in parsed_ball)

    sweep_rows = []
    for threshold in thresholds:
        stats = sweep_stats[threshold]
        stats["frames_without_detection"] = stats["total_frames"] - stats["frames_with_ball"]
        denom = max(stats["frames_with_ball"], 1)
        stats["avg_confidence_when_detected"] = round(stats.pop("confidence_sum") / denom, 4)
        sweep_rows.append(stats)

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    metadata = {
        "detector_mode": "rfdetr_only",
        "bucket": bucket,
        "prefix": prefix,
        "limit": limit,
        "frames_loaded": len(keys),
        "scope": "rfdetr_raw_dump_and_ball_threshold_sweep",
        "raw_dump_threshold": min_threshold,
        "raw_top_n": raw_top_n,
        "sports_ball_class_id": detector.sports_ball_class_id,
        "sports_ball_class_name": rfdetr_raw_name(detector, detector.sports_ball_class_id),
        "preprocessing": {
            "frame_read": "cv2.imdecode(..., cv2.IMREAD_COLOR) returns BGR numpy array",
            "adapter_conversion": "BGR numpy array converted to RGB before RFDETRBase.predict",
            "rfdetr_input_contract": "RFDETRBase.predict expects RGB numpy/PIL or normalized torch.Tensor",
            "rfdetr_internal_resize": getattr(getattr(detector.model, "model", None), "resolution", None),
            "rfdetr_internal_normalization": {
                "means": getattr(detector.model, "means", None),
                "stds": getattr(detector.model, "stds", None),
            },
        },
    }

    raw_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
        "detector_statuses": {
            "rfdetr": {
                "detector_status": detector_status(detector),
                "message": detector_message(detector),
            }
        },
        "class_counts_top_detections": sorted(
            class_counts.values(),
            key=lambda item: item["detections"],
            reverse=True,
        ),
        "frames": raw_frames,
    }
    raw_paths = write_json_csv(
        output_dir,
        f"{timestamp}_rfdetr_raw_dump_{safe_token(prefix)}_limit{int(limit)}_threshold{min_threshold}",
        raw_payload,
        raw_rows,
        ["frame_key", "class_id", "class_name", "confidence", "bbox"],
    )

    sweep_payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
        "detector_statuses": raw_payload["detector_statuses"],
        "threshold_sweep": sweep_rows,
    }
    sweep_paths = write_json_csv(
        output_dir,
        f"{timestamp}_rfdetr_threshold_sweep_{safe_token(prefix)}_limit{int(limit)}",
        sweep_payload,
        sweep_rows,
        [
            "threshold",
            "total_frames",
            "frames_with_ball",
            "ball_detections",
            "frames_without_detection",
            "avg_confidence_when_detected",
            "raw_sports_ball_frames",
            "raw_sports_ball_detections",
        ],
    )

    return {
        "raw_dump": raw_paths,
        "threshold_sweep": sweep_paths,
        "metadata": metadata,
        "threshold_sweep_summary": sweep_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark offline de detectores ai_worker_v1.")
    parser.add_argument("--bucket", default=os.getenv("AI_WORKER_V1_R2_INPUT_BUCKET", "one-frame"))
    parser.add_argument("--prefix", default="training_frames/")
    parser.add_argument("--limit", type=int, default=146)
    parser.add_argument("--detector-mode", choices=DETECTOR_MODES, default="yolo_primary")
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR.parent / "benchmark_outputs"))
    parser.add_argument("--rfdetr-raw-dump", action="store_true")
    parser.add_argument("--rfdetr-threshold-sweep", default="")
    parser.add_argument("--raw-top-n", type=int, default=20)
    args = parser.parse_args()

    config = VisionConfig()
    config.detector_mode = args.detector_mode
    resolve_worker_model_path(config)
    client = build_r2_client()
    keys = list_frame_keys(client, args.bucket, args.prefix, args.limit)
    if not keys:
        raise RuntimeError(f"No se encontraron frames en bucket={args.bucket} prefix={args.prefix}")

    detectors = []
    detector_statuses: Dict[str, Dict[str, str]] = {}

    if args.detector_mode != "rfdetr_only":
        try:
            yolo_detector = YOLODetectorAdapter(BallDetector(config), inference_only=True)
            detectors.append(yolo_detector)
            detector_statuses[yolo_detector.name] = {
                "detector_status": detector_status(yolo_detector),
                "message": "",
            }
        except Exception as exc:
            detector_statuses["yolo"] = {
                "detector_status": "error",
                "message": str(exc),
            }

    try:
        rfdetr_detector = RFDETRDetectorAdapter(detection_factory=Detection, config=config.rfdetr)
        detectors.append(rfdetr_detector)
        detector_statuses[rfdetr_detector.name] = {
            "detector_status": detector_status(rfdetr_detector),
            "message": detector_message(rfdetr_detector),
        }
    except Exception as exc:
        detector_statuses["rfdetr"] = {
            "detector_status": "error",
            "message": str(exc),
        }

    if not detectors:
        raise RuntimeError("No hay detectores instanciados para benchmark.")

    thresholds = parse_thresholds(args.rfdetr_threshold_sweep)
    if args.rfdetr_raw_dump or thresholds:
        rfdetr_detector = next((detector for detector in detectors if detector.name == "rfdetr"), None)
        if rfdetr_detector is None:
            raise RuntimeError("RF-DETR no fue instanciado para diagnóstico.")
        result = run_rfdetr_diagnostics(
            rfdetr_detector,
            client,
            args.bucket,
            keys,
            Path(args.output_dir),
            args.prefix,
            args.limit,
            thresholds or [rfdetr_detector.conf_threshold],
            args.raw_top_n,
        )
        print(json.dumps(result, indent=2))
        return 0

    rows = []
    for key in keys:
        frame = read_frame(client, args.bucket, key)
        for detector in detectors:
            result = run_detector(detector, frame)
            rows.append(
                {
                    "detector_mode": args.detector_mode,
                    "frame_key": key,
                    "detector": detector.name,
                    **result,
                }
            )

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    report_prefix = (
        f"{timestamp}_{safe_token(args.detector_mode)}_"
        f"{safe_token(args.prefix)}_limit{int(args.limit)}"
    )
    metadata = {
        "detector_mode": args.detector_mode,
        "bucket": args.bucket,
        "prefix": args.prefix,
        "limit": args.limit,
        "frames_loaded": len(keys),
        "scope": "ball_detection_only",
        "yolo_benchmark_inference": "direct_model_predict_without_bytetrack",
    }
    report_paths = write_reports(
        rows,
        Path(args.output_dir),
        report_prefix,
        metadata,
        detector_statuses,
    )
    print(
        json.dumps(
            {
                "frames": len(keys),
                "reports": report_paths,
                "detector_statuses": detector_statuses,
                "summary": summarize(rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
