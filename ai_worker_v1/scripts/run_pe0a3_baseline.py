#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
AI_WORKER_SRC = AI_WORKER_ROOT / "src"
PROD_SRC = REPO_ROOT / "src"
for path in (str(AI_WORKER_SRC), str(PROD_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

try:
    import pandas as pd  # noqa: E402
except Exception:  # pragma: no cover
    pd = None

from video_source import cleanup_resolved_source, resolve_video_source  # noqa: E402
from video_calibration.profile_io import profile_hash  # noqa: E402


PHASE = "PE-0A3"
SCRIPT_NAME = "ai_worker_v1/scripts/run_pe0a3_baseline.py"
CANONICAL_PERSON = "on_field_person"
CANONICAL_BALL = "ball"
PERSON_NAMES = {"person"}
BALL_NAMES = {"ball", "sports ball", "sports_ball"}
HARD_NEGATIVE_HINTS = {"line", "shoe", "sock", "advertising", "light", "net", "grass_mark"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def maybe_write_parquet(path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if pd is None:
        write_text(path.with_suffix(".parquet.unavailable.txt"), "pandas/pyarrow unavailable in this environment\n")
        return {"status": "unavailable", "reason": "pandas_or_pyarrow_unavailable"}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return {"status": "produced", "path": str(path)}
    except Exception as exc:
        write_text(path.with_suffix(".parquet.error.txt"), str(exc) + "\n")
        return {"status": "error", "reason": str(exc)}


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "detections": run_dir / "detections",
        "ball": run_dir / "ball",
        "metrics": run_dir / "metrics",
        "overlays": run_dir / "overlays",
        "dataset_images": run_dir / "dataset_seed" / "images",
        "dataset_annotations": run_dir / "dataset_seed" / "annotations",
        "dataset_yolo": run_dir / "dataset_seed" / "pseudo_annotations_yolo",
        "review": run_dir / "review",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def locate_capture(capture_id: str, capture_root: Path) -> Dict[str, Path]:
    matches = list(capture_root.glob(f"**/{capture_id}/calibration_capture_bundle.json"))
    if not matches:
        raise FileNotFoundError(f"capture_id not found under {capture_root}: {capture_id}")
    if len(matches) > 1:
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    bundle = matches[0]
    capture_dir = bundle.parent
    return {
        "capture_dir": capture_dir,
        "bundle": bundle,
        "reference_frame": capture_dir / "reference_frame.png",
    }


def find_materialized_profile(capture_id: str, capture_root: Path) -> Path:
    candidates = list(capture_root.glob(f"**/*{capture_id}*/video_calibration.json"))
    if not candidates:
        raise FileNotFoundError(f"video_calibration.json not found for capture_id={capture_id}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def download_source_video(bundle: Dict[str, Any], run_dir: Path) -> Path:
    source = (
        bundle.get("video", {}).get("source_reference_sanitized")
        or bundle.get("calibration_input", {}).get("source_reference")
    )
    if not source:
        raise ValueError("Capture bundle has no source video reference")
    resolved = resolve_video_source(source, {"temp_dir": str(run_dir / "_download")})
    try:
        expected_hash = bundle.get("video", {}).get("video_hash")
        actual_hash = sha256_file(resolved["local_path"])
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"video_hash mismatch: expected {expected_hash}, got {actual_hash}")
        destination = run_dir / "source_video.mp4"
        shutil.copyfile(resolved["local_path"], destination)
        return destination
    finally:
        cleanup_resolved_source(resolved)


def video_meta(video_path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = frame_count / fps if fps > 0 else 0.0
    return {
        "path": str(video_path),
        "sha256": sha256_file(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def choose_segment(reference_ts: float, duration: float, requested_duration: float) -> Tuple[float, float]:
    target_duration = min(max(requested_duration, 5.0), duration)
    start = max(0.0, reference_ts - target_duration / 2.0)
    if start + target_duration > duration:
        start = max(0.0, duration - target_duration)
    return start, min(target_duration, duration - start)


def make_source_segment(video_path: Path, output_path: Path, start_sec: float, duration_sec: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "18",
        "-an",
        "-loglevel",
        "error",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok and frame is not None else None


def make_contact_sheet(video_path: Path, output_path: Path, start_sec: float, duration_sec: float, fps: float) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("could not open source video for contact sheet")
    samples = []
    images = []
    t = start_sec
    while t <= start_sec + duration_sec + 1e-6:
        frame_index = int(round(t * fps))
        frame = read_frame_at(cap, frame_index)
        if frame is not None:
            thumb = cv2.resize(frame, (320, 180))
            cv2.putText(thumb, f"f={frame_index} t={t:.1f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            images.append(thumb)
            samples.append({"frame_index": frame_index, "timestamp_sec": round(t, 3)})
        t += 10.0
    cap.release()
    if not images:
        raise RuntimeError("no contact sheet frames extracted")
    cols = min(3, len(images))
    rows = int(math.ceil(len(images) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        sheet[row * 180 : (row + 1) * 180, col * 320 : (col + 1) * 320] = img
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)
    return samples


def polygon_points(profile: Dict[str, Any]) -> np.ndarray:
    points = profile.get("detection_roi", {}).get("polygon_pixels_reference") or []
    return np.array(points, dtype=np.float32)


def point_spatial_status(point: Tuple[float, float], polygon: np.ndarray) -> str:
    if polygon.size == 0:
        return "unavailable"
    distance = cv2.pointPolygonTest(polygon, point, True)
    if distance > 3.0:
        return "inside"
    if distance >= -8.0:
        return "boundary_uncertain"
    return "outside"


def xywh_from_xyxy(box: Iterable[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def center_from_xyxy(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bottom_center_from_xyxy(box: Iterable[float]) -> Tuple[float, float]:
    x1, _y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, y2


def bbox_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def canonical_class(name: str) -> Optional[str]:
    lowered = str(name or "").lower().replace("_", " ")
    if lowered in PERSON_NAMES:
        return CANONICAL_PERSON
    if lowered in BALL_NAMES:
        return CANONICAL_BALL
    return None


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clamp_box(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]


def yolo_predict(model: YOLO, frame: np.ndarray, imgsz: int, conf: float, iou: float) -> List[Dict[str, Any]]:
    results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    names = getattr(model, "names", {}) or {}
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            class_id = int(box.cls[0].item()) if box.cls is not None else -1
            conf_value = float(box.conf[0].item()) if box.conf is not None else 0.0
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": conf_value,
                    "bbox_xyxy": xyxy,
                }
            )
    return detections


def init_rfdetr(model_path: str, device: str):
    from rfdetr import RFDETRBase

    return RFDETRBase(pretrain_weights=model_path, device=device)


def rfdetr_predict(model: Any, frame: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
    rgb = frame[:, :, ::-1].copy()
    results = model.predict(rgb, threshold=threshold)
    class_names = getattr(model, "class_names", {}) or {}
    rows = []
    if results is None:
        return rows
    if all(hasattr(results, attr) for attr in ("xyxy", "confidence", "class_id")):
        for xyxy, conf, cls in zip(results.xyxy, results.confidence, results.class_id):
            class_id = int(cls)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": str(class_names.get(class_id, class_id)),
                    "confidence": float(conf),
                    "bbox_xyxy": [float(v) for v in xyxy],
                }
            )
        return rows
    if isinstance(results, dict):
        iterable = results.get("predictions") or results.get("detections") or []
    else:
        iterable = getattr(results, "predictions", results)
    for pred in iterable or []:
        if not isinstance(pred, dict):
            pred = {k: getattr(pred, k) for k in ("xmin", "ymin", "xmax", "ymax", "confidence", "class_id") if hasattr(pred, k)}
        xyxy = pred.get("xyxy")
        if xyxy is None:
            xyxy = [pred.get("xmin", pred.get("x1", 0.0)), pred.get("ymin", pred.get("y1", 0.0)), pred.get("xmax", pred.get("x2", 0.0)), pred.get("ymax", pred.get("y2", 0.0))]
        class_id = int(pred.get("class_id", pred.get("class", pred.get("category_id", -1))))
        rows.append(
            {
                "class_id": class_id,
                "class_name": str(class_names.get(class_id, pred.get("class_name", class_id))),
                "confidence": float(pred.get("confidence", pred.get("score", 0.0)) or 0.0),
                "bbox_xyxy": [float(v) for v in xyxy],
            }
        )
    return rows


def generate_tiles(width: int, height: int, tile_size: int, overlap: float, roi_polygon: np.ndarray) -> List[Dict[str, Any]]:
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []
    tile_id = 0
    roi_bbox = None
    if roi_polygon.size:
        xs = roi_polygon[:, 0]
        ys = roi_polygon[:, 1]
        margin = tile_size // 3
        roi_bbox = (
            max(0, int(xs.min()) - margin),
            max(0, int(ys.min()) - margin),
            min(width, int(xs.max()) + margin),
            min(height, int(ys.max()) + margin),
        )
    for y in range(0, max(1, height), stride):
        for x in range(0, max(1, width), stride):
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            if roi_bbox:
                rx1, ry1, rx2, ry2 = roi_bbox
                if x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2:
                    continue
            tiles.append({"tile_id": f"tile_{tile_id:04d}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            tile_id += 1
        if y + tile_size >= height:
            break
    return tiles


def make_detection_row(
    *,
    run_id: str,
    frame_index: int,
    processed_index: int,
    timestamp_sec: float,
    model_name: str,
    source_pass: str,
    tile_id: Optional[str],
    raw: Dict[str, Any],
    width: int,
    height: int,
    roi_polygon: np.ndarray,
) -> Dict[str, Any]:
    box = clamp_box(raw["bbox_xyxy"], width, height)
    class_name = str(raw.get("class_name", ""))
    canonical = canonical_class(class_name)
    center = center_from_xyxy(box)
    bottom = bottom_center_from_xyxy(box)
    spatial_point = bottom if canonical == CANONICAL_PERSON else center
    spatial = point_spatial_status(spatial_point, roi_polygon)
    x, y, w, h = xywh_from_xyxy(box)
    return {
        "run_id": run_id,
        "frame_index": int(frame_index),
        "processed_frame_index": int(processed_index),
        "timestamp_sec": round(float(timestamp_sec), 6),
        "source_model": model_name,
        "source_pass": source_pass,
        "tile_id": tile_id,
        "raw_class_name": class_name,
        "class_id": int(raw.get("class_id", -1)),
        "canonical_class": canonical,
        "confidence": round(float(raw.get("confidence", 0.0)), 6),
        "bbox_xyxy": [round(float(v), 4) for v in box],
        "bbox_xywh": [round(x, 4), round(y, 4), round(w, 4), round(h, 4)],
        "center": [round(center[0], 4), round(center[1], 4)],
        "bottom_center": [round(bottom[0], 4), round(bottom[1], 4)],
        "width_px": round(w, 4),
        "height_px": round(h, 4),
        "area_px": round(w * h, 4),
        "spatial_status": spatial,
        "accepted_for_review": canonical in {CANONICAL_PERSON, CANONICAL_BALL},
        "review_status": "unreviewed",
        "rejection_reason": None if canonical in {CANONICAL_PERSON, CANONICAL_BALL} else "non_target_class",
        "detection_id": stable_id(run_id, frame_index, model_name, source_pass, tile_id, class_name, box),
    }


def dedupe_ball_candidates(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model = defaultdict(list)
    for row in rows:
        by_model[(row["source_model"], row["frame_index"])].append(row)
    output = []
    group_counter = 0
    for _key, items in by_model.items():
        items = sorted(items, key=lambda row: row["confidence"], reverse=True)
        groups: List[List[Dict[str, Any]]] = []
        for item in items:
            matched = None
            for group in groups:
                best = group[0]
                center_dist = math.hypot(item["center"][0] - best["center"][0], item["center"][1] - best["center"][1])
                size_gate = max(item["width_px"], item["height_px"], best["width_px"], best["height_px"], 12.0)
                if bbox_iou(item["bbox_xyxy"], best["bbox_xyxy"]) >= 0.3 or center_dist <= size_gate:
                    matched = group
                    break
            if matched is None:
                groups.append([item])
            else:
                matched.append(item)
        for group in groups:
            group_id = f"dup_{group_counter:06d}"
            group_counter += 1
            for idx, item in enumerate(group):
                item = dict(item)
                item["duplicate_group"] = group_id
                item["duplicate_rank"] = idx
                output.append(item)
    return output


def ball_agreement(yolo_rows: List[Dict[str, Any]], rfdetr_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frames = sorted({row["frame_index"] for row in yolo_rows + rfdetr_rows})
    y_by_frame = defaultdict(list)
    r_by_frame = defaultdict(list)
    for row in yolo_rows:
        y_by_frame[row["frame_index"]].append(row)
    for row in rfdetr_rows:
        r_by_frame[row["frame_index"]].append(row)
    both = []
    yolo_only = []
    rfdetr_only = []
    for frame in frames:
        y_rows = y_by_frame.get(frame, [])
        r_rows = r_by_frame.get(frame, [])
        agreed = False
        for y in y_rows:
            for r in r_rows:
                if bbox_iou(y["bbox_xyxy"], r["bbox_xyxy"]) >= 0.25 or math.hypot(y["center"][0] - r["center"][0], y["center"][1] - r["center"][1]) <= 25:
                    agreed = True
        if agreed:
            both.append(frame)
        elif y_rows:
            yolo_only.append(frame)
        elif r_rows:
            rfdetr_only.append(frame)
    return {"both": both, "yolo_only": yolo_only, "rfdetr_only": rfdetr_only}


def draw_roi(frame: np.ndarray, roi_polygon: np.ndarray) -> None:
    if roi_polygon.size:
        pts = roi_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)


def draw_detections(frame: np.ndarray, rows: List[Dict[str, Any]], model_label: str) -> np.ndarray:
    out = frame.copy()
    colors = {
        CANONICAL_PERSON: (60, 220, 60),
        CANONICAL_BALL: (60, 60, 255),
        None: (128, 128, 128),
    }
    for row in rows:
        if row.get("canonical_class") not in {CANONICAL_PERSON, CANONICAL_BALL}:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        color = colors.get(row.get("canonical_class"), (128, 128, 128))
        if row.get("spatial_status") == "boundary_uncertain":
            color = (0, 165, 255)
        elif row.get("spatial_status") == "outside":
            color = (140, 140, 140)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{model_label}:{row['canonical_class']} {row['confidence']:.2f} {row['spatial_status']}"
        cv2.putText(out, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def write_overlay_header(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(1.0, float(fps)), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def save_seed_image(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


def coco_annotation_for(row: Dict[str, Any], image_id: int, annotation_id: int) -> Dict[str, Any]:
    category_id = 1 if row["canonical_class"] == CANONICAL_PERSON else 2
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": row["bbox_xywh"],
        "area": row["area_px"],
        "iscrowd": 0,
        "source_model": row["source_model"],
        "confidence": row["confidence"],
        "agreement": row.get("agreement_status"),
        "review_status": "unreviewed",
        "pseudo_label": True,
        "ground_truth": False,
    }


def run_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.run_id
    run_dir = AI_WORKER_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(run_dir)
    started = time.time()

    located = locate_capture(args.capture_id, Path(args.capture_root))
    bundle = load_json(located["bundle"])
    profile_path = Path(args.profile) if args.profile else find_materialized_profile(args.capture_id, Path(args.capture_root))
    profile = load_json(profile_path)
    ref_hash = sha256_file(located["reference_frame"])
    if ref_hash != bundle["reference_frame"]["sha256"]:
        raise ValueError("reference frame hash mismatch")
    if profile.get("calibration_id") != args.calibration_id:
        raise ValueError(f"calibration_id mismatch: {profile.get('calibration_id')} != {args.calibration_id}")

    source_video = download_source_video(bundle, run_dir)
    source_meta = video_meta(source_video)
    reference_ts = float(bundle["reference_frame"]["timestamp_sec"])
    start_sec, duration_sec = choose_segment(reference_ts, source_meta["duration_sec"], args.duration_sec)
    if args.smoke:
        duration_sec = min(float(args.smoke_seconds), duration_sec)
    segment_path = run_dir / "source_segment.mp4"
    make_source_segment(source_video, segment_path, start_sec, duration_sec)
    segment_meta = video_meta(segment_path)
    contact_samples = make_contact_sheet(source_video, run_dir / "contact_sheet.png", start_sec, duration_sec, source_meta["fps"])

    write_json(
        run_dir / "source_segment_manifest.json",
        {
            "run_id": run_id,
            "capture_id": args.capture_id,
            "source_video": str(source_video),
            "source_video_hash": source_meta["sha256"],
            "expected_video_hash": bundle.get("video", {}).get("video_hash"),
            "source_segment": str(segment_path),
            "source_segment_hash": segment_meta["sha256"],
            "start_sec": start_sec,
            "duration_sec": duration_sec,
            "fps_original": source_meta["fps"],
            "fps_segment": segment_meta["fps"],
            "resolution": [source_meta["width"], source_meta["height"]],
            "reference_timestamp_sec": reference_ts,
            "contact_sheet_samples": contact_samples,
        },
    )

    roi_polygon = polygon_points(profile)
    yolo_model_path = Path(args.yolo_model)
    rfdetr_model_path = Path(args.rfdetr_model)
    yolo_sha = sha256_file(yolo_model_path)
    rfdetr_sha = sha256_file(rfdetr_model_path)

    import ultralytics
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(yolo_model_path))
    rfdetr = init_rfdetr(str(rfdetr_model_path), device=device)

    model_manifest = {
        "yolo": {
            "checkpoint": str(yolo_model_path),
            "sha256": yolo_sha,
            "architecture": "ultralytics_yolo",
            "ultralytics_version": ultralytics.__version__,
            "input_resolution": args.yolo_imgsz,
            "confidence": args.confidence,
            "iou": args.iou,
            "device": device,
            "precision": "default",
        },
        "rfdetr": {
            "variant": "RFDETRBase",
            "checkpoint": str(rfdetr_model_path),
            "sha256": rfdetr_sha,
            "package_version": "unknown",
            "input_resolution": "native_model_preprocess",
            "confidence": args.rfdetr_confidence,
            "device": device,
            "precision": "default",
        },
    }
    write_json(run_dir / "model_manifest.json", model_manifest)

    process_every = max(1, int(round(source_meta["fps"] / max(0.1, args.target_fps))))
    output_fps = source_meta["fps"] / process_every
    width, height = int(source_meta["width"]), int(source_meta["height"])
    tiles = generate_tiles(width, height, args.tile_size, args.tile_overlap, roi_polygon)

    cap = cv2.VideoCapture(str(source_video))
    start_frame = int(round(start_sec * source_meta["fps"]))
    end_frame = int(round((start_sec + duration_sec) * source_meta["fps"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writers = {
        "yolo": open_video_writer(dirs["overlays"] / "yolo_baseline.mp4", output_fps, width, height),
        "rfdetr": open_video_writer(dirs["overlays"] / "rfdetr_baseline.mp4", output_fps, width, height),
        "comparison": open_video_writer(dirs["overlays"] / "comparison_baseline.mp4", output_fps, width * 2, height),
        "fusion": open_video_writer(dirs["overlays"] / "fusion_review.mp4", output_fps, width, height),
        "outside": open_video_writer(dirs["overlays"] / "diagnostic_outside_roi.mp4", output_fps, width, height),
    }

    yolo_raw: List[Dict[str, Any]] = []
    rfdetr_raw: List[Dict[str, Any]] = []
    canonical_rows: List[Dict[str, Any]] = []
    yolo_ball: List[Dict[str, Any]] = []
    rfdetr_ball: List[Dict[str, Any]] = []
    hard_negatives: List[Dict[str, Any]] = []
    runtime_rows = []
    review_items = []
    seed_frames: Dict[int, np.ndarray] = {}
    processed = 0
    frames_seen = 0
    errors = []

    while True:
        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_index > end_frame:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames_seen += 1
        if (frame_index - start_frame) % process_every != 0:
            continue
        timestamp = frame_index / source_meta["fps"]
        processed += 1
        y_start = time.time()
        y_global_raw = yolo_predict(yolo, frame, args.yolo_imgsz, args.confidence, args.iou)
        y_runtime = time.time() - y_start
        r_start = time.time()
        r_global_raw = rfdetr_predict(rfdetr, frame, args.rfdetr_confidence)
        r_runtime = time.time() - r_start

        y_frame_rows = [
            make_detection_row(
                run_id=run_id,
                frame_index=frame_index,
                processed_index=processed - 1,
                timestamp_sec=timestamp,
                model_name="yolo",
                source_pass="global",
                tile_id=None,
                raw=raw,
                width=width,
                height=height,
                roi_polygon=roi_polygon,
            )
            for raw in y_global_raw
        ]
        r_frame_rows = [
            make_detection_row(
                run_id=run_id,
                frame_index=frame_index,
                processed_index=processed - 1,
                timestamp_sec=timestamp,
                model_name="rfdetr",
                source_pass="global",
                tile_id=None,
                raw=raw,
                width=width,
                height=height,
                roi_polygon=roi_polygon,
            )
            for raw in r_global_raw
        ]

        # Tile pass for ball candidates only. This preserves original coordinates.
        for tile in tiles:
            crop = frame[tile["y1"] : tile["y2"], tile["x1"] : tile["x2"]]
            if crop.size == 0:
                continue
            for raw in yolo_predict(yolo, crop, args.tile_size, args.confidence, args.iou):
                if canonical_class(str(raw.get("class_name", ""))) != CANONICAL_BALL:
                    continue
                raw = dict(raw)
                raw["bbox_xyxy"] = [
                    raw["bbox_xyxy"][0] + tile["x1"],
                    raw["bbox_xyxy"][1] + tile["y1"],
                    raw["bbox_xyxy"][2] + tile["x1"],
                    raw["bbox_xyxy"][3] + tile["y1"],
                ]
                y_frame_rows.append(
                    make_detection_row(
                        run_id=run_id,
                        frame_index=frame_index,
                        processed_index=processed - 1,
                        timestamp_sec=timestamp,
                        model_name="yolo",
                        source_pass="tile",
                        tile_id=tile["tile_id"],
                        raw=raw,
                        width=width,
                        height=height,
                        roi_polygon=roi_polygon,
                    )
                )
            for raw in rfdetr_predict(rfdetr, crop, args.rfdetr_confidence):
                if canonical_class(str(raw.get("class_name", ""))) != CANONICAL_BALL:
                    continue
                raw = dict(raw)
                raw["bbox_xyxy"] = [
                    raw["bbox_xyxy"][0] + tile["x1"],
                    raw["bbox_xyxy"][1] + tile["y1"],
                    raw["bbox_xyxy"][2] + tile["x1"],
                    raw["bbox_xyxy"][3] + tile["y1"],
                ]
                r_frame_rows.append(
                    make_detection_row(
                        run_id=run_id,
                        frame_index=frame_index,
                        processed_index=processed - 1,
                        timestamp_sec=timestamp,
                        model_name="rfdetr",
                        source_pass="tile",
                        tile_id=tile["tile_id"],
                        raw=raw,
                        width=width,
                        height=height,
                        roi_polygon=roi_polygon,
                    )
                )

        yolo_raw.extend(y_frame_rows)
        rfdetr_raw.extend(r_frame_rows)
        canonical_rows.extend([row for row in y_frame_rows + r_frame_rows if row["canonical_class"] in {CANONICAL_PERSON, CANONICAL_BALL}])
        yolo_ball.extend([row for row in y_frame_rows if row["canonical_class"] == CANONICAL_BALL])
        rfdetr_ball.extend([row for row in r_frame_rows if row["canonical_class"] == CANONICAL_BALL])
        hard_negatives.extend([row for row in y_frame_rows + r_frame_rows if row["canonical_class"] is None and row["confidence"] >= args.hard_negative_confidence])

        if len(seed_frames) < args.max_seed_frames and (
            any(row["canonical_class"] == CANONICAL_BALL for row in y_frame_rows + r_frame_rows)
            or any(row["spatial_status"] == "boundary_uncertain" for row in y_frame_rows + r_frame_rows)
            or processed % max(1, int(output_fps * 3)) == 0
        ):
            seed_frames[frame_index] = frame.copy()

        y_vis = draw_detections(frame, [row for row in y_frame_rows if row["source_pass"] == "global"], "YOLO")
        r_vis = draw_detections(frame, [row for row in r_frame_rows if row["source_pass"] == "global"], "RFDETR")
        fusion_vis = draw_detections(frame, [row for row in y_frame_rows + r_frame_rows if row["canonical_class"] in {CANONICAL_BALL, CANONICAL_PERSON}], "FUSION")
        outside_vis = draw_detections(frame, [row for row in y_frame_rows + r_frame_rows if row["canonical_class"] == CANONICAL_PERSON], "SPATIAL")
        for vis in (y_vis, r_vis, fusion_vis, outside_vis):
            draw_roi(vis, roi_polygon)
        write_overlay_header(y_vis, f"YOLO f={frame_index} t={timestamp:.2f}s")
        write_overlay_header(r_vis, f"RF-DETR f={frame_index} t={timestamp:.2f}s")
        write_overlay_header(fusion_vis, f"Fusion review f={frame_index} t={timestamp:.2f}s")
        write_overlay_header(outside_vis, f"Outside ROI diagnostic f={frame_index} t={timestamp:.2f}s")
        writers["yolo"].write(y_vis)
        writers["rfdetr"].write(r_vis)
        writers["comparison"].write(np.concatenate([y_vis, r_vis], axis=1))
        writers["fusion"].write(fusion_vis)
        writers["outside"].write(outside_vis)

        runtime_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp, 6),
                "yolo_runtime_sec": y_runtime,
                "rfdetr_runtime_sec": r_runtime,
                "total_runtime_sec": y_runtime + r_runtime,
                "tile_count": len(tiles),
            }
        )

        if args.max_processed_frames and processed >= args.max_processed_frames:
            break

    cap.release()
    for writer in writers.values():
        writer.release()

    yolo_ball = dedupe_ball_candidates(yolo_ball)
    rfdetr_ball = dedupe_ball_candidates(rfdetr_ball)
    agreement = ball_agreement(yolo_ball, rfdetr_ball)
    for row in yolo_ball + rfdetr_ball:
        if row["frame_index"] in agreement["both"]:
            row["agreement_status"] = "both"
        elif row["source_model"] == "yolo":
            row["agreement_status"] = "yolo_only"
        else:
            row["agreement_status"] = "rfdetr_only"
        row["pseudo_label"] = True
        row["ground_truth"] = False

    for row in canonical_rows:
        if row["canonical_class"] == CANONICAL_PERSON and row["spatial_status"] in {"boundary_uncertain", "outside"}:
            review_items.append(
                {
                    "priority": "P0" if row["spatial_status"] == "outside" else "P1",
                    "frame_index": row["frame_index"],
                    "timestamp_sec": row["timestamp_sec"],
                    "reason": f"person_{row['spatial_status']}",
                    "action_required": "review spatial ROI classification",
                    "detection_id": row["detection_id"],
                }
            )
    for row in yolo_ball + rfdetr_ball:
        priority = "P0" if row.get("agreement_status") != "both" or row.get("source_pass") == "tile" else "P1"
        reason = "ball_tile_only" if row.get("source_pass") == "tile" else f"ball_{row.get('agreement_status')}"
        review_items.append(
            {
                "priority": priority,
                "frame_index": row["frame_index"],
                "timestamp_sec": row["timestamp_sec"],
                "reason": reason,
                "action_required": "confirm ball candidate",
                "detection_id": row["detection_id"],
            }
        )

    review_items.sort(key=lambda item: (item["priority"], item["frame_index"]))
    priority_sequences = {
        "P0": [item for item in review_items if item["priority"] == "P0"][:200],
        "P1": [item for item in review_items if item["priority"] == "P1"][:200],
        "P2": [item for item in review_items if item["priority"] == "P2"][:200],
    }

    write_jsonl(dirs["detections"] / "yolo_raw.jsonl", yolo_raw)
    write_jsonl(dirs["detections"] / "rfdetr_raw.jsonl", rfdetr_raw)
    write_jsonl(dirs["detections"] / "canonical_detections.jsonl", canonical_rows)
    write_jsonl(dirs["ball"] / "yolo_ball_candidates.jsonl", yolo_ball)
    write_jsonl(dirs["ball"] / "rfdetr_ball_candidates.jsonl", rfdetr_ball)
    maybe_write_parquet(dirs["detections"] / "yolo_raw.parquet", yolo_raw)
    maybe_write_parquet(dirs["detections"] / "rfdetr_raw.parquet", rfdetr_raw)

    write_json(dirs["ball"] / "ball_disagreements.json", agreement)
    write_json(dirs["ball"] / "hard_negative_candidates.json", hard_negatives[:1000])
    runtime_total = time.time() - started
    write_json(
        dirs["metrics"] / "runtime_metrics.json",
        {
            "total_runtime_sec": runtime_total,
            "processed_frames": processed,
            "effective_processed_fps": processed / runtime_total if runtime_total else 0,
            "target_fps": args.target_fps,
            "process_every_n_frames": process_every,
            "per_frame": runtime_rows,
            "errors": errors,
        },
    )
    spatial_counts = Counter(row["spatial_status"] for row in canonical_rows)
    detection_counts = {
        "processed_frames": processed,
        "yolo_total_detections": len(yolo_raw),
        "rfdetr_total_detections": len(rfdetr_raw),
        "yolo_person_detections": sum(1 for row in yolo_raw if row["canonical_class"] == CANONICAL_PERSON),
        "rfdetr_person_detections": sum(1 for row in rfdetr_raw if row["canonical_class"] == CANONICAL_PERSON),
        "yolo_ball_candidates": len(yolo_ball),
        "rfdetr_ball_candidates": len(rfdetr_ball),
        "frames_with_yolo_ball": len({row["frame_index"] for row in yolo_ball}),
        "frames_with_rfdetr_ball": len({row["frame_index"] for row in rfdetr_ball}),
        "spatial_counts": dict(spatial_counts),
    }
    write_json(dirs["metrics"] / "detection_counts.json", detection_counts)
    write_json(
        dirs["metrics"] / "disagreement_metrics.json",
        {
            "ball_frames_both": len(agreement["both"]),
            "ball_frames_yolo_only": len(agreement["yolo_only"]),
            "ball_frames_rfdetr_only": len(agreement["rfdetr_only"]),
            "candidate_maximum_gap": candidate_max_gap(sorted({row["frame_index"] for row in yolo_ball + rfdetr_ball})),
        },
    )
    write_json(dirs["metrics"] / "spatial_metrics.json", dict(spatial_counts))

    # Dataset seed.
    coco_images = []
    coco_annotations = []
    yolo_lines_by_frame = defaultdict(list)
    ann_id = 1
    for image_id, (frame_index, frame) in enumerate(sorted(seed_frames.items()), start=1):
        filename = f"frame_{frame_index:08d}.jpg"
        image_path = dirs["dataset_images"] / filename
        save_seed_image(image_path, frame)
        coco_images.append({"id": image_id, "file_name": filename, "width": width, "height": height, "frame_index": frame_index})
        frame_rows = [row for row in canonical_rows if row["frame_index"] == frame_index]
        for row in frame_rows:
            coco_annotations.append(coco_annotation_for(row, image_id, ann_id))
            ann_id += 1
            x, y, w, h = row["bbox_xywh"]
            class_id = 0 if row["canonical_class"] == CANONICAL_PERSON else 1
            yolo_lines_by_frame[filename].append(
                f"{class_id} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"
            )
    for filename, lines in yolo_lines_by_frame.items():
        write_text(dirs["dataset_yolo"] / f"{Path(filename).stem}.txt", "\n".join(lines) + "\n")
    write_json(
        run_dir / "dataset_seed" / "pseudo_annotations_coco.json",
        {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": CANONICAL_PERSON}, {"id": 2, "name": CANONICAL_BALL}],
            "review_status": "unreviewed",
            "pseudo_label": True,
            "ground_truth": False,
        },
    )
    write_json(
        run_dir / "dataset_seed" / "dataset_manifest.json",
        {
            "frames": len(coco_images),
            "pseudo_annotations": len(coco_annotations),
            "review_status": "unreviewed",
            "ground_truth": False,
            "selection_policy": "PE-0A3 diagnostic seed, not training-approved",
        },
    )
    write_json(run_dir / "dataset_seed" / "review_manifest.json", {"items": review_items, "review_status": "unreviewed"})
    write_text(
        run_dir / "dataset_seed" / "README.md",
        "# PE-0A3 Dataset Seed\n\nPseudo labels only. Do not train until human review approves a subset.\n",
    )
    write_json(dirs["review"] / "review_queue.json", {"items": review_items})
    write_json(dirs["review"] / "priority_sequences.json", priority_sequences)
    write_text(
        dirs["review"] / "review_instructions.md",
        "# Review Instructions\n\nPrioritize P0 ball/person disagreements. Mark reviewed items outside this run; no labels here are ground truth.\n",
    )
    make_review_contact_sheet(dirs["review"] / "review_contact_sheet.png", sorted(seed_frames.items())[:24])

    run_manifest = {
        "run_id": run_id,
        "phase": PHASE,
        "status": "completed",
        "created_at": utc_now(),
        "capture_id": args.capture_id,
        "match_id": bundle.get("job", {}).get("match_id"),
        "job_id": bundle.get("job", {}).get("job_id"),
        "calibration_id": profile.get("calibration_id"),
        "profile_hash": profile_hash_obj(profile),
        "video_hash": source_meta["sha256"],
        "reference_frame_hash": ref_hash,
        "segment": {"start_sec": start_sec, "duration_sec": duration_sec, "smoke": bool(args.smoke)},
        "production_intact": True,
        "external_services": {"runpod": False, "supabase": False, "r2": False},
    }
    write_json(run_dir / "run_manifest.json", run_manifest)

    report = build_final_report(
        run_id=run_id,
        source_meta=source_meta,
        segment={"start_sec": start_sec, "duration_sec": duration_sec, "fps": source_meta["fps"], "resolution": [width, height]},
        model_manifest=model_manifest,
        detection_counts=detection_counts,
        agreement=agreement,
        runtime_total=runtime_total,
        output_fps=output_fps,
        dataset_frames=len(coco_images),
        pseudo_annotations=len(coco_annotations),
        review_items=review_items,
        run_dir=run_dir,
        errors=errors,
        capture_id=args.capture_id,
        calibration_id=profile.get("calibration_id"),
    )
    write_text(run_dir / "PE0A3_FINAL_REPORT.md", report)
    write_artifact_manifest(run_dir)
    return {
        "status": "completed",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "processed_frames": processed,
        "runtime_sec": runtime_total,
        "detection_counts": detection_counts,
    }


def profile_hash_obj(profile: Dict[str, Any]) -> str:
    class Wrapper:
        def __init__(self, data):
            self._data = data
        def to_dict(self):
            return self._data
    try:
        return profile_hash(Wrapper(profile))
    except Exception:
        return hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest()


def candidate_max_gap(frames: List[int]) -> int:
    if len(frames) < 2:
        return 0
    return max(b - a for a, b in zip(frames, frames[1:]))


def make_review_contact_sheet(path: Path, frames: List[Tuple[int, np.ndarray]]) -> None:
    if not frames:
        blank = np.zeros((180, 320, 3), dtype=np.uint8)
        cv2.putText(blank, "No seed frames", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(str(path), blank)
        return
    thumbs = []
    for frame_index, frame in frames:
        thumb = cv2.resize(frame, (320, 180))
        cv2.putText(thumb, f"f={frame_index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        thumbs.append(thumb)
    cols = min(4, len(thumbs))
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        sheet[row * 180 : (row + 1) * 180, col * 320 : (col + 1) * 320] = thumb
    cv2.imwrite(str(path), sheet)


def build_final_report(**kwargs) -> str:
    counts = kwargs["detection_counts"]
    agreement = kwargs["agreement"]
    model_manifest = kwargs["model_manifest"]
    run_dir = kwargs["run_dir"]
    review_items = kwargs["review_items"]
    priority_counts = Counter(item["priority"] for item in review_items)
    return "\n".join(
        [
            "# PE-0A3 Final Report",
            "",
            f"- status: completed",
            f"- run_id: `{kwargs['run_id']}`",
            f"- capture_id: `{kwargs['capture_id']}`",
            f"- calibration_id: `{kwargs['calibration_id']}`",
            f"- video_hash: `{kwargs['source_meta']['sha256']}`",
            f"- segment: `{kwargs['segment']}`",
            "",
            "## Models",
            f"- YOLO checkpoint: `{model_manifest['yolo']['checkpoint']}`",
            f"- YOLO hash: `{model_manifest['yolo']['sha256']}`",
            f"- RF-DETR variant: `{model_manifest['rfdetr']['variant']}`",
            f"- RF-DETR checkpoint: `{model_manifest['rfdetr']['checkpoint']}`",
            f"- RF-DETR hash: `{model_manifest['rfdetr']['sha256']}`",
            "",
            "## Counts",
            f"- processed_frames: `{counts['processed_frames']}`",
            f"- yolo_person_detections: `{counts['yolo_person_detections']}`",
            f"- rfdetr_person_detections: `{counts['rfdetr_person_detections']}`",
            f"- yolo_ball_candidates: `{counts['yolo_ball_candidates']}`",
            f"- rfdetr_ball_candidates: `{counts['rfdetr_ball_candidates']}`",
            f"- spatial_counts: `{counts['spatial_counts']}`",
            f"- ball both: `{len(agreement['both'])}`",
            f"- ball yolo_only: `{len(agreement['yolo_only'])}`",
            f"- ball rfdetr_only: `{len(agreement['rfdetr_only'])}`",
            "",
            "## Runtime",
            f"- total_runtime_sec: `{kwargs['runtime_total']:.3f}`",
            f"- output_fps: `{kwargs['output_fps']:.3f}`",
            f"- errors: `{kwargs['errors']}`",
            "",
            "## Dataset Seed",
            f"- frames: `{kwargs['dataset_frames']}`",
            f"- pseudo_annotations: `{kwargs['pseudo_annotations']}`",
            f"- review_status: `unreviewed`",
            "",
            "## Review Queue",
            f"- P0: `{priority_counts.get('P0', 0)}`",
            f"- P1: `{priority_counts.get('P1', 0)}`",
            f"- P2: `{priority_counts.get('P2', 0)}`",
            "",
            "## Videos",
            f"- yolo: `{run_dir / 'overlays/yolo_baseline.mp4'}`",
            f"- rfdetr: `{run_dir / 'overlays/rfdetr_baseline.mp4'}`",
            f"- comparison: `{run_dir / 'overlays/comparison_baseline.mp4'}`",
            f"- fusion: `{run_dir / 'overlays/fusion_review.mp4'}`",
            f"- outside_roi: `{run_dir / 'overlays/diagnostic_outside_roi.mp4'}`",
            "",
            "No production integration, Supabase, R2, RunPod, GameReferee, Goal Safety Net, or training was executed.",
            "",
        ]
    )


def write_artifact_manifest(run_dir: Path) -> None:
    artifacts = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "created_at": utc_now(),
                "phase": PHASE,
            }
        )
    write_json(run_dir / "artifact_manifest.json", {"artifacts": artifacts})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PE-0A3 real person + ball baseline.")
    parser.add_argument("--capture-id", default="ccb_ec8836978221c786ed55a0ab")
    parser.add_argument("--capture-root", default="/tmp/oneframe_pe0a2c_capture")
    parser.add_argument("--calibration-id", default="vc_a90e53754cb6083389782e25")
    parser.add_argument("--profile")
    parser.add_argument("--run-id", default=f"pe0a3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    parser.add_argument("--duration-sec", type=float, default=90.0)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-seconds", type=float, default=5.0)
    parser.add_argument("--max-processed-frames", type=int, default=0)
    parser.add_argument("--yolo-model", default=str(AI_WORKER_SRC / "oneframe_v3_best.pt"))
    parser.add_argument("--rfdetr-model", default=str(AI_WORKER_ROOT / "rfdetr_cache" / "rf-detr-base.pth"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--rfdetr-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-seed-frames", type=int, default=400)
    parser.add_argument("--hard-negative-confidence", type=float, default=0.35)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_baseline(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
