from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


FRAME_W = 1920
FRAME_H = 1080
BALL_NAMES = {"ball", "sports ball", "sports_ball"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_class(name: str) -> Optional[str]:
    return "ball" if str(name or "").lower().replace("_", " ") in BALL_NAMES else None


def full_field_points() -> List[List[float]]:
    return [
        [354.0, 966.0],
        [4.0, 948.0],
        [4.0, 286.0],
        [540.0, 96.0],
        [1210.0, 108.0],
        [1510.0, 145.0],
        [1918.0, 352.0],
        [1900.0, 1052.0],
        [1578.0, 1078.0],
        [363.0, 1068.0],
    ]


def selected_sequences() -> List[Dict[str, Any]]:
    return [
        {"sequence_id": "test_ball_mixed_01", "split": "within_video_test_v0", "start_sec": 204.0, "end_sec": 209.0, "review_order": 1},
        {"sequence_id": "valid_ball_mixed_01", "split": "valid", "start_sec": 154.0, "end_sec": 159.0, "review_order": 2},
        {"sequence_id": "train_ball_hard_03", "split": "train", "start_sec": 70.0, "end_sec": 75.0, "review_order": 3},
        {"sequence_id": "train_ball_context_02", "split": "train", "start_sec": 31.0, "end_sec": 36.0, "review_order": 4},
    ]


def clamp_box(box: List[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(float(width), float(x1))),
        max(0.0, min(float(height), float(y1))),
        max(0.0, min(float(width), float(x2))),
        max(0.0, min(float(height), float(y2))),
    ]


def center_from_xyxy(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


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


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> List[Dict[str, Any]]:
    roi = np.array(full_field_points(), dtype=np.float32)
    xs, ys = roi[:, 0], roi[:, 1]
    margin = tile_size // 3
    roi_bbox = (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin),
        min(height, int(ys.max()) + margin),
    )
    stride = max(1, int(tile_size * (1.0 - overlap)))
    tiles = []
    tile_id = 0
    for y in range(0, max(1, height), stride):
        for x in range(0, max(1, width), stride):
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            rx1, ry1, rx2, ry2 = roi_bbox
            if x2 < rx1 or x1 > rx2 or y2 < ry1 or y1 > ry2:
                continue
            tiles.append({"tile_id": f"tile_{tile_id:04d}", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            tile_id += 1
        if y + tile_size >= height:
            break
    return tiles


def yolo_predict(model: YOLO, frame: np.ndarray, imgsz: int, conf: float, iou: float) -> List[Dict[str, Any]]:
    results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    names = getattr(model, "names", {}) or {}
    detections = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0].item()) if box.cls is not None else -1
            class_name = str(names.get(cls, cls))
            if canonical_class(class_name) != "ball":
                continue
            detections.append(
                {
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": float(box.conf[0].item()) if box.conf is not None else 0.0,
                    "bbox_xyxy": [float(v) for v in box.xyxy[0].tolist()],
                }
            )
    return detections


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
            class_name = str(class_names.get(class_id, class_id))
            if canonical_class(class_name) != "ball":
                continue
            rows.append({"class_id": class_id, "class_name": class_name, "confidence": float(conf), "bbox_xyxy": [float(v) for v in xyxy]})
    return rows


def make_row(
    *,
    run_id: str,
    frame_id: str,
    sequence_id: str,
    split: str,
    frame_index: int,
    timestamp_sec: float,
    source_model: str,
    source_pass: str,
    tile: Optional[Dict[str, Any]],
    raw: Dict[str, Any],
    width: int,
    height: int,
) -> Dict[str, Any]:
    box = clamp_box(raw["bbox_xyxy"], width, height)
    center = center_from_xyxy(box)
    tile_bbox = [tile["x1"], tile["y1"], tile["x2"], tile["y2"]] if tile else None
    return {
        "run_id": run_id,
        "frame_id": frame_id,
        "sequence_id": sequence_id,
        "split": split,
        "frame_index": int(frame_index),
        "timestamp_sec": round(float(timestamp_sec), 6),
        "source_model": source_model,
        "source_pass": source_pass,
        "tile_id": tile["tile_id"] if tile else None,
        "tile_bbox_xyxy": tile_bbox,
        "scale": 1.0,
        "padding": [0, 0, 0, 0],
        "bbox_tile_xyxy": raw.get("bbox_tile_xyxy"),
        "bbox_xyxy": [round(float(v), 4) for v in box],
        "center": [round(center[0], 4), round(center[1], 4)],
        "class_id": int(raw.get("class_id", -1)),
        "class_name": str(raw.get("class_name", "")),
        "confidence": round(float(raw.get("confidence", 0.0)), 6),
        "agreement": "pending_fusion",
        "candidate_status": "unreviewed",
        "pseudo_label": True,
        "ground_truth": False,
        "reviewed": False,
        "tracking_generated": False,
        "interpolated": False,
    }


def fuse_agreement(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[row["frame_id"]].append(row)
    fused = []
    for frame_rows in by_frame.values():
        yolo_rows = [row for row in frame_rows if row["source_model"] == "yolo"]
        rfdetr_rows = [row for row in frame_rows if row["source_model"] == "rfdetr"]
        for row in frame_rows:
            others = rfdetr_rows if row["source_model"] == "yolo" else yolo_rows
            agreement = f"{row['source_model']}_only"
            for other in others:
                dist = math.hypot(row["center"][0] - other["center"][0], row["center"][1] - other["center"][1])
                if bbox_iou(row["bbox_xyxy"], other["bbox_xyxy"]) >= 0.25 or dist <= 25:
                    agreement = "both"
                    break
            item = dict(row)
            item["agreement"] = agreement
            fused.append(item)
    return fused


def validate_outputs(frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]], expected_count: int, width: int, height: int) -> Dict[str, Any]:
    frame_ids = [frame["frame_id"] for frame in frames]
    errors = []
    if len(frames) != expected_count:
        errors.append("frame_count_mismatch")
    if len(set(frame_ids)) != len(frame_ids):
        errors.append("duplicate_frame_ids")
    for row in candidates:
        x1, y1, x2, y2 = row["bbox_xyxy"]
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
            errors.append(f"bbox_invalid:{row['frame_id']}:{row['source_model']}:{row['source_pass']}")
        if row.get("ground_truth") or row.get("reviewed") or row.get("interpolated") or row.get("tracking_generated"):
            errors.append(f"candidate_semantics_invalid:{row['frame_id']}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "frames_expected": expected_count,
        "frames_processed": len(frames),
        "candidate_count": len(candidates),
        "unique_frames": len(set(frame_ids)),
        "bbox_errors": [e for e in errors if e.startswith("bbox_invalid")],
    }


def artifact_manifest(output_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(p for p in output_dir.rglob("*") if p.is_file()):
        artifacts.append({"relative_path": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    return {"created_at": utc_now(), "artifacts": artifacts}


def run_gpu_preannotation(
    *,
    video_path: str,
    output_dir: str,
    run_id: str,
    yolo_model_path: str,
    rfdetr_model_path: str = "",
    device: str = "cuda",
    confidence: float = 0.25,
    iou: float = 0.45,
    rfdetr_confidence: float = 0.25,
    yolo_imgsz: int = 640,
    tile_size: int = 640,
    tile_overlap: float = 0.2,
) -> Dict[str, Any]:
    started = time.time()
    output = Path(output_dir)
    pre_dir = output / "preannotations_gpu"
    pre_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or FRAME_W)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or FRAME_H)
    step = max(1, int(round(fps / 15.0)))
    tiles = generate_tiles(width, height, tile_size, tile_overlap)

    from rfdetr import RFDETRBase

    device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(yolo_model_path)
    model_kwargs = {"device": device}
    rfdetr = RFDETRBase(pretrain_weights=rfdetr_model_path, **model_kwargs) if rfdetr_model_path and Path(rfdetr_model_path).exists() else RFDETRBase(**model_kwargs)

    environment = {
        "created_at": utc_now(),
        "device_requested": device,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "rfdetr_version": "1.3.0",
        "yolo_checkpoint": yolo_model_path,
        "yolo_checkpoint_sha256": sha256_file(Path(yolo_model_path)) if Path(yolo_model_path).exists() else None,
        "rfdetr_checkpoint": rfdetr_model_path or "RFDETRBase_default",
        "rfdetr_checkpoint_sha256": sha256_file(Path(rfdetr_model_path)) if rfdetr_model_path and Path(rfdetr_model_path).exists() else None,
        "config": {
            "confidence": confidence,
            "iou": iou,
            "rfdetr_confidence": rfdetr_confidence,
            "yolo_imgsz": yolo_imgsz,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "sampling_fps": 15,
            "sequences": selected_sequences(),
        },
    }
    environment["config_hash"] = payload_hash(environment["config"])
    write_json(pre_dir / "environment_manifest.json", environment)

    frames = []
    rfdetr_global_rows = []
    rfdetr_tile_rows = []
    yolo_rows = []
    frame_status = []
    counters = Counter()
    expected = 0
    for seq in selected_sequences():
        expected += len(range(int(round(seq["start_sec"] * fps)), int(round(seq["end_sec"] * fps)), step))

    for seq in selected_sequences():
        frame_index = int(round(seq["start_sec"] * fps))
        end_frame = int(round(seq["end_sec"] * fps))
        while frame_index < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame_status.append({"frame_index": frame_index, "frame_id": None, "global_done": False, "tiles_done": False, "yolo_done": False, "artifacts_written": False, "validation_passed": False})
                frame_index += step
                continue
            timestamp = frame_index / fps
            frame_id = f"{seq['sequence_id']}_f{frame_index:08d}"
            frame_meta = {
                "frame_id": frame_id,
                "sequence_id": seq["sequence_id"],
                "split": seq["split"],
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp, 6),
                "review_status": "pending",
                "pseudo_label": True,
                "ground_truth": False,
            }
            frames.append(frame_meta)

            y_raw = yolo_predict(yolo, frame, yolo_imgsz, confidence, iou)
            y_frame_rows = [make_row(run_id=run_id, frame_id=frame_id, sequence_id=seq["sequence_id"], split=seq["split"], frame_index=frame_index, timestamp_sec=timestamp, source_model="yolo", source_pass="global", tile=None, raw=raw, width=width, height=height) for raw in y_raw]
            yolo_rows.extend(y_frame_rows)

            r_raw = rfdetr_predict(rfdetr, frame, rfdetr_confidence)
            r_global_frame = [make_row(run_id=run_id, frame_id=frame_id, sequence_id=seq["sequence_id"], split=seq["split"], frame_index=frame_index, timestamp_sec=timestamp, source_model="rfdetr", source_pass="global", tile=None, raw=raw, width=width, height=height) for raw in r_raw]
            rfdetr_global_rows.extend(r_global_frame)

            tile_frame_rows = []
            for tile in tiles:
                crop = frame[tile["y1"] : tile["y2"], tile["x1"] : tile["x2"]]
                if crop.size == 0:
                    continue
                for raw in rfdetr_predict(rfdetr, crop, rfdetr_confidence):
                    raw = dict(raw)
                    raw["bbox_tile_xyxy"] = [float(v) for v in raw["bbox_xyxy"]]
                    raw["bbox_xyxy"] = [
                        raw["bbox_xyxy"][0] + tile["x1"],
                        raw["bbox_xyxy"][1] + tile["y1"],
                        raw["bbox_xyxy"][2] + tile["x1"],
                        raw["bbox_xyxy"][3] + tile["y1"],
                    ]
                    tile_frame_rows.append(make_row(run_id=run_id, frame_id=frame_id, sequence_id=seq["sequence_id"], split=seq["split"], frame_index=frame_index, timestamp_sec=timestamp, source_model="rfdetr", source_pass="tile", tile=tile, raw=raw, width=width, height=height))
            rfdetr_tile_rows.extend(tile_frame_rows)

            frame_candidates = y_frame_rows + r_global_frame + tile_frame_rows
            if any(row["source_model"] == "rfdetr" and row["source_pass"] == "global" for row in frame_candidates):
                counters["rfdetr_global_frames"] += 1
            if any(row["source_model"] == "rfdetr" and row["source_pass"] == "tile" for row in frame_candidates):
                counters["rfdetr_tile_frames"] += 1
            if y_frame_rows:
                counters["yolo_frames"] += 1
            if not frame_candidates:
                counters["frames_without_candidate"] += 1
            frame_status.append({"frame_id": frame_id, "sequence_id": seq["sequence_id"], "split": seq["split"], "frame_index": frame_index, "timestamp_sec": round(timestamp, 6), "global_done": True, "tiles_done": True, "yolo_done": True, "artifacts_written": True, "validation_passed": True})
            if len(frames) % 25 == 0:
                print(json.dumps({"progress": "pe0a4r_gpu_preannotation", "frames": len(frames), "expected": expected}, sort_keys=True), flush=True)
            frame_index += step

    cap.release()
    fused_rows = fuse_agreement(yolo_rows + rfdetr_global_rows + rfdetr_tile_rows)
    counters["conflict_frames"] = len({row["frame_id"] for row in fused_rows if row["agreement"] != "both"})
    validation = validate_outputs(frames, fused_rows, expected, width, height)

    write_jsonl(pre_dir / "rfdetr_global_candidates.jsonl", rfdetr_global_rows)
    write_jsonl(pre_dir / "rfdetr_tile_candidates.jsonl", rfdetr_tile_rows)
    write_jsonl(pre_dir / "yolo_candidates.jsonl", yolo_rows)
    write_jsonl(pre_dir / "fused_candidates.jsonl", fused_rows)
    write_jsonl(pre_dir / "frame_status.jsonl", frame_status)
    write_json(pre_dir / "preannotation_gpu_validation.json", validation)
    runtime = {
        "started_at": environment["created_at"],
        "finished_at": utc_now(),
        "duration_sec": round(time.time() - started, 3),
        "frames_expected": expected,
        "frames_processed": len(frames),
        "rfdetr_global_candidates": len(rfdetr_global_rows),
        "rfdetr_tile_candidates": len(rfdetr_tile_rows),
        "yolo_candidates": len(yolo_rows),
        "frames_without_candidate": counters["frames_without_candidate"],
        "frames_with_conflict": counters["conflict_frames"],
    }
    write_json(pre_dir / "runtime_metrics.json", runtime)
    write_text(
        pre_dir / "preannotation_gpu_summary.md",
        "# PE-0A4R GPU Preannotation Summary\n\n"
        f"- validation: `{validation['status']}`\n"
        f"- frames_expected: `{expected}`\n"
        f"- frames_processed: `{len(frames)}`\n"
        f"- rfdetr_global_candidates: `{len(rfdetr_global_rows)}`\n"
        f"- rfdetr_tile_candidates: `{len(rfdetr_tile_rows)}`\n"
        f"- yolo_candidates: `{len(yolo_rows)}`\n"
        f"- frames_without_candidate: `{counters['frames_without_candidate']}`\n"
        f"- frames_with_conflict: `{counters['conflict_frames']}`\n",
    )
    write_json(pre_dir / "artifact_manifest.json", artifact_manifest(pre_dir))
    return {
        "status": "completed" if validation["status"] == "passed" else "failed",
        "preannotations_dir": str(pre_dir),
        "environment": environment,
        "runtime": runtime,
        "validation": validation,
        "summary": {
            "rfdetr_global_candidates": len(rfdetr_global_rows),
            "rfdetr_tile_candidates": len(rfdetr_tile_rows),
            "yolo_candidates": len(yolo_rows),
            "frames_without_candidate": counters["frames_without_candidate"],
            "frames_with_conflict": counters["conflict_frames"],
        },
    }
