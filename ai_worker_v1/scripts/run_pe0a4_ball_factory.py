#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
PE0A3_RUN = AI_WORKER_ROOT / "runs" / "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps"
PE0A3_SCRIPT_DIR = AI_WORKER_ROOT / "scripts"
if str(PE0A3_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PE0A3_SCRIPT_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from run_pe0a3_baseline import (  # noqa: E402
    BALL_NAMES,
    CANONICAL_BALL,
    CANONICAL_PERSON,
    bbox_iou,
    canonical_class,
    center_from_xyxy,
    clamp_box,
    dedupe_ball_candidates,
    generate_tiles,
    init_rfdetr,
    load_json,
    make_detection_row,
    maybe_write_parquet,
    polygon_points,
    rfdetr_predict,
    sha256_file,
    stable_id,
    video_meta,
    write_json,
    write_jsonl,
    write_text,
    yolo_predict,
)


PHASE = "PE-0A4"
CAPTURE_ID = "ccb_ec8836978221c786ed55a0ab"
CALIBRATION_ID = "vc_a90e53754cb6083389782e25"
VIDEO_HASH = "885c106cbf89a61b3fd38e9de015aad10decb9f6b47487843711e21115b4f2f9"
PE0A3_RUN_ID = "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps"
MATCH_ID = "test_match_2026-07-15T02-16-23-996Z"
CLASS_BALL_ID = 1


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "preannotations": run_dir / "preannotations",
        "frames": run_dir / "dataset" / "frames",
        "crops": run_dir / "dataset" / "crops",
        "annotations": run_dir / "dataset" / "annotations",
        "manifests": run_dir / "dataset" / "manifests",
        "review": run_dir / "review",
        "overlays": run_dir / "overlays",
        "metrics": run_dir / "metrics",
        "training": run_dir / "training",
        "evaluation": run_dir / "evaluation",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def artifact_manifest(run_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == ".DS_Store":
            continue
        artifacts.append(
            {
                "relative_path": str(path.relative_to(run_dir)),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "phase": PHASE,
                "created_at": utc_now(),
            }
        )
    return {"phase": PHASE, "run_dir": str(run_dir), "artifacts": artifacts}


def inspect_yolo_checkpoint(model_path: Path, confidence: float) -> Dict[str, Any]:
    try:
        model = YOLO(str(model_path))
        names = {int(k): str(v) for k, v in (getattr(model, "names", {}) or {}).items()}
        lower = {v.lower().replace("_", " ") for v in names.values()}
        if "person" not in lower and BALL_NAMES.intersection(lower):
            reason = "checkpoint_ball_only"
        elif "person" not in lower:
            reason = "checkpoint_without_person"
        else:
            yolo_rows = read_jsonl(PE0A3_RUN / "detections" / "yolo_raw.jsonl")
            raw_person = sum(1 for row in yolo_rows if str(row.get("raw_class_name", "")).lower() == "person")
            canonical_person = sum(1 for row in yolo_rows if row.get("canonical_class") == CANONICAL_PERSON)
            if raw_person > 0 and canonical_person == 0:
                reason = "class_mapping_error"
            elif confidence > 0.5:
                reason = "confidence_filter_error"
            else:
                reason = "valid_zero_detections"
        return {
            "status": "ok",
            "checkpoint": str(model_path),
            "sha256": sha256_file(model_path),
            "names": names,
            "person_class_present": "person" in lower,
            "ball_class_present": bool(BALL_NAMES.intersection(lower)),
            "classification": reason,
        }
    except Exception as exc:
        return {
            "status": "error",
            "checkpoint": str(model_path),
            "classification": "inference_error",
            "error": str(exc),
        }


def baseline_sanity(run_dir: Path, yolo_model_path: Path) -> Dict[str, Any]:
    pe3_counts = load_json(PE0A3_RUN / "metrics" / "detection_counts.json")
    canonical = read_jsonl(PE0A3_RUN / "detections" / "canonical_detections.jsonl")
    person_spatial = Counter(row["spatial_status"] for row in canonical if row.get("canonical_class") == CANONICAL_PERSON)
    ball_spatial = Counter(row["spatial_status"] for row in canonical if row.get("canonical_class") == CANONICAL_BALL)
    all_spatial = Counter(row["spatial_status"] for row in canonical)
    report = {
        "phase": PHASE,
        "pe0a3_run_id": PE0A3_RUN_ID,
        "yolo_person_zero": inspect_yolo_checkpoint(yolo_model_path, 0.25),
        "spatial_metrics_inconsistency": {
            "historical_spatial_counts": pe3_counts.get("spatial_counts", {}),
            "historical_sum": sum(pe3_counts.get("spatial_counts", {}).values()),
            "rfdetr_person_detections": pe3_counts.get("rfdetr_person_detections"),
            "determination": "historical PE-0A3 spatial_counts counted all canonical pseudoannotations, not only RF-DETR persons",
            "person_spatial_counts": dict(person_spatial),
            "ball_spatial_counts": dict(ball_spatial),
            "all_annotation_counts": dict(all_spatial),
        },
    }
    write_json(run_dir / "baseline_sanity_report.json", report)
    md = [
        "# PE-0A4 Baseline Sanity Report",
        "",
        f"- YOLO person zero reason: `{report['yolo_person_zero']['classification']}`",
        f"- YOLO names: `{report['yolo_person_zero'].get('names')}`",
        "- Spatial metric fix: new reports split person, ball, and all annotations.",
        f"- person_spatial_counts: `{dict(person_spatial)}`",
        f"- ball_spatial_counts: `{dict(ball_spatial)}`",
        f"- all_annotation_counts: `{dict(all_spatial)}`",
        "",
    ]
    write_text(run_dir / "baseline_sanity_report.md", "\n".join(md))
    return report


def select_sequences(fps: float) -> List[Dict[str, Any]]:
    # Fixed from PE-0A3 evidence, deliberately non-overlapping and mixed.
    specs = [
        ("seq_pos_01", 82.525, 86.525, "positive", "RF-DETR burst at segment start", ["rfdetr"]),
        ("seq_neg_01", 88.525, 92.525, "negative_or_ambiguous", "low agreement window; review hard negatives", ["candidate_absence_or_false_candidates"]),
        ("seq_hard_01", 94.525, 98.525, "difficult", "YOLO-only ball window; likely small/blurred ball", ["yolo"]),
        ("seq_pos_02", 102.525, 106.525, "positive", "agreement-adjacent candidate burst", ["rfdetr", "yolo"]),
        ("seq_pos_03", 110.525, 114.525, "positive", "RF-DETR-only candidate burst near reference frame", ["rfdetr"]),
        ("seq_hard_02", 122.525, 126.525, "difficult", "post-reference disagreement and possible occlusion", ["rfdetr", "yolo"]),
        ("seq_neg_02", 128.525, 132.525, "negative_or_ambiguous", "ambiguous interval for false candidate mining", ["hard_negative"]),
        ("seq_pos_04", 136.525, 140.525, "positive", "late RF-DETR/Yolo candidate sequence", ["rfdetr", "yolo"]),
    ]
    splits = {
        "seq_pos_01": "train",
        "seq_neg_01": "train",
        "seq_hard_01": "train",
        "seq_pos_02": "train",
        "seq_pos_03": "train",
        "seq_hard_02": "train",
        "seq_neg_02": "valid",
        "seq_pos_04": "within_video_test_v0",
    }
    rows = []
    for sequence_id, start, end, kind, reason, sources in specs:
        rows.append(
            {
                "sequence_id": sequence_id,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": round(end - start, 3),
                "original_frame_start": int(round(start * fps)),
                "original_frame_end": int(round(end * fps)) - 1,
                "fps_original": fps,
                "sampling_fps": 15,
                "sequence_type": kind,
                "motivo": reason,
                "candidate_sources": sources,
                "assigned_split": splits[sequence_id],
                "review_status": "pending",
            }
        )
    return rows


def leakage_report(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    overlaps = []
    seen_ids = set()
    for idx, left in enumerate(sequences):
        if left["sequence_id"] in seen_ids:
            overlaps.append({"type": "duplicate_sequence_id", "sequence_id": left["sequence_id"]})
        seen_ids.add(left["sequence_id"])
        for right in sequences[idx + 1 :]:
            if not (left["end_sec"] <= right["start_sec"] or right["end_sec"] <= left["start_sec"]):
                overlaps.append({"type": "temporal_overlap", "left": left["sequence_id"], "right": right["sequence_id"]})
    return {
        "status": "passed" if not overlaps else "failed",
        "overlaps": overlaps,
        "rule": "complete sequences are assigned to exactly one split; no crops/tiles cross split boundaries",
        "split_counts": dict(Counter(seq["assigned_split"] for seq in sequences)),
    }


def crop_box_512(center: Tuple[float, float], width: int, height: int, jitter: Tuple[int, int] = (0, 0)) -> Tuple[int, int, int, int]:
    cx = int(round(center[0] + jitter[0]))
    cy = int(round(center[1] + jitter[1]))
    half = 256
    x1 = max(0, min(width - 512, cx - half)) if width >= 512 else 0
    y1 = max(0, min(height - 512, cy - half)) if height >= 512 else 0
    return x1, y1, min(width, x1 + 512), min(height, y1 + 512)


def bbox_to_crop(box: List[float], crop: Tuple[int, int, int, int]) -> Optional[List[float]]:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop
    nx1 = max(0.0, x1 - cx1)
    ny1 = max(0.0, y1 - cy1)
    nx2 = min(float(cx2 - cx1), x2 - cx1)
    ny2 = min(float(cy2 - cy1), y2 - cy1)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return [nx1, ny1, nx2, ny2]


def agreement_for_candidate(candidate: Dict[str, Any], yolo_ball: List[Dict[str, Any]], rfdetr_ball: List[Dict[str, Any]]) -> str:
    other_rows = rfdetr_ball if candidate["source_model"] == "yolo" else yolo_ball
    for other in other_rows:
        if other["frame_index"] != candidate["frame_index"]:
            continue
        distance = math.hypot(candidate["center"][0] - other["center"][0], candidate["center"][1] - other["center"][1])
        if bbox_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) >= 0.25 or distance <= 25:
            return "both"
    return f"{candidate['source_model']}_only"


def draw_preannotation(frame: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    out = frame.copy()
    for row in candidates:
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        color = (0, 0, 255) if row["source_model"] == "rfdetr" else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{row['source_model']}:{row['source_pass']} {row['confidence']:.2f}"
        cv2.putText(out, label, (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def write_review_assets(run_dir: Path, frames_manifest: List[Dict[str, Any]], frame_candidates: Dict[str, List[Dict[str, Any]]]) -> None:
    review_dir = run_dir / "review"
    queue = []
    for frame in frames_manifest:
        candidates = frame_candidates.get(frame["frame_id"], [])
        has_conflict = len({c["source_model"] for c in candidates}) > 1 and not any(c["agreement"] == "both" for c in candidates)
        priority = "P0" if candidates and (has_conflict or any(c["source_pass"] == "tile" for c in candidates)) else "P1"
        if not candidates:
            priority = "P2"
        queue.append(
            {
                "frame_id": frame["frame_id"],
                "sequence_id": frame["sequence_id"],
                "split": frame["split"],
                "timestamp_sec": frame["timestamp_sec"],
                "frame_image": frame["image_relpath"],
                "crop_image": frame.get("review_crop_relpath"),
                "candidate_count": len(candidates),
                "priority": priority,
                "status": "pending",
            }
        )
    write_json(review_dir / "review_queue.json", {"items": queue})
    write_json(review_dir / "review_progress.json", {"total_frames": len(queue), "pending": len(queue), "reviewed_ball": 0, "reviewed_no_ball": 0, "reviewed_uncertain": 0})
    write_jsonl(review_dir / "review_decisions.jsonl", [])
    write_jsonl(review_dir / "review_audit_log.jsonl", [])
    write_json(review_dir / "reviewed_dataset_manifest.json", {"status": "pending_human_review", "ground_truth": False, "frames": 0})
    write_json(review_dir / "reviewed_annotations_coco.json", {"images": [], "annotations": [], "categories": [{"id": CLASS_BALL_ID, "name": "ball"}]})
    write_text(
        review_dir / "README.md",
        "# Ball Review\n\nRun `python ai_worker_v1/review_tools/ball_review/server.py --run-dir <this-run>` and review every pending frame.\n",
    )


def run_factory(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.run_id or f"pe0a4_ball_specialist_v0_{utc_now_compact()}"
    run_dir = AI_WORKER_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dirs = ensure_dirs(run_dir)
    started = time.time()

    source_video = PE0A3_RUN / "source_video.mp4"
    if not source_video.exists():
        raise FileNotFoundError(f"PE-0A3 source_video not found: {source_video}")
    source_meta = video_meta(source_video)
    if source_meta["sha256"] != VIDEO_HASH:
        raise ValueError(f"source video hash mismatch: {source_meta['sha256']}")

    profile_path = Path(args.profile)
    profile = load_json(profile_path)
    if profile.get("calibration_id") != CALIBRATION_ID:
        raise ValueError("calibration_id mismatch")
    roi_polygon = polygon_points(profile)
    yolo_model_path = Path(args.yolo_model)
    rfdetr_model_path = Path(args.rfdetr_model)

    sanity = baseline_sanity(run_dir, yolo_model_path)
    sequences = select_sequences(source_meta["fps"])
    write_json(run_dir / "sequences_manifest.json", {"sequences": sequences})
    split_manifest = {
        "splits": {
            "train": [seq["sequence_id"] for seq in sequences if seq["assigned_split"] == "train"],
            "valid": [seq["sequence_id"] for seq in sequences if seq["assigned_split"] == "valid"],
            "within_video_test_v0": [seq["sequence_id"] for seq in sequences if seq["assigned_split"] == "within_video_test_v0"],
        },
        "rule": "split by whole sequence; no temporal leakage across train/valid/within_video_test_v0",
    }
    write_json(run_dir / "split_manifest.json", split_manifest)
    leakage = leakage_report(sequences)
    write_json(run_dir / "leakage_report.json", leakage)
    write_text(
        run_dir / "split_summary.md",
        "# PE-0A4 Split Summary\n\n"
        f"- train sequences: `{len(split_manifest['splits']['train'])}`\n"
        f"- valid sequences: `{len(split_manifest['splits']['valid'])}`\n"
        f"- within_video_test_v0 sequences: `{len(split_manifest['splits']['within_video_test_v0'])}`\n"
        f"- leakage: `{leakage['status']}`\n",
    )

    import torch
    import ultralytics

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(yolo_model_path))
    rfdetr = init_rfdetr(str(rfdetr_model_path), device=device)
    model_manifest = {
        "yolo": {
            "checkpoint": str(yolo_model_path),
            "sha256": sha256_file(yolo_model_path),
            "ultralytics_version": ultralytics.__version__,
            "confidence": args.confidence,
            "iou": args.iou,
            "device": device,
        },
        "rfdetr": {
            "class": "RFDETRBase",
            "checkpoint": str(rfdetr_model_path),
            "sha256": sha256_file(rfdetr_model_path),
            "confidence": args.rfdetr_confidence,
            "device": device,
        },
        "training_target": {
            "class": "RFDETRSmall",
            "resolution": "512x512",
            "status": "not_started_pending_human_review",
        },
    }
    write_json(run_dir / "model_manifest.json", model_manifest)

    width, height = int(source_meta["width"]), int(source_meta["height"])
    tiles = generate_tiles(width, height, args.tile_size, args.tile_overlap, roi_polygon)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError("could not open source video")

    frames_manifest: List[Dict[str, Any]] = []
    all_candidates: List[Dict[str, Any]] = []
    frame_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    runtime_rows = []
    crop_rows = []
    coco_images = []
    coco_annotations = []
    ann_id = 1
    image_id = 1

    for seq in sequences:
        seq_id = seq["sequence_id"]
        process_every = max(1, int(round(source_meta["fps"] / 15.0)))
        start_frame = int(round(seq["start_sec"] * source_meta["fps"]))
        end_frame = int(round(seq["end_sec"] * source_meta["fps"]))
        frame_index = start_frame
        seq_counter = 0
        while frame_index < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / source_meta["fps"]
            frame_id = f"{seq_id}_f{frame_index:08d}"
            image_name = f"{frame_id}.jpg"
            image_path = dirs["frames"] / seq["assigned_split"] / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)

            t0 = time.time()
            y_raw = yolo_predict(yolo, frame, args.yolo_imgsz, args.confidence, args.iou)
            r_global = rfdetr_predict(rfdetr, frame, args.rfdetr_confidence)
            y_rows = [
                make_detection_row(
                    run_id=run_id,
                    frame_index=frame_index,
                    processed_index=seq_counter,
                    timestamp_sec=timestamp,
                    model_name="yolo",
                    source_pass="global",
                    tile_id=None,
                    raw=raw,
                    width=width,
                    height=height,
                    roi_polygon=roi_polygon,
                )
                for raw in y_raw
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]
            r_rows = [
                make_detection_row(
                    run_id=run_id,
                    frame_index=frame_index,
                    processed_index=seq_counter,
                    timestamp_sec=timestamp,
                    model_name="rfdetr",
                    source_pass="global",
                    tile_id=None,
                    raw=raw,
                    width=width,
                    height=height,
                    roi_polygon=roi_polygon,
                )
                for raw in r_global
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]

            for tile in tiles:
                crop = frame[tile["y1"] : tile["y2"], tile["x1"] : tile["x2"]]
                if crop.size == 0:
                    continue
                for raw in yolo_predict(yolo, crop, args.tile_size, args.confidence, args.iou):
                    if canonical_class(str(raw.get("class_name", ""))) != CANONICAL_BALL:
                        continue
                    raw = dict(raw)
                    raw["bbox_xyxy"] = [raw["bbox_xyxy"][0] + tile["x1"], raw["bbox_xyxy"][1] + tile["y1"], raw["bbox_xyxy"][2] + tile["x1"], raw["bbox_xyxy"][3] + tile["y1"]]
                    y_rows.append(
                        make_detection_row(run_id=run_id, frame_index=frame_index, processed_index=seq_counter, timestamp_sec=timestamp, model_name="yolo", source_pass="tile", tile_id=tile["tile_id"], raw=raw, width=width, height=height, roi_polygon=roi_polygon)
                    )
                for raw in rfdetr_predict(rfdetr, crop, args.rfdetr_confidence):
                    if canonical_class(str(raw.get("class_name", ""))) != CANONICAL_BALL:
                        continue
                    raw = dict(raw)
                    raw["bbox_xyxy"] = [raw["bbox_xyxy"][0] + tile["x1"], raw["bbox_xyxy"][1] + tile["y1"], raw["bbox_xyxy"][2] + tile["x1"], raw["bbox_xyxy"][3] + tile["y1"]]
                    r_rows.append(
                        make_detection_row(run_id=run_id, frame_index=frame_index, processed_index=seq_counter, timestamp_sec=timestamp, model_name="rfdetr", source_pass="tile", tile_id=tile["tile_id"], raw=raw, width=width, height=height, roi_polygon=roi_polygon)
                    )

            y_rows = dedupe_ball_candidates(y_rows)
            r_rows = dedupe_ball_candidates(r_rows)
            candidates = []
            for row in y_rows + r_rows:
                row = dict(row)
                row["frame_id"] = frame_id
                row["sequence_id"] = seq_id
                row["split"] = seq["assigned_split"]
                row["agreement"] = agreement_for_candidate(row, y_rows, r_rows)
                row["candidate_status"] = "unreviewed"
                row["pseudo_label"] = True
                row["ground_truth"] = False
                candidates.append(row)
            all_candidates.extend(candidates)
            frame_candidates[frame_id].extend(candidates)

            crop_center = center_from_xyxy(candidates[0]["bbox_xyxy"]) if candidates else (width / 2.0, height / 2.0)
            review_crop = crop_box_512(crop_center, width, height)
            crop_img = frame[review_crop[1] : review_crop[3], review_crop[0] : review_crop[2]]
            crop_rel = Path("dataset") / "crops" / seq["assigned_split"] / f"{frame_id}_review.jpg"
            crop_path = run_dir / crop_rel
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop_img)

            # One candidate training crop per frame until human review decides final label.
            crop_rows.append(
                {
                    "frame_id": frame_id,
                    "sequence_id": seq_id,
                    "split": seq["assigned_split"],
                    "crop_relpath": str(crop_rel),
                    "crop_xyxy_in_frame": list(review_crop),
                    "candidate_count": len(candidates),
                    "crop_status": "pending_human_review",
                    "intended_role": "positive_candidate" if candidates else "hard_negative_candidate",
                }
            )
            coco_images.append({"id": image_id, "file_name": str(crop_rel), "width": crop_img.shape[1], "height": crop_img.shape[0], "frame_id": frame_id, "split": seq["assigned_split"]})
            for candidate in candidates:
                rel_box = bbox_to_crop(candidate["bbox_xyxy"], review_crop)
                if rel_box is None:
                    continue
                x1, y1, x2, y2 = rel_box
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": CLASS_BALL_ID,
                        "bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)],
                        "area": round((x2 - x1) * (y2 - y1), 4),
                        "iscrowd": 0,
                        "pseudo_label": True,
                        "ground_truth": False,
                        "candidate_status": "unreviewed",
                        "source_model": candidate["source_model"],
                        "source_pass": candidate["source_pass"],
                    }
                )
                ann_id += 1
            image_id += 1

            overlay = draw_preannotation(frame, candidates)
            overlay_rel = Path("overlays") / f"{frame_id}.jpg"
            overlay_path = run_dir / overlay_rel
            cv2.imwrite(str(overlay_path), overlay)

            frames_manifest.append(
                {
                    "frame_id": frame_id,
                    "sequence_id": seq_id,
                    "split": seq["assigned_split"],
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp, 6),
                    "image_relpath": str(Path("dataset") / "frames" / seq["assigned_split"] / image_name),
                    "review_crop_relpath": str(crop_rel),
                    "overlay_relpath": str(overlay_rel),
                    "candidate_count": len(candidates),
                    "status": "pending",
                }
            )
            runtime_rows.append({"frame_id": frame_id, "frame_index": frame_index, "timestamp_sec": timestamp, "runtime_sec": time.time() - t0, "candidate_count": len(candidates)})
            seq_counter += 1
            frame_index += process_every
            if args.max_frames and len(frames_manifest) >= args.max_frames:
                break
        if args.max_frames and len(frames_manifest) >= args.max_frames:
            break
    cap.release()

    write_json(run_dir / "source_manifest.json", {"source_video": str(source_video), "video_hash": source_meta["sha256"], "capture_id": CAPTURE_ID, "calibration_id": CALIBRATION_ID, "match_id": MATCH_ID, "fps": source_meta["fps"], "resolution": [width, height]})
    write_json(run_dir / "dataset" / "manifests" / "frames_manifest.json", {"frames": frames_manifest})
    write_json(run_dir / "dataset" / "manifests" / "crops_manifest.json", {"crops": crop_rows})
    write_jsonl(dirs["preannotations"] / "ball_candidates.jsonl", all_candidates)
    maybe_write_parquet(dirs["preannotations"] / "ball_candidates.parquet", all_candidates)
    write_json(dirs["annotations"] / "preannotations_coco.json", {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": CLASS_BALL_ID, "name": "ball"}], "ground_truth": False, "pseudo_label": True})
    write_review_assets(run_dir, frames_manifest, frame_candidates)

    split_frame_counts = Counter(frame["split"] for frame in frames_manifest)
    candidate_frames = {row["frame_id"] for row in all_candidates}
    summary = {
        "phase": PHASE,
        "run_id": run_id,
        "status": "ready_for_human_review",
        "frames": len(frames_manifest),
        "sequences": len(sequences),
        "candidate_frames": len(candidate_frames),
        "positive_candidate_frames": len(candidate_frames),
        "negative_candidate_frames": len(frames_manifest) - len(candidate_frames),
        "uncertain": 0,
        "candidate_count": len(all_candidates),
        "split_frame_counts": dict(split_frame_counts),
        "runtime_sec": time.time() - started,
        "review_required": True,
        "training_status": "blocked_pending_human_review",
    }
    write_json(run_dir / "metrics" / "factory_summary.json", summary)
    write_json(run_dir / "metrics" / "runtime_metrics.json", {"per_frame": runtime_rows, "total_runtime_sec": summary["runtime_sec"]})
    write_json(run_dir / "dataset" / "manifests" / "review_gate.json", {"training_allowed": False, "blocked_by": ["pending_human_review"], "required_phrase": "REVIEW BALL DATASET COMPLETED"})

    card = [
        "# OneFrame Ball Dataset v0 Candidate",
        "",
        "Status: pending human review. Do not train from this dataset yet.",
        f"Frames: {summary['frames']}",
        f"Candidate frames: {summary['candidate_frames']}",
        f"Negative candidate frames: {summary['negative_candidate_frames']}",
        "Ground truth: false until review decisions are completed.",
        "",
    ]
    write_text(run_dir / "dataset" / "DATASET_CARD.md", "\n".join(card))
    report = [
        "# PE-0A4 Ball Data Review Report",
        "",
        f"- run_id: `{run_id}`",
        f"- status: `{summary['status']}`",
        f"- sequences: `{summary['sequences']}`",
        f"- frames: `{summary['frames']}`",
        f"- candidate frames: `{summary['candidate_frames']}`",
        f"- negative candidate frames: `{summary['negative_candidate_frames']}`",
        f"- candidates: `{summary['candidate_count']}`",
        f"- yolo_person_zero_reason: `{sanity['yolo_person_zero']['classification']}`",
        f"- spatial metrics fix: `{sanity['spatial_metrics_inconsistency']['determination']}`",
        "",
        "Training is blocked until the operator completes every pending review frame.",
    ]
    write_text(run_dir / "PE0A4_FINAL_REPORT.md", "\n".join(report) + "\n")
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {"run_dir": str(run_dir), **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PE-0A4 ball data factory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile", default="/tmp/oneframe_pe0a2c_capture/session_20260715T013505Z/materialized/pe0a2c_real_capture_ccb_ec8836978221c786ed55a0ab/video_calibration.json")
    parser.add_argument("--yolo-model", default=str(AI_WORKER_ROOT / "src" / "oneframe_v3_best.pt"))
    parser.add_argument("--rfdetr-model", default=str(AI_WORKER_ROOT / "rfdetr_cache" / "rf-detr-base.pth"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--rfdetr-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_factory(parse_args())
