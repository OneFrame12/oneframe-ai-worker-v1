#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
AI_WORKER_SRC = AI_WORKER_ROOT / "src"
for path in (str(AI_WORKER_ROOT / "scripts"), str(AI_WORKER_SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from run_pe0a3_baseline import (  # noqa: E402
    CANONICAL_BALL,
    bbox_iou,
    canonical_class,
    center_from_xyxy,
    clamp_box,
    dedupe_ball_candidates,
    generate_tiles,
    init_rfdetr,
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


PHASE = "PE-0A4R"
ORIGINAL_RUN_ID = "pe0a3r_full_field_ball_data_20260715T0345Z"
ORIGINAL_RUN = AI_WORKER_ROOT / "runs" / ORIGINAL_RUN_ID
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_REVIEW_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"
SOURCE_RUN = AI_WORKER_ROOT / "runs" / "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps"
SOURCE_VIDEO = SOURCE_RUN / "source_video.mp4"
CLASS_BALL_ID = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "candidate_windows": run_dir / "candidate_windows",
        "preannotations": run_dir / "preannotations",
        "frames": run_dir / "dataset" / "frames",
        "crops": run_dir / "dataset" / "crops",
        "manifests": run_dir / "dataset" / "manifests",
        "overlays": run_dir / "overlays",
        "review": run_dir / "review",
        "split_repair": run_dir / "split_repair",
        "tests": run_dir / "tests",
        "metrics": run_dir / "metrics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def review_hash_from_manifest(manifest: Dict[str, Any]) -> str:
    return str(manifest.get("review_hash", ""))


def validate_original_freeze() -> Dict[str, Any]:
    freeze_dir = ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID
    manifest_path = freeze_dir / "review_frozen_manifest.json"
    root_manifest_path = ORIGINAL_RUN / "review" / "review_frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    root_manifest = read_json(root_manifest_path) if root_manifest_path.exists() else {}
    errors = []
    if manifest.get("freeze_id") != ORIGINAL_FREEZE_ID:
        errors.append("freeze_id_mismatch")
    if review_hash_from_manifest(manifest) != ORIGINAL_REVIEW_HASH:
        errors.append("review_hash_mismatch")
    if root_manifest and review_hash_from_manifest(root_manifest) != ORIGINAL_REVIEW_HASH:
        errors.append("root_review_hash_mismatch")
    for item in manifest.get("frozen_files", []):
        path = Path(item["path"])
        if not path.exists():
            path = freeze_dir / item["file"]
        if not path.exists():
            errors.append(f"missing_frozen_file:{item['file']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"frozen_file_hash_mismatch:{item['file']}")
    if errors:
        raise RuntimeError(f"Original frozen review is not intact: {errors}")
    return {"status": "passed", "freeze_id": ORIGINAL_FREEZE_ID, "review_hash": ORIGINAL_REVIEW_HASH}


def latest_original_statuses() -> Dict[str, Any]:
    queue = read_json(ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID / "review_queue.json")["items"]
    decisions = read_jsonl(ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID / "review_decisions.jsonl")
    by_sequence: Dict[str, Counter] = defaultdict(Counter)
    for item in queue:
        by_sequence[item["sequence_id"]][item.get("status", "pending")] += 1
    return {"queue_items": queue, "decisions_total": len(decisions), "by_sequence": {seq: dict(counts) for seq, counts in by_sequence.items()}}


def write_split_repair(run_dir: Path, original_stats: Dict[str, Any]) -> Dict[str, Any]:
    sequence_stats = original_stats["by_sequence"]
    proposal = {
        "phase": PHASE,
        "created_at": utc_now(),
        "source_freeze_id": ORIGINAL_FREEZE_ID,
        "source_review_hash": ORIGINAL_REVIEW_HASH,
        "policy": "sequence_level_split_repair; reviewed_uncertain excluded from train/valid/test metrics",
        "sequences": [
            {
                "sequence_id": "dense_normal_passes_01",
                "previous_split": "train",
                "new_split": "train",
                "decision": "keep_positive_sequence_in_train",
                "reason": "contains original 53 reviewed_ball frames",
                "counts": sequence_stats.get("dense_normal_passes_01", {}),
            },
            {
                "sequence_id": "dense_goal_approach_01",
                "previous_split": "within_video_test_v0",
                "new_split": "train",
                "decision": "move_negative_sequence_to_train",
                "reason": "contains original 33 reviewed_no_ball hard negatives; uncertain frames remain excluded",
                "counts": sequence_stats.get("dense_goal_approach_01", {}),
            },
            {
                "sequence_id": "dense_feet_cluster_01",
                "previous_split": "valid",
                "new_split": "excluded_uncertain_pool",
                "decision": "exclude_uncertain_only_sequence",
                "reason": "contains only reviewed_uncertain frames; uncertain cannot become negatives",
                "counts": sequence_stats.get("dense_feet_cluster_01", {}),
            },
        ],
    }
    write_json(run_dir / "split_repair" / "split_repair_proposal.json", proposal)
    lines = [
        "# PE-0A4R Split Repair Report",
        "",
        f"- source_freeze_id: `{ORIGINAL_FREEZE_ID}`",
        f"- source_review_hash: `{ORIGINAL_REVIEW_HASH}`",
        "- policy: sequence-level split repair; uncertain labels are excluded from model splits",
        "",
        "| sequence | previous | new | decision | counts |",
        "|---|---|---|---|---|",
    ]
    for row in proposal["sequences"]:
        lines.append(f"| {row['sequence_id']} | {row['previous_split']} | {row['new_split']} | {row['decision']} | `{row['counts']}` |")
    write_text(run_dir / "split_repair" / "split_repair_report.md", "\n".join(lines) + "\n")
    return proposal


def candidate_window_specs(duration: float) -> List[Dict[str, Any]]:
    raw = [
        ("cw_01_early_build", 18.0, 5.5, "early buildup, different from original positives"),
        ("cw_02_mid_pass", 31.0, 5.5, "midfield pass/control context"),
        ("cw_03_transition", 52.0, 5.5, "transition with open play"),
        ("cw_04_feet_cluster", 70.0, 5.5, "feet/cluster hard context candidate"),
        ("cw_05_wide_play", 82.0, 5.5, "wide play candidate, separated from original train"),
        ("cw_06_between_originals", 119.0, 5.5, "between original valid and test windows"),
        ("cw_07_post_goal_side", 154.0, 5.5, "post-original mixed validation candidate"),
        ("cw_08_late_midfield", 171.0, 5.5, "late midfield mixed candidate"),
        ("cw_09_late_attack", 190.0, 5.5, "late attack test candidate"),
        ("cw_10_final_phase", 204.0, 5.5, "final phase test candidate"),
    ]
    output = []
    for window_id, start, window_duration, reason in raw:
        if start + window_duration >= duration:
            continue
        output.append({"window_id": window_id, "start_sec": start, "end_sec": start + window_duration, "duration_sec": window_duration, "rationale": reason})
    return output


def temporal_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def write_preview(video_path: Path, out_path: Path, start_sec: float, end_sec: float, source_fps: float, preview_fps: int = 5) -> None:
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), preview_fps, (width, height))
    step = max(1, int(round(source_fps / preview_fps)))
    frame_index = int(round(start_sec * source_fps))
    end_frame = int(round(end_sec * source_fps))
    while frame_index < end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        cv2.putText(frame, f"{out_path.stem} f={frame_index} t={frame_index/source_fps:.2f}s", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        writer.write(frame)
        frame_index += step
    writer.release()
    cap.release()


def make_candidate_contact_sheet(video_path: Path, run_dir: Path, windows: List[Dict[str, Any]], source_fps: float) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    thumbs = []
    annotated = []
    for window in windows:
        frame_indices = [
            int(round(window["start_sec"] * source_fps)),
            int(round(((window["start_sec"] + window["end_sec"]) / 2.0) * source_fps)),
            int(round((window["end_sec"] - 0.1) * source_fps)),
        ]
        samples = []
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            thumb = cv2.resize(frame, (320, 180))
            cv2.putText(thumb, f"{window['window_id']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(thumb, f"t={frame_index/source_fps:.1f}s", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            thumbs.append(thumb)
            samples.append({"frame_index": frame_index, "timestamp_sec": round(frame_index / source_fps, 3)})
        window["samples"] = samples
        preview_rel = Path("candidate_windows") / f"{window['window_id']}_preview_5fps.mp4"
        write_preview(video_path, run_dir / preview_rel, window["start_sec"], window["end_sec"], source_fps)
        window["preview_mp4"] = str(preview_rel)
        annotated.append(window)
    cap.release()
    if thumbs:
        cols = 3
        rows = int(math.ceil(len(thumbs) / cols))
        sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
        for idx, thumb in enumerate(thumbs):
            row, col = divmod(idx, cols)
            sheet[row * 180 : row * 180 + 180, col * 320 : col * 320 + 320] = thumb
        cv2.imwrite(str(run_dir / "candidate_windows" / "candidate_windows_contact_sheet.png"), sheet)
    write_json(run_dir / "candidate_windows" / "candidate_windows.json", {"windows": annotated})
    return annotated


def selected_sequences() -> List[Dict[str, Any]]:
    return [
        {
            "sequence_id": "test_ball_mixed_01",
            "split": "within_video_test_v0",
            "start_sec": 204.0,
            "end_sec": 209.0,
            "reason": "test first: late distinct context with mixed ball/no-ball frames; not used for parameter selection",
            "source_window_id": "cw_10_final_phase",
            "review_order": 1,
        },
        {
            "sequence_id": "valid_ball_mixed_01",
            "split": "valid",
            "start_sec": 154.0,
            "end_sec": 159.0,
            "reason": "validation mixed sequence after original test window, separated from train",
            "source_window_id": "cw_07_post_goal_side",
            "review_order": 2,
        },
        {
            "sequence_id": "train_ball_hard_03",
            "split": "train",
            "start_sec": 70.0,
            "end_sec": 75.0,
            "reason": "hard train context near feet/players/partial occlusion",
            "source_window_id": "cw_04_feet_cluster",
            "review_order": 3,
        },
        {
            "sequence_id": "train_ball_context_02",
            "split": "train",
            "start_sec": 31.0,
            "end_sec": 36.0,
            "reason": "normal play with pass/control context different from original train sequence",
            "source_window_id": "cw_02_mid_pass",
            "review_order": 4,
        },
    ]


def selection_checks(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    original = [
        ("dense_normal_passes_01", 94.525, 99.525),
        ("dense_feet_cluster_01", 108.525, 113.525),
        ("dense_goal_approach_01", 132.525, 137.525),
    ]
    overlaps = []
    too_close = []
    for seq in sequences:
        for name, start, end in original:
            overlap = temporal_overlap(seq["start_sec"], seq["end_sec"], start, end)
            if overlap > 0:
                overlaps.append({"sequence_id": seq["sequence_id"], "other": name, "overlap_sec": overlap})
    sorted_seq = sorted(sequences, key=lambda row: row["start_sec"])
    for left, right in zip(sorted_seq, sorted_seq[1:]):
        gap = right["start_sec"] - left["end_sec"]
        if gap < 15.0:
            too_close.append({"left": left["sequence_id"], "right": right["sequence_id"], "gap_sec": round(gap, 3)})
    return {"status": "passed" if not overlaps and not too_close else "failed", "overlaps": overlaps, "too_close": too_close}


def crop_box_512(center: Tuple[float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(max(0, width - 512), int(round(center[0] - 256))))
    y1 = max(0, min(max(0, height - 512), int(round(center[1] - 256))))
    return x1, y1, min(width, x1 + 512), min(height, y1 + 512)


def agreement_for_candidate(candidate: Dict[str, Any], yolo_rows: List[Dict[str, Any]], rfdetr_rows: List[Dict[str, Any]]) -> str:
    others = rfdetr_rows if candidate["source_model"] == "yolo" else yolo_rows
    for other in others:
        if other["frame_index"] != candidate["frame_index"]:
            continue
        dist = math.hypot(candidate["center"][0] - other["center"][0], candidate["center"][1] - other["center"][1])
        if bbox_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) >= 0.25 or dist <= 25:
            return "both"
    return f"{candidate['source_model']}_only"


def draw_overlay(frame: np.ndarray, candidates: List[Dict[str, Any]]) -> np.ndarray:
    overlay = frame.copy()
    for row in candidates:
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        color = (0, 0, 255) if row["source_model"] == "rfdetr" else (0, 255, 255)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, f"{row['source_model']}:{row['source_pass']} {row['confidence']}", (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return overlay


def write_review_assets(run_dir: Path, frames: List[Dict[str, Any]], frame_candidates: Dict[str, List[Dict[str, Any]]]) -> None:
    order = {"test_ball_mixed_01": 0, "valid_ball_mixed_01": 1, "train_ball_hard_03": 2, "train_ball_context_02": 3}
    queue = []
    for frame in sorted(frames, key=lambda row: (order.get(row["sequence_id"], 99), row["frame_index"])):
        candidates = frame_candidates.get(frame["frame_id"], [])
        queue.append(
            {
                "frame_id": frame["frame_id"],
                "sequence_id": frame["sequence_id"],
                "split": frame["split"],
                "timestamp_sec": frame["timestamp_sec"],
                "frame_image": frame["image_relpath"],
                "crop_image": frame["review_crop_relpath"],
                "candidate_count": len(candidates),
                "priority": "P0" if candidates else "P2",
                "status": "pending",
                "review_scope": "supplemental_only",
            }
        )
    review_dir = run_dir / "review"
    write_json(review_dir / "supplemental_review_queue.json", {"items": queue})
    write_json(review_dir / "supplemental_review_progress.json", {"total_frames": len(queue), "pending": len(queue), "reviewed_ball": 0, "reviewed_no_ball": 0, "reviewed_uncertain": 0})
    write_jsonl(review_dir / "supplemental_review_decisions.jsonl", [])
    write_jsonl(review_dir / "supplemental_review_audit_log.jsonl", [])
    write_json(review_dir / "supplemental_reviewed_annotations_coco.json", {"images": [], "annotations": [], "categories": [{"id": CLASS_BALL_ID, "name": "ball"}]})
    write_json(review_dir / "supplemental_reviewed_dataset_manifest.json", {"status": "pending_human_review", "ground_truth": False, "scope": "supplemental_only"})
    write_text(
        review_dir / "supplemental_review_instructions.md",
        "# PE-0A4R Supplemental Ball Review\n\n"
        "Review only supplemental frames. Required order: test_ball_mixed_01, valid_ball_mixed_01, train_ball_hard_03, train_ball_context_02.\n"
        "Do not use training metrics. Every frame starts as pending/pseudo_label=true/ground_truth=false.\n",
    )


def process_selected_sequences(args: argparse.Namespace, run_dir: Path, source_meta: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    profile = read_json(ORIGINAL_RUN / "calibration" / "video_calibration_full_field.json")
    roi_polygon = polygon_points(profile)
    yolo = YOLO(str(args.yolo_model))
    rfdetr = init_rfdetr(str(args.rfdetr_model), device=device)

    width, height = int(source_meta["width"]), int(source_meta["height"])
    tiles = generate_tiles(width, height, args.tile_size, args.tile_overlap, roi_polygon)
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    frames_manifest = []
    all_candidates: List[Dict[str, Any]] = []
    frame_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counters = Counter()
    process_every = max(1, int(round(source_meta["fps"] / 15.0)))

    for seq in selected_sequences():
        frame_index = int(round(seq["start_sec"] * source_meta["fps"]))
        end_frame = int(round(seq["end_sec"] * source_meta["fps"]))
        processed_in_seq = 0
        while frame_index < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / source_meta["fps"]
            frame_id = f"{seq['sequence_id']}_f{frame_index:08d}"
            image_rel = Path("dataset") / "frames" / seq["split"] / f"{frame_id}.jpg"
            image_path = run_dir / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)

            y_raw = yolo_predict(yolo, frame, args.yolo_imgsz, args.confidence, args.iou)
            r_global = rfdetr_predict(rfdetr, frame, args.rfdetr_confidence)
            y_rows = [
                make_detection_row(run_id=run_dir.name, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="yolo", source_pass="global", tile_id=None, raw=raw, width=width, height=height, roi_polygon=roi_polygon)
                for raw in y_raw
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]
            r_rows = [
                make_detection_row(run_id=run_dir.name, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="rfdetr", source_pass="global", tile_id=None, raw=raw, width=width, height=height, roi_polygon=roi_polygon)
                for raw in r_global
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]
            counters["rfdetr_global_frames"] += 1 if r_rows else 0
            counters["yolo_frames"] += 1 if y_rows else 0

            tile_rows = []
            for tile in tiles:
                crop = frame[tile["y1"] : tile["y2"], tile["x1"] : tile["x2"]]
                if crop.size == 0:
                    continue
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
                    tile_rows.append(make_detection_row(run_id=run_dir.name, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="rfdetr", source_pass="tile", tile_id=tile["tile_id"], raw=raw, width=width, height=height, roi_polygon=roi_polygon))
            r_rows.extend(tile_rows)
            y_rows = dedupe_ball_candidates(y_rows)
            r_rows = dedupe_ball_candidates(r_rows)
            counters["rfdetr_tile_frames"] += 1 if any(row["source_pass"] == "tile" for row in r_rows) else 0

            candidates = []
            for row in y_rows + r_rows:
                row = dict(row)
                row["frame_id"] = frame_id
                row["sequence_id"] = seq["sequence_id"]
                row["split"] = seq["split"]
                row["agreement"] = agreement_for_candidate(row, y_rows, r_rows)
                row["candidate_status"] = "unreviewed"
                row["pseudo_label"] = True
                row["ground_truth"] = False
                candidates.append(row)
            if candidates:
                counters["frames_with_candidates"] += 1
            if len({c["agreement"] for c in candidates}) > 1 or any(c["agreement"] != "both" for c in candidates):
                counters["conflict_frames"] += 1 if candidates else 0
            all_candidates.extend(candidates)
            frame_candidates[frame_id] = candidates

            center = center_from_xyxy(candidates[0]["bbox_xyxy"]) if candidates else (width / 2.0, height / 2.0)
            x1, y1, x2, y2 = crop_box_512(center, width, height)
            crop_rel = Path("dataset") / "crops" / seq["split"] / f"{frame_id}_review.jpg"
            crop_path = run_dir / crop_rel
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            overlay_rel = Path("overlays") / f"{frame_id}.jpg"
            cv2.imwrite(str(run_dir / overlay_rel), draw_overlay(frame, candidates))

            frames_manifest.append(
                {
                    "frame_id": frame_id,
                    "sequence_id": seq["sequence_id"],
                    "split": seq["split"],
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp, 6),
                    "image_relpath": str(image_rel),
                    "review_crop_relpath": str(crop_rel),
                    "overlay_relpath": str(overlay_rel),
                    "candidate_count": len(candidates),
                    "review_status": "pending",
                    "pseudo_label": True,
                    "ground_truth": False,
                }
            )
            frame_index += process_every
            processed_in_seq += 1
            if len(frames_manifest) % 25 == 0:
                print(json.dumps({"progress": "pe0a4r_preannotation", "frames": len(frames_manifest), "candidates": len(all_candidates)}, sort_keys=True), flush=True)
    cap.release()

    write_json(run_dir / "dataset" / "manifests" / "frames_manifest.json", {"frames": frames_manifest})
    write_jsonl(run_dir / "preannotations" / "ball_candidates.jsonl", all_candidates)
    maybe_write_parquet(run_dir / "preannotations" / "ball_candidates.parquet", all_candidates)
    write_review_assets(run_dir, frames_manifest, frame_candidates)

    return {
        "frames": len(frames_manifest),
        "candidate_count": len(all_candidates),
        "rfdetr_global_frames": counters["rfdetr_global_frames"],
        "rfdetr_tile_frames": counters["rfdetr_tile_frames"],
        "yolo_frames": counters["yolo_frames"],
        "conflict_frames": counters["conflict_frames"],
        "frames_without_candidate": len(frames_manifest) - counters["frames_with_candidates"],
        "device": device,
    }


def write_selected_manifest(run_dir: Path, source_meta: Dict[str, Any]) -> Dict[str, Any]:
    sequences = []
    for seq in selected_sequences():
        row = dict(seq)
        row["duration_sec"] = round(row["end_sec"] - row["start_sec"], 3)
        row["sampling_fps"] = 15
        row["original_frame_start"] = int(round(row["start_sec"] * source_meta["fps"]))
        row["original_frame_end"] = int(round(row["end_sec"] * source_meta["fps"])) - 1
        sequences.append(row)
    checks = selection_checks(sequences)
    manifest = {"phase": PHASE, "created_at": utc_now(), "sequences": sequences, "selection_checks": checks}
    write_json(run_dir / "selected_sequences_manifest.json", manifest)
    lines = [
        "# PE-0A4R Selection Rationale",
        "",
        "Selected after generating candidate windows and contact/previews. Detection counts are secondary evidence; windows were selected for split repair structure, temporal separation, and visual review coverage.",
        "",
        "| sequence | split | start | end | reason |",
        "|---|---|---:|---:|---|",
    ]
    for seq in sequences:
        lines.append(f"| {seq['sequence_id']} | {seq['split']} | {seq['start_sec']} | {seq['end_sec']} | {seq['reason']} |")
    lines.append("")
    lines.append(f"- overlap_check: `{checks['status']}`")
    write_text(run_dir / "selection_rationale.md", "\n".join(lines) + "\n")
    if checks["status"] != "passed":
        raise RuntimeError(f"Selected sequence checks failed: {checks}")
    return manifest


def artifact_manifest(run_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == ".DS_Store":
            continue
        artifacts.append({"relative_path": str(path.relative_to(run_dir)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"phase": PHASE, "run_id": run_dir.name, "created_at": utc_now(), "artifacts": artifacts}


def write_final_report(run_dir: Path, summary: Dict[str, Any], split_repair: Dict[str, Any], selected: Dict[str, Any]) -> None:
    lines = [
        "# PE-0A4R SUPPLEMENTAL BALL DATA",
        "",
        f"- status: `{summary['status']}`",
        f"- original_freeze_id: `{ORIGINAL_FREEZE_ID}`",
        f"- original_review_hash: `{ORIGINAL_REVIEW_HASH}`",
        f"- supplemental_frames_pending: `{summary['frames']}`",
        f"- rfdetr_global_candidates_frames: `{summary['rfdetr_global_frames']}`",
        f"- rfdetr_tile_candidates_frames: `{summary['rfdetr_tile_frames']}`",
        f"- yolo_candidates_frames: `{summary['yolo_frames']}`",
        f"- conflicts: `{summary['conflict_frames']}`",
        f"- frames_without_candidate: `{summary['frames_without_candidate']}`",
        "",
        "Training/export remain blocked until SUPPLEMENTAL BALL REVIEW COMPLETED and combined preflight passes.",
        "",
        "## Review Tool",
        "",
        "```bash",
        f"python3 ai_worker_v1/review_tools/ball_review/server.py --run-dir {run_dir} --session-prefix supplemental_",
        "```",
        "",
        "Review order: test_ball_mixed_01 -> valid_ball_mixed_01 -> train_ball_hard_03 -> train_ball_context_02.",
    ]
    write_text(run_dir / "PE0A4R_FINAL_REPORT.md", "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.time()
    run_id = args.run_id or f"pe0a4r_supplemental_ball_{utc_now_compact()}"
    run_dir = AI_WORKER_ROOT / "runs" / run_id
    ensure_dirs(run_dir)
    source_meta = video_meta(SOURCE_VIDEO)
    freeze_check = validate_original_freeze()
    original_stats = latest_original_statuses()
    split_repair = write_split_repair(run_dir, original_stats)
    windows = make_candidate_contact_sheet(SOURCE_VIDEO, run_dir, candidate_window_specs(source_meta["duration_sec"]), source_meta["fps"])
    selected = write_selected_manifest(run_dir, source_meta)
    preannotation_summary = process_selected_sequences(args, run_dir, source_meta)

    write_json(
        run_dir / "source_manifest.json",
        {
            "source_video": str(SOURCE_VIDEO),
            "source_video_sha256": source_meta["sha256"],
            "source_run": str(SOURCE_RUN),
            "original_run": str(ORIGINAL_RUN),
            "original_freeze_id": ORIGINAL_FREEZE_ID,
            "original_review_hash": ORIGINAL_REVIEW_HASH,
        },
    )
    checks = {
        "freeze_original_intact": freeze_check,
        "selection_windows_count": len(windows),
        "selected_sequence_check": selected["selection_checks"],
        "extraction_fps": 15,
        "review_scope": "supplemental_only",
        "src_intact_policy": "no src/ production files modified by this script",
    }
    write_json(run_dir / "tests" / "pe0a4r_generation_checks.json", checks)
    summary = {
        "phase": PHASE,
        "run_id": run_id,
        "status": "ready_for_supplemental_review",
        "original_review_integrity": freeze_check["status"],
        "candidate_windows": len(windows),
        "selected_sequences": len(selected["sequences"]),
        "runtime_sec": round(time.time() - started, 3),
        **preannotation_summary,
    }
    write_json(run_dir / "metrics" / "pe0a4r_summary.json", summary)
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
    write_final_report(run_dir, summary, split_repair, selected)
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--yolo-model", type=Path, default=AI_WORKER_ROOT / "src" / "oneframe_v3_best.pt")
    parser.add_argument("--rfdetr-model", type=Path, default=AI_WORKER_ROOT / "rfdetr_cache" / "rf-detr-base.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--rfdetr-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
