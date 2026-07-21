#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
RUN_DIR = AI_WORKER_ROOT / "runs" / "pe0a4r_supplemental_ball_20260715T060000Z"
SOURCE_VIDEO = AI_WORKER_ROOT / "runs" / "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps" / "source_video.mp4"
ORIGINAL_RUN = AI_WORKER_ROOT / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z"
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_REVIEW_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"
SAMPLING_FPS = 15.0
YOLO_TIME_LIMIT_SEC = 600.0


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_original_freeze_manifest() -> Dict[str, Any]:
    freeze_dir = ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID
    manifest_path = freeze_dir / "review_frozen_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("freeze_id") != ORIGINAL_FREEZE_ID:
        raise RuntimeError("Original review freeze_id mismatch")
    if manifest.get("review_hash") != ORIGINAL_REVIEW_HASH:
        raise RuntimeError("Original review hash mismatch")
    return manifest


def selected_sequences() -> List[Dict[str, Any]]:
    manifest = read_json(RUN_DIR / "selected_sequences_manifest.json")
    sequences = manifest["sequences"]
    order = {
        "test_ball_mixed_01": 0,
        "valid_ball_mixed_01": 1,
        "train_ball_hard_03": 2,
        "train_ball_context_02": 3,
    }
    return sorted(sequences, key=lambda row: order[row["sequence_id"]])


def public_split(split: str) -> str:
    return "test" if split == "within_video_test_v0" else split


def video_meta(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fps": fps,
        "frames": frames,
        "duration_sec": frames / fps if fps else 0.0,
        "width": width,
        "height": height,
    }


def clamp_bbox(box: List[float], width: int, height: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def center_from_bbox(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def crop_512(center: Tuple[float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(max(0, width - 512), int(round(center[0] - 256))))
    y1 = max(0, min(max(0, height - 512), int(round(center[1] - 256))))
    return x1, y1, min(width, x1 + 512), min(height, y1 + 512)


def extract_frames(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    step = max(1, int(round(meta["fps"] / SAMPLING_FPS)))
    rows: List[Dict[str, Any]] = []
    width, height = int(meta["width"]), int(meta["height"])
    for seq in selected_sequences():
        split = public_split(seq["split"])
        frame_index = int(round(seq["start_sec"] * meta["fps"]))
        end_frame = int(round(seq["end_sec"] * meta["fps"]))
        while frame_index < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / meta["fps"]
            frame_id = f"{seq['sequence_id']}_f{frame_index:08d}"
            image_rel = Path("dataset") / "supplemental_frames" / split / f"{frame_id}.jpg"
            image_path = RUN_DIR / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)
            crop_rel = Path("dataset") / "supplemental_crops" / split / f"{frame_id}_review.jpg"
            crop_path = RUN_DIR / crop_rel
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            x1, y1, x2, y2 = crop_512((width / 2.0, height / 2.0), width, height)
            cv2.imwrite(str(crop_path), frame[y1:y2, x1:x2])
            rows.append(
                {
                    "frame_id": frame_id,
                    "sequence_id": seq["sequence_id"],
                    "split": split,
                    "frame_index_original": frame_index,
                    "timestamp_sec_original": round(timestamp, 6),
                    "timestamp_sec": round(timestamp, 6),
                    "image_path": str(image_rel),
                    "image_relpath": str(image_rel),
                    "review_crop_relpath": str(crop_rel),
                    "review_status": "pending",
                    "preannotation_status": "none",
                    "candidate_sources": [],
                    "candidate_count": 0,
                    "candidates": [],
                }
            )
            frame_index += step
    cap.release()
    return rows


def yolo_class_is_ball(class_id: int, class_name: str, names: Dict[int, str]) -> bool:
    name = str(class_name or "").lower().replace("_", " ")
    if any(token in name for token in ["ball", "balon", "balón", "pelota", "sports ball"]):
        return True
    if len(names) == 1:
        return True
    return class_id in {0, 32, 37}


def run_yolo_candidates(frames: List[Dict[str, Any]], meta: Dict[str, Any], model_path: Path, time_limit_sec: float) -> Dict[str, Any]:
    started = time.time()
    candidates: List[Dict[str, Any]] = []
    status = "not_run"
    error = None
    frames_with_candidates = set()
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        names = {int(k): str(v) for k, v in getattr(model, "names", {}).items()}
        status = "completed"
        for idx, frame in enumerate(frames):
            if time.time() - started > time_limit_sec:
                status = "timeout"
                break
            image = RUN_DIR / frame["image_relpath"]
            results = model.predict(str(image), conf=0.25, iou=0.45, imgsz=640, verbose=False)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    cls = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                    class_name = names.get(cls, str(cls))
                    if not yolo_class_is_ball(cls, class_name, names):
                        continue
                    xyxy_raw = box.xyxy[0].detach().cpu().tolist()
                    xyxy = clamp_bbox(xyxy_raw, int(meta["width"]), int(meta["height"]))
                    if xyxy is None:
                        continue
                    conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                    cx, cy = center_from_bbox(xyxy)
                    candidates.append(
                        {
                            "frame_id": frame["frame_id"],
                            "sequence_id": frame["sequence_id"],
                            "split": frame["split"],
                            "frame_index_original": frame["frame_index_original"],
                            "timestamp_sec_original": frame["timestamp_sec_original"],
                            "source_model": "yolo",
                            "source_pass": "global",
                            "source_run": "local_manual_first",
                            "inference_complete": status == "completed",
                            "class_id": cls,
                            "class_name": class_name,
                            "confidence": round(conf, 6),
                            "bbox_xyxy": xyxy,
                            "center": [round(cx, 4), round(cy, 4)],
                            "candidate_status": "unreviewed",
                            "pseudo_label": True,
                            "ground_truth": False,
                        }
                    )
                    frames_with_candidates.add(frame["frame_id"])
            if (idx + 1) % 50 == 0:
                print(json.dumps({"progress": "yolo_manual_first", "frames": idx + 1, "candidates": len(candidates)}, sort_keys=True), flush=True)
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error = str(exc)
    return {
        "status": status,
        "error": error,
        "runtime_sec": round(time.time() - started, 3),
        "candidates": candidates,
        "frames_with_candidates": len(frames_with_candidates),
    }


def attach_candidates(frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_frame: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_frame[row["frame_id"]].append(row)
    for frame in frames:
        rows = by_frame.get(frame["frame_id"], [])
        frame["candidates"] = rows
        frame["candidate_count"] = len(rows)
        frame["candidate_sources"] = sorted({f"{row['source_model']}:{row['source_pass']}:{row['source_run']}" for row in rows})
        if rows:
            source_runs = {row.get("source_run") for row in rows}
            frame["preannotation_status"] = "partial" if "cpu_partial" in source_runs else "available"
    return by_frame


def draw_overlay(frame_path: Path, out_path: Path, candidates: List[Dict[str, Any]], label: str) -> None:
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return
    for row in candidates:
        x1, y1, x2, y2 = [int(round(v)) for v in row["bbox_xyxy"]]
        color = (0, 255, 255) if row["source_model"] == "yolo" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{row['source_model']} {row['confidence']:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def write_contact_sheet(name: str, frame_rows: List[Dict[str, Any]], by_frame: Dict[str, List[Dict[str, Any]]], max_per_sequence: int = 24) -> str:
    thumbs = []
    selected = frame_rows[:max_per_sequence]
    for row in selected:
        image_path = RUN_DIR / row["image_relpath"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        for cand in by_frame.get(row["frame_id"], []):
            x1, y1, x2, y2 = [int(round(v)) for v in cand["bbox_xyxy"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        thumb = cv2.resize(frame, (320, 180))
        text = f"{row['sequence_id']} f={row['frame_index_original']} c={row['candidate_count']}"
        cv2.putText(thumb, text[:42], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        thumbs.append(thumb)
    if not thumbs:
        return ""
    cols = 4
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = np.zeros((rows * 180, cols * 320, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        r, c = divmod(idx, cols)
        sheet[r * 180 : r * 180 + 180, c * 320 : c * 320 + 320] = thumb
    out_rel = Path("review") / name
    cv2.imwrite(str(RUN_DIR / out_rel), sheet)
    return str(out_rel)


def write_review_files(frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]], by_frame: Dict[str, List[Dict[str, Any]]], yolo_summary: Dict[str, Any]) -> Dict[str, Any]:
    review_dir = RUN_DIR / "review"
    pre_dir = RUN_DIR / "preannotations"
    review_dir.mkdir(parents=True, exist_ok=True)
    pre_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(pre_dir / "supplemental_ball_candidates.jsonl", candidates)

    queue = []
    for row in frames:
        queue.append(
            {
                "frame_id": row["frame_id"],
                "sequence_id": row["sequence_id"],
                "split": row["split"],
                "frame_index_original": row["frame_index_original"],
                "timestamp_sec_original": row["timestamp_sec_original"],
                "timestamp_sec": row["timestamp_sec_original"],
                "image_path": row["image_relpath"],
                "frame_image": row["image_relpath"],
                "crop_image": row["review_crop_relpath"],
                "candidates": [],
                "candidate_sources": row["candidate_sources"],
                "candidate_count": row["candidate_count"],
                "preannotation_status": row["preannotation_status"],
                "review_status": "pending",
                "status": "pending",
                "priority": "P0" if row["candidate_count"] else "P2",
                "review_scope": "supplemental_manual_first",
            }
        )

    progress = {
        "total_frames": len(queue),
        "pending": len(queue),
        "reviewed_ball": 0,
        "reviewed_no_ball": 0,
        "reviewed_uncertain": 0,
        "by_split": {
            split: {
                "total": sum(1 for row in queue if row["split"] == split),
                "pending": sum(1 for row in queue if row["split"] == split),
            }
            for split in ["test", "valid", "train"]
        },
    }
    write_json(review_dir / "supplemental_review_queue.json", {"items": queue})
    write_json(review_dir / "supplemental_review_progress.json", progress)
    write_jsonl(review_dir / "supplemental_review_decisions.jsonl", [])
    write_jsonl(
        review_dir / "supplemental_review_audit_log.jsonl",
        [
            {
                "created_at": utc_now(),
                "event": "UI_TEST",
                "status": "not_human_review",
                "details": "Manual-first review queue initialized; validation will use synthetic save/reset checks.",
            }
        ],
    )
    write_json(
        review_dir / "supplemental_review_manifest.json",
        {
            "phase": "PE-0A4R",
            "status": "ready_for_supplemental_review",
            "created_at": utc_now(),
            "review_scope": "supplemental_only",
            "original_freeze_id": ORIGINAL_FREEZE_ID,
            "original_review_hash": ORIGINAL_REVIEW_HASH,
            "order": ["test", "valid", "train_ball_hard_03", "train_ball_context_02"],
            "frames": len(queue),
            "candidate_sources_policy": ["rfdetr_cpu_partial_if_serialized", "yolo_local_if_completed", "empty_candidates_allowed"],
            "yolo_summary": {k: v for k, v in yolo_summary.items() if k != "candidates"},
            "training_allowed": False,
            "decision": "wait_for_supplemental_ball_review_completed",
        },
    )
    write_text(
        review_dir / "supplemental_review_instructions.md",
        "# PE-0A4R Manual-First Supplemental Ball Review\n\n"
        "Review every supplemental frame. The absence of candidates is valid and expected.\n\n"
        "Controls:\n"
        "- A: accept selected/drawn bbox\n"
        "- N: reviewed_no_ball\n"
        "- U: reviewed_uncertain\n"
        "- D: delete selected/current bbox\n"
        "- S: save\n"
        "- Arrow keys: navigate\n"
        "- Draw bbox: click and drag over the visible ball\n\n"
        "Visible reminder: if a frame has no candidates, draw a box if the ball is visible.\n\n"
        "Do not train until pending total is zero and the combined preflight passes.\n",
    )
    by_sequence = defaultdict(list)
    for row in frames:
        by_sequence[row["sequence_id"]].append(row)
    sheets = {
        "supplemental_review_contact_sheet": write_contact_sheet("supplemental_review_contact_sheet.png", frames, by_frame, max_per_sequence=48),
        "test_sequence_contact_sheet": write_contact_sheet("test_sequence_contact_sheet.png", by_sequence["test_ball_mixed_01"], by_frame),
        "valid_sequence_contact_sheet": write_contact_sheet("valid_sequence_contact_sheet.png", by_sequence["valid_ball_mixed_01"], by_frame),
        "train_sequences_contact_sheet": write_contact_sheet("train_sequences_contact_sheet.png", by_sequence["train_ball_hard_03"] + by_sequence["train_ball_context_02"], by_frame, max_per_sequence=48),
    }
    return {"queue": queue, "progress": progress, "contact_sheets": sheets}


def validate_and_summarize(frames: List[Dict[str, Any]], candidates: List[Dict[str, Any]], yolo_summary: Dict[str, Any], contact_sheets: Dict[str, str]) -> Dict[str, Any]:
    frame_ids = [row["frame_id"] for row in frames]
    errors = []
    if len(frame_ids) != len(set(frame_ids)):
        errors.append("duplicate_frame_ids")
    if not any(row["candidate_count"] == 0 for row in frames):
        errors.append("queue_missing_zero_candidate_frames")
    seqs = selected_sequences()
    for i, left in enumerate(seqs):
        for right in seqs[i + 1 :]:
            if max(left["start_sec"], right["start_sec"]) < min(left["end_sec"], right["end_sec"]):
                errors.append(f"temporal_overlap:{left['sequence_id']}:{right['sequence_id']}")
    counts_by_split = Counter(row["split"] for row in frames)
    candidate_frame_ids = {row["frame_id"] for row in candidates}
    summary = {
        "phase": "PE-0A4R MANUAL-FIRST SUPPLEMENTAL REVIEW",
        "status": "ready_for_supplemental_review" if not errors else "blocked",
        "created_at": utc_now(),
        "frames_total": len(frames),
        "frames_by_split": dict(counts_by_split),
        "frames_with_candidates": len(candidate_frame_ids),
        "frames_without_candidates": len(frames) - len(candidate_frame_ids),
        "rfdetr_cpu_partial_candidate_frames": 0,
        "yolo_candidate_frames": yolo_summary.get("frames_with_candidates", 0),
        "candidate_count": len(candidates),
        "yolo_status": yolo_summary.get("status"),
        "yolo_error": yolo_summary.get("error"),
        "contact_sheets": contact_sheets,
        "errors": errors,
        "original_freeze_id": ORIGINAL_FREEZE_ID,
        "original_review_hash": ORIGINAL_REVIEW_HASH,
        "original_freeze_intact": True,
        "gpu_image": {
            "image": "oneframecontent/oneframe-ai-worker-v1:pe0a4r-gpu-preannotate-20260715",
            "digest": "sha256:7b2efaeeec3f6e0bacb9bdf2941bbd0a037c0070847a4578f62a0dbfee791d8c",
            "status": "available_but_not_executed",
            "promotion_status": "none",
            "purpose": "diagnostic_shadow_only",
        },
        "training_allowed": False,
    }
    write_json(RUN_DIR / "metrics" / "manual_first_supplemental_summary.json", summary)
    return summary


def write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# PE-0A4R Manual-First Supplemental Review",
        "",
        f"- status: `{summary['status']}`",
        f"- total_frames: `{summary['frames_total']}`",
        f"- test: `{summary['frames_by_split'].get('test', 0)}`",
        f"- valid: `{summary['frames_by_split'].get('valid', 0)}`",
        f"- train: `{summary['frames_by_split'].get('train', 0)}`",
        f"- yolo_candidate_frames: `{summary['yolo_candidate_frames']}`",
        f"- rfdetr_cpu_partial_candidate_frames: `{summary['rfdetr_cpu_partial_candidate_frames']}`",
        f"- frames_without_candidates: `{summary['frames_without_candidates']}`",
        f"- original_freeze_id: `{ORIGINAL_FREEZE_ID}`",
        f"- original_review_hash: `{ORIGINAL_REVIEW_HASH}`",
        "",
        "## GPU Attempts",
        "",
        "- attempt_1: `85fad0f8-9819-4aed-a0f7-842fcbf40257-u2` cancelled after staying IN_QUEUE with workers 0",
        "- attempt_2: `5d82c116-bab0-4401-b48c-f6a0182c6cef-u1` cancelled after staying IN_QUEUE with worker throttled=1",
        "- image: `oneframecontent/oneframe-ai-worker-v1:pe0a4r-gpu-preannotate-20260715`",
        "- image_status: `available_but_not_executed`",
        "",
        "## Review",
        "",
        "```bash",
        f"python3 ai_worker_v1/review_tools/ball_review/server.py --run-dir {RUN_DIR} --session-prefix supplemental_",
        "```",
        "",
        "Order: test_ball_mixed_01 -> valid_ball_mixed_01 -> train_ball_hard_03 -> train_ball_context_02.",
        "",
        "Decision: wait for SUPPLEMENTAL BALL REVIEW COMPLETED. Do not train.",
    ]
    write_text(RUN_DIR / "PE0A4R_MANUAL_FIRST_REVIEW_REPORT.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--yolo-model", type=Path, default=AI_WORKER_ROOT / "src" / "oneframe_v3_best.pt")
    parser.add_argument("--yolo-time-limit-sec", type=float, default=YOLO_TIME_LIMIT_SEC)
    args = parser.parse_args()

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    freeze_manifest = load_original_freeze_manifest()
    meta = video_meta(SOURCE_VIDEO)
    frames = extract_frames(meta)

    yolo_summary: Dict[str, Any] = {"status": "skipped", "error": None, "runtime_sec": 0.0, "candidates": [], "frames_with_candidates": 0}
    if not args.skip_yolo:
        yolo_summary = run_yolo_candidates(frames, meta, args.yolo_model, args.yolo_time_limit_sec)

    candidates = list(yolo_summary.get("candidates", []))
    by_frame = attach_candidates(frames, candidates)

    # Write per-frame overlays after candidates are attached.
    for frame in frames:
        overlay_rel = Path("review") / "supplemental_overlays" / f"{frame['frame_id']}.jpg"
        draw_overlay(RUN_DIR / frame["image_relpath"], RUN_DIR / overlay_rel, by_frame.get(frame["frame_id"], []), frame["frame_id"])
        frame["overlay_relpath"] = str(overlay_rel)

    write_json(RUN_DIR / "dataset" / "manifests" / "supplemental_frames_manifest.json", {"frames": frames})
    review = write_review_files(frames, candidates, by_frame, yolo_summary)
    summary = validate_and_summarize(frames, candidates, yolo_summary, review["contact_sheets"])
    write_json(
        RUN_DIR / "source_manifest_manual_first.json",
        {
            "created_at": utc_now(),
            "source_video": str(SOURCE_VIDEO),
            "source_video_sha256": meta["sha256"],
            "source_video_meta": meta,
            "selected_sequences_manifest": str(RUN_DIR / "selected_sequences_manifest.json"),
            "original_freeze_id": ORIGINAL_FREEZE_ID,
            "original_review_hash": ORIGINAL_REVIEW_HASH,
            "freeze_manifest_hash": sha256_file(ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID / "review_frozen_manifest.json"),
            "freeze_manifest_seen": freeze_manifest.get("freeze_id"),
        },
    )
    write_report(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "ready_for_supplemental_review":
        sys.exit(1)


if __name__ == "__main__":
    main()

