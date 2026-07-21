#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = REPO_ROOT / "ai_worker_v1" / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def freeze_review(run_dir: Path) -> Dict[str, Any]:
    review_dir = run_dir / "review"
    freeze_id = datetime.now(timezone.utc).strftime("review_freeze_%Y%m%dT%H%M%SZ")
    freeze_dir = review_dir / "frozen" / freeze_id
    freeze_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "review_decisions.jsonl",
        "review_progress.json",
        "review_audit_log.jsonl",
        "review_queue.json",
        "reviewed_annotations_coco.json",
    ]
    frozen_files = []
    for name in files:
        src = review_dir / name
        if not src.exists():
            continue
        dst = freeze_dir / name
        shutil.copy2(src, dst)
        frozen_files.append(
            {
                "file": name,
                "path": str(dst),
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size,
                "line_count": sum(1 for _ in dst.open()) if dst.suffix == ".jsonl" else None,
            }
        )
    review_hash_payload = "|".join(f"{item['file']}:{item['sha256']}" for item in sorted(frozen_files, key=lambda row: row["file"]))
    review_hash = hashlib.sha256(review_hash_payload.encode("utf-8")).hexdigest()
    manifest = {
        "created_at": utc_now(),
        "freeze_id": freeze_id,
        "review_hash": review_hash,
        "source_run_dir": str(run_dir),
        "frozen_files": frozen_files,
        "decision_count": next((item["line_count"] for item in frozen_files if item["file"] == "review_decisions.jsonl"), 0),
    }
    write_json(review_dir / "review_frozen_manifest.json", manifest)
    write_json(freeze_dir / "review_frozen_manifest.json", manifest)
    write_text(
        review_dir / "review_completion_report.md",
        "# Review Completion Report\n\n"
        f"- freeze_id: `{freeze_id}`\n"
        f"- review_hash: `{review_hash}`\n"
        f"- decisions: `{manifest['decision_count']}`\n"
        f"- frozen_at: `{manifest['created_at']}`\n",
    )
    return manifest


def validate_bbox(box: Any) -> List[str]:
    errors = []
    if not isinstance(box, list) or len(box) != 4:
        return ["bbox_missing_or_invalid_shape"]
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        errors.append("bbox_zero_or_negative_area")
    if x1 < 0 or y1 < 0 or x2 > FRAME_WIDTH or y2 > FRAME_HEIGHT:
        errors.append("bbox_out_of_bounds")
    return errors


def preflight(run_dir: Path, review_manifest: Dict[str, Any]) -> Dict[str, Any]:
    preflight_dir = run_dir / "preflight"
    queue = read_json(run_dir / "review" / "review_queue.json")["items"]
    decisions = read_jsonl(run_dir / "review" / "review_decisions.jsonl")
    latest_decision = {}
    duplicate_decisions = Counter()
    for decision in decisions:
        frame_id = decision["frame_id"]
        duplicate_decisions[frame_id] += 1
        latest_decision[frame_id] = decision

    status_counts = Counter(item.get("status", "pending") for item in queue)
    split_counts = defaultdict(Counter)
    seq_by_split = defaultdict(set)
    frame_to_split = {}
    duplicate_frame_ids = []
    bbox_errors = []
    annotations_by_sequence = defaultdict(int)
    boxes_accepted = 0
    boxes_drawn_manual = 0
    boxes_corrected = 0
    candidates_deleted = 0

    for item in queue:
        frame_id = item["frame_id"]
        split = item["split"]
        status = item.get("status", "pending")
        split_counts[split][status] += 1
        seq_by_split[item["sequence_id"]].add(split)
        if frame_id in frame_to_split:
            duplicate_frame_ids.append(frame_id)
        frame_to_split[frame_id] = split
        decision = latest_decision.get(frame_id)
        if status == "reviewed_ball":
            boxes_accepted += 1
            if decision and decision.get("bbox_xyxy"):
                boxes_drawn_manual += 1
                boxes_corrected += 1 if item.get("candidate_count", 0) else 0
                annotations_by_sequence[item["sequence_id"]] += 1
                for err in validate_bbox(decision.get("bbox_xyxy")):
                    bbox_errors.append({"frame_id": frame_id, "error": err, "bbox_xyxy": decision.get("bbox_xyxy")})
            else:
                bbox_errors.append({"frame_id": frame_id, "error": "reviewed_ball_without_bbox", "bbox_xyxy": None})
        elif status in {"reviewed_no_ball", "reviewed_uncertain"}:
            if item.get("candidate_count", 0):
                candidates_deleted += item.get("candidate_count", 0)

    sequence_leakage = {seq: sorted(splits) for seq, splits in seq_by_split.items() if len(splits) > 1}
    split_summary = {split: dict(counts) for split, counts in split_counts.items()}
    validation_errors = []
    if status_counts.get("pending", 0) != 0:
        validation_errors.append("pending_frames_remaining")
    for split_name in ("valid", "within_video_test_v0"):
        if split_counts[split_name].get("pending", 0) != 0:
            validation_errors.append(f"{split_name}_not_fully_reviewed")
    if duplicate_frame_ids:
        validation_errors.append("duplicate_frame_ids")
    if sequence_leakage:
        validation_errors.append("sequence_split_leakage")
    if bbox_errors:
        validation_errors.append("bbox_validation_errors")
    if split_counts["valid"].get("reviewed_ball", 0) == 0:
        validation_errors.append("valid_split_has_zero_positive_ball_frames")
    if split_counts["within_video_test_v0"].get("reviewed_ball", 0) == 0:
        validation_errors.append("test_split_has_zero_positive_ball_frames")
    if split_counts["valid"].get("reviewed_no_ball", 0) == 0:
        validation_errors.append("valid_split_has_zero_negative_frames")

    statistics = {
        "frames_total": len(queue),
        "reviewed_ball": status_counts.get("reviewed_ball", 0),
        "reviewed_no_ball": status_counts.get("reviewed_no_ball", 0),
        "reviewed_uncertain": status_counts.get("reviewed_uncertain", 0),
        "pending": status_counts.get("pending", 0),
        "boxes_accepted": boxes_accepted,
        "boxes_drawn_manual": boxes_drawn_manual,
        "boxes_corrected": boxes_corrected,
        "candidates_eliminated_or_rejected": candidates_deleted,
        "decisions_total": len(decisions),
        "frames_with_multiple_decisions": sum(1 for count in duplicate_decisions.values() if count > 1),
        "decisions_by_split": split_summary,
        "annotations_by_sequence": dict(annotations_by_sequence),
        "review_hash": review_manifest["review_hash"],
    }
    validation = {
        "status": "passed" if not validation_errors else "failed",
        "errors": validation_errors,
        "bbox_errors": bbox_errors,
        "duplicate_frame_ids": duplicate_frame_ids,
        "sequence_split_leakage": sequence_leakage,
        "uncertain_policy": "reviewed_uncertain excluded from train/valid/test metrics; never converted to negatives",
        "ground_truth_policy": "only latest human reviewed_ball bbox can become ground_truth=true",
    }
    leakage = {
        "status": "passed" if not duplicate_frame_ids and not sequence_leakage else "failed",
        "duplicate_frame_ids": duplicate_frame_ids,
        "sequence_split_leakage": sequence_leakage,
        "split_summary": split_summary,
    }
    write_json(preflight_dir / "review_statistics.json", statistics)
    write_json(preflight_dir / "dataset_validation.json", validation)
    write_json(preflight_dir / "leakage_report.json", leakage)
    status = "blocked_dataset_validation" if validation_errors else "ready_for_dataset_export"
    write_text(
        preflight_dir / "PREFLIGHT_REPORT.md",
        "# PE-0A4 Preflight Report\n\n"
        f"- status: `{status}`\n"
        f"- review_hash: `{review_manifest['review_hash']}`\n"
        f"- frames_total: `{statistics['frames_total']}`\n"
        f"- reviewed_ball: `{statistics['reviewed_ball']}`\n"
        f"- reviewed_no_ball: `{statistics['reviewed_no_ball']}`\n"
        f"- reviewed_uncertain: `{statistics['reviewed_uncertain']}`\n"
        f"- pending: `{statistics['pending']}`\n"
        f"- decisions_by_split: `{split_summary}`\n"
        f"- validation_errors: `{validation_errors}`\n\n"
        "Dataset export and training are blocked unless validation status is passed.\n",
    )
    return {"status": status, "statistics": statistics, "validation": validation, "leakage": leakage}


def main() -> None:
    manifest = freeze_review(RUN_DIR)
    result = preflight(RUN_DIR, manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
