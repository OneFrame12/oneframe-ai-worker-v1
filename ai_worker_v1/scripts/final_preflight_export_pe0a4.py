#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"

ORIGINAL_RUN = AI_WORKER_ROOT / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z"
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"
ORIGINAL_FREEZE = ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID

SUPP_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_supplemental_ball_20260715T060000Z"
SUPP_FREEZE_ID = "supplemental_review_freeze_20260715T163151Z"
SUPP_HASH = "1ac2d800a46290c9142836ad4a3295425efe4d03445e11524f820081bccf7912"
SUPP_FREEZE = SUPP_RUN / "review" / "frozen" / SUPP_FREEZE_ID

CORRECTION_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_uncertain_correction_20260715T165053Z"
CORRECTION_REVIEW = CORRECTION_RUN / "review"

COMBINED_DIR = CORRECTION_RUN / "combined_review"
PREFLIGHT_DIR = CORRECTION_RUN / "final_preflight"
DATASET_DIR = AI_WORKER_ROOT / "datasets" / "OneFrame_Ball_v0"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
GATES = {
    "train": {"ball": 80, "no_ball": 30, "positive_sequences": 3},
    "valid": {"ball": 15, "no_ball": 10},
    "test": {"ball": 15, "no_ball": 10},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def latest_by_frame(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["frame_id"]] = row
    return latest


def validate_bbox(box: Any) -> List[str]:
    if not isinstance(box, list) or len(box) != 4:
        return ["bbox_missing_or_invalid_shape"]
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:  # noqa: BLE001
        return ["bbox_non_numeric"]
    errors = []
    if x2 <= x1 or y2 <= y1:
        errors.append("bbox_zero_or_negative_area")
    if x1 < 0 or y1 < 0 or x2 > FRAME_WIDTH or y2 > FRAME_HEIGHT:
        errors.append("bbox_out_of_bounds")
    return errors


def verify_freeze(freeze_dir: Path, manifest_name: str, freeze_id: str, review_hash: str) -> Dict[str, Any]:
    manifest = read_json(freeze_dir / manifest_name)
    errors = []
    if manifest.get("freeze_id") != freeze_id:
        errors.append("freeze_id_mismatch")
    if manifest.get("review_hash") != review_hash:
        errors.append("review_hash_mismatch")
    for item in manifest.get("frozen_files", []):
        path = Path(item["path"])
        if not path.exists():
            path = freeze_dir / item["file"]
        if not path.exists():
            errors.append(f"missing:{item['file']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash_mismatch:{item['file']}")
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "freeze_id": freeze_id,
        "review_hash": review_hash,
    }


def correction_completion() -> Dict[str, Any]:
    queue = read_json(CORRECTION_REVIEW / "uncertain_correction_queue.json")["items"]
    latest = latest_by_frame(read_jsonl(CORRECTION_REVIEW / "uncertain_correction_decisions.jsonl"))
    errors = []
    if len(queue) != 127:
        errors.append(f"queue_count_expected_127_actual_{len(queue)}")
    if len({item["frame_id"] for item in queue}) != len(queue):
        errors.append("duplicate_frame_ids_in_queue")
    if len(latest) != len(queue):
        errors.append(f"unique_decisions_expected_{len(queue)}_actual_{len(latest)}")

    counts = Counter()
    by_split = defaultdict(Counter)
    by_sequence = defaultdict(Counter)
    bboxes_drawn = 0
    bboxes_corrected = 0
    missing_decisions = []
    invalid_bboxes = []
    for item in queue:
        decision = latest.get(item["frame_id"])
        if not decision:
            missing_decisions.append(item["frame_id"])
            continue
        if decision.get("reviewed_by") != "human":
            errors.append(f"non_human_decision:{item['frame_id']}")
        status = decision.get("new_status")
        counts[status] += 1
        by_split[item["split"]][status] += 1
        by_sequence[item["sequence_id"]][status] += 1
        if status == "reviewed_ball":
            box_errors = validate_bbox(decision.get("final_bbox_xyxy"))
            if box_errors:
                invalid_bboxes.append({"frame_id": item["frame_id"], "errors": box_errors})
            else:
                bboxes_drawn += 1
                if item.get("final_bbox_xyxy") != decision.get("final_bbox_xyxy"):
                    bboxes_corrected += 1
        elif decision.get("final_bbox_xyxy") is not None:
            invalid_bboxes.append({"frame_id": item["frame_id"], "errors": ["non_ball_decision_has_bbox"]})
    if missing_decisions:
        errors.append(f"missing_decisions:{len(missing_decisions)}")
    if invalid_bboxes:
        errors.append(f"invalid_bboxes:{len(invalid_bboxes)}")
    pending = len(queue) - len(latest)
    return {
        "status": "passed" if not errors and pending == 0 else "failed",
        "errors": errors,
        "queue_items": len(queue),
        "unique_decisions": len(latest),
        "pending": pending,
        "reviewed_ball": counts.get("reviewed_ball", 0),
        "reviewed_no_ball": counts.get("reviewed_no_ball", 0),
        "reviewed_uncertain": counts.get("reviewed_uncertain", 0),
        "bboxes_drawn": bboxes_drawn,
        "bboxes_corrected": bboxes_corrected,
        "decisions_by_split": {split: dict(counter) for split, counter in by_split.items()},
        "decisions_by_sequence": {seq: dict(counter) for seq, counter in by_sequence.items()},
        "invalid_bboxes": invalid_bboxes,
        "missing_decisions": missing_decisions,
    }


def freeze_correction(completion: Dict[str, Any]) -> Dict[str, Any]:
    freeze_id = f"uncertain_correction_freeze_{compact_now()}"
    freeze_dir = CORRECTION_REVIEW / "frozen" / freeze_id
    if freeze_dir.exists():
        raise FileExistsError(freeze_dir)
    freeze_dir.mkdir(parents=True)
    file_names = [
        "uncertain_correction_queue.json",
        "uncertain_correction_decisions.jsonl",
        "uncertain_correction_progress.json",
        "uncertain_correction_audit_log.jsonl",
        "uncertain_correction_manifest.json",
    ]
    frozen_files = []
    for name in file_names:
        src = CORRECTION_REVIEW / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = freeze_dir / name
        shutil.copy2(src, dst)
        frozen_files.append(
            {
                "file": name,
                "path": str(dst),
                "sha256": sha256_file(dst),
                "size_bytes": dst.stat().st_size,
                "line_count": sum(1 for _ in dst.open(encoding="utf-8")) if dst.suffix == ".jsonl" else None,
            }
        )
    payload = "|".join(f"{row['file']}:{row['sha256']}" for row in sorted(frozen_files, key=lambda item: item["file"]))
    correction_hash = sha256_text(payload)
    manifest = {
        "created_at": utc_now(),
        "freeze_id": freeze_id,
        "review_hash": correction_hash,
        "source_run_dir": str(CORRECTION_RUN),
        "source_review_dir": str(CORRECTION_REVIEW),
        "original_freeze_id": ORIGINAL_FREEZE_ID,
        "original_review_hash": ORIGINAL_HASH,
        "supplemental_freeze_id": SUPP_FREEZE_ID,
        "supplemental_review_hash": SUPP_HASH,
        "frozen_files": frozen_files,
        "completion": completion,
    }
    write_json(freeze_dir / "uncertain_correction_frozen_manifest.json", manifest)
    write_text(freeze_dir / "uncertain_correction_hash.txt", correction_hash + "\n")
    write_text(
        freeze_dir / "uncertain_correction_completion_report.md",
        "# Uncertain Correction Completion Report\n\n"
        f"- freeze_id: `{freeze_id}`\n"
        f"- correction_hash: `{correction_hash}`\n"
        f"- queue_items: `{completion['queue_items']}`\n"
        f"- reviewed_ball: `{completion['reviewed_ball']}`\n"
        f"- reviewed_no_ball: `{completion['reviewed_no_ball']}`\n"
        f"- reviewed_uncertain: `{completion['reviewed_uncertain']}`\n"
        f"- pending: `{completion['pending']}`\n"
        f"- bboxes_drawn: `{completion['bboxes_drawn']}`\n"
        f"- bboxes_corrected: `{completion['bboxes_corrected']}`\n"
        f"- status: `{completion['status']}`\n",
    )
    write_json(CORRECTION_REVIEW / "uncertain_correction_frozen_manifest.json", manifest)
    return manifest


def original_effective_split(item: Dict[str, Any]) -> str:
    if item["sequence_id"] == "dense_goal_approach_01":
        return "train"
    if item["sequence_id"] == "dense_feet_cluster_01":
        return "excluded_uncertain_pool"
    if item["split"] == "within_video_test_v0":
        return "test"
    return item["split"]


def supplemental_effective_split(item: Dict[str, Any]) -> str:
    return "test" if item["split"] == "within_video_test_v0" else item["split"]


def source_image_path(source: str, item: Dict[str, Any]) -> str:
    if source == "original":
        rel = item.get("frame_image") or item.get("image_path") or item.get("review_crop_relpath")
        return str(ORIGINAL_RUN / rel) if rel else ""
    rel = item.get("frame_image") or item.get("image_path") or item.get("review_crop_relpath")
    return str(SUPP_RUN / rel) if rel else ""


def source_rows() -> Dict[str, Dict[str, Any]]:
    original_queue = read_json(ORIGINAL_FREEZE / "review_queue.json")["items"]
    original_decisions = latest_by_frame(read_jsonl(ORIGINAL_FREEZE / "review_decisions.jsonl"))
    supp_queue = read_json(SUPP_FREEZE / "supplemental_review_queue.json")["items"]
    supp_decisions = latest_by_frame(read_jsonl(SUPP_FREEZE / "supplemental_review_decisions.jsonl"))
    rows: Dict[str, Dict[str, Any]] = {}
    for item in original_queue:
        frame_id = item["frame_id"]
        decision = original_decisions.get(frame_id, {})
        rows[frame_id] = {
            "frame_id": frame_id,
            "sequence_id": item["sequence_id"],
            "source_review": "original",
            "source_freeze_id": ORIGINAL_FREEZE_ID,
            "source_review_hash": ORIGINAL_HASH,
            "source_split": item["split"],
            "split": original_effective_split(item),
            "status": item.get("status", "pending"),
            "bbox_xyxy": decision.get("bbox_xyxy"),
            "timestamp_sec": item.get("timestamp_sec_original") or item.get("timestamp_sec"),
            "frame_index": item.get("frame_index_original") or item.get("frame_index"),
            "image_path": source_image_path("original", item),
            "ground_truth": item.get("status") in {"reviewed_ball", "reviewed_no_ball"},
            "pseudo_label": False,
        }
    for item in supp_queue:
        frame_id = item["frame_id"]
        decision = supp_decisions.get(frame_id, {})
        rows[frame_id] = {
            "frame_id": frame_id,
            "sequence_id": item["sequence_id"],
            "source_review": "supplemental",
            "source_freeze_id": SUPP_FREEZE_ID,
            "source_review_hash": SUPP_HASH,
            "source_split": item["split"],
            "split": supplemental_effective_split(item),
            "status": item.get("status", "pending"),
            "bbox_xyxy": decision.get("bbox_xyxy"),
            "timestamp_sec": item.get("timestamp_sec_original") or item.get("timestamp_sec"),
            "frame_index": item.get("frame_index_original") or item.get("frame_index"),
            "image_path": source_image_path("supplemental", item),
            "ground_truth": item.get("status") in {"reviewed_ball", "reviewed_no_ball"},
            "pseudo_label": False,
        }
    return rows


def apply_corrections(correction_manifest: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = source_rows()
    queue_items = {item["frame_id"]: item for item in read_json(CORRECTION_REVIEW / "uncertain_correction_queue.json")["items"]}
    corrections = latest_by_frame(read_jsonl(CORRECTION_REVIEW / "uncertain_correction_decisions.jsonl"))
    errors = []
    applied = []
    rejected = []
    for frame_id, correction in corrections.items():
        source = rows.get(frame_id)
        queue_item = queue_items.get(frame_id)
        reasons = []
        if not source:
            reasons.append("frame_not_found_in_source_freezes")
        if not queue_item:
            reasons.append("frame_not_found_in_correction_queue")
        if source and source["status"] != "reviewed_uncertain":
            reasons.append(f"source_status_not_uncertain:{source['status']}")
        if queue_item and correction.get("old_status") != "reviewed_uncertain":
            reasons.append(f"correction_old_status_not_uncertain:{correction.get('old_status')}")
        if queue_item and correction.get("source_freeze_id") != queue_item.get("source_freeze_id"):
            reasons.append("source_freeze_id_mismatch_queue")
        if queue_item and correction.get("source_review_hash") != queue_item.get("source_review_hash"):
            reasons.append("source_review_hash_mismatch_queue")
        if source and correction.get("source_freeze_id") != source.get("source_freeze_id"):
            reasons.append("source_freeze_id_mismatch_source")
        if source and correction.get("source_review_hash") != source.get("source_review_hash"):
            reasons.append("source_review_hash_mismatch_source")
        if source and queue_item:
            if correction.get("sequence_id") != source.get("sequence_id") or queue_item.get("sequence_id") != source.get("sequence_id"):
                reasons.append("sequence_id_changed")
            if correction.get("split") != source.get("split") or queue_item.get("split") != source.get("split"):
                reasons.append("split_changed")
            if str(queue_item.get("timestamp_sec")) != str(source.get("timestamp_sec")):
                reasons.append("timestamp_changed")
        if reasons:
            rejected.append({"frame_id": frame_id, "reasons": reasons})
            continue
        if correction["new_status"] == "reviewed_ball":
            box_errors = validate_bbox(correction.get("final_bbox_xyxy"))
            if box_errors:
                rejected.append({"frame_id": frame_id, "reasons": box_errors})
                continue
        rows[frame_id]["status"] = correction["new_status"]
        rows[frame_id]["bbox_xyxy"] = correction.get("final_bbox_xyxy")
        rows[frame_id]["source_review"] = "uncertain_correction"
        rows[frame_id]["correction_freeze_id"] = correction_manifest["freeze_id"]
        rows[frame_id]["correction_review_hash"] = correction_manifest["review_hash"]
        rows[frame_id]["ground_truth"] = correction["new_status"] in {"reviewed_ball", "reviewed_no_ball"}
        rows[frame_id]["pseudo_label"] = False
        applied.append(frame_id)
    if rejected:
        errors.append(f"rejected_corrections:{len(rejected)}")
    final_rows = sorted(rows.values(), key=lambda row: (row.get("split") or "", row["sequence_id"], float(row.get("timestamp_sec") or 0), row["frame_id"]))
    report = {
        "created_at": utc_now(),
        "precedence": ["original", "supplemental", "uncertain_correction"],
        "corrections_seen": len(corrections),
        "corrections_applied": len(applied),
        "corrections_rejected": len(rejected),
        "rejected": rejected,
        "errors": errors,
    }
    return final_rows, report


def temporal_overlaps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seq = defaultdict(list)
    for row in rows:
        if row["split"] in {"train", "valid", "test"}:
            by_seq[row["sequence_id"]].append(float(row.get("timestamp_sec") or 0.0))
    ranges = {}
    seq_split = {}
    for seq, times in by_seq.items():
        ranges[seq] = (min(times), max(times))
        seq_split[seq] = next(row["split"] for row in rows if row["sequence_id"] == seq)
    overlaps = []
    seqs = sorted(ranges)
    for index, left in enumerate(seqs):
        for right in seqs[index + 1 :]:
            if seq_split[left] == seq_split[right]:
                continue
            overlap = max(0.0, min(ranges[left][1], ranges[right][1]) - max(ranges[left][0], ranges[right][0]))
            if overlap > 0:
                overlaps.append({"left": left, "right": right, "overlap_sec": round(overlap, 6)})
    return overlaps


def final_preflight(final_rows: List[Dict[str, Any]], freeze_integrity: Dict[str, Any], precedence: Dict[str, Any]) -> Dict[str, Any]:
    stats = defaultdict(Counter)
    seqs_by_split = defaultdict(set)
    positive_sequences = defaultdict(set)
    bbox_errors = []
    frame_ids = Counter()
    image_by_split = {}
    image_duplicates_between_splits = []
    sequence_splits = defaultdict(set)
    uncertain_included = []
    pseudo_labels = []
    pending = []
    dataset_rows = []
    uncertain_rows = []
    for row in final_rows:
        frame_ids[row["frame_id"]] += 1
        split = row["split"]
        status = row["status"]
        sequence_splits[row["sequence_id"]].add(split)
        if split not in {"train", "valid", "test"}:
            if status == "reviewed_uncertain":
                uncertain_rows.append(row)
            continue
        seqs_by_split[split].add(row["sequence_id"])
        stats[split][status] += 1
        if status == "pending":
            pending.append(row["frame_id"])
        if status == "reviewed_uncertain":
            uncertain_rows.append(row)
            continue
        if row.get("pseudo_label"):
            pseudo_labels.append(row["frame_id"])
        if row["image_path"] in image_by_split and image_by_split[row["image_path"]] != split:
            image_duplicates_between_splits.append({"image_path": row["image_path"], "splits": sorted([image_by_split[row["image_path"]], split])})
        image_by_split[row["image_path"]] = split
        if status == "reviewed_ball":
            errors = validate_bbox(row.get("bbox_xyxy"))
            if errors:
                bbox_errors.append({"frame_id": row["frame_id"], "errors": errors})
            positive_sequences[split].add(row["sequence_id"])
            dataset_rows.append(row)
        elif status == "reviewed_no_ball":
            if row.get("bbox_xyxy") is not None:
                bbox_errors.append({"frame_id": row["frame_id"], "errors": ["no_ball_has_bbox"]})
            dataset_rows.append(row)

    duplicate_ids = [frame_id for frame_id, count in frame_ids.items() if count > 1]
    sequence_leakage = {seq: sorted(splits) for seq, splits in sequence_splits.items() if len({s for s in splits if s in {"train", "valid", "test"}}) > 1}
    overlaps = temporal_overlaps(final_rows)

    gate_results = {
        "train_ball": {"value": stats["train"].get("reviewed_ball", 0), "minimum": GATES["train"]["ball"]},
        "train_no_ball": {"value": stats["train"].get("reviewed_no_ball", 0), "minimum": GATES["train"]["no_ball"]},
        "train_positive_sequences": {"value": len(positive_sequences["train"]), "minimum": GATES["train"]["positive_sequences"]},
        "valid_ball": {"value": stats["valid"].get("reviewed_ball", 0), "minimum": GATES["valid"]["ball"]},
        "valid_no_ball": {"value": stats["valid"].get("reviewed_no_ball", 0), "minimum": GATES["valid"]["no_ball"]},
        "test_ball": {"value": stats["test"].get("reviewed_ball", 0), "minimum": GATES["test"]["ball"]},
        "test_no_ball": {"value": stats["test"].get("reviewed_no_ball", 0), "minimum": GATES["test"]["no_ball"]},
    }
    for result in gate_results.values():
        result["passed"] = result["value"] >= result["minimum"]
        result["deficit"] = max(0, result["minimum"] - result["value"])

    validation_errors = []
    for name, result in gate_results.items():
        if not result["passed"]:
            validation_errors.append({"type": "gate_failed", "requirement": name, **result})
    if pending:
        validation_errors.append({"type": "pending_in_final_review", "count": len(pending)})
    if bbox_errors:
        validation_errors.append({"type": "bbox_errors", "count": len(bbox_errors)})
    if duplicate_ids:
        validation_errors.append({"type": "duplicate_frame_ids", "count": len(duplicate_ids)})
    if image_duplicates_between_splits:
        validation_errors.append({"type": "image_duplicates_between_splits", "count": len(image_duplicates_between_splits)})
    if sequence_leakage:
        validation_errors.append({"type": "sequence_leakage", "count": len(sequence_leakage)})
    if overlaps:
        validation_errors.append({"type": "temporal_overlap", "count": len(overlaps)})
    if uncertain_rows:
        validation_errors.append({"type": "uncertain_excluded_from_dataset", "count": len(uncertain_rows), "blocking": False})
    if pseudo_labels:
        validation_errors.append({"type": "pseudo_labels_included", "count": len(pseudo_labels)})
    if precedence["errors"]:
        validation_errors.append({"type": "decision_precedence_errors", "errors": precedence["errors"]})
    for name, report in freeze_integrity.items():
        if report["status"] != "passed":
            validation_errors.append({"type": "freeze_integrity_failed", "freeze": name, "errors": report["errors"]})

    blocking_errors = [row for row in validation_errors if row.get("blocking", True)]
    split_statistics = {
        split: {
            "sequences": len(seqs_by_split[split]),
            "ball": stats[split].get("reviewed_ball", 0),
            "no_ball": stats[split].get("reviewed_no_ball", 0),
            "uncertain": stats[split].get("reviewed_uncertain", 0),
            "pending": stats[split].get("pending", 0),
            "annotations": stats[split].get("reviewed_ball", 0),
            "positive_sequences": sorted(positive_sequences[split]),
            "positive_sequence_count": len(positive_sequences[split]),
        }
        for split in ["train", "valid", "test"]
    }
    sequence_statistics = {
        seq: {"splits": sorted(splits), "counts": dict(Counter(row["status"] for row in final_rows if row["sequence_id"] == seq))}
        for seq, splits in sorted(sequence_splits.items())
    }
    review_statistics = {
        "total_final_rows": len(final_rows),
        "dataset_candidate_rows": len(dataset_rows),
        "uncertain_excluded": len(uncertain_rows),
        "pending": len(pending),
        "status_counts": dict(Counter(row["status"] for row in final_rows)),
    }
    return {
        "status": "passed" if not blocking_errors else "blocked_dataset_validation",
        "validation_errors": validation_errors,
        "blocking_errors": blocking_errors,
        "gate_results": gate_results,
        "review_statistics": review_statistics,
        "split_statistics": split_statistics,
        "sequence_statistics": sequence_statistics,
        "bbox_validation": {"errors": bbox_errors, "error_count": len(bbox_errors)},
        "duplicate_report": {"duplicate_frame_ids": duplicate_ids, "images_between_splits": image_duplicates_between_splits},
        "temporal_overlap_report": {"overlaps": overlaps, "count": len(overlaps)},
        "leakage_report": {"sequence_leakage": sequence_leakage, "image_duplicates_between_splits": image_duplicates_between_splits},
        "dataset_validation": {
            "pending": len(pending),
            "bbox_errors": len(bbox_errors),
            "duplicate_frame_ids": len(duplicate_ids),
            "images_duplicated_between_splits": len(image_duplicates_between_splits),
            "sequence_leakage": len(sequence_leakage),
            "temporal_overlap": len(overlaps),
            "uncertain_included_in_dataset": 0,
            "pseudo_labels_included": len(pseudo_labels),
            "ground_truth_only": True,
            "coco_validation": "not_run_dataset_not_exported" if blocking_errors else "pending_export",
        },
    }


def write_combined_artifacts(final_rows: List[Dict[str, Any]], precedence: Dict[str, Any], correction_manifest: Dict[str, Any]) -> Dict[str, Any]:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": utc_now(),
        "phase": "PE-0A4 FINAL COMBINED PREFLIGHT AND DATASET EXPORT",
        "precedence": ["original", "supplemental", "uncertain_correction"],
        "original_freeze_id": ORIGINAL_FREEZE_ID,
        "original_review_hash": ORIGINAL_HASH,
        "supplemental_freeze_id": SUPP_FREEZE_ID,
        "supplemental_review_hash": SUPP_HASH,
        "correction_freeze_id": correction_manifest["freeze_id"],
        "correction_review_hash": correction_manifest["review_hash"],
        "final_decision_count": len(final_rows),
    }
    write_json(COMBINED_DIR / "decision_precedence_report.json", precedence)
    write_jsonl(COMBINED_DIR / "final_decisions.jsonl", final_rows)
    write_json(COMBINED_DIR / "final_review_manifest.json", manifest)
    write_json(COMBINED_DIR / "final_review_audit.json", {"created_at": utc_now(), "manifest": manifest, "precedence": precedence})
    return manifest


def write_preflight_artifacts(preflight: Dict[str, Any], freeze_integrity: Dict[str, Any], correction_manifest: Dict[str, Any], combined_manifest: Dict[str, Any]) -> None:
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(PREFLIGHT_DIR / "review_statistics.json", preflight["review_statistics"])
    write_json(PREFLIGHT_DIR / "split_statistics.json", preflight["split_statistics"])
    write_json(PREFLIGHT_DIR / "sequence_statistics.json", preflight["sequence_statistics"])
    write_json(PREFLIGHT_DIR / "bbox_validation.json", preflight["bbox_validation"])
    write_json(PREFLIGHT_DIR / "duplicate_report.json", preflight["duplicate_report"])
    write_json(PREFLIGHT_DIR / "temporal_overlap_report.json", preflight["temporal_overlap_report"])
    write_json(PREFLIGHT_DIR / "leakage_report.json", preflight["leakage_report"])
    write_json(PREFLIGHT_DIR / "freeze_integrity_report.json", freeze_integrity)
    write_json(PREFLIGHT_DIR / "dataset_validation.json", preflight["dataset_validation"])
    write_json(
        PREFLIGHT_DIR / "artifact_manifest.json",
        {
            "created_at": utc_now(),
            "status": preflight["status"],
            "artifacts": [
                "review_statistics.json",
                "split_statistics.json",
                "sequence_statistics.json",
                "bbox_validation.json",
                "duplicate_report.json",
                "temporal_overlap_report.json",
                "leakage_report.json",
                "freeze_integrity_report.json",
                "dataset_validation.json",
                "FINAL_PREFLIGHT_REPORT.md",
            ],
            "combined_manifest": combined_manifest,
            "correction_freeze_id": correction_manifest["freeze_id"],
            "correction_review_hash": correction_manifest["review_hash"],
        },
    )
    rows = [
        "# PE-0A4 Final Combined Preflight\n",
        f"- status: `{preflight['status']}`",
        f"- correction_freeze_id: `{correction_manifest['freeze_id']}`",
        f"- correction_hash: `{correction_manifest['review_hash']}`",
        "",
        "## Gates",
    ]
    for name, result in preflight["gate_results"].items():
        rows.append(f"- {name}: `{result['value']}/{result['minimum']}` passed=`{result['passed']}` deficit=`{result['deficit']}`")
    rows.extend(["", "## Split Statistics"])
    for split, stats in preflight["split_statistics"].items():
        rows.append(f"- {split}: sequences=`{stats['sequences']}`, ball=`{stats['ball']}`, no_ball=`{stats['no_ball']}`, uncertain=`{stats['uncertain']}`, annotations=`{stats['annotations']}`")
    rows.extend(["", "## Blocking Errors"])
    if preflight["blocking_errors"]:
        for error in preflight["blocking_errors"]:
            rows.append(f"- `{error}`")
    else:
        rows.append("- none")
    rows.extend(["", "## Dataset Export"])
    rows.append("- not exported because final preflight is blocked" if preflight["status"] != "passed" else "- ready for export")
    write_text(PREFLIGHT_DIR / "FINAL_PREFLIGHT_REPORT.md", "\n".join(rows) + "\n")


def main() -> None:
    completion = correction_completion()
    if completion["pending"] > 0 or completion["status"] != "passed":
        write_json(PREFLIGHT_DIR / "blocked_review_incomplete.json", completion)
        print(json.dumps({"status": "blocked_review_incomplete", "completion": completion}, indent=2, sort_keys=True))
        return

    correction_manifest = freeze_correction(completion)
    freeze_integrity = {
        "original": verify_freeze(ORIGINAL_FREEZE, "review_frozen_manifest.json", ORIGINAL_FREEZE_ID, ORIGINAL_HASH),
        "supplemental": verify_freeze(SUPP_FREEZE, "supplemental_review_frozen_manifest.json", SUPP_FREEZE_ID, SUPP_HASH),
        "correction": verify_freeze(
            CORRECTION_REVIEW / "frozen" / correction_manifest["freeze_id"],
            "uncertain_correction_frozen_manifest.json",
            correction_manifest["freeze_id"],
            correction_manifest["review_hash"],
        ),
    }
    final_rows, precedence = apply_corrections(correction_manifest)
    combined_manifest = write_combined_artifacts(final_rows, precedence, correction_manifest)
    preflight = final_preflight(final_rows, freeze_integrity, precedence)
    write_preflight_artifacts(preflight, freeze_integrity, correction_manifest, combined_manifest)

    summary = {
        "phase": "PE-0A4 FINAL COMBINED PREFLIGHT AND DATASET EXPORT",
        "status": "dataset_ready" if preflight["status"] == "passed" else preflight["status"],
        "correction": {
            "freeze_id": correction_manifest["freeze_id"],
            "hash": correction_manifest["review_hash"],
            "reviewed_ball": completion["reviewed_ball"],
            "reviewed_no_ball": completion["reviewed_no_ball"],
            "reviewed_uncertain": completion["reviewed_uncertain"],
            "pending": completion["pending"],
            "integrity": freeze_integrity["correction"]["status"],
        },
        "splits": preflight["split_statistics"],
        "gates": preflight["gate_results"],
        "validation": preflight["dataset_validation"],
        "final_preflight_dir": str(PREFLIGHT_DIR),
        "combined_review_dir": str(COMBINED_DIR),
        "dataset_path": str(DATASET_DIR) if preflight["status"] == "passed" else None,
        "dataset_exported": preflight["status"] == "passed",
        "blocking_errors": preflight["blocking_errors"],
    }
    write_json(CORRECTION_RUN / "PE0A4_FINAL_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
