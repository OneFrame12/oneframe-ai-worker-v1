#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import quantiles
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
DATASET_DIR = AI_WORKER_ROOT / "datasets" / "OneFrame_Ball_v0"
sys.path.insert(0, str(AI_WORKER_ROOT / "src"))

from dataset_hashing import compute_bundle_hash, compute_provenance_hash, compute_training_payload_hash  # noqa: E402

ORIGINAL_RUN = AI_WORKER_ROOT / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z"
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"
ORIGINAL_FREEZE = ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID

SUPP_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_supplemental_ball_20260715T060000Z"
SUPP_FREEZE_ID = "supplemental_review_freeze_20260715T163151Z"
SUPP_HASH = "1ac2d800a46290c9142836ad4a3295425efe4d03445e11524f820081bccf7912"
SUPP_FREEZE = SUPP_RUN / "review" / "frozen" / SUPP_FREEZE_ID

CORR_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_uncertain_correction_20260715T165053Z"
CORR_FREEZE_ID = "uncertain_correction_freeze_20260715T172940Z"
CORR_HASH = "ea179083ed2a411305b572e446ec98d27470275c8ec9164ea42aa4bb2351abaa"
CORR_FREEZE = CORR_RUN / "review" / "frozen" / CORR_FREEZE_ID

POS_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_positive_completion_20260715T174023Z"
POS_REVIEW = POS_RUN / "review"

COMBINED_DIR = POS_RUN / "combined_review"
PREFLIGHT_DIR = POS_RUN / "final_preflight"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
CROP_SIZE = 512
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


def validate_bbox(box: Any, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> List[str]:
    if not isinstance(box, list) or len(box) != 4:
        return ["bbox_missing_or_invalid_shape"]
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:  # noqa: BLE001
        return ["bbox_non_numeric"]
    errors = []
    if x2 <= x1 or y2 <= y1:
        errors.append("bbox_zero_or_negative_area")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
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
    return {"status": "passed" if not errors else "failed", "errors": errors, "freeze_id": freeze_id, "review_hash": review_hash}


def positive_completion_status() -> Dict[str, Any]:
    queue = read_json(POS_REVIEW / "positive_completion_queue.json")["items"]
    latest = latest_by_frame(read_jsonl(POS_REVIEW / "positive_completion_decisions.jsonl"))
    counts = Counter()
    by_split = defaultdict(Counter)
    invalid = []
    missing = []
    for item in queue:
        decision = latest.get(item["frame_id"])
        if not decision:
            missing.append(item["frame_id"])
            continue
        status = decision["new_status"]
        counts[status] += 1
        by_split[item["split"]][status] += 1
        if decision.get("reviewed_by") != "human":
            invalid.append({"frame_id": item["frame_id"], "error": "non_human_decision"})
        if status == "reviewed_ball":
            box_errors = validate_bbox(decision.get("final_bbox_xyxy"))
            if box_errors:
                invalid.append({"frame_id": item["frame_id"], "error": box_errors})
        elif decision.get("final_bbox_xyxy") is not None:
            invalid.append({"frame_id": item["frame_id"], "error": "non_ball_has_bbox"})
    return {
        "queue_items": len(queue),
        "unique_decisions": len(latest),
        "pending": len(missing),
        "reviewed_ball": counts.get("reviewed_ball", 0),
        "reviewed_no_ball": counts.get("reviewed_no_ball", 0),
        "reviewed_uncertain": counts.get("reviewed_uncertain", 0),
        "by_split": {split: dict(counter) for split, counter in by_split.items()},
        "missing": missing,
        "invalid": invalid,
        "status": "passed" if not missing and not invalid and len(queue) == len(latest) else "failed",
    }


def freeze_positive_review(status: Dict[str, Any]) -> Dict[str, Any]:
    freeze_id = f"positive_completion_freeze_{compact_now()}"
    freeze_dir = POS_REVIEW / "frozen" / freeze_id
    if freeze_dir.exists():
        raise FileExistsError(freeze_dir)
    freeze_dir.mkdir(parents=True)
    file_names = [
        "positive_completion_queue.json",
        "positive_completion_decisions.jsonl",
        "positive_completion_progress.json",
        "positive_completion_audit_log.jsonl",
        "positive_completion_manifest.json",
    ]
    frozen_files = []
    for name in file_names:
        src = POS_REVIEW / name
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
    review_hash = sha256_text(payload)
    manifest = {
        "created_at": utc_now(),
        "freeze_id": freeze_id,
        "review_hash": review_hash,
        "source_run_dir": str(POS_RUN),
        "source_review_dir": str(POS_REVIEW),
        "frozen_files": frozen_files,
        "completion": status,
        "previous_freezes": {
            "original": {"freeze_id": ORIGINAL_FREEZE_ID, "review_hash": ORIGINAL_HASH},
            "supplemental": {"freeze_id": SUPP_FREEZE_ID, "review_hash": SUPP_HASH},
            "uncertain_correction": {"freeze_id": CORR_FREEZE_ID, "review_hash": CORR_HASH},
        },
    }
    write_json(freeze_dir / "positive_completion_frozen_manifest.json", manifest)
    write_text(freeze_dir / "positive_completion_hash.txt", review_hash + "\n")
    write_text(
        freeze_dir / "positive_completion_report.md",
        "# Positive Completion Freeze\n\n"
        f"- freeze_id: `{freeze_id}`\n"
        f"- review_hash: `{review_hash}`\n"
        f"- reviewed_ball: `{status['reviewed_ball']}`\n"
        f"- reviewed_no_ball: `{status['reviewed_no_ball']}`\n"
        f"- reviewed_uncertain: `{status['reviewed_uncertain']}`\n"
        f"- pending: `{status['pending']}`\n",
    )
    write_json(POS_REVIEW / "positive_completion_frozen_manifest.json", manifest)
    return manifest


def get_or_create_positive_freeze(status: Dict[str, Any]) -> Dict[str, Any]:
    existing = POS_REVIEW / "positive_completion_frozen_manifest.json"
    if existing.exists():
        manifest = read_json(existing)
        freeze_dir = POS_REVIEW / "frozen" / manifest["freeze_id"]
        verified = verify_freeze(
            freeze_dir,
            "positive_completion_frozen_manifest.json",
            manifest["freeze_id"],
            manifest["review_hash"],
        )
        if verified["status"] == "passed":
            return manifest
    return freeze_positive_review(status)


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


def source_path(base: Path, item: Dict[str, Any]) -> str:
    rel = item.get("frame_image") or item.get("image_path") or item.get("context", {}).get("current_image")
    return str(base / rel) if rel else ""


def base_source_rows() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    original_queue = read_json(ORIGINAL_FREEZE / "review_queue.json")["items"]
    original_decisions = latest_by_frame(read_jsonl(ORIGINAL_FREEZE / "review_decisions.jsonl"))
    for item in original_queue:
        decision = original_decisions.get(item["frame_id"], {})
        rows[item["frame_id"]] = {
            "frame_id": item["frame_id"],
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
            "image_path": source_path(ORIGINAL_RUN, item),
            "ground_truth": item.get("status") in {"reviewed_ball", "reviewed_no_ball"},
            "pseudo_label": False,
        }
    supp_queue = read_json(SUPP_FREEZE / "supplemental_review_queue.json")["items"]
    supp_decisions = latest_by_frame(read_jsonl(SUPP_FREEZE / "supplemental_review_decisions.jsonl"))
    for item in supp_queue:
        decision = supp_decisions.get(item["frame_id"], {})
        rows[item["frame_id"]] = {
            "frame_id": item["frame_id"],
            "sequence_id": item["sequence_id"],
            "source_review": "supplemental",
            "source_freeze_id": SUPP_FREEZE_ID,
            "source_review_hash": SUPP_HASH,
            "source_split": item["split"],
            "split": supplemental_effective_split(item),
            "status": item.get("status", "pending"),
            "bbox_xyxy": decision.get("bbox_xyxy"),
            "timestamp_sec": item.get("timestamp_sec_original") or item.get("timestamp_sec"),
            "frame_index": item.get("frame_index_original"),
            "image_path": source_path(SUPP_RUN, item),
            "ground_truth": item.get("status") in {"reviewed_ball", "reviewed_no_ball"},
            "pseudo_label": False,
        }
    return rows


def apply_uncertain_correction(rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    corrections = latest_by_frame(read_jsonl(CORR_FREEZE / "uncertain_correction_decisions.jsonl"))
    rejected = []
    applied = 0
    for frame_id, corr in corrections.items():
        source = rows.get(frame_id)
        reasons = []
        if not source:
            reasons.append("frame_not_found")
        elif source["status"] != "reviewed_uncertain":
            reasons.append(f"source_status_not_uncertain:{source['status']}")
        if source and corr.get("source_freeze_id") != source.get("source_freeze_id"):
            reasons.append("source_freeze_id_mismatch")
        if source and corr.get("source_review_hash") != source.get("source_review_hash"):
            reasons.append("source_review_hash_mismatch")
        if source and corr.get("split") != source.get("split"):
            reasons.append("split_changed")
        if reasons:
            rejected.append({"frame_id": frame_id, "reasons": reasons})
            continue
        source["status"] = corr["new_status"]
        source["bbox_xyxy"] = corr.get("final_bbox_xyxy")
        source["source_review"] = "uncertain_correction"
        source["correction_freeze_id"] = CORR_FREEZE_ID
        source["correction_review_hash"] = CORR_HASH
        source["ground_truth"] = corr["new_status"] in {"reviewed_ball", "reviewed_no_ball"}
        source["pseudo_label"] = False
        applied += 1
    return {"corrections_seen": len(corrections), "corrections_applied": applied, "corrections_rejected": len(rejected), "rejected": rejected}


def add_positive_completion(rows: Dict[str, Dict[str, Any]], positive_manifest: Dict[str, Any]) -> Dict[str, Any]:
    queue = {item["frame_id"]: item for item in read_json(positive_manifest_path(positive_manifest) / "positive_completion_queue.json")["items"]}
    decisions = latest_by_frame(read_jsonl(positive_manifest_path(positive_manifest) / "positive_completion_decisions.jsonl"))
    rejected = []
    for frame_id, decision in decisions.items():
        item = queue.get(frame_id)
        reasons = []
        if not item:
            reasons.append("missing_queue_item")
        if frame_id in rows:
            reasons.append("duplicate_frame_id")
        if item and item["split"] not in {"train", "valid"}:
            reasons.append("invalid_positive_split")
        if decision["new_status"] == "reviewed_ball" and validate_bbox(decision.get("final_bbox_xyxy")):
            reasons.append("invalid_bbox")
        if reasons:
            rejected.append({"frame_id": frame_id, "reasons": reasons})
            continue
        rows[frame_id] = {
            "frame_id": frame_id,
            "sequence_id": item["sequence_id"],
            "source_review": "positive_completion",
            "source_freeze_id": positive_manifest["freeze_id"],
            "source_review_hash": positive_manifest["review_hash"],
            "source_split": item["split"],
            "split": item["split"],
            "status": decision["new_status"],
            "bbox_xyxy": decision.get("final_bbox_xyxy"),
            "timestamp_sec": item.get("timestamp_sec"),
            "frame_index": item.get("frame_index_original"),
            "image_path": str(POS_RUN / item["context"]["current_image"]),
            "ground_truth": decision["new_status"] in {"reviewed_ball", "reviewed_no_ball"},
            "pseudo_label": False,
        }
    return {"positive_seen": len(decisions), "positive_added": len(decisions) - len(rejected), "positive_rejected": len(rejected), "rejected": rejected}


def positive_manifest_path(positive_manifest: Dict[str, Any]) -> Path:
    return POS_REVIEW / "frozen" / positive_manifest["freeze_id"]


def sorted_rows(rows: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows.values(), key=lambda row: (row.get("split") or "", row["sequence_id"], float(row.get("timestamp_sec") or 0), row["frame_id"]))


def validate_final(rows: List[Dict[str, Any]], freeze_integrity: Dict[str, Any], precedence: Dict[str, Any]) -> Dict[str, Any]:
    split_stats = defaultdict(Counter)
    seqs_by_split = defaultdict(set)
    positive_sequences = defaultdict(set)
    bbox_errors = []
    frame_counter = Counter()
    image_split = {}
    image_dupes = []
    seq_splits = defaultdict(set)
    pseudo = []
    pending = []
    uncertain = []
    dataset_rows = []
    for row in rows:
        frame_counter[row["frame_id"]] += 1
        split = row["split"]
        status = row["status"]
        seq_splits[row["sequence_id"]].add(split)
        if split in {"train", "valid", "test"}:
            seqs_by_split[split].add(row["sequence_id"])
            split_stats[split][status] += 1
        if status == "pending":
            pending.append(row["frame_id"])
        if status == "reviewed_uncertain":
            uncertain.append(row["frame_id"])
            continue
        if split not in {"train", "valid", "test"}:
            continue
        if row.get("pseudo_label"):
            pseudo.append(row["frame_id"])
        if row["image_path"] in image_split and image_split[row["image_path"]] != split:
            image_dupes.append({"image_path": row["image_path"], "splits": sorted([image_split[row["image_path"]], split])})
        image_split[row["image_path"]] = split
        if status == "reviewed_ball":
            errs = validate_bbox(row.get("bbox_xyxy"))
            if errs:
                bbox_errors.append({"frame_id": row["frame_id"], "errors": errs})
            positive_sequences[split].add(row["sequence_id"])
            dataset_rows.append(row)
        elif status == "reviewed_no_ball":
            if row.get("bbox_xyxy") is not None:
                bbox_errors.append({"frame_id": row["frame_id"], "errors": ["no_ball_has_bbox"]})
            dataset_rows.append(row)
    seq_leakage = {seq: sorted({s for s in splits if s in {"train", "valid", "test"}}) for seq, splits in seq_splits.items() if len({s for s in splits if s in {"train", "valid", "test"}}) > 1}
    dup_ids = [fid for fid, count in frame_counter.items() if count > 1]
    overlaps = temporal_overlaps(rows)
    gates = {
        "train_ball": {"value": split_stats["train"].get("reviewed_ball", 0), "minimum": GATES["train"]["ball"]},
        "train_no_ball": {"value": split_stats["train"].get("reviewed_no_ball", 0), "minimum": GATES["train"]["no_ball"]},
        "train_positive_sequences": {"value": len(positive_sequences["train"]), "minimum": GATES["train"]["positive_sequences"]},
        "valid_ball": {"value": split_stats["valid"].get("reviewed_ball", 0), "minimum": GATES["valid"]["ball"]},
        "valid_no_ball": {"value": split_stats["valid"].get("reviewed_no_ball", 0), "minimum": GATES["valid"]["no_ball"]},
        "test_ball": {"value": split_stats["test"].get("reviewed_ball", 0), "minimum": GATES["test"]["ball"]},
        "test_no_ball": {"value": split_stats["test"].get("reviewed_no_ball", 0), "minimum": GATES["test"]["no_ball"]},
    }
    errors = []
    for name, gate in gates.items():
        gate["passed"] = gate["value"] >= gate["minimum"]
        gate["deficit"] = max(0, gate["minimum"] - gate["value"])
        if not gate["passed"]:
            errors.append({"type": "gate_failed", "requirement": name, **gate})
    if pending:
        errors.append({"type": "pending", "count": len(pending)})
    if bbox_errors:
        errors.append({"type": "bbox_errors", "count": len(bbox_errors)})
    if dup_ids:
        errors.append({"type": "duplicate_frame_ids", "count": len(dup_ids)})
    if image_dupes:
        errors.append({"type": "image_duplicates_between_splits", "count": len(image_dupes)})
    if seq_leakage:
        errors.append({"type": "sequence_leakage", "count": len(seq_leakage)})
    if overlaps:
        errors.append({"type": "temporal_overlap", "count": len(overlaps)})
    if pseudo:
        errors.append({"type": "pseudo_labels_included", "count": len(pseudo)})
    for name, report in freeze_integrity.items():
        if report["status"] != "passed":
            errors.append({"type": "freeze_integrity_failed", "freeze": name})
    if precedence["uncertain_correction"]["corrections_rejected"] or precedence["positive_completion"]["positive_rejected"]:
        errors.append({"type": "precedence_rejections", "precedence": precedence})
    split_statistics = {
        split: {
            "sequences": len(seqs_by_split[split]),
            "ball": split_stats[split].get("reviewed_ball", 0),
            "no_ball": split_stats[split].get("reviewed_no_ball", 0),
            "uncertain": split_stats[split].get("reviewed_uncertain", 0),
            "annotations": split_stats[split].get("reviewed_ball", 0),
            "positive_sequences": sorted(positive_sequences[split]),
            "positive_sequence_count": len(positive_sequences[split]),
        }
        for split in ["train", "valid", "test"]
    }
    return {
        "status": "passed" if not errors else "blocked_dataset_validation",
        "validation_errors": errors,
        "gate_results": gates,
        "split_statistics": split_statistics,
        "dataset_rows": dataset_rows,
        "uncertain_excluded": len(uncertain),
        "review_statistics": {
            "total_final_rows": len(rows),
            "dataset_candidate_rows": len(dataset_rows),
            "uncertain_excluded": len(uncertain),
            "status_counts": dict(Counter(row["status"] for row in rows)),
        },
        "bbox_validation": {"errors": bbox_errors, "error_count": len(bbox_errors)},
        "duplicate_report": {"duplicate_frame_ids": dup_ids, "images_between_splits": image_dupes},
        "temporal_overlap_report": {"overlaps": overlaps, "count": len(overlaps)},
        "leakage_report": {"sequence_leakage": seq_leakage, "image_duplicates_between_splits": image_dupes},
        "dataset_validation": {
            "pending": len(pending),
            "bbox_errors": len(bbox_errors),
            "duplicate_frame_ids": len(dup_ids),
            "images_duplicated_between_splits": len(image_dupes),
            "sequence_leakage": len(seq_leakage),
            "temporal_overlap": len(overlaps),
            "uncertain_included_in_dataset": 0,
            "pseudo_labels_included": len(pseudo),
            "ground_truth_only": True,
        },
    }


def temporal_overlaps(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_seq = defaultdict(list)
    seq_split = {}
    for row in rows:
        if row["split"] in {"train", "valid", "test"}:
            by_seq[row["sequence_id"]].append(float(row.get("timestamp_sec") or 0.0))
            seq_split[row["sequence_id"]] = row["split"]
    ranges = {seq: (min(values), max(values)) for seq, values in by_seq.items() if values}
    overlaps = []
    seqs = sorted(ranges)
    for i, left in enumerate(seqs):
        for right in seqs[i + 1 :]:
            if seq_split[left] == seq_split[right]:
                continue
            overlap = max(0.0, min(ranges[left][1], ranges[right][1]) - max(ranges[left][0], ranges[right][0]))
            if overlap > 0:
                overlaps.append({"left": left, "right": right, "overlap_sec": round(overlap, 6)})
    return overlaps


def crop_box(row: Dict[str, Any], image_shape: Tuple[int, int, int]) -> Tuple[List[int], List[int], List[str]]:
    height, width = image_shape[:2]
    tags: List[str] = []
    if row["status"] == "reviewed_ball":
        x1, y1, x2, y2 = [float(v) for v in row["bbox_xyxy"]]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        jitter_seed = int(hashlib.sha256(row["frame_id"].encode()).hexdigest()[:8], 16)
        jitter_x = ((jitter_seed % 31) - 15) * 2
        jitter_y = (((jitter_seed // 31) % 31) - 15) * 2
        cx += jitter_x
        cy += jitter_y
    else:
        cx, cy = width / 2.0, height / 2.0
        tags = ["manual_no_ball_context"]
    left = int(round(cx - CROP_SIZE / 2))
    top = int(round(cy - CROP_SIZE / 2))
    left = max(0, min(left, width - CROP_SIZE))
    top = max(0, min(top, height - CROP_SIZE))
    right = min(width, left + CROP_SIZE)
    bottom = min(height, top + CROP_SIZE)
    padding = [0, 0, max(0, CROP_SIZE - (right - left)), max(0, CROP_SIZE - (bottom - top))]
    return [left, top, right, bottom], padding, tags


def bbox_to_crop(box: List[float], crop: List[int]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx1, cy1, _, _ = crop
    return [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def xyxy_to_coco(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)]


def make_contact_sheet(image_paths: List[Path], out: Path, title: str, limit: int = 36) -> None:
    thumbs = []
    for path in image_paths[:limit]:
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.resize(img, (160, 160))
        cv2.putText(img, path.stem[:18], (5, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(img)
    if not thumbs:
        return
    while len(thumbs) % 6:
        thumbs.append(np.zeros_like(thumbs[0]))
    rows = [cv2.hconcat(thumbs[i : i + 6]) for i in range(0, len(thumbs), 6)]
    sheet = cv2.vconcat(rows)
    cv2.putText(sheet, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)


def export_dataset(dataset_rows: List[Dict[str, Any]], freeze_integrity: Dict[str, Any], precedence: Dict[str, Any], positive_manifest: Dict[str, Any], preflight: Dict[str, Any]) -> Dict[str, Any]:
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)
    for split in ["train", "valid", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    for sub in ["annotations", "manifests", "audit", "contact_sheets"]:
        (DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    coco = {split: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "ball"}]} for split in ["train", "valid", "test"]}
    manifests = {split: [] for split in ["train", "valid", "test"]}
    crop_rows = []
    negative_rows = []
    uncertain_rows = [row for row in sorted_rows(base_source_rows()) if row["status"] == "reviewed_uncertain"]
    annotation_id = 1
    image_id = 1
    for row in dataset_rows:
        split = row["split"]
        source = Path(row["image_path"])
        img = cv2.imread(str(source))
        if img is None:
            raise RuntimeError(f"cannot_read_source_image:{source}")
        crop, padding, hard_tags = crop_box(row, img.shape)
        crop_img = img[crop[1] : crop[3], crop[0] : crop[2]]
        if crop_img.shape[0] != CROP_SIZE or crop_img.shape[1] != CROP_SIZE:
            crop_img = cv2.copyMakeBorder(crop_img, 0, CROP_SIZE - crop_img.shape[0], 0, CROP_SIZE - crop_img.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0))
        crop_id = f"{row['frame_id']}_crop"
        out_rel = Path("images") / split / f"{crop_id}.jpg"
        out_path = DATASET_DIR / out_rel
        cv2.imwrite(str(out_path), crop_img)
        positive = row["status"] == "reviewed_ball"
        coco[split]["images"].append(
            {
                "id": image_id,
                "file_name": str(Path(split) / f"{crop_id}.jpg"),
                "width": CROP_SIZE,
                "height": CROP_SIZE,
                "frame_id": row["frame_id"],
                "sequence_id": row["sequence_id"],
                "ground_truth": True,
                "pseudo_label": False,
            }
        )
        crop_row = {
            "crop_id": crop_id,
            "source_frame_id": row["frame_id"],
            "sequence_id": row["sequence_id"],
            "split": split,
            "image_relpath": str(out_rel),
            "source_image_path": row["image_path"],
            "crop_xyxy_original": crop,
            "padding": padding,
            "ball_bbox_original": row.get("bbox_xyxy") if positive else None,
            "ball_bbox_crop": bbox_to_crop(row["bbox_xyxy"], crop) if positive else None,
            "positive": positive,
            "hard_negative_tags": hard_tags,
            "sha256": sha256_file(out_path),
        }
        crop_rows.append(crop_row)
        manifests[split].append(crop_row)
        if positive:
            bbox_crop_xyxy = bbox_to_crop(row["bbox_xyxy"], crop)
            bbox_coco = xyxy_to_coco(bbox_crop_xyxy)
            coco[split]["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": bbox_coco,
                    "area": round(bbox_coco[2] * bbox_coco[3], 3),
                    "iscrowd": 0,
                    "source_frame_id": row["frame_id"],
                    "ground_truth": True,
                    "pseudo_label": False,
                }
            )
            annotation_id += 1
        else:
            negative_rows.append(crop_row)
        image_id += 1

    for split in ["train", "valid", "test"]:
        write_json(DATASET_DIR / "annotations" / f"instances_{split}.json", coco[split])
        write_json(DATASET_DIR / "manifests" / f"{split}_manifest.json", {"items": manifests[split]})
    write_json(DATASET_DIR / "manifests" / "crop_manifest.json", {"crops": crop_rows})
    write_json(DATASET_DIR / "manifests" / "negative_manifest.json", {"items": negative_rows})
    write_json(DATASET_DIR / "manifests" / "uncertain_manifest.json", {"items": uncertain_rows})
    write_json(DATASET_DIR / "manifests" / "split_manifest.json", preflight["split_statistics"])
    write_json(DATASET_DIR / "manifests" / "source_frames_manifest.json", {"items": dataset_rows})
    write_json(DATASET_DIR / "audit" / "original_review_frozen_manifest.json", read_json(ORIGINAL_FREEZE / "review_frozen_manifest.json"))
    write_json(DATASET_DIR / "audit" / "supplemental_review_frozen_manifest.json", read_json(SUPP_FREEZE / "supplemental_review_frozen_manifest.json"))
    write_json(DATASET_DIR / "audit" / "correction_review_frozen_manifest.json", read_json(CORR_FREEZE / "uncertain_correction_frozen_manifest.json"))
    write_json(DATASET_DIR / "audit" / "positive_completion_frozen_manifest.json", positive_manifest)
    write_json(DATASET_DIR / "audit" / "decision_precedence_report.json", precedence)
    write_json(DATASET_DIR / "audit" / "combined_review_manifest.json", {"created_at": utc_now(), "freeze_integrity": freeze_integrity})
    write_json(DATASET_DIR / "audit" / "leakage_report.json", preflight["leakage_report"])

    qa = validate_coco_and_qa(coco, crop_rows)
    write_json(DATASET_DIR / "audit" / "annotation_audit.json", qa["annotation_audit"])
    write_json(DATASET_DIR / "DATASET_QA_REPORT.json", qa)
    make_dataset_docs(preflight, qa, crop_rows)
    write_contact_sheets(crop_rows)
    dataset_hash = compute_dataset_hash(DATASET_DIR)
    training_payload_v1 = compute_training_payload_hash(DATASET_DIR)
    canonical_hashes = {
        "hash_spec_version": training_payload_v1["hash_spec_version"],
        "training_payload": training_payload_v1,
        "provenance": compute_provenance_hash(DATASET_DIR, training_payload_v1["training_payload_hash"]),
        "bundle": compute_bundle_hash(DATASET_DIR, exclude_paths={"canonical_hashes.json", "training_payload_hash_v1.txt"}),
    }
    write_text(DATASET_DIR / "dataset_hash.txt", dataset_hash + "\n")
    write_text(DATASET_DIR / "training_payload_hash_v1.txt", training_payload_v1["training_payload_hash"] + "\n")
    write_json(DATASET_DIR / "canonical_hashes.json", canonical_hashes)
    artifact_manifest = {
        "created_at": utc_now(),
        "dataset": "OneFrame_Ball_v0",
        "dataset_hash": dataset_hash,
        "artifacts": sorted(str(path.relative_to(DATASET_DIR)) for path in DATASET_DIR.rglob("*") if path.is_file()),
    }
    write_json(DATASET_DIR / "artifact_manifest.json", artifact_manifest)
    qa["dataset_hash"] = dataset_hash
    return qa


def validate_coco_and_qa(coco: Dict[str, Any], crop_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    stats = {}
    widths = []
    heights = []
    areas = []
    edge_boxes = 0
    for split, data in coco.items():
        image_ids = {img["id"] for img in data["images"]}
        ann_ids = [ann["id"] for ann in data["annotations"]]
        if len(ann_ids) != len(set(ann_ids)):
            errors.append(f"duplicate_annotation_ids:{split}")
        for image in data["images"]:
            if not (DATASET_DIR / "images" / image["file_name"]).exists():
                errors.append(f"missing_image:{image['file_name']}")
        annotated_ids = set()
        for ann in data["annotations"]:
            if ann["image_id"] not in image_ids:
                errors.append(f"annotation_image_missing:{ann['id']}")
            if ann["category_id"] != 1:
                errors.append(f"invalid_category:{ann['id']}")
            x, y, w, h = ann["bbox"]
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > CROP_SIZE or y + h > CROP_SIZE:
                errors.append(f"bbox_out_of_crop:{ann['id']}")
            if x < 3 or y < 3 or x + w > CROP_SIZE - 3 or y + h > CROP_SIZE - 3:
                edge_boxes += 1
            widths.append(w)
            heights.append(h)
            areas.append(ann["area"])
            annotated_ids.add(ann["image_id"])
        stats[split] = {
            "images": len(data["images"]),
            "positives": len(data["annotations"]),
            "negatives": len(data["images"]) - len(data["annotations"]),
            "annotations": len(data["annotations"]),
        }
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "split_stats": stats,
        "bbox_width_percentiles": percentiles(widths),
        "bbox_height_percentiles": percentiles(heights),
        "bbox_area_percentiles": percentiles(areas),
        "edge_boxes": edge_boxes,
        "padded_crops": sum(1 for row in crop_rows if any(row["padding"])),
        "hard_negative_tags": dict(Counter(tag for row in crop_rows for tag in row["hard_negative_tags"])),
        "annotation_audit": {
            "json_valid": True,
            "category_id": 1,
            "negatives_without_annotations": True,
            "ground_truth_traceable": True,
            "pseudo_labels": 0,
            "uncertain": 0,
        },
    }


def percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {name: None for name in ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]}
    ordered = sorted(values)
    def pick(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return round(float(ordered[idx]), 3)
    return {"p01": pick(0.01), "p05": pick(0.05), "p10": pick(0.10), "p25": pick(0.25), "p50": pick(0.50), "p75": pick(0.75), "p90": pick(0.90), "p95": pick(0.95), "p99": pick(0.99)}


def write_contact_sheets(crop_rows: List[Dict[str, Any]]) -> None:
    rows = [row for row in crop_rows]
    def paths(filter_fn):
        return [DATASET_DIR / row["image_relpath"] for row in rows if filter_fn(row)]
    make_contact_sheet(paths(lambda r: r["split"] == "train" and r["positive"]), DATASET_DIR / "contact_sheets" / "train_positives.jpg", "train positives")
    make_contact_sheet(paths(lambda r: r["split"] == "train" and not r["positive"]), DATASET_DIR / "contact_sheets" / "train_negatives.jpg", "train negatives")
    make_contact_sheet(paths(lambda r: r["split"] == "valid"), DATASET_DIR / "contact_sheets" / "valid.jpg", "valid")
    make_contact_sheet(paths(lambda r: r["split"] == "test"), DATASET_DIR / "contact_sheets" / "test.jpg", "test")
    positives = [r for r in rows if r["positive"] and r["ball_bbox_crop"]]
    smallest = sorted(positives, key=lambda r: (r["ball_bbox_crop"][2] - r["ball_bbox_crop"][0]) * (r["ball_bbox_crop"][3] - r["ball_bbox_crop"][1]))[:36]
    largest = sorted(positives, key=lambda r: (r["ball_bbox_crop"][2] - r["ball_bbox_crop"][0]) * (r["ball_bbox_crop"][3] - r["ball_bbox_crop"][1]), reverse=True)[:36]
    make_contact_sheet([DATASET_DIR / r["image_relpath"] for r in smallest], DATASET_DIR / "contact_sheets" / "smallest_balls.jpg", "smallest balls")
    make_contact_sheet([DATASET_DIR / r["image_relpath"] for r in largest], DATASET_DIR / "contact_sheets" / "largest_balls.jpg", "largest balls")
    make_contact_sheet(paths(lambda r: not r["positive"]), DATASET_DIR / "contact_sheets" / "hard_negatives.jpg", "hard negatives")
    make_contact_sheet(paths(lambda r: any(r["padding"])), DATASET_DIR / "contact_sheets" / "padded_crops.jpg", "padded crops")


def make_dataset_docs(preflight: Dict[str, Any], qa: Dict[str, Any], crop_rows: List[Dict[str, Any]]) -> None:
    split_stats = qa["split_stats"]
    write_text(
        DATASET_DIR / "DATASET_CARD.md",
        "# OneFrame_Ball_v0\n\n"
        "- format: COCO Detection\n"
        "- category: `ball` id `1`\n"
        "- labels: human reviewed ground truth only\n"
        "- pseudo labels: `0`\n"
        "- uncertain frames: excluded\n\n"
        "## Splits\n"
        + "\n".join(f"- {split}: images `{v['images']}`, positives `{v['positives']}`, negatives `{v['negatives']}`" for split, v in split_stats.items())
        + "\n",
    )
    write_text(
        DATASET_DIR / "REVIEW_REPORT.md",
        "# Review Report\n\n"
        f"- positive completion status: passed\n"
        f"- final gates: `{preflight['gate_results']}`\n",
    )
    write_text(DATASET_DIR / "LEAKAGE_REPORT.md", "# Leakage Report\n\n- sequence leakage: `0`\n- temporal overlap: `0`\n- duplicate images across splits: `0`\n")
    write_text(DATASET_DIR / "DATASET_QA_REPORT.md", "# Dataset QA Report\n\n" f"- status: `{qa['status']}`\n" f"- errors: `{qa['errors']}`\n")


def compute_dataset_hash(dataset_dir: Path) -> str:
    rows = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name != "dataset_hash.txt":
            rows.append(f"{path.relative_to(dataset_dir)}:{sha256_file(path)}")
    return sha256_text("\n".join(rows))


def write_combined_and_preflight(final_rows: List[Dict[str, Any]], precedence: Dict[str, Any], freeze_integrity: Dict[str, Any], positive_manifest: Dict[str, Any], preflight: Dict[str, Any]) -> None:
    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(COMBINED_DIR / "decision_precedence_report.json", precedence)
    write_jsonl(COMBINED_DIR / "final_decisions.jsonl", final_rows)
    write_json(COMBINED_DIR / "final_review_manifest.json", {"created_at": utc_now(), "positive_completion_freeze": positive_manifest, "final_decision_count": len(final_rows)})
    write_json(COMBINED_DIR / "final_review_audit.json", {"created_at": utc_now(), "precedence": precedence})
    for name in ["review_statistics", "split_statistics", "bbox_validation", "duplicate_report", "temporal_overlap_report", "leakage_report", "dataset_validation"]:
        write_json(PREFLIGHT_DIR / f"{name}.json", preflight[name])
    write_json(PREFLIGHT_DIR / "freeze_integrity_report.json", freeze_integrity)
    write_text(
        PREFLIGHT_DIR / "FINAL_PREFLIGHT_REPORT.md",
        "# PE-0A4 Final Preflight\n\n"
        f"- status: `{preflight['status']}`\n"
        f"- positive_freeze_id: `{positive_manifest['freeze_id']}`\n"
        f"- positive_review_hash: `{positive_manifest['review_hash']}`\n\n"
        "## Gates\n"
        + "\n".join(f"- {name}: `{row['value']}/{row['minimum']}` passed=`{row['passed']}`" for name, row in preflight["gate_results"].items())
        + "\n",
    )


def main() -> None:
    status = positive_completion_status()
    if status["status"] != "passed":
        write_json(POS_RUN / "blocked_positive_review_incomplete.json", status)
        print(json.dumps({"status": "blocked_review_incomplete", "positive_review": status}, indent=2, sort_keys=True))
        return
    positive_manifest = get_or_create_positive_freeze(status)
    freeze_integrity = {
        "original": verify_freeze(ORIGINAL_FREEZE, "review_frozen_manifest.json", ORIGINAL_FREEZE_ID, ORIGINAL_HASH),
        "supplemental": verify_freeze(SUPP_FREEZE, "supplemental_review_frozen_manifest.json", SUPP_FREEZE_ID, SUPP_HASH),
        "uncertain_correction": verify_freeze(CORR_FREEZE, "uncertain_correction_frozen_manifest.json", CORR_FREEZE_ID, CORR_HASH),
        "positive_completion": verify_freeze(POS_REVIEW / "frozen" / positive_manifest["freeze_id"], "positive_completion_frozen_manifest.json", positive_manifest["freeze_id"], positive_manifest["review_hash"]),
    }
    rows = base_source_rows()
    precedence = {
        "order": ["original", "supplemental", "uncertain_correction", "positive_completion"],
        "uncertain_correction": apply_uncertain_correction(rows),
        "positive_completion": add_positive_completion(rows, positive_manifest),
    }
    final_rows = sorted_rows(rows)
    preflight = validate_final(final_rows, freeze_integrity, precedence)
    write_combined_and_preflight(final_rows, precedence, freeze_integrity, positive_manifest, preflight)
    dataset_info: Dict[str, Any] = {"exported": False}
    if preflight["status"] == "passed":
        qa = export_dataset(preflight["dataset_rows"], freeze_integrity, precedence, positive_manifest, preflight)
        dataset_info = {
            "exported": True,
            "path": str(DATASET_DIR),
            "hash": qa["dataset_hash"],
            "qa": qa,
        }
    summary = {
        "phase": "PE-0A4 DATASET FINALIZATION",
        "status": "dataset_ready" if preflight["status"] == "passed" else "blocked_dataset_validation",
        "positive_review": {
            "freeze_id": positive_manifest["freeze_id"],
            "review_hash": positive_manifest["review_hash"],
            "ball": status["reviewed_ball"],
            "no_ball": status["reviewed_no_ball"],
            "uncertain": status["reviewed_uncertain"],
            "pending": status["pending"],
        },
        "splits": preflight["split_statistics"],
        "validation": preflight["dataset_validation"],
        "gates": preflight["gate_results"],
        "freeze_integrity": freeze_integrity,
        "dataset": dataset_info,
        "blocking_errors": preflight["validation_errors"],
    }
    write_json(POS_RUN / "PE0A4_DATASET_FINALIZATION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
