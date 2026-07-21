#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def verify_freeze(freeze_dir: Path, freeze_id: str, expected_hash: str, manifest_name: str) -> Dict[str, Any]:
    manifest = read_json(freeze_dir / manifest_name)
    errors = []
    if manifest.get("freeze_id") != freeze_id:
        errors.append("freeze_id_mismatch")
    if manifest.get("review_hash") != expected_hash:
        errors.append("review_hash_mismatch")
    for row in manifest.get("frozen_files", []):
        path = Path(row["path"])
        if not path.exists():
            path = freeze_dir / row["file"]
        if not path.exists():
            errors.append(f"missing:{row['file']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"hash_mismatch:{row['file']}")
    return {"status": "passed" if not errors else "failed", "errors": errors, "freeze_id": freeze_id, "review_hash": expected_hash}


def latest_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest = {}
    for row in rows:
        latest[row["frame_id"]] = row
    return latest


def original_effective_split(item: Dict[str, Any]) -> str:
    if item["sequence_id"] == "dense_goal_approach_01":
        return "train"
    if item["sequence_id"] == "dense_feet_cluster_01":
        return "excluded_uncertain_pool"
    return "test" if item["split"] == "within_video_test_v0" else item["split"]


def load_frozen_sources() -> List[Dict[str, Any]]:
    original_queue = read_json(ORIGINAL_FREEZE / "review_queue.json")["items"]
    original_decisions = latest_decisions(read_jsonl(ORIGINAL_FREEZE / "review_decisions.jsonl"))
    supp_queue = read_json(SUPP_FREEZE / "supplemental_review_queue.json")["items"]
    supp_decisions = latest_decisions(read_jsonl(SUPP_FREEZE / "supplemental_review_decisions.jsonl"))
    rows = []
    for item in original_queue:
        rows.append(
            {
                "source": "original",
                "source_run": ORIGINAL_RUN,
                "source_freeze_id": ORIGINAL_FREEZE_ID,
                "source_review_hash": ORIGINAL_HASH,
                "effective_split": original_effective_split(item),
                "decision": original_decisions.get(item["frame_id"]),
                "item": item,
            }
        )
    for item in supp_queue:
        rows.append(
            {
                "source": "supplemental",
                "source_run": SUPP_RUN,
                "source_freeze_id": SUPP_FREEZE_ID,
                "source_review_hash": SUPP_HASH,
                "effective_split": "test" if item["split"] == "within_video_test_v0" else item["split"],
                "decision": supp_decisions.get(item["frame_id"]),
                "item": item,
            }
        )
    return rows


def base_counts(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    split_counts = defaultdict(Counter)
    positive_sequences = defaultdict(set)
    for row in rows:
        item = row["item"]
        status = item.get("status", "pending")
        split = row["effective_split"]
        if status == "reviewed_uncertain":
            split_counts[split]["uncertain"] += 1
            continue
        if split == "excluded_uncertain_pool":
            continue
        if status == "reviewed_ball":
            split_counts[split]["ball"] += 1
            positive_sequences[split].add(item["sequence_id"])
        elif status == "reviewed_no_ball":
            split_counts[split]["no_ball"] += 1
        elif status == "pending":
            split_counts[split]["pending"] += 1
    return {
        "splits": {split: dict(counts) for split, counts in split_counts.items()},
        "positive_sequences": {split: sorted(values) for split, values in positive_sequences.items()},
    }


def deficits(counts: Dict[str, Any]) -> Dict[str, Any]:
    splits = counts["splits"]
    positives = counts["positive_sequences"]
    return {
        "train_ball": max(0, GATES["train"]["ball"] - splits.get("train", {}).get("ball", 0)),
        "valid_ball": max(0, GATES["valid"]["ball"] - splits.get("valid", {}).get("ball", 0)),
        "valid_no_ball": max(0, GATES["valid"]["no_ball"] - splits.get("valid", {}).get("no_ball", 0)),
        "test_no_ball": max(0, GATES["test"]["no_ball"] - splits.get("test", {}).get("no_ball", 0)),
        "train_positive_sequences": {
            "current": len(positives.get("train", [])),
            "minimum": GATES["train"]["positive_sequences"],
            "deficit": max(0, GATES["train"]["positive_sequences"] - len(positives.get("train", []))),
            "sequences": positives.get("train", []),
        },
    }


def group_by_sequence(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        item = row["item"]
        grouped[item["sequence_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: float(row["item"].get("timestamp_sec_original") or row["item"].get("timestamp_sec") or 0.0))
    return grouped


def pick_spread(rows: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if len(rows) <= limit:
        return list(rows)
    if limit <= 1:
        return [rows[0]]
    selected = []
    used = set()
    for idx in [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)]:
        while idx in used and idx + 1 < len(rows):
            idx += 1
        used.add(idx)
        selected.append(rows[idx])
    return selected


def selected_uncertain_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    uncertain = [row for row in rows if row["item"].get("status") == "reviewed_uncertain"]
    test_rows = [row for row in uncertain if row["effective_split"] == "test"]
    valid_rows = [row for row in uncertain if row["effective_split"] == "valid"]
    train_rows = [row for row in uncertain if row["effective_split"] == "train"]
    grouped_train = group_by_sequence(train_rows)

    # Initial train block: emphasize the sequence with no positives yet, then spread the rest.
    train_selected = []
    allocation = {
        "train_ball_hard_03": 20,
        "train_ball_context_02": 10,
        "dense_goal_approach_01": 8,
        "dense_normal_passes_01": 7,
    }
    for seq_id, limit in allocation.items():
        train_selected.extend(pick_spread(grouped_train.get(seq_id, []), limit))
    # Deduplicate while preserving order.
    seen = set()
    train_unique = []
    for row in train_selected:
        frame_id = row["item"]["frame_id"]
        if frame_id not in seen:
            seen.add(frame_id)
            train_unique.append(row)
    return test_rows + valid_rows + train_unique[:45]


def source_image_path(row: Dict[str, Any]) -> Path:
    item = row["item"]
    rel = item.get("frame_image") or item.get("image_path")
    return row["source_run"] / rel


def copy_context_assets(run_dir: Path, selected: List[Dict[str, Any]], all_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = group_by_sequence(all_rows)
    selected_ids = {row["item"]["frame_id"] for row in selected}
    queue = []
    for row in selected:
        item = row["item"]
        seq_rows = grouped[item["sequence_id"]]
        idx = next(i for i, candidate in enumerate(seq_rows) if candidate["item"]["frame_id"] == item["frame_id"])
        contexts = {
            "prev": seq_rows[idx - 1] if idx > 0 else None,
            "current": row,
            "next": seq_rows[idx + 1] if idx + 1 < len(seq_rows) else None,
        }
        context_rel = {}
        for label, ctx in contexts.items():
            if ctx is None:
                context_rel[f"{label}_image"] = None
                continue
            src = source_image_path(ctx)
            dst_rel = Path("frames") / item["sequence_id"] / f"{item['frame_id']}_{label}.jpg"
            dst = run_dir / "review" / dst_rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            context_rel[f"{label}_image"] = str(Path("review") / dst_rel)
        decision = row.get("decision") or {}
        old_bbox = decision.get("bbox_xyxy")
        if old_bbox == [0, 0, 0, 0]:
            old_bbox = None
        queue.append(
            {
                "frame_id": item["frame_id"],
                "sequence_id": item["sequence_id"],
                "split": row["effective_split"],
                "source": row["source"],
                "source_freeze_id": row["source_freeze_id"],
                "source_review_hash": row["source_review_hash"],
                "old_status": "reviewed_uncertain",
                "new_status": "pending",
                "final_bbox_xyxy": old_bbox,
                "candidate_count": item.get("candidate_count", 0),
                "timestamp_sec": item.get("timestamp_sec_original") or item.get("timestamp_sec"),
                "frame_index_original": item.get("frame_index_original") or item.get("frame_index"),
                "queue_status": "pending",
                "reviewed_by": None,
                "context": context_rel,
                "selection_reason": selection_reason(row, selected_ids),
            }
        )
    return queue


def selection_reason(row: Dict[str, Any], selected_ids: set) -> str:
    split = row["effective_split"]
    seq = row["item"]["sequence_id"]
    if split == "test":
        return "test_uncertain_target_no_ball_deficit"
    if split == "valid":
        return "valid_uncertain_target_ball_and_no_ball_deficit"
    if seq == "train_ball_hard_03":
        return "train_uncertain_priority_new_positive_sequence"
    return "train_uncertain_spread_temporal_context"


def write_review_files(run_dir: Path, queue: List[Dict[str, Any]], counts: Dict[str, Any], deficit_rows: Dict[str, Any]) -> None:
    review = run_dir / "review"
    write_json(review / "uncertain_correction_queue.json", {"items": queue})
    write_jsonl(review / "uncertain_correction_decisions.jsonl", [])
    write_jsonl(
        review / "uncertain_correction_audit_log.jsonl",
        [
            {
                "created_at": utc_now(),
                "event": "SESSION_CREATED",
                "queue_items": len(queue),
                "synthetic": False,
            }
        ],
    )
    progress = {
        "total": len(queue),
        "pending": len(queue),
        "reviewed_ball": 0,
        "reviewed_no_ball": 0,
        "reviewed_uncertain": 0,
        "base_counts": counts,
        "deficits": deficit_rows,
        "gates": GATES,
        "gate_status": "pending_corrections",
    }
    write_json(review / "uncertain_correction_progress.json", progress)
    write_json(
        review / "uncertain_correction_manifest.json",
        {
            "created_at": utc_now(),
            "phase": "PE-0A4R-C TARGETED UNCERTAIN CORRECTION",
            "status": "ready_for_uncertain_correction",
            "original_freeze_id": ORIGINAL_FREEZE_ID,
            "original_review_hash": ORIGINAL_HASH,
            "supplemental_freeze_id": SUPP_FREEZE_ID,
            "supplemental_review_hash": SUPP_HASH,
            "queue_items": len(queue),
            "queue_policy": "test uncertain, valid uncertain, then selected train uncertain block",
            "additive_corrections": True,
            "training_allowed": False,
        },
    )
    write_text(
        review / "uncertain_correction_instructions.md",
        "# PE-0A4R-C Targeted Uncertain Correction\n\n"
        "Correct only the loaded uncertain frames. Decisions are additive and never edit frozen reviews.\n\n"
        "Rules:\n"
        "- A: reviewed_ball, requires an existing or drawn bbox.\n"
        "- N: reviewed_no_ball when no ball is visually identifiable.\n"
        "- U: keep reviewed_uncertain only when a concrete object is visible but ambiguous.\n"
        "- D: clear bbox.\n"
        "- S: save current status.\n"
        "- Arrow keys: navigate.\n\n"
        "The previous and next frames are context only. The decision applies only to the center frame.\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or f"pe0a4r_uncertain_correction_{compact_now()}"
    run_dir = AI_WORKER_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    original_integrity = verify_freeze(ORIGINAL_FREEZE, ORIGINAL_FREEZE_ID, ORIGINAL_HASH, "review_frozen_manifest.json")
    supp_integrity = verify_freeze(SUPP_FREEZE, SUPP_FREEZE_ID, SUPP_HASH, "supplemental_review_frozen_manifest.json")
    if original_integrity["status"] != "passed" or supp_integrity["status"] != "passed":
        raise RuntimeError({"original": original_integrity, "supplemental": supp_integrity})

    rows = load_frozen_sources()
    counts = base_counts(rows)
    deficit_rows = deficits(counts)
    selected = selected_uncertain_rows(rows)
    queue = copy_context_assets(run_dir, selected, rows)
    write_review_files(run_dir, queue, counts, deficit_rows)

    summary = {
        "phase": "PE-0A4R-C TARGETED UNCERTAIN CORRECTION",
        "status": "ready_for_uncertain_correction",
        "run_dir": str(run_dir),
        "created_at": utc_now(),
        "freezes": {"original": original_integrity, "supplemental": supp_integrity},
        "deficits": deficit_rows,
        "queue": {
            "test_frames": sum(1 for item in queue if item["split"] == "test"),
            "valid_frames": sum(1 for item in queue if item["split"] == "valid"),
            "train_frames_initial": sum(1 for item in queue if item["split"] == "train"),
            "total": len(queue),
            "pending": len(queue),
        },
    }
    write_json(run_dir / "UNCERTAIN_CORRECTION_SETUP_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

