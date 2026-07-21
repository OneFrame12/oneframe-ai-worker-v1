#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
ORIGINAL_RUN = AI_WORKER_ROOT / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z"
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"
ORIGINAL_FREEZE = ORIGINAL_RUN / "review" / "frozen" / ORIGINAL_FREEZE_ID
SUPP_RUN = AI_WORKER_ROOT / "runs" / "pe0a4r_supplemental_ball_20260715T060000Z"
SUPP_REVIEW = SUPP_RUN / "review"
OUT_DIR = SUPP_RUN / "combined_preflight"
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def freeze_supplemental_review() -> Dict[str, Any]:
    freeze_id = f"supplemental_review_freeze_{compact_now()}"
    freeze_dir = SUPP_REVIEW / "frozen" / freeze_id
    files = [
        "supplemental_review_decisions.jsonl",
        "supplemental_review_progress.json",
        "supplemental_review_audit_log.jsonl",
        "supplemental_review_queue.json",
        "supplemental_review_manifest.json",
    ]
    frozen_files = []
    freeze_dir.mkdir(parents=True, exist_ok=True)
    for name in files:
        src = SUPP_REVIEW / name
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
    review_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "created_at": utc_now(),
        "freeze_id": freeze_id,
        "review_hash": review_hash,
        "source_run_dir": str(SUPP_RUN),
        "source_review_dir": str(SUPP_REVIEW),
        "original_freeze_id": ORIGINAL_FREEZE_ID,
        "original_review_hash": ORIGINAL_HASH,
        "frozen_files": frozen_files,
        "decision_count": next((row["line_count"] for row in frozen_files if row["file"] == "supplemental_review_decisions.jsonl"), 0),
    }
    write_json(freeze_dir / "supplemental_review_frozen_manifest.json", manifest)
    write_json(SUPP_REVIEW / "supplemental_review_frozen_manifest.json", manifest)
    return manifest


def verify_original_freeze() -> Dict[str, Any]:
    manifest = read_json(ORIGINAL_FREEZE / "review_frozen_manifest.json")
    errors = []
    if manifest.get("freeze_id") != ORIGINAL_FREEZE_ID:
        errors.append("freeze_id_mismatch")
    if manifest.get("review_hash") != ORIGINAL_HASH:
        errors.append("review_hash_mismatch")
    for item in manifest.get("frozen_files", []):
        path = Path(item["path"])
        if not path.exists():
            path = ORIGINAL_FREEZE / item["file"]
        if not path.exists():
            errors.append(f"missing:{item['file']}")
        elif sha256_file(path) != item["sha256"]:
            errors.append(f"hash_mismatch:{item['file']}")
    return {"status": "passed" if not errors else "failed", "errors": errors, "hash": manifest.get("review_hash")}


def latest_decisions(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row["frame_id"]] = row
    return latest


def original_effective_split(item: Dict[str, Any]) -> str:
    seq = item["sequence_id"]
    if seq == "dense_goal_approach_01":
        return "train"
    if seq == "dense_feet_cluster_01":
        return "excluded_uncertain_pool"
    if item["split"] == "within_video_test_v0":
        return "test"
    return item["split"]


def supplemental_effective_split(item: Dict[str, Any]) -> str:
    return "test" if item["split"] == "within_video_test_v0" else item["split"]


def collect_items() -> Tuple[List[Dict[str, Any]], Dict[str, Counter], Dict[str, Counter]]:
    original_queue = read_json(ORIGINAL_FREEZE / "review_queue.json")["items"]
    original_decisions = latest_decisions(read_jsonl(ORIGINAL_FREEZE / "review_decisions.jsonl"))
    supp_queue = read_json(SUPP_REVIEW / "supplemental_review_queue.json")["items"]
    supp_decisions = latest_decisions(read_jsonl(SUPP_REVIEW / "supplemental_review_decisions.jsonl"))

    items = []
    original_counts = Counter()
    supplemental_counts = Counter()

    for item in original_queue:
        status = item.get("status", "pending")
        original_counts[status] += 1
        items.append(
            {
                "source_review": "original",
                "frame_id": item["frame_id"],
                "sequence_id": item["sequence_id"],
                "source_split": item["split"],
                "effective_split": original_effective_split(item),
                "status": status,
                "decision": original_decisions.get(item["frame_id"]),
                "frame_index_original": item.get("frame_index") or item.get("frame_index_original"),
                "timestamp_sec_original": item.get("timestamp_sec") or item.get("timestamp_sec_original"),
            }
        )

    for item in supp_queue:
        status = item.get("status", "pending")
        supplemental_counts[status] += 1
        items.append(
            {
                "source_review": "supplemental",
                "frame_id": item["frame_id"],
                "sequence_id": item["sequence_id"],
                "source_split": item["split"],
                "effective_split": supplemental_effective_split(item),
                "status": status,
                "decision": supp_decisions.get(item["frame_id"]),
                "frame_index_original": item.get("frame_index_original"),
                "timestamp_sec_original": item.get("timestamp_sec_original") or item.get("timestamp_sec"),
            }
        )
    return items, {"original": original_counts}, {"supplemental": supplemental_counts}


def sequence_time_ranges() -> Dict[str, Dict[str, Any]]:
    ranges = {
        "dense_normal_passes_01": {"start": 94.525, "end": 99.525, "source": "original"},
        "dense_feet_cluster_01": {"start": 108.525, "end": 113.525, "source": "original"},
        "dense_goal_approach_01": {"start": 132.525, "end": 137.525, "source": "original"},
    }
    for row in read_json(SUPP_RUN / "selected_sequences_manifest.json")["sequences"]:
        ranges[row["sequence_id"]] = {"start": row["start_sec"], "end": row["end_sec"], "source": "supplemental"}
    return ranges


def temporal_overlaps() -> List[Dict[str, Any]]:
    ranges = sequence_time_ranges()
    rows = []
    seqs = sorted(ranges)
    for i, left in enumerate(seqs):
        for right in seqs[i + 1 :]:
            a, b = ranges[left], ranges[right]
            overlap = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
            if overlap > 0:
                rows.append({"left": left, "right": right, "overlap_sec": round(overlap, 6)})
    return rows


def preflight(supp_manifest: Dict[str, Any]) -> Dict[str, Any]:
    original_integrity = verify_original_freeze()
    items, original_counts_wrapper, supp_counts_wrapper = collect_items()
    original_counts = original_counts_wrapper["original"]
    supp_counts = supp_counts_wrapper["supplemental"]

    bbox_errors = []
    duplicate_frame_ids = []
    frame_seen = {}
    sequence_splits = defaultdict(set)
    split_counts = defaultdict(Counter)
    uncertain_excluded = 0
    usable_items = []

    for item in items:
        frame_id = item["frame_id"]
        if frame_id in frame_seen:
            duplicate_frame_ids.append(frame_id)
        frame_seen[frame_id] = item["source_review"]
        sequence_splits[item["sequence_id"]].add(item["effective_split"])
        status = item["status"]
        effective_split = item["effective_split"]
        if status == "pending":
            split_counts[effective_split]["pending"] += 1
        if status == "reviewed_uncertain":
            split_counts[effective_split]["uncertain"] += 1
            uncertain_excluded += 1
            continue
        if effective_split == "excluded_uncertain_pool":
            continue
        if status == "reviewed_ball":
            decision = item.get("decision")
            box = decision.get("bbox_xyxy") if decision else None
            errors = validate_bbox(box)
            if errors:
                bbox_errors.append({"frame_id": frame_id, "source_review": item["source_review"], "errors": errors, "bbox_xyxy": box})
            split_counts[effective_split]["ball"] += 1
            usable_items.append(item)
        elif status == "reviewed_no_ball":
            split_counts[effective_split]["no_ball"] += 1
            usable_items.append(item)

    leakage = {seq: sorted(splits) for seq, splits in sequence_splits.items() if len(splits) > 1}
    overlaps = temporal_overlaps()
    validation_errors = []
    pending_total = sum(1 for item in items if item["status"] == "pending")
    if pending_total:
        validation_errors.append("pending_frames_remaining")
    for split in ["train", "valid", "test"]:
        if split_counts[split]["ball"] == 0:
            validation_errors.append(f"{split}_has_zero_ball")
        if split_counts[split]["no_ball"] == 0:
            validation_errors.append(f"{split}_has_zero_no_ball")
    if bbox_errors:
        validation_errors.append("bbox_errors")
    if duplicate_frame_ids:
        validation_errors.append("duplicate_frame_ids")
    if overlaps:
        validation_errors.append("temporal_overlap")
    if leakage:
        validation_errors.append("split_leakage")
    if original_integrity["status"] != "passed":
        validation_errors.append("original_freeze_not_intact")

    result = {
        "phase": "PE-0A4R COMBINED BALL PREFLIGHT",
        "status": "passed" if not validation_errors else "blocked",
        "created_at": utc_now(),
        "original_review": {
            "ball": original_counts.get("reviewed_ball", 0),
            "no_ball": original_counts.get("reviewed_no_ball", 0),
            "uncertain": original_counts.get("reviewed_uncertain", 0),
            "pending": original_counts.get("pending", 0),
            "hash": ORIGINAL_HASH,
        },
        "supplemental_review": {
            "ball": supp_counts.get("reviewed_ball", 0),
            "no_ball": supp_counts.get("reviewed_no_ball", 0),
            "uncertain": supp_counts.get("reviewed_uncertain", 0),
            "pending": supp_counts.get("pending", 0),
            "hash": supp_manifest["review_hash"],
            "freeze_id": supp_manifest["freeze_id"],
        },
        "splits": {
            split: {
                "ball": split_counts[split].get("ball", 0),
                "no_ball": split_counts[split].get("no_ball", 0),
                "uncertain": split_counts[split].get("uncertain", 0),
                "pending": split_counts[split].get("pending", 0),
            }
            for split in ["train", "valid", "test", "excluded_uncertain_pool"]
        },
        "validation": {
            "errors": validation_errors,
            "bbox": {"status": "passed" if not bbox_errors else "failed", "errors": bbox_errors},
            "duplicates": {"status": "passed" if not duplicate_frame_ids else "failed", "duplicate_frame_ids": duplicate_frame_ids},
            "temporal_overlap": {"status": "passed" if not overlaps else "failed", "overlaps": overlaps},
            "leakage": {"status": "passed" if not leakage else "failed", "sequence_split_leakage": leakage},
            "uncertain_excluded": {"status": "passed", "count": uncertain_excluded},
            "original_freeze_intact": original_integrity,
        },
        "decision": "exportar OneFrame_Ball_v0" if not validation_errors else "agregar únicamente una secuencia al split deficitario",
        "deficits": [
            err
            for err in validation_errors
            if err.endswith("_has_zero_ball") or err.endswith("_has_zero_no_ball")
        ],
        "usable_items": len(usable_items),
    }
    write_json(OUT_DIR / "combined_preflight_result.json", result)
    lines = [
        "# PE-0A4R Combined Ball Preflight",
        "",
        f"- status: `{result['status']}`",
        f"- original_hash: `{ORIGINAL_HASH}`",
        f"- supplemental_hash: `{supp_manifest['review_hash']}`",
        f"- validation_errors: `{validation_errors}`",
        "",
        "## Splits",
        "",
        "| split | ball | no_ball | uncertain | pending |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, counts in result["splits"].items():
        lines.append(f"| {split} | {counts['ball']} | {counts['no_ball']} | {counts['uncertain']} | {counts['pending']} |")
    lines += [
        "",
        f"Decision: `{result['decision']}`",
        "",
        "No dataset export or training was run by this preflight.",
    ]
    write_text(OUT_DIR / "COMBINED_PREFLIGHT_REPORT.md", "\n".join(lines) + "\n")
    return result


def main() -> None:
    supp_manifest = freeze_supplemental_review()
    result = preflight(supp_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

