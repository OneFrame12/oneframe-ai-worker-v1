#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import cv2


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
RUN_DIR = AI_WORKER_ROOT / "runs" / "pe0a4r_positive_completion_20260715T174023Z"
VIDEO_PATH = AI_WORKER_ROOT / "runs" / "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps" / "source_video.mp4"

ORIGINAL_FREEZE = AI_WORKER_ROOT / "runs" / "pe0a3r_full_field_ball_data_20260715T0345Z" / "review" / "frozen" / "review_freeze_20260715T052342Z"
ORIGINAL_FREEZE_ID = "review_freeze_20260715T052342Z"
ORIGINAL_HASH = "6701e364750fa69824a9114070949eeea4d519a6f6d18acf05116b2a88259703"

SUPP_FREEZE = AI_WORKER_ROOT / "runs" / "pe0a4r_supplemental_ball_20260715T060000Z" / "review" / "frozen" / "supplemental_review_freeze_20260715T163151Z"
SUPP_FREEZE_ID = "supplemental_review_freeze_20260715T163151Z"
SUPP_HASH = "1ac2d800a46290c9142836ad4a3295425efe4d03445e11524f820081bccf7912"

CORR_FREEZE = AI_WORKER_ROOT / "runs" / "pe0a4r_uncertain_correction_20260715T165053Z" / "review" / "frozen" / "uncertain_correction_freeze_20260715T172940Z"
CORR_FREEZE_ID = "uncertain_correction_freeze_20260715T172940Z"
CORR_HASH = "ea179083ed2a411305b572e446ec98d27470275c8ec9164ea42aa4bb2351abaa"

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
SAMPLE_FPS = 15
SELECTED = [
    {
        "sequence_id": "valid_positive_completion_01",
        "source_window_id": "pcw_09",
        "split": "valid",
        "start_sec": 190.0,
        "end_sec": 192.5,
        "visual_classification": "positive_clear",
        "reason": "ball visible in most sampled frames near goal area; validation split needs positives",
        "review_order": 1,
    },
    {
        "sequence_id": "train_positive_completion_01",
        "source_window_id": "pcw_01",
        "split": "train",
        "start_sec": 8.0,
        "end_sec": 10.5,
        "visual_classification": "positive_clear",
        "reason": "early attacking play; new train positive sequence distinct from existing train positives",
        "review_order": 2,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def verify_freeze(freeze_dir: Path, manifest_name: str, freeze_id: str, review_hash: str) -> Dict[str, Any]:
    manifest = read_json(freeze_dir / manifest_name)
    errors = []
    if manifest.get("freeze_id") != freeze_id:
        errors.append("freeze_id_mismatch")
    if manifest.get("review_hash") != review_hash:
        errors.append("review_hash_mismatch")
    for row in manifest.get("frozen_files", []):
        path = Path(row["path"])
        if not path.exists():
            path = freeze_dir / row["file"]
        if not path.exists():
            errors.append(f"missing:{row['file']}")
        elif sha256_file(path) != row["sha256"]:
            errors.append(f"hash_mismatch:{row['file']}")
    return {"status": "passed" if not errors else "failed", "errors": errors, "freeze_id": freeze_id, "review_hash": review_hash}


def overlap(a: float, b: float, c: float, d: float) -> float:
    return max(0.0, min(b, d) - max(a, c))


def load_used_ranges() -> List[Dict[str, Any]]:
    ranges = read_json(RUN_DIR / "used_time_ranges.json")["used_ranges"]
    for selected in SELECTED:
        selected["overlap_status"] = "clear"
        selected["overlaps"] = []
        for used in ranges:
            ov = overlap(selected["start_sec"], selected["end_sec"], used["start_sec"], used["end_sec"])
            if ov:
                selected["overlap_status"] = "overlap_rejected"
                selected["overlaps"].append({"sequence_id": used["sequence_id"], "overlap_sec": ov})
    return ranges


def update_candidate_windows() -> Dict[str, Any]:
    path = RUN_DIR / "candidate_windows.json"
    payload = read_json(path)
    selected_by_id = {row["source_window_id"]: row for row in SELECTED}
    classifications = {
        "pcw_01": "positive_clear",
        "pcw_02": "ambiguous",
        "pcw_03": "rejected",
        "pcw_04": "ambiguous",
        "pcw_05": "ambiguous",
        "pcw_06": "rejected",
        "pcw_07": "ambiguous",
        "pcw_08": "ambiguous",
        "pcw_09": "positive_clear",
        "pcw_10": "rejected",
    }
    for window in payload["windows"]:
        window["visual_classification"] = classifications.get(window["window_id"], "pending")
        window["selected_for_review"] = window["window_id"] in selected_by_id
        if window["window_id"] in selected_by_id:
            window["selected_sequence_id"] = selected_by_id[window["window_id"]]["sequence_id"]
            window["selected_split"] = selected_by_id[window["window_id"]]["split"]
    write_json(path, payload)
    return payload


def extract_sequence_frames() -> List[Dict[str, Any]]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"cannot_open_video:{VIDEO_PATH}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
        raise RuntimeError(f"unexpected_resolution:{width}x{height}")
    queue_items = []
    for seq in SELECTED:
        seq_dir = RUN_DIR / "review" / "frames" / seq["sequence_id"]
        seq_dir.mkdir(parents=True, exist_ok=True)
        start = float(seq["start_sec"])
        end = float(seq["end_sec"])
        frame_count = int(round((end - start) * SAMPLE_FPS))
        frames = []
        for local_idx in range(frame_count):
            timestamp = start + local_idx / SAMPLE_FPS
            frame_index = int(round(timestamp * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            frame_id = f"{seq['sequence_id']}_f{frame_index:08d}"
            current_rel = Path("review") / "frames" / seq["sequence_id"] / f"{frame_id}_current.jpg"
            cv2.imwrite(str(RUN_DIR / current_rel), frame)
            frames.append(
                {
                    "frame_id": frame_id,
                    "timestamp_sec": round(timestamp, 6),
                    "frame_index_original": frame_index,
                    "current_image": str(current_rel),
                }
            )
        for index, frame in enumerate(frames):
            prev_rel = frames[index - 1]["current_image"] if index > 0 else None
            next_rel = frames[index + 1]["current_image"] if index + 1 < len(frames) else None
            queue_items.append(
                {
                    "frame_id": frame["frame_id"],
                    "sequence_id": seq["sequence_id"],
                    "split": seq["split"],
                    "source_window_id": seq["source_window_id"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "frame_index_original": frame["frame_index_original"],
                    "review_status": "pending",
                    "new_status": "pending",
                    "ground_truth": False,
                    "pseudo_label": True,
                    "final_bbox_xyxy": None,
                    "candidate_count": 0,
                    "queue_status": "pending",
                    "reviewed_by": None,
                    "context": {
                        "prev_image": prev_rel,
                        "current_image": frame["current_image"],
                        "next_image": next_rel,
                    },
                }
            )
        # Preview MP4 for selected windows only.
        preview_path = RUN_DIR / "candidate_previews" / f"{seq['source_window_id']}_{start:.1f}_{end:.1f}.mp4"
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*"mp4v"), SAMPLE_FPS, (width, height))
        for item in [row for row in queue_items if row["sequence_id"] == seq["sequence_id"]]:
            img = cv2.imread(str(RUN_DIR / item["context"]["current_image"]))
            if img is not None:
                writer.write(img)
        writer.release()
        seq["frames"] = len([row for row in queue_items if row["sequence_id"] == seq["sequence_id"]])
        seq["preview_mp4"] = str(preview_path.relative_to(RUN_DIR))
    cap.release()
    return queue_items


def base_counts() -> Dict[str, Any]:
    summary = read_json(AI_WORKER_ROOT / "runs" / "pe0a4r_uncertain_correction_20260715T165053Z" / "PE0A4_FINAL_SUMMARY.json")
    return {
        "splits": {
            split: {
                "ball": values["ball"],
                "no_ball": values["no_ball"],
                "uncertain": values["uncertain"],
                "annotations": values["annotations"],
            }
            for split, values in summary["splits"].items()
        },
        "positive_sequences": {
            split: values.get("positive_sequences", [])
            for split, values in summary["splits"].items()
        },
        "gates": summary["gates"],
    }


def write_review_files(queue_items: List[Dict[str, Any]], freezes: Dict[str, Any], candidate_payload: Dict[str, Any], used_ranges: List[Dict[str, Any]]) -> None:
    review = RUN_DIR / "review"
    counts = base_counts()
    write_json(review / "positive_completion_queue.json", {"items": queue_items})
    write_jsonl(review / "positive_completion_decisions.jsonl", [])
    write_jsonl(review / "positive_completion_audit_log.jsonl", [{"event": "SESSION_CREATED", "created_at": utc_now(), "queue_items": len(queue_items), "synthetic": False}])
    write_json(
        review / "positive_completion_progress.json",
        {
            "total": len(queue_items),
            "pending": len(queue_items),
            "reviewed_ball": 0,
            "reviewed_no_ball": 0,
            "reviewed_uncertain": 0,
            "base_counts": counts,
            "gate_status": "pending_positive_review",
        },
    )
    write_json(
        review / "positive_completion_manifest.json",
        {
            "created_at": utc_now(),
            "phase": "PE-0A4R-P FINAL POSITIVE SPLIT COMPLETION",
            "status": "ready_for_positive_review",
            "source_video": str(VIDEO_PATH.resolve()),
            "source_video_sha256": sha256_file(VIDEO_PATH),
            "sample_fps": SAMPLE_FPS,
            "selected_sequences": SELECTED,
            "queue_items": len(queue_items),
            "freezes": freezes,
            "base_counts": counts,
            "additive_positive_completion": True,
            "training_allowed": False,
        },
    )
    write_json(
        RUN_DIR / "selected_sequences_manifest.json",
        {
            "created_at": utc_now(),
            "selection_checks": {
                "status": "passed" if all(row["overlap_status"] == "clear" for row in SELECTED) else "failed",
                "overlaps": [row for row in SELECTED if row["overlap_status"] != "clear"],
            },
            "sequences": SELECTED,
        },
    )
    selected_ids = {row["source_window_id"] for row in SELECTED}
    write_text(
        RUN_DIR / "selection_report.md",
        "# PE-0A4R-P Positive Completion Selection\n\n"
        f"- candidate_windows: `{len(candidate_payload['windows'])}`\n"
        f"- selected_windows: `{', '.join(sorted(selected_ids))}`\n"
        "- selection_basis: visual review of contact sheets, not detector confidence\n"
        "- overlap_status: `clear`\n\n"
        "## Selected\n"
        + "\n".join(f"- `{row['sequence_id']}` split=`{row['split']}` window=`{row['source_window_id']}` start=`{row['start_sec']}` end=`{row['end_sec']}` frames=`{row.get('frames')}`" for row in SELECTED)
        + "\n",
    )
    write_json(RUN_DIR / "used_time_ranges.json", {"created_at": utc_now(), "used_ranges": used_ranges, "selected_new_ranges": SELECTED})


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    freezes = {
        "original": verify_freeze(ORIGINAL_FREEZE, "review_frozen_manifest.json", ORIGINAL_FREEZE_ID, ORIGINAL_HASH),
        "supplemental": verify_freeze(SUPP_FREEZE, "supplemental_review_frozen_manifest.json", SUPP_FREEZE_ID, SUPP_HASH),
        "uncertain_correction": verify_freeze(CORR_FREEZE, "uncertain_correction_frozen_manifest.json", CORR_FREEZE_ID, CORR_HASH),
    }
    if any(row["status"] != "passed" for row in freezes.values()):
        raise RuntimeError(freezes)
    used_ranges = load_used_ranges()
    if any(row["overlap_status"] != "clear" for row in SELECTED):
        raise RuntimeError({"selected_overlap": SELECTED})
    candidate_payload = update_candidate_windows()
    queue_items = extract_sequence_frames()
    write_review_files(queue_items, freezes, candidate_payload, used_ranges)
    summary = {
        "phase": "PE-0A4R-P FINAL POSITIVE SPLIT COMPLETION",
        "status": "ready_for_positive_review",
        "run_dir": str(RUN_DIR),
        "freezes": freezes,
        "selected_sequences": SELECTED,
        "queue": {
            "pending": len(queue_items),
            "total": len(queue_items),
            "valid_frames": sum(1 for item in queue_items if item["split"] == "valid"),
            "train_frames": sum(1 for item in queue_items if item["split"] == "train"),
        },
        "deficits": {
            "train_ball": 9,
            "train_positive_sequences": 1,
            "valid_ball": 14,
        },
    }
    write_json(RUN_DIR / "POSITIVE_COMPLETION_SETUP_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
