#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "ai_worker_v1" / "runs" / "pe0_multivideo_ingestion_20260717T024829Z"
V31_FREEZE_SUMMARY = "calibration/manual_roi_v3_1_final/roi_v3_1_freeze_summary.json"

PERSON_ARTIFACTS = [
    ("persons_rfdetr_base_full.mp4", "full_person_detector", "manual_roi_v3_approved"),
    ("persons_yolo_baseline_full.mp4", "full_person_detector", "manual_roi_v3_approved"),
    ("persons_rfdetr_vs_yolo_full.mp4", "person_detector_comparison", "manual_roi_v3_approved"),
    ("persons_ground_truth.mp4", "person_ground_truth", "person_gold_set_frozen"),
    ("persons_rfdetr_evaluation.mp4", "person_evaluation", "person_gold_set_frozen"),
    ("persons_yolo_evaluation.mp4", "person_evaluation", "person_gold_set_frozen"),
    ("persons_error_review.mp4", "person_error_review", "person_gold_set_frozen"),
]

BALL_ARTIFACTS = [
    ("ball_yolo_baseline_full.mp4", "full_ball_detector", "manual_roi_v3_approved"),
    ("ball_rfdetr_base_full.mp4", "full_ball_detector", "manual_roi_v3_approved"),
    ("ball_specialist_v0_full.mp4", "full_ball_detector", "ball_specialist_v0_available"),
    ("ball_three_way_comparison_full.mp4", "ball_detector_comparison", "ball_specialist_v0_available"),
    ("ball_error_review.mp4", "ball_error_review", "ball_specialist_v0_available"),
]

COMBINED_ARTIFACTS = [
    ("combined_perception_diagnostic.mp4", "combined_visual_diagnostic", "manual_roi_v3_approved"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 4), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_inventory(run_dir: Path) -> Dict[str, Any]:
    inventory_path = run_dir / "inventory" / "video_inventory.json"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Missing inventory: {inventory_path}")
    return json.loads(inventory_path.read_text(encoding="utf-8"))


def source_stem(video: Dict[str, Any]) -> str:
    filename = video.get("filename") or video.get("metadata", {}).get("filename")
    if not filename:
        ffprobe_filename = video.get("ffprobe", {}).get("format", {}).get("filename", "")
        filename = Path(ffprobe_filename).name
    return Path(filename).stem


def expected_assets_for_video(video: Dict[str, Any], evaluation_dir: Path) -> List[Dict[str, Any]]:
    video_id = video["video_id"]
    stem = source_stem(video)
    video_dir = evaluation_dir / video_id
    specs: List[Dict[str, str]] = []
    for filename, category, prerequisite in PERSON_ARTIFACTS:
        specs.append({"filename": filename, "category": category, "prerequisite": prerequisite})
    specs.append({
        "filename": f"{stem}_person_errors.mp4",
        "category": "person_error_reel",
        "prerequisite": "person_gold_set_frozen",
    })
    for filename, category, prerequisite in BALL_ARTIFACTS:
        specs.append({"filename": filename, "category": category, "prerequisite": prerequisite})
    specs.append({
        "filename": f"{stem}_ball_errors.mp4",
        "category": "ball_error_reel",
        "prerequisite": "ball_specialist_v0_available",
    })
    for filename, category, prerequisite in COMBINED_ARTIFACTS:
        specs.append({"filename": filename, "category": category, "prerequisite": prerequisite})

    assets = []
    for spec in specs:
        path = video_dir / spec["filename"]
        exists = path.exists()
        assets.append({
            "filename": spec["filename"],
            "path": str(path),
            "category": spec["category"],
            "prerequisite": spec["prerequisite"],
            "status": "present" if exists else "missing",
            "size_bytes": path.stat().st_size if exists else None,
            "sha256": sha256_file(path) if exists else None,
        })
    return assets


def status_for_assets(assets: Iterable[Dict[str, Any]]) -> str:
    return "visual_evidence_complete" if all(asset["status"] == "present" for asset in assets) else "blocked_missing_visual_evidence"


def build_video_summary(video: Dict[str, Any], assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing = [asset for asset in assets if asset["status"] != "present"]
    present = [asset for asset in assets if asset["status"] == "present"]
    return {
        "video_id": video["video_id"],
        "source_filename": source_stem(video) + ".mp4",
        "status": status_for_assets(assets),
        "required_visual_artifacts": len(assets),
        "present_visual_artifacts": len(present),
        "missing_visual_artifacts": len(missing),
        "assets": assets,
        "blocked_reasons": sorted({asset["prerequisite"] for asset in missing}),
        "completion_rule": "Do not declare detector benchmark completed until every required MP4 is present and hashed.",
    }


def build_report(comparison: Dict[str, Any]) -> str:
    lines = [
        "# Multivideo Visual Evaluation Gate",
        "",
        f"- generated_at: `{comparison['generated_at']}`",
        f"- status: `{comparison['status']}`",
        f"- previous_gate_status: `{comparison['previous_gate_status']}`",
        f"- current_gate: `{comparison['current_gate']}`",
        f"- videos: `{len(comparison['videos'])}`",
        "",
        "This gate records the mandatory visual evidence required before detector evaluation can be called complete.",
        "It does not run inference and it does not create placeholder MP4 evidence.",
        "",
        "## Per Video Status",
        "",
        "| video | present | missing | status | blocked reasons |",
        "|---|---:|---:|---|---|",
    ]
    for video in comparison["videos"]:
        lines.append(
            f"| `{video['video_id']}` | {video['present_visual_artifacts']} | "
            f"{video['missing_visual_artifacts']} | `{video['status']}` | "
            f"{', '.join(video['blocked_reasons']) or 'none'} |"
        )

    lines.extend([
        "",
        "## Required Output Families",
        "",
        "- Full person detector videos per model.",
        "- Side-by-side person model comparison.",
        "- Full ball detector videos for YOLO, RF-DETR base, and Ball Specialist v0.",
        "- Three-way ball model comparison.",
        "- Combined perception diagnostic.",
        "- Ground-truth person evaluation videos after Person Gold Set freeze.",
        "- Person and ball error reels.",
        "",
        "## Current Decision",
        "",
    ])
    if comparison["status"] == "visual_evidence_complete":
        lines.append("Detector visual evidence is complete and can proceed to metric review.")
    elif comparison["status"] == "blocked_pending_v3_1_freeze":
        lines.append("Visual evidence generation is blocked until the three manual ROI V3.1 profiles are frozen and approved for offline visual smoke.")
    else:
        lines.append("Detector visual evidence is blocked. Generate and hash all missing MP4 artifacts before benchmark completion.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate required visual evidence for multivideo detector evaluation.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    inventory = load_inventory(run_dir)
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = run_dir / V31_FREEZE_SUMMARY
    freeze_summary = None
    if freeze_path.exists():
        freeze_summary = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_approved = bool(freeze_summary and freeze_summary.get("status") == "approved_for_offline_visual_smoke")

    summaries = []
    for video in inventory.get("videos", []):
        assets = expected_assets_for_video(video, evaluation_dir)
        summary = build_video_summary(video, assets)
        stem = source_stem(video)
        write_json(evaluation_dir / f"{stem}_detector_summary.json", summary)
        summaries.append(summary)

    if not freeze_approved:
        status = "blocked_pending_v3_1_freeze"
    elif all(s["status"] == "visual_evidence_complete" for s in summaries):
        status = "visual_evidence_complete"
    else:
        status = "blocked_missing_visual_evidence"

    comparison = {
        "phase": "PE-0 MULTIVIDEO VISUAL EVIDENCE GATE",
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "status": status,
        "previous_gate_status": "obsolete_pre_final_roi_export",
        "previous_gate_reason": "The earlier 0/15 result was generated before final manual ROI V3 exports were reconciled.",
        "current_gate": "starts_after_v3_1_freeze" if freeze_approved else "blocked_until_v3_1_freeze",
        "v3_1_freeze_summary": str(freeze_path),
        "v3_1_freeze_status": freeze_summary.get("status") if freeze_summary else "missing",
        "videos": summaries,
        "production": {
            "src_touched": False,
            "runpod_active": False,
            "cost_active": False,
            "inference_executed_by_this_script": False,
        },
    }
    write_json(evaluation_dir / "multivideo_detector_comparison.json", comparison)
    write_text(evaluation_dir / "MULTIVIDEO_VISUAL_EVALUATION_REPORT.md", build_report(comparison))

    print(json.dumps({
        "status": comparison["status"],
        "evaluation_dir": str(evaluation_dir),
        "report": str(evaluation_dir / "MULTIVIDEO_VISUAL_EVALUATION_REPORT.md"),
        "comparison": str(evaluation_dir / "multivideo_detector_comparison.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
