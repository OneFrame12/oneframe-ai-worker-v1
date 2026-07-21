#!/usr/bin/env python3
"""DF-0A Ball Specialist raw inference trace.

This audit is conservative: it uses existing visual-smoke evidence, verifies the
Ball Specialist checkpoint, inspects the inference script for class/threshold
paths, and records whether a real raw trace can run locally. It does not train,
does not modify datasets, and does not touch production.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ai_worker_v1" / "data_factory" / "audit" / "df0a_ball_specialist_trace"
SMOKE_SUMMARY = (
    ROOT
    / "ai_worker_v1"
    / "runs"
    / "pe0_multivideo_visual_smoke_20260717T062015Z"
    / "remote_outputs"
    / "visual_smoke_summary.json"
)
SMOKE_SCRIPT = ROOT / "ai_worker_v1" / "scripts" / "run_multivideo_visual_smoke.py"
PACKAGED_SMOKE_SCRIPT = (
    ROOT
    / "ai_worker_v1"
    / "runs"
    / "pe0_multivideo_visual_smoke_20260717T062015Z"
    / "package_root"
    / "oneframe_visual_smoke"
    / "scripts"
    / "run_multivideo_visual_smoke.py"
)
CHECKPOINT = (
    ROOT
    / "ai_worker_v1"
    / "training"
    / "ball_v0"
    / "rfdetr_s_ball_v0_t3_20260716T221213Z"
    / "remote_outputs"
    / "rfdetr_s_ball_v0_20260716T225601Z"
    / "checkpoints"
    / "best.pth"
)
EXPECTED_SHA = "9dda20e4e7363a284a9775ff3aac4c10280ecd4c86299127be2c5e77a7b64d55"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_probe(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {
            "args": args,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-2000:],
        }
    except Exception as exc:
        return {"args": args, "error": repr(exc)}


def inspect_script(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text()
    constants: list[Any] = []
    try:
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float)):
                constants.append(node.value)
    except Exception:
        pass
    interesting = [
        value
        for value in constants
        if isinstance(value, (int, float))
        or any(token in str(value).lower() for token in ["ball", "threshold", "class", "rfdetr", "yolo"])
    ]
    lines = []
    for idx, line in enumerate(text.splitlines(), start=1):
        low = line.lower()
        if any(token in low for token in ["specialist", "threshold", "class", "rfdetr", "predict", "ball"]):
            lines.append({"line": idx, "text": line[:240]})
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "interesting_constants": interesting[:200],
        "interesting_lines": lines[:200],
    }


def summarize_smoke() -> dict[str, Any]:
    smoke = read_json(SMOKE_SUMMARY, {"videos": []})
    totals = {
        "frames_processed": 0,
        "yolo_ball_frames": 0,
        "rfdetr_ball_frames": 0,
        "specialist_ball_frames": 0,
        "ball_disagreement_frames": 0,
        "yolo_person_detections": 0,
        "rfdetr_person_detections": 0,
    }
    for video in smoke.get("videos", []):
        counters = video.get("counters", {})
        for key in totals:
            totals[key] += int(counters.get(key, 0))
    return {"totals": totals, "videos": smoke.get("videos", [])}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint_sha = sha256_file(CHECKPOINT)
    smoke_summary = summarize_smoke()
    torch_probe = command_probe([sys.executable, "-c", "import torch, json; print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()}))"])
    module_probe = command_probe([sys.executable, "-c", "import rfdetr, ultralytics; print('rfdetr_ok'); print('ultralytics', ultralytics.__version__)"])
    script_reports = [
        inspect_script(SMOKE_SCRIPT),
        inspect_script(PACKAGED_SMOKE_SCRIPT),
    ]
    totals = smoke_summary["totals"]
    if checkpoint_sha != EXPECTED_SHA:
        root_cause = "checkpoint_mismatch"
        classification = "pipeline_regression"
    elif totals["specialist_ball_frames"] == 0 and (
        totals["yolo_ball_frames"] > 0 or totals["rfdetr_ball_frames"] > 0
    ):
        root_cause = "raw_trace_required_threshold_mapping_or_decoding_suspect"
        classification = "threshold_or_mapping_failure_unproven_pending_raw_trace"
    else:
        root_cause = "not_reproduced_from_existing_smoke"
        classification = "functioning_as_expected"

    report = {
        "schema_version": "df0a.ball_specialist_trace.v0",
        "created_at": now_iso(),
        "checkpoint": {
            "path": str(CHECKPOINT.relative_to(ROOT)),
            "expected_sha256": EXPECTED_SHA,
            "actual_sha256": checkpoint_sha,
            "sha256_match": checkpoint_sha == EXPECTED_SHA,
        },
        "environment_probe": {
            "torch": torch_probe,
            "rfdetr_ultralytics": module_probe,
            "local_raw_inference_executed": False,
            "reason": "Local DF-0A did not execute GPU inference; raw trace must run in the same CUDA/model environment or an isolated diagnostic pod.",
        },
        "existing_visual_smoke": smoke_summary["totals"],
        "script_inspection": script_reports,
        "legacy_parity": {
            "status": "pending",
            "reason": "Needs raw frame-level run against known legacy output; aggregate smoke is insufficient.",
        },
        "raw_outputs": {
            "status": "missing",
            "required": True,
            "fields_needed": [
                "raw_logits",
                "raw_boxes",
                "raw_scores",
                "class_ids_before_mapping",
                "class_names_after_mapping",
                "threshold_trace",
                "roi_filter_trace",
                "global_tile_coordinate_trace",
            ],
        },
        "rfdetr_yolo_iou_disagreement": {
            "status": "blocked_raw_boxes_missing",
            "reason": "Visual smoke did not persist per-frame boxes; aggregate disagreement is not enough for IoU.",
        },
        "root_cause": root_cause,
        "final_classification": classification,
    }
    write_json(OUT / "df0a_ball_specialist_trace_report.json", report)
    md = f"""# DF-0A Ball Specialist Raw Inference Trace

- checkpoint SHA match: `{report['checkpoint']['sha256_match']}`
- existing YOLO ball frames: `{totals['yolo_ball_frames']}`
- existing RF-DETR base ball frames: `{totals['rfdetr_ball_frames']}`
- existing Ball Specialist frames: `{totals['specialist_ball_frames']}`
- root cause: `{root_cause}`
- final classification: `{classification}`

Raw GPU inference was not executed locally in this audit. The next diagnostic
must persist raw outputs and per-frame boxes from the CUDA environment before
using Ball Specialist v0 as a preannotator.
"""
    (OUT / "DF0A_BALL_SPECIALIST_TRACE_REPORT.md").write_text(md)
    print(json.dumps(report, indent=2, sort_keys=True)[:5000])


if __name__ == "__main__":
    main()
