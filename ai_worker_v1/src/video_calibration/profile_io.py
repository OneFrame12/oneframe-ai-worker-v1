from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

from .legacy_roi_adapter import build_legacy_import_report
from .qa import update_profile_qa
from .renderer import render_calibration_overlay
from .schema import VideoCalibrationProfile


def canonical_profile_json(profile: VideoCalibrationProfile) -> str:
    return json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def profile_hash(profile: VideoCalibrationProfile) -> str:
    return sha256(canonical_profile_json(profile).encode("utf-8")).hexdigest()


def save_profile(path: str | Path, profile: VideoCalibrationProfile) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_profile(path: str | Path) -> VideoCalibrationProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return VideoCalibrationProfile.from_dict(data)


def save_profile_run_artifacts(
    run_id: str,
    profile: VideoCalibrationProfile,
    runs_root: str | Path = "ai_worker_v1/runs",
) -> Dict[str, str]:
    update_profile_qa(profile)
    run_dir = Path(runs_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    profile_path = save_profile(run_dir / "video_calibration.json", profile)
    qa_path = run_dir / "video_calibration_qa.json"
    legacy_path = run_dir / "legacy_import_report.json"
    overlay_path = run_dir / "video_calibration_overlay.png"

    qa_payload = {
        "calibration_id": profile.calibration_id,
        "profile_hash": profile_hash(profile),
        "qa": profile.qa.__dict__,
        "roi_validation": profile.detection_roi.validation,
        "homography": profile.homography.__dict__,
    }
    qa_path.write_text(json.dumps(qa_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    legacy_path.write_text(
        json.dumps(build_legacy_import_report(profile), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_calibration_overlay(profile, overlay_path)

    return {
        "profile": str(profile_path),
        "qa": str(qa_path),
        "legacy_import_report": str(legacy_path),
        "overlay": str(overlay_path),
    }
