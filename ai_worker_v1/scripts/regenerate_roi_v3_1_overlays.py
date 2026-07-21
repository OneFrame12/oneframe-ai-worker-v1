#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from reconcile_manual_roi_v3_1 import (
    AI_ROOT,
    OUTPUT_DIR,
    REPO_ROOT,
    FRAME_SIZE,
    draw_polygon,
    sha256_file,
    write_json,
)


MANIFEST_PATH = AI_ROOT / "tools" / "manual_roi_calibration" / "assets" / "videos_manifest.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_visual_asset_path(path_value: str) -> Path | None:
    raw = Path(path_value)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend([
        REPO_ROOT / raw,
        AI_ROOT / "tools" / "manual_roi_calibration" / raw,
        AI_ROOT / "runs" / "pe0_multivideo_ingestion_20260717T024829Z" / raw,
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def overlay_one(image: np.ndarray, geometry: Dict[str, Any], label: str) -> np.ndarray:
    out = image.copy()
    draw_polygon(out, geometry["perception_roi"], (255, 0, 0), "perception_roi")
    draw_polygon(out, geometry["detection_field_roi"], (0, 230, 0), "detection_field_roi")
    draw_polygon(out, geometry["goal_zones"]["near_goal"], (0, 255, 255), "near_goal_zone")
    draw_polygon(out, geometry["goal_zones"]["far_goal"], (0, 255, 255), "far_goal_zone")
    cv2.rectangle(out, (0, 0), (1080, 54), (0, 0, 0), -1)
    cv2.putText(out, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def make_overlay(video_manifest: Dict[str, Any], geometry: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    tiles: List[np.ndarray] = []
    frame_checks: List[Dict[str, Any]] = []
    for frame in video_manifest["frames"]:
        frame_path = resolve_visual_asset_path(frame["path"])
        image = cv2.imread(str(frame_path)) if frame_path else None
        if image is None:
            image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
            status = "read_failed"
        else:
            status = "ok"
        label = f"{video_manifest['video_id']} t={frame['timestamp_sec']}s ROI V3.1"
        overlaid = overlay_one(image, geometry, label)
        tiles.append(cv2.resize(overlaid, (640, 360), interpolation=cv2.INTER_AREA))
        frame_checks.append({
            "path": str(frame_path) if frame_path else None,
            "status": status,
            "mean_pixel": round(float(image.mean()), 3),
            "max_pixel": int(image.max()),
            "timestamp_sec": frame["timestamp_sec"],
        })
    sheet = np.zeros((720, 1920, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles[:6]):
        row, col = divmod(idx, 3)
        sheet[row * 360 : row * 360 + 360, col * 640 : col * 640 + 640] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    saved = cv2.imread(str(out_path))
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "mean_pixel": round(float(saved.mean()), 3) if saved is not None else None,
        "max_pixel": int(saved.max()) if saved is not None else None,
        "frame_checks": frame_checks,
        "nonblank": bool(saved is not None and saved.mean() > 20.0),
    }


def main() -> int:
    manifest = load_json(MANIFEST_PATH)
    report = {
        "status": "completed",
        "geometry_changed": False,
        "profiles_modified": False,
        "masks_modified": False,
        "videos": [],
    }
    for video in manifest["videos"]:
        video_id = video["video_id"]
        profile_path = OUTPUT_DIR / video_id / "roi_manual_v3_1_profile.json"
        profile = load_json(profile_path)
        overlay_path = OUTPUT_DIR / video_id / "roi_overlay_multi_timestamp_v3_1.jpg"
        overlay = make_overlay(video, profile["geometry"], overlay_path)
        expected_ok = overlay["nonblank"] and all(item["status"] == "ok" for item in overlay["frame_checks"])
        report["videos"].append({
            "video_id": video_id,
            "overlay": overlay,
            "visual_asset_path_corrected": True,
            "background_real": expected_ok,
            "colors": {
                "perception_roi": "blue",
                "detection_field_roi": "green",
                "near_goal_zone": "yellow",
                "far_goal_zone": "yellow",
            },
        })
        if not expected_ok:
            report["status"] = "blocked_overlay_validation"
    write_json(OUTPUT_DIR / "roi_v3_1_overlay_regeneration_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
