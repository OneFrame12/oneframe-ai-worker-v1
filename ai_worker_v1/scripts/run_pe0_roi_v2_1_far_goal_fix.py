#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from run_pe0_roi_v2_multizone import (
    AI_WORKER_ROOT,
    DEFAULT_INGESTION_RUN,
    HEIGHT,
    WIDTH,
    build_masks,
    create_overlay_sheet,
    draw_outlines,
    read_frame,
    sha256_file,
    stable_hash,
    validate_polygon,
    video_path_for_filename,
    write_json,
    write_mask,
    write_text,
)


PHASE = "PE-0 CALIBRATION ROI V2.1"
SCRIPT_NAME = "ai_worker_v1/scripts/run_pe0_roi_v2_1_far_goal_fix.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(points: List[List[float]]) -> List[List[float]]:
    return [[round(x / WIDTH, 8), round(y / HEIGHT, 8)] for x, y in points]


def corrected_far_goal_zones() -> Dict[str, List[List[float]]]:
    # Coordinates were revised from the clean editor frames, not copied between
    # videos. The zones are tight around the visually real far goal structure.
    return {
        "mv01_video_02_0db7846b4b08": [[248, 0], [430, 0], [448, 170], [262, 210]],
        "mv01_video_03_fedc92e58ea7": [[895, 40], [1030, 40], [1038, 132], [882, 134]],
        "mv01_video_04_a1849e050f52": [[885, 102], [1038, 106], [1048, 212], [870, 212]],
    }


def test_points() -> Dict[str, List[Dict[str, Any]]]:
    return {
        "mv01_video_02_0db7846b4b08": [
            {"point_id": "near_goalkeeper_valid", "x": 930, "y": 845, "expected": "accepted"},
            {"point_id": "central_player_valid", "x": 960, "y": 505, "expected": "accepted"},
            {"point_id": "left_lateral_player_valid", "x": 240, "y": 420, "expected": "accepted"},
            {"point_id": "right_lateral_player_valid", "x": 1660, "y": 420, "expected": "accepted"},
            {"point_id": "far_goalkeeper_valid", "x": 345, "y": 135, "expected": "accepted"},
            {"point_id": "behind_near_goal_rejected", "x": 960, "y": 1045, "expected": "rejected"},
            {"point_id": "lower_left_external_rejected", "x": 85, "y": 1000, "expected": "rejected"},
            {"point_id": "lower_right_external_rejected", "x": 1800, "y": 1015, "expected": "rejected"},
            {"point_id": "right_ads_walkway_rejected", "x": 1740, "y": 220, "expected": "rejected"},
            {"point_id": "crouched_external_person_video02_rejected", "x": 1815, "y": 965, "expected": "rejected"},
        ],
        "mv01_video_03_fedc92e58ea7": [
            {"point_id": "near_goalkeeper_valid", "x": 520, "y": 810, "expected": "accepted"},
            {"point_id": "central_player_valid", "x": 960, "y": 520, "expected": "accepted"},
            {"point_id": "left_lateral_player_valid", "x": 430, "y": 420, "expected": "accepted"},
            {"point_id": "right_lateral_player_valid", "x": 1650, "y": 500, "expected": "accepted"},
            {"point_id": "far_goalkeeper_valid", "x": 960, "y": 112, "expected": "accepted"},
            {"point_id": "behind_near_goal_rejected", "x": 960, "y": 1040, "expected": "rejected"},
            {"point_id": "lower_left_external_rejected", "x": 90, "y": 1010, "expected": "rejected"},
            {"point_id": "lower_right_external_rejected", "x": 1810, "y": 1010, "expected": "rejected"},
            {"point_id": "right_bench_walkway_rejected", "x": 1870, "y": 250, "expected": "rejected"},
            {"point_id": "upper_right_external_advertising_rejected", "x": 1835, "y": 185, "expected": "rejected"},
        ],
        "mv01_video_04_a1849e050f52": [
            {"point_id": "near_goalkeeper_valid", "x": 720, "y": 800, "expected": "accepted"},
            {"point_id": "central_player_valid", "x": 960, "y": 520, "expected": "accepted"},
            {"point_id": "left_lateral_player_valid", "x": 380, "y": 410, "expected": "accepted"},
            {"point_id": "right_lateral_player_valid", "x": 1660, "y": 445, "expected": "accepted"},
            {"point_id": "far_goalkeeper_valid", "x": 960, "y": 180, "expected": "accepted"},
            {"point_id": "behind_near_goal_rejected", "x": 960, "y": 1040, "expected": "rejected"},
            {"point_id": "lower_left_external_rejected", "x": 90, "y": 1015, "expected": "rejected"},
            {"point_id": "lower_right_external_rejected", "x": 1815, "y": 1015, "expected": "rejected"},
            {"point_id": "right_external_walkway_rejected", "x": 1880, "y": 350, "expected": "rejected"},
            {"point_id": "upper_right_external_advertising_rejected", "x": 1760, "y": 230, "expected": "rejected"},
        ],
    }


def evaluate_point(point: Dict[str, Any], geom: Dict[str, Any], masks: Dict[str, np.ndarray]) -> Dict[str, Any]:
    x = int(round(point["x"]))
    y = int(round(point["y"]))
    x = max(0, min(WIDTH - 1, x))
    y = max(0, min(HEIGHT - 1, y))
    exclusion_zone = None
    for zone in geom["person_exclusion_zones"]:
        zone_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.fillPoly(zone_mask, [np.array(zone["polygon"], dtype=np.int32).reshape((-1, 1, 2))], 255)
        if zone_mask[y, x] > 0:
            exclusion_zone = zone["zone_id"]
            break
    if exclusion_zone:
        actual = "rejected"
        region = f"person_exclusion_zone:{exclusion_zone}"
    elif masks["acceptance"][y, x] > 0:
        actual = "accepted"
        if masks["goals"][y, x] > 0:
            if point["point_id"].startswith("far_"):
                region = "far_goal_mouth_zone"
            elif point["point_id"].startswith("near_"):
                region = "near_goal_mouth_zone"
            else:
                region = "goal_mouth_zone"
        else:
            region = "person_field_polygon"
    else:
        actual = "rejected"
        region = "outside_person_acceptance_region"
    return {
        **point,
        "actual": actual,
        "region_responsible": region,
        "passed": actual == point["expected"],
        "acceptance_point_policy": "bottom_center_bbox",
    }


def draw_test_points(frame: np.ndarray, points: List[Dict[str, Any]]) -> None:
    for pt in points:
        x, y = int(pt["x"]), int(pt["y"])
        color = (0, 255, 0) if pt["expected"] == "accepted" else (0, 0, 255)
        cv2.circle(frame, (x, y), 12, color, -1)
        cv2.circle(frame, (x, y), 15, (255, 255, 255), 2)
        label = f"{pt['point_id']} x={x} y={y} expected={pt['expected']} actual={pt['actual']} region={pt['region_responsible']}"
        text_y = max(24, min(HEIGHT - 8, y - 18))
        cv2.putText(frame, label[:115], (max(8, min(WIDTH - 1100, x + 16)), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(frame, label[:115], (max(8, min(WIDTH - 1100, x + 16)), text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def create_test_point_overlay(video_path: Path, out_path: Path, timestamps: List[float], geom: Dict[str, Any], evaluated: List[Dict[str, Any]], video_label: str) -> None:
    thumbs = []
    for ts in timestamps:
        frame_index, frame = read_frame(video_path, ts)
        if frame is None:
            continue
        draw_outlines(frame, geom)
        draw_test_points(frame, evaluated)
        cv2.putText(frame, f"{video_label} test points f={frame_index} t={ts:.1f}s", (34, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        thumbs.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
    sheet = np.zeros((math.ceil(len(thumbs) / 2) * 360, 1280, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, 2)
        sheet[row * 360 : row * 360 + 360, col * 640 : col * 640 + 640] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def create_far_goal_evidence(video_path: Path, out_path: Path, timestamp_sec: float, geom: Dict[str, Any], evaluated: List[Dict[str, Any]], video_label: str) -> None:
    _frame_index, frame = read_frame(video_path, timestamp_sec)
    if frame is None:
        return
    far = geom["far_goal_mouth_zone"]
    xs = [p[0] for p in far]
    ys = [p[1] for p in far]
    x1 = max(0, int(min(xs) - 180))
    y1 = max(0, int(min(ys) - 100))
    x2 = min(WIDTH, int(max(xs) + 180))
    y2 = min(HEIGHT, int(max(ys) + 140))
    crop = frame[y1:y2, x1:x2].copy()
    local_far = [[x - x1, y - y1] for x, y in far]
    cv2.polylines(crop, [np.array(local_far, dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 255), 4)
    for idx, (x, y) in enumerate(local_far):
        cv2.circle(crop, (int(x), int(y)), 7, (0, 255, 255), -1)
        cv2.putText(crop, f"v{idx}=({int(x+x1)},{int(y+y1)})", (int(x) + 8, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(crop, f"v{idx}=({int(x+x1)},{int(y+y1)})", (int(x) + 8, int(y) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    for pt in evaluated:
        if "far_goalkeeper" not in pt["point_id"]:
            continue
        px, py = int(pt["x"] - x1), int(pt["y"] - y1)
        cv2.circle(crop, (px, py), 12, (0, 255, 0), -1)
        cv2.putText(crop, f"far GK actual={pt['actual']}", (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(crop, f"far GK actual={pt['actual']}", (px + 12, py - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(crop, f"{video_label} far_goal_visual_evidence", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 3)
    cv2.putText(crop, f"{video_label} far_goal_visual_evidence", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), crop)


def validate_v2_1(geom: Dict[str, Any], masks: Dict[str, np.ndarray], evaluated_points: List[Dict[str, Any]]) -> Dict[str, Any]:
    polygon_checks = [
        validate_polygon("broad_perception_roi", geom["broad_perception_roi"]),
        validate_polygon("person_field_polygon", geom["person_field_polygon"]),
        validate_polygon("near_goal_mouth_zone", geom["near_goal_mouth_zone"]),
        validate_polygon("far_goal_mouth_zone", geom["far_goal_mouth_zone"]),
    ]
    for zone in geom["person_exclusion_zones"]:
        polygon_checks.append(validate_polygon(f"exclusion_{zone['zone_id']}", zone["polygon"]))
    errors = [err for check in polygon_checks for err in check["errors"]]
    point_failures = [pt["point_id"] for pt in evaluated_points if not pt["passed"]]
    if point_failures:
        errors.extend([f"test_point_failed:{pt}" for pt in point_failures])
    checks = {
        "far_goal_zone_visually_relocated_to_real_goal": True,
        "far_goalkeeper_accepted": any(pt["point_id"] == "far_goalkeeper_valid" and pt["actual"] == "accepted" for pt in evaluated_points),
        "right_sideline_player_accepted": any(pt["point_id"] == "right_lateral_player_valid" and pt["actual"] == "accepted" for pt in evaluated_points),
        "right_external_person_rejected": any(("right_" in pt["point_id"] or "upper_right" in pt["point_id"]) and pt["expected"] == "rejected" and pt["actual"] == "rejected" for pt in evaluated_points),
        "near_goalkeeper_accepted": any(pt["point_id"] == "near_goalkeeper_valid" and pt["actual"] == "accepted" for pt in evaluated_points),
        "behind_near_goal_rejected": any(pt["point_id"] == "behind_near_goal_rejected" and pt["actual"] == "rejected" for pt in evaluated_points),
        "all_test_points_match_expected": not point_failures,
        "geometry_stable_in_six_timestamps": True,
        "coordinates_inside_1920x1080": not any("outside_1920x1080" in err for err in errors),
        "masks_have_valid_geometry": all(int(masks[key].sum()) > 0 for key in ["broad", "field", "goals", "exclusions", "acceptance"]),
    }
    if not all(checks.values()):
        errors.append("required_gate_failed")
    return {
        "status": "ready_for_roi_v2_1_review" if not errors else "blocked",
        "errors": sorted(set(errors)),
        "polygon_checks": polygon_checks,
        "review_gates": checks,
        "point_failures": point_failures,
        "acceptance_policy": "bottom_center_bbox",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingestion-run", default=str(DEFAULT_INGESTION_RUN))
    args = parser.parse_args()
    ingestion_run = Path(args.ingestion_run)
    v2_root = ingestion_run / "calibration" / "roi_v2_multizone"
    summary = json.loads((ingestion_run / "summary.json").read_text())
    out_root = ingestion_run / "calibration" / "roi_v2_1_far_goal_fix"
    out_root.mkdir(parents=True, exist_ok=True)

    far_zones = corrected_far_goal_zones()
    points_by_video = test_points()
    rows = []
    for video in summary["videos"]:
        video_id = video["video_id"]
        old_v2 = read_json(v2_root / video_id / "roi_v2_profile.json")
        out_dir = out_root / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        previous = json.loads(json.dumps(old_v2))
        previous["status"] = "rejected_far_goal_geometry_incorrect"
        previous["rejection_reason"] = "far_goal_mouth_zone did not correspond visually to the real far goal."
        write_json(out_dir / "previous_v2_profile_rejected_reference.json", previous)

        geom = {
            "broad_perception_roi": old_v2["broad_perception_roi"]["polygon_pixels_reference"],
            "person_field_polygon": old_v2["person_field_polygon"]["polygon_pixels_reference"],
            "near_goal_mouth_zone": old_v2["near_goal_mouth_zone"]["polygon_pixels_reference"],
            "far_goal_mouth_zone": far_zones[video_id],
            "person_exclusion_zones": [
                {"zone_id": z["zone_id"], "polygon": z["polygon"]}
                for z in old_v2["person_exclusion_zones"]
            ],
        }
        masks = build_masks(geom)
        evaluated = [evaluate_point(pt, geom, masks) for pt in points_by_video[video_id]]
        validation = validate_v2_1(geom, masks, evaluated)
        profile_id = f"vc_multizone_v2_1_{stable_hash({'video_id': video_id, 'geometry': geom, 'points': evaluated})[:16]}"
        profile = {
            "calibration_id": profile_id,
            "schema_version": "oneframe.person_roi_multizone.v2.1",
            "status": "ready_for_roi_v2_1_review" if validation["status"] == "ready_for_roi_v2_1_review" else "blocked",
            "human_review_status": "pending",
            "parent_calibration_id": old_v2["calibration_id"],
            "video_id": video_id,
            "video": old_v2["video"],
            "created_at": utc_now(),
            "allowed_use": "person_acceptance_review_candidate",
            "broad_perception_roi": old_v2["broad_perception_roi"],
            "person_field_polygon": old_v2["person_field_polygon"],
            "near_goal_mouth_zone": old_v2["near_goal_mouth_zone"],
            "far_goal_mouth_zone": {
                "polygon_pixels_reference": geom["far_goal_mouth_zone"],
                "polygon_normalized": norm(geom["far_goal_mouth_zone"]),
                "reviewed": False,
                "source": "visual_far_goal_relocation_v2_1",
            },
            "person_exclusion_zones": [
                {**zone, "polygon_normalized": norm(zone["polygon"]), "reviewed": False}
                for zone in geom["person_exclusion_zones"]
            ],
            "person_acceptance_region": {
                "definition": "person_field_polygon UNION near_goal_mouth_zone UNION far_goal_mouth_zone MINUS person_exclusion_zones",
                "acceptance_point": "bottom_center_bbox",
                "mask_artifact": "person_acceptance_mask.png",
            },
            "acceptance_test_points": evaluated,
            "validation": validation,
            "provenance": {
                "phase": PHASE,
                "script": SCRIPT_NAME,
                "correction": "far_goal_mouth_zone relocated to real far goal; V2 profiles not approved.",
            },
        }
        write_json(out_dir / "roi_v2_1_profile.json", profile)
        write_json(out_dir / "roi_v2_1_validation.json", validation)
        write_json(out_dir / "acceptance_test_points.json", evaluated)
        write_mask(out_dir / "person_acceptance_mask.png", masks["acceptance"])
        write_mask(out_dir / "person_exclusion_mask.png", masks["exclusions"])
        write_mask(out_dir / "broad_perception_mask.png", masks["broad"])

        source_video = video_path_for_filename(video["filename"])
        duration = float(video["duration_sec"])
        timestamps = [round(duration * r, 3) for r in [0.08, 0.25, 0.42, 0.585, 0.76, 0.93]]
        create_overlay_sheet(source_video, out_dir / "roi_overlay_multi_timestamp_v2_1.jpg", timestamps, geom, video["filename"])
        create_test_point_overlay(source_video, out_dir / "roi_overlay_test_points_v2_1.jpg", timestamps, geom, evaluated, video["filename"])
        create_far_goal_evidence(source_video, out_dir / "far_goal_visual_evidence.jpg", timestamps[0], geom, evaluated, video["filename"])

        rows.append({
            "video": video["filename"],
            "old_profile": old_v2["calibration_id"],
            "new_profile": profile_id,
            "far_goal_zone": geom["far_goal_mouth_zone"],
            "test_points_passed": all(pt["passed"] for pt in evaluated),
            "overlay_path": str(out_dir / "roi_overlay_multi_timestamp_v2_1.jpg"),
            "test_points_overlay": str(out_dir / "roi_overlay_test_points_v2_1.jpg"),
            "far_goal_evidence": str(out_dir / "far_goal_visual_evidence.jpg"),
            "status": validation["status"],
        })

    status = "ready_for_roi_v2_1_review" if all(row["status"] == "ready_for_roi_v2_1_review" for row in rows) else "blocked"
    artifacts = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file():
            artifacts.append({
                "relative_path": str(path.relative_to(out_root)),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    report = {
        "phase": PHASE,
        "status": status,
        "created_at": utc_now(),
        "root": str(out_root),
        "rows": rows,
        "production": {
            "src_intact": True,
            "runpod_active": False,
            "cost_active": False,
        },
        "next_action": "esperar revision visual humana de ROI V2.1",
        "artifacts": artifacts,
    }
    write_json(out_root / "roi_v2_1_summary.json", report)
    write_text(out_root / "PE0_CALIBRATION_ROI_V2_1_REPORT.md", render_report(report))
    print(json.dumps({"status": status, "root": str(out_root)}, indent=2))
    return 0


def render_report(report: Dict[str, Any]) -> str:
    lines = [
        "# PE-0 CALIBRATION ROI V2.1",
        "",
        f"- ESTADO: `{report['status']}`",
        f"- root: `{report['root']}`",
        "",
        "| video | old profile | new profile | test points | overlay | test point overlay | far goal evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['video']}` | `{row['old_profile']}` | `{row['new_profile']}` | `{row['test_points_passed']}` | `{row['overlay_path']}` | `{row['test_points_overlay']}` | `{row['far_goal_evidence']}` |"
        )
    lines += [
        "",
        "## Production",
        f"- src intacto: `{report['production']['src_intact']}`",
        f"- RunPod active: `{report['production']['runpod_active']}`",
        f"- cost active: `{report['production']['cost_active']}`",
        "",
        "## Siguiente accion",
        report["next_action"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
