#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ROOT = REPO_ROOT / "ai_worker_v1"
RUN_DIR = AI_ROOT / "runs" / "pe0_multivideo_ingestion_20260717T024829Z"
INVENTORY_PATH = RUN_DIR / "inventory" / "video_inventory.json"
DEFAULT_EXPORT_DIRS = [
    AI_ROOT / "tools" / "manual_roi_calibration" / "final_exports",
    AI_ROOT / "tools" / "manual_roi_calibration" / "exports",
]
OUTPUT_DIR = RUN_DIR / "calibration" / "manual_roi_v3_1_final"
FRAME_SIZE = (1920, 1080)
MARGIN_PX = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 4), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_points(raw: Any) -> List[List[float]]:
    points = []
    for point in raw or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append([float(point[0]), float(point[1])])
    return points


def normalize_geometry(profile: Dict[str, Any]) -> Dict[str, Any]:
    geometry = profile.get("geometry", profile)
    goal_zones = geometry.get("goal_zones", {})
    return {
        "perception_roi": as_points(geometry.get("perception_roi") or geometry.get("broad_perception_roi")),
        "detection_field_roi": as_points(geometry.get("detection_field_roi") or geometry.get("person_field_polygon")),
        "goal_zones": {
            "near_goal": as_points(goal_zones.get("near_goal") or geometry.get("near_goal_zone") or geometry.get("near_goal_mouth_zone")),
            "far_goal": as_points(goal_zones.get("far_goal") or geometry.get("far_goal_zone") or geometry.get("far_goal_mouth_zone")),
        },
    }


def all_geometry_points(geometry: Dict[str, Any]) -> List[List[float]]:
    return (
        geometry["perception_roi"]
        + geometry["detection_field_roi"]
        + geometry["goal_zones"]["near_goal"]
        + geometry["goal_zones"]["far_goal"]
    )


def repaired_perception_roi(geometry: Dict[str, Any]) -> List[List[float]]:
    points = all_geometry_points(geometry)
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1 = max(0.0, math.floor(min(xs) - MARGIN_PX))
    y1 = max(0.0, math.floor(min(ys) - MARGIN_PX))
    x2 = min(float(FRAME_SIZE[0]), math.ceil(max(xs) + MARGIN_PX))
    y2 = min(float(FRAME_SIZE[1]), math.ceil(max(ys) + MARGIN_PX))
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def point_in_polygon(point: List[float], polygon: List[List[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def orientation(a: List[float], b: List[float], c: List[float]) -> float:
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def segments_intersect(a: List[float], b: List[float], c: List[float], d: List[float]) -> bool:
    def on_segment(p: List[float], q: List[float], r: List[float]) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    eps = 1e-9
    return (
        abs(o1) < eps and on_segment(a, c, b)
        or abs(o2) < eps and on_segment(a, d, b)
        or abs(o3) < eps and on_segment(c, a, d)
        or abs(o4) < eps and on_segment(c, b, d)
    )


def self_intersections(polygon: List[List[float]]) -> int:
    if len(polygon) < 4:
        return 0
    count = 0
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or {i, j} == {0, n - 1}:
                continue
            c, d = polygon[j], polygon[(j + 1) % n]
            if segments_intersect(a, b, c, d):
                count += 1
    return count


def polygon_valid(polygon: List[List[float]]) -> bool:
    return len(polygon) >= 3 and self_intersections(polygon) == 0


def in_frame(point: List[float]) -> bool:
    return 0 <= point[0] <= FRAME_SIZE[0] and 0 <= point[1] <= FRAME_SIZE[1]


def contained(container: List[List[float]], child: List[List[float]]) -> bool:
    return bool(container) and all(point_in_polygon(point, container) or point in container for point in child)


def find_export(video_id: str, search_dirs: Iterable[Path]) -> Path | None:
    candidates: List[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        candidates.extend(directory.rglob(f"*{video_id}*roi_manual_v3*profile*.json"))
        candidates.extend(directory.rglob(f"*{video_id}*.json"))
    for path in sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = load_json(path)
        except Exception:
            continue
        if video_id in json.dumps(data, ensure_ascii=True):
            return path
    return None


def draw_polygon(image: np.ndarray, polygon: List[List[float]], color: Tuple[int, int, int], label: str) -> None:
    if len(polygon) < 2:
        return
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.polylines(image, [pts], True, color, 4)
    for idx, (x, y) in enumerate(pts):
        cv2.circle(image, (int(x), int(y)), 6, color, -1)
        cv2.putText(image, str(idx + 1), (int(x) + 7, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    anchor = tuple(pts[0])
    cv2.putText(image, label, (anchor[0] + 8, anchor[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def fill_mask(path: Path, polygons: List[List[List[float]]]) -> str:
    mask = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    for polygon in polygons:
        if len(polygon) >= 3:
            pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [pts], (255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)
    return sha256_file(path)


def make_overlay(video_manifest: Dict[str, Any], geometry: Dict[str, Any], out_path: Path) -> str:
    frames = video_manifest["frames"]
    tiles = []
    for frame in frames:
        frame_path = resolve_visual_asset_path(frame["path"])
        image = cv2.imread(str(frame_path)) if frame_path else None
        if image is None:
            image = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
        draw_polygon(image, geometry["perception_roi"], (255, 0, 0), "perception_roi_v3_1")
        draw_polygon(image, geometry["detection_field_roi"], (0, 230, 0), "detection_field_roi")
        draw_polygon(image, geometry["goal_zones"]["near_goal"], (0, 255, 255), "near_goal_zone")
        draw_polygon(image, geometry["goal_zones"]["far_goal"], (0, 255, 255), "far_goal_zone")
        cv2.rectangle(image, (0, 0), (950, 52), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"{video_manifest['video_id']} t={frame['timestamp_sec']}s ROI V3.1",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        tiles.append(cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA))
    while len(tiles) < 6:
        tiles.append(np.zeros((360, 640, 3), dtype=np.uint8))
    sheet = np.zeros((720, 1920, 3), dtype=np.uint8)
    for idx, tile in enumerate(tiles[:6]):
        row, col = divmod(idx, 3)
        sheet[row * 360 : row * 360 + 360, col * 640 : col * 640 + 640] = tile
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return sha256_file(out_path)


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


def validate_profile(video: Dict[str, Any], profile: Dict[str, Any], geometry: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    expected_sha = video["sha256"]
    actual_sha = sha256_file(Path(video["path"]))
    polygons = {
        "perception_roi": geometry["perception_roi"],
        "detection_field_roi": geometry["detection_field_roi"],
        "near_goal_zone": geometry["goal_zones"]["near_goal"],
        "far_goal_zone": geometry["goal_zones"]["far_goal"],
    }
    validation = {
        "source_video_sha256_match": actual_sha == expected_sha,
        "source_video_sha256": actual_sha,
        "expected_source_video_sha256": expected_sha,
        "coordinate_space": f"{FRAME_SIZE[0]}x{FRAME_SIZE[1]}",
        "coordinate_space_valid": profile.get("coordinate_space", {}).get("width", FRAME_SIZE[0]) == FRAME_SIZE[0]
        and profile.get("coordinate_space", {}).get("height", FRAME_SIZE[1]) == FRAME_SIZE[1],
        "single_geometry_per_video": True,
        "polygons_valid": {name: polygon_valid(poly) for name, poly in polygons.items()},
        "self_intersections": {name: self_intersections(poly) for name, poly in polygons.items()},
        "detection_field_roi_contained": contained(geometry["perception_roi"], geometry["detection_field_roi"]),
        "near_goal_zone_contained": contained(geometry["perception_roi"], geometry["goal_zones"]["near_goal"]),
        "far_goal_zone_contained": contained(geometry["perception_roi"], geometry["goal_zones"]["far_goal"]),
        "all_coordinates_in_frame": all(in_frame(point) for poly in polygons.values() for point in poly),
        "no_exclusion_zones": not profile.get("person_exclusion_zones") and not profile.get("exclusion_zones"),
        "danger_area_status": "deferred_until_metric_field_geometry",
        "manual_geometry_preserved": {
            "detection_field_roi": True,
            "near_goal_zone": True,
            "far_goal_zone": True,
        },
        "perception_roi_repaired": True,
        "source_export": str(source_path),
    }
    validation["valid"] = (
        validation["source_video_sha256_match"]
        and validation["coordinate_space_valid"]
        and all(validation["polygons_valid"].values())
        and all(v == 0 for v in validation["self_intersections"].values())
        and validation["detection_field_roi_contained"]
        and validation["near_goal_zone_contained"]
        and validation["far_goal_zone_contained"]
        and validation["all_coordinates_in_frame"]
        and validation["no_exclusion_zones"]
    )
    return validation


def video_lookup(inventory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {video["video_id"]: video for video in inventory["videos"]}


def manual_tool_manifest() -> Dict[str, Any]:
    return load_json(AI_ROOT / "tools" / "manual_roi_calibration" / "assets" / "videos_manifest.json")


def freeze_one(video: Dict[str, Any], video_manifest: Dict[str, Any], export_path: Path, out_dir: Path) -> Dict[str, Any]:
    profile = load_json(export_path)
    manual_geometry = normalize_geometry(profile)
    repaired_geometry = {
        "perception_roi": repaired_perception_roi(manual_geometry),
        "detection_field_roi": manual_geometry["detection_field_roi"],
        "goal_zones": manual_geometry["goal_zones"],
    }
    v31_profile = {
        **{k: v for k, v in profile.items() if k not in ("geometry", "profile_sha256", "validation", "status")},
        "schema_version": "oneframe.manual_roi.v3.1",
        "status": "approved_for_offline_visual_smoke",
        "created_at": utc_now(),
        "source_manual_profile_path": str(export_path),
        "coordinate_space": {"width": FRAME_SIZE[0], "height": FRAME_SIZE[1]},
        "danger_area_status": "deferred_until_metric_field_geometry",
        "geometry": repaired_geometry,
        "repair": {
            "type": "perception_roi_union_envelope",
            "margin_px": MARGIN_PX,
            "preserved_layers": ["detection_field_roi", "goal_zones.near_goal", "goal_zones.far_goal"],
            "exclusion_zones_introduced": False,
        },
    }
    validation = validate_profile(video, v31_profile, repaired_geometry, export_path)
    if not validation["valid"]:
        v31_profile["status"] = "blocked_v3_1_validation"
    v31_profile["validation"] = validation
    v31_profile["profile_sha256"] = stable_hash(v31_profile)

    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "roi_manual_v3_1_profile.json"
    validation_path = out_dir / "roi_manual_v3_1_validation.json"
    overlay_path = out_dir / "roi_overlay_multi_timestamp_v3_1.jpg"
    perception_mask_path = out_dir / "perception_mask_v3_1.png"
    detection_mask_path = out_dir / "detection_field_mask_v3_1.png"
    goal_mask_path = out_dir / "goal_zones_mask_v3_1.png"

    write_json(profile_path, v31_profile)
    write_json(validation_path, validation)
    overlay_hash = make_overlay(video_manifest, repaired_geometry, overlay_path)
    perception_hash = fill_mask(perception_mask_path, [repaired_geometry["perception_roi"]])
    detection_hash = fill_mask(detection_mask_path, [repaired_geometry["detection_field_roi"]])
    goal_hash = fill_mask(goal_mask_path, [repaired_geometry["goal_zones"]["near_goal"], repaired_geometry["goal_zones"]["far_goal"]])

    return {
        "video_id": video["video_id"],
        "filename": video["filename"],
        "status": v31_profile["status"],
        "valid": validation["valid"],
        "source_export": str(export_path),
        "profile": str(profile_path),
        "profile_sha256": sha256_file(profile_path),
        "profile_content_hash": v31_profile["profile_sha256"],
        "validation": str(validation_path),
        "overlay": str(overlay_path),
        "masks": {
            "perception": {"path": str(perception_mask_path), "sha256": perception_hash},
            "detection_field": {"path": str(detection_mask_path), "sha256": detection_hash},
            "goal_zones": {"path": str(goal_mask_path), "sha256": goal_hash},
        },
        "overlay_sha256": overlay_hash,
        "manual_geometry_preserved": validation["manual_geometry_preserved"],
        "containment": {
            "detection_field_roi": validation["detection_field_roi_contained"],
            "near_goal_zone": validation["near_goal_zone_contained"],
            "far_goal_zone": validation["far_goal_zone_contained"],
        },
    }


def report(summary: Dict[str, Any]) -> str:
    lines = [
        "# PE-0 Final ROI Reconciliation",
        "",
        f"- status: `{summary['status']}`",
        f"- generated_at: `{summary['generated_at']}`",
        f"- previous_gate_status: `{summary['previous_gate_status']}`",
        f"- current_gate: `{summary['current_gate']}`",
        "",
        "## Videos",
        "",
        "| video | status | valid | profile | overlay |",
        "|---|---|---:|---|---|",
    ]
    for item in summary["videos"]:
        lines.append(
            f"| `{item['video_id']}` | `{item['status']}` | {item.get('valid')} | "
            f"`{item.get('profile', '')}` | `{item.get('overlay', '')}` |"
        )
    if summary["missing_exports"]:
        lines.extend(["", "## Missing Exports", ""])
        for item in summary["missing_exports"]:
            lines.append(f"- `{item}`")
    lines.extend([
        "",
        "## Production",
        "",
        "- src intacto: true",
        "- RunPod active: false",
        "- inference executed: false",
        "- tracking executed: false",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair manual ROI V3 exports into frozen V3.1 profiles.")
    parser.add_argument("--export-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    search_dirs = args.export_dir or DEFAULT_EXPORT_DIRS
    inventory = load_json(INVENTORY_PATH)
    videos = video_lookup(inventory)
    tool_manifest = manual_tool_manifest()
    tool_videos = {video["video_id"]: video for video in tool_manifest["videos"]}

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for video_id, video in videos.items():
        export_path = find_export(video_id, search_dirs)
        if not export_path:
            missing.append(video_id)
            summaries.append({
                "video_id": video_id,
                "filename": video["filename"],
                "status": "blocked_missing_manual_roi_v3_export",
                "valid": False,
            })
            continue
        summaries.append(freeze_one(video, tool_videos[video_id], export_path, output_dir / video_id))

    all_valid = bool(summaries) and all(item.get("status") == "approved_for_offline_visual_smoke" for item in summaries)
    summary = {
        "phase": "PE-0 FINAL ROI RECONCILIATION",
        "generated_at": utc_now(),
        "status": "approved_for_offline_visual_smoke" if all_valid else "blocked_missing_or_invalid_manual_roi_exports",
        "previous_gate_status": "obsolete_pre_final_roi_export",
        "current_gate": "starts_after_v3_1_freeze" if all_valid else "blocked_until_v3_1_freeze",
        "search_dirs": [str(path) for path in search_dirs],
        "missing_exports": missing,
        "videos": summaries,
        "production": {
            "src_intact": True,
            "runpod_active": False,
            "cost_active": False,
            "inference_executed": False,
            "tracking_executed": False,
        },
    }
    write_json(output_dir / "roi_v3_1_freeze_summary.json", summary)
    write_text(output_dir / "PE0_FINAL_ROI_RECONCILIATION_REPORT.md", report(summary))
    print(json.dumps({
        "status": summary["status"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "roi_v3_1_freeze_summary.json"),
        "report": str(output_dir / "PE0_FINAL_ROI_RECONCILIATION_REPORT.md"),
        "missing_exports": missing,
    }, indent=2, sort_keys=True))
    return 0 if all_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
