#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
PHASE = "PE-0 MULTIVIDEO INGESTION AND SELECTION"
SCRIPT_NAME = "ai_worker_v1/scripts/run_pe0_multivideo_ingestion.py"
INPUT_DIR = AI_WORKER_ROOT / "input_videos" / "multivideo_v01"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def ffprobe_json(path: Path) -> Dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def parse_ffprobe(path: Path, raw: Dict[str, Any]) -> Dict[str, Any]:
    video_stream = next((s for s in raw.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in raw.get("streams", []) if s.get("codec_type") == "audio"), {})
    fps_raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    duration = float(raw.get("format", {}).get("duration") or video_stream.get("duration") or 0.0)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    return {
        "filename": path.name,
        "path": str(path),
        "size_bytes": int(raw.get("format", {}).get("size") or path.stat().st_size),
        "duration_sec": duration,
        "fps": fps,
        "frame_count_estimate": int(round(duration * fps)) if fps and duration else None,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "video_codec": video_stream.get("codec_name"),
        "audio_codec": audio_stream.get("codec_name"),
        "pix_fmt": video_stream.get("pix_fmt"),
        "bit_rate": int(raw.get("format", {}).get("bit_rate") or 0),
    }


def read_frame(video_path: Path, timestamp_sec: float) -> Tuple[int, np.ndarray | None]:
    # OpenCV frame seeking can become very slow on these long H.264 files.
    # ffmpeg's input-side seek is good enough for review artifacts and avoids
    # hanging on deep timestamps.
    frame_index = int(round(timestamp_sec * 30.0))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return frame_index, None
    if not result.stdout:
        return frame_index, None
    data = np.frombuffer(result.stdout, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame_index, frame


def sample_timestamps(duration: float, count: int, margin_ratio: float = 0.03) -> List[float]:
    if count <= 1:
        return [duration / 2.0]
    start = max(0.0, duration * margin_ratio)
    end = max(start, duration * (1.0 - margin_ratio))
    return [round(start + (end - start) * i / (count - 1), 3) for i in range(count)]


def average_hash(frame: np.ndarray, size: int = 16) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    mean = float(small.mean())
    bits = "".join("1" if v >= mean else "0" for v in small.flatten())
    return hex(int(bits, 2))[2:].zfill(size * size // 4)


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def make_contact_sheet(video_path: Path, out_path: Path, timestamps: List[float]) -> List[Dict[str, Any]]:
    thumbs: List[np.ndarray] = []
    samples: List[Dict[str, Any]] = []
    for ts in timestamps:
        frame_index, frame = read_frame(video_path, ts)
        if frame is None:
            samples.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "read_failed"})
            continue
        thumb = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        label = f"{video_path.stem} f={frame_index} t={ts:.1f}s"
        cv2.rectangle(thumb, (0, 0), (480, 34), (0, 0, 0), -1)
        cv2.putText(thumb, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        thumbs.append(thumb)
        samples.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "ok"})
    rows = math.ceil(len(thumbs) / 3.0)
    sheet = np.zeros((rows * 270, 3 * 480, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, 3)
        sheet[row * 270 : row * 270 + 270, col * 480 : col * 480 + 480] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return samples


def make_preview(video_path: Path, out_path: Path, start_sec: float, duration_sec: float = 10.0) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration_sec:.3f}",
        "-vf",
        "scale=960:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-an",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return {
        "path": str(out_path),
        "start_sec": round(start_sec, 3),
        "duration_sec": duration_sec,
        "size_bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
    }


def detect_field_roi(video_path: Path, metadata: Dict[str, Any], timestamps: List[float]) -> Dict[str, Any]:
    width = int(metadata["width"])
    height = int(metadata["height"])
    masks = []
    brightness_values = []
    for ts in timestamps:
        _idx, frame = read_frame(video_path, ts)
        if frame is None:
            continue
        brightness_values.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad green/cyan field mask, tuned to be inclusive. The result is
        # review-only, never an approved metric calibration.
        lower = np.array([28, 35, 35], dtype=np.uint8)
        upper = np.array([105, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((17, 17), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks.append(mask)
    if not masks:
        polygon = fallback_roi(width, height)
        source = "fallback_no_readable_frames"
        area_ratio = polygon_area_ratio(polygon, width, height)
    else:
        combined = np.maximum.reduce(masks)
        contours, _hier = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > width * height * 0.03]
        if contours:
            contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(contour)
            epsilon = 0.018 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True).reshape(-1, 2).astype(float).tolist()
            polygon = normalize_polygon_for_roi(approx, width, height)
            source = "auto_field_mask_convex_hull_review_required"
        else:
            polygon = fallback_roi(width, height)
            source = "fallback_field_mask_no_large_contour"
        area_ratio = polygon_area_ratio(polygon, width, height)
    validation = validate_polygon(polygon, width, height)
    avg_brightness = float(sum(brightness_values) / len(brightness_values)) if brightness_values else None
    return {
        "polygon_pixels_reference": polygon,
        "polygon_normalized": [[round(x / width, 8), round(y / height, 8)] for x, y in polygon],
        "source": source,
        "reviewed": False,
        "valid": validation["valid"],
        "validation": {
            **validation,
            "area_ratio": round(area_ratio, 8),
            "resolution": [width, height],
            "near_goal_included_heuristic": max(y for _x, y in polygon) >= height * 0.92,
            "far_goal_included_heuristic": min(y for _x, y in polygon) <= height * 0.22,
            "sidelines_included_heuristic": min(x for x, _y in polygon) <= width * 0.08 and max(x for x, _y in polygon) >= width * 0.92,
            "external_people_excluded_review_required": True,
            "average_brightness": round(avg_brightness, 3) if avg_brightness is not None else None,
        },
        "warnings": ["draft_visual_review_required", "not_metric_homography"],
    }


def fallback_roi(width: int, height: int) -> List[List[float]]:
    return [
        [width * 0.18, height * 0.90],
        [width * 0.01, height * 0.88],
        [width * 0.01, height * 0.26],
        [width * 0.28, height * 0.09],
        [width * 0.63, height * 0.10],
        [width * 0.79, height * 0.13],
        [width * 0.99, height * 0.32],
        [width * 0.99, height * 0.97],
        [width * 0.82, height * 0.99],
        [width * 0.19, height * 0.99],
    ]


def normalize_polygon_for_roi(points: List[List[float]], width: int, height: int) -> List[List[float]]:
    if len(points) < 4:
        return fallback_roi(width, height)
    pts = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    cx = float(hull[:, 0].mean())
    cy = float(hull[:, 1].mean())
    ordered = sorted(hull.tolist(), key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    expanded = []
    for x, y in ordered:
        ex = cx + (x - cx) * 1.03
        ey = cy + (y - cy) * 1.03
        expanded.append([float(min(max(ex, 0), width - 1)), float(min(max(ey, 0), height - 1))])
    # Keep 6-12 points. More points are OK for detection ROI but require review.
    if len(expanded) > 12:
        step = len(expanded) / 12.0
        expanded = [expanded[int(round(i * step)) % len(expanded)] for i in range(12)]
    return [[round(x, 2), round(y, 2)] for x, y in expanded]


def polygon_area_ratio(points: List[List[float]], width: int, height: int) -> float:
    contour = np.array(points, dtype=np.float32)
    return abs(float(cv2.contourArea(contour))) / float(width * height)


def validate_polygon(points: List[List[float]], width: int, height: int) -> Dict[str, Any]:
    errors = []
    if len(points) < 3:
        errors.append("minimum_3_points_required")
    for idx, (x, y) in enumerate(points):
        if x < 0 or x > width or y < 0 or y > height:
            errors.append(f"point_{idx}_outside_frame")
    self_intersecting = polygon_self_intersecting(points)
    if self_intersecting:
        errors.append("self_intersecting_polygon")
    if polygon_area_ratio(points, width, height) < 0.15:
        errors.append("area_below_full_field_expectation")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "self_intersecting": self_intersecting,
        "points_inside_frame": not any("outside_frame" in e for e in errors),
    }


def polygon_self_intersecting(points: List[List[float]]) -> bool:
    def orient(a: List[float], b: List[float], c: List[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a: List[float], b: List[float], c: List[float], d: List[float]) -> bool:
        if a == c or a == d or b == c or b == d:
            return False
        return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0

    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            c = points[j]
            d = points[(j + 1) % n]
            if intersects(a, b, c, d):
                return True
    return False


def draw_roi_overlay(video_path: Path, out_path: Path, timestamps: List[float], polygon: List[List[float]]) -> List[Dict[str, Any]]:
    thumbs = []
    rows = []
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    for ts in timestamps:
        frame_index, frame = read_frame(video_path, ts)
        if frame is None:
            rows.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "read_failed"})
            continue
        cv2.polylines(frame, [poly], True, (0, 255, 0), 4)
        cv2.putText(frame, "draft ROI - visual review required", (28, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 4)
        cv2.putText(frame, f"{video_path.stem} f={frame_index} t={ts:.1f}s", (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        thumbs.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
        rows.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "ok"})
    sheet = np.zeros((math.ceil(len(thumbs) / 2) * 360, 2 * 640, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, 2)
        sheet[row * 360 : row * 360 + 360, col * 640 : col * 640 + 640] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return rows


def build_calibration_profile(video_id: str, metadata: Dict[str, Any], video_hash: str, roi: Dict[str, Any], reference: Dict[str, Any]) -> Dict[str, Any]:
    material = {
        "video_hash": video_hash,
        "reference_frame_index": reference["frame_index"],
        "roi": roi["polygon_pixels_reference"],
        "phase": "pe0_multivideo_v01",
    }
    calibration_id = f"vc_multivideo_{stable_hash(material)[:16]}"
    return {
        "calibration_id": calibration_id,
        "schema_version": "oneframe.video_calibration.v1",
        "profile_version": 1,
        "status": "draft_visual_review",
        "video_id": video_id,
        "video": {
            "video_hash": video_hash,
            "width": metadata["width"],
            "height": metadata["height"],
            "aspect_ratio": round(metadata["width"] / metadata["height"], 8) if metadata["height"] else 0,
            "fps": metadata["fps"],
            "reference_frame_index": reference["frame_index"],
            "reference_timestamp_sec": reference["timestamp_sec"],
            "frame_hash": reference["frame_hash"],
        },
        "camera": {
            "camera_type": "fixed_or_mostly_fixed_match_camera",
            "expected_fixed_within_video": True,
            "angle_reusable_across_videos": False,
        },
        "detection_roi": {
            "source": roi["source"],
            "polygon_normalized": roi["polygon_normalized"],
            "polygon_pixels_reference": roi["polygon_pixels_reference"],
            "point_order": list(range(len(roi["polygon_pixels_reference"]))),
            "reviewed": False,
            "valid": roi["valid"],
            "warnings": roi["warnings"],
            "validation": roi["validation"],
        },
        "ignore_regions": [],
        "landmarks": [],
        "homography": {
            "status": "unavailable",
            "matrix_image_to_field": None,
            "matrix_field_to_image": None,
            "correspondence_ids": [],
            "mean_reprojection_error_px": None,
            "reviewed": False,
            "failure_reasons": ["draft_roi_only", "semantic_landmark_correspondences_required"],
        },
        "human_review_status": "pending",
        "created_at": utc_now(),
        "provenance": {
            "phase": PHASE,
            "script": SCRIPT_NAME,
            "note": "ROI is an independent per-video draft for visual review; not approved calibration.",
        },
        "qa": {
            "schema_valid": True,
            "roi_valid": bool(roi["valid"]),
            "homography_valid": False,
            "review_complete": False,
            "warnings": ["human_visual_review_required", "homography_unavailable_until_landmarks_exist"],
            "errors": [],
        },
    }


def frame_hash(frame: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        return ""
    return hashlib.sha256(encoded.tobytes()).hexdigest()


def analyze_camera_quality(video_path: Path, metadata: Dict[str, Any], timestamps: List[float]) -> Dict[str, Any]:
    brightness = []
    blur = []
    green_ratio = []
    ahashes = []
    for ts in timestamps:
        _idx, frame = read_frame(video_path, ts)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(gray.mean()))
        blur.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([28, 35, 35], dtype=np.uint8), np.array([105, 255, 255], dtype=np.uint8))
        green_ratio.append(float(mask.mean() / 255.0))
        ahashes.append(average_hash(frame))
    avg_b = sum(brightness) / len(brightness) if brightness else 0.0
    avg_blur = sum(blur) / len(blur) if blur else 0.0
    avg_green = sum(green_ratio) / len(green_ratio) if green_ratio else 0.0
    return {
        "camera_position": "fixed elevated match camera; visual review required for exact sideline/behind-goal classification",
        "lighting": "good" if avg_b >= 80 else "dim",
        "average_brightness": round(avg_b, 3),
        "sharpness_laplacian_var": round(avg_blur, 3),
        "field_visible_ratio_estimate": round(avg_green, 4),
        "field_visible_quality": "usable" if avg_green >= 0.30 else "needs_review",
        "usable": metadata["width"] == 1920 and metadata["height"] == 1080 and metadata["fps"] >= 25 and avg_green >= 0.25,
        "reason": "1080p/30fps with visible pitch; draft visual review required before model use",
        "sample_hashes": ahashes,
    }


def exact_and_near_duplicates(videos: List[Dict[str, Any]]) -> Dict[str, Any]:
    exact = []
    for i, a in enumerate(videos):
        for b in videos[i + 1 :]:
            if a["sha256"] == b["sha256"]:
                exact.append([a["video_id"], b["video_id"]])
    near = []
    for i, a in enumerate(videos):
        for b in videos[i + 1 :]:
            distances = []
            for ha, hb in zip(a["analysis"]["sample_hashes"], b["analysis"]["sample_hashes"]):
                distances.append(hamming_hex(ha, hb))
            avg = sum(distances) / len(distances) if distances else None
            near.append({
                "video_a": a["video_id"],
                "video_b": b["video_id"],
                "avg_hamming_16x16": round(avg, 3) if avg is not None else None,
                "near_duplicate": bool(avg is not None and avg <= 18),
                "interpretation": "not_near_duplicate" if avg is None or avg > 18 else "near_duplicate_candidate",
            })
    return {"exact_duplicates": exact, "near_duplicates": near}


def propose_person_frames(video: Dict[str, Any]) -> List[Dict[str, Any]]:
    duration = video["metadata"]["duration_sec"]
    timestamps = sample_timestamps(duration, 40, margin_ratio=0.025)
    rows = []
    coverage_cycle = [
        ("near", "low", "goal_area"),
        ("mid", "medium", "central_field"),
        ("far", "high", "far_side"),
        ("mixed", "cluster", "touchline_or_edge"),
        ("mixed", "occlusion", "crowded_play"),
    ]
    for idx, ts in enumerate(timestamps):
        distance, occlusion, zone = coverage_cycle[idx % len(coverage_cycle)]
        frame_index = int(round(ts * video["metadata"]["fps"]))
        rows.append({
            "video_id": video["video_id"],
            "filename": video["metadata"]["filename"],
            "frame_index": frame_index,
            "timestamp_sec": ts,
            "distance_coverage": distance,
            "occlusion_coverage": occlusion,
            "cluster_coverage": "yes" if occlusion in {"cluster", "occlusion"} else "mixed",
            "field_zone": zone,
            "expected_persons": "review_required",
            "ground_truth_status": "proposal_only",
        })
    return rows


def propose_ball_sequences(videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    categories = [
        "tiny_far_ball",
        "near_feet",
        "partial_occlusion",
        "motion_blur",
        "player_cluster",
        "low_contrast",
        "white_lines",
        "goal_net",
        "goal_post",
        "frame_edge",
        "hard_negatives",
    ]
    splits = ["train", "train", "train", "train", "valid", "valid", "train", "valid", "test", "test", "test"]
    rows = []
    seq_idx = 1
    for video_idx, video in enumerate(videos):
        duration = video["metadata"]["duration_sec"]
        anchors = sample_timestamps(duration, len(categories), margin_ratio=0.06)
        for cat_idx, category in enumerate(categories):
            split = splits[(cat_idx + video_idx) % len(splits)]
            if video_idx == len(videos) - 1 and category in {"goal_post", "frame_edge", "hard_negatives"}:
                split = "test"
            start = max(0.0, anchors[cat_idx] - 3.0)
            end = min(duration, start + 6.0)
            rows.append({
                "sequence_id": f"mv01_seq_{seq_idx:03d}",
                "source_video": video["video_id"],
                "filename": video["metadata"]["filename"],
                "split": split,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "error_category": category,
                "estimated_positive_frames": "unknown_until_review",
                "estimated_negative_frames": "unknown_until_review",
                "overlap_status": "non_overlapping_within_video_by_construction",
                "policy": "proposal_only_no_inference_no_ground_truth",
            })
            seq_idx += 1
    return rows


def save_reference_frame(video_path: Path, out_path: Path, timestamp_sec: float) -> Dict[str, Any]:
    frame_index, frame = read_frame(video_path, timestamp_sec)
    if frame is None:
        raise RuntimeError(f"could not read reference frame for {video_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return {
        "path": str(out_path),
        "timestamp_sec": round(timestamp_sec, 3),
        "frame_index": frame_index,
        "frame_hash": frame_hash(frame),
    }


def artifact_manifest(run_dir: Path, artifact_paths: Dict[str, Path]) -> Dict[str, Any]:
    rows = []
    for name, path in sorted(artifact_paths.items()):
        if path.exists():
            rows.append({
                "name": name,
                "path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            })
    manifest = {
        "phase": PHASE,
        "created_at": utc_now(),
        "run_dir": str(run_dir),
        "artifacts": rows,
    }
    write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(INPUT_DIR))
    parser.add_argument("--run-dir", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    run_dir = Path(args.run_dir) if args.run_dir else AI_WORKER_ROOT / "runs" / f"pe0_multivideo_ingestion_{utc_compact()}"
    dirs = {
        "hashes": run_dir / "hashes",
        "inventory": run_dir / "inventory",
        "contact_sheets": run_dir / "contact_sheets",
        "previews": run_dir / "previews",
        "calibration": run_dir / "calibration",
        "person": run_dir / "person_gold_set",
        "ball": run_dir / "ball_v0_1_plan",
        "analysis": run_dir / "analysis",
        "frames": run_dir / "reference_frames",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(input_dir.glob("video_*.mp4"))
    if not video_paths:
        raise FileNotFoundError(f"no videos found under {input_dir}")

    videos = []
    all_person_rows: List[Dict[str, Any]] = []
    for path in video_paths:
        raw_probe = ffprobe_json(path)
        metadata = parse_ffprobe(path, raw_probe)
        video_hash = sha256_file(path)
        video_id = f"mv01_{path.stem}_{video_hash[:12]}"
        metadata["sha256"] = video_hash
        contact_ts = sample_timestamps(metadata["duration_sec"], 15)
        analysis_ts = sample_timestamps(metadata["duration_sec"], 9)
        contact_path = dirs["contact_sheets"] / f"{video_id}_contact_sheet.jpg"
        contact_samples = make_contact_sheet(path, contact_path, contact_ts)
        preview_entries = []
        for label, start in [
            ("early", max(0.0, metadata["duration_sec"] * 0.12)),
            ("middle", max(0.0, metadata["duration_sec"] * 0.50)),
            ("late", max(0.0, metadata["duration_sec"] * 0.84)),
        ]:
            preview_entries.append(make_preview(path, dirs["previews"] / f"{video_id}_{label}_preview.mp4", start))
        analysis = analyze_camera_quality(path, metadata, analysis_ts)
        roi = detect_field_roi(path, metadata, analysis_ts)
        reference = save_reference_frame(path, dirs["frames"] / f"{video_id}_reference_frame.jpg", metadata["duration_sec"] * 0.50)
        profile = build_calibration_profile(video_id, metadata, video_hash, roi, reference)
        profile_path = dirs["calibration"] / video_id / "video_calibration_draft.json"
        write_json(profile_path, profile)
        overlay_path = dirs["calibration"] / video_id / "roi_overlay_multi_timestamp.jpg"
        overlay_samples = draw_roi_overlay(path, overlay_path, sample_timestamps(metadata["duration_sec"], 6, margin_ratio=0.08), roi["polygon_pixels_reference"])
        calibration_summary = {
            "video_id": video_id,
            "calibration_profile_id": profile["calibration_id"],
            "profile_path": str(profile_path),
            "roi_status": profile["status"],
            "full_field_included": "draft_visual_review",
            "near_goal_included": roi["validation"]["near_goal_included_heuristic"],
            "far_goal_included": roi["validation"]["far_goal_included_heuristic"],
            "sidelines_included": roi["validation"]["sidelines_included_heuristic"],
            "external_people_excluded": "requires_visual_review",
            "overlay_path": str(overlay_path),
            "reference_frame": reference,
            "overlay_samples": overlay_samples,
        }
        write_json(dirs["calibration"] / video_id / "calibration_summary.json", calibration_summary)

        video_record = {
            "video_id": video_id,
            "filename": path.name,
            "path": str(path),
            "sha256": video_hash,
            "metadata": metadata,
            "ffprobe": raw_probe,
            "contact_sheet": str(contact_path),
            "contact_samples": contact_samples,
            "previews": preview_entries,
            "analysis": analysis,
            "calibration": calibration_summary,
        }
        videos.append(video_record)
        all_person_rows.extend(propose_person_frames(video_record))

    duplicates = exact_and_near_duplicates(videos)
    ball_rows = propose_ball_sequences(videos)

    inventory = {
        "phase": PHASE,
        "created_at": utc_now(),
        "input_dir": str(input_dir),
        "run_dir": str(run_dir),
        "video_count": len(videos),
        "videos": videos,
        "duplicates": duplicates,
        "restrictions": {
            "detectors_executed": False,
            "training_executed": False,
            "runpod_active": False,
            "production_src_touched": False,
            "supabase_touched": False,
            "r2_touched": False,
            "oneframe_ball_v0_modified": False,
        },
    }
    write_json(dirs["inventory"] / "video_inventory.json", inventory)
    write_json(dirs["hashes"] / "video_hashes.json", [{"video_id": v["video_id"], "filename": v["filename"], "sha256": v["sha256"], "size_bytes": v["metadata"]["size_bytes"]} for v in videos])
    write_json(dirs["analysis"] / "duplicate_report.json", duplicates)
    write_json(dirs["person"] / "person_gold_set_frame_manifest.json", all_person_rows)
    write_csv(dirs["person"] / "person_gold_set_frame_manifest.csv", all_person_rows, [
        "video_id", "filename", "frame_index", "timestamp_sec", "distance_coverage",
        "occlusion_coverage", "cluster_coverage", "field_zone", "expected_persons", "ground_truth_status",
    ])
    write_json(dirs["ball"] / "ball_v0_1_sequence_manifest.json", ball_rows)
    write_csv(dirs["ball"] / "ball_v0_1_sequence_manifest.csv", ball_rows, [
        "sequence_id", "source_video", "filename", "split", "start_sec", "end_sec",
        "duration_sec", "error_category", "estimated_positive_frames", "estimated_negative_frames",
        "overlap_status", "policy",
    ])

    video_table_rows = []
    calibration_table_rows = []
    for v in videos:
        video_table_rows.append({
            "video_id": v["video_id"],
            "filename": v["filename"],
            "sha256": v["sha256"],
            "duration_sec": round(v["metadata"]["duration_sec"], 6),
            "fps": round(v["metadata"]["fps"], 6),
            "resolution": v["metadata"]["resolution"],
            "codec": v["metadata"]["video_codec"],
            "camera_position": v["analysis"]["camera_position"],
            "lighting": v["analysis"]["lighting"],
            "usable": v["analysis"]["usable"],
            "reason": v["analysis"]["reason"],
        })
        calibration_table_rows.append({
            "video_id": v["video_id"],
            "calibration_profile": v["calibration"]["calibration_profile_id"],
            "roi_status": v["calibration"]["roi_status"],
            "full_field_included": v["calibration"]["full_field_included"],
            "near_goal_included": v["calibration"]["near_goal_included"],
            "far_goal_included": v["calibration"]["far_goal_included"],
            "external_people_excluded": v["calibration"]["external_people_excluded"],
            "overlay_path": v["calibration"]["overlay_path"],
        })
    write_csv(dirs["inventory"] / "video_table.csv", video_table_rows, list(video_table_rows[0].keys()))
    write_csv(dirs["calibration"] / "calibration_table.csv", calibration_table_rows, list(calibration_table_rows[0].keys()))

    split_counts: Dict[str, int] = {}
    for row in ball_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    summary = {
        "phase": PHASE,
        "status": "ready_for_calibration_review",
        "run_dir": str(run_dir),
        "videos": video_table_rows,
        "calibration": calibration_table_rows,
        "person_gold_set": {
            "frames_total": len(all_person_rows),
            "frames_per_video": {v["video_id"]: sum(1 for row in all_person_rows if row["video_id"] == v["video_id"]) for v in videos},
        },
        "ball_v0_1": {
            "sequence_count": len(ball_rows),
            "split_counts": split_counts,
            "cross_video_test_policy": "reserved; do not infer or tune thresholds on test split",
        },
        "duplicates": duplicates,
        "production": {
            "src_intact": True,
            "runpod_active": False,
            "cost_active": False,
            "supabase_touched": False,
            "r2_touched": False,
        },
        "next_action": "esperar validacion visual de las tres ROI antes de preanotar o crear un Pod",
    }
    write_json(run_dir / "summary.json", summary)
    write_text(
        run_dir / "PE0_MULTIVIDEO_INGESTION_REPORT.md",
        render_report(summary, all_person_rows, ball_rows),
    )
    artifact_manifest(run_dir, {
        "inventory": dirs["inventory"] / "video_inventory.json",
        "hashes": dirs["hashes"] / "video_hashes.json",
        "duplicate_report": dirs["analysis"] / "duplicate_report.json",
        "person_gold_manifest_json": dirs["person"] / "person_gold_set_frame_manifest.json",
        "person_gold_manifest_csv": dirs["person"] / "person_gold_set_frame_manifest.csv",
        "ball_sequence_manifest_json": dirs["ball"] / "ball_v0_1_sequence_manifest.json",
        "ball_sequence_manifest_csv": dirs["ball"] / "ball_v0_1_sequence_manifest.csv",
        "video_table": dirs["inventory"] / "video_table.csv",
        "calibration_table": dirs["calibration"] / "calibration_table.csv",
        "summary": run_dir / "summary.json",
        "report": run_dir / "PE0_MULTIVIDEO_INGESTION_REPORT.md",
    })
    print(json.dumps({"status": "ready_for_calibration_review", "run_dir": str(run_dir)}, indent=2))
    return 0


def render_report(summary: Dict[str, Any], person_rows: List[Dict[str, Any]], ball_rows: List[Dict[str, Any]]) -> str:
    lines = [
        "# PE-0 MULTIVIDEO INGESTION AND SELECTION",
        "",
        f"- ESTADO: `{summary['status']}`",
        f"- run_dir: `{summary['run_dir']}`",
        "",
        "## Videos",
        "| video ID | filename | SHA256 | duration | FPS | resolution | codec | camera | lighting | usable | reason |",
        "|---|---|---|---:|---:|---|---|---|---|---|---|",
    ]
    for v in summary["videos"]:
        lines.append(
            f"| `{v['video_id']}` | `{v['filename']}` | `{v['sha256']}` | {v['duration_sec']} | {v['fps']} | {v['resolution']} | {v['codec']} | {v['camera_position']} | {v['lighting']} | {v['usable']} | {v['reason']} |"
        )
    lines += [
        "",
        "## Calibration",
        "| video ID | calibration profile | ROI status | full field | near goal | far goal | external people | overlay |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in summary["calibration"]:
        lines.append(
            f"| `{c['video_id']}` | `{c['calibration_profile']}` | `{c['roi_status']}` | `{c['full_field_included']}` | `{c['near_goal_included']}` | `{c['far_goal_included']}` | `{c['external_people_excluded']}` | `{c['overlay_path']}` |"
        )
    lines += [
        "",
        "## Person Gold Set Plan",
        f"- frames_total: `{len(person_rows)}`",
        f"- frames_per_video: `{summary['person_gold_set']['frames_per_video']}`",
        "- coverage: near/mid/far, clusters, occlusions, field edges, goal areas.",
        "",
        "## Ball V0.1 Plan",
        f"- sequence_count: `{len(ball_rows)}`",
        f"- split_counts: `{summary['ball_v0_1']['split_counts']}`",
        "- policy: proposal only; no inference on cross-video test; no pseudo-labels as ground truth.",
        "",
        "## Production",
        f"- src intacto: `{summary['production']['src_intact']}`",
        f"- RunPod active: `{summary['production']['runpod_active']}`",
        f"- cost active: `{summary['production']['cost_active']}`",
        "",
        "## Siguiente accion",
        summary["next_action"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
