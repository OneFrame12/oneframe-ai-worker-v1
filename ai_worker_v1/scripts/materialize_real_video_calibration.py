#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


AI_WORKER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = AI_WORKER_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from video_calibration.coordinates import (  # noqa: E402
    coerce_points,
    denormalize_polygon,
    distance_point_to_segment,
    normalize_polygon,
    polygon_area,
)
from video_calibration.legacy_roi_adapter import (  # noqa: E402
    build_legacy_import_report,
    build_profile_from_legacy_roi,
)
from video_calibration.profile_io import profile_hash, save_profile  # noqa: E402
from video_calibration.qa import update_profile_qa  # noqa: E402
from video_calibration.renderer import render_calibration_overlay  # noqa: E402
from video_calibration.schema import SCHEMA_VERSION  # noqa: E402


PHASE = "PE-0A2"
PRODUCED_BY = "ai_worker_v1/scripts/materialize_real_video_calibration.py"


@dataclass
class VideoMetadataProbe:
    path: str
    exists: bool
    sha256: Optional[str]
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float
    aspect_ratio: float
    codec: Optional[str]
    rotation: Optional[float]
    probe_method: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "exists": self.exists,
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "aspect_ratio": self.aspect_ratio,
            "codec": self.codec,
            "rotation": self.rotation,
            "probe_method": self.probe_method,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Dict[str, Any] | List[Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return destination


def write_text(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return destination


def _try_ffprobe_rotation(video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream_tags=rotate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    value = result.stdout.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _fourcc_to_str(raw_value: float) -> Optional[str]:
    try:
        value = int(raw_value)
    except Exception:
        return None
    chars = [chr((value >> 8 * idx) & 0xFF) for idx in range(4)]
    text = "".join(chars).strip("\x00 ")
    return text or None


def read_video_metadata(video_path: str | Path) -> VideoMetadataProbe:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {path}")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to read video metadata") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    codec = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()

    duration = frame_count / fps if fps > 0 else 0.0
    aspect_ratio = width / float(height) if width and height else 0.0
    return VideoMetadataProbe(
        path=str(path),
        exists=True,
        sha256=file_sha256(path),
        width=width,
        height=height,
        fps=round(fps, 6),
        frame_count=frame_count,
        duration_sec=round(duration, 6),
        aspect_ratio=round(aspect_ratio, 8),
        codec=codec,
        rotation=_try_ffprobe_rotation(path),
        probe_method="opencv_ffprobe_rotation_optional",
    )


def extract_reference_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    frame_index: int,
) -> Path:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is required to extract the reference frame") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not extract reference frame {frame_index}")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise RuntimeError(f"Could not write reference frame: {destination}")
    return destination


def _first_present(mapping: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_roi_payload(path: str | Path) -> Dict[str, Any]:
    payload_path = Path(path)
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"roi_points": data}
    if not isinstance(data, dict):
        raise ValueError("ROI payload must be a JSON object or a list of points")
    nested_input = data.get("input") if isinstance(data.get("input"), dict) else {}
    merged = {**nested_input, **data}
    roi_points = _first_present(merged, ["roi_points", "roi", "polygon", "playable_roi"])
    if not roi_points:
        raise ValueError("ROI payload missing roi_points")
    return {
        "source_path": str(payload_path),
        "raw": data,
        "roi_points": roi_points,
        "danger_zone": _first_present(merged, ["danger_zone", "legacy_danger_zone"]),
        "match_id": _first_present(merged, ["match_id", "match_uuid"]),
        "reference_frame_index": _first_present(merged, ["reference_frame_index", "frame_index"]),
        "reference_timestamp_sec": _first_present(merged, ["reference_timestamp_sec", "timestamp_sec"]),
        "reference_width": _first_present(merged, ["reference_width", "frame_width", "video_width"]),
        "reference_height": _first_present(merged, ["reference_height", "frame_height", "video_height"]),
        "canvas_width": _first_present(merged, ["canvas_width", "display_width", "rendered_width"]),
        "canvas_height": _first_present(merged, ["canvas_height", "display_height", "rendered_height"]),
        "padding_x": float(_first_present(merged, ["padding_x", "pad_x"]) or 0.0),
        "padding_y": float(_first_present(merged, ["padding_y", "pad_y"]) or 0.0),
    }


def _points_max_error(first: List[Tuple[float, float]], second: List[Tuple[float, float]]) -> float:
    if len(first) != len(second):
        return float("inf")
    return max((math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(first, second)), default=0.0)


def _scale_points(
    points: Iterable,
    *,
    source_width: float,
    source_height: float,
    target_width: float,
    target_height: float,
    padding_x: float = 0.0,
    padding_y: float = 0.0,
) -> List[List[float]]:
    coerced = coerce_points(points)
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_width and source_height must be known for coordinate scaling")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return [[round((x - padding_x) * scale_x, 4), round((y - padding_y) * scale_y, 4)] for x, y in coerced]


def resolve_coordinate_system(payload: Dict[str, Any], video: VideoMetadataProbe) -> Dict[str, Any]:
    roi_points = coerce_points(payload["roi_points"])
    danger_zone = coerce_points(payload["danger_zone"]) if payload.get("danger_zone") else []
    ref_width = payload.get("reference_width")
    ref_height = payload.get("reference_height")
    canvas_width = payload.get("canvas_width")
    canvas_height = payload.get("canvas_height")
    padding_x = float(payload.get("padding_x") or 0.0)
    padding_y = float(payload.get("padding_y") or 0.0)

    transform = {
        "status": "direct_frame_pixels",
        "canvas_width": None,
        "canvas_height": None,
        "video_width": video.width,
        "video_height": video.height,
        "scale_x": 1.0,
        "scale_y": 1.0,
        "padding_x": 0.0,
        "padding_y": 0.0,
        "round_trip_max_error_px": 0.0,
        "notes": [],
    }

    if ref_width and ref_height:
        ref_width_f = float(ref_width)
        ref_height_f = float(ref_height)
        if int(round(ref_width_f)) == video.width and int(round(ref_height_f)) == video.height:
            transformed_roi = [[float(x), float(y)] for x, y in roi_points]
            transformed_danger = [[float(x), float(y)] for x, y in danger_zone]
            transform["notes"].append("payload reference resolution matches video resolution")
        else:
            raise ValueError(
                "ROI reference resolution differs from video resolution and no explicit canvas transform is allowed"
            )
    elif canvas_width and canvas_height:
        canvas_width_f = float(canvas_width)
        canvas_height_f = float(canvas_height)
        transformed_roi = _scale_points(
            roi_points,
            source_width=canvas_width_f,
            source_height=canvas_height_f,
            target_width=video.width,
            target_height=video.height,
            padding_x=padding_x,
            padding_y=padding_y,
        )
        transformed_danger = _scale_points(
            danger_zone,
            source_width=canvas_width_f,
            source_height=canvas_height_f,
            target_width=video.width,
            target_height=video.height,
            padding_x=padding_x,
            padding_y=padding_y,
        ) if danger_zone else []
        restored_roi = _scale_points(
            transformed_roi,
            source_width=video.width,
            source_height=video.height,
            target_width=canvas_width_f,
            target_height=canvas_height_f,
            padding_x=0.0,
            padding_y=0.0,
        )
        round_trip_error = _points_max_error(roi_points, [tuple(point) for point in restored_roi])
        transform.update(
            {
                "status": "canvas_to_video_scaled",
                "canvas_width": canvas_width_f,
                "canvas_height": canvas_height_f,
                "scale_x": round(video.width / canvas_width_f, 8),
                "scale_y": round(video.height / canvas_height_f, 8),
                "padding_x": padding_x,
                "padding_y": padding_y,
                "round_trip_max_error_px": round(round_trip_error, 6),
            }
        )
    else:
        transformed_roi = [[float(x), float(y)] for x, y in roi_points]
        transformed_danger = [[float(x), float(y)] for x, y in danger_zone]
        transform["notes"].append("no canvas metadata found; treating roi_points as original frame pixels")

    return {
        "roi_points_video_pixels": transformed_roi,
        "danger_zone_video_pixels": transformed_danger,
        "transform": transform,
    }


def polygon_bbox(points: List[Tuple[float, float]]) -> Dict[str, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "min_x": round(min(xs), 4),
        "min_y": round(min(ys), 4),
        "max_x": round(max(xs), 4),
        "max_y": round(max(ys), 4),
        "width": round(max(xs) - min(xs), 4),
        "height": round(max(ys) - min(ys), 4),
    }


def min_point_distance(points: List[Tuple[float, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    best = min(
        math.hypot(a[0] - b[0], a[1] - b[1])
        for idx, a in enumerate(points)
        for b in points[idx + 1 :]
    )
    return round(best, 4)


def min_distance_to_frame_edge(points: List[Tuple[float, float]], width: int, height: int) -> Optional[float]:
    if not points:
        return None
    distances = []
    for x, y in points:
        distances.extend([x, y, width - x, height - y])
    return round(min(distances), 4)


def max_round_trip_error(points: List[Tuple[float, float]], width: int, height: int) -> float:
    normalized = normalize_polygon(points, width, height)
    restored = denormalize_polygon(normalized, width, height)
    return round(_points_max_error(points, [tuple(point) for point in restored]), 6)


def build_qa_payload(
    *,
    profile,
    video_metadata: VideoMetadataProbe,
    transform: Dict[str, Any],
    artifact_hashes: Dict[str, str],
) -> Dict[str, Any]:
    points = coerce_points(profile.detection_roi.polygon_pixels_reference)
    area_px = polygon_area(points)
    update_profile_qa(profile)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "calibration_id": profile.calibration_id,
        "profile_hash": profile_hash(profile),
        "qa": profile.qa.__dict__,
        "roi_validation": profile.detection_roi.validation,
        "geometric_metrics": {
            "point_count": len(points),
            "polygon_area_px": round(area_px, 4),
            "polygon_area_ratio": profile.detection_roi.validation.get("area_ratio"),
            "bounding_rectangle": polygon_bbox(points) if points else None,
            "min_point_distance_px": min_point_distance(points),
            "min_distance_to_frame_edge_px": min_distance_to_frame_edge(points, video_metadata.width, video_metadata.height),
            "orientation": profile.detection_roi.validation.get("orientation"),
            "round_trip_pixel_normalized_pixel_max_error_px": max_round_trip_error(points, video_metadata.width, video_metadata.height),
        },
        "coordinate_transform": transform,
        "frame": {
            "reference_frame_index": profile.video.reference_frame_index,
            "reference_timestamp_sec": profile.video.reference_timestamp_sec,
            "width": video_metadata.width,
            "height": video_metadata.height,
            "frame_hash": profile.video.frame_hash,
        },
        "homography": profile.homography.__dict__,
        "artifact_hashes": artifact_hashes,
        "review_complete": False,
    }


def build_qa_markdown(qa_payload: Dict[str, Any]) -> str:
    metrics = qa_payload["geometric_metrics"]
    transform = qa_payload["coordinate_transform"]
    homography = qa_payload["homography"]
    return "\n".join(
        [
            "# PE-0A2 Video Calibration QA",
            "",
            f"- calibration_id: `{qa_payload['calibration_id']}`",
            f"- schema_version: `{qa_payload['schema_version']}`",
            f"- roi_valid: `{qa_payload['qa']['roi_valid']}`",
            f"- schema_valid: `{qa_payload['qa']['schema_valid']}`",
            f"- review_complete: `{qa_payload['review_complete']}`",
            "",
            "## Geometry",
            f"- point_count: `{metrics['point_count']}`",
            f"- polygon_area_px: `{metrics['polygon_area_px']}`",
            f"- polygon_area_ratio: `{metrics['polygon_area_ratio']}`",
            f"- bounding_rectangle: `{metrics['bounding_rectangle']}`",
            f"- min_point_distance_px: `{metrics['min_point_distance_px']}`",
            f"- min_distance_to_frame_edge_px: `{metrics['min_distance_to_frame_edge_px']}`",
            f"- orientation: `{metrics['orientation']}`",
            f"- round_trip_error_px: `{metrics['round_trip_pixel_normalized_pixel_max_error_px']}`",
            "",
            "## Coordinate Transform",
            f"- status: `{transform['status']}`",
            f"- canvas: `{transform.get('canvas_width')}x{transform.get('canvas_height')}`",
            f"- video: `{transform.get('video_width')}x{transform.get('video_height')}`",
            f"- scale: `{transform.get('scale_x')}, {transform.get('scale_y')}`",
            f"- padding: `{transform.get('padding_x')}, {transform.get('padding_y')}`",
            "",
            "## Homography",
            f"- status: `{homography['status']}`",
            f"- failure_reasons: `{homography['failure_reasons']}`",
            "",
            "## Warnings",
            *[f"- `{warning}`" for warning in qa_payload["qa"].get("warnings", [])],
            "",
        ]
    )


def build_human_review_checklist() -> str:
    questions = [
        "El ROI contiene toda el area donde pueden aparecer jugadores?",
        "Excluye correctamente publico y zonas externas?",
        "Corta jugadores que pisan los bordes?",
        "Los puntos coinciden con los clicks de la herramienta?",
        "Existe desplazamiento por resize o padding?",
        "El frame corresponde al video correcto?",
        "La zona detras del arco fue incluida intencionalmente?",
        "La danger zone legacy coincide con lo marcado originalmente?",
        "Debe corregirse algun vertice?",
        "Se aprueba el profile para baseline de deteccion?",
    ]
    lines = ["# Human Review Checklist", ""]
    for idx, question in enumerate(questions, start=1):
        lines.append(f"{idx}. [ ] {question}")
    lines.extend(
        [
            "",
            "Decision requerida: approve | correct_points | wrong_frame | wrong_video | reject",
            "",
        ]
    )
    return "\n".join(lines)


def artifact_record(
    path: Path,
    *,
    artifact_type: str,
    video_hash: Optional[str],
    calibration_id: Optional[str],
) -> Dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "artifact_type": artifact_type,
        "produced_by": PRODUCED_BY,
        "schema_version": SCHEMA_VERSION,
        "video_hash": video_hash,
        "calibration_id": calibration_id,
        "created_at": utc_now_iso(),
    }


def materialize_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    video_metadata = read_video_metadata(args.video)
    roi_payload = load_roi_payload(args.roi_payload)

    if args.reference_frame_index is not None:
        frame_index = int(args.reference_frame_index)
        timestamp_sec = frame_index / video_metadata.fps if video_metadata.fps > 0 else 0.0
        timestamp_source = "calculated_from_frame_index_and_fps"
    else:
        timestamp_sec = float(args.reference_timestamp_sec)
        frame_index = int(round(timestamp_sec * video_metadata.fps)) if video_metadata.fps > 0 else 0
        timestamp_source = "calculated_from_timestamp_and_fps"

    if frame_index < 0 or (video_metadata.frame_count and frame_index >= video_metadata.frame_count):
        raise ValueError(f"reference frame index {frame_index} outside video frame count {video_metadata.frame_count}")

    transform_result = resolve_coordinate_system(roi_payload, video_metadata)
    roi_points = transform_result["roi_points_video_pixels"]
    danger_zone = transform_result["danger_zone_video_pixels"]
    transform = transform_result["transform"]

    calibration_input = {
        "phase": PHASE,
        "video": str(args.video),
        "roi_payload": str(args.roi_payload),
        "match_id": args.match_id or roi_payload.get("match_id"),
        "reference_frame_index": frame_index,
        "reference_timestamp_sec": round(timestamp_sec, 6),
        "timestamp_source": timestamp_source,
        "roi_points_original_payload": roi_payload["roi_points"],
        "roi_points_video_pixels": roi_points,
        "danger_zone_original_payload": roi_payload.get("danger_zone") or [],
        "danger_zone_video_pixels": danger_zone,
        "coordinate_transform": transform,
    }
    write_json(run_dir / "calibration_input.json", calibration_input)

    video_metadata_path = write_json(run_dir / "source_video_metadata.json", video_metadata.to_dict())
    frame_path = extract_reference_frame(args.video, run_dir / "source_reference_frame.png", frame_index=frame_index)
    frame_hash = file_sha256(frame_path)

    profile = build_profile_from_legacy_roi(
        roi_points=roi_points,
        frame_width=video_metadata.width,
        frame_height=video_metadata.height,
        match_id=args.match_id or roi_payload.get("match_id"),
        video_hash=video_metadata.sha256 or "",
        reference_frame_index=frame_index,
        reference_timestamp_sec=round(timestamp_sec, 6),
        danger_zone=danger_zone,
        fps=video_metadata.fps,
        expected_aspect_ratio=video_metadata.aspect_ratio,
    )
    profile.status = "draft"
    profile.detection_roi.reviewed = False
    profile.qa.review_complete = False
    profile.video.frame_hash = frame_hash
    update_profile_qa(profile)

    profile_path = save_profile(run_dir / "video_calibration.json", profile)

    legacy_report = build_legacy_import_report(profile)
    legacy_report.update(
        {
            "danger_zone_present": bool(danger_zone),
            "danger_zone_pixels_reference": danger_zone,
            "danger_zone_normalized": normalize_polygon(danger_zone, video_metadata.width, video_metadata.height) if danger_zone else [],
            "danger_zone_treatment": "preserved for diagnostic visualization only; non-metric; not detection ROI",
        }
    )
    legacy_path = write_json(run_dir / "legacy_import_report.json", legacy_report)

    overlay_path = render_calibration_overlay(
        profile,
        run_dir / "video_calibration_overlay.png",
        image_path=frame_path,
        danger_zone_pixels=danger_zone,
    )
    overlay_clean_path = render_calibration_overlay(
        profile,
        run_dir / "video_calibration_overlay_clean.png",
        image_path=frame_path,
        danger_zone_pixels=None,
        clean=True,
    )

    preliminary_hashes = {
        "source_video_metadata.json": file_sha256(video_metadata_path),
        "source_reference_frame.png": frame_hash,
        "calibration_input.json": file_sha256(run_dir / "calibration_input.json"),
        "video_calibration.json": file_sha256(profile_path),
        "legacy_import_report.json": file_sha256(legacy_path),
        "video_calibration_overlay.png": file_sha256(overlay_path),
        "video_calibration_overlay_clean.png": file_sha256(overlay_clean_path),
    }
    qa_payload = build_qa_payload(
        profile=profile,
        video_metadata=video_metadata,
        transform=transform,
        artifact_hashes=preliminary_hashes,
    )
    qa_path = write_json(run_dir / "video_calibration_qa.json", qa_payload)
    qa_md_path = write_text(run_dir / "video_calibration_qa.md", build_qa_markdown(qa_payload))
    checklist_path = write_text(run_dir / "human_review_checklist.md", build_human_review_checklist())

    artifacts = [
        artifact_record(run_dir / "calibration_input.json", artifact_type="calibration_input", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(video_metadata_path, artifact_type="source_video_metadata", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(frame_path, artifact_type="source_reference_frame", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(profile_path, artifact_type="video_calibration_profile", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(qa_path, artifact_type="video_calibration_qa_json", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(qa_md_path, artifact_type="video_calibration_qa_markdown", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(legacy_path, artifact_type="legacy_import_report", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(overlay_path, artifact_type="video_calibration_overlay", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(overlay_clean_path, artifact_type="video_calibration_overlay_clean", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
        artifact_record(checklist_path, artifact_type="human_review_checklist", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id),
    ]
    artifact_manifest_path = write_json(run_dir / "artifact_manifest.json", {"artifacts": artifacts})
    artifacts.append(artifact_record(artifact_manifest_path, artifact_type="artifact_manifest", video_hash=video_metadata.sha256, calibration_id=profile.calibration_id))

    run_manifest = {
        "run_id": args.run_id,
        "phase": PHASE,
        "status": "completed",
        "video_hash": video_metadata.sha256,
        "match_id": args.match_id or roi_payload.get("match_id"),
        "calibration_id": profile.calibration_id,
        "reference_frame_index": frame_index,
        "reference_timestamp_sec": round(timestamp_sec, 6),
        "code_version": "local_workspace",
        "git_commit": None,
        "artifacts": artifacts,
        "warnings": profile.qa.warnings,
        "errors": profile.qa.errors,
        "human_review_status": "pending",
    }
    run_manifest_path = write_json(run_dir / "run_manifest.json", run_manifest)

    report = "\n".join(
        [
            "# PE-0A2 Run Report",
            "",
            f"- status: completed",
            f"- run_id: `{args.run_id}`",
            f"- match_id: `{args.match_id or roi_payload.get('match_id')}`",
            f"- calibration_id: `{profile.calibration_id}`",
            f"- video_hash: `{video_metadata.sha256}`",
            f"- frame_index: `{frame_index}`",
            f"- timestamp_sec: `{round(timestamp_sec, 6)}`",
            f"- coordinate_transform: `{transform['status']}`",
            f"- homography_status: `{profile.homography.status}`",
            f"- human_review_status: `pending`",
            "",
            "No production integration was performed.",
            "",
        ]
    )
    report_path = write_text(run_dir / "PE0A2_RUN_REPORT.md", report)

    for path, artifact_type in [
        (run_manifest_path, "run_manifest"),
        (report_path, "pe0a2_run_report"),
    ]:
        artifacts.append(artifact_record(path, artifact_type=artifact_type, video_hash=video_metadata.sha256, calibration_id=profile.calibration_id))
    write_json(artifact_manifest_path, {"artifacts": artifacts})

    return {
        "run_dir": str(run_dir),
        "run_manifest": str(run_manifest_path),
        "artifact_manifest": str(artifact_manifest_path),
        "profile": str(profile_path),
        "qa": str(qa_path),
        "overlay": str(overlay_path),
        "overlay_clean": str(overlay_clean_path),
        "human_review_checklist": str(checklist_path),
        "calibration_id": profile.calibration_id,
        "video_hash": video_metadata.sha256,
        "frame_hash": frame_hash,
        "status": "completed",
    }


def materialize_from_capture_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.output_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(Path(args.capture_bundle).read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "oneframe.calibration_capture_bundle.v1":
        raise ValueError("capture bundle schema_version mismatch")

    bundle_copy_path = run_dir / "calibration_capture_bundle.json"
    shutil.copyfile(args.capture_bundle, bundle_copy_path)

    video = bundle.get("video", {})
    calibration_input = bundle.get("calibration_input", {})
    reference = bundle.get("reference_frame", {})
    roi_points = calibration_input.get("roi_points") or []
    danger_zone = calibration_input.get("danger_zone") or []
    context = calibration_input.get("calibration_context") or {}

    reference_frame_source = Path(args.reference_frame or reference.get("artifact_path") or "")
    if not reference_frame_source.exists():
        raise FileNotFoundError("reference frame is required when materializing from a capture bundle")
    expected_frame_hash = reference.get("sha256")
    actual_frame_hash = file_sha256(reference_frame_source)
    if expected_frame_hash and expected_frame_hash != actual_frame_hash:
        raise ValueError("reference frame sha256 does not match capture bundle")

    calibration_input_path = write_json(run_dir / "calibration_input.json", {
        "phase": PHASE,
        "source": "CalibrationCaptureBundle",
        "capture_bundle": str(args.capture_bundle),
        "roi_points_video_pixels": roi_points,
        "danger_zone_video_pixels": danger_zone,
        "calibration_context": context,
    })
    video_metadata_path = write_json(run_dir / "source_video_metadata.json", video)
    frame_path = run_dir / "source_reference_frame.png"
    shutil.copyfile(reference_frame_source, frame_path)

    frame_index = reference.get("frame_index")
    timestamp_sec = reference.get("timestamp_sec")
    frame_width = int(video.get("width") or reference.get("width") or 0)
    frame_height = int(video.get("height") or reference.get("height") or 0)
    profile = build_profile_from_legacy_roi(
        roi_points=roi_points,
        frame_width=frame_width,
        frame_height=frame_height,
        match_id=bundle.get("job", {}).get("match_id"),
        video_hash=video.get("video_hash") or "",
        reference_frame_index=int(frame_index or 0),
        reference_timestamp_sec=float(timestamp_sec or 0.0),
        danger_zone=danger_zone,
        fps=video.get("fps"),
        expected_aspect_ratio=(frame_width / float(frame_height) if frame_width and frame_height else None),
    )
    profile.status = "draft"
    profile.detection_roi.reviewed = False
    profile.qa.review_complete = False
    profile.video.frame_hash = actual_frame_hash
    update_profile_qa(profile)
    profile_path = save_profile(run_dir / "video_calibration.json", profile)

    legacy_report = build_legacy_import_report(profile)
    legacy_report.update(
        {
            "capture_bundle_id": bundle.get("capture_id"),
            "danger_zone_present": bool(danger_zone),
            "danger_zone_pixels_reference": danger_zone,
            "danger_zone_normalized": normalize_polygon(danger_zone, frame_width, frame_height) if danger_zone else [],
            "danger_zone_treatment": "preserved for diagnostic visualization only; non-metric; not detection ROI",
        }
    )
    legacy_path = write_json(run_dir / "legacy_import_report.json", legacy_report)
    overlay_path = render_calibration_overlay(
        profile,
        run_dir / "video_calibration_overlay.png",
        image_path=frame_path,
        danger_zone_pixels=danger_zone,
    )
    overlay_clean_path = render_calibration_overlay(
        profile,
        run_dir / "video_calibration_overlay_clean.png",
        image_path=frame_path,
        clean=True,
    )

    preliminary_hashes = {
        "source_video_metadata.json": file_sha256(video_metadata_path),
        "source_reference_frame.png": actual_frame_hash,
        "calibration_input.json": file_sha256(calibration_input_path),
        "video_calibration.json": file_sha256(profile_path),
        "legacy_import_report.json": file_sha256(legacy_path),
        "video_calibration_overlay.png": file_sha256(overlay_path),
        "video_calibration_overlay_clean.png": file_sha256(overlay_clean_path),
    }
    probe = VideoMetadataProbe(
        path=str(video.get("source_reference_sanitized") or ""),
        exists=False,
        sha256=video.get("video_hash"),
        width=frame_width,
        height=frame_height,
        fps=float(video.get("fps") or 0.0),
        frame_count=int(video.get("frame_count") or 0),
        duration_sec=float(video.get("duration_sec") or 0.0),
        aspect_ratio=round(frame_width / float(frame_height), 8) if frame_width and frame_height else 0.0,
        codec=video.get("codec"),
        rotation=None,
        probe_method="capture_bundle",
    )
    qa_payload = build_qa_payload(
        profile=profile,
        video_metadata=probe,
        transform={
            "status": "from_capture_bundle",
            "canvas_width": context.get("display_width"),
            "canvas_height": context.get("display_height"),
            "video_width": frame_width,
            "video_height": frame_height,
            "scale_x": context.get("display_transform", {}).get("scale_x", 1.0),
            "scale_y": context.get("display_transform", {}).get("scale_y", 1.0),
            "padding_x": context.get("display_transform", {}).get("offset_x", 0.0),
            "padding_y": context.get("display_transform", {}).get("offset_y", 0.0),
            "round_trip_max_error_px": 0.0,
            "notes": ["materialized from CalibrationCaptureBundle"],
        },
        artifact_hashes=preliminary_hashes,
    )
    qa_path = write_json(run_dir / "video_calibration_qa.json", qa_payload)
    qa_md_path = write_text(run_dir / "video_calibration_qa.md", build_qa_markdown(qa_payload))
    checklist_path = write_text(run_dir / "human_review_checklist.md", build_human_review_checklist())

    artifacts = [
        artifact_record(bundle_copy_path, artifact_type="calibration_capture_bundle", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(calibration_input_path, artifact_type="calibration_input", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(video_metadata_path, artifact_type="source_video_metadata", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(frame_path, artifact_type="source_reference_frame", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(profile_path, artifact_type="video_calibration_profile", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(qa_path, artifact_type="video_calibration_qa_json", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(qa_md_path, artifact_type="video_calibration_qa_markdown", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(legacy_path, artifact_type="legacy_import_report", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(overlay_path, artifact_type="video_calibration_overlay", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(overlay_clean_path, artifact_type="video_calibration_overlay_clean", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        artifact_record(checklist_path, artifact_type="human_review_checklist", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
    ]
    artifact_manifest_path = write_json(run_dir / "artifact_manifest.json", {"artifacts": artifacts})
    artifacts.append(artifact_record(artifact_manifest_path, artifact_type="artifact_manifest", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id))

    run_manifest = {
        "run_id": args.run_id,
        "phase": PHASE,
        "status": "completed",
        "video_hash": video.get("video_hash"),
        "match_id": bundle.get("job", {}).get("match_id"),
        "calibration_id": profile.calibration_id,
        "reference_frame_index": frame_index,
        "reference_timestamp_sec": timestamp_sec,
        "code_version": "local_workspace",
        "git_commit": None,
        "artifacts": artifacts,
        "warnings": profile.qa.warnings,
        "errors": profile.qa.errors,
        "human_review_status": "pending",
    }
    run_manifest_path = write_json(run_dir / "run_manifest.json", run_manifest)
    report_path = write_text(
        run_dir / "PE0A2C_RUN_REPORT.md",
        "\n".join(
            [
                "# PE-0A2C Run Report",
                "",
                "- status: completed",
                f"- source: CalibrationCaptureBundle `{bundle.get('capture_id')}`",
                f"- calibration_id: `{profile.calibration_id}`",
                "- human_review_status: `pending`",
                "",
            ]
        ),
    )
    artifacts.extend(
        [
            artifact_record(run_manifest_path, artifact_type="run_manifest", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
            artifact_record(report_path, artifact_type="pe0a2c_run_report", video_hash=video.get("video_hash"), calibration_id=profile.calibration_id),
        ]
    )
    write_json(artifact_manifest_path, {"artifacts": artifacts})
    return {
        "run_dir": str(run_dir),
        "run_manifest": str(run_manifest_path),
        "artifact_manifest": str(artifact_manifest_path),
        "profile": str(profile_path),
        "qa": str(qa_path),
        "overlay": str(overlay_path),
        "overlay_clean": str(overlay_clean_path),
        "human_review_checklist": str(checklist_path),
        "calibration_id": profile.calibration_id,
        "video_hash": video.get("video_hash"),
        "frame_hash": actual_frame_hash,
        "status": "completed",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Materialize a real per-video calibration profile from legacy ROI payload.")
    parser.add_argument("--video", help="Path to the real local video file.")
    parser.add_argument("--roi-payload", help="Path to JSON payload containing roi_points.")
    parser.add_argument("--capture-bundle", help="Path to CalibrationCaptureBundle v1 JSON.")
    parser.add_argument("--reference-frame", help="Optional reference_frame.png path for --capture-bundle mode.")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--reference-frame-index", type=int)
    reference.add_argument("--reference-timestamp-sec", type=float)
    parser.add_argument("--match-id", default=None)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", default=str(AI_WORKER_ROOT / "runs"))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.capture_bundle:
        result = materialize_from_capture_bundle(args)
    else:
        if not args.video or not args.roi_payload:
            parser.error("--video and --roi-payload are required unless --capture-bundle is used")
        if args.reference_frame_index is None and args.reference_timestamp_sec is None:
            parser.error("--reference-frame-index or --reference-timestamp-sec is required without --capture-bundle")
        result = materialize_calibration(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
