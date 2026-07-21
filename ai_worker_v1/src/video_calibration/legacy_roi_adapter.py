from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, Iterable, Optional

from .coordinates import coerce_points, normalize_polygon
from .homography import set_homography_from_landmarks
from .qa import update_profile_qa
from .schema import (
    DetectionROI,
    VideoCalibrationProfile,
    VideoMetadata,
    derive_calibration_id,
)
from .validator import validate_roi


def _frame_hash_from_inputs(
    video_hash: str,
    reference_frame_index: int,
    reference_timestamp_sec: float,
    width: int,
    height: int,
) -> str:
    material = f"{video_hash}:{reference_frame_index}:{reference_timestamp_sec:.6f}:{width}x{height}"
    return sha256(material.encode("utf-8")).hexdigest()


def build_profile_from_legacy_roi(
    *,
    roi_points: Iterable,
    frame_width: int,
    frame_height: int,
    match_id: Optional[str],
    video_hash: str,
    reference_frame_index: int = 0,
    reference_timestamp_sec: float = 0.0,
    danger_zone: Optional[Iterable] = None,
    fps: Optional[float] = None,
    profile_version: int = 1,
    match_uuid: Optional[str] = None,
    expected_aspect_ratio: Optional[float] = None,
) -> VideoCalibrationProfile:
    pixel_points = coerce_points(roi_points)
    validation = validate_roi(
        pixel_points,
        frame_width,
        frame_height,
        expected_aspect_ratio=expected_aspect_ratio,
    )
    normalized = normalize_polygon(pixel_points, frame_width, frame_height)
    calibration_id = derive_calibration_id(video_hash, reference_frame_index, profile_version)
    aspect_ratio = frame_width / float(frame_height) if frame_width and frame_height else 0.0

    detection_roi = DetectionROI(
        source="existing_calibration_tool",
        polygon_normalized=normalized,
        polygon_pixels_reference=[[float(x), float(y)] for x, y in pixel_points],
        point_order=list(range(len(pixel_points))),
        reviewed=False,
        valid=bool(validation["valid"]),
        warnings=list(validation.get("warnings", [])),
        validation=validation,
    )

    profile = VideoCalibrationProfile(
        calibration_id=calibration_id,
        profile_version=profile_version,
        status="draft" if validation["valid"] else "invalid",
        video=VideoMetadata(
            video_hash=video_hash,
            match_id=match_id,
            match_uuid=match_uuid,
            width=int(frame_width),
            height=int(frame_height),
            aspect_ratio=round(aspect_ratio, 8),
            fps=fps,
            reference_frame_index=int(reference_frame_index),
            reference_timestamp_sec=float(reference_timestamp_sec),
            frame_hash=_frame_hash_from_inputs(
                video_hash,
                reference_frame_index,
                reference_timestamp_sec,
                frame_width,
                frame_height,
            ),
        ),
        detection_roi=detection_roi,
        provenance={
            "source": "legacy_roi_adapter",
            "ui": "app/components/calibration-tool.js",
            "payload_format": "[[x, y], ...]",
        },
        legacy_metadata={
            "danger_zone_pixels_reference": (
                [[float(x), float(y)] for x, y in coerce_points(danger_zone)]
                if danger_zone
                else []
            ),
            "danger_zone_note": "legacy danger_zone preserved separately; not metric geometry",
        },
    )
    set_homography_from_landmarks(profile)
    update_profile_qa(profile)
    return profile


def build_legacy_import_report(profile: VideoCalibrationProfile) -> Dict[str, Any]:
    return {
        "calibration_id": profile.calibration_id,
        "source": "existing_calibration_tool",
        "imported_fields": [
            "roi_points",
            "frame_width",
            "frame_height",
            "match_id",
            "video_hash",
            "reference_frame_index",
            "reference_timestamp_sec",
            "danger_zone",
        ],
        "omitted_fields": [
            "homography_correspondences",
            "field_point_m",
            "goalpost_landmarks",
            "ignore_regions",
        ],
        "warnings": list(profile.qa.warnings),
        "errors": list(profile.qa.errors),
        "homography_status": profile.homography.status,
    }
