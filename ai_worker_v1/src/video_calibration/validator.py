from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .coordinates import (
    coerce_points,
    has_duplicate_points,
    polygon_area,
    polygon_orientation,
    polygon_self_intersecting,
)


def validate_roi(
    roi_points: Iterable,
    width: int,
    height: int,
    min_area_ratio: float = 0.02,
    expected_aspect_ratio: float | None = None,
    aspect_ratio_tolerance: float = 0.02,
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        points = coerce_points(roi_points)
    except ValueError as exc:
        return {
            "valid": False,
            "errors": [str(exc)],
            "warnings": [],
            "area_ratio": 0.0,
            "orientation": "counter_clockwise",
            "self_intersecting": False,
        }

    if width <= 0 or height <= 0:
        errors.append("resolution_unknown")
    if len(points) < 3:
        errors.append("minimum_3_points_required")
    if has_duplicate_points(points):
        errors.append("duplicate_points")

    for idx, (x, y) in enumerate(points):
        if width > 0 and height > 0 and not (0 <= x <= width and 0 <= y <= height):
            errors.append(f"point_{idx}_outside_image")

    area_ratio = 0.0
    orientation = "counter_clockwise"
    self_intersecting = False
    if len(points) >= 3 and width > 0 and height > 0:
        area = polygon_area(points)
        area_ratio = area / float(width * height)
        orientation = polygon_orientation(points)
        self_intersecting = polygon_self_intersecting(points)
        if area <= 0:
            errors.append("area_not_positive")
        if area_ratio < min_area_ratio:
            errors.append("area_below_minimum")
        if self_intersecting:
            errors.append("self_intersecting_polygon")

    if width > 0 and height > 0:
        aspect_ratio = width / float(height)
        if expected_aspect_ratio is not None:
            delta = abs(aspect_ratio - expected_aspect_ratio)
            if delta > aspect_ratio_tolerance:
                errors.append("aspect_ratio_mismatch")
        elif aspect_ratio <= 0:
            errors.append("aspect_ratio_unknown")
    else:
        errors.append("aspect_ratio_unknown")

    if len(points) > 4:
        warnings.append("roi_has_more_than_four_points")
    if len(points) == 4:
        warnings.append("roi_four_points_are_not_metric_correspondences")

    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "area_ratio": round(area_ratio, 8),
        "orientation": orientation,
        "self_intersecting": self_intersecting,
    }


def compare_static_frame_evidence(
    reference_frame_index: int,
    comparison_frame_index: int,
    translation_px: float,
    scale_delta: float,
    rotation_delta_deg: float,
    *,
    max_translation_px: float = 12.0,
    max_scale_delta: float = 0.015,
    max_rotation_delta_deg: float = 1.0,
) -> Dict[str, Any]:
    """Lightweight future-facing stability validator.

    PE-0A1 does not run this on real video. This helper only defines the
    decision contract for later feature/static-background checks.
    """
    moved = (
        abs(float(translation_px)) > max_translation_px
        or abs(float(scale_delta)) > max_scale_delta
        or abs(float(rotation_delta_deg)) > max_rotation_delta_deg
    )
    return {
        "reference_frame_index": int(reference_frame_index),
        "comparison_frame_index": int(comparison_frame_index),
        "translation_px": float(translation_px),
        "scale_delta": float(scale_delta),
        "rotation_delta_deg": float(rotation_delta_deg),
        "status": "moved" if moved else "mostly_fixed",
        "thresholds": {
            "max_translation_px": max_translation_px,
            "max_scale_delta": max_scale_delta,
            "max_rotation_delta_deg": max_rotation_delta_deg,
        },
    }
