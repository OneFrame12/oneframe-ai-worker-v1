from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float]


def coerce_point(point) -> Point:
    if isinstance(point, dict):
        return float(point.get("x", 0.0)), float(point.get("y", 0.0))
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        return float(point[0]), float(point[1])
    raise ValueError(f"Invalid point: {point!r}")


def coerce_points(points: Iterable) -> List[Point]:
    return [coerce_point(point) for point in points or []]


def normalize_point(point: Point, width: int, height: int) -> List[float]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [round(point[0] / float(width), 8), round(point[1] / float(height), 8)]


def denormalize_point(point: Sequence[float], width: int, height: int) -> List[float]:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return [round(float(point[0]) * width, 4), round(float(point[1]) * height, 4)]


def normalize_polygon(points: Iterable, width: int, height: int) -> List[List[float]]:
    return [normalize_point(point, width, height) for point in coerce_points(points)]


def denormalize_polygon(points: Iterable, width: int, height: int) -> List[List[float]]:
    return [denormalize_point(point, width, height) for point in points or []]


def signed_area(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def polygon_area(points: Sequence[Point]) -> float:
    return abs(signed_area(points))


def polygon_orientation(points: Sequence[Point]) -> str:
    # Screen coordinates have y downward, so positive shoelace reads clockwise.
    return "clockwise" if signed_area(points) > 0 else "counter_clockwise"


def has_duplicate_points(points: Sequence[Point], tolerance: float = 1e-6) -> bool:
    seen = set()
    for x, y in points:
        key = (round(x / tolerance), round(y / tolerance))
        if key in seen:
            return True
        seen.add(key)
    return False


def _orientation(a: Point, b: Point, c: Point) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1: Point, q1: Point, p2: Point, q2: Point) -> bool:
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(p1, p2, q1))
        or (o2 == 0 and _on_segment(p1, q2, q1))
        or (o3 == 0 and _on_segment(p2, p1, q2))
        or (o4 == 0 and _on_segment(p2, q1, q2))
    )


def polygon_self_intersecting(points: Sequence[Point]) -> bool:
    n = len(points)
    if n < 4:
        return False
    for i in range(n):
        a1 = points[i]
        a2 = points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            b1 = points[j]
            b2 = points[(j + 1) % n]
            if segments_intersect(a1, a2, b1, b2):
                return True
    return False


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def distance_point_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denom))
    proj_x = sx + t * dx
    proj_y = sy + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def distance_to_polygon_boundary(point: Point, polygon: Sequence[Point]) -> Optional[float]:
    if len(polygon) < 2:
        return None
    return min(
        distance_point_to_segment(point, polygon[idx], polygon[(idx + 1) % len(polygon)])
        for idx in range(len(polygon))
    )


def bbox_bottom_center(bbox_xyxy: Sequence[float]) -> List[float]:
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain [x1, y1, x2, y2]")
    x1, _y1, x2, y2 = [float(value) for value in bbox_xyxy]
    return [(x1 + x2) / 2.0, y2]


def person_inside_detection_roi(
    bbox_xyxy: Sequence[float],
    polygon_pixels: Sequence[Point],
    boundary_margin_px: float = 8.0,
) -> str:
    if not polygon_pixels:
        return "unavailable"
    anchor = tuple(bbox_bottom_center(bbox_xyxy))
    distance = distance_to_polygon_boundary(anchor, polygon_pixels)
    if distance is not None and distance <= boundary_margin_px:
        return "boundary_uncertain"
    return "inside" if point_in_polygon(anchor, polygon_pixels) else "outside"


def classify_person_roi_status(
    bbox_xyxy: Sequence[float],
    polygon_pixels: Sequence[Point],
    boundary_margin_px: float = 8.0,
) -> str:
    status = person_inside_detection_roi(bbox_xyxy, polygon_pixels, boundary_margin_px)
    if status == "inside":
        return "candidate_on_field_person"
    if status == "boundary_uncertain":
        return "boundary_uncertain"
    if status == "outside":
        return "exclude_from_sport_tracking"
    return "unavailable"


def classify_ball_roi_status(
    bbox_xyxy: Sequence[float],
    polygon_pixels: Sequence[Point],
    boundary_margin_px: float = 8.0,
) -> str:
    # Ball detections are never hard-excluded by ROI in the new contract.
    if not polygon_pixels:
        return "unavailable"
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    distance = distance_to_polygon_boundary(center, polygon_pixels)
    if distance is not None and distance <= boundary_margin_px:
        return "boundary_uncertain"
    return "inside" if point_in_polygon(center, polygon_pixels) else "outside"
