from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .schema import VideoCalibrationProfile


def render_calibration_overlay(
    profile: VideoCalibrationProfile,
    output_path: str | Path,
    image_path: Optional[str | Path] = None,
    *,
    danger_zone_pixels: Optional[Iterable] = None,
    clean: bool = False,
    extra_lines: Optional[list[str]] = None,
) -> Path:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError("Pillow is required for video calibration overlay rendering") from exc

    width = max(int(profile.video.width or 1280), 1)
    height = max(int(profile.video.height or 720), 1)
    if image_path:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
    else:
        image = Image.new("RGB", (width, height), (18, 22, 28))

    draw = ImageDraw.Draw(image, "RGBA")
    points = [tuple(point) for point in profile.detection_roi.polygon_pixels_reference]
    if len(points) >= 3:
        draw.polygon(points, fill=(0, 180, 90, 45), outline=(0, 255, 140, 230))
        for idx, (x, y) in enumerate(points):
            r = 6
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 230), outline=(0, 0, 0, 255))
            draw.text((x + 8, y - 8), str(idx), fill=(255, 255, 255, 255))

    danger_points = [tuple(point) for point in danger_zone_pixels or []]
    if len(danger_points) >= 3:
        draw.polygon(danger_points, fill=(255, 130, 0, 38), outline=(255, 155, 0, 220))
        label_x, label_y = danger_points[0]
        draw.text((label_x + 8, label_y + 8), "LEGACY DANGER ZONE - NON-METRIC", fill=(255, 220, 140, 255))

    if clean:
        label = f"{profile.calibration_id} ROI draft"
        draw.rectangle((0, 0, min(width, 420), 34), fill=(0, 0, 0, 150))
        draw.text((12, 10), label, fill=(255, 255, 255, 255))
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, format="PNG")
        return destination

    area_ratio = profile.detection_roi.validation.get("area_ratio", 0.0)
    orientation = profile.detection_roi.validation.get("orientation", "unknown")
    status = "VALID" if profile.detection_roi.valid else "INVALID"
    video_hash_short = (profile.video.video_hash or "")[:10]
    lines = [
        f"VideoCalibrationProfile {profile.schema_version}",
        f"calibration_id: {profile.calibration_id}",
        f"resolution: {profile.video.width}x{profile.video.height}",
        f"video_hash: {video_hash_short}",
        f"frame: {profile.video.reference_frame_index} t={profile.video.reference_timestamp_sec:.3f}s",
        f"ROI: {status} area={area_ratio:.2%} orientation={orientation}",
        "person ROI anchor: bbox bottom-center",
        "ball ROI status: never hard-excluded",
        f"homography: {profile.homography.status.upper()}",
    ]
    if profile.homography.status == "unavailable":
        lines.append("HOMOGRAPHY UNAVAILABLE - ROI ONLY")
    if danger_points:
        lines.append("legacy danger zone: present, non-metric")
    for warning in profile.qa.warnings[:4]:
        lines.append(f"warning: {warning}")
    if extra_lines:
        lines.extend(extra_lines)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    margin = 12
    line_height = 15
    box_height = margin * 2 + line_height * len(lines)
    draw.rectangle((0, 0, min(width, 760), box_height), fill=(0, 0, 0, 160))
    for idx, line in enumerate(lines):
        draw.text((margin, margin + idx * line_height), line, fill=(255, 255, 255, 255), font=font)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG")
    return destination
