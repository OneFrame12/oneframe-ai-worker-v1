from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, List, Optional


SCHEMA_VERSION = "oneframe.video_calibration.v1"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def derive_calibration_id(
    video_hash: str,
    reference_frame_index: int,
    profile_version: int = 1,
) -> str:
    material = f"{video_hash}:{int(reference_frame_index)}:{int(profile_version)}"
    digest = sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"vc_{digest}"


@dataclass
class VideoMetadata:
    video_hash: str
    match_id: Optional[str] = None
    match_uuid: Optional[str] = None
    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    fps: Optional[float] = None
    reference_frame_index: int = 0
    reference_timestamp_sec: float = 0.0
    frame_hash: Optional[str] = None


@dataclass
class CameraMetadata:
    camera_type: str = "fixed_or_mostly_fixed_behind_goal"
    expected_fixed_within_video: bool = True
    angle_reusable_across_videos: bool = False


@dataclass
class DetectionROI:
    source: str = "existing_calibration_tool"
    polygon_normalized: List[List[float]] = field(default_factory=list)
    polygon_pixels_reference: List[List[float]] = field(default_factory=list)
    point_order: List[int] = field(default_factory=list)
    reviewed: bool = False
    valid: bool = False
    warnings: List[str] = field(default_factory=list)
    validation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Landmark:
    landmark_id: str
    landmark_type: str
    image_point_normalized: List[float]
    image_point_pixels_reference: List[float]
    field_point_m: Optional[List[float]] = None
    source: str = "manual"
    reviewed: bool = False


@dataclass
class HomographyState:
    status: str = "unavailable"
    matrix_image_to_field: Optional[List[List[float]]] = None
    matrix_field_to_image: Optional[List[List[float]]] = None
    correspondence_ids: List[str] = field(default_factory=list)
    mean_reprojection_error_px: Optional[float] = None
    reviewed: bool = False
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class StabilityState:
    status: str = "unknown"
    valid_from_frame: Optional[int] = None
    valid_to_frame: Optional[int] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class QAState:
    schema_valid: bool = False
    roi_valid: bool = False
    homography_valid: bool = False
    review_complete: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class VideoCalibrationProfile:
    calibration_id: str
    schema_version: str = SCHEMA_VERSION
    profile_version: int = 1
    status: str = "draft"
    video: VideoMetadata = field(default_factory=lambda: VideoMetadata(video_hash=""))
    camera: CameraMetadata = field(default_factory=CameraMetadata)
    detection_roi: DetectionROI = field(default_factory=DetectionROI)
    ignore_regions: List[Dict[str, Any]] = field(default_factory=list)
    foreground_goal_occlusion: Optional[Dict[str, Any]] = None
    landmarks: List[Landmark] = field(default_factory=list)
    homography: HomographyState = field(default_factory=HomographyState)
    stability: StabilityState = field(default_factory=StabilityState)
    qa: QAState = field(default_factory=QAState)
    created_at: str = field(default_factory=utc_now_iso)
    provenance: Dict[str, Any] = field(default_factory=dict)
    legacy_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoCalibrationProfile":
        video = VideoMetadata(**data.get("video", {}))
        camera = CameraMetadata(**data.get("camera", {}))
        detection_roi = DetectionROI(**data.get("detection_roi", {}))
        homography = HomographyState(**data.get("homography", {}))
        stability = StabilityState(**data.get("stability", {}))
        qa = QAState(**data.get("qa", {}))
        landmarks = [Landmark(**item) for item in data.get("landmarks", [])]
        return cls(
            calibration_id=data["calibration_id"],
            schema_version=data.get("schema_version", SCHEMA_VERSION),
            profile_version=int(data.get("profile_version", 1)),
            status=data.get("status", "draft"),
            video=video,
            camera=camera,
            detection_roi=detection_roi,
            ignore_regions=list(data.get("ignore_regions", [])),
            foreground_goal_occlusion=data.get("foreground_goal_occlusion"),
            landmarks=landmarks,
            homography=homography,
            stability=stability,
            qa=qa,
            created_at=data.get("created_at", utc_now_iso()),
            provenance=dict(data.get("provenance", {})),
            legacy_metadata=dict(data.get("legacy_metadata", {})),
        )
