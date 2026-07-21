from __future__ import annotations

from .schema import SCHEMA_VERSION, VideoCalibrationProfile


def update_profile_qa(profile: VideoCalibrationProfile) -> VideoCalibrationProfile:
    errors = []
    warnings = []

    if profile.schema_version != SCHEMA_VERSION:
        errors.append("schema_version_mismatch")
    if not profile.calibration_id:
        errors.append("missing_calibration_id")
    if profile.video.width <= 0 or profile.video.height <= 0:
        errors.append("invalid_video_resolution")
    if not profile.video.video_hash:
        errors.append("missing_video_hash")

    roi_validation = profile.detection_roi.validation or {}
    for warning in roi_validation.get("warnings", []):
        warnings.append(warning)
    for error in roi_validation.get("errors", []):
        errors.append(error)

    if profile.homography.status == "unavailable":
        warnings.append("homography_unavailable_until_semantic_landmarks_exist")
    elif profile.homography.status in {"invalid", "degraded"}:
        errors.extend(profile.homography.failure_reasons)

    profile.qa.schema_valid = not any(
        error in {"schema_version_mismatch", "missing_calibration_id", "invalid_video_resolution", "missing_video_hash"}
        for error in errors
    )
    profile.qa.roi_valid = bool(profile.detection_roi.valid)
    profile.qa.homography_valid = profile.homography.status == "valid"
    profile.qa.review_complete = profile.status in {"reviewed", "active"}
    profile.qa.warnings = sorted(set(warnings))
    profile.qa.errors = sorted(set(errors))
    return profile
