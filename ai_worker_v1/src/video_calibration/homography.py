from __future__ import annotations

from typing import List

from .schema import HomographyState, VideoCalibrationProfile


def eligible_landmark_correspondences(profile: VideoCalibrationProfile) -> List:
    return [
        landmark
        for landmark in profile.landmarks
        if landmark.field_point_m is not None
        and len(landmark.field_point_m) >= 2
        and len(landmark.image_point_pixels_reference) >= 2
    ]


def set_homography_from_landmarks(profile: VideoCalibrationProfile) -> HomographyState:
    correspondences = eligible_landmark_correspondences(profile)
    if not correspondences:
        profile.homography = HomographyState(
            status="unavailable",
            failure_reasons=[
                "roi_points_are_detection_roi_only",
                "semantic_landmark_correspondences_required",
            ],
        )
        return profile.homography
    if len(correspondences) < 4:
        profile.homography = HomographyState(
            status="provisional",
            correspondence_ids=[item.landmark_id for item in correspondences],
            failure_reasons=["minimum_4_landmark_correspondences_required_for_valid_homography"],
        )
        return profile.homography

    try:
        import cv2
        import numpy as np
    except Exception:
        profile.homography = HomographyState(
            status="provisional",
            correspondence_ids=[item.landmark_id for item in correspondences],
            failure_reasons=["opencv_unavailable_for_homography_estimation"],
        )
        return profile.homography

    image_points = np.float32([item.image_point_pixels_reference[:2] for item in correspondences])
    field_points = np.float32([item.field_point_m[:2] for item in correspondences])
    matrix, _mask = cv2.findHomography(image_points, field_points)
    inverse = None
    if matrix is not None:
        inverse = np.linalg.inv(matrix)

    if matrix is None or inverse is None:
        profile.homography = HomographyState(
            status="invalid",
            correspondence_ids=[item.landmark_id for item in correspondences],
            failure_reasons=["cv2_findHomography_failed"],
        )
        return profile.homography

    profile.homography = HomographyState(
        status="provisional",
        matrix_image_to_field=matrix.tolist(),
        matrix_field_to_image=inverse.tolist(),
        correspondence_ids=[item.landmark_id for item in correspondences],
        mean_reprojection_error_px=None,
        reviewed=False,
        failure_reasons=["homography_requires_manual_review"],
    )
    return profile.homography
