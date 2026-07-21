#!/usr/bin/env python3
"""
Synthetic validation for RF-DETR segment physics filters.

This does not process video and does not call RunPod. It exercises the real
SegmentPhysicsFilter used by process_segment_shadow with audit-derived cases.
"""
import os
import sys
import types


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

sys.modules.setdefault("gdown", types.SimpleNamespace(download=lambda *args, **kwargs: None))
sys.modules.setdefault("ultralytics", types.SimpleNamespace(__version__="synthetic-test"))

from config import VisionConfig
from engine import Detection
from handler import SegmentPhysicsFilter


def ball(x, y, w, h, confidence, source="rfdetr_primary"):
    return Detection(
        x=x,
        y=y,
        w=w,
        h=h,
        confidence=confidence,
        class_id=37 if source.startswith("rfdetr") else -1,
        class_name="ball",
        detector_source=source,
    )


def run_case(filter_obj, frame_index, timestamp_sec, rfdetr_det, yolo_det=None):
    accepted, discarded = filter_obj.filter(
        [rfdetr_det],
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        yolo_detections=[yolo_det] if yolo_det else [],
        diagnostic_mode=True,
    )
    return accepted, discarded


def assert_accepted(name, accepted, reason):
    assert accepted, f"{name}: expected accepted detection"
    metadata = accepted[0].metadata
    assert metadata["final_status"] == "accepted", f"{name}: wrong final_status {metadata}"
    assert metadata["acceptance_reason"] == reason, f"{name}: wrong reason {metadata}"
    return metadata


def assert_discarded(name, discarded):
    assert discarded, f"{name}: expected discarded detection"
    assert discarded[0]["kalman_used"] is False, f"{name}: stale state must not use Kalman"
    return discarded[0]


def main():
    config = VisionConfig()
    filt = SegmentPhysicsFilter(config)

    accepted, discarded = run_case(
        filt,
        60,
        2.002,
        ball(1335.975, 394.445, 39.68, 20.813, 0.492307),
    )
    meta60 = assert_accepted("frame 60", accepted, "high_confidence_geometry_when_stale")
    assert meta60["motion_state_status"] == "uninitialized", meta60
    assert not discarded

    accepted, discarded = run_case(
        filt,
        70,
        2.335,
        ball(1210.86, 560.537, 27.239, 29.512, 0.566498),
    )
    meta70 = assert_accepted("frame 70", accepted, "fresh_motion_gate")
    assert meta70["motion_state_status"] == "fresh", meta70
    assert meta70["kalman_used"] is False, meta70
    assert not discarded

    accepted, discarded = run_case(
        filt,
        500,
        16.677,
        ball(805.618, 200.041, 13.49, 10.677, 0.492115),
        yolo_det=ball(804.728, 200.981, 10.963, 19.023, 0.641932, source="yolo_compare"),
    )
    meta500 = assert_accepted("frame 500", accepted, "diagnostic_yolo_agreement")
    assert meta500["motion_state_status"] == "stale", meta500
    assert meta500["diagnostic_override_used"] is True, meta500
    assert meta500["kalman_used"] is False, meta500
    assert meta500["yolo_center_distance_px"] <= 25.0, meta500
    assert not discarded

    accepted, discarded = run_case(
        filt,
        949,
        31.687,
        ball(1174.974, 238.176, 17.067, 13.382, 0.632075),
        yolo_det=ball(1179.259, 237.991, 17.494, 16.717, 0.496145, source="yolo_compare"),
    )
    meta949 = assert_accepted("frame 949", accepted, "diagnostic_yolo_agreement")
    assert meta949["motion_state_status"] == "stale", meta949
    assert meta949["diagnostic_override_used"] is True, meta949
    assert meta949["kalman_used"] is False, meta949
    assert meta949["yolo_center_distance_px"] <= 25.0, meta949
    assert not discarded

    accepted, discarded = run_case(
        filt,
        1099,
        36.69,
        ball(623.328, 297.562, 15.898, 11.752, 0.308712),
        yolo_det=ball(1423.039, 287.158, 10.595, 16.36, 0.328348, source="yolo_compare"),
    )
    row1099 = assert_discarded("frame 1099", discarded)
    assert not accepted
    assert row1099["motion_state_status"] == "stale", row1099
    assert row1099["yolo_agreement"] is False, row1099
    assert row1099["yolo_center_distance_px"] > 700.0, row1099
    assert row1099["reason"] == "stale_motion_low_confidence", row1099

    print("OK synthetic RF-DETR filter validation")
    print("frame 500 accepted:", meta500["acceptance_reason"])
    print("frame 949 accepted:", meta949["acceptance_reason"])
    print("frame 1099 discarded:", row1099["reason"])


if __name__ == "__main__":
    main()
