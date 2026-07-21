import logging
import math
from typing import Any, Optional

from .interfaces import DetectionFactory, DetectorResult
from .rfdetr_adapter import RFDETRDetectorAdapter
from .yolo_adapter import YOLODetectorAdapter

logger = logging.getLogger("OneFrame.Detectors.Factory")


VALID_DETECTOR_MODES = {
    "yolo_primary",
    "rfdetr_primary",
    "rfdetr_primary_shadow",
    "rfdetr_only",
    "dual_compare",
    "ensemble",
}


class CompositeDetectorAdapter:
    def __init__(
        self,
        mode: str,
        yolo: YOLODetectorAdapter,
        rfdetr: Optional[RFDETRDetectorAdapter] = None,
    ):
        self.mode = mode if mode in VALID_DETECTOR_MODES else "yolo_primary"
        self.yolo = yolo
        self.rfdetr = rfdetr
        self.frame_exporter = yolo.frame_exporter
        self._stats = {
            "mode": self.mode,
            "frames": 0,
            "yolo_frames_with_ball": 0,
            "rfdetr_frames_with_ball": 0,
            "rfdetr_only_frames": 0,
            "yolo_only_frames": 0,
            "both_detectors_frames": 0,
            "ball_detections_discarded_by_physics_filter": 0,
            "rfdetr_available": bool(rfdetr and rfdetr.is_available),
        }
        self._last_rfdetr_center = None
        self._discarded_debug = []

    @property
    def name(self) -> str:
        return self.mode

    @property
    def is_available(self) -> bool:
        return self.yolo.is_available or bool(self.rfdetr and self.rfdetr.is_available)

    def detect(
        self,
        frame: Any,
        use_sahi: bool = False,
        match_id: str = "unknown",
        frame_num: Optional[int] = None,
    ) -> DetectorResult:
        self._stats["frames"] += 1

        if self.mode == "yolo_primary":
            yolo_ball, yolo_people = self.yolo.detect(frame, use_sahi, match_id, frame_num)
            self._tag_detections(yolo_ball, "yolo_primary", frame_num)
            self._record(yolo_ball, [])
            return yolo_ball, yolo_people

        if self.mode == "rfdetr_only":
            rfdetr_ball = self.rfdetr.detect_ball(frame) if self.rfdetr else []
            self._tag_detections(rfdetr_ball, "rfdetr", frame_num)
            self._record([], rfdetr_ball)
            return rfdetr_ball, []

        yolo_ball, yolo_people = self.yolo.detect(frame, use_sahi, match_id, frame_num)
        rfdetr_ball = self.rfdetr.detect_ball(frame) if self.rfdetr else []
        self._tag_detections(yolo_ball, "yolo_compare", frame_num)
        self._tag_detections(rfdetr_ball, "rfdetr", frame_num)
        self._record(yolo_ball, rfdetr_ball)

        if self.mode == "rfdetr_primary_shadow":
            accepted_rfdetr = self._filter_rfdetr_physics(rfdetr_ball, frame_num)
            if accepted_rfdetr:
                return accepted_rfdetr, yolo_people
            fallback_enabled = bool(
                getattr(getattr(self.rfdetr, "config", None), "yolo_fallback_enabled", False)
            )
            if fallback_enabled and yolo_ball:
                self._tag_detections(yolo_ball, "yolo_fallback", frame_num)
                return yolo_ball, yolo_people
            return [], yolo_people

        if self.mode == "rfdetr_primary":
            return (rfdetr_ball or yolo_ball), yolo_people

        if self.mode == "dual_compare":
            return yolo_ball, yolo_people

        if self.mode == "ensemble":
            return self._merge_ball_detections(yolo_ball, rfdetr_ball), yolo_people

        return yolo_ball, yolo_people

    def detect_ball(self, frame: Any):
        ball_dets, _ = self.detect(frame)
        return ball_dets

    def _record(self, yolo_ball: list, rfdetr_ball: list) -> None:
        if yolo_ball:
            self._stats["yolo_frames_with_ball"] += 1
        if rfdetr_ball:
            self._stats["rfdetr_frames_with_ball"] += 1
        if yolo_ball and rfdetr_ball:
            self._stats["both_detectors_frames"] += 1
        elif yolo_ball:
            self._stats["yolo_only_frames"] += 1
        elif rfdetr_ball:
            self._stats["rfdetr_only_frames"] += 1

    def _tag_detections(self, detections: list, source: str, frame_num: Optional[int]) -> None:
        for det in detections or []:
            setattr(det, "detector_source", source)
            setattr(det, "detector_mode", self.mode)
            setattr(det, "frame_index", frame_num)
            if not getattr(det, "threshold", 0.0):
                threshold = 0.0
                if source.startswith("rfdetr") and self.rfdetr:
                    threshold = float(getattr(self.rfdetr, "conf_threshold", 0.0) or 0.0)
                setattr(det, "threshold", threshold)

    def _filter_rfdetr_physics(self, detections: list, frame_num: Optional[int]) -> list:
        if not detections:
            return []

        config = getattr(self.rfdetr, "config", None)
        min_size = float(getattr(config, "min_ball_size_px", 4.0) or 4.0)
        max_size = float(getattr(config, "max_ball_size_px", 90.0) or 90.0)
        min_ar = float(getattr(config, "min_aspect_ratio", 0.45) or 0.45)
        max_ar = float(getattr(config, "max_aspect_ratio", 2.2) or 2.2)
        max_jump = float(getattr(config, "max_jump_px", 280.0) or 280.0)

        accepted = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            reason = self._rfdetr_reject_reason(det, min_size, max_size, min_ar, max_ar, max_jump)
            if reason:
                self._stats["ball_detections_discarded_by_physics_filter"] += 1
                self._discarded_debug.append(
                    {
                        "frame_index": frame_num,
                        "reason": reason,
                        "confidence": float(getattr(det, "confidence", 0.0) or 0.0),
                        "bbox": [
                            float(getattr(det, "x", 0.0) or 0.0),
                            float(getattr(det, "y", 0.0) or 0.0),
                            float(getattr(det, "w", 0.0) or 0.0),
                            float(getattr(det, "h", 0.0) or 0.0),
                        ],
                    }
                )
                continue
            accepted.append(det)

        if accepted:
            best = accepted[0]
            self._last_rfdetr_center = (float(best.x), float(best.y))
        return accepted

    def _rfdetr_reject_reason(
        self,
        det: Any,
        min_size: float,
        max_size: float,
        min_ar: float,
        max_ar: float,
        max_jump: float,
    ) -> str:
        width = float(getattr(det, "w", 0.0) or 0.0)
        height = float(getattr(det, "h", 0.0) or 0.0)
        if min(width, height) < min_size or max(width, height) > max_size:
            return "bbox_size"

        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio < min_ar or aspect_ratio > max_ar:
            return "aspect_ratio"

        if self._last_rfdetr_center is not None:
            jump = math.hypot(float(det.x) - self._last_rfdetr_center[0], float(det.y) - self._last_rfdetr_center[1])
            if jump > max_jump:
                return "jump_distance"

        # Kalman compatibility is evaluated by the tracker in the full engine path.
        return ""

    def _merge_ball_detections(self, yolo_ball: list, rfdetr_ball: list) -> list:
        candidates = sorted(
            list(yolo_ball) + list(rfdetr_ball),
            key=lambda det: det.confidence,
            reverse=True,
        )
        kept = []
        for candidate in candidates:
            if all(self._bbox_iou(candidate, existing) <= 0.5 for existing in kept):
                kept.append(candidate)
        return kept

    def _bbox_iou(self, first: Any, second: Any) -> float:
        ax1 = first.x - first.w / 2.0
        ay1 = first.y - first.h / 2.0
        ax2 = first.x + first.w / 2.0
        ay2 = first.y + first.h / 2.0
        bx1 = second.x - second.w / 2.0
        by1 = second.y - second.h / 2.0
        bx2 = second.x + second.w / 2.0
        by2 = second.y + second.h / 2.0

        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = inter_w * inter_h
        union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        union += max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union -= intersection
        if union <= 0 or math.isclose(union, 0.0):
            return 0.0
        return intersection / union

    def get_stats(self) -> dict:
        stats = dict(self._stats)
        stats["yolo"] = self.yolo.get_stats()
        if self.rfdetr:
            stats["rfdetr"] = self.rfdetr.get_stats()
        stats["discarded_debug"] = list(self._discarded_debug[-200:])
        return stats


class DetectorFactory:
    @staticmethod
    def create(
        config: Any,
        yolo_detector: Any,
        detection_factory: DetectionFactory,
        device: str = "cuda",
    ) -> CompositeDetectorAdapter:
        mode = getattr(config, "detector_mode", "yolo_primary") or "yolo_primary"
        if mode not in VALID_DETECTOR_MODES:
            logger.warning("detector_mode invalido '%s'; usando yolo_primary", mode)
            mode = "yolo_primary"

        yolo_adapter = YOLODetectorAdapter(yolo_detector)
        rfdetr_adapter = None
        if mode in {"rfdetr_primary", "rfdetr_primary_shadow", "rfdetr_only", "dual_compare", "ensemble"}:
            rfdetr_adapter = RFDETRDetectorAdapter(
                detection_factory=detection_factory,
                config=getattr(config, "rfdetr", None),
                device=device,
            )

        logger.info("DetectorFactory activo: mode=%s", mode)
        return CompositeDetectorAdapter(mode=mode, yolo=yolo_adapter, rfdetr=rfdetr_adapter)
