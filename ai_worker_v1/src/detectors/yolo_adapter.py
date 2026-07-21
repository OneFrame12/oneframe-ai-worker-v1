from typing import Any, Optional

from .interfaces import DetectionList, DetectorResult


class YOLODetectorAdapter:
    name = "yolo"

    def __init__(self, ball_detector: Any, inference_only: bool = False):
        self.ball_detector = ball_detector
        self.inference_only = inference_only
        self.frame_exporter = getattr(ball_detector, "frame_exporter", None)
        self._calls = 0
        self._ball_detections = 0

    @property
    def is_available(self) -> bool:
        return getattr(self.ball_detector, "model", None) is not None

    @property
    def detector_status(self) -> str:
        return "available" if self.is_available else "unavailable"

    def detect(
        self,
        frame: Any,
        use_sahi: bool = False,
        match_id: str = "unknown",
        frame_num: Optional[int] = None,
    ) -> DetectorResult:
        self._calls += 1
        if self.inference_only:
            cfg = self.ball_detector.config.yolo
            results = self.ball_detector.model(
                frame,
                conf=cfg.confidence,
                imgsz=cfg.imgsz,
                verbose=False,
            )
            ball_dets, person_dets = self.ball_detector._parse_yolo_results(results)
        else:
            ball_dets, person_dets = self.ball_detector.detect(
                frame,
                use_sahi=use_sahi,
                match_id=match_id,
                frame_num=frame_num,
            )
        self._ball_detections += len(ball_dets)
        return ball_dets, person_dets

    def detect_ball(self, frame: Any) -> DetectionList:
        ball_dets, _ = self.detect(frame)
        return ball_dets

    def get_stats(self) -> dict:
        return {
            "detector": self.name,
            "detector_status": self.detector_status,
            "inference_only": self.inference_only,
            "calls": self._calls,
            "ball_detections": self._ball_detections,
        }
