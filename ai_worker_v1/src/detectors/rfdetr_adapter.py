import importlib
import logging
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import numpy as np

from .interfaces import DetectionFactory, DetectionList, DetectorResult

logger = logging.getLogger("OneFrame.Detectors.RFDETR")


class RFDETRDetectorAdapter:
    name = "rfdetr"

    BALL_CLASS_THRESHOLD = 0.25
    BALL_SIZE_MIN = 0.003
    BALL_SIZE_MAX = 0.09

    def __init__(
        self,
        detection_factory: DetectionFactory,
        config: Optional[Any] = None,
        model_path: Optional[str] = None,
        device: str = "cuda",
    ):
        self.detection_factory = detection_factory
        self.config = config
        self.model_path = (
            model_path
            or os.getenv("AI_WORKER_V1_RFDETR_MODEL_PATH", "")
            or str(getattr(config, "model_path", "") or "")
        )
        self.conf_threshold = float(getattr(config, "conf_threshold", self.BALL_CLASS_THRESHOLD))
        self.device = os.getenv("AI_WORKER_V1_RFDETR_DEVICE", "") or str(
            getattr(config, "device", device) or device
        )
        self.sports_ball_class_id = int(getattr(config, "sports_ball_class_id", 37) or 37)
        self.ball_class_name = str(getattr(config, "ball_class_name", "ball") or "ball")
        self.class_names = {}
        self.model = None
        self.frame_exporter = None
        self.unavailable_reason = ""
        self.load_error = ""
        self._minimal_inference_ok = False
        self._calls = 0
        self._ball_detections = 0
        self._load_model()

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @property
    def detector_status(self) -> str:
        if self.is_available and self._minimal_inference_ok:
            return "available"
        if self.unavailable_reason:
            return "unavailable"
        if self.load_error:
            return "error"
        return "unavailable"

    def _load_model(self) -> None:
        try:
            rfdetr_module = importlib.import_module("rfdetr")
            RFDETRBase = getattr(rfdetr_module, "RFDETRBase")
            model_kwargs = {"device": self.device} if self.device else {}

            if self.model_path and Path(self.model_path).exists():
                self.model = RFDETRBase(pretrain_weights=self.model_path, **model_kwargs)
                logger.info("RF-DETR adapter cargado con pesos: %s", self.model_path)
            elif self.model_path:
                self.unavailable_reason = f"RF-DETR weights not found: {self.model_path}"
                logger.warning(self.unavailable_reason)
                return
            else:
                self.model = RFDETRBase(**model_kwargs)
                logger.info("RF-DETR adapter cargado con modelo base")
            self.class_names = dict(getattr(self.model, "class_names", {}) or {})
            self._run_minimal_inference_check()
        except ImportError as exc:
            self.model = None
            self.unavailable_reason = str(exc)
            logger.warning("RF-DETR adapter no disponible: %s", exc)
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            logger.warning("RF-DETR adapter fallo iniciando: %s", exc)

    def _run_minimal_inference_check(self) -> None:
        if self.model is None:
            return
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.model.predict(self._prepare_frame_for_model(frame), threshold=self.conf_threshold)
        self._minimal_inference_ok = True

    def detect(
        self,
        frame: Any,
        use_sahi: bool = False,
        match_id: str = "unknown",
        frame_num: Optional[int] = None,
    ) -> DetectorResult:
        return self.detect_ball(frame), []

    def detect_ball(self, frame: Any) -> DetectionList:
        self._calls += 1
        if not self.is_available:
            return []

        try:
            h, w = frame.shape[:2]
            detections = []
            for pred in self.raw_detections(frame, threshold=self.conf_threshold):
                parsed = self._prediction_to_detection(pred, frame_w=w, frame_h=h)
                if parsed is not None:
                    detections.append(parsed)
            detections.sort(key=lambda det: det.confidence, reverse=True)
            self._ball_detections += len(detections)
            return detections
        except Exception as exc:
            logger.debug("RF-DETR detect fallo: %s", exc)
            return []

    def raw_detections(
        self,
        frame: Any,
        threshold: Optional[float] = None,
        top_n: Optional[int] = None,
    ) -> List[dict]:
        if not self.is_available:
            return []

        raw_results = self.model.predict(
            self._prepare_frame_for_model(frame),
            threshold=self.conf_threshold if threshold is None else float(threshold),
        )
        detections = [self._normalize_prediction(pred) for pred in self._iter_predictions(raw_results)]
        detections = [det for det in detections if det is not None]
        detections.sort(key=lambda det: det.get("confidence", 0.0), reverse=True)
        if top_n is not None:
            return detections[: int(top_n)]
        return detections

    def _prepare_frame_for_model(self, frame: Any) -> Any:
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[2] == 3:
            # Frames read by cv2 are BGR. RF-DETR expects RGB numpy/PIL inputs.
            return frame[:, :, ::-1].copy()
        return frame

    def _iter_predictions(self, results: Any) -> Iterable[Any]:
        if results is None:
            return []
        if isinstance(results, dict):
            return results.get("predictions") or results.get("detections") or []
        if hasattr(results, "predictions"):
            return getattr(results, "predictions") or []
        if all(hasattr(results, attr) for attr in ("xyxy", "confidence", "class_id")):
            rows = []
            for xyxy, confidence, class_id in zip(
                getattr(results, "xyxy"),
                getattr(results, "confidence"),
                getattr(results, "class_id"),
            ):
                rows.append(
                    {
                        "xyxy": xyxy,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )
            return rows
        return results

    def _prediction_to_detection(self, pred: Any, frame_w: int, frame_h: int):
        pred = self._normalize_prediction(pred)
        if pred is None:
            return None

        if not self._is_ball_prediction(pred):
            return None

        xmin = float(pred.get("xmin", pred.get("x1", 0.0)) or 0.0)
        ymin = float(pred.get("ymin", pred.get("y1", 0.0)) or 0.0)
        xmax = float(pred.get("xmax", pred.get("x2", 0.0)) or 0.0)
        ymax = float(pred.get("ymax", pred.get("y2", 0.0)) or 0.0)
        bw = max(0.0, xmax - xmin)
        bh = max(0.0, ymax - ymin)
        rel_w = bw / max(frame_w, 1)
        rel_h = bh / max(frame_h, 1)

        if not (
            self.BALL_SIZE_MIN < rel_w < self.BALL_SIZE_MAX
            and self.BALL_SIZE_MIN < rel_h < self.BALL_SIZE_MAX
        ):
            return None

        confidence = float(pred.get("confidence", pred.get("score", 0.0)) or 0.0)
        return self.detection_factory(
            x=xmin + bw / 2.0,
            y=ymin + bh / 2.0,
            w=bw,
            h=bh,
            confidence=confidence,
            class_id=self.sports_ball_class_id,
            class_name=self.ball_class_name,
            track_id=None,
        )

    def _normalize_prediction(self, pred: Any) -> Optional[dict]:
        if not isinstance(pred, dict):
            pred = self._object_to_dict(pred)

        if not pred:
            return None

        if "xyxy" in pred:
            xyxy = pred.get("xyxy")
            if xyxy is not None and len(xyxy) >= 4:
                pred["xmin"] = float(xyxy[0])
                pred["ymin"] = float(xyxy[1])
                pred["xmax"] = float(xyxy[2])
                pred["ymax"] = float(xyxy[3])

        class_id = self._extract_class_id(pred)
        if class_id is not None:
            pred["class_id"] = class_id
            pred["class_name"] = self.class_names.get(class_id, str(pred.get("class_name", "")))

        confidence = pred.get("confidence", pred.get("score", 0.0))
        pred["confidence"] = float(confidence or 0.0)
        return pred

    def _is_ball_prediction(self, pred: dict) -> bool:
        raw_class = self._extract_class_id(pred)
        if raw_class is None:
            raw_class = pred.get("class_name") or pred.get("label")
        if raw_class is None:
            return False
        try:
            return int(raw_class) == self.sports_ball_class_id
        except (TypeError, ValueError):
            return str(raw_class).lower().replace("_", " ") in {
                "sports ball",
                "ball",
                self.ball_class_name.lower(),
            }

    def _extract_class_id(self, pred: dict) -> Optional[int]:
        for key in ("class_id", "class", "category_id"):
            if pred.get(key) is None:
                continue
            try:
                return int(pred.get(key))
            except (TypeError, ValueError):
                continue
        return None

    def _object_to_dict(self, pred: Any) -> dict:
        data = {}
        for key in (
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "x1",
            "y1",
            "x2",
            "y2",
            "confidence",
            "score",
            "class_id",
            "class",
            "category_id",
            "label",
            "class_name",
            "xyxy",
        ):
            if hasattr(pred, key):
                value = getattr(pred, key)
                if isinstance(value, np.generic):
                    value = value.item()
                data[key] = value
        return data

    def get_stats(self) -> dict:
        return {
            "detector": self.name,
            "available": self.is_available,
            "detector_status": self.detector_status,
            "unavailable_reason": self.unavailable_reason,
            "load_error": self.load_error,
            "conf_threshold": self.conf_threshold,
            "sports_ball_class_id": self.sports_ball_class_id,
            "sports_ball_class_name": self.class_names.get(self.sports_ball_class_id, ""),
            "calls": self._calls,
            "ball_detections": self._ball_detections,
        }
