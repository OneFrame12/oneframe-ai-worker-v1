"""
OneFrame V57 — Engine de Tracking y Análisis
==============================================
Tracking robusto de balón para partidos completos filmados con GoPro.

Clases principales:
  BallDetector   — YOLO inference + auto-detección de clase "ball"
  BallTracker    — Kalman filter + candidate scoring + continuidad
  AudioAnalyzer  — STA/LTA para picos de audio
  AudioPreScanner — Pre-escaneo de audio para ventanas candidatas
  VisionEngine   — Orquestador: procesa video completo, genera telemetría
  GameReferee    — Eventos (danger zone, incoming shots) + clips
"""

import cv2
import numpy as np
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from scipy.io import wavfile

try:
    from filterpy.kalman import KalmanFilter as FilterPyKF
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False

try:
    from shapely.geometry import Point, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from config import VisionConfig, DEFAULT_CONFIG, CameraCalibrationConfig
from learner import EvolutiveLearner
from frame_exporter import FrameExporter
from goal_detector import GoalDetector
from homography import HomographyTransformer
from player_detector import PlayerDetector as TeamPlayerDetector
from detectors import DetectorFactory
# Fase 0 shadow: RF-DETR solo entra por detector_mode experimental; SAM3 y CoTracker siguen desactivados.

logger = logging.getLogger("OneFrame.Engine")


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    class_name: str = ""
    track_id: Optional[int] = None
    detector_source: str = ""
    detector_mode: str = ""
    threshold: float = 0.0
    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackPoint:
    frame_idx: int
    timestamp_sec: float
    x: float
    y: float
    speed_px: float
    speed_px_per_sec: float
    speed_kmh: float
    confidence: float
    is_detected: bool
    is_gk_kick: bool = False
    direction_deg: Optional[float] = None
    approach_angle_deg: Optional[float] = None
    dist_to_danger_center: Optional[float] = None
    velocity_class: str = "low"
    players_in_danger_zone: int = 0
    track_id: Optional[int] = None


@dataclass
class RejectedCandidate:
    frame: int
    timestamp: float
    conf: float
    dist_px: float
    max_jump: float


@dataclass
class EventCandidate:
    timestamp_sec: float
    frame_idx: int
    event_type: str
    score: float
    speed: float
    in_danger_zone: bool
    speed_kmh: float = 0.0
    velocity_class: str = "low"
    is_approaching: bool = False
    has_audio_peak: bool = False
    approach_angle_deg: Optional[float] = None
    score_breakdown: Dict[str, float] = field(default_factory=dict)


# ============================================================
# CAMERA UNDISTORTER
# ============================================================

class CameraUndistorter:
    """
    Corrige distorsión de lente gran angular usando cv2.undistort().
    Se inicializa una vez y aplica la corrección a cada frame.
    Costo: ~2ms por frame en GPU, insignificante vs YOLO (~30ms).
    """

    def __init__(self, config: CameraCalibrationConfig):
        self.config = config
        self.enabled = config.enabled
        self._camera_matrix = None
        self._dist_coeffs = None
        self._new_camera_matrix = None
        self._map1 = None
        self._map2 = None
        self._initialized = False

    def initialize(self, frame_width: int, frame_height: int):
        """
        Pre-calcula los mapas de remapeo para la resolución del video.
        Llamar una vez al inicio del procesamiento.
        """
        if not self.enabled:
            return

        cfg = self.config

        self._camera_matrix = np.array([
            [cfg.fx, 0, cfg.cx],
            [0, cfg.fy, cfg.cy],
            [0, 0, 1],
        ], dtype=np.float64)

        self._dist_coeffs = np.array(
            [cfg.k1, cfg.k2, cfg.p1, cfg.p2, cfg.k3],
            dtype=np.float64,
        )

        scale_x = frame_width / cfg.expected_width
        scale_y = frame_height / cfg.expected_height
        if abs(scale_x - 1.0) > 0.01 or abs(scale_y - 1.0) > 0.01:
            self._camera_matrix[0, 0] *= scale_x  # fx
            self._camera_matrix[1, 1] *= scale_y  # fy
            self._camera_matrix[0, 2] *= scale_x  # cx
            self._camera_matrix[1, 2] *= scale_y  # cy

        self._new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self._camera_matrix,
            self._dist_coeffs,
            (frame_width, frame_height),
            alpha=0.5,
            newImgSize=(frame_width, frame_height),
        )

        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self._camera_matrix,
            self._dist_coeffs,
            None,
            self._new_camera_matrix,
            (frame_width, frame_height),
            cv2.CV_16SC2,
        )

        self._initialized = True
        logger.info(
            "📸 CameraUndistorter inicializado: %dx%d, k1=%.3f, k2=%.3f, scale=%.2f/%.2f",
            frame_width, frame_height, cfg.k1, cfg.k2, scale_x, scale_y,
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Corrige la distorsión de un frame. ~2ms por frame."""
        if not self.enabled or not self._initialized:
            return frame
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    def undistort_point(self, x: float, y: float) -> Tuple[float, float]:
        """Transforma un punto de coordenadas distorsionadas a corregidas."""
        if not self.enabled or not self._initialized:
            return (x, y)
        points = np.array([[[x, y]]], dtype=np.float64)
        undistorted = cv2.undistortPoints(
            points,
            self._camera_matrix,
            self._dist_coeffs,
            P=self._new_camera_matrix,
        )
        return (float(undistorted[0][0][0]), float(undistorted[0][0][1]))


# ============================================================
# BALL DETECTOR
# ============================================================

class FailureDetector:
    """Detecta dónde falló el modelo y clasifica el tipo de fallo."""

    def analyze(
        self,
        speed_kmh: float,
        frames_lost: int,
        in_danger_zone: bool,
        audio_peak: bool = False,
        ball_detected: bool = False,
    ) -> dict:
        """
        Retorna tipo de fallo para clasificar frames de entrenamiento.
        """
        failures = []

        # Velocidad físicamente imposible (ya filtrada pero por seguridad).
        if speed_kmh > 120.0:
            failures.append("false_speed")

        # Kalman predice pero YOLO no confirma (frames_lost 1-3).
        if 0 < frames_lost <= 3:
            failures.append("missed_ball")

        # Balón en zona peligro y se pierde.
        if in_danger_zone and frames_lost > 0:
            if "missed_ball" not in failures:
                failures.append("danger_zone_lost")
            else:
                failures.remove("missed_ball")
                failures.append("danger_zone_lost")

        # Audio peak sin balón detectado (posible gol no visto).
        if audio_peak and not ball_detected:
            failures.append("audio_no_ball")

        severity = "high" if any(
            f in ["false_speed", "danger_zone_lost", "audio_no_ball"]
            for f in failures
        ) else "medium"

        return {
            "has_failure": len(failures) > 0,
            "types": failures,
            "primary_type": failures[0] if failures else None,
            "severity": severity,
            "should_export": len(failures) > 0,
        }


class BallDetector:
    """Detecta el balón con YOLO. Auto-detecta la clase correcta."""

    def __init__(self, config: VisionConfig):
        self.config = config
        self.model = None
        self.ball_classes: List[int] = []
        self._previous_sahi_ball_center: Optional[Tuple[float, float]] = None
        export_cfg = self.config.training_frames
        self.frame_exporter = (
            FrameExporter(
                confidence_threshold=export_cfg.confidence_threshold,
                max_frames_per_match=export_cfg.max_frames_per_match,
                jpeg_quality=export_cfg.jpeg_quality,
            )
            if export_cfg.enabled
            else None
        )
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            path = self.config.yolo.model_path
            logger.info(f"👁️ Cargando modelo YOLO: {path}")
            self.model = YOLO(path)

            if self.config.yolo.target_classes is not None:
                self.ball_classes = self.config.yolo.target_classes
                logger.info(f"✅ Clases configuradas manualmente: {self.ball_classes}")
            else:
                self.ball_classes = self._auto_detect_ball_class()
                logger.info(f"✅ Clases auto-detectadas: {self.ball_classes}")

            if not self.ball_classes:
                logger.warning("⚠️ No se encontró clase de balón. Se usarán TODAS las clases.")

        except Exception as exc:
            logger.error(f"❌ Error cargando YOLO: {exc}")
            raise

    def _auto_detect_ball_class(self) -> List[int]:
        """Busca clases que contengan 'ball' en el nombre."""
        if not hasattr(self.model, "names") or not self.model.names:
            return [32]

        found = []
        for idx, name in self.model.names.items():
            name_lower = name.lower().strip()
            if "ball" in name_lower or "balon" in name_lower or "pelota" in name_lower:
                found.append(int(idx))
                logger.info(f"  🎯 Clase [{idx}] '{name}' detectada como balón")

        if not found:
            num_classes = len(self.model.names)
            if num_classes <= 5:
                found = list(range(num_classes))
                logger.info(f"  📋 Modelo custom con {num_classes} clases, usando todas: {found}")
            else:
                found = [32]
                logger.info("  📋 Fallback a clase COCO 32 (sports ball)")

        return found

    def _correct_gopro_distortion(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        # Parámetros genéricos GoPro Wide mode
        # K: matriz intrínseca de cámara
        # D: coeficientes de distorsión (barrel/fisheye)
        K = np.array([
            [w * 0.75,  0,        w / 2],
            [0,         w * 0.75, h / 2],
            [0,         0,        1    ]
        ], dtype=np.float64)
        D = np.array([-0.3, 0.1, 0.0, 0.0], dtype=np.float64)
        corrected = cv2.undistort(frame, K, D)
        return corrected

    def _parse_yolo_results(
        self,
        results,
        x_offset: float = 0.0,
        y_offset: float = 0.0,
    ) -> Tuple[List[Detection], List[Detection]]:
        """Convierte resultados YOLO en coordenadas del frame completo."""
        cfg = self.config.yolo
        ball_detections: List[Detection] = []
        person_detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id
            for box_idx, box in enumerate(result.boxes):
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, "unknown").lower()
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                cx = x_offset + x1 + w / 2
                cy = y_offset + y1 + h / 2
                track_id = None
                if track_ids is not None:
                    try:
                        track_id = int(track_ids[box_idx].item())
                    except Exception:
                        track_id = None

                detection = Detection(
                    x=cx, y=cy, w=w, h=h,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    track_id=track_id,
                )

                if cls_name in ("ball", "sports ball", "sports_ball"):
                    if (
                        min(w, h) >= cfg.min_ball_size_px
                        and max(w, h) <= cfg.max_ball_size_px
                    ):
                        ball_detections.append(detection)
                elif cls_name == "person":
                    person_detections.append(detection)

        return ball_detections, person_detections

    def _detect_full_frame(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        cfg = self.config.yolo
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=cfg.confidence,
            imgsz=cfg.imgsz,
            verbose=False,
        )
        return self._parse_yolo_results(results)

    def _slice_positions(self, full_size: int, slice_size: int, step: int) -> List[int]:
        if full_size <= slice_size:
            return [0]

        positions = list(range(0, max(full_size - slice_size + 1, 1), step))
        final_pos = max(full_size - slice_size, 0)
        if positions[-1] != final_pos:
            positions.append(final_pos)
        return positions

    def _detect_sahi(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        cfg = self.config.yolo
        frame_h, frame_w = frame.shape[:2]
        slice_size = max(64, int(cfg.sahi_slice_size or cfg.imgsz or 640))
        overlap = min(max(float(cfg.sahi_overlap), 0.0), 0.9)
        step = max(1, int(slice_size * (1.0 - overlap)))

        x_positions = self._slice_positions(frame_w, slice_size, step)
        y_positions = self._slice_positions(frame_h, slice_size, step)
        all_ball_detections: List[Detection] = []
        all_person_detections: List[Detection] = []

        for y1 in y_positions:
            for x1 in x_positions:
                x2 = min(x1 + slice_size, frame_w)
                y2 = min(y1 + slice_size, frame_h)
                tile = frame[y1:y2, x1:x2]
                if tile.size == 0:
                    continue

                results = self.model(
                    tile,
                    conf=cfg.confidence,
                    imgsz=cfg.imgsz,
                    verbose=False,
                )
                ball_dets, person_dets = self._parse_yolo_results(
                    results,
                    x_offset=float(x1),
                    y_offset=float(y1),
                )
                all_ball_detections.extend(ball_dets)
                all_person_detections.extend(person_dets)

        ball_detections = self._apply_nms(all_ball_detections, cfg.iou_threshold, cfg.max_det)
        person_detections = self._apply_nms(all_person_detections, cfg.iou_threshold, cfg.max_det)
        return self._filter_sahi_ball_motion(ball_detections), person_detections

    def _filter_sahi_ball_motion(self, detections: List[Detection]) -> List[Detection]:
        if not detections:
            return []

        previous_center = self._previous_sahi_ball_center
        best_detection = max(detections, key=lambda det: det.confidence)
        self._previous_sahi_ball_center = (best_detection.x, best_detection.y)

        if previous_center is None:
            return detections

        min_movement = max(0.0, float(getattr(self.config.yolo, "sahi_min_movement_px", 30)))
        if min_movement <= 0:
            return detections

        filtered = [
            det
            for det in detections
            if math.hypot(det.x - previous_center[0], det.y - previous_center[1]) > min_movement
        ]
        if detections and not filtered:
            logger.debug(
                "🔎 SAHI filtro movimiento: %s detecciones descartadas (< %.1fpx)",
                len(detections),
                min_movement,
            )

        return filtered

    def _apply_nms(
        self,
        detections: List[Detection],
        iou_threshold: float,
        max_det: int,
    ) -> List[Detection]:
        if not detections:
            return []

        kept: List[Detection] = []
        by_class: Dict[int, List[Detection]] = {}
        for det in detections:
            by_class.setdefault(det.class_id, []).append(det)

        for class_detections in by_class.values():
            candidates = sorted(
                class_detections,
                key=lambda det: det.confidence,
                reverse=True,
            )
            while candidates:
                current = candidates.pop(0)
                kept.append(current)
                if max_det and len(kept) >= max_det:
                    break
                candidates = [
                    det
                    for det in candidates
                    if self._bbox_iou(current, det) <= iou_threshold
                ]

        kept.sort(key=lambda det: det.confidence, reverse=True)
        return kept[:max_det] if max_det else kept

    def _bbox_iou(self, first: Detection, second: Detection) -> float:
        ax1 = first.x - first.w / 2
        ay1 = first.y - first.h / 2
        ax2 = first.x + first.w / 2
        ay2 = first.y + first.h / 2
        bx1 = second.x - second.w / 2
        by1 = second.y - second.h / 2
        bx2 = second.x + second.w / 2
        by2 = second.y + second.h / 2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def detect(
        self,
        frame: np.ndarray,
        use_sahi: bool = False,
        match_id: str = "unknown",
        frame_num: Optional[int] = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        """
        Detecta balón y opcionalmente personas.
        Retorna: (ball_detections, person_detections)
        """
        if self.model is None:
            return [], []

        if use_sahi:
            ball_detections, person_detections = self._detect_sahi(frame)
        else:
            ball_detections, person_detections = self._detect_full_frame(frame)

        ball_detections.sort(key=lambda det: det.confidence, reverse=True)
        person_detections.sort(key=lambda det: det.confidence, reverse=True)
        return ball_detections, person_detections


class PlayerDetector:
    def __init__(self, config: VisionConfig):
        self.config = config
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            import os
            if os.path.exists("/app/yolo11n.pt"):
                model_path = "/app/yolo11n.pt"
            elif os.path.exists("yolo11n.pt"):
                model_path = "yolo11n.pt"
            else:
                logger.warning("⚠️ PlayerDetector no disponible: yolo11n.pt no existe localmente")
                self.model = None
                return
            self.model = YOLO(model_path)
            logger.info("✅ PlayerDetector cargado (yolo11n COCO)")
        except Exception as e:
            logger.warning(f"⚠️ PlayerDetector no disponible: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        cfg = self.config.player_detection
        results = self.model.predict(
            source=frame,
            conf=cfg.confidence,
            classes=[0],
            verbose=False,
        )
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                size = max(w, h)
                if size < cfg.min_player_size_px or size > cfg.max_player_size_px:
                    continue
                detections.append(Detection(
                    x=x1 + w / 2, y=y1 + h / 2,
                    w=w, h=h,
                    confidence=float(box.conf[0].item()),
                    class_id=0,
                ))
        return detections

    def count_in_zone(
        self,
        detections: List[Detection],
        danger_polygon,
        roi_polygon,
    ) -> int:
        if not detections or danger_polygon is None:
            return 0
        count = 0
        for det in detections:
            if self._point_in_polygon(det.x, det.y, danger_polygon):
                count += 1
        return count

    def _point_in_polygon(self, x, y, polygon) -> bool:
        from shapely.geometry import Point
        return polygon.contains(Point(x, y))


# ============================================================
# BALL TRACKER — Kalman Filter + Candidate Scoring
# ============================================================

class BallTracker:
    """
    Tracking robusto con Kalman Filter.
    - Predicción cuando YOLO pierde el balón (hasta 1.5s)
    - Max pixel jump ADAPTATIVO
    - GK kick filtering
    - Registro de candidatos rechazados
    """

    def __init__(self, config: VisionConfig):
        self.config = config
        self.kf = None
        self.is_initialized = False
        self.frames_since_detection = 0
        self.confirmed_frames = 0
        self.recent_speeds: List[float] = []
        self.last_position: Optional[Tuple[float, float]] = None
        self.last_timestamp_sec: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self.rejected_candidates: List[RejectedCandidate] = []
        self.fps = 30.0
        self._danger_center: Optional[Tuple[float, float]] = None

    def set_fps(self, fps: float):
        self.fps = max(1.0, fps)

    def set_danger_center(self, center: Optional[Tuple[float, float]]):
        self._danger_center = center

    def _set_kalman_transition(self, dt: float):
        if self.kf is None:
            return

        safe_dt = max(float(dt), 1.0 / max(self.fps, 1.0), 1e-3)
        self.kf.F = np.array(
            [
                [1, 0, safe_dt, 0],
                [0, 1, 0, safe_dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        q = self.config.kalman.process_noise
        dt2 = safe_dt * safe_dt
        dt3 = dt2 * safe_dt
        dt4 = dt2 * dt2
        self.kf.Q = q * np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=float,
        )

    def _init_kalman(self, x: float, y: float):
        if not HAS_FILTERPY:
            self.last_position = (x, y)
            self.last_timestamp_sec = None
            self._last_update_time = None
            self.is_initialized = True
            return

        kf = FilterPyKF(dim_x=4, dim_z=2)
        kf.x = np.array([x, y, 0.0, 0.0])
        kf.H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=float,
        )
        r = self.config.kalman.measurement_noise
        kf.R = np.array([[r, 0], [0, r]], dtype=float)
        kf.P *= 500.0

        self.kf = kf
        self._set_kalman_transition(1.0 / max(self.fps, 1.0))
        self.is_initialized = True
        self.last_position = (x, y)
        self.last_timestamp_sec = None
        self._last_update_time = None
        self.frames_since_detection = 0
        self.confirmed_frames = 1

    def _predict(self, dt: float) -> Tuple[float, float]:
        if self.kf is not None:
            self._set_kalman_transition(dt)
            self.kf.predict()
            return float(self.kf.x[0]), float(self.kf.x[1])
        return self.last_position if self.last_position else (0.0, 0.0)

    def _get_max_jump(self) -> float:
        cfg = self.config.tracking
        base = cfg.base_max_pixel_jump
        if self.recent_speeds:
            avg = float(np.mean(self.recent_speeds[-10:]))
            adaptive = avg * cfg.adaptive_jump_factor
            computed = max(base, adaptive)
        else:
            computed = base
        return min(computed, cfg.absolute_max_pixel_jump)

    def _score_candidate(
        self,
        det: Detection,
        pred_x: float,
        pred_y: float,
        max_jump: float,
    ) -> float:
        dist = math.hypot(det.x - pred_x, det.y - pred_y)
        if dist > max_jump:
            return float("inf")
        cfg = self.config.tracking
        norm_dist = dist / max(max_jump, 1.0)
        norm_conf = 1.0 - det.confidence
        return (cfg.distance_weight * norm_dist) + (cfg.confidence_weight * norm_conf)

    def _check_gk_kick(self, y: float, speed: float, frame_h: int) -> bool:
        if not self.config.tracking.filter_gk_kicks:
            return False
        zone_y = frame_h * self.config.tracking.gk_kick_zone_bottom_pct
        return y > zone_y and speed > self.config.tracking.gk_kick_min_speed

    def _calc_direction(self, x: float, y: float) -> Optional[float]:
        if not self.last_position:
            return None
        dx = x - self.last_position[0]
        dy = y - self.last_position[1]
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        return round(math.degrees(math.atan2(dy, dx)), 2)

    def _calc_dist_to_danger(self, x: float, y: float) -> Optional[float]:
        if not self._danger_center:
            return None
        return round(math.hypot(x - self._danger_center[0], y - self._danger_center[1]), 2)

    def _calc_approach_angle(self, x: float, y: float) -> Optional[float]:
        if self._danger_center is None or self.last_position is None:
            return None
        move_dx = x - self.last_position[0]
        move_dy = y - self.last_position[1]
        move_mag = math.hypot(move_dx, move_dy)
        if move_mag < 1.0:
            return None
        target_dx = self._danger_center[0] - x
        target_dy = self._danger_center[1] - y
        target_mag = math.hypot(target_dx, target_dy)
        if target_mag < 1.0:
            return 0.0
        dot = (move_dx * target_dx + move_dy * target_dy) / (move_mag * target_mag)
        dot = max(-1.0, min(1.0, dot))
        return round(math.degrees(math.acos(dot)), 1)

    def reset(self):
        self.kf = None
        self.is_initialized = False
        self.frames_since_detection = 0
        self.confirmed_frames = 0
        self.recent_speeds.clear()
        self.last_position = None
        self.last_timestamp_sec = None
        self._last_update_time = None

    def update(
        self,
        detections: List[Detection],
        frame_idx: int,
        timestamp_sec: float,
        frame_height: int,
    ) -> Optional[TrackPoint]:
        if not self.is_initialized:
            if detections:
                best = detections[0]
                self._init_kalman(best.x, best.y)
                self.last_timestamp_sec = timestamp_sec
                self._last_update_time = timestamp_sec
                return TrackPoint(
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    x=best.x,
                    y=best.y,
                    speed_px=0.0,
                    speed_px_per_sec=0.0,
                    speed_kmh=0.0,
                    confidence=best.confidence,
                    is_detected=True,
                    direction_deg=None,
                    approach_angle_deg=None,
                    dist_to_danger_center=self._calc_dist_to_danger(best.x, best.y),
                    track_id=best.track_id,
                )
            return None

        if self._last_update_time is not None:
            dt = max(timestamp_sec - self._last_update_time, 1e-3)
        else:
            dt = 1.0 / max(self.fps, 1.0)

        pred_x, pred_y = self._predict(dt)
        self._last_update_time = timestamp_sec
        max_jump = self._get_max_jump()

        best_det = None
        best_score = float("inf")

        for det in detections:
            dist = math.hypot(det.x - pred_x, det.y - pred_y)
            if self.confirmed_frames > 5 and dist > max_jump:
                self.rejected_candidates.append(
                    RejectedCandidate(
                        frame=frame_idx,
                        timestamp=round(timestamp_sec, 3),
                        conf=round(det.confidence, 3),
                        dist_px=round(dist, 2),
                        max_jump=round(max_jump, 2),
                    )
                )
                continue

            score = self._score_candidate(det, pred_x, pred_y, max_jump)
            if score < best_score:
                best_score = score
                best_det = det

            if dist > max_jump:
                self.rejected_candidates.append(
                    RejectedCandidate(
                        frame=frame_idx,
                        timestamp=round(timestamp_sec, 3),
                        conf=round(det.confidence, 3),
                        dist_px=round(dist, 2),
                        max_jump=round(max_jump, 2),
                    )
                )

        if best_det is not None and best_score < float("inf"):
            mx, my = best_det.x, best_det.y
            if self.kf is not None:
                self.kf.update(np.array([mx, my]))

            step_distance = 0.0
            speed_px_per_sec = 0.0
            if self.last_position and self.last_timestamp_sec is not None:
                step_distance = math.hypot(
                    mx - self.last_position[0],
                    my - self.last_position[1],
                )
                real_dt = max(timestamp_sec - self.last_timestamp_sec, 1e-3)
                speed_px_per_sec = step_distance / real_dt

            direction_deg = self._calc_direction(mx, my)
            approach_angle = self._calc_approach_angle(mx, my)
            self.last_position = (mx, my)
            self.last_timestamp_sec = timestamp_sec
            self.frames_since_detection = 0
            self.confirmed_frames += 1
            self.recent_speeds.append(step_distance)
            if len(self.recent_speeds) > 90:
                self.recent_speeds = self.recent_speeds[-90:]

            is_gk = self._check_gk_kick(my, speed_px_per_sec, frame_height)

            return TrackPoint(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                x=mx,
                y=my,
                speed_px=step_distance,
                speed_px_per_sec=speed_px_per_sec,
                speed_kmh=0.0,
                confidence=best_det.confidence,
                is_detected=True,
                is_gk_kick=is_gk,
                direction_deg=direction_deg,
                approach_angle_deg=approach_angle,
                dist_to_danger_center=self._calc_dist_to_danger(mx, my),
                track_id=best_det.track_id,
            )

        self.frames_since_detection += 1

        max_frames_lost = int(
            getattr(
                self.config.tracking,
                "max_frames_lost",
                self.config.kalman.max_prediction_frames,
            )
        )
        max_frames_lost = max(max_frames_lost, self.config.kalman.confirm_after_frames)

        if self.frames_since_detection > max_frames_lost:
            self.reset()
            return None

        if self.confirmed_frames < self.config.kalman.confirm_after_frames:
            self.reset()
            return None

        step_distance = 0.0
        speed_px_per_sec = 0.0
        if self.last_position and self.last_timestamp_sec is not None:
            step_distance = math.hypot(
                pred_x - self.last_position[0],
                pred_y - self.last_position[1],
            )
            real_dt = max(timestamp_sec - self.last_timestamp_sec, 1e-3)
            speed_px_per_sec = step_distance / real_dt
        direction_deg = self._calc_direction(pred_x, pred_y)
        approach_angle = self._calc_approach_angle(pred_x, pred_y)
        self.last_position = (pred_x, pred_y)
        self.last_timestamp_sec = timestamp_sec

        return TrackPoint(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            x=pred_x,
            y=pred_y,
            speed_px=step_distance,
            speed_px_per_sec=speed_px_per_sec,
            speed_kmh=0.0,
            confidence=0.0,
            is_detected=False,
            direction_deg=direction_deg,
            approach_angle_deg=approach_angle,
            dist_to_danger_center=self._calc_dist_to_danger(pred_x, pred_y),
        )


# ============================================================
# AUDIO ANALYZER — STA/LTA
# ============================================================

class AudioAnalyzer:
    """Análisis de audio STA/LTA para detectar picos (gritos, silbatos, goles)."""

    def __init__(self, config: VisionConfig):
        self.config = config

    def analyze(self, video_path: str) -> Tuple[List[float], List[float]]:
        audio_data, sample_rate = self._extract_audio(video_path)
        if audio_data is None or len(audio_data) == 0:
            logger.warning("⚠️ No se pudo extraer audio del video")
            return [], []

        if self.config.audio.normalize:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

        energy_per_sec = self._compute_energy_per_second(audio_data, sample_rate)
        peaks = self._sta_lta(audio_data, sample_rate)
        return peaks, energy_per_sec

    def _extract_audio(self, video_path: str) -> Tuple[Optional[np.ndarray], int]:
        target_sr = 16000
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_file.close()
        tmp_wav = temp_file.name

        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(target_sr),
                "-ac",
                "1",
                "-loglevel",
                "error",
                tmp_wav,
            ]
            subprocess.run(cmd, capture_output=True, timeout=120, check=True)

            sample_rate, audio = wavfile.read(tmp_wav)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            if np.issubdtype(audio.dtype, np.integer):
                info = np.iinfo(audio.dtype)
                scale = float(max(abs(info.min), info.max))
                audio = audio.astype(np.float32) / max(scale, 1.0)
            else:
                audio = audio.astype(np.float32)

            return audio, int(sample_rate)
        except Exception as exc:
            logger.warning(f"⚠️ ffmpeg audio extraction failed: {exc}")
            return None, target_sr
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

    def _compute_energy_per_second(self, audio: np.ndarray, sample_rate: int) -> List[float]:
        samples_per_sec = sample_rate
        total_seconds = len(audio) // samples_per_sec
        energy = []
        for sec in range(total_seconds):
            start = sec * samples_per_sec
            end = start + samples_per_sec
            chunk = audio[start:end]
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            energy.append(rms)
        if energy:
            max_e = max(energy) if max(energy) > 0 else 1.0
            energy = [round((value / max_e) * 2.0, 3) for value in energy]
        return energy

    def _sta_lta(self, audio: np.ndarray, sample_rate: int) -> List[float]:
        cfg = self.config.audio
        sta_samples = int(cfg.sta_window_sec * sample_rate)
        lta_samples = int(cfg.lta_window_sec * sample_rate)
        min_sep_samples = int(cfg.min_peak_separation_sec * sample_rate)

        if len(audio) < lta_samples + sta_samples:
            return []

        energy = audio ** 2
        peaks = []
        last_peak_sample = -min_sep_samples * 2

        for i in range(lta_samples, len(energy) - sta_samples):
            sta_start = i
            sta_end = i + sta_samples
            if sta_end > len(energy):
                break
            sta_val = np.mean(energy[sta_start:sta_end])
            lta_start = max(0, i - lta_samples)
            lta_val = np.mean(energy[lta_start:i])
            if lta_val > 0:
                ratio = sta_val / lta_val
                if ratio > cfg.trigger_ratio and (i - last_peak_sample) > min_sep_samples:
                    peak_sec = round(i / sample_rate, 3)
                    peaks.append(peak_sec)
                    last_peak_sample = i
        return peaks


# ============================================================
# AUDIO PRE-SCANNER — Ventanas candidatas por audio
# ============================================================

class AudioPreScanner:
    """Pre-escaneo de audio para generar ventanas candidatas antes de correr YOLO."""

    def __init__(
        self,
        config: VisionConfig,
        window_margin_sec: float = 20.0,
        min_peak_score: float = 0.6,
        merge_gap_sec: float = 10.0,
    ):
        self.config = config
        self.window_margin_sec = window_margin_sec
        self.min_peak_score = min_peak_score  # reservado: STA/LTA actual no devuelve score
        self.merge_gap_sec = merge_gap_sec
        self._audio = AudioAnalyzer(config)

    def scan(self, video_path: str) -> List[Dict]:
        audio_data, sample_rate = self._audio._extract_audio(video_path)
        if audio_data is None or len(audio_data) == 0:
            return []

        total_duration_sec = len(audio_data) / sample_rate

        if self.config.audio.normalize:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

        peaks = self._audio._sta_lta(audio_data, sample_rate)
        if not peaks:
            return []

        windows = []
        for peak_sec in peaks:
            start = max(0.0, peak_sec - self.window_margin_sec)
            end = min(total_duration_sec, peak_sec + self.window_margin_sec)
            windows.append({"start": start, "end": end, "peak_score": 1.0, "peak_sec": peak_sec})

        windows.sort(key=lambda w: w["start"])
        merged: List[Dict] = []
        for w in windows:
            if merged and w["start"] <= merged[-1]["end"] + self.merge_gap_sec:
                merged[-1]["end"] = max(merged[-1]["end"], w["end"])
                if w["peak_score"] > merged[-1]["peak_score"]:
                    merged[-1]["peak_score"] = w["peak_score"]
                    merged[-1]["peak_sec"] = w["peak_sec"]
            else:
                merged.append(dict(w))

        return merged

    def get_coverage_pct(self, windows: List[Dict], total_sec: float) -> float:
        if not windows or total_sec <= 0:
            return 0.0
        covered = sum(w["end"] - w["start"] for w in windows)
        return round(min(100.0, covered / total_sec * 100), 1)


# ============================================================
# VISION ENGINE — Orquestador principal
# ============================================================

class VisionEngine:
    def __init__(self, config: VisionConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.device = os.getenv("RFDETR_DEVICE", "cuda")
        self.yolo_detector = BallDetector(self.config)
        self.detector = DetectorFactory.create(
            config=self.config,
            yolo_detector=self.yolo_detector,
            detection_factory=Detection,
            device=self.device,
        )
        self.dual_detector = None
        self.sam3 = None
        logger.info(
            "🧪 Detector shadow activo: mode=%s; SAM3/CoTracker desactivados.",
            getattr(self.config, "detector_mode", "yolo_primary"),
        )
        self.player_detector = TeamPlayerDetector(self.config.player_detection)
        self.tracker = BallTracker(self.config)
        self.audio_analyzer = AudioAnalyzer(self.config)
        self.undistorter = CameraUndistorter(self.config.camera)
        self.homography = HomographyTransformer()
        self.failure_detector = FailureDetector()
        self.goal_detector = GoalDetector(
            frame_height_px=self.frame_height if hasattr(self, "frame_height") else 720
        )
        logger.info("✅ GoalDetector inicializado")

    def process_video(
        self,
        video_path: str,
        roi_points=None,
        danger_zone=None,
        match_id: str = "unknown",
        candidate_windows: Optional[List[Dict]] = None,
        debug_video: bool = False,
    ) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"🎬 Procesando video: {video_path}")
        logger.info(f"🎯 Match ID: {match_id}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        total_duration_sec = total_frames / fps if fps > 0 else 0.0

        logger.info(
            f"📐 Video: {frame_w}x{frame_h} @ {fps:.1f}fps, "
            f"{total_frames} frames, {total_duration_sec:.1f}s"
        )

        self.tracker.set_fps(fps)
        self.undistorter.initialize(frame_w, frame_h)

        roi_polygon = self._make_polygon(roi_points)
        danger_polygon = self._make_polygon(danger_zone)
        danger_center = self._compute_polygon_center(danger_polygon, danger_zone)
        previous_goalpost_left = getattr(self.homography, "goalpost_left_px", None)
        previous_goalpost_right = getattr(self.homography, "goalpost_right_px", None)
        self.homography = HomographyTransformer()
        self.homography.calibrate_from_roi(roi_points or [])
        self.goal_detector.frame_height_px = frame_h
        self.goal_detector.reset()
        if previous_goalpost_left and previous_goalpost_right:
            self.homography.goalpost_left_px = previous_goalpost_left
            self.homography.goalpost_right_px = previous_goalpost_right
        if (
            hasattr(self.homography, "goalpost_left_px")
            and self.homography.goalpost_left_px
            and self.homography.goalpost_right_px
        ):
            self.goal_detector.calibrate(
                goalpost_left_px=self.homography.goalpost_left_px,
                goalpost_right_px=self.homography.goalpost_right_px,
            )
        self.tracker.set_danger_center(danger_center)

        if roi_polygon:
            logger.info(f"✅ ROI polygon: {len(roi_points)} puntos")
        if danger_polygon:
            logger.info(f"✅ Danger zone polygon: {len(danger_zone)} puntos")
        if danger_center:
            logger.info(
                "🎯 Centro danger zone: (%.1f, %.1f)",
                danger_center[0],
                danger_center[1],
            )
        if self.config.yolo.use_sahi:
            logger.info(
                "🔎 SAHI activado: slices=%d overlap=%.2f",
                self.config.yolo.sahi_slice_size,
                self.config.yolo.sahi_overlap,
            )

        track_points: List[TrackPoint] = []
        ball_positions_history: List[Tuple[float, float]] = []
        ball_timestamps_history: List[float] = []
        goal_events: List[Dict[str, Any]] = []
        debug_records: Dict[int, Dict[str, Any]] = {}
        person_counts_by_sec: Dict[int, Dict[str, Any]] = {}
        player_tracks: Dict[int, Dict[str, Any]] = {}
        player_team_by_id: Dict[int, str] = {}
        frame_idx = 0
        process_every = self.config.tracking.process_every_n_frames
        frame_skip = max(1, int(getattr(self.config.yolo, "frame_skip", 1) or 1))
        player_detection_interval = max(
            1,
            int(getattr(self.config.player_detection, "detection_interval", 15) or 15),
        )
        latest_players_in_zone = 0
        frames_with_detection = 0
        missed_detection_exports = 0
        processed_count = 0
        log_interval = max(1, total_frames // 20)
        estimated_processed_count = max(1, (total_frames + frame_skip - 1) // frame_skip)

        logger.info(
            "📹 Procesando %s frames (de %s totales, skip=%s) a %.1ffps efectivo",
            estimated_processed_count,
            total_frames,
            frame_skip,
            round(fps / max(frame_skip, 1), 1),
        )
        logger.info("🔊 Analizando audio...")
        audio_peaks, audio_energy = self.audio_analyzer.analyze(video_path)
        logger.info(f"✅ Audio: {len(audio_peaks)} picos detectados")
        logger.info("🔄 Iniciando tracking frame por frame...")

        if candidate_windows is None:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                should_process = True
                if frame_skip > 1 and frame_idx % frame_skip != 0:
                    should_process = False
                if process_every > 1 and frame_idx % process_every != 0:
                    should_process = False

                if should_process:
                    processed_count += 1
                    timestamp_sec = frame_idx / fps
                    frame = self.undistorter.undistort(frame)
                    ball_dets, person_dets = self._detect_ball_candidates(
                        frame,
                        use_sahi=self.config.yolo.use_sahi,
                        match_id=match_id,
                        frame_num=frame_idx,
                    )
                    detections = ball_dets
                    sec = int(timestamp_sec)

                    players = []
                    players_in_zone = latest_players_in_zone
                    if self.config.player_detection.enabled:
                        players = self.player_detector.detect_players(frame)
                        players = self._filter_players_by_polygon(players, roi_polygon)
                        if frame_idx % 30 == 0 and self.sam3 and self.sam3.is_available:
                            team_context = self._analyze_teams_with_sam3(frame, players, danger_polygon)
                            self._update_player_team_assignments(team_context, player_team_by_id)
                            previous_context = person_counts_by_sec.get(sec, {})
                            if team_context.get("total_players", 0) >= previous_context.get("total_players", 0):
                                person_counts_by_sec[sec] = team_context
                            players_in_zone = int(team_context.get("players_in_danger_zone", 0) or 0)
                            latest_players_in_zone = players_in_zone
                        elif frame_idx % player_detection_interval == 0:
                            team_context = self.player_detector.analyze_players(
                                players,
                                danger_polygon=danger_polygon,
                            )
                            self._update_player_team_assignments(team_context, player_team_by_id)
                            previous_context = person_counts_by_sec.get(sec, {})
                            if team_context.get("total_players", 0) >= previous_context.get("total_players", 0):
                                person_counts_by_sec[sec] = team_context
                            players_in_zone = int(team_context.get("players_in_danger_zone", 0) or 0)
                            latest_players_in_zone = players_in_zone
                        self._update_player_tracks(
                            player_tracks,
                            players,
                            player_team_by_id,
                            frame_idx,
                            fps,
                            self.homography,
                        )
                        players = self._players_with_team(players, player_team_by_id)

                    if roi_polygon and detections:
                        detections = [
                            det
                            for det in detections
                            if self._point_in_polygon(det.x, det.y, roi_polygon)
                        ]

                    tp = self.tracker.update(detections, frame_idx, timestamp_sec, frame_h)
                    if tp is not None:
                        tp.speed_kmh = round(self.homography.speed_px_to_kmh(tp.speed_px_per_sec), 2)
                        tp.players_in_danger_zone = players_in_zone
                        track_points.append(tp)
                        in_danger_zone = bool(
                            danger_polygon and self._point_in_polygon(tp.x, tp.y, danger_polygon)
                        )
                        has_audio_peak = self._has_audio_peak_near(
                            audio_peaks,
                            timestamp_sec,
                            self.config.scoring.audio_search_window_sec,
                        )
                        best_confidence = detections[0].confidence if detections else tp.confidence
                        goal_event = self.goal_detector.update(
                            ball_x=tp.x if tp.is_detected else None,
                            ball_y=tp.y if tp.is_detected else None,
                            timestamp_s=timestamp_sec,
                            audio_peak=has_audio_peak,
                            ball_confidence=best_confidence if tp.is_detected else 0.0,
                        )
                        if goal_event:
                            logger.info(f"⚽ Clip de gol automático en t={timestamp_sec:.1f}s")
                            goal_events.append(goal_event)
                            if hasattr(self, "current_clip_metadata"):
                                self.current_clip_metadata["auto_verdict"] = "probable_goal"
                                self.current_clip_metadata["score"] = 1.0
                                self.current_clip_metadata["goal_confidence"] = goal_event["confidence"]
                        if tp.is_detected:
                            frames_with_detection += 1
                            if detections and best_confidence > 0.3:
                                ball_positions_history.append((detections[0].x, detections[0].y))
                                ball_timestamps_history.append(timestamp_sec)
                            self._export_low_confidence_frame(
                                frame=frame,
                                match_id=match_id,
                                frame_idx=frame_idx,
                                track_point=tp,
                                best_confidence=best_confidence,
                                in_danger_zone=in_danger_zone,
                                has_audio_peak=has_audio_peak,
                            )
                        else:
                            missed_detection_exports += self._export_missed_detection_frame(
                                frame=frame,
                                match_id=match_id,
                                frame_idx=frame_idx,
                                track_point=tp,
                                current_count=missed_detection_exports,
                                in_danger_zone=in_danger_zone,
                                audio_peak=has_audio_peak,
                            )
                    if debug_video:
                        debug_records[frame_idx] = {
                            "timestamp_sec": timestamp_sec,
                            "detections": list(detections),
                            "track_point": tp,
                            "players": players,
                        }

                frame_idx += 1

                if frame_idx % log_interval == 0:
                    pct = round(frame_idx / max(total_frames, 1) * 100)
                    logger.info(f"  📹 Progreso: {pct}% ({frame_idx}/{total_frames})")

        else:
            WINDOW_FRAME_SKIP = 3
            logger.info(
                "🪟 Modo ventanas: %d segmentos, frame_skip=%d",
                len(candidate_windows), WINDOW_FRAME_SKIP,
            )
            for win_idx, window in enumerate(candidate_windows):
                cap.set(cv2.CAP_PROP_POS_MSEC, window["start"] * 1000)
                local_idx = 0
                logger.info(
                    "  🪟 Ventana %d/%d: %.1fs → %.1fs",
                    win_idx + 1, len(candidate_windows),
                    window["start"], window["end"],
                )
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    abs_frame_idx = int(timestamp_sec * fps)
                    if timestamp_sec > window["end"]:
                        break
                    if local_idx % WINDOW_FRAME_SKIP == 0:
                        processed_count += 1
                        frame = self.undistorter.undistort(frame)
                        ball_dets, person_dets = self._detect_ball_candidates(
                            frame,
                            use_sahi=self.config.yolo.use_sahi,
                            match_id=match_id,
                            frame_num=abs_frame_idx,
                        )
                        detections = ball_dets
                        sec = int(timestamp_sec)

                        players = []
                        players_in_zone = latest_players_in_zone
                        if self.config.player_detection.enabled:
                            players = self.player_detector.detect_players(frame)
                            players = self._filter_players_by_polygon(players, roi_polygon)
                            if abs_frame_idx % 30 == 0 and self.sam3 and self.sam3.is_available:
                                team_context = self._analyze_teams_with_sam3(frame, players, danger_polygon)
                                self._update_player_team_assignments(team_context, player_team_by_id)
                                previous_context = person_counts_by_sec.get(sec, {})
                                if team_context.get("total_players", 0) >= previous_context.get("total_players", 0):
                                    person_counts_by_sec[sec] = team_context
                                players_in_zone = int(team_context.get("players_in_danger_zone", 0) or 0)
                                latest_players_in_zone = players_in_zone
                            elif abs_frame_idx % player_detection_interval == 0:
                                team_context = self.player_detector.analyze_players(
                                    players,
                                    danger_polygon=danger_polygon,
                                )
                                self._update_player_team_assignments(team_context, player_team_by_id)
                                previous_context = person_counts_by_sec.get(sec, {})
                                if team_context.get("total_players", 0) >= previous_context.get("total_players", 0):
                                    person_counts_by_sec[sec] = team_context
                                players_in_zone = int(team_context.get("players_in_danger_zone", 0) or 0)
                                latest_players_in_zone = players_in_zone
                            self._update_player_tracks(
                                player_tracks,
                                players,
                                player_team_by_id,
                                abs_frame_idx,
                                fps,
                                self.homography,
                            )
                            players = self._players_with_team(players, player_team_by_id)

                        if roi_polygon and detections:
                            detections = [
                                det for det in detections
                                if self._point_in_polygon(det.x, det.y, roi_polygon)
                            ]

                        tp = self.tracker.update(detections, abs_frame_idx, timestamp_sec, frame_h)
                        if tp is not None:
                            tp.speed_kmh = round(self.homography.speed_px_to_kmh(tp.speed_px_per_sec), 2)
                            tp.players_in_danger_zone = players_in_zone
                            track_points.append(tp)
                            in_danger_zone = bool(
                                danger_polygon and self._point_in_polygon(tp.x, tp.y, danger_polygon)
                            )
                            has_audio_peak = self._has_audio_peak_near(
                                audio_peaks,
                                timestamp_sec,
                                self.config.scoring.audio_search_window_sec,
                            )
                            best_confidence = detections[0].confidence if detections else tp.confidence
                            goal_event = self.goal_detector.update(
                                ball_x=tp.x if tp.is_detected else None,
                                ball_y=tp.y if tp.is_detected else None,
                                timestamp_s=timestamp_sec,
                                audio_peak=has_audio_peak,
                                ball_confidence=best_confidence if tp.is_detected else 0.0,
                            )
                            if goal_event:
                                logger.info(f"⚽ Clip de gol automático en t={timestamp_sec:.1f}s")
                                goal_events.append(goal_event)
                                if hasattr(self, "current_clip_metadata"):
                                    self.current_clip_metadata["auto_verdict"] = "probable_goal"
                                    self.current_clip_metadata["score"] = 1.0
                                    self.current_clip_metadata["goal_confidence"] = goal_event["confidence"]
                            if tp.is_detected:
                                frames_with_detection += 1
                                if detections and best_confidence > 0.3:
                                    ball_positions_history.append((detections[0].x, detections[0].y))
                                    ball_timestamps_history.append(timestamp_sec)
                                self._export_low_confidence_frame(
                                    frame=frame,
                                    match_id=match_id,
                                    frame_idx=abs_frame_idx,
                                    track_point=tp,
                                    best_confidence=best_confidence,
                                    in_danger_zone=in_danger_zone,
                                    has_audio_peak=has_audio_peak,
                                )
                            else:
                                missed_detection_exports += self._export_missed_detection_frame(
                                    frame=frame,
                                    match_id=match_id,
                                    frame_idx=abs_frame_idx,
                                    track_point=tp,
                                    current_count=missed_detection_exports,
                                    in_danger_zone=in_danger_zone,
                                    audio_peak=has_audio_peak,
                                )
                        if debug_video:
                            debug_records[abs_frame_idx] = {
                                "timestamp_sec": timestamp_sec,
                                "detections": list(detections),
                                "track_point": tp,
                                "players": players,
                            }
                    local_idx += 1

        cap.release()
        logger.info(
            f"✅ Tracking completado: {len(track_points)} puntos, "
            f"{frames_with_detection} con detección de balón"
        )
        if hasattr(self.detector, "get_stats"):
            logger.info("📊 Detector stats: %s", self.detector.get_stats())
        if missed_detection_exports:
            logger.info(
                "🧪 Missed detections exportados: %d frames para reentrenamiento",
                missed_detection_exports,
            )

        telemetry = self._build_telemetry(
            track_points,
            audio_energy,
            danger_polygon,
            fps,
            total_duration_sec,
        )

        detection_by_minute = self._build_detection_by_minute(
            track_points,
            fps,
            total_duration_sec,
        )

        detected_seconds = set()
        for tp in track_points:
            if tp.is_detected:
                detected_seconds.add(int(tp.timestamp_sec))
        ball_detected_sec = len(detected_seconds)

        debug_video_path = None
        if debug_video:
            try:
                game_states = self._classify_debug_game_state(track_points)
                debug_video_path = self._write_debug_video(
                    video_path=video_path,
                    output_path="/tmp/debug_video.mp4",
                    debug_records=debug_records,
                    game_states=game_states,
                    roi_points=roi_points,
                    danger_zone=danger_zone,
                    fps=fps,
                    frame_w=frame_w,
                    frame_h=frame_h,
                )
                logger.info("🧪 Debug video generado: %s", debug_video_path)
            except Exception as exc:
                logger.warning("⚠️ No se pudo generar debug video: %s", exc)

        rejected = self.tracker.rejected_candidates
        rejected.sort(key=lambda row: row.conf, reverse=True)
        rejected_export = [
            {
                "frame": row.frame,
                "timestamp": row.timestamp,
                "conf": row.conf,
                "dist_px": row.dist_px,
                "max_jump": row.max_jump,
            }
            for row in rejected[:200]
        ]

        processing_time = round(time.time() - start_time, 2)
        max_speed = max((tp.speed_px_per_sec for tp in track_points), default=0.0)
        max_speed_kmh = max((tp.speed_kmh for tp in track_points), default=0.0)
        avg_speed_kmh = (
            float(np.mean([tp.speed_kmh for tp in track_points if tp.speed_kmh > 0]))
            if track_points
            else 0.0
        )
        trajectory_outliers_count = 0
        if hasattr(self.homography, "validate_trajectory"):
            trajectory_stats = self.homography.validate_trajectory(
                positions=ball_positions_history,
                timestamps=ball_timestamps_history,
            )
            max_speed_kmh = trajectory_stats["max_speed_kmh"]
            avg_speed_kmh = trajectory_stats["avg_speed_kmh"]
            trajectory_outliers_count = len(trajectory_stats["outliers"])
            logger.info(
                "📊 Trayectoria balón: max=%.1f km/h, avg=%.1f km/h, outliers descartados=%d",
                max_speed_kmh,
                avg_speed_kmh,
                trajectory_outliers_count,
            )
        logger.info(
            "⏱️ Procesamiento completado en %.1fs (%.1f min) — video: %.1fs, "
            "frames: %s (skip=%s), fps_efectivo: %.1f, clips: %s",
            processing_time,
            processing_time / 60,
            total_duration_sec,
            processed_count,
            frame_skip,
            processed_count / max(processing_time, 1e-3),
            0,
        )

        return {
            "status": "success",
            "match_id": match_id,
            "message": (
                "Processed successfully. Tracking: "
                f"{ball_detected_sec}s detected of {int(total_duration_sec)}s total."
            ),
            "telemetry": telemetry,
            "detection_by_minute": detection_by_minute,
            "rejected_candidates": rejected_export,
            "audio_peaks_sec": audio_peaks,
            "total_video_sec": round(total_duration_sec, 3),
            "ball_detected_sec": float(ball_detected_sec),
            "processing_time_sec": processing_time,
            "max_speed_px_per_sec": round(max_speed, 2),
            "max_speed_kmh": round(max_speed_kmh, 2),
            "avg_speed_kmh": round(avg_speed_kmh, 2),
            "trajectory_outliers_count": int(trajectory_outliers_count),
            "goals_detected": len(goal_events),
            "goal_timestamps": [round(float(event["timestamp_s"]), 3) for event in goal_events],
            "goal_events": goal_events,
            "processed_frame_count": int(processed_count),
            "frame_skip": int(frame_skip),
            "infractions_count": 0,
            "clips": [],
            "training_frames": (
                self.detector.frame_exporter.exported_frames
                if self.detector.frame_exporter
                else []
            ),
            "player_tracks": self._finalize_player_tracks(player_tracks),
            "person_counts_by_sec": person_counts_by_sec,
            "debug_video_path": debug_video_path,
            "_track_points": track_points,
            "_danger_polygon": danger_polygon,
            "_danger_center": danger_center,
            "_audio_peaks": audio_peaks,
            "_audio_energy": audio_energy,
            "_fps": fps,
        }

    def _detect_ball_candidates(
        self,
        frame: np.ndarray,
        use_sahi: bool,
        match_id: str,
        frame_num: int,
    ) -> Tuple[List[Detection], List[Detection]]:
        ball_dets, person_dets = self.detector.detect(
            frame,
            use_sahi=use_sahi,
            match_id=match_id,
            frame_num=frame_num,
        )
        if self.dual_detector:
            ball_dets = self.dual_detector.confirm_detections(
                frame,
                ball_dets,
                Detection,
            )
            ball_dets.sort(key=lambda det: det.confidence, reverse=True)
        return ball_dets, person_dets

    def _has_audio_peak_near(
        self,
        audio_peaks: List[float],
        timestamp_sec: float,
        window_sec: float,
    ) -> bool:
        return any(abs(float(peak) - timestamp_sec) <= window_sec for peak in audio_peaks)

    def _sam3_teacher_confidence(
        self,
        frame,
        frame_idx: int,
        best_confidence: float,
    ) -> float:
        if best_confidence >= 0.3 or not self.sam3 or not self.sam3.is_available:
            return 0.0
        sam3_result = self.sam3.find_ball(frame)
        if not sam3_result:
            return 0.0
        teacher_conf = float(sam3_result.get("confidence", 0.0) or 0.0)
        logger.debug(
            "SAM3 encontro balon conf=%.2f frame %s",
            teacher_conf,
            frame_idx,
        )
        return teacher_conf

    def _analyze_teams_with_sam3(
        self,
        frame,
        players: List[Dict[str, Any]],
        danger_polygon,
    ) -> Dict[str, Any]:
        team_data = self.sam3.identify_teams(frame)
        if team_data.get("method") == "unavailable":
            return self.player_detector.analyze_players(players, danger_polygon=danger_polygon)

        players_in_zone = self.player_detector.count_in_zone(players, danger_polygon)
        logger.debug(
            "SAM3 equipos: A=%s B=%s",
            team_data.get("team_a", 0),
            team_data.get("team_b", 0),
        )
        return {
            "team_a": [],
            "team_b": [],
            "team_a_color": "dark",
            "team_b_color": "light",
            "team_a_count": int(team_data.get("team_a", 0) or 0),
            "team_b_count": int(team_data.get("team_b", 0) or 0),
            "team_a_in_zone_count": 0,
            "team_b_in_zone_count": 0,
            "players_in_danger_zone": players_in_zone,
            "total_players": len(players),
            "method": team_data.get("method", "grounding_dino"),
        }

    def _export_low_confidence_frame(
        self,
        frame,
        match_id: str,
        frame_idx: int,
        track_point: TrackPoint,
        best_confidence: float,
        in_danger_zone: bool,
        has_audio_peak: bool,
    ) -> Optional[Dict[str, Any]]:
        if self.detector.frame_exporter is None:
            return None

        speed_kmh = getattr(track_point, "speed_kmh", 0.0)
        failure = self.failure_detector.analyze(
            speed_kmh=speed_kmh,
            frames_lost=getattr(self.tracker, "frames_since_detection", 0),
            in_danger_zone=in_danger_zone,
            audio_peak=has_audio_peak,
            ball_detected=(best_confidence >= 0.3),
        )

        if failure["has_failure"] or best_confidence < 0.3:
            teacher_conf = self._sam3_teacher_confidence(frame, frame_idx, best_confidence)
            return self.detector.frame_exporter.export_frame(
                frame=frame,
                match_id=match_id,
                frame_num=frame_idx,
                confidence=best_confidence,
                label=failure["primary_type"] or "low_confidence",
                hard_case_type=failure["primary_type"],
                label_quality="C",
                teacher_confidence=teacher_conf,
                physics_ok=(speed_kmh <= 120.0),
            )
        return None

    def _export_missed_detection_frame(
        self,
        frame,
        match_id: str,
        frame_idx: int,
        track_point: Optional[TrackPoint],
        current_count: int,
        in_danger_zone: bool = False,
        audio_peak: bool = False,
    ) -> int:
        export_cfg = self.config.training_frames
        if not export_cfg.enabled or not getattr(export_cfg, "export_missed_detections", True):
            return 0
        max_exports = int(getattr(export_cfg, "max_missed_exports_per_match", 100) or 100)
        if current_count >= max_exports:
            return 0
        if track_point is None or track_point.is_detected:
            return 0
        frames_lost = int(getattr(self.tracker, "frames_since_detection", 0) or 0)
        if frames_lost <= 0 or frames_lost > 3:
            return 0
        if self.detector.frame_exporter is None:
            return 0

        speed_kmh = getattr(track_point, "speed_kmh", 0.0)
        failure = self.failure_detector.analyze(
            speed_kmh=speed_kmh,
            frames_lost=frames_lost,
            in_danger_zone=in_danger_zone,
            audio_peak=audio_peak,
            ball_detected=False,
        )

        teacher_conf = self._sam3_teacher_confidence(frame, frame_idx, 0.0)
        exported = self.detector.frame_exporter.export_frame(
            frame=frame,
            match_id=match_id,
            frame_num=frame_idx,
            confidence=0.0,
            label=failure["primary_type"] or "missed_detection",
            force=True,
            max_frames=max_exports,
            filename_prefix="missed",
            hard_case_type=failure["primary_type"],
            label_quality="C",
            teacher_confidence=teacher_conf,
            physics_ok=(speed_kmh <= 120.0),
        )
        if exported:
            logger.debug(
                "🧪 Missed detection exportado frame=%s lost=%s pred=(%.1f, %.1f)",
                frame_idx,
                frames_lost,
                track_point.x,
                track_point.y,
            )
            return 1
        return 0

    def _filter_players_by_polygon(self, players: List[Dict[str, Any]], polygon) -> List[Dict[str, Any]]:
        if polygon is None:
            return players
        return [
            player for player in players
            if self._point_in_polygon(player["center"][0], player["center"][1], polygon)
        ]

    def _update_player_team_assignments(
        self,
        team_context: Dict[str, Any],
        player_team_by_id: Dict[int, str],
    ) -> None:
        for player in team_context.get("team_a", []):
            track_id = player.get("track_id")
            if track_id is not None:
                player_team_by_id[int(track_id)] = "A"
        for player in team_context.get("team_b", []):
            track_id = player.get("track_id")
            if track_id is not None:
                player_team_by_id[int(track_id)] = "B"

    def _update_player_tracks(
        self,
        player_tracks: Dict[int, Dict[str, Any]],
        players: List[Dict[str, Any]],
        player_team_by_id: Dict[int, str],
        frame_idx: int,
        fps: float,
        homography: HomographyTransformer,
    ) -> None:
        for player in players:
            track_id = player.get("track_id")
            center = player.get("center") or []
            if track_id is None or len(center) < 2:
                continue

            tid = int(track_id)
            x = float(center[0])
            y = float(center[1])
            track = player_tracks.setdefault(
                tid,
                {
                    "positions": [],
                    "team": player_team_by_id.get(tid, "unknown"),
                    "distance_total": 0.0,
                    "distance_total_px": 0.0,
                    "_speed_sum": 0.0,
                    "_speed_count": 0,
                    "_last": None,
                },
            )
            track["team"] = player_team_by_id.get(tid, track.get("team", "unknown"))
            last = track.get("_last")
            if last is not None:
                last_x, last_y, last_frame = last
                distance_px = math.hypot(x - last_x, y - last_y)
                distance_m = homography.px_to_meters(distance_px)
                dt = max((frame_idx - last_frame) / max(fps, 1.0), 1e-3)
                track["distance_total"] += distance_m
                track["distance_total_px"] += distance_px
                track["_speed_sum"] += (distance_m / dt) * 3.6
                track["_speed_count"] += 1
            track["positions"].append((round(x, 2), round(y, 2), int(frame_idx)))
            track["_last"] = (x, y, int(frame_idx))

    def _players_with_team(
        self,
        players: List[Dict[str, Any]],
        player_team_by_id: Dict[int, str],
    ) -> List[Dict[str, Any]]:
        enriched = []
        for player in players:
            track_id = player.get("track_id")
            team = "unknown"
            if track_id is not None:
                team = player_team_by_id.get(int(track_id), "unknown")
            enriched.append(
                {
                    "track_id": track_id,
                    "team": team,
                    "bbox": player.get("bbox", []),
                    "center": player.get("center", []),
                    "confidence": player.get("confidence", 0.0),
                }
            )
        return enriched

    def _finalize_player_tracks(
        self,
        player_tracks: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        output = {}
        for track_id, track in player_tracks.items():
            speed_count = int(track.get("_speed_count", 0) or 0)
            speed_avg = (
                float(track.get("_speed_sum", 0.0) or 0.0) / speed_count
                if speed_count > 0
                else 0.0
            )
            output[str(track_id)] = {
                "positions": track.get("positions", []),
                "team": track.get("team", "unknown"),
                "speed_avg": round(speed_avg, 2),
                "distance_total": round(float(track.get("distance_total", 0.0) or 0.0), 2),
                "distance_total_px": round(float(track.get("distance_total_px", 0.0) or 0.0), 2),
            }
        return output

    def _make_polygon(self, points):
        if not points or not HAS_SHAPELY:
            return None
        try:
            coords = [
                (p[0], p[1]) if isinstance(p, (list, tuple))
                else (p.get("x", 0), p.get("y", 0))
                for p in points
            ]
            if len(coords) < 3:
                return None
            poly = Polygon(coords)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception as exc:
            logger.warning(f"⚠️ Error creando polígono: {exc}")
            return None

    def _point_in_polygon(self, x: float, y: float, polygon) -> bool:
        if polygon is None:
            return True
        try:
            return polygon.contains(Point(x, y)) or polygon.touches(Point(x, y))
        except Exception:
            return True

    def _classify_debug_game_state(self, track_points: List[TrackPoint]) -> Dict[int, str]:
        speeds_by_sec: Dict[int, List[float]] = {}
        for tp in track_points:
            speeds_by_sec.setdefault(int(tp.timestamp_sec), []).append(tp.speed_px_per_sec)

        if not speeds_by_sec:
            return {}

        max_sec = max(speeds_by_sec.keys())
        states = {sec: "active" for sec in range(0, max_sec + 1)}
        avg_speed_by_sec = {
            sec: float(np.mean(values)) if values else 0.0
            for sec, values in speeds_by_sec.items()
        }
        dead_ball_periods: List[Tuple[int, int]] = []
        current_dead_start = None
        consecutive_slow_sec = 0

        for sec in range(0, max_sec + 1):
            if avg_speed_by_sec.get(sec, 0.0) < 50.0:
                if current_dead_start is None:
                    current_dead_start = sec
                consecutive_slow_sec += 1
                continue

            if current_dead_start is not None and consecutive_slow_sec >= 1.5:
                dead_ball_periods.append((current_dead_start, sec - 1))
            current_dead_start = None
            consecutive_slow_sec = 0

        if current_dead_start is not None and consecutive_slow_sec >= 1.5:
            dead_ball_periods.append((current_dead_start, max_sec))

        for start_sec, end_sec in dead_ball_periods:
            for sec in range(start_sec, end_sec + 1):
                states[sec] = "dead_ball"

            restart_sec = None
            for sec in range(end_sec + 1, min(max_sec + 1, end_sec + 8)):
                if avg_speed_by_sec.get(sec, 0.0) > 100.0:
                    restart_sec = sec
                    break
            if restart_sec is not None:
                for sec in range(restart_sec, min(max_sec + 1, restart_sec + 3)):
                    if states.get(sec) != "dead_ball":
                        states[sec] = "restart"

        return states

    def _write_debug_video(
        self,
        video_path: str,
        output_path: str,
        debug_records: Dict[int, Dict[str, Any]],
        game_states: Dict[int, str],
        roi_points,
        danger_zone,
        fps: float,
        frame_w: int,
        frame_h: int,
    ) -> Optional[str]:
        if not debug_records:
            logger.warning("⚠️ Debug video solicitado sin frames procesados.")
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo reabrir video para debug: {video_path}")

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except OSError:
                pass

        frame_indices = sorted(debug_records.keys())
        if len(frame_indices) > 1:
            debug_fps = min(max(fps / max(frame_indices[1] - frame_indices[0], 1), 1.0), fps)
        else:
            debug_fps = max(min(fps, 30.0), 1.0)

        frames_dir = "/tmp/debug_frames"
        shutil.rmtree(frames_dir, ignore_errors=True)
        os.makedirs(frames_dir, exist_ok=True)
        saved_frames = 0

        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = self.undistorter.undistort(frame)
                record = debug_records[frame_idx]
                self._draw_debug_overlay(
                    frame=frame,
                    frame_idx=frame_idx,
                    detections=record.get("detections", []),
                    track_point=record.get("track_point"),
                    players=record.get("players", []),
                    game_state=game_states.get(int(record.get("timestamp_sec", 0)), "active"),
                    roi_points=roi_points,
                    danger_zone=danger_zone,
                )
                frame_path = os.path.join(frames_dir, f"frame_{saved_frames:06d}.jpg")
                if cv2.imwrite(frame_path, frame):
                    saved_frames += 1

            if saved_frames <= 0:
                raise RuntimeError("No se pudo escribir ningún frame JPEG para debug.")

            try:
                ffmpeg_bin = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
                subprocess.run(
                    [
                        ffmpeg_bin,
                        "-y",
                        "-framerate",
                        str(debug_fps),
                        "-i",
                        os.path.join(frames_dir, "frame_%06d.jpg"),
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        "-preset",
                        "fast",
                        "-movflags",
                        "faststart",
                        "-pix_fmt",
                        "yuv420p",
                        output_path,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError as exc:
                stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
                raise RuntimeError(f"ffmpeg no pudo compilar debug video H264: {stderr[-1200:]}") from exc
            logger.info(
                "🧪 Debug video H264 generado con ffmpeg: %d frames, %.2f fps",
                saved_frames,
                debug_fps,
            )
        finally:
            cap.release()
            shutil.rmtree(frames_dir, ignore_errors=True)

        return output_path if os.path.exists(output_path) else None

    def _draw_debug_overlay(
        self,
        frame,
        frame_idx: int,
        detections: List[Detection],
        track_point: Optional[TrackPoint],
        players: List[Dict[str, Any]],
        game_state: str,
        roi_points,
        danger_zone,
    ) -> None:
        self._draw_filled_polygon(frame, danger_zone, color=(48, 90, 216), alpha=0.28)
        self._draw_dotted_polygon(frame, danger_zone, color=(48, 90, 216), thickness=2)
        self._draw_dotted_polygon(frame, roi_points, color=(255, 255, 255), thickness=2)

        for det in detections:
            x1 = int(max(det.x - det.w / 2, 0))
            y1 = int(max(det.y - det.h / 2, 0))
            x2 = int(min(det.x + det.w / 2, frame.shape[1] - 1))
            y2 = int(min(det.y + det.h / 2, frame.shape[0] - 1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 190, 80), 2)
            cv2.putText(
                frame,
                f"{det.confidence * 100:.0f}%"
                + (f" #{det.track_id}" if det.track_id is not None else ""),
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 190, 80),
                2,
                cv2.LINE_AA,
            )

        self._draw_player_debug_overlay(frame, players)

        if track_point is not None:
            center = (int(track_point.x), int(track_point.y))
            if track_point.is_detected:
                cv2.circle(frame, center, 7, (30, 190, 80), 2)
            else:
                self._draw_dotted_box(frame, center, size=30, color=(35, 35, 220), thickness=2)
                cv2.putText(
                    frame,
                    "KALMAN MISS",
                    (center[0] + 14, max(18, center[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (35, 35, 220),
                    2,
                    cv2.LINE_AA,
                )

        speed = track_point.speed_px_per_sec if track_point else 0.0
        speed_kmh = track_point.speed_kmh if track_point else 0.0
        score = min(max(speed / max(self.config.scoring.reference_speed, 1.0), 0.0), 1.0)
        label = f"Frame {frame_idx} | Speed: {speed_kmh:.0f} km/h | Score: {score:.2f}"
        bg_color = (117, 158, 29) if game_state == "active" else (48, 90, 216)
        self._draw_debug_text(frame, label, bg_color)

    def _draw_player_debug_overlay(self, frame, players: List[Dict[str, Any]]) -> None:
        for player in players:
            bbox = player.get("bbox") or []
            if len(bbox) < 4:
                continue
            team = player.get("team", "unknown")
            track_id = player.get("track_id")
            if team == "A":
                color = (255, 90, 40)
            elif team == "B":
                color = (40, 210, 255)
            else:
                color = (220, 220, 220)

            x1, y1, x2, y2 = [int(value) for value in bbox[:4]]
            x1 = max(0, min(frame.shape[1] - 1, x1))
            x2 = max(0, min(frame.shape[1] - 1, x2))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            y2 = max(0, min(frame.shape[0] - 1, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{team}"
            if track_id is not None:
                label = f"{team} #{track_id}"
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_debug_text(self, frame, text: str, bg_color: Tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.72
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = 18, 18
        cv2.rectangle(
            frame,
            (x - 8, y - 8),
            (x + text_w + 8, y + text_h + baseline + 8),
            bg_color,
            -1,
        )
        cv2.putText(
            frame,
            text,
            (x, y + text_h),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_filled_polygon(self, frame, points, color: Tuple[int, int, int], alpha: float) -> None:
        coords = self._normalize_points(points)
        if len(coords) < 3:
            return
        overlay = frame.copy()
        cv2.fillPoly(overlay, [np.array(coords, dtype=np.int32)], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_dotted_polygon(self, frame, points, color: Tuple[int, int, int], thickness: int = 1) -> None:
        coords = self._normalize_points(points)
        if len(coords) < 2:
            return
        for idx, start in enumerate(coords):
            end = coords[(idx + 1) % len(coords)]
            self._draw_dotted_line(frame, start, end, color, thickness)

    def _draw_dotted_box(
        self,
        frame,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        half = size // 2
        points = [
            (center[0] - half, center[1] - half),
            (center[0] + half, center[1] - half),
            (center[0] + half, center[1] + half),
            (center[0] - half, center[1] + half),
        ]
        self._draw_dotted_polygon(frame, points, color, thickness)

    def _draw_dotted_line(
        self,
        frame,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        gap: int = 12,
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        distance = max(int(math.hypot(x2 - x1, y2 - y1)), 1)
        for i in range(0, distance, gap * 2):
            t1 = i / distance
            t2 = min((i + gap) / distance, 1.0)
            p1 = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
            p2 = (int(x1 + (x2 - x1) * t2), int(y1 + (y2 - y1) * t2))
            cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    def _normalize_points(self, points) -> List[Tuple[int, int]]:
        if not points:
            return []
        coords = []
        for point in points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                coords.append((int(point[0]), int(point[1])))
            elif isinstance(point, dict):
                coords.append((int(point.get("x", 0)), int(point.get("y", 0))))
        return coords

    def _compute_polygon_center(self, polygon, raw_points) -> Optional[Tuple[float, float]]:
        if polygon is not None and HAS_SHAPELY:
            try:
                centroid = polygon.centroid
                return (float(centroid.x), float(centroid.y))
            except Exception:
                pass

        if not raw_points:
            return None

        coords = []
        for point in raw_points:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                coords.append((float(point[0]), float(point[1])))
            elif isinstance(point, dict):
                coords.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))

        if len(coords) < 3:
            return None

        xs = [coord[0] for coord in coords]
        ys = [coord[1] for coord in coords]
        return (float(np.mean(xs)), float(np.mean(ys)))

    def _build_telemetry(
        self,
        track_points: List[TrackPoint],
        audio_energy: List[float],
        danger_polygon,
        fps: float,
        total_duration: float,
    ) -> List[Dict]:
        total_seconds = int(total_duration)
        if total_seconds <= 0:
            return []

        points_by_sec: Dict[int, List[TrackPoint]] = {}
        for tp in track_points:
            sec = int(tp.timestamp_sec)
            if sec not in points_by_sec:
                points_by_sec[sec] = []
            points_by_sec[sec].append(tp)

        telemetry = []
        for sec in range(total_seconds):
            pts = points_by_sec.get(sec, [])

            if pts:
                speeds = [tp.speed_px_per_sec for tp in pts]
                speeds_kmh = [tp.speed_kmh for tp in pts]
                avg_speed = float(np.mean(speeds))
                max_speed = float(np.max(speeds))
                avg_speed_kmh = float(np.mean(speeds_kmh))
                max_speed_kmh = float(np.max(speeds_kmh))
                ball_track_id = next((tp.track_id for tp in pts if tp.track_id is not None), None)
            else:
                avg_speed = 0.0
                max_speed = 0.0
                avg_speed_kmh = 0.0
                max_speed_kmh = 0.0
                ball_track_id = None

            audio_val = audio_energy[sec] if sec < len(audio_energy) else 0.0

            incoming = 0.0
            if danger_polygon and pts:
                for tp in pts:
                    if (
                        self._point_in_polygon(tp.x, tp.y, danger_polygon)
                        and tp.speed_px_per_sec > self.config.danger_zone.incoming_min_speed
                    ):
                        incoming = max(incoming, tp.speed_px_per_sec)

            speed_norm = min(max_speed / 600.0, 1.0)
            audio_norm = min(audio_val / 2.0, 1.0)
            score = (0.6 * speed_norm) + (0.4 * audio_norm)

            telemetry.append(
                {
                    "t": sec,
                    "speed": round(max_speed, 2),
                    "avg_speed": round(avg_speed, 2),
                    "speed_kmh": round(max_speed_kmh, 2),
                    "avg_speed_kmh": round(avg_speed_kmh, 2),
                    "audio": round(audio_val, 3),
                    "score": round(score, 3),
                    "incoming": round(incoming, 2),
                    "ball_track_id": ball_track_id,
                }
            )

        return telemetry

    def _build_detection_by_minute(
        self,
        track_points: List[TrackPoint],
        fps: float,
        total_duration: float,
    ) -> List[Dict]:
        total_minutes = int(total_duration / 60) + 1
        if total_minutes <= 0:
            return []

        by_minute: Dict[int, List[TrackPoint]] = {}
        for tp in track_points:
            minute = int(tp.timestamp_sec / 60)
            if minute not in by_minute:
                by_minute[minute] = []
            by_minute[minute].append(tp)

        diagnostics = []
        for minute in range(total_minutes):
            pts = by_minute.get(minute, [])
            total_sec_in_minute = min(60, int(total_duration - (minute * 60)))
            if total_sec_in_minute <= 0:
                continue

            detected_seconds = set()
            gk_count = 0
            speeds = []
            gap_count = 0
            gap_frames = 0
            in_gap = False

            for tp in pts:
                if tp.is_detected:
                    detected_seconds.add(int(tp.timestamp_sec))
                    if in_gap:
                        in_gap = False
                    if tp.is_gk_kick:
                        gk_count += 1
                else:
                    if not in_gap:
                        gap_count += 1
                        in_gap = True
                    gap_frames += 1

                speeds.append(tp.speed_px_per_sec)

            visible_sec = len(detected_seconds)
            pct = round(visible_sec / max(total_sec_in_minute, 1) * 100)
            avg_speed = round(float(np.mean(speeds)), 2) if speeds else 0.0
            max_speed = round(float(np.max(speeds)), 2) if speeds else 0.0

            diagnostics.append(
                {
                    "minute": minute,
                    "total_sec": total_sec_in_minute,
                    "visible_sec": visible_sec,
                    "pct": pct,
                    "avg_speed": avg_speed,
                    "max_speed": max_speed,
                    "gk_filtered": gk_count,
                    "gap_count": gap_count,
                    "gap_frames": gap_frames,
                }
            )

        return diagnostics


# ============================================================
# FEEDBACK LEARNER
# ============================================================

class FeedbackLearner:
    """
    Aprende de clips clasificados para ajustar umbrales dinámicamente.
    Usa velocidad, clase de velocidad y trayectoria para perfilar highlights.
    """

    def __init__(self, config):
        self.config = config
        self.approved_clips: List[Dict] = []
        self.rejected_clips: List[Dict] = []
        self.has_feedback = False
        self.feedback_level = "none"

        self.approved_speed_range: Tuple[float, float] = (0, 9999)
        self.rejected_speed_range: Tuple[float, float] = (0, 0)
        self.approved_angle_range: Tuple[float, float] = (0, 180)
        self.good_velocity_classes: set = set()
        self.bad_velocity_classes: set = set()
        self.rejected_approaching_rate = 0.0
        self.rejected_avg_angle: Optional[float] = None

    def load_from_supabase(self):
        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
        )
        if not url or not key:
            return

        try:
            from supabase import create_client

            client = create_client(url, key)
            approved_res = (
                client.table("clips")
                .select(
                    "speed, score, velocity_class, is_approaching, "
                    "has_audio_peak, approach_angle"
                )
                .eq("is_confirmed", True)
                .not_.is_("speed", "null")
                .execute()
            )
            rejected_res = (
                client.table("clips")
                .select(
                    "speed, score, velocity_class, is_approaching, "
                    "has_audio_peak, approach_angle"
                )
                .in_("review_status", ["rejected", "discarded", "trash"])
                .not_.is_("speed", "null")
                .execute()
            )

            self.approved_clips = [
                row
                for row in (approved_res.data or [])
                if row.get("speed") and float(row["speed"]) > 0
            ]
            self.rejected_clips = [
                row
                for row in (rejected_res.data or [])
                if row.get("speed") and float(row["speed"]) > 0
            ]

            total = len(self.approved_clips) + len(self.rejected_clips)
            if total == 0:
                self.feedback_level = "none"
                logger.info("📊 FeedbackLearner: Sin historial todavía")
                return

            self.has_feedback = True
            if total < 10:
                self.feedback_level = "basic"
            elif total < 30:
                self.feedback_level = "profile"
            else:
                self.feedback_level = "full"

            self._build_profiles()

            logger.info(
                "📊 FeedbackLearner nivel=%s: %s aprobados, %s rechazados, "
                "speed_buena=[%.0f-%.0f], speed_mala=[%.0f-%.0f], "
                "clases_buenas=%s, clases_malas=%s",
                self.feedback_level,
                len(self.approved_clips),
                len(self.rejected_clips),
                self.approved_speed_range[0],
                self.approved_speed_range[1],
                self.rejected_speed_range[0],
                self.rejected_speed_range[1],
                self.good_velocity_classes,
                self.bad_velocity_classes,
            )
        except Exception as exc:
            logger.warning(f"⚠️ FeedbackLearner: {exc}")

    def _build_profiles(self):
        self.rejected_approaching_rate = 0.0
        self.rejected_avg_angle = None

        if self.approved_clips:
            approved_speeds = [float(clip["speed"]) for clip in self.approved_clips]
            self.approved_speed_range = (
                float(np.percentile(approved_speeds, 10)),
                float(np.percentile(approved_speeds, 90)),
            )

        if self.rejected_clips:
            rejected_speeds = [float(clip["speed"]) for clip in self.rejected_clips]
            self.rejected_speed_range = (
                float(np.percentile(rejected_speeds, 10)),
                float(np.percentile(rejected_speeds, 90)),
            )
            self.rejected_approaching_rate = sum(
                1 for clip in self.rejected_clips if clip.get("is_approaching")
            ) / max(len(self.rejected_clips), 1)
            rejected_angles = [
                float(clip["approach_angle"])
                for clip in self.rejected_clips
                if clip.get("approach_angle") is not None
            ]
            if rejected_angles:
                self.rejected_avg_angle = float(np.mean(rejected_angles))

        for clip in self.approved_clips:
            velocity_class = clip.get("velocity_class", "")
            if velocity_class:
                self.good_velocity_classes.add(velocity_class)

        for clip in self.rejected_clips:
            velocity_class = clip.get("velocity_class", "")
            if velocity_class:
                self.bad_velocity_classes.add(velocity_class)

        ambiguous = self.good_velocity_classes & self.bad_velocity_classes
        if ambiguous:
            for velocity_class in ambiguous:
                approved_count = sum(
                    1
                    for clip in self.approved_clips
                    if clip.get("velocity_class") == velocity_class
                )
                rejected_count = sum(
                    1
                    for clip in self.rejected_clips
                    if clip.get("velocity_class") == velocity_class
                )
                if rejected_count > approved_count * 2:
                    self.good_velocity_classes.discard(velocity_class)
                elif approved_count > rejected_count * 2:
                    self.bad_velocity_classes.discard(velocity_class)

        if self.rejected_clips:
            logger.info(
                "📊 Perfil de rechazados: approaching_rate=%.1f%%, avg_angle=%.1f°",
                self.rejected_approaching_rate * 100,
                self.rejected_avg_angle if self.rejected_avg_angle is not None else 0.0,
            )

    def adjust_score(
        self,
        speed: float,
        base_score: float,
        velocity_class: str = "",
        is_approaching: bool = False,
    ) -> float:
        if not self.has_feedback:
            return base_score

        adjusted = base_score

        approved_min, approved_max = self.approved_speed_range
        rejected_min, rejected_max = self.rejected_speed_range
        if approved_min <= speed <= approved_max:
            adjusted *= 1.15
        elif self.rejected_clips and rejected_min <= speed <= rejected_max:
            adjusted *= 0.70

        if self.feedback_level == "basic":
            return min(adjusted, 1.0)

        if velocity_class in self.bad_velocity_classes and velocity_class not in self.good_velocity_classes:
            adjusted *= 0.60
        elif velocity_class in self.good_velocity_classes and velocity_class not in self.bad_velocity_classes:
            adjusted *= 1.10

        if self.feedback_level == "profile":
            return min(adjusted, 1.0)

        if self.approved_clips:
            approaching_rate = sum(
                1 for clip in self.approved_clips if clip.get("is_approaching")
            ) / max(len(self.approved_clips), 1)
            if approaching_rate > 0.7 and not is_approaching:
                adjusted *= 0.75
        if self.rejected_clips and self.rejected_avg_angle is not None:
            # El angulo ya se filtra en el scoring principal; aqui solo dejamos
            # el perfil cargado para debug y futuros ajustes.
            pass
        if self.rejected_clips and self.rejected_approaching_rate < 0.3 and not is_approaching:
            adjusted *= 0.85

        return min(adjusted, 1.0)


# ============================================================
# AI MEMORY
# ============================================================

class AIMemory:
    """
    Memoria persistente que sobrevive entre videos y no depende de clips.
    Guarda conclusiones, promedios y patrones aprendidos.
    """

    def __init__(self):
        self.memories: Dict[str, Any] = {}
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
        )
        if not url or not key:
            return None

        try:
            from supabase import create_client

            self._client = create_client(url, key)
        except Exception as exc:
            logger.warning(f"⚠️ AIMemory client: {exc}")
            self._client = None

        return self._client

    def load(self):
        client = self._get_client()
        if not client:
            return

        try:
            response = client.table("ai_memory").select("memory_key, memory_value").execute()
            self.memories = {}
            for row in (response.data or []):
                key = row.get("memory_key")
                if key:
                    self.memories[key] = row.get("memory_value")
            logger.info(f"🧠 AI Memory: {len(self.memories)} memorias cargadas")
        except Exception as exc:
            logger.warning(f"⚠️ AIMemory load: {exc}")

    def save(self, key: str, value: Any, description: str = ""):
        self.memories[key] = value
        client = self._get_client()
        if not client:
            return

        try:
            client.table("ai_memory").upsert(
                {
                    "memory_key": key,
                    "memory_value": value,
                    "description": description or None,
                    "updated_at": datetime.utcnow().isoformat(),
                },
                on_conflict="memory_key",
            ).execute()
        except Exception as exc:
            logger.warning(f"⚠️ AIMemory save: {exc}")

    def get(self, key: str, default=None):
        return self.memories.get(key, default)

    def update_after_processing(
        self,
        engine_results: Dict[str, Any],
        classified_clips: Optional[List[Dict[str, Any]]] = None,
    ):
        tracking_history = self.get("tracking_history", [])
        total_sec = float(engine_results.get("total_video_sec", 0) or 0)
        detected_sec = float(engine_results.get("ball_detected_sec", 0) or 0)
        detection_pct = round((detected_sec / max(total_sec, 1.0)) * 100, 1)
        tracking_history.append(
            {
                "total_sec": total_sec,
                "detected_sec": detected_sec,
                "detection_pct": detection_pct,
                "max_speed": float(engine_results.get("max_speed_px_per_sec", 0) or 0),
                "clips_generated": len(classified_clips or engine_results.get("clips", []) or []),
            }
        )
        tracking_history = tracking_history[-50:]
        self.save("tracking_history", tracking_history, "Historial de tracking por video")

        if tracking_history:
            avg_detection = np.mean([row.get("detection_pct", 0) for row in tracking_history])
            avg_max_speed = np.mean([row.get("max_speed", 0) for row in tracking_history])
            self.save(
                "avg_detection_pct",
                round(float(avg_detection), 1),
                "Promedio de deteccion global",
            )
            self.save(
                "avg_max_speed",
                round(float(avg_max_speed), 1),
                "Velocidad maxima promedio por video",
            )

        videos_processed = int(self.get("videos_processed", 0) or 0) + 1
        self.save("videos_processed", videos_processed, "Total de videos procesados")

        logger.info(
            "🧠 AI Memory actualizada: %s videos procesados, deteccion promedio: %s%%",
            videos_processed,
            self.get("avg_detection_pct", 0),
        )


# ============================================================
# GAME REFEREE — Eventos + Clips
# ============================================================

class GameReferee:
    def __init__(self, config: VisionConfig = None):
        self.config = config or DEFAULT_CONFIG
        self.feedback = FeedbackLearner(self.config)
        self.memory = AIMemory()
        self.learner = EvolutiveLearner()
        self.learner.load_from_ai_memory()
        self._audio_peaks_sec: List[float] = []

    def _build_danger_zone(
        self,
        engine_results: Dict[str, Any],
    ):
        danger_polygon = engine_results.get("_danger_polygon") or engine_results.get("danger_polygon")
        danger_center = engine_results.get("_danger_center") or engine_results.get("danger_center")
        return danger_polygon, danger_center

    def _classify_game_state(
        self,
        track_points: List[TrackPoint],
        fps: float = 30.0,
    ) -> Dict[int, str]:
        """
        Clasifica cada segundo del video como active, dead_ball o restart.

        dead_ball: velocidad promedio muy baja durante varios segundos.
        restart: tras un dead_ball sostenido, primer arranque de velocidad alta y
                 solo el segundo inicial queda penalizado.
        """
        if not track_points:
            return {}

        by_second: Dict[int, List[TrackPoint]] = {}
        for track_point in track_points:
            second = int(track_point.timestamp_sec)
            by_second.setdefault(second, []).append(track_point)

        max_sec = max(by_second.keys()) if by_second else 0
        avg_speed_by_second: Dict[int, float] = {}
        for sec, points in by_second.items():
            avg_speed_by_second[sec] = sum(tp.speed_px_per_sec for tp in points) / max(len(points), 1)

        moving_speeds = [speed for speed in avg_speed_by_second.values() if speed > 0]
        speed_p75 = float(np.percentile(moving_speeds, 75)) if moving_speeds else 0.0

        states: Dict[int, str] = {}
        dead_ball_threshold = max(25.0, min(55.0, speed_p75 * 0.18))
        restart_threshold = max(180.0, min(450.0, speed_p75 * 0.75))
        dead_ball_min_duration = 3.0
        restart_cooldown_sec = 1

        consecutive_slow_sec = 0.0
        current_dead_start: Optional[int] = None
        dead_ball_periods: List[Tuple[int, int]] = []

        for sec in range(max_sec + 1):
            points = by_second.get(sec, [])
            if not points:
                if current_dead_start is not None and consecutive_slow_sec >= dead_ball_min_duration:
                    dead_ball_periods.append((current_dead_start, sec - 1))
                current_dead_start = None
                consecutive_slow_sec = 0.0
                continue

            avg_speed = avg_speed_by_second.get(sec, 0.0)
            if avg_speed < dead_ball_threshold:
                if current_dead_start is None:
                    current_dead_start = sec
                consecutive_slow_sec += 1.0
            else:
                if current_dead_start is not None and consecutive_slow_sec >= dead_ball_min_duration:
                    dead_ball_periods.append((current_dead_start, sec - 1))
                current_dead_start = None
                consecutive_slow_sec = 0.0

        if current_dead_start is not None and consecutive_slow_sec >= dead_ball_min_duration:
            dead_ball_periods.append((current_dead_start, max_sec))

        for sec in range(max_sec + 1):
            states[sec] = "active"

        for start_sec, end_sec in dead_ball_periods:
            for sec in range(start_sec, end_sec + 1):
                states[sec] = "dead_ball"

            restart_sec = None
            for sec in range(end_sec + 1, min(max_sec + 1, end_sec + 6)):
                if avg_speed_by_second.get(sec, 0.0) > restart_threshold:
                    restart_sec = sec
                    break
                if states.get(sec) == "dead_ball":
                    break

            if restart_sec is not None:
                for sec in range(restart_sec, min(max_sec + 1, restart_sec + restart_cooldown_sec)):
                    if states.get(sec) != "dead_ball":
                        states[sec] = "restart"

        return states

    def _compute_intensity_scores(
        self,
        track_points: List[TrackPoint],
        game_states: Dict[int, str],
        danger_polygon,
        danger_center,
        person_counts_by_sec: Dict[int, int],
        window_sec: float = 3.0,
        fps: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        IntensityScorer V1-V6.
        Calcula un score por segundo usando 6 variables ponderadas.
        """
        if not track_points:
            return []

        by_second: Dict[int, List[TrackPoint]] = {}
        for tp in track_points:
            sec = int(tp.timestamp_sec)
            by_second.setdefault(sec, []).append(tp)

        max_sec = max(by_second.keys()) if by_second else 0
        window_half = int(window_sec / 2)
        cfg = self.config.scoring

        all_speeds = [tp.speed_px_per_sec for tp in track_points if tp.speed_px_per_sec > 0]
        speed_p95 = float(np.percentile(all_speeds, 95)) if all_speeds else cfg.reference_speed

        max_dist_to_center = 1.0
        if danger_center:
            distances = [
                tp.dist_to_danger_center for tp in track_points
                if tp.dist_to_danger_center is not None
            ]
            if distances:
                max_dist_to_center = max(float(np.percentile(distances, 95)), 1.0)

        intensity_scores: List[Dict[str, Any]] = []

        for center_sec in range(max_sec + 1):
            window_points: List[TrackPoint] = []
            for s in range(max(0, center_sec - window_half),
                          min(max_sec + 1, center_sec + window_half + 1)):
                window_points.extend(by_second.get(s, []))

            if not window_points:
                continue

            # === V1: PROXIMIDAD AL ARCO (peso 0.30) ===
            if danger_center:
                distances = [
                    tp.dist_to_danger_center for tp in window_points
                    if tp.dist_to_danger_center is not None
                ]
                if distances:
                    min_dist = min(distances)
                    v1 = max(0.0, 1.0 - (min_dist / max_dist_to_center))
                else:
                    v1 = 0.0
            else:
                v1 = 0.5

            # === V2: VELOCIDAD DEL BALÓN (peso 0.25) ===
            speeds = [tp.speed_px_per_sec for tp in window_points]
            speeds_kmh = [tp.speed_kmh for tp in window_points]
            max_speed = max(speeds)
            avg_speed = sum(speeds) / len(speeds)
            max_speed_kmh = max(speeds_kmh) if speeds_kmh else 0.0
            avg_speed_kmh = sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0
            v2 = min((max_speed * 0.7 + avg_speed * 0.3) / max(speed_p95, 1.0), 1.0)

            # === V3: DIRECCIÓN AL ARCO (peso 0.25) ===
            angles = [
                tp.approach_angle_deg for tp in window_points
                if tp.approach_angle_deg is not None
            ]
            avg_angle = sum(angles) / len(angles) if angles else 90.0
            min_angle = min(angles) if angles else avg_angle
            max_angle = cfg.max_approach_angle_deg

            if max_speed >= self.config.danger_zone.incoming_min_speed:
                scoring_angle = min(avg_angle, min_angle + 20.0)
                if scoring_angle <= max_angle:
                    v3 = 1.0
                elif avg_angle <= 90:
                    v3 = max(0.0, 1.0 - ((scoring_angle - max_angle) / max(90.0 - max_angle, 1.0)))
                else:
                    v3 = 0.15 if min_angle <= 75 else 0.0
            else:
                v3 = 0.0

            # === V4: AUDIO (peso 0.10) ===
            has_audio = self._has_audio_peak_near(
                self._audio_peaks_sec, center_sec, cfg.audio_search_window_sec,
            )
            v4 = 1.0 if has_audio else 0.0

            # === V5: DENSIDAD/TRAYECTORIA (peso 0.10) ===
            person_count = person_counts_by_sec.get(center_sec, 0)
            if person_count > 0:
                v5 = min(person_count / 6.0, 1.0)
            else:
                if len(speeds) > 2:
                    mid = len(speeds) // 2
                    first_avg = sum(speeds[:mid]) / max(mid, 1)
                    second_avg = sum(speeds[mid:]) / max(len(speeds) - mid, 1)
                    v5 = 1.0 if second_avg > first_avg else 0.3
                else:
                    v5 = 0.3

            # === V6: GAME STATE (multiplicador) ===
            game_state = game_states.get(center_sec, "active")
            if game_state == "dead_ball":
                v6 = cfg.dead_ball_multiplier
            elif game_state == "restart":
                v6 = cfg.restart_multiplier
            else:
                v6 = 1.0

            if avg_angle > 90:
                v6 *= 0.70

            # === SCORE FINAL ===
            raw_score = (
                v1 * 0.30 +
                v2 * 0.25 +
                v3 * 0.25 +
                v4 * 0.10 +
                v5 * 0.10
            )
            final_score = raw_score * v6

            is_gk_origin = False
            point_indices = {id(tp): idx for idx, tp in enumerate(track_points)}
            for tp in window_points:
                idx = point_indices.get(id(tp), -1)
                if idx >= 0 and self._originates_from_gk_zone(
                    idx, track_points, danger_polygon, danger_center
                ):
                    is_gk_origin = True
                    break
            if is_gk_origin:
                final_score *= 0.08

            velocity_class = self._classify_velocity(max_speed)
            final_score = self.feedback.adjust_score(
                speed=max_speed,
                base_score=final_score,
                velocity_class=velocity_class,
                is_approaching=(avg_angle <= 60),
            )
            final_score = self.learner.predict(
                final_score,
                {
                    "v1_proximity": v1,
                    "v2_speed": v2,
                    "v3_direction": v3,
                    "v4_audio": v4,
                    "v5_density": v5,
                    "v6_state": v6,
                },
            )

            best_tp = max(window_points, key=lambda tp: tp.speed_px_per_sec)

            breakdown = {
                "v1_proximity": round(v1, 3),
                "v2_speed": round(v2, 3),
                "v3_direction": round(v3, 3),
                "v4_audio": round(v4, 3),
                "v5_density": round(v5, 3),
                "v6_state": round(v6, 3),
                "raw_score": round(raw_score, 3),
                "game_state": game_state,
                "avg_angle": round(avg_angle, 1),
                "min_angle": round(min_angle, 1),
                "max_speed": round(max_speed, 1),
                "max_speed_kmh": round(max_speed_kmh, 1),
                "speed_p95": round(speed_p95, 1),
                "is_gk_origin": is_gk_origin,
                "person_count": person_count,
            }

            intensity_scores.append({
                "sec": center_sec,
                "score": round(min(final_score, 1.0), 4),
                "max_speed": round(max_speed, 1),
                "avg_speed": round(avg_speed, 1),
                "max_speed_kmh": round(max_speed_kmh, 1),
                "avg_speed_kmh": round(avg_speed_kmh, 1),
                "avg_angle": round(avg_angle, 1),
                "game_state": game_state,
                "best_tp": best_tp,
                "breakdown": breakdown,
            })

        return intensity_scores

    def detect_events(self, engine_results: Dict[str, Any]) -> List[EventCandidate]:
        """
        V30: Intensity Score Engine.
        Evalúa intensidad por segundo usando ventana deslizante y estado de juego.
        """
        track_points = engine_results.get("track_points") or engine_results.get("_track_points", [])
        if not track_points:
            logger.warning("⚠️ Sin track points para analizar")
            return []

        self._audio_peaks_sec = (
            engine_results.get("audio_peaks_sec")
            or engine_results.get("_audio_peaks")
            or []
        )

        danger_polygon, danger_center = self._build_danger_zone(engine_results)
        fps = engine_results.get("fps") or engine_results.get("_fps", 30.0)
        cfg_score = self.config.scoring
        cfg_dz = self.config.danger_zone

        person_counts_raw = engine_results.get("person_counts_by_sec", {})
        person_counts_by_sec: Dict[int, int] = {}
        for sec_str, count in person_counts_raw.items():
            try:
                if isinstance(count, dict):
                    density_count = (
                        count.get("team_a_in_zone_count")
                        or count.get("team_a_count")
                        or count.get("players_in_danger_zone")
                        or count.get("total_players")
                        or 0
                    )
                    person_counts_by_sec[int(sec_str)] = int(density_count)
                else:
                    person_counts_by_sec[int(sec_str)] = int(count)
            except (ValueError, TypeError):
                pass

        game_states = self._classify_game_state(track_points, fps)
        active_count = sum(1 for state in game_states.values() if state == "active")
        dead_count = sum(1 for state in game_states.values() if state == "dead_ball")
        restart_count = sum(1 for state in game_states.values() if state == "restart")
        logger.info(
            "⚽ Game states: %s activos, %s dead_ball, %s restart (de %s segundos)",
            active_count,
            dead_count,
            restart_count,
            len(game_states),
        )

        intensity_scores = self._compute_intensity_scores(
            track_points,
            game_states,
            danger_polygon,
            danger_center,
            person_counts_by_sec,
            window_sec=3.0,
            fps=fps,
        )

        self._intensity_timeline = [
            {
                "t": s["sec"],
                "score": s["score"],
                "v1": s["breakdown"]["v1_proximity"],
                "v2": s["breakdown"]["v2_speed"],
                "v3": s["breakdown"]["v3_direction"],
                "v4": s["breakdown"]["v4_audio"],
                "v5": s["breakdown"]["v5_density"],
                "v6": s["breakdown"]["v6_state"],
                "state": s["game_state"],
                "speed": s["max_speed"],
                "speed_kmh": s.get("max_speed_kmh", 0.0),
            }
            for s in intensity_scores
        ]

        if not intensity_scores:
            logger.warning("⚠️ Sin intensity scores calculados")
            return []

        min_score = cfg_score.min_clip_score
        raw_candidates = [score_row for score_row in intensity_scores if score_row["score"] >= min_score]
        score_by_sec = {score_row["sec"]: score_row["score"] for score_row in intensity_scores}
        sustained_candidates = []
        for score_row in raw_candidates:
            left_score = score_by_sec.get(score_row["sec"] - 1, 0.0)
            right_score = score_by_sec.get(score_row["sec"] + 1, 0.0)
            is_sustained = (
                left_score >= (min_score * 0.85)
                or right_score >= (min_score * 0.85)
                or score_row["score"] >= min(min_score + 0.20, 0.95)
            )
            if is_sustained:
                sustained_candidates.append(score_row)
        candidates = sustained_candidates

        logger.info(
            "📊 Intensity scores: %s segundos analizados, %s cruzan umbral, %s sostenidos (%.2f)",
            len(intensity_scores),
            len(raw_candidates),
            len(sustained_candidates),
            min_score,
        )

        for score_row in intensity_scores:
            if score_row["score"] < min_score and score_row["max_speed"] > 300:
                logger.info(
                    "  🚫 Descartado sec=%s speed=%.1f score=%.3f state=%s angle=%.1f breakdown=%s",
                    score_row["sec"],
                    score_row["max_speed"],
                    score_row["score"],
                    score_row["game_state"],
                    score_row["avg_angle"],
                    score_row["breakdown"],
                )

        events: List[EventCandidate] = []
        point_indices = {id(track_point): idx for idx, track_point in enumerate(track_points)}
        for candidate in candidates:
            track_point = candidate["best_tp"]
            speed = candidate["max_speed"]
            speed_kmh = candidate.get("max_speed_kmh", getattr(track_point, "speed_kmh", 0.0))
            velocity_class = self._classify_velocity(speed)
            approach_angle = candidate["avg_angle"]
            is_approaching = approach_angle <= 60
            in_danger_zone = (
                self._point_in_danger_zone(track_point, danger_polygon)
                if danger_polygon is not None
                else False
            )
            has_audio_peak = candidate["breakdown"].get("v4_audio", 0) > 0.5
            is_long_range = (
                (not in_danger_zone)
                and speed >= cfg_dz.long_range_min_speed
                and is_approaching
            ) if danger_polygon is not None else False

            event_type = self._resolve_event_type(
                velocity_class,
                in_danger_zone,
                has_audio_peak,
                is_long_range,
                is_approaching=is_approaching,
            )

            event_ts = track_point.timestamp_sec
            if velocity_class == "shot" or is_long_range:
                index = point_indices.get(id(track_point), len(track_points) - 1)
                accel_start = self._find_acceleration_start(
                    index,
                    track_points,
                    100.0,
                )
                if accel_start < event_ts - 1.5:
                    event_ts = accel_start

            events.append(
                EventCandidate(
                    timestamp_sec=event_ts,
                    frame_idx=track_point.frame_idx,
                    event_type=event_type,
                    score=candidate["score"],
                    speed=speed,
                    speed_kmh=speed_kmh,
                    in_danger_zone=in_danger_zone,
                    velocity_class=velocity_class,
                    is_approaching=is_approaching,
                    has_audio_peak=has_audio_peak,
                    approach_angle_deg=approach_angle,
                    score_breakdown=candidate["breakdown"],
                )
            )

            logger.info(
                "🎯 Evento sec=%s ts=%.2fs speed=%.1fpx/s %.1fkm/h vel=%s score=%.3f "
                "state=%s angle=%.1f° long_range=%s breakdown=%s",
                candidate["sec"],
                event_ts,
                speed,
                speed_kmh,
                velocity_class,
                candidate["score"],
                candidate["game_state"],
                approach_angle,
                is_long_range,
                candidate["breakdown"],
            )

        events.sort(key=lambda event: event.timestamp_sec)

        deduped: List[EventCandidate] = []
        for event in events:
            if deduped and abs(event.timestamp_sec - deduped[-1].timestamp_sec) < 1.0:
                if event.score > deduped[-1].score:
                    deduped[-1] = event
            else:
                deduped.append(event)

        sequences: List[List[EventCandidate]] = []
        current_seq: List[EventCandidate] = []
        for event in deduped:
            if current_seq and (
                event.timestamp_sec - current_seq[-1].timestamp_sec
            ) > cfg_score.sequence_window_sec:
                sequences.append(current_seq)
                current_seq = []
            current_seq.append(event)
        if current_seq:
            sequences.append(current_seq)

        final_events: List[EventCandidate] = []
        for seq in sequences:
            sorted_by_score = sorted(seq, key=lambda event: event.score, reverse=True)
            selected: List[EventCandidate] = []

            for candidate in sorted_by_score:
                if len(selected) >= cfg_score.max_moments_per_sequence:
                    break

                too_close = any(
                    abs(candidate.timestamp_sec - chosen.timestamp_sec)
                    < cfg_score.min_moment_separation_sec
                    for chosen in selected
                )
                if not too_close:
                    selected.append(candidate)

            selected.sort(key=lambda event: event.timestamp_sec)

            if selected and seq:
                earliest_ts = seq[0].timestamp_sec
                if selected[0].timestamp_sec > earliest_ts + 2.0:
                    selected[0] = EventCandidate(
                        timestamp_sec=earliest_ts,
                        frame_idx=seq[0].frame_idx,
                        event_type=selected[0].event_type,
                        score=selected[0].score,
                        speed=selected[0].speed,
                        speed_kmh=getattr(selected[0], "speed_kmh", 0.0),
                        in_danger_zone=selected[0].in_danger_zone,
                        is_approaching=getattr(selected[0], "is_approaching", False),
                        has_audio_peak=getattr(selected[0], "has_audio_peak", False),
                        velocity_class=getattr(selected[0], "velocity_class", "unknown"),
                        approach_angle_deg=getattr(selected[0], "approach_angle_deg", None),
                        score_breakdown=getattr(selected[0], "score_breakdown", {}),
                    )

            final_events.extend(selected)

            if selected:
                logger.info(
                    "  📦 Secuencia %.1fs–%.1fs: %s candidatos → %s clips",
                    seq[0].timestamp_sec,
                    seq[-1].timestamp_sec,
                    len(seq),
                    len(selected),
            )

        auto_goal_events = []
        for goal_event in engine_results.get("goal_events", []) or []:
            try:
                ts = float(goal_event.get("timestamp_s", 0.0))
            except (TypeError, ValueError):
                continue
            auto_goal_events.append(
                EventCandidate(
                    timestamp_sec=ts,
                    frame_idx=int(ts * fps),
                    event_type="goal",
                    score=1.0,
                    speed=0.0,
                    speed_kmh=0.0,
                    in_danger_zone=True,
                    velocity_class="shot",
                    is_approaching=True,
                    has_audio_peak=bool(goal_event.get("audio_confirmed", False)),
                    approach_angle_deg=None,
                    score_breakdown={
                        "auto_verdict": goal_event.get("auto_verdict", "probable_goal"),
                        "goal_confidence": goal_event.get("confidence", 0.0),
                        "frames_confirmed": goal_event.get("frames_confirmed", 0),
                    },
                )
            )

        if auto_goal_events:
            logger.info("⚽ Goles automáticos añadidos a clips: %s", len(auto_goal_events))
            final_events = auto_goal_events + final_events

        logger.info(
            "⚽ Resultado final: %s clips (de %s intensity peaks, %s deduplicados, %s secuencias)",
            len(final_events),
            len(candidates),
            len(deduped),
            len(sequences),
        )
        return final_events

    def _classify_velocity(self, speed: float) -> str:
        vel_cfg = self.config.velocity
        if speed <= vel_cfg.possession_max:
            return "possession"
        if speed <= vel_cfg.short_pass_max:
            return "short_pass"
        if speed <= vel_cfg.long_pass_max:
            return "long_pass"
        if speed <= vel_cfg.shot_max:
            return "shot"
        return "artifact"

    def _point_in_danger_zone(self, track_point: TrackPoint, polygon) -> bool:
        if polygon is None or not HAS_SHAPELY:
            return False
        try:
            point = Point(track_point.x, track_point.y)
            return polygon.contains(point) or polygon.touches(point)
        except Exception:
            return False

    def _is_near_danger_zone(
        self,
        track_point: TrackPoint,
        polygon,
        danger_center: Optional[Tuple[float, float]],
    ) -> bool:
        if polygon is None:
            return False
        if self._point_in_danger_zone(track_point, polygon):
            return True
        if track_point.dist_to_danger_center is None or danger_center is None:
            return False
        try:
            minx, miny, maxx, maxy = polygon.bounds
            radius = max(maxx - minx, maxy - miny) * 0.75
        except Exception:
            radius = 180.0
        return track_point.dist_to_danger_center <= max(radius, 120.0)

    def _has_audio_peak_near(
        self,
        audio_peaks: List[float],
        timestamp_sec: float,
        window_sec: float,
    ) -> bool:
        return any(abs(float(peak) - timestamp_sec) <= window_sec for peak in audio_peaks)

    def _get_audio_energy(self, audio_energy: List[float], timestamp_sec: float) -> float:
        if not audio_energy:
            return 0.0
        sec_idx = int(timestamp_sec)
        if sec_idx < 0 or sec_idx >= len(audio_energy):
            return 0.0
        return min(1.0, float(audio_energy[sec_idx]) / 2.0)

    def _analyze_trajectory(self, index: int, track_points: List[TrackPoint]) -> bool:
        lookback = self.config.scoring.trajectory_lookback_frames
        if index <= 0:
            return False

        start_index = max(0, index - lookback)
        window = [
            tp
            for tp in track_points[start_index : index + 1]
            if tp.dist_to_danger_center is not None
        ]
        if len(window) < 2:
            return False

        reductions = 0
        comparisons = 0
        for prev, curr in zip(window, window[1:]):
            comparisons += 1
            if curr.dist_to_danger_center < prev.dist_to_danger_center:
                reductions += 1

        if comparisons == 0:
            return False

        return (reductions / comparisons) >= 0.5

    def _find_acceleration_start(
        self,
        index: int,
        track_points: List[TrackPoint],
        min_speed_threshold: float = 100.0,
    ) -> float:
        """
        Busca hacia atras desde el pico de velocidad para encontrar el momento
        donde el balon EMPEZO a acelerar. Retorna el timestamp de ese momento.
        Util para dar mas contexto a tiros lejanos.
        """
        if not track_points:
            return 0.0
        if index <= 0:
            return track_points[max(0, index)].timestamp_sec

        lookback = min(90, index)
        for back_index in range(index - 1, max(0, index - lookback) - 1, -1):
            track_point = track_points[back_index]
            if track_point.speed_px_per_sec < min_speed_threshold:
                return track_points[min(back_index + 1, index)].timestamp_sec

        return track_points[max(0, index - lookback)].timestamp_sec

    def _originates_from_gk_zone(
        self,
        index: int,
        track_points: List[TrackPoint],
        danger_polygon,
        danger_center: Optional[Tuple[float, float]],
    ) -> bool:
        """
        Verifica si el movimiento rapido empezo desde el fondo de la danger zone,
        lo que suele corresponder a un saque del arquero.
        """
        if danger_polygon is None or danger_center is None:
            return False

        lookback = min(30, index)
        if lookback < 3:
            return False

        try:
            minx, miny, maxx, maxy = danger_polygon.bounds
        except Exception:
            return False

        zone_height = maxy - miny
        zone_width = maxx - minx
        if zone_height <= 0 and zone_width <= 0:
            return False

        gk_pct = self.config.danger_zone.gk_origin_zone_pct
        frame_center_y = 540
        if danger_center[1] > frame_center_y:
            gk_zone_y_start = maxy - (zone_height * gk_pct)
            is_in_gk_zone = lambda x, y: y >= gk_zone_y_start
        else:
            gk_zone_y_end = miny + (zone_height * gk_pct)
            is_in_gk_zone = lambda x, y: y <= gk_zone_y_end

        prev_points = track_points[max(0, index - lookback) : index]
        if not prev_points:
            return False

        slow_in_gk_zone = 0
        for track_point in prev_points:
            in_gk = is_in_gk_zone(track_point.x, track_point.y)
            is_slow = track_point.speed_px_per_sec < self.config.velocity.short_pass_max
            if in_gk and is_slow:
                slow_in_gk_zone += 1

        return (slow_in_gk_zone / max(len(prev_points), 1)) >= 0.4

    def _resolve_event_type(
        self,
        velocity_class: str,
        in_danger_zone: bool,
        has_audio_peak: bool,
        is_long_range: bool = False,
        is_approaching: bool = False,
    ) -> str:
        if is_long_range and velocity_class == "shot":
            return "long_range_shot"
        if velocity_class == "shot":
            return "shot"
        if velocity_class == "long_pass" and is_approaching and (in_danger_zone or has_audio_peak):
            return "danger_build_up"
        if velocity_class == "short_pass" and in_danger_zone and has_audio_peak and is_approaching:
            return "chance"
        return "play"

    def generate_clips(
        self,
        events: List[EventCandidate],
        video_path: str,
        output_dir: str,
    ) -> List[Dict]:
        os.makedirs(output_dir, exist_ok=True)
        cfg = self.config.clips
        clips = []

        for index, event in enumerate(events):
            clip_num = index + 1
            if getattr(event, "event_type", "") == "long_range_shot":
                pre_sec = cfg.long_range_pre_event_sec
            else:
                pre_sec = cfg.pre_event_sec
            start = max(0.0, event.timestamp_sec - pre_sec)
            duration = max(cfg.min_clip_duration_sec, pre_sec + cfg.post_event_sec)
            duration = min(duration, cfg.max_clip_duration_sec)

            filename = f"clip_{clip_num:03d}_{event.event_type}_{int(event.timestamp_sec)}s.mp4"
            output_path = os.path.join(output_dir, filename)

            try:
                # Intento 1: re-encode para corte preciso.
                cmd_reencode = [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    video_path,
                    "-t",
                    f"{duration:.3f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    "-avoid_negative_ts",
                    "make_zero",
                    output_path,
                ]
                result = subprocess.run(cmd_reencode, capture_output=True, timeout=120)

                # Si re-encode falla, fallback a copy para no perder el clip.
                if (
                    result.returncode != 0
                    or not os.path.exists(output_path)
                    or os.path.getsize(output_path) < 1000
                ):
                    logger.warning(f"  ⚠️ Re-encode falló clip #{clip_num}, usando -c copy")
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    cmd_copy = [
                        "ffmpeg",
                        "-y",
                        "-ss",
                        f"{start:.3f}",
                        "-i",
                        video_path,
                        "-t",
                        f"{duration:.3f}",
                        "-c",
                        "copy",
                        "-avoid_negative_ts",
                        "make_zero",
                        output_path,
                    ]
                    subprocess.run(cmd_copy, capture_output=True, timeout=60)

                if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                    clips.append(
                        {
                            "clip_number": clip_num,
                            "start_time": round(start, 2),
                            "timestamp_sec": round(event.timestamp_sec, 2),
                            "event_type": event.event_type,
                            "score": event.score,
                            "speed": event.speed,
                            "speed_kmh": getattr(event, "speed_kmh", 0.0),
                            "in_danger_zone": event.in_danger_zone,
                            "velocity_class": getattr(event, "velocity_class", "unknown"),
                            "is_approaching": getattr(event, "is_approaching", False),
                            "has_audio_peak": getattr(event, "has_audio_peak", False),
                            "approach_angle_deg": getattr(event, "approach_angle_deg", None),
                            "score_breakdown": getattr(event, "score_breakdown", {}),
                            "auto_verdict": getattr(event, "score_breakdown", {}).get("auto_verdict"),
                            "goal_confidence": getattr(event, "score_breakdown", {}).get("goal_confidence"),
                            "local_path": output_path,
                            "filename": filename,
                        }
                    )
                    logger.info(
                        f"  🎬 Clip #{clip_num}: {filename} ({start:.1f}s → {start + duration:.1f}s)"
                    )
                else:
                    logger.warning(f"  ⚠️ Clip #{clip_num} vacío después de ambos intentos")
            except subprocess.TimeoutExpired:
                logger.warning(f"  ⚠️ Clip #{clip_num} timeout")
            except Exception as exc:
                logger.warning(f"  ⚠️ Error clip #{clip_num}: {exc}")

        logger.info(f"✅ {len(clips)} clips generados en {output_dir}")
        return clips

    def process_and_clip(
        self,
        engine_results: Dict[str, Any],
        video_path: str,
        output_dir: str = "/tmp/clips",
    ) -> Dict[str, Any]:
        clip_stage_start = time.time()
        self.memory.load()
        self.feedback.load_from_supabase()
        avg_detection = self.memory.get("avg_detection_pct")
        if avg_detection and avg_detection < 30:
            logger.info(
                "🧠 Deteccion promedio baja (%.1f%%), manteniendo heuristica permisiva",
                avg_detection,
            )

        events = self.detect_events(engine_results)
        clips = self.generate_clips(events, video_path, output_dir)
        self.memory.update_after_processing(engine_results, clips)
        clip_stage_time = time.time() - clip_stage_start
        visual_processing_time = float(engine_results.get("processing_time_sec", 0) or 0)
        total_pipeline_time = visual_processing_time + clip_stage_time

        logger.info(
            "⏱️ Pipeline completo en %.1fs (%.1f min) — video: %.1fs, "
            "frames: %s (skip=%s), fps_efectivo: %.1f, clips: %s",
            total_pipeline_time,
            total_pipeline_time / 60,
            float(engine_results.get("total_video_sec", 0) or 0),
            int(engine_results.get("processed_frame_count", 0) or 0),
            int(engine_results.get("frame_skip", 1) or 1),
            float(engine_results.get("processed_frame_count", 0) or 0)
            / max(visual_processing_time, 1e-3),
            len(clips),
        )

        engine_results["clips"] = clips
        engine_results["infractions_count"] = len(clips)
        engine_results["intensity_timeline"] = getattr(self, "_intensity_timeline", [])
        engine_results["message"] = (
            f"Processed successfully. Found {len(clips)} highlights. "
            f"Tracking: {int(engine_results.get('ball_detected_sec', 0))}s detected "
            f"of {int(engine_results.get('total_video_sec', 0))}s total."
        )

        for key in list(engine_results.keys()):
            if key.startswith("_"):
                del engine_results[key]

        return engine_results
