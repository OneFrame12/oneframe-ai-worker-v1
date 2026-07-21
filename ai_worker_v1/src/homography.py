"""
Transformación simple de píxeles a metros para cancha de fútbol 7.
"""

import logging
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


FIELD_LENGTH_M = 65.0
FIELD_WIDTH_M = 45.0
logger = logging.getLogger("OneFrame.Homography")


class HomographyTransformer:
    def __init__(self):
        self.H = None
        self.pixels_per_meter: Optional[float] = None
        self.px_per_meter: Optional[float] = None
        self.meters_per_px: Optional[float] = None
        self.calibration_method: Optional[str] = None
        self.goalpost_left_px: Optional[Tuple[float, float]] = None
        self.goalpost_right_px: Optional[Tuple[float, float]] = None

    def calibrate_from_roi(
        self,
        roi_points: list,
        field_length: float = FIELD_LENGTH_M,
        field_width: float = FIELD_WIDTH_M,
    ):
        points = self._normalize_points(roi_points)
        if len(points) < 4:
            self._estimate_from_roi_area(points)
            return

        dst_points = np.float32(
            [
                [0, 0],
                [field_length, 0],
                [field_length, field_width],
                [0, field_width],
            ]
        )
        src_points = np.float32(points[:4])
        self.H, _ = cv2.findHomography(src_points, dst_points)
        self.pixels_per_meter = self._estimate_scale_from_corners(
            src_points,
            field_length,
            field_width,
        )
        if not self.pixels_per_meter or self.pixels_per_meter <= 0:
            self._estimate_from_roi_area(points)
        self._sync_scale_aliases("roi")

    def _estimate_from_roi_area(self, roi_points):
        points = self._normalize_points(roi_points)
        if len(points) < 3:
            self.pixels_per_meter = 8.0
            self._sync_scale_aliases("roi_area_default")
            return
        pts = np.array(points, dtype=np.float32)
        area_px = max(cv2.contourArea(pts), 1.0)
        area_m2 = FIELD_LENGTH_M * FIELD_WIDTH_M
        self.pixels_per_meter = float(np.sqrt(area_px / area_m2))
        self._sync_scale_aliases("roi_area")

    def px_to_meters(self, px_distance: float) -> float:
        if self.meters_per_px:
            return float(px_distance) * self.meters_per_px
        if self.px_per_meter:
            return float(px_distance) / max(self.px_per_meter, 1e-6)
        if self.pixels_per_meter:
            return float(px_distance) / max(self.pixels_per_meter, 1e-6)
        return float(px_distance) / 8.0

    def speed_px_to_kmh(self, speed_px_per_sec: float) -> float:
        """Convierte velocidad px/s a km/h con validación física."""
        speed_ms = self.px_to_meters(speed_px_per_sec)
        speed_kmh = speed_ms * 3.6
        # Filtro físico: un balón de fútbol 7 amateur no puede superar 120 km/h.
        MAX_BALL_SPEED_KMH = 120.0
        if speed_kmh > MAX_BALL_SPEED_KMH:
            return 0.0
        return round(speed_kmh, 2)

    def calibrate_from_goalposts(
        self,
        post_left_px: tuple,
        post_right_px: tuple,
        goalpost_width_m: float = 6.0,
    ) -> bool:
        """
        Calibra la escala px->metros usando los postes del arco.

        Args:
            post_left_px: (x, y) pixel del poste izquierdo
            post_right_px: (x, y) pixel del poste derecho
            goalpost_width_m: ancho real del arco en metros
                              (futbol 7 = 6.0m, futbol 11 = 7.32m)

        Returns:
            True si la calibracion fue exitosa
        """
        try:
            dx = post_right_px[0] - post_left_px[0]
            dy = post_right_px[1] - post_left_px[1]
            distance_px = math.sqrt(dx**2 + dy**2)

            if distance_px < 10:
                logger.warning("Postes muy cercanos en px; calibracion ignorada")
                return False

            self.px_per_meter = distance_px / goalpost_width_m
            self.pixels_per_meter = self.px_per_meter
            self.meters_per_px = goalpost_width_m / distance_px
            self.calibration_method = "goalposts"
            self.goalpost_left_px = (float(post_left_px[0]), float(post_left_px[1]))
            self.goalpost_right_px = (float(post_right_px[0]), float(post_right_px[1]))

            logger.info(
                "Calibracion por postes: %.1fpx = %.2fm -> %.2f px/m",
                distance_px,
                goalpost_width_m,
                self.px_per_meter,
            )
            return True

        except Exception as e:
            logger.warning(f"Error en calibracion por postes: {e}")
            return False

    def validate_trajectory(
        self,
        positions: list,
        timestamps: list,
        max_speed_kmh: float = 120.0,
        max_acceleration_ms2: float = 50.0,
    ) -> dict:
        """
        Valida que la trayectoria del balon sea fisicamente posible.

        Returns dict con:
        - valid_positions: lista de posiciones validas
        - outliers: indices de posiciones descartadas
        - max_speed_kmh: velocidad maxima real
        - avg_speed_kmh: velocidad promedio
        """
        if len(positions) < 2:
            return {
                "valid_positions": positions,
                "outliers": [],
                "max_speed_kmh": 0.0,
                "avg_speed_kmh": 0.0,
            }

        valid = [positions[0]]
        valid_times = [timestamps[0]]
        outliers = []
        speeds = []
        previous_speed_ms = None

        for i in range(1, len(positions)):
            dt = max(timestamps[i] - timestamps[i - 1], 1e-3)
            dx = positions[i][0] - valid[-1][0]
            dy = positions[i][1] - valid[-1][1]
            dist_px = math.sqrt(dx**2 + dy**2)

            speed_px_s = dist_px / dt
            speed_kmh = self.speed_px_to_kmh(speed_px_s)
            speed_ms = self.px_to_meters(speed_px_s)
            acceleration_ms2 = (
                abs(speed_ms - previous_speed_ms) / dt
                if previous_speed_ms is not None
                else 0.0
            )

            if (
                (speed_kmh == 0.0 and speed_px_s > 0)
                or speed_kmh > max_speed_kmh
                or acceleration_ms2 > max_acceleration_ms2
            ):
                outliers.append(i)
            else:
                valid.append(positions[i])
                valid_times.append(timestamps[i])
                previous_speed_ms = speed_ms
                if speed_kmh > 0:
                    speeds.append(speed_kmh)

        return {
            "valid_positions": valid,
            "outliers": outliers,
            "max_speed_kmh": max(speeds) if speeds else 0.0,
            "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
        }

    def _estimate_scale_from_corners(
        self,
        src_points: np.ndarray,
        field_length: float,
        field_width: float,
    ) -> float:
        p0, p1, p2, p3 = src_points[:4]
        top = np.linalg.norm(p1 - p0) / max(field_length, 1e-6)
        bottom = np.linalg.norm(p2 - p3) / max(field_length, 1e-6)
        right = np.linalg.norm(p2 - p1) / max(field_width, 1e-6)
        left = np.linalg.norm(p3 - p0) / max(field_width, 1e-6)
        values = [value for value in (top, bottom, right, left) if value > 0]
        return float(np.mean(values)) if values else 8.0

    def _sync_scale_aliases(self, method: str):
        if self.pixels_per_meter and self.pixels_per_meter > 0:
            self.px_per_meter = self.pixels_per_meter
            self.meters_per_px = 1.0 / self.pixels_per_meter
            self.calibration_method = method

    def _normalize_points(self, points) -> List[Tuple[float, float]]:
        normalized = []
        for point in points or []:
            if isinstance(point, dict):
                normalized.append((float(point.get("x", 0.0)), float(point.get("y", 0.0))))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                normalized.append((float(point[0]), float(point[1])))
        return normalized
