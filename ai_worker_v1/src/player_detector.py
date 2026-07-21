"""
Detección de jugadores y clasificación de equipos por color de camiseta.

Este módulo replica el bloque táctico usado por sistemas profesionales:
YOLO para detectar personas + K-means sobre la región de camiseta.
"""

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


logger = logging.getLogger("OneFrame.PlayerDetector")


class PlayerDetector:
    """Detecta jugadores y los clasifica por equipo."""

    def __init__(self, config=None, model_path: str = "yolo11n.pt"):
        self.config = config
        self.model_path = self._resolve_model_path(model_path)
        self.model = None
        self.confidence = float(getattr(config, "confidence", 0.5))
        self.min_player_size_px = float(getattr(config, "min_player_size_px", 20.0))
        self.max_player_size_px = float(getattr(config, "max_player_size_px", 400.0))
        self.min_players_for_clustering = int(getattr(config, "min_players_for_clustering", 4))
        self.jersey_region_ratio = float(getattr(config, "jersey_region_ratio", 0.4))
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
            logger.info("✅ PlayerDetector cargado: %s", self.model_path)
        except Exception as exc:
            logger.warning("⚠️ PlayerDetector no disponible: %s", exc)
            self.model = None

    def detect_players(self, frame) -> List[Dict[str, Any]]:
        """Detecta todas las personas en el frame."""
        if self.model is None:
            return []

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )
        players: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None:
                continue
            track_ids = result.boxes.id
            for box_idx, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)
                size = max(width, height)
                if size < self.min_player_size_px or size > self.max_player_size_px:
                    continue

                ix1 = max(0, int(x1))
                iy1 = max(0, int(y1))
                ix2 = min(frame.shape[1], int(x2))
                jersey_bottom = min(frame.shape[0], int(y1 + height * self.jersey_region_ratio))
                jersey_region = frame[iy1:jersey_bottom, ix1:ix2]
                track_id = None
                if track_ids is not None:
                    try:
                        track_id = int(track_ids[box_idx].item())
                    except Exception:
                        track_id = None

                players.append(
                    {
                        "track_id": track_id,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "jersey_region": jersey_region,
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "confidence": float(box.conf[0].item()),
                    }
                )
        return players

    def classify_teams(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        """K-means clustering en colores de camiseta."""
        if len(players) < self.min_players_for_clustering:
            return self._empty_team_context(players)

        try:
            from sklearn.cluster import KMeans
        except Exception as exc:
            logger.warning("⚠️ sklearn no disponible para K-means de equipos: %s", exc)
            return self._empty_team_context(players)

        valid_players = []
        colors = []
        for player in players:
            jersey = player.get("jersey_region")
            if jersey is None or jersey.size == 0:
                continue
            color = self._dominant_jersey_color(jersey)
            if color is None:
                continue
            valid_players.append(player)
            colors.append(color)

        if len(colors) < self.min_players_for_clustering:
            return self._empty_team_context(players)

        n_clusters = min(3, len(colors))
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=7)
        labels = kmeans.fit_predict(np.array(colors))
        counts = Counter(labels)
        top_clusters = counts.most_common(2)
        if len(top_clusters) < 2:
            return self._empty_team_context(players)

        team_a_label = top_clusters[0][0]
        team_b_label = top_clusters[1][0]
        team_a = [
            self._strip_player(player)
            for player, label in zip(valid_players, labels)
            if label == team_a_label
        ]
        team_b = [
            self._strip_player(player)
            for player, label in zip(valid_players, labels)
            if label == team_b_label
        ]

        return {
            "team_a": team_a,
            "team_b": team_b,
            "team_a_color": kmeans.cluster_centers_[team_a_label].tolist(),
            "team_b_color": kmeans.cluster_centers_[team_b_label].tolist(),
            "team_a_count": len(team_a),
            "team_b_count": len(team_b),
            "total_players": len(players),
        }

    def analyze_frame(self, frame, danger_polygon=None, roi_polygon=None) -> Dict[str, Any]:
        players = self.detect_players(frame)
        return self.analyze_players(players, danger_polygon=danger_polygon, roi_polygon=roi_polygon)

    def analyze_players(
        self,
        players: List[Dict[str, Any]],
        danger_polygon=None,
        roi_polygon=None,
    ) -> Dict[str, Any]:
        if roi_polygon is not None:
            players = [
                player for player in players
                if self._point_in_polygon(player["center"][0], player["center"][1], roi_polygon)
            ]

        teams = self.classify_teams(players)
        team_a_in_zone = self._count_players_in_zone(teams.get("team_a", []), danger_polygon)
        team_b_in_zone = self._count_players_in_zone(teams.get("team_b", []), danger_polygon)
        total_in_zone = self._count_players_in_zone(
            [self._strip_player(player) for player in players],
            danger_polygon,
        )

        return {
            **teams,
            "team_a_in_zone_count": team_a_in_zone,
            "team_b_in_zone_count": team_b_in_zone,
            "players_in_danger_zone": total_in_zone,
        }

    def count_in_zone(self, players: List[Dict[str, Any]], danger_polygon, roi_polygon=None) -> int:
        if roi_polygon is not None:
            players = [
                player for player in players
                if self._point_in_polygon(player["center"][0], player["center"][1], roi_polygon)
            ]
        return self._count_players_in_zone(players, danger_polygon)

    def _dominant_jersey_color(self, jersey: np.ndarray) -> Optional[np.ndarray]:
        if jersey.size == 0:
            return None
        resized = cv2.resize(jersey, (24, 24), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        saturation_mask = hsv[:, :, 1] > 25
        pixels = resized[saturation_mask] if np.any(saturation_mask) else resized.reshape(-1, 3)
        if pixels.size == 0:
            return None
        return pixels.reshape(-1, 3).mean(axis=0)

    def _count_players_in_zone(self, players: List[Dict[str, Any]], danger_polygon) -> int:
        if not players or danger_polygon is None:
            return 0
        return sum(
            1 for player in players
            if self._point_in_polygon(player["center"][0], player["center"][1], danger_polygon)
        )

    def _point_in_polygon(self, x: float, y: float, polygon) -> bool:
        try:
            from shapely.geometry import Point

            return polygon.contains(Point(x, y))
        except Exception:
            return False

    def _empty_team_context(self, players: List[Dict[str, Any]]) -> Dict[str, Any]:
        stripped_players = [self._strip_player(player) for player in players]
        return {
            "team_a": stripped_players,
            "team_b": [],
            "team_a_color": None,
            "team_b_color": None,
            "team_a_count": len(stripped_players),
            "team_b_count": 0,
            "total_players": len(players),
        }

    def _strip_player(self, player: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "track_id": player.get("track_id"),
            "bbox": player.get("bbox", []),
            "center": player.get("center", []),
            "confidence": player.get("confidence", 0.0),
        }

    def _resolve_model_path(self, model_path: str) -> str:
        for candidate in (
            os.getenv("PLAYER_MODEL_PATH"),
            "/app/yolo11n.pt",
            model_path,
        ):
            if not candidate:
                continue
            if os.path.exists(candidate):
                return candidate
        return model_path
