"""
OneFrame Evolutive Learner.

Aprende pesos simples desde training_samples y los guarda en ai_memory para que
el siguiente procesamiento pueda ajustar el score sin depender de deploys.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("OneFrame.Learner")

LEARNER_WEIGHTS_KEY = "evolutive_learner_weights"
LEARNER_SAMPLE_COUNT_KEY = "evolutive_learner_sample_count"
DEFAULT_WEIGHTS = {
    "v1_proximity": 0.30,
    "v2_speed": 0.25,
    "v3_direction": 0.25,
    "v4_audio": 0.10,
    "v5_density": 0.10,
    "v6_game_state": 1.0,
}


class EvolutiveLearner:
    def __init__(self, supabase_client=None):
        self.client = supabase_client
        self.samples: List[Dict[str, Any]] = []
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)
        self.sample_count = 0
        self.previous_sample_count = 0
        self.enabled = False

    def _get_client(self):
        if self.client is not None:
            return self.client

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

            self.client = create_client(url, key)
        except Exception as exc:
            logger.warning("⚠️ EvolutiveLearner client: %s", exc)
            self.client = None

        return self.client

    def load_from_ai_memory(self) -> Dict[str, float]:
        client = self._get_client()
        if not client:
            return self.weights

        try:
            response = (
                client.table("ai_memory")
                .select("memory_key, memory_value")
                .in_("memory_key", [LEARNER_WEIGHTS_KEY, LEARNER_SAMPLE_COUNT_KEY])
                .execute()
            )
            for row in response.data or []:
                key = row.get("memory_key")
                value = row.get("memory_value")
                if key == LEARNER_WEIGHTS_KEY and isinstance(value, dict):
                    self.weights = self._normalize_weights(value)
                    self.enabled = True
                elif key == LEARNER_SAMPLE_COUNT_KEY:
                    self.previous_sample_count = int(float(value or 0))
            if self.enabled:
                logger.info("🧠 EvolutiveLearner pesos cargados: %s", self.weights)
        except Exception as exc:
            logger.warning("⚠️ EvolutiveLearner load ai_memory: %s", exc)

        return self.weights

    def load_samples(self, limit: int = 5000) -> List[Dict[str, Any]]:
        client = self._get_client()
        if not client:
            return []

        try:
            response = (
                client.table("training_samples")
                .select(
                    "label, speed_px_sec, approach_angle, v1_proximity, v2_speed, "
                    "v3_direction, v6_game_state, intensity_score, has_audio_peak, "
                    "game_state, is_highlight, created_at"
                )
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            self.samples = list(response.data or [])
            self.sample_count = len(self.samples)
            logger.info("🧠 EvolutiveLearner muestras cargadas: %s", self.sample_count)
        except Exception as exc:
            logger.warning("⚠️ EvolutiveLearner load samples: %s", exc)
            self.samples = []
            self.sample_count = 0

        return self.samples

    def new_sample_count(self) -> int:
        return max(0, int(self.sample_count) - int(self.previous_sample_count))

    def should_evolve(self, min_new_samples: int = 10) -> bool:
        return self.sample_count >= min_new_samples and self.new_sample_count() >= min_new_samples

    def evolve(self) -> Dict[str, float]:
        if not self.samples:
            return self.weights

        positives = [sample for sample in self.samples if self._is_positive(sample)]
        negatives = [sample for sample in self.samples if not self._is_positive(sample)]
        if len(positives) < 2 or len(negatives) < 2:
            logger.info(
                "🧠 EvolutiveLearner sin clases suficientes: %s positivos, %s negativos",
                len(positives),
                len(negatives),
            )
            return self.weights

        signals = ["v1_proximity", "v2_speed", "v3_direction"]
        learned = {}
        for signal in signals:
            pos_mean = self._mean(positives, signal)
            neg_mean = self._mean(negatives, signal)
            learned[signal] = max(0.05, pos_mean - neg_mean)

        audio_lift = (
            self._rate(positives, "has_audio_peak") - self._rate(negatives, "has_audio_peak")
        )
        learned["v4_audio"] = max(0.05, audio_lift)
        learned["v5_density"] = DEFAULT_WEIGHTS["v5_density"]

        total = sum(learned.values()) or 1.0
        normalized = {key: value / total for key, value in learned.items()}
        self.weights = {
            "v1_proximity": round(self._blend(DEFAULT_WEIGHTS["v1_proximity"], normalized["v1_proximity"]), 4),
            "v2_speed": round(self._blend(DEFAULT_WEIGHTS["v2_speed"], normalized["v2_speed"]), 4),
            "v3_direction": round(self._blend(DEFAULT_WEIGHTS["v3_direction"], normalized["v3_direction"]), 4),
            "v4_audio": round(self._blend(DEFAULT_WEIGHTS["v4_audio"], normalized["v4_audio"]), 4),
            "v5_density": round(self._blend(DEFAULT_WEIGHTS["v5_density"], normalized["v5_density"]), 4),
            "v6_game_state": 1.0,
        }
        self.enabled = True
        logger.info("🧬 EvolutiveLearner evolucionó pesos: %s", self.weights)
        return self.weights

    def save_to_ai_memory(self) -> bool:
        client = self._get_client()
        if not client:
            return False

        try:
            now = datetime.utcnow().isoformat()
            client.table("ai_memory").upsert(
                {
                    "memory_key": LEARNER_WEIGHTS_KEY,
                    "memory_value": self.weights,
                    "description": "Pesos aprendidos desde training_samples",
                    "updated_at": now,
                },
                on_conflict="memory_key",
            ).execute()
            client.table("ai_memory").upsert(
                {
                    "memory_key": LEARNER_SAMPLE_COUNT_KEY,
                    "memory_value": self.sample_count,
                    "description": "Cantidad de muestras vistas por EvolutiveLearner",
                    "updated_at": now,
                },
                on_conflict="memory_key",
            ).execute()
            self.previous_sample_count = self.sample_count
            return True
        except Exception as exc:
            logger.warning("⚠️ EvolutiveLearner save ai_memory: %s", exc)
            return False

    def predict(self, base_score: float, features: Dict[str, Any]) -> float:
        if not self.enabled:
            return base_score

        learned_score = (
            self._num(features.get("v1_proximity")) * self.weights["v1_proximity"]
            + self._num(features.get("v2_speed")) * self.weights["v2_speed"]
            + self._num(features.get("v3_direction")) * self.weights["v3_direction"]
            + self._num(features.get("v4_audio")) * self.weights["v4_audio"]
            + self._num(features.get("v5_density")) * self.weights["v5_density"]
        )
        learned_score *= self._num(features.get("v6_state"), self.weights["v6_game_state"])
        adjusted = (base_score * 0.70) + (learned_score * 0.30)
        return min(max(adjusted, 0.0), 1.0)

    def _normalize_weights(self, value: Dict[str, Any]) -> Dict[str, float]:
        weights = dict(DEFAULT_WEIGHTS)
        for key in weights:
            numeric = self._num(value.get(key), weights[key])
            weights[key] = numeric
        return weights

    @staticmethod
    def _is_positive(sample: Dict[str, Any]) -> bool:
        label = str(sample.get("label") or "").lower()
        return bool(sample.get("is_highlight")) or label in {"gol", "tiro_arco", "tiro", "ocasion"}

    @staticmethod
    def _num(value, fallback: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        if not np.isfinite(numeric):
            return fallback
        return numeric

    def _mean(self, samples: List[Dict[str, Any]], key: str) -> float:
        values = [self._num(sample.get(key)) for sample in samples]
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _rate(samples: List[Dict[str, Any]], key: str) -> float:
        if not samples:
            return 0.0
        return sum(1 for sample in samples if sample.get(key)) / len(samples)

    @staticmethod
    def _blend(default: float, learned: float, alpha: float = 0.65) -> float:
        return (default * (1.0 - alpha)) + (learned * alpha)
