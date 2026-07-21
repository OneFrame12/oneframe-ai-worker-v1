"""
Detector de gol automatico para OneFrame.
Usa homografia + postes del arco + confirmacion temporal.
"""

import logging
from collections import deque
from typing import Optional


logger = logging.getLogger("OneFrame.GoalDetector")


class GoalDetector:
    """
    Detecta goles automaticamente.

    Reglas combinadas:
    1. Balon cruza la linea de gol
    2. Dentro del ancho del arco
    3. Confirmacion temporal: 3+ frames consecutivos
    4. Audio peak refuerza la deteccion
    """

    def __init__(
        self,
        goalpost_left_px: tuple = None,
        goalpost_right_px: tuple = None,
        goalpost_width_m: float = 6.0,
        frame_height_px: int = 720,
    ):
        self.goalpost_left_px = goalpost_left_px
        self.goalpost_right_px = goalpost_right_px
        self.goalpost_width_m = goalpost_width_m
        self.frame_height_px = frame_height_px

        self._ball_history = deque(maxlen=30)
        self._time_history = deque(maxlen=30)
        self._frames_in_goal = 0
        self._last_goal_time = -10.0
        self._goal_cooldown_s = 5.0
        self._last_ball_time = None
        self._max_consecutive_gap_s = 0.5

        self._goal_line_y = None
        self._goal_left_x = None
        self._goal_right_x = None
        if goalpost_left_px and goalpost_right_px:
            self._calculate_goal_line()

    def _calculate_goal_line(self):
        """Calcula la linea de gol desde los postes."""
        if not self.goalpost_left_px or not self.goalpost_right_px:
            return

        self._goal_line_y = (
            self.goalpost_left_px[1] + self.goalpost_right_px[1]
        ) / 2
        self._goal_left_x = min(self.goalpost_left_px[0], self.goalpost_right_px[0])
        self._goal_right_x = max(self.goalpost_left_px[0], self.goalpost_right_px[0])

        logger.info(
            "GoalDetector calibrado: linea Y=%.0fpx, arco X=[%.0f, %.0f]px",
            self._goal_line_y,
            self._goal_left_x,
            self._goal_right_x,
        )

    def calibrate(
        self,
        goalpost_left_px: tuple,
        goalpost_right_px: tuple,
        goalpost_width_m: float = 6.0,
    ):
        """Calibra el detector con las coordenadas de los postes."""
        self.goalpost_left_px = goalpost_left_px
        self.goalpost_right_px = goalpost_right_px
        self.goalpost_width_m = goalpost_width_m
        self._calculate_goal_line()

    @property
    def is_calibrated(self) -> bool:
        return self._goal_line_y is not None

    def update(
        self,
        ball_x: Optional[float],
        ball_y: Optional[float],
        timestamp_s: float,
        audio_peak: bool = False,
        ball_confidence: float = 0.0,
    ) -> Optional[dict]:
        """
        Actualiza el detector con la posicion actual del balon.

        Returns:
            dict con info del gol si se detecto, None si no.
        """
        if not self.is_calibrated:
            return None

        if ball_x is None or ball_y is None or ball_confidence < 0.25:
            self._frames_in_goal = 0
            return None

        self._ball_history.append((ball_x, ball_y))
        self._time_history.append(timestamp_s)

        inside_x = self._goal_left_x <= ball_x <= self._goal_right_x
        past_line = ball_y >= self._goal_line_y

        if inside_x and past_line:
            if (
                self._last_ball_time is not None
                and timestamp_s - self._last_ball_time > self._max_consecutive_gap_s
            ):
                self._frames_in_goal = 0
            self._frames_in_goal += 1
        else:
            self._frames_in_goal = 0
        self._last_ball_time = timestamp_s

        if timestamp_s - self._last_goal_time < self._goal_cooldown_s:
            return None

        confidence = 0.0
        if self._frames_in_goal >= 3:
            confidence += 0.50
        elif self._frames_in_goal >= 1:
            confidence += 0.20

        if inside_x:
            confidence += 0.30
        if past_line:
            confidence += 0.10
        if audio_peak:
            confidence += 0.10

        if confidence >= 0.70 and self._frames_in_goal >= 3:
            frames_confirmed = self._frames_in_goal
            self._last_goal_time = timestamp_s
            self._frames_in_goal = 0

            goal_event = {
                "event_type": "goal",
                "timestamp_s": timestamp_s,
                "ball_position": (ball_x, ball_y),
                "confidence": confidence,
                "frames_confirmed": frames_confirmed,
                "audio_confirmed": audio_peak,
                "auto_verdict": "probable_goal",
            }

            logger.info(
                "GOL DETECTADO en t=%.1fs conf=%.2f pos=(%.0f,%.0f)",
                timestamp_s,
                confidence,
                ball_x,
                ball_y,
            )
            return goal_event

        return None

    def reset(self):
        """Resetea el estado para un nuevo partido."""
        self._ball_history.clear()
        self._time_history.clear()
        self._frames_in_goal = 0
        self._last_goal_time = -10.0
        self._last_ball_time = None
