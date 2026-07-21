from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BallStateRecord:
    run_id: str
    match_id: str
    frame_id: str
    timestamp_sec: float
    visible: bool
    position_px: Optional[List[float]] = None
    position_field: Optional[List[float]] = None
    speed_px_s: float = 0.0
    speed_kmh: float = 0.0
    state: str = "unknown"
    source_track_id: Optional[str] = None
    source_detection_id: Optional[str] = None
    quality: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)
