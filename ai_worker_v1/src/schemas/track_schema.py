from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrackRecord:
    run_id: str
    match_id: str
    frame_id: str
    track_id: str
    object_class: str
    timestamp_sec: float
    point_xy: List[float]
    bbox_xyxy: List[float] = field(default_factory=list)
    confidence: float = 0.0
    state: str = "active"
    velocity_xy: List[float] = field(default_factory=list)
    source_detection_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
