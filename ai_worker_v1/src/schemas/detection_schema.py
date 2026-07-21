from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DetectionRecord:
    run_id: str
    match_id: str
    frame_id: str
    detector_source: str
    detector_mode: str
    class_id: int
    class_name: str
    normalized_class: str
    confidence: float
    bbox_xyxy: List[float]
    bbox_xywh: List[float]
    threshold: float = 0.0
    detection_id: Optional[str] = None
    is_primary: bool = False
    filters_applied: List[str] = field(default_factory=list)
    rejection_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
