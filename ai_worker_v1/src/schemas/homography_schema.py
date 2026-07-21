from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class HomographyRecord:
    run_id: str
    match_id: str
    status: str
    matrix: Optional[List[List[float]]] = None
    image_points: List[List[float]] = field(default_factory=list)
    field_points: List[List[float]] = field(default_factory=list)
    calibration_source: str = ""
    reprojection_error: Optional[float] = None
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
