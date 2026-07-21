from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TacticalMetricRecord:
    run_id: str
    match_id: str
    metric_id: str
    metric_type: str
    timestamp_sec: float
    value: Any
    team_id: Optional[str] = None
    confidence: float = 0.0
    source_events: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
