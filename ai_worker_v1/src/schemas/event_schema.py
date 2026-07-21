from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EventRecord:
    run_id: str
    match_id: str
    event_id: str
    event_type: str
    timestamp_sec: float
    confidence: float
    source: str
    involved_tracks: List[str] = field(default_factory=list)
    ball_state_id: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    qa_status: str = "unchecked"
    metadata: Dict[str, Any] = field(default_factory=dict)
