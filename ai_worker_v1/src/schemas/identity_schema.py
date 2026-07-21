from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class IdentityAssignmentRecord:
    run_id: str
    match_id: str
    frame_id: str
    track_id: str
    identity_id: Optional[str] = None
    team_id: Optional[str] = None
    role: str = ""
    confidence: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
