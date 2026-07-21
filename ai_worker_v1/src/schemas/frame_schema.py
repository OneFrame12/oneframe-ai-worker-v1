from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class FrameRecord:
    run_id: str
    match_id: str
    frame_id: str
    frame_index: int
    timestamp_sec: float
    source_uri: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    sample_level: str = "frame"
    dedupe_key: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
