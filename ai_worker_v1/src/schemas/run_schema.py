from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RunMetadata:
    run_id: str
    match_id: str
    worker_type: str
    environment: str
    detector_mode: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "running"
    input_video_uri: str = ""
    output_prefix: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
