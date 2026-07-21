from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


ArtifactStatus = Literal["produced", "not_available", "degraded", "invalid", "skipped"]


@dataclass
class RunArtifact:
    run_id: str
    match_id: str
    name: str
    status: ArtifactStatus
    path: str = ""
    reason: Optional[str] = ""
    content_type: str = ""
    size_bytes: Optional[int] = None
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
