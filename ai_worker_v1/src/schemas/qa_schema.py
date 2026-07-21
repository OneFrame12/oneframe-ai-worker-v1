from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QAReport:
    run_id: str
    match_id: str
    status: str
    checks: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    artifact_status: Dict[str, str] = field(default_factory=dict)
    summary: str = ""
    created_at: str = ""
