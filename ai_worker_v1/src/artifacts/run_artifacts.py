from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from schemas import RunArtifact


ARTIFACT_STATUSES = ("produced", "not_available", "degraded", "invalid", "skipped")

REQUIRED_ARTIFACT_NAMES = (
    "metadata.json",
    "detections.json",
    "tracks.json",
    "ball_state.json",
    "homography.json",
    "identity_assignments.json",
    "events.json",
    "tactics.json",
    "qa_report.json",
    "qa_report.md",
    "overlay_video.mp4",
    "event_clips/",
    "keyframes/",
    "run_summary.md",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_placeholder_artifact(
    run_id: str,
    match_id: str,
    name: str,
    reason: str,
    status: str = "not_available",
    output_prefix: str = "",
) -> RunArtifact:
    if status not in ARTIFACT_STATUSES:
        raise ValueError(f"Invalid artifact status: {status}")

    path = f"{output_prefix.rstrip('/')}/{name}" if output_prefix else name
    return RunArtifact(
        run_id=run_id,
        match_id=match_id,
        name=name,
        status=status,  # type: ignore[arg-type]
        path=path,
        reason=reason,
        created_at=utc_now_iso(),
    )


def create_produced_artifact(
    run_id: str,
    match_id: str,
    name: str,
    path: str,
    content_type: str = "",
    size_bytes: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> RunArtifact:
    return RunArtifact(
        run_id=run_id,
        match_id=match_id,
        name=name,
        status="produced",
        path=path,
        reason=None,  # type: ignore[arg-type]
        content_type=content_type,
        size_bytes=size_bytes,
        created_at=utc_now_iso(),
        metadata=metadata or {},
    )


def build_required_artifacts(
    run_id: str,
    match_id: str,
    produced: Optional[Dict[str, RunArtifact]] = None,
    output_prefix: str = "",
    default_reason: str = "Artifact not produced by this run mode.",
) -> List[RunArtifact]:
    produced = produced or {}
    artifacts: List[RunArtifact] = []
    for name in REQUIRED_ARTIFACT_NAMES:
        artifacts.append(
            produced.get(name)
            or create_placeholder_artifact(
                run_id=run_id,
                match_id=match_id,
                name=name,
                reason=default_reason,
                output_prefix=output_prefix,
            )
        )
    return artifacts


def artifact_manifest(artifacts: Iterable[RunArtifact]) -> List[dict]:
    return [asdict(artifact) for artifact in artifacts]
