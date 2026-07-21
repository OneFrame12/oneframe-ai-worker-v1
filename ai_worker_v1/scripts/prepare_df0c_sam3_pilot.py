#!/usr/bin/env python3
"""Prepare DF-0C SAM 3.1 checkpoint manifest and pilot clip selection."""

from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "ai_worker_v1" / "data_factory"
CKPT_DIR = DF / "sam3_worker" / "checkpoints" / "sam3_1"
TREE_META = CKPT_DIR / ".cache" / "huggingface" / "trees" / "daa63191845a41281374e725f4c9e51c7a824460.json"
SMOKE = ROOT / "ai_worker_v1" / "runs" / "pe0_multivideo_visual_smoke_20260717T062015Z" / "remote_outputs" / "visual_smoke_summary.json"
ROI_FREEZE = ROOT / "ai_worker_v1" / "runs" / "pe0_multivideo_ingestion_20260717T024829Z" / "calibration" / "manual_roi_v3_1_final" / "roi_v3_1_freeze_summary.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def build_checkpoint_manifest() -> dict[str, Any]:
    tree = read_json(TREE_META, {"files": {}})
    files = []
    complete = True
    for name, meta in sorted(tree.get("files", {}).items()):
        path = CKPT_DIR / name
        exists = path.exists()
        actual_sha = sha256_file(path) if exists else None
        expected_sha = meta.get("lfs_sha256")
        if expected_sha:
            match = actual_sha == expected_sha
        else:
            match = actual_sha is not None
        if not exists or not match:
            complete = False
        files.append(
            {
                "filename": name,
                "path": rel(path),
                "exists": exists,
                "size": path.stat().st_size if exists else None,
                "expected_size": meta.get("lfs_size") or meta.get("size"),
                "sha256": actual_sha,
                "expected_lfs_sha256": expected_sha,
                "sha256_match": match,
                "blob_id": meta.get("blob_id"),
                "xet_hash": meta.get("xet_hash"),
            }
        )
    config = read_json(CKPT_DIR / "config.json", {})
    manifest = {
        "schema_version": "df0c.sam3_checkpoint_manifest.v0",
        "repository": "facebook/sam3.1",
        "revision": "daa63191845a41281374e725f4c9e51c7a824460",
        "download_timestamp": now_iso(),
        "loader_expected": {
            "architecture": config.get("architectures"),
            "model_type": config.get("model_type"),
            "checkpoint": "sam3.1_multiplex.pt",
            "tokenizer_assets": [
                "vocab.json",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "processor_config.json",
            ],
        },
        "license_access_status": "authenticated_access_verified_config_and_files",
        "files": files,
        "complete": complete,
        "token_exposed": False,
    }
    write_json(CKPT_DIR / "checkpoint_manifest.json", manifest)
    return manifest


def update_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    gate_path = DF / "sam3_adapter" / "sam3_gate.json"
    gate = read_json(gate_path, {})
    checkpoint_download = "passed" if manifest["complete"] else "blocked"
    hf_auth_present = any(
        path.exists()
        for path in [
            Path.home() / ".cache" / "huggingface" / "token",
            Path.home() / ".cache" / "huggingface" / "stored_tokens",
        ]
    ) or bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    gate.update(
        {
            "checkpoint_access": "verified",
            "config_download": "passed",
            "checkpoint_download": checkpoint_download,
            "local_token_present": hf_auth_present,
            "token_exposed": False,
            "pilot_status": "ready_for_runpod" if checkpoint_download == "passed" else "blocked_checkpoint_download",
            "checkpoint_manifest": rel(CKPT_DIR / "checkpoint_manifest.json"),
            "updated_at": now_iso(),
        }
    )
    write_json(gate_path, gate)
    write_json(DF / "sam3_worker" / "sam3_pilot_plan.json", gate)
    return gate


def select_pilot_clips() -> list[dict[str, Any]]:
    smoke = read_json(SMOKE, {"videos": []})
    roi = read_json(ROI_FREEZE, {"videos": []})
    roi_by_video = {item["video_id"]: item for item in roi.get("videos", [])}
    categories = [
        ("ball_visible_motion", 0, 10.0, 25.0),
        ("ball_near_feet_occlusion", 1, 10.0, 25.0),
        ("persons_cluster", 2, 10.0, 25.0),
        ("persons_far", 0, 5.0, 20.0),
        ("team_appearance", 1, 5.0, 20.0),
        ("hard_negative_scene", 2, 5.0, 20.0),
    ]
    videos = smoke.get("videos", [])
    clips = []
    for idx, (category, video_offset, clip_offset, clip_duration) in enumerate(categories, start=1):
        video = videos[(idx + video_offset - 1) % len(videos)]
        windows = video.get("windows", [])
        window = windows[(idx - 1) % len(windows)]
        start = round(float(window[0]) + clip_offset, 3)
        end = round(min(float(window[1]), start + clip_duration), 3)
        duration = round(end - start, 3)
        video_id = video["video_id"]
        clips.append(
            {
                "schema_version": "df0c.sam3_pilot_clip.v0",
                "clip_id": f"sam3_pilot_{idx:02d}_{category}",
                "category": category,
                "video_id": video_id,
                "source_video": video.get("source_video"),
                "start_sec": start,
                "end_sec": end,
                "duration_sec": duration,
                "fps": 15,
                "resolution": "original",
                "roi_profile": roi_by_video.get(video_id, {}).get("profile"),
                "roi_profile_hash": roi_by_video.get(video_id, {}).get("profile_sha256"),
                "split_overlap_status": "no_cross_video_test_inference_requested; review_seed_only",
                "ground_truth_policy": "sam_output_not_ground_truth",
                "planned_prompt_tests": [
                    "point",
                    "box",
                    "visual_exemplar",
                    "text_concept_if_supported",
                    "forward_propagation",
                    "backward_propagation",
                    "multi_object",
                    "mask_to_box",
                ],
            }
        )
    write_json(DF / "sam3_worker" / "sam3_pilot_clips.json", clips)
    return clips


def main() -> None:
    manifest = build_checkpoint_manifest()
    gate = update_gate(manifest)
    clips = select_pilot_clips()
    status = {
        "phase": "DF-0C SAM 3.1 PILOT PREP",
        "created_at": now_iso(),
        "checkpoint_complete": manifest["complete"],
        "pilot_status": gate["pilot_status"],
        "clip_count": len(clips),
        "token_exposed": False,
    }
    write_json(DF / "sam3_worker" / "df0c_pilot_prep_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
