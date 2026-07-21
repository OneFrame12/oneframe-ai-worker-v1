#!/usr/bin/env python3
"""Build OneFrame Automated Data Factory v0 artifacts.

This script is intentionally offline-only. It reads existing PE-0 artifacts,
creates a local SQLite/JSONL catalog, writes executable schema contracts, and
prepares sampling/review/training/registry scaffolding without touching
production, Supabase, R2, RunPod, or source videos.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DF_ROOT = ROOT / "ai_worker_v1" / "data_factory"
RUN_ID = "df0_automated_data_factory_v0"
RUN_DIR = DF_ROOT / "audit" / RUN_ID

INGESTION_RUN = ROOT / "ai_worker_v1" / "runs" / "pe0_multivideo_ingestion_20260717T024829Z"
VISUAL_SMOKE_RUN = ROOT / "ai_worker_v1" / "runs" / "pe0_multivideo_visual_smoke_20260717T062015Z"
BALL_V0 = ROOT / "ai_worker_v1" / "datasets" / "OneFrame_Ball_v0"
BALL_SPECIALIST_BEST = (
    ROOT
    / "ai_worker_v1"
    / "training"
    / "ball_v0"
    / "rfdetr_s_ball_v0_t3_20260716T221213Z"
    / "remote_outputs"
    / "rfdetr_s_ball_v0_20260716T225601Z"
    / "checkpoints"
    / "best.pth"
)
EXPECTED_BALL_SPECIALIST_SHA256 = (
    "9dda20e4e7363a284a9775ff3aac4c10280ecd4c86299127be2c5e77a7b64d55"
)

MODULE_DIRS = [
    "catalog",
    "sampling",
    "sampling/contact_sheets",
    "sampling/previews",
    "prelabel",
    "sam3_adapter",
    "sam3_worker",
    "tracklets",
    "review",
    "review/app",
    "datasets",
    "training",
    "evaluation",
    "registry",
    "schemas",
    "audit",
]

RECORD_TYPES = [
    "VideoRecord",
    "SequenceRecord",
    "FrameRecord",
    "DetectionRecord",
    "TrackletRecord",
    "AnnotationRecord",
    "ReviewDecision",
    "DatasetVersion",
    "ModelVersion",
    "EvaluationRun",
]

COMMON_REQUIRED_FIELDS = [
    "schema_version",
    "match_id",
    "camera_id",
    "video_id",
    "sequence_id",
    "frame_id",
    "timestamp_ms",
    "source",
    "source_model_version",
    "review_status",
    "created_at",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


CREATED_AT = now_iso()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def mkdirs() -> None:
    for d in MODULE_DIRS:
        (DF_ROOT / d).mkdir(parents=True, exist_ok=True)


def build_schemas() -> dict[str, Any]:
    properties = {
        field: {"type": ["string", "number", "integer", "null"]}
        for field in COMMON_REQUIRED_FIELDS
    }
    properties.update(
        {
            "record_id": {"type": "string"},
            "payload": {"type": "object"},
            "bbox_xyxy": {"type": ["array", "null"], "items": {"type": "number"}},
            "mask_ref": {"type": ["string", "null"]},
            "confidence": {"type": ["number", "null"]},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
        }
    )
    schemas = {
        name: {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": name,
            "type": "object",
            "required": COMMON_REQUIRED_FIELDS,
            "properties": properties,
            "additionalProperties": True,
        }
        for name in RECORD_TYPES
    }
    write_json(DF_ROOT / "schemas" / "records.schema.json", schemas)
    for name, schema in schemas.items():
        write_json(DF_ROOT / "schemas" / f"{name}.schema.json", schema)
    deepstream = {
        "schema_version": "df0.deepstream_ready.v0",
        "deepstream_status": "deferred_until_perception_and_tracking_validated",
        "interfaces": {
            "FramePacket": {
                "required": [
                    "schema_version",
                    "video_id",
                    "camera_id",
                    "frame_id",
                    "timestamp_ms",
                    "image_uri",
                    "roi_profile_hash",
                ]
            },
            "DetectionRecord": {
                "required": COMMON_REQUIRED_FIELDS
                + ["class_name", "confidence", "bbox_xyxy", "detector_name"]
            },
            "TrackletRecord": {
                "required": COMMON_REQUIRED_FIELDS
                + ["tracklet_id", "object_class", "frame_span", "source_algorithm"]
            },
            "CameraState": {
                "required": [
                    "schema_version",
                    "camera_id",
                    "calibration_profile_id",
                    "roi_profile_hash",
                    "homography_status",
                    "created_at",
                ]
            },
        },
        "notes": [
            "Schema-only contract for future DeepStream ingestion.",
            "No DeepStream runtime or sport logic is installed in DF-0.",
        ],
    }
    write_json(DF_ROOT / "schemas" / "deepstream_ready_interfaces.schema.json", deepstream)
    return schemas


def load_sources() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    inventory = read_json(INGESTION_RUN / "inventory" / "video_inventory.json", {"videos": []})
    roi_freeze = read_json(
        INGESTION_RUN
        / "calibration"
        / "manual_roi_v3_1_final"
        / "roi_v3_1_freeze_summary.json",
        {"videos": []},
    )
    smoke = read_json(
        VISUAL_SMOKE_RUN / "remote_outputs" / "visual_smoke_summary.json", {"videos": []}
    )
    return inventory, roi_freeze, smoke


def build_video_records(
    inventory: dict[str, Any], roi_freeze: dict[str, Any], smoke: dict[str, Any]
) -> list[dict[str, Any]]:
    roi_by_video = {v.get("video_id"): v for v in roi_freeze.get("videos", [])}
    smoke_by_video = {v.get("video_id"): v for v in smoke.get("videos", [])}
    records: list[dict[str, Any]] = []
    for item in inventory.get("videos", []):
        video_id = item["video_id"]
        ffprobe = item.get("ffprobe", {})
        fmt = ffprobe.get("format", {})
        streams = ffprobe.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        roi = roi_by_video.get(video_id, {})
        smoke_item = smoke_by_video.get(video_id, {})
        records.append(
            {
                "schema_version": "df0.v0",
                "record_type": "VideoRecord",
                "record_id": f"video::{video_id}",
                "match_id": "multivideo_v01",
                "camera_id": f"camera::{video_id}",
                "video_id": video_id,
                "sequence_id": "full_video",
                "frame_id": "not_applicable",
                "timestamp_ms": 0,
                "source": "local_multivideo_ingestion",
                "source_model_version": "not_applicable",
                "review_status": roi.get("status", "draft_visual_review"),
                "created_at": CREATED_AT,
                "filename": item.get("filename"),
                "path": rel(item.get("path")),
                "sha256": item.get("sha256") or item.get("hashes", {}).get("sha256"),
                "duration_sec": item.get("duration_sec") or fmt.get("duration"),
                "fps": item.get("fps") or video_stream.get("r_frame_rate"),
                "resolution": {
                    "width": item.get("width") or video_stream.get("width"),
                    "height": item.get("height") or video_stream.get("height"),
                },
                "codec": video_stream.get("codec_name") or item.get("codec"),
                "camera_position": item.get("analysis", {}).get("camera_position"),
                "lighting": item.get("analysis", {}).get("lighting"),
                "usable": item.get("analysis", {}).get("usable"),
                "roi_profile": rel(roi.get("profile")),
                "roi_profile_hash": roi.get("profile_sha256"),
                "roi_profile_content_hash": roi.get("profile_content_hash"),
                "roi_overlay": rel(roi.get("overlay")),
                "roi_status": roi.get("status"),
                "previous_artifacts": {
                    "contact_sheet": rel(item.get("contact_sheet")),
                    "previews_dir": rel(INGESTION_RUN / "previews"),
                    "visual_smoke_outputs": {
                        key: value | {"path": rel(value.get("path"))}
                        for key, value in smoke_item.get("outputs", {}).items()
                    },
                },
                "existing_predictions": {
                    "visual_smoke_summary": rel(
                        VISUAL_SMOKE_RUN / "remote_outputs" / "visual_smoke_summary.json"
                    ),
                    "counts": smoke_item.get("counters", {}),
                    "raw_detection_records_available": False,
                },
                "split_history": "not_assigned_df0_catalog_only",
            }
        )
    dataset_hash = (BALL_V0 / "dataset_hash.txt").read_text().strip() if (BALL_V0 / "dataset_hash.txt").exists() else None
    records.append(
        {
            "schema_version": "df0.v0",
            "record_type": "VideoRecord",
            "record_id": "video::oneframe_ball_v0_source_reference",
            "match_id": "oneframe_ball_v0",
            "camera_id": "camera::oneframe_ball_v0_source_unknown",
            "video_id": "oneframe_ball_v0_source_reference",
            "sequence_id": "oneframe_ball_v0_dataset_sequences",
            "frame_id": "dataset_frames",
            "timestamp_ms": 0,
            "source": "OneFrame_Ball_v0_dataset_artifacts",
            "source_model_version": "human_reviewed_ground_truth",
            "review_status": "frozen_dataset_reference",
            "created_at": CREATED_AT,
            "filename": "source_video_not_present_as_full_mp4_in_workspace",
            "sha256": None,
            "duration_sec": None,
            "fps": None,
            "resolution": None,
            "codec": None,
            "camera_position": "unknown_original_v0_video",
            "lighting": "unknown",
            "usable": True,
            "roi_profile": None,
            "roi_profile_hash": None,
            "source_video_status": "missing_full_video_reference_indexed_from_dataset_manifests",
            "previous_artifacts": {
                "dataset": rel(BALL_V0),
                "artifact_manifest": rel(BALL_V0 / "artifact_manifest.json"),
                "dataset_hash": dataset_hash,
                "source_frames_manifest": rel(BALL_V0 / "manifests" / "source_frames_manifest.json"),
            },
            "existing_predictions": {},
            "split_history": read_json(BALL_V0 / "manifests" / "split_manifest.json", {}),
        }
    )
    return records


def build_sequence_records(video_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sequence_manifest = read_json(
        INGESTION_RUN / "ball_v0_1_plan" / "ball_v0_1_sequence_manifest.json", []
    )
    for item in sequence_manifest:
        records.append(
            {
                "schema_version": "df0.v0",
                "record_type": "SequenceRecord",
                "record_id": f"sequence::{item['sequence_id']}",
                "match_id": "multivideo_v01",
                "camera_id": f"camera::{item['source_video']}",
                "video_id": item["source_video"],
                "sequence_id": item["sequence_id"],
                "frame_id": "sequence",
                "timestamp_ms": int(float(item["start_sec"]) * 1000),
                "source": "pe0_multivideo_ball_v0_1_plan",
                "source_model_version": "not_applicable",
                "review_status": "proposal_only",
                "created_at": CREATED_AT,
                **item,
            }
        )
    for v in video_records:
        if v["match_id"] != "multivideo_v01":
            continue
        records.append(
            {
                "schema_version": "df0.v0",
                "record_type": "SequenceRecord",
                "record_id": f"sequence::{v['video_id']}::full_video",
                "match_id": v["match_id"],
                "camera_id": v["camera_id"],
                "video_id": v["video_id"],
                "sequence_id": "full_video",
                "frame_id": "full_video",
                "timestamp_ms": 0,
                "source": "video_catalog",
                "source_model_version": "not_applicable",
                "review_status": "catalog_reference",
                "created_at": CREATED_AT,
                "duration_sec": v.get("duration_sec"),
                "split": "unassigned",
            }
        )
    return records


def build_frame_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    person_frames = read_json(
        INGESTION_RUN / "person_gold_set" / "person_gold_set_frame_manifest.json", []
    )
    for item in person_frames:
        frame_id = f"{item['video_id']}::frame_{int(item['frame_index']):06d}"
        records.append(
            {
                "schema_version": "df0.v0",
                "record_type": "FrameRecord",
                "record_id": f"frame::{frame_id}",
                "match_id": "multivideo_v01",
                "camera_id": f"camera::{item['video_id']}",
                "video_id": item["video_id"],
                "sequence_id": "person_gold_set_candidate",
                "frame_id": frame_id,
                "timestamp_ms": int(float(item["timestamp_sec"]) * 1000),
                "source": "pe0_person_gold_set_plan",
                "source_model_version": "not_applicable",
                "review_status": item.get("ground_truth_status", "proposal_only"),
                "created_at": CREATED_AT,
                **item,
            }
        )
    return records


def build_pipeline_audit(smoke: dict[str, Any]) -> dict[str, Any]:
    checkpoint_sha = sha256_file(BALL_SPECIALIST_BEST)
    totals = {
        "frames_processed": 0,
        "yolo_ball_frames": 0,
        "rfdetr_ball_frames": 0,
        "specialist_ball_frames": 0,
        "ball_disagreement_frames": 0,
        "yolo_person_detections": 0,
        "rfdetr_person_detections": 0,
        "previous_bool_person_disagreement_frames": 0,
    }
    per_video: list[dict[str, Any]] = []
    for item in smoke.get("videos", []):
        c = item.get("counters", {})
        for k in totals:
            totals[k] += int(c.get(k, 0))
        per_video.append({"video_id": item.get("video_id"), "counters": c})
    if checkpoint_sha != EXPECTED_BALL_SPECIALIST_SHA256:
        classification = "pipeline_regression"
    elif totals["specialist_ball_frames"] == 0 and (
        totals["yolo_ball_frames"] > 0 or totals["rfdetr_ball_frames"] > 0
    ):
        classification = "threshold_or_mapping_failure"
    else:
        classification = "functioning_as_expected"

    audit = {
        "schema_version": "df0.pipeline_audit.v0",
        "created_at": CREATED_AT,
        "checkpoint": {
            "model_name": "rfdetr_s_oneframe_ball_v0",
            "path": rel(BALL_SPECIALIST_BEST),
            "expected_sha256": EXPECTED_BALL_SPECIALIST_SHA256,
            "actual_sha256": checkpoint_sha,
            "sha256_match": checkpoint_sha == EXPECTED_BALL_SPECIALIST_SHA256,
        },
        "classification": classification,
        "totals": totals,
        "per_video": per_video,
        "legacy_parity": {
            "status": "not_executed_in_df0",
            "reason": "DF-0 does not run new inference; parity must be run before using the model as a preannotator.",
        },
        "raw_outputs_trace": {
            "status": "missing_for_ball_specialist_v0_visual_smoke",
            "required_before_preannotation": True,
        },
        "class_mapping_trace": {
            "status": "suspect",
            "reason": "Ball Specialist v0 produced zero ball frames while YOLO/RF-DETR base produced nonzero frames.",
        },
        "preprocessing_trace": {
            "status": "pending_raw_probe",
            "required_fields": [
                "input_color_order",
                "resize",
                "normalization",
                "class_id_to_ball_mapping",
                "threshold",
                "global_vs_tile_coordinate_mapping",
            ],
        },
        "roi_trace": {
            "status": "available",
            "roi_profiles": rel(
                INGESTION_RUN / "calibration" / "manual_roi_v3_1_final" / "roi_v3_1_freeze_summary.json"
            ),
        },
        "person_disagreement_correction": {
            "previous_metric": "presence_only_counted_zero_disagreement_when both models saw at least one person",
            "correct_metric": "per-frame bipartite IoU matching over person boxes with count mismatch and low-IoU unmatched detections",
            "status": "definition_ready_raw_boxes_missing",
            "reason": "Visual smoke summaries store aggregate counts but not per-frame person boxes.",
        },
    }
    write_json(DF_ROOT / "audit" / "pipeline_preannotator_audit.json", audit)
    return audit


def build_detection_records(smoke: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in smoke.get("videos", []):
        video_id = item.get("video_id")
        for model_name, frame_key in [
            ("yolo_ball_baseline", "yolo_ball_frames"),
            ("rfdetr_small_base_ball", "rfdetr_ball_frames"),
            ("rfdetr_s_oneframe_ball_v0", "specialist_ball_frames"),
            ("yolo_person_baseline", "yolo_person_detections"),
            ("rfdetr_small_base_person", "rfdetr_person_detections"),
        ]:
            records.append(
                {
                    "schema_version": "df0.v0",
                    "record_type": "DetectionRecord",
                    "record_id": f"detection_aggregate::{video_id}::{model_name}",
                    "match_id": "multivideo_v01",
                    "camera_id": f"camera::{video_id}",
                    "video_id": video_id,
                    "sequence_id": "visual_smoke_windows",
                    "frame_id": "aggregate_only_raw_boxes_not_persisted",
                    "timestamp_ms": 0,
                    "source": "pe0_multivideo_visual_smoke_summary",
                    "source_model_version": model_name,
                    "review_status": "diagnostic_aggregate_only",
                    "created_at": CREATED_AT,
                    "metric_name": frame_key,
                    "value": item.get("counters", {}).get(frame_key, 0),
                    "pipeline_audit_classification": audit["classification"],
                }
            )
    return records


def midpoint_window(window: list[float]) -> tuple[float, float]:
    start, end = float(window[0]), float(window[1])
    center = (start + end) / 2.0
    return round(center - 5.0, 3), round(center + 5.0, 3)


def build_sampling_outputs(smoke: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    sequence_counter = 1
    for item in smoke.get("videos", []):
        video_id = item.get("video_id")
        counters = item.get("counters", {})
        for window_index, window in enumerate(item.get("windows", []), start=1):
            start, end = midpoint_window(window)
            reason_codes = [
                "model_disagreement_ball",
                "specialist_zero_candidates",
                "count_mismatch_person_requires_iou_review",
            ]
            if counters.get("ball_disagreement_frames", 0):
                reason_codes.append("ball_disagreement")
            if counters.get("rfdetr_ball_frames", 0) and counters.get("yolo_ball_frames", 0):
                reason_codes.append("rfdetr_yolo_overlap_unknown")
            if counters.get("rfdetr_person_detections") != counters.get("yolo_person_detections"):
                reason_codes.append("person_count_mismatch")
            sequence_id = f"df0_candidate_seq_{sequence_counter:03d}"
            sequence_counter += 1
            candidates.append(
                {
                    "schema_version": "df0.sampling.v0",
                    "candidate_type": "conflict_window",
                    "sequence_id": sequence_id,
                    "video_id": video_id,
                    "source_window_sec": window,
                    "start_sec": start,
                    "end_sec": end,
                    "duration_sec": round(end - start, 3),
                    "reason_codes": reason_codes,
                    "priority": "high",
                    "review_status": "pending_tracklet_review",
                    "split_policy": "unassigned_until_review",
                    "notes": "10s review window centered inside existing 30s smoke window.",
                }
            )
            mid = round((start + end) / 2.0, 3)
            for offset in [-2.0, 0.0, 2.0]:
                ts = round(mid + offset, 3)
                frames.append(
                    {
                        "schema_version": "df0.sampling.v0",
                        "frame_id": f"{video_id}::{sequence_id}::{int(ts * 1000)}ms",
                        "sequence_id": sequence_id,
                        "video_id": video_id,
                        "timestamp_sec": ts,
                        "timestamp_ms": int(ts * 1000),
                        "reason_codes": reason_codes,
                        "review_status": "pending_tracklet_review",
                    }
                )

    report = {
        "schema_version": "df0.coverage_report.v0",
        "created_at": CREATED_AT,
        "candidate_sequences": len(candidates),
        "candidate_frames": len(frames),
        "videos": sorted({c["video_id"] for c in candidates}),
        "ball_priorities_covered": [
            "disagreement",
            "zero_candidates",
            "near_feet_requires_review",
            "tiny_far_requires_review",
            "motion_blur_requires_review",
            "player_cluster_requires_review",
            "net_lines_shoes_edge_low_contrast_requires_review",
        ],
        "person_priorities_covered": [
            "count_mismatch",
            "low_iou_match_pending_raw_boxes",
            "far",
            "occlusion",
            "cluster",
            "goalkeeper",
            "border",
            "outside_field_candidate",
        ],
        "limitations": [
            "Raw per-frame boxes were not persisted by the visual smoke, so IoU conflict frames require the next diagnostic pass.",
            "Candidate frames are review seeds, not labels.",
        ],
    }
    write_jsonl(DF_ROOT / "sampling" / "candidate_sequences.jsonl", candidates)
    write_jsonl(DF_ROOT / "sampling" / "candidate_frames.jsonl", frames)
    write_json(DF_ROOT / "sampling" / "coverage_report.json", report)
    copy_sampling_visuals()
    return candidates, frames, report


def copy_sampling_visuals() -> None:
    contact_dst = DF_ROOT / "sampling" / "contact_sheets"
    preview_dst = DF_ROOT / "sampling" / "previews"
    copied: list[dict[str, str]] = []
    for src in (INGESTION_RUN / "contact_sheets").glob("*.jpg"):
        dst = contact_dst / src.name
        shutil.copy2(src, dst)
        copied.append({"type": "contact_sheet", "path": rel(dst), "sha256": sha256_file(dst)})
    for src in (INGESTION_RUN / "previews").glob("*.mp4"):
        dst = preview_dst / src.name
        shutil.copy2(src, dst)
        copied.append({"type": "preview", "path": rel(dst), "sha256": sha256_file(dst)})
    write_json(DF_ROOT / "sampling" / "visual_artifacts_manifest.json", copied)


def build_sam3_adapter_docs() -> dict[str, Any]:
    gate = {
        "schema_version": "df0.sam3_gate.v0",
        "created_at": CREATED_AT,
        "official_sources": [
            "https://github.com/facebookresearch/sam3",
            "https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md",
            "https://huggingface.co/facebook/sam3.1",
        ],
        "environment_requirements": {
            "python": ">=3.12",
            "pytorch": ">=2.7",
            "cuda": ">=12.6",
            "checkpoint_access": "Hugging Face authentication/access required",
        },
        "object_multiplex": {
            "available_in_sam3_1": True,
            "status": "planned_for_pilot_only",
        },
        "local_token_present": bool(
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        ),
        "pilot_status": "blocked_missing_verified_sam3_checkpoint_access",
        "runpod_pod_created": False,
        "reason": "DF-0 cannot run SAM 3.1 pilot without verified gated checkpoint access; no token values are printed or stored.",
        "selected_clips": [
            "df0_candidate_seq_001",
            "df0_candidate_seq_002",
            "df0_candidate_seq_004",
            "df0_candidate_seq_005",
            "df0_candidate_seq_007",
            "df0_candidate_seq_008",
        ],
        "metrics_to_collect_when_unblocked": [
            "gpu_name",
            "vram_peak_mb",
            "runtime_sec",
            "objects",
            "masks",
            "continuity",
            "identity_switches",
            "propagation_failures",
            "estimated_human_corrections",
        ],
    }
    write_json(DF_ROOT / "sam3_adapter" / "sam3_gate.json", gate)
    adapter = '''"""SAM 3.1 adapter contract for the isolated Data Factory worker.

This module intentionally does not import sam3 at module import time. SAM 3.1
must run in a separate Python 3.12+/CUDA 12.6+ environment.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sam3Prompt:
    kind: str
    frame_id: str
    points: list[tuple[float, float]] = field(default_factory=list)
    box_xyxy: tuple[float, float, float, float] | None = None
    visual_exemplar_ref: str | None = None
    text_concept: str | None = None


@dataclass
class Sam3Result:
    status: str
    object_id: str | None
    mask_ref: str | None
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float | None
    failure_reason: str | None = None


class Sam3Adapter:
    supported_prompts = {"point", "box", "visual_exemplar", "text_concept"}

    def __init__(self, backend: Any | None = None) -> None:
        self.backend = backend

    def ensure_backend(self) -> Any:
        if self.backend is None:
            raise RuntimeError(
                "SAM 3.1 backend is not loaded. Use sam3_worker in an isolated "
                "Python 3.12+/CUDA 12.6+ environment."
            )
        return self.backend

    def segment_frame(self, prompt: Sam3Prompt) -> Sam3Result:
        self.ensure_backend()
        raise NotImplementedError("Bound by sam3_worker after official API inspection.")

    def propagate(self, object_id: str, direction: str = "forward") -> list[Sam3Result]:
        self.ensure_backend()
        raise NotImplementedError("Bound by sam3_worker after pilot.")

    @staticmethod
    def mask_to_box(mask: Any) -> tuple[float, float, float, float] | None:
        raise NotImplementedError("Implemented in sam3_worker with numpy/cv2.")
'''
    (DF_ROOT / "sam3_adapter" / "adapter.py").write_text(adapter)
    worker_readme = """# SAM 3.1 Worker (DF-0)

Independent worker for SAM 3.1 pilots. Do not install SAM 3.1 in the RF-DETR
environment.

Official prereqs from the SAM 3 repository: Python 3.12+, PyTorch 2.7+, CUDA
12.6+, and authenticated access to the SAM 3.1 Hugging Face checkpoints.

Supported contract:
- point prompt
- box prompt
- visual exemplar prompt
- text concept prompt
- forward/backward/multi-object propagation
- mask-to-box conversion
- confidence and failure reporting
- object identity traces

Outputs are preannotations only. They are never ground truth until human review.
"""
    (DF_ROOT / "sam3_worker" / "README.md").write_text(worker_readme)
    (DF_ROOT / "sam3_worker" / "requirements.sam3.txt").write_text(
        "torch>=2.7\n"
        "torchvision\n"
        "huggingface_hub\n"
        "# Install latest official facebookresearch/sam3 in this isolated worker.\n"
        "git+https://github.com/facebookresearch/sam3.git\n"
    )
    write_json(DF_ROOT / "sam3_worker" / "sam3_pilot_plan.json", gate)
    return gate


def build_review_app() -> None:
    app = DF_ROOT / "review" / "app"
    (app / "index.html").write_text(
        """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>OneFrame Data Factory Review</title>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header>
    <h1>OneFrame Data Factory Review</h1>
    <p>Revisar conflictos, no todos los frames. Las decisiones humanas son obligatorias.</p>
  </header>
  <main>
    <section class="panel">
      <h2>Cola priorizada</h2>
      <textarea id="queue" readonly></textarea>
    </section>
    <section class="viewer">
      <div class="toolbar">
        <button id="prev">Frame anterior</button>
        <button id="next">Frame siguiente</button>
        <button id="zoomIn">Zoom +</button>
        <button id="zoomOut">Zoom -</button>
      </div>
      <div class="canvas">Cargar clip/tracklet desde candidate_sequences.jsonl</div>
      <div class="actions">
        <button>Aceptar</button>
        <button>Corregir</button>
        <button>Rechazar</button>
        <button>Split</button>
        <button>Merge</button>
        <button>Missing</button>
        <button>No object</button>
        <button>Uncertain</button>
        <button>Propagar</button>
        <button>Guardar</button>
      </div>
    </section>
    <section class="panel">
      <h2>Contexto</h2>
      <ul>
        <li>Modelo fuente</li>
        <li>Acuerdos/desacuerdos</li>
        <li>Confianza</li>
        <li>ROI</li>
        <li>Frames conflictivos</li>
      </ul>
    </section>
  </main>
  <script src="app.js"></script>
</body>
</html>
"""
    )
    (app / "styles.css").write_text(
        """body{font-family:system-ui,Arial,sans-serif;margin:0;background:#101214;color:#f4f6f8}
header{padding:16px 24px;border-bottom:1px solid #30343a}
main{display:grid;grid-template-columns:320px 1fr 280px;gap:16px;padding:16px}
.panel,.viewer{background:#181b20;border:1px solid #30343a;border-radius:6px;padding:12px}
textarea{width:100%;height:70vh;background:#0d0f12;color:#dce3ea;border:1px solid #30343a}
.toolbar,.actions{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px}
button{background:#2d6cdf;color:white;border:0;border-radius:4px;padding:8px 10px}
.canvas{height:62vh;display:grid;place-items:center;background:#08090b;border:1px solid #30343a}
"""
    )
    (app / "app.js").write_text(
        """fetch('../../sampling/candidate_sequences.jsonl')
  .then(r => r.ok ? r.text() : '')
  .then(text => {
    document.getElementById('queue').value = text || 'Abrir desde servidor local para cargar JSONL.';
  })
  .catch(() => {
    document.getElementById('queue').value = 'Cola no cargada. Use el manifest local.';
  });
"""
    )
    review_contract = {
        "schema_version": "df0.review_app.v0",
        "status": "ready_for_conflict_review_once_tracklets_exist",
        "supported_actions": [
            "accept",
            "correct",
            "reject",
            "split",
            "merge",
            "missing",
            "no_object",
            "uncertain",
            "propagate",
            "save",
        ],
        "prioritization": "conflicts_first_not_full_frame_sweep",
    }
    write_json(DF_ROOT / "review" / "review_app_contract.json", review_contract)


def build_dataset_training_registry(audit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    ball_v0_hash = (
        (BALL_V0 / "dataset_hash.txt").read_text().strip()
        if (BALL_V0 / "dataset_hash.txt").exists()
        else None
    )
    dataset_plan = {
        "schema_version": "df0.dataset_builder.v0",
        "created_at": CREATED_AT,
        "datasets": [
            {
                "name": "OneFrame_Ball_v0.1",
                "status": "not_ready_waiting_for_human_tracklet_review",
                "base_dataset": "OneFrame_Ball_v0",
                "base_dataset_hash": ball_v0_hash,
                "policy": "additive_only_no_modification_to_v0",
                "split_policy": "sequence_level_train_valid_cross_video_test_no_leakage",
                "required_annotation_metadata": [
                    "source_model_version",
                    "tracklet_id",
                    "review_status",
                    "reviewer_action",
                    "correction_history",
                ],
                "hash_spec": rel(BALL_V0 / "HASH_SPEC_V1.md"),
            },
            {
                "name": "OneFrame_Person_v0_candidate",
                "status": "not_ready_waiting_for_gold_set_review",
                "split_policy": "sequence_video_aware",
                "ground_truth_policy": "no_pseudo_labels_as_ground_truth",
            },
        ],
    }
    write_json(DF_ROOT / "datasets" / "dataset_builder_plan.json", dataset_plan)
    training_plan = {
        "schema_version": "df0.training_orchestrator.v0",
        "created_at": CREATED_AT,
        "status": "prepared_not_allowed_to_train_until_dataset_ready",
        "jobs": [
            {
                "target": "RF-DETR-S Ball Specialist v0.1",
                "dataset": "OneFrame_Ball_v0.1",
                "gate": "dataset_ready_and_human_review_complete",
                "action": "blocked_no_training_in_df0",
            },
            {
                "target": "Person Specialist candidate",
                "dataset": "OneFrame_Person_v0_candidate",
                "gate": "gold_set_complete",
                "action": "blocked_no_training_in_df0",
            },
        ],
    }
    write_json(DF_ROOT / "training" / "training_orchestrator_plan.json", training_plan)
    registry = {
        "schema_version": "df0.model_registry.v0",
        "created_at": CREATED_AT,
        "allowed_statuses": [
            "diagnostic",
            "prelabel_only",
            "rejected",
            "rejected_diagnostic",
            "shadow_candidate",
            "shadow_approved",
            "production_candidate",
        ],
        "models": [
            {
                "model_id": "rfdetr_s_oneframe_ball_v0",
                "status": "rejected_diagnostic",
                "allowed_uses": ["preannotation", "error_mining", "offline_comparison"],
                "not_allowed_uses": ["ground_truth", "production", "threshold_selection"],
                "checkpoint_sha256": audit["checkpoint"]["actual_sha256"],
                "checkpoint_sha256_match_expected": audit["checkpoint"]["sha256_match"],
                "diagnostic_classification": audit["classification"],
                "reason": "Visual smoke produced zero specialist ball frames while other detectors found ball frames.",
            },
            {
                "model_id": "rfdetr_small_base",
                "status": "diagnostic",
                "allowed_uses": ["preannotation_candidate", "error_mining", "offline_comparison"],
            },
            {
                "model_id": "yolo_baseline",
                "status": "diagnostic",
                "allowed_uses": ["baseline_comparison", "error_mining"],
            },
        ],
    }
    write_json(DF_ROOT / "registry" / "model_registry.json", registry)
    return dataset_plan, registry


def build_prelabel_tracklet_evaluation_plans(
    audit: dict[str, Any], sampling_report: dict[str, Any], sam_gate: dict[str, Any]
) -> None:
    prelabel_plan = {
        "schema_version": "df0.prelabel.v0",
        "created_at": CREATED_AT,
        "status": "prepared_blocked_for_ball_specialist_v0",
        "policy": "prelabels_are_candidates_only_never_ground_truth",
        "ball_specialist_v0": {
            "allowed": False,
            "classification": audit["classification"],
            "required_fix_before_use": [
                "raw_output_trace",
                "class_mapping_trace",
                "threshold_trace",
                "preprocessing_trace",
                "global_tile_coordinate_mapping_trace",
                "legacy_parity_probe",
            ],
        },
        "rfdetr_small_base": {
            "allowed": True,
            "use": "candidate_generation_and_error_mining_only",
        },
        "yolo_baseline": {
            "allowed": True,
            "use": "baseline_comparison_and_error_mining_only",
        },
    }
    write_json(DF_ROOT / "prelabel" / "prelabel_policy.json", prelabel_plan)

    tracklet_plan = {
        "schema_version": "df0.tracklets.v0",
        "created_at": CREATED_AT,
        "status": "blocked_until_sam_pilot_or_raw_detector_tracklets_exist",
        "candidate_sequences": sampling_report["candidate_sequences"],
        "candidate_frames": sampling_report["candidate_frames"],
        "sam3_pilot_status": sam_gate["pilot_status"],
        "no_tracking_executed": True,
        "review_actions": [
            "accept",
            "correct",
            "reject",
            "split",
            "merge",
            "missing",
            "no_object",
            "uncertain",
            "propagate",
            "save",
        ],
    }
    write_json(DF_ROOT / "tracklets" / "tracklet_review_plan.json", tracklet_plan)

    evaluation_plan = {
        "schema_version": "df0.evaluation.v0",
        "created_at": CREATED_AT,
        "status": "prepared_no_precision_recall_without_gold_set",
        "inputs": {
            "visual_smoke_summary": rel(
                VISUAL_SMOKE_RUN / "remote_outputs" / "visual_smoke_summary.json"
            ),
            "pipeline_audit": rel(DF_ROOT / "audit" / "pipeline_preannotator_audit.json"),
        },
        "metrics_ready": [
            "aggregate_model_counts",
            "ball_disagreement_frames",
            "runtime_errors",
            "checkpoint_hash_match",
        ],
        "metrics_blocked": [
            "person_iou_disagreement_without_raw_boxes",
            "precision",
            "recall",
            "tp_fp_fn",
        ],
    }
    write_json(DF_ROOT / "evaluation" / "evaluation_plan.json", evaluation_plan)


def init_sqlite(
    video_records: list[dict[str, Any]],
    sequence_records: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    detection_records: list[dict[str, Any]],
    dataset_plan: dict[str, Any],
    registry: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    db_path = DF_ROOT / "catalog" / "oneframe_data_factory_v0.sqlite"
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "create table records (record_type text not null, record_id text primary key, payload_json text not null, created_at text not null)"
    )
    conn.execute("create index idx_records_type on records(record_type)")
    all_rows = (
        video_records
        + sequence_records
        + frame_records
        + detection_records
        + [
            {
                "record_type": "EvaluationRun",
                "record_id": "evaluation::pipeline_preannotator_audit",
                "schema_version": "df0.v0",
                "match_id": "multivideo_v01",
                "camera_id": "all",
                "video_id": "all",
                "sequence_id": "all",
                "frame_id": "aggregate",
                "timestamp_ms": 0,
                "source": "df0_pipeline_audit",
                "source_model_version": "mixed",
                "review_status": "diagnostic",
                "created_at": CREATED_AT,
                "payload": audit,
            }
        ]
    )
    for ds in dataset_plan.get("datasets", []):
        all_rows.append(
            {
                "record_type": "DatasetVersion",
                "record_id": f"dataset::{ds['name']}",
                "schema_version": "df0.v0",
                "match_id": "multivideo_v01",
                "camera_id": "all",
                "video_id": "all",
                "sequence_id": "all",
                "frame_id": "dataset",
                "timestamp_ms": 0,
                "source": "df0_dataset_builder_plan",
                "source_model_version": "not_applicable",
                "review_status": ds["status"],
                "created_at": CREATED_AT,
                "payload": ds,
            }
        )
    for model in registry.get("models", []):
        all_rows.append(
            {
                "record_type": "ModelVersion",
                "record_id": f"model::{model['model_id']}",
                "schema_version": "df0.v0",
                "match_id": "global",
                "camera_id": "global",
                "video_id": "global",
                "sequence_id": "global",
                "frame_id": "model",
                "timestamp_ms": 0,
                "source": "df0_model_registry",
                "source_model_version": model["model_id"],
                "review_status": model["status"],
                "created_at": CREATED_AT,
                "payload": model,
            }
        )
    for row in all_rows:
        conn.execute(
            "insert into records(record_type, record_id, payload_json, created_at) values (?, ?, ?, ?)",
            (
                row["record_type"],
                row["record_id"],
                json.dumps(row, sort_keys=True),
                row["created_at"],
            ),
        )
    conn.commit()
    conn.close()
    write_jsonl(DF_ROOT / "catalog" / "videos.jsonl", video_records)
    write_jsonl(DF_ROOT / "catalog" / "sequences.jsonl", sequence_records)
    write_jsonl(DF_ROOT / "catalog" / "frames.jsonl", frame_records)
    write_jsonl(DF_ROOT / "catalog" / "detections.jsonl", detection_records)
    write_json(
        DF_ROOT / "catalog" / "catalog_build_report.json",
        {
            "schema_version": "df0.catalog_build.v0",
            "created_at": CREATED_AT,
            "sqlite": rel(db_path),
            "counts": {
                "VideoRecord": len(video_records),
                "SequenceRecord": len(sequence_records),
                "FrameRecord": len(frame_records),
                "DetectionRecord": len(detection_records),
                "DatasetVersion": len(dataset_plan.get("datasets", [])),
                "ModelVersion": len(registry.get("models", [])),
                "EvaluationRun": 1,
            },
            "duplicates_policy": "deterministic record_id primary key; rebuild replaces sqlite atomically",
        },
    )


def write_report(
    audit: dict[str, Any],
    sampling_report: dict[str, Any],
    sam_gate: dict[str, Any],
    dataset_plan: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    status = "blocked"
    reason = "SAM 3.1 pilot is blocked until checkpoint access/token is verified; tracklet review has only candidate queues, not SAM-propagated tracklets."
    report = f"""# DF-0 — OneFrame Automated Data Factory V0

FASE: DF-0 ONEFRAME AUTOMATED DATA FACTORY V0

ESTADO: {status}

Motivo: {reason}

## Pipeline Audit

- Ball Specialist checkpoint hash match: `{audit['checkpoint']['sha256_match']}`
- Pipeline classification: `{audit['classification']}`
- YOLO ball frames in visual smoke: `{audit['totals']['yolo_ball_frames']}`
- RF-DETR base ball frames in visual smoke: `{audit['totals']['rfdetr_ball_frames']}`
- Ball Specialist v0 frames in visual smoke: `{audit['totals']['specialist_ball_frames']}`
- Person disagreement correction: `{audit['person_disagreement_correction']['status']}`

## Catalog

- SQLite: `ai_worker_v1/data_factory/catalog/oneframe_data_factory_v0.sqlite`
- Video records: 4 (3 multivideo files + OneFrame_Ball_v0 source reference)
- JSONL mirrors: `videos.jsonl`, `sequences.jsonl`, `frames.jsonl`, `detections.jsonl`
- Duplicate policy: deterministic record IDs, no duplicate videos/predictions.

## SAM 3.1 Pilot

- Status: `{sam_gate['pilot_status']}`
- RunPod pod created: `{sam_gate['runpod_pod_created']}`
- Token present locally: `{sam_gate['local_token_present']}`
- Official environment: Python {sam_gate['environment_requirements']['python']}, PyTorch {sam_gate['environment_requirements']['pytorch']}, CUDA {sam_gate['environment_requirements']['cuda']}

## SAM Gate

SAM outputs are allowed only as preannotations. They are blocked from ground truth
until human review. SAM 3.1 remains isolated in `data_factory/sam3_worker/`.

## Sampling

- Candidate sequences: `{sampling_report['candidate_sequences']}`
- Candidate frames: `{sampling_report['candidate_frames']}`
- Contact sheets copied: `ai_worker_v1/data_factory/sampling/contact_sheets/`
- Previews copied: `ai_worker_v1/data_factory/sampling/previews/`

## Tracklets

- Tracklet schema created.
- Tracklet review UI scaffold: `ai_worker_v1/data_factory/review/app/index.html`
- Current tracklet status: `blocked_until_sam_or_detector_tracklet_candidates_exist`

## Datasets

- OneFrame_Ball_v0.1: `{dataset_plan['datasets'][0]['status']}`
- OneFrame_Person_v0_candidate: `{dataset_plan['datasets'][1]['status']}`
- OneFrame_Ball_v0 was not modified.

## Registry

- Registered models: `{len(registry['models'])}`
- `rfdetr_s_oneframe_ball_v0`: `rejected_diagnostic`
- Allowed uses: preannotation, error_mining, offline_comparison.

## Production

- `src/` productivo: not touched by this DF-0 script.
- Supabase/R2/endpoints: not touched.
- RunPod: no pod created in DF-0.
- DeepStream/MV3DT/AutoMagicCalib/tracking/events: deferred, not executed.

## Siguiente Accion

Unblock SAM 3.1 checkpoint access for an isolated pilot, or run a raw-output
trace for Ball Specialist v0 to resolve the threshold/mapping failure before
using it as a preannotator.
"""
    (DF_ROOT / "audit" / "DF0_FINAL_REPORT.md").write_text(report)
    write_json(
        DF_ROOT / "audit" / "status.json",
        {
            "phase": "DF-0 ONEFRAME AUTOMATED DATA FACTORY V0",
            "status": status,
            "reason": reason,
            "created_at": CREATED_AT,
            "pipeline_classification": audit["classification"],
            "catalog_sqlite": rel(DF_ROOT / "catalog" / "oneframe_data_factory_v0.sqlite"),
            "sam_pilot_status": sam_gate["pilot_status"],
            "production": {
                "src_touched": False,
                "supabase_touched": False,
                "r2_touched": False,
                "runpod_created": False,
            },
        },
    )


def write_csv_summary(video_records: list[dict[str, Any]]) -> None:
    path = DF_ROOT / "catalog" / "video_catalog.csv"
    fields = [
        "video_id",
        "filename",
        "sha256",
        "duration_sec",
        "fps",
        "resolution",
        "codec",
        "roi_status",
        "roi_profile_hash",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in video_records:
            writer.writerow(
                {
                    "video_id": row.get("video_id"),
                    "filename": row.get("filename"),
                    "sha256": row.get("sha256"),
                    "duration_sec": row.get("duration_sec"),
                    "fps": row.get("fps"),
                    "resolution": json.dumps(row.get("resolution"), sort_keys=True),
                    "codec": row.get("codec"),
                    "roi_status": row.get("roi_status"),
                    "roi_profile_hash": row.get("roi_profile_hash"),
                }
            )


def main() -> None:
    mkdirs()
    build_schemas()
    inventory, roi_freeze, smoke = load_sources()
    video_records = build_video_records(inventory, roi_freeze, smoke)
    sequence_records = build_sequence_records(video_records)
    frame_records = build_frame_records()
    audit = build_pipeline_audit(smoke)
    detection_records = build_detection_records(smoke, audit)
    _, _, sampling_report = build_sampling_outputs(smoke)
    sam_gate = build_sam3_adapter_docs()
    build_review_app()
    dataset_plan, registry = build_dataset_training_registry(audit)
    build_prelabel_tracklet_evaluation_plans(audit, sampling_report, sam_gate)
    init_sqlite(
        video_records,
        sequence_records,
        frame_records,
        detection_records,
        dataset_plan,
        registry,
        audit,
    )
    write_csv_summary(video_records)
    write_report(audit, sampling_report, sam_gate, dataset_plan, registry)
    print(json.dumps(read_json(DF_ROOT / "audit" / "status.json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
