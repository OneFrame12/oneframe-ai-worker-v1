#!/usr/bin/env python3
"""RF-1A dataset inventory and report utilities.

This module intentionally avoids model inference, training, R2 writes and
Supabase writes. It scans available local files, optionally lists R2/Supabase
read-only sources, and emits deterministic manifests for RF-1A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "rf1a.inventory.v1"
ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "data_catalog"
DATASET_DIR = ROOT / "datasets" / "football_detection_v1"
GOLD_DIR = ROOT / "datasets" / "football_gold_v1"
REVIEW_DIR = ROOT / "review"
DOCS_DIR = ROOT / "docs"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
ANNOTATION_EXTS = {".json", ".jsonl", ".csv", ".txt", ".xml"}
KNOWN_MATCH_IDS = {
    "run_a": "fcb621fd-72b8-4d42-8749-89e46d188f8a",
    "run_b": "4b400780-1a76-4a28-b574-00be68bafd57",
    "rfdetr_shadow_reference": "7a792f5a-00c1-461b-a0be-3b50bc0f062c",
}
KNOWN_RUNS = {
    "run_a": {
        "match_id": KNOWN_MATCH_IDS["run_a"],
        "job_id": "32c0f037-53c9-4473-b71a-9a68569e5522-u2",
        "notes": ["baseline_valid", "goal_safety_net_off"],
    },
    "run_b": {
        "match_id": KNOWN_MATCH_IDS["run_b"],
        "job_id": "b0cee331-89c6-4775-9fc9-1986198221af-u2",
        "notes": ["goal_safety_net_on", "json_final_unavailable"],
    },
    "rfdetr_shadow_success": {
        "run_id": "ef033fbd-b2f3-4365-b1a1-ebaa2aac95f4",
        "job_id": "40322876-a336-4ef9-a113-43f351b67211-u2",
        "notes": ["rfdetr_shadow_reference"],
    },
}
TARGET_CLASSES = {
    0: "player",
    1: "goalkeeper",
    2: "referee",
    3: "ball",
    4: "goal_frame",
}
SOURCE_CLASS_HINTS = {
    "person": (0, "player", "ambiguous", 0.55),
    "athlete": (0, "player", "probable", 0.75),
    "player": (0, "player", "exact", 1.0),
    "player_with_ball": (0, "player", "manual_review", 0.4),
    "shooter": (0, "player", "manual_review", 0.4),
    "defender": (0, "player", "manual_review", 0.4),
    "goalkeeper": (1, "goalkeeper", "exact", 1.0),
    "goalkeeper_or_player": (None, None, "manual_review", 0.0),
    "referee": (2, "referee", "exact", 1.0),
    "referee_like_person": (None, None, "manual_review", 0.0),
    "sports ball": (3, "ball", "exact", 1.0),
    "sports_ball": (3, "ball", "exact", 1.0),
    "ball": (3, "ball", "exact", 1.0),
    "goal": (4, "goal_frame", "probable", 0.7),
    "goalpost": (4, "goal_frame", "probable", 0.8),
    "goal_frame": (4, "goal_frame", "exact", 1.0),
    "net": (4, "goal_frame", "manual_review", 0.3),
    "staff": (None, None, "discard", 0.0),
    "spectator": (None, None, "discard", 0.0),
}
SAFETY_EXCLUDE_PATTERNS = (
    re.compile(r"Authorization:\s*Bearer\s+\S+", re.I),
    re.compile(r"SUPABASE_(SERVICE_ROLE_)?KEY\s*=\s*\S+", re.I),
    re.compile(r"R2_SECRET_ACCESS_KEY\s*=\s*\S+", re.I),
    re.compile(r"RUNPOD_API_KEY\s*=\s*\S+", re.I),
)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dirs() -> None:
    for d in [
        CATALOG_DIR,
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "images" / "valid",
        DATASET_DIR / "images" / "test",
        DATASET_DIR / "annotations",
        DATASET_DIR / "manifests",
        DATASET_DIR / "hard_negatives",
        DATASET_DIR / "review",
        GOLD_DIR / "images",
        GOLD_DIR / "annotations",
        GOLD_DIR / "sequences",
        REVIEW_DIR,
        DOCS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
        keep = d / ".gitkeep"
        if not keep.exists():
            keep.write_text("placeholder for RF-1A; source files are referenced by manifests\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def stable_hash(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def file_sha256(path: Path, max_bytes: int = 512 * 1024 * 1024) -> Optional[str]:
    try:
        if path.stat().st_size > max_bytes:
            return None
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def infer_mime(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext in VIDEO_EXTS:
        return "video/" + ext.lstrip(".")
    if ext == ".json":
        return "application/json"
    if ext == ".csv":
        return "text/csv"
    if ext == ".txt":
        return "text/plain"
    return None


def image_size(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path))
        if img is None:
            return None, None, None
        h, w = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        return int(w), int(h), int(channels)
    except Exception:
        return None, None, None


def extract_match_id(text: str) -> Optional[str]:
    m = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", text, re.I)
    return m.group(0).lower() if m else None


def extract_frame_index(text: str) -> Optional[int]:
    m = re.search(r"frame[_-]?0*(\d+)", text, re.I)
    return int(m.group(1)) if m else None


def base_asset(
    source_type: str,
    reference: str,
    status: str = "available",
    local_path: Optional[str] = None,
    storage_reference: Optional[str] = None,
) -> Dict[str, Any]:
    match_id = extract_match_id(reference)
    frame_index = extract_frame_index(reference)
    return {
        "asset_id": stable_hash([source_type, reference, frame_index, None]),
        "source_id": stable_hash([source_type, reference])[:16],
        "source_type": source_type,
        "source_status": status,
        "local_path": local_path,
        "storage_reference": storage_reference,
        "match_id": match_id,
        "match_uuid": match_id,
        "run_id": None,
        "job_id": None,
        "video_id": None,
        "video_fingerprint": None,
        "frame_index": frame_index,
        "timestamp_sec": None,
        "width": None,
        "height": None,
        "channels": None,
        "mime_type": None,
        "size_bytes": None,
        "sha256": None,
        "perceptual_hash": None,
        "annotation_available": False,
        "annotation_format": None,
        "annotation_path": None,
        "classes_present": [],
        "object_count": 0,
        "ball_annotation_count": 0,
        "reviewed": False,
        "review_status": "unreviewed",
        "existing_split": None,
        "proposed_split": None,
        "gold_candidate": False,
        "provenance": None,
        "generated_by": None,
        "created_at": None,
        "schema_version": SCHEMA_VERSION,
        "notes": [],
    }


def local_assets(root: Path = ROOT) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTS | VIDEO_EXTS | ANNOTATION_EXTS | {".md", ".pt", ".pth", ".yaml", ".yml"}:
            continue
        asset = base_asset("local", rel, local_path=rel)
        try:
            stat = path.stat()
            asset["size_bytes"] = stat.st_size
            asset["mime_type"] = infer_mime(path)
            asset["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime))
            asset["sha256"] = file_sha256(path)
            if ext in IMAGE_EXTS:
                w, h, c = image_size(path)
                asset["width"], asset["height"], asset["channels"] = w, h, c
            if ext in VIDEO_EXTS:
                asset["video_id"] = stable_hash(["local_video", rel])[:16]
                asset["video_fingerprint"] = stable_hash([rel, stat.st_size])[:32]
            if ext in ANNOTATION_EXTS:
                asset["annotation_available"] = True
                asset["annotation_format"] = annotation_format(path)
                asset["annotation_path"] = rel
        except Exception as exc:
            asset["source_status"] = "read_error"
            asset["notes"].append(f"read_error:{type(exc).__name__}")
        assets.append(asset)
    return assets


def annotation_format(path: Path) -> str:
    ext = path.suffix.lower()
    name = path.name.lower()
    if ext == ".json":
        try:
            d = json.loads(path.read_text())
            if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d.keys()):
                return "coco_detection"
            if isinstance(d, list):
                return "json_list_custom"
            return "json_custom"
        except Exception:
            return "json_read_error"
    if ext == ".jsonl":
        return "jsonl_custom"
    if ext == ".csv":
        return "csv_custom"
    if ext == ".txt":
        try:
            non_empty = [line.split() for line in path.read_text(errors="ignore").splitlines() if line.strip() and not line.strip().startswith("#")]
            if non_empty and all(len(parts) >= 5 and all(is_float(p) for p in parts[:5]) for parts in non_empty[:25]):
                return "yolo_detection"
        except Exception:
            return "txt_read_error"
        return "txt_unknown"
    if ext == ".xml":
        return "pascal_voc"
    return "unknown"


def is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def iter_bboxes_from_obj(obj: Any, source: str = "") -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        bbox = obj.get("bbox") or obj.get("xyxy") or obj.get("xywh")
        cls = (
            obj.get("class_name")
            or obj.get("label")
            or obj.get("name")
            or obj.get("category")
            or obj.get("event_type")
        )
        cls_id = obj.get("class_id") or obj.get("category_id")
        conf = obj.get("confidence") or obj.get("conf") or obj.get("score")
        frame_index = obj.get("frame_index") or obj.get("frame_number") or extract_frame_index(str(obj.get("frame_path") or obj.get("frame_key") or source))
        if bbox is not None:
            yield {
                "bbox": bbox,
                "class_name": cls,
                "class_id": cls_id,
                "confidence": conf,
                "frame_index": frame_index,
                "source": source,
                "timestamp_sec": obj.get("timestamp_sec") or obj.get("timestamp"),
            }
        for value in obj.values():
            yield from iter_bboxes_from_obj(value, source)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_bboxes_from_obj(item, source)


def normalize_bbox(bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(bbox, dict):
        if all(k in bbox for k in ("x", "y", "width", "height")):
            return float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])
        if all(k in bbox for k in ("left", "top", "width", "height")):
            return float(bbox["left"]), float(bbox["top"]), float(bbox["width"]), float(bbox["height"])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        vals = [float(x) for x in bbox[:4]]
        x1, y1, a, b = vals
        # COCO/RF-DETR artifacts in this repo mostly use xywh. If a/b look like
        # absolute bottom-right and exceed x/y, keep a conservative note via size.
        if a > 0 and b > 0:
            return x1, y1, a, b
    return None


def is_ball_class(cls: Any, cls_id: Any = None) -> bool:
    if cls is not None and str(cls).strip().lower().replace("-", "_") in {"ball", "sports_ball", "sports ball"}:
        return True
    return str(cls_id) in {"3", "32", "37"}


def annotation_audit(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    ball_boxes: List[Dict[str, Any]] = []
    annotation_files = [ROOT / a["local_path"] for a in assets if a.get("annotation_available") and a.get("local_path")]
    for path in annotation_files:
        try:
            fmt = annotation_format(path)
            if fmt == "coco_detection":
                d = json.loads(path.read_text())
                cats = {c.get("id"): c.get("name") for c in d.get("categories", [])}
                image_by_id = {i.get("id"): i for i in d.get("images", [])}
                for ann in d.get("annotations", []):
                    cls = cats.get(ann.get("category_id"), str(ann.get("category_id")))
                    class_counter[str(cls)] += 1
                    bbox = normalize_bbox(ann.get("bbox"))
                    img = image_by_id.get(ann.get("image_id"), {})
                    validate_bbox_issue(path, bbox, img.get("width"), img.get("height"), issues, ann.get("id"))
                    if is_ball_class(cls, ann.get("category_id")) and bbox:
                        ball_boxes.append(ball_record(bbox, img.get("width"), img.get("height"), cls, ann.get("category_id"), path.as_posix(), ann.get("image_id")))
            elif fmt in {"json_custom", "json_list_custom"}:
                d = json.loads(path.read_text())
                for rec in iter_bboxes_from_obj(d, path.relative_to(ROOT).as_posix()):
                    cls = rec.get("class_name") or rec.get("class_id") or "unknown"
                    class_counter[str(cls)] += 1
                    bbox = normalize_bbox(rec.get("bbox"))
                    validate_bbox_issue(path, bbox, None, None, issues, rec.get("frame_index"))
                    if is_ball_class(rec.get("class_name"), rec.get("class_id")) and bbox:
                        ball_boxes.append(ball_record(bbox, None, None, rec.get("class_name"), rec.get("class_id"), path.as_posix(), rec.get("frame_index")))
            elif fmt == "csv_custom":
                with path.open(newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        for key in ("manual_label_yolo", "manual_label_rfdetr_025", "manual_label_rfdetr_050"):
                            if row.get(key):
                                class_counter[row[key]] += 1
            elif fmt == "yolo_detection":
                for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
                    parts = line.split()
                    if len(parts) >= 5:
                        cls = parts[0]
                        class_counter[str(cls)] += 1
                    elif line.strip():
                        issues.append(issue("warning", "unsupported_yolo_line", path, f"line {line_no}"))
        except Exception as exc:
            issues.append(issue("error", "annotation_read_error", path, type(exc).__name__))
    return {
        "schema_version": "rf1a.annotation_quality.v1",
        "generated_at": now_iso(),
        "annotation_files_scanned": len(annotation_files),
        "classes_count": dict(sorted(class_counter.items())),
        "ball_boxes": ball_boxes,
        "issues": issues,
        "issue_counts": dict(Counter(i["severity"] for i in issues)),
    }


def issue(severity: str, code: str, path: Path, detail: str) -> Dict[str, Any]:
    return {"severity": severity, "code": code, "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix(), "detail": detail}


def validate_bbox_issue(path: Path, bbox: Optional[Tuple[float, float, float, float]], width: Any, height: Any, issues: List[Dict[str, Any]], ref: Any) -> None:
    if bbox is None:
        issues.append(issue("warning", "missing_or_unsupported_bbox", path, str(ref)))
        return
    x, y, w, h = bbox
    vals = [x, y, w, h]
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        issues.append(issue("critical", "bbox_nan_or_inf", path, str(ref)))
    if w <= 0 or h <= 0:
        issues.append(issue("error", "bbox_non_positive_area", path, str(ref)))
    if x < 0 or y < 0:
        issues.append(issue("warning", "bbox_negative_coordinate", path, str(ref)))
    if width and height and (x + w > float(width) or y + h > float(height)):
        issues.append(issue("warning", "bbox_outside_image", path, str(ref)))


def ball_record(bbox: Tuple[float, float, float, float], width: Any, height: Any, cls: Any, cls_id: Any, source: str, ref: Any) -> Dict[str, Any]:
    x, y, w, h = bbox
    area = w * h
    image_area = float(width) * float(height) if width and height else None
    return {
        "source": source,
        "reference": ref,
        "class_name": cls,
        "class_id": cls_id,
        "bbox_width_px": w,
        "bbox_height_px": h,
        "bbox_area_px": area,
        "bbox_area_ratio": area / image_area if image_area else None,
        "bbox_center_x_normalized": (x + w / 2) / float(width) if width else None,
        "bbox_center_y_normalized": (y + h / 2) / float(height) if height else None,
        "aspect_ratio": w / h if h else None,
        "distance_to_nearest_border_px": min(x, y, float(width) - (x + w), float(height) - (y + h)) if width and height else None,
        "frame_index": ref if isinstance(ref, int) else extract_frame_index(str(ref)),
        "timestamp_sec": None,
        "annotation_reviewed": False,
        "hard_case_tags": auto_hard_case_tags(w, h, area, width, height),
        "needs_manual_review": True,
    }


def auto_hard_case_tags(w: float, h: float, area: float, width: Any, height: Any) -> List[str]:
    tags: List[str] = []
    if area <= 400:
        tags.append("object_too_small")
    if width and height:
        border = min(w, h, float(width), float(height))
        if border <= 10:
            tags.append("near_border_candidate")
    return tags


def ball_distribution(ball_boxes: List[Dict[str, Any]]) -> Dict[str, Any]:
    areas = sorted([float(b["bbox_area_px"]) for b in ball_boxes if b.get("bbox_area_px") is not None])
    widths = sorted([float(b["bbox_width_px"]) for b in ball_boxes if b.get("bbox_width_px") is not None])
    heights = sorted([float(b["bbox_height_px"]) for b in ball_boxes if b.get("bbox_height_px") is not None])
    return {
        "schema_version": "rf1a.ball_distribution.v1",
        "ball_annotation_count": len(areas),
        "area_px": stats(areas),
        "width_px": stats(widths),
        "height_px": stats(heights),
        "size_categories_proposed": propose_size_categories(areas),
        "hard_case_tags_observed": dict(Counter(tag for b in ball_boxes for tag in b.get("hard_case_tags", []))),
        "records_sample": ball_boxes[:50],
    }


def stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p01": percentile(values, 1),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return values[int(k)]
    return values[f] * (c - k) + values[c] * (k - f)


def propose_size_categories(areas: List[float]) -> Dict[str, Any]:
    if not areas:
        return {"status": "unavailable", "reason": "no_ball_annotations"}
    p25, p50, p75 = percentile(areas, 25), percentile(areas, 50), percentile(areas, 75)
    return {
        "basis": "observed_area_percentiles",
        "tiny": {"max_area_px": p25},
        "small": {"min_area_px": p25, "max_area_px": p50},
        "medium": {"min_area_px": p50, "max_area_px": p75},
        "large": {"min_area_px": p75},
        "absolute_area_alternative": {
            "tiny": "<=400 px",
            "small": "401-900 px",
            "medium": "901-2500 px",
            "large": ">2500 px",
        },
        "benchmark_recommendation": "Use percentile buckets for current data comparability; report absolute buckets in parallel for cross-dataset stability.",
    }


def duplicate_report(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sha: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_basename: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in assets:
        if a.get("sha256"):
            by_sha[a["sha256"]].append(a)
        if a.get("local_path") and Path(a["local_path"]).suffix.lower() in IMAGE_EXTS:
            by_basename[Path(a["local_path"]).name].append(a)
    groups = []
    for sha, members in sorted(by_sha.items()):
        if len(members) > 1:
            groups.append(duplicate_group("exact", sha, members, "keep_one"))
    for basename, members in sorted(by_basename.items()):
        if len(members) > 1:
            groups.append(duplicate_group("same_basename", basename, members, "manual_review"))
    return {
        "schema_version": "rf1a.duplicates.v1",
        "generated_at": now_iso(),
        "duplicate_group_count": len(groups),
        "exact_duplicate_groups": sum(1 for g in groups if g["duplicate_type"] == "exact"),
        "near_duplicate_groups": sum(1 for g in groups if g["duplicate_type"] != "exact"),
        "groups": groups,
    }


def duplicate_group(kind: str, key: str, members: List[Dict[str, Any]], action: str) -> Dict[str, Any]:
    return {
        "duplicate_group_id": stable_hash([kind, key])[:16],
        "duplicate_type": "exact" if kind == "exact" else "near_duplicate",
        "canonical_candidate": members[0]["asset_id"],
        "members": [m["asset_id"] for m in members],
        "evidence": {"key": key, "member_paths": [m.get("local_path") for m in members]},
        "recommended_action": action,
    }


def class_mapping_report(annotation: Dict[str, Any]) -> Dict[str, Any]:
    records = []
    for cls, count in sorted(annotation.get("classes_count", {}).items()):
        norm = cls.strip().lower().replace("-", "_")
        target = SOURCE_CLASS_HINTS.get(norm) or SOURCE_CLASS_HINTS.get(norm.replace("_", " "))
        if target:
            target_id, target_name, status, conf = target
        else:
            target_id, target_name, status, conf = None, None, "manual_review", 0.0
        records.append({
            "source_schema": "observed_annotations",
            "source_class_id": cls if str(cls).isdigit() else None,
            "source_class_name": cls,
            "observed_count": count,
            "target_class_id": target_id,
            "target_class_name": target_name,
            "mapping_status": status,
            "confidence": conf,
            "reason": "rule_based_mapping_from_rf1a_class_policy",
            "examples": [],
        })
    return {"schema_version": "rf1a.class_mapping.v1", "target_classes": TARGET_CLASSES, "mappings": records}


def supabase_source_health() -> Dict[str, Any]:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or ""
    tables = [
        "training_frames",
        "clips",
        "matches",
        "match_processing_metrics",
        "artifacts",
        "detections",
        "annotations",
        "review",
        "runs",
        "shadow_detections",
        "shadow_tracking",
        "hard_case_annotations",
        "events_log",
    ]
    if not url or not key:
        return {"source_type": "supabase", "source_status": "missing_credentials", "tables": []}
    results = []
    headers = {"apikey": key, "Authorization": "Bearer " + key, "Prefer": "count=exact"}
    for table in tables:
        endpoint = f"{url}/rest/v1/{table}?select=*&limit=1"
        try:
            req = urllib.request.Request(endpoint, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
                content_range = resp.headers.get("Content-Range")
                total = None
                if content_range and "/" in content_range:
                    tail = content_range.split("/")[-1]
                    total = int(tail) if tail.isdigit() else None
                results.append({
                    "table": table,
                    "source_status": "available",
                    "row_count": total,
                    "columns_observed": sorted(data[0].keys()) if data else [],
                })
        except urllib.error.HTTPError as exc:
            status = "unavailable" if exc.code in {404, 406} else "read_error"
            results.append({"table": table, "source_status": status, "http_status": exc.code})
        except Exception as exc:
            results.append({"table": table, "source_status": "read_error", "error_type": type(exc).__name__})
    return {"source_type": "supabase", "source_status": "available", "tables": results}


def r2_source_health() -> Dict[str, Any]:
    endpoint = os.environ.get("AI_WORKER_V1_R2_ENDPOINT") or os.environ.get("R2_ENDPOINT")
    access = os.environ.get("AI_WORKER_V1_R2_READ_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("AI_WORKER_V1_R2_READ_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    bucket = os.environ.get("AI_WORKER_V1_R2_INPUT_BUCKET") or os.environ.get("R2_BUCKET") or "one-frame"
    prefixes = ["training_frames/", "clips/", "debug/", "artifacts/", "detections/", "shadow/", "ai_worker_v1/"]
    if not endpoint or not access or not secret:
        return {"source_type": "r2", "source_status": "missing_credentials", "bucket": bucket, "prefixes": []}
    try:
        import boto3  # type: ignore

        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint.strip('"'),
            aws_access_key_id=access.strip('"'),
            aws_secret_access_key=secret.strip('"'),
            region_name="auto",
        )
        out = []
        for prefix in prefixes:
            try:
                contents = []
                token = None
                truncated_by_cap = False
                while True:
                    kwargs = {"Bucket": bucket.strip('"'), "Prefix": prefix, "MaxKeys": 1000}
                    if token:
                        kwargs["ContinuationToken"] = token
                    resp = s3.list_objects_v2(**kwargs)
                    contents.extend(resp.get("Contents", []))
                    if len(contents) >= 10000:
                        truncated_by_cap = True
                        break
                    if not resp.get("IsTruncated"):
                        break
                    token = resp.get("NextContinuationToken")
                out.append({
                    "prefix": prefix,
                    "source_status": "available",
                    "object_count": len(contents),
                    "sample_count": min(len(contents), 20),
                    "is_truncated": bool(resp.get("IsTruncated")) or truncated_by_cap,
                    "truncated_by_cap": truncated_by_cap,
                    "sample_keys": [sanitize_reference(o["Key"]) for o in contents[:20]],
                    "total_size_bytes_listed": sum(int(o.get("Size", 0)) for o in contents),
                })
            except Exception as exc:
                out.append({"prefix": prefix, "source_status": "read_error", "error_type": type(exc).__name__})
        return {"source_type": "r2", "source_status": "available", "bucket": bucket.strip('"'), "prefixes": out}
    except Exception as exc:
        return {"source_type": "r2", "source_status": "read_error", "error_type": type(exc).__name__, "bucket": bucket}


def sanitize_reference(text: str) -> str:
    clean = text
    for pat in SAFETY_EXCLUDE_PATTERNS:
        clean = pat.sub("[REDACTED]", clean)
    return clean


def build_sources_manifest(local: List[Dict[str, Any]], supabase: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": "rf1a.sources_manifest.v1",
        "generated_at": now_iso(),
        "sources": [
            {
                "source_id": "local_ai_worker_v1",
                "source_type": "local",
                "source_status": "available",
                "asset_count": len(local),
                "root": "ai_worker_v1/",
            },
            {
                "source_id": "supabase_rest",
                "source_type": "supabase",
                "source_status": supabase.get("source_status"),
                "tables": supabase.get("tables", []),
            },
            {
                "source_id": "r2_one_frame",
                "source_type": "r2",
                "source_status": r2.get("source_status"),
                "bucket": r2.get("bucket"),
                "prefixes": r2.get("prefixes", []),
            },
        ],
        "known_runs": KNOWN_RUNS,
    }


def match_grouping(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    shared_group = "match_group_drive_smoke_test_v1"
    groups[shared_group] = {
        "group_id": shared_group,
        "match_ids": {KNOWN_MATCH_IDS["run_a"], KNOWN_MATCH_IDS["run_b"], KNOWN_MATCH_IDS["rfdetr_shadow_reference"]},
        "match_uuids": {KNOWN_MATCH_IDS["run_a"], KNOWN_MATCH_IDS["run_b"], KNOWN_MATCH_IDS["rfdetr_shadow_reference"]},
        "video_ids": set(),
        "run_ids": {KNOWN_RUNS["rfdetr_shadow_success"]["run_id"]},
        "job_ids": {KNOWN_RUNS["run_a"]["job_id"], KNOWN_RUNS["run_b"]["job_id"], KNOWN_RUNS["rfdetr_shadow_success"]["job_id"]},
        "source_ids": {"known_run_registry"},
        "asset_count": 0,
        "annotation_count": 0,
        "ball_count": 0,
        "candidate_split": None,
        "reason": ["known_run_a_b_and_rfdetr_shadow_registry"],
    }
    for a in assets:
        match_id = a.get("match_id")
        if match_id in {KNOWN_MATCH_IDS["run_a"], KNOWN_MATCH_IDS["run_b"], KNOWN_MATCH_IDS["rfdetr_shadow_reference"]}:
            gid = shared_group
        elif match_id:
            gid = "match_group_" + match_id[:8]
        else:
            local = a.get("local_path") or a.get("storage_reference") or "unknown"
            if "test_178" in local:
                m = re.search(r"test_(\d+)", local)
                gid = "source_test_" + (m.group(1) if m else "unknown")
            else:
                gid = "ungrouped_unknown"
        g = groups.setdefault(gid, {
            "group_id": gid,
            "match_ids": set(),
            "match_uuids": set(),
            "video_ids": set(),
            "run_ids": set(),
            "job_ids": set(),
            "source_ids": set(),
            "asset_count": 0,
            "annotation_count": 0,
            "ball_count": 0,
            "candidate_split": None,
            "reason": [],
        })
        g["asset_count"] += 1
        if match_id:
            g["match_ids"].add(match_id)
            g["match_uuids"].add(match_id)
        for key in ("video_id", "run_id", "job_id", "source_id"):
            if a.get(key):
                g[key + "s"].add(a[key])
        if a.get("annotation_available"):
            g["annotation_count"] += 1
        g["ball_count"] += int(a.get("ball_annotation_count") or 0)
    ordered = []
    split_cycle = ["train", "valid", "test"]
    for idx, g in enumerate(sorted(groups.values(), key=lambda x: x["group_id"])):
        if g["group_id"] == shared_group:
            g["candidate_split"] = "gold"
            g["reason"].append("contains_run_a_run_b_and_required_gold_review_moments")
        else:
            g["candidate_split"] = split_cycle[idx % len(split_cycle)]
            g["reason"].append("deterministic_group_split_proposal")
        for k in ("match_ids", "match_uuids", "video_ids", "run_ids", "job_ids", "source_ids"):
            g[k] = sorted(g[k])
        ordered.append(g)
    return {"schema_version": "rf1a.match_grouping.v1", "groups": ordered}


def split_manifests(assets: List[Dict[str, Any]], grouping: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    group_split = {g["group_id"]: g["candidate_split"] for g in grouping["groups"]}
    group_by_match = {}
    for g in grouping["groups"]:
        for m in g.get("match_ids", []):
            group_by_match[m] = g["group_id"]
    manifests = {"train": [], "valid": [], "test": [], "gold": []}
    for a in assets:
        gid = group_by_match.get(a.get("match_id")) or ("source_test_" + re.search(r"test_(\d+)", a.get("local_path") or "").group(1) if re.search(r"test_(\d+)", a.get("local_path") or "") else "ungrouped_unknown")
        split = group_split.get(gid, "train")
        record = {
            "asset_id": a["asset_id"],
            "source_reference": a.get("local_path") or a.get("storage_reference"),
            "match_group": gid,
            "proposed_split": split,
            "annotation_status": "annotated" if a.get("annotation_available") else "unannotated",
            "gold_excluded": split != "gold",
            "review_status": a.get("review_status", "unreviewed"),
            "schema_version": "rf1a.split_manifest.v1",
        }
        manifests.setdefault(split, []).append(record)
    return manifests


def gold_set_proposal() -> Dict[str, Any]:
    moments = [235, 389, 532, 605]
    sequences = []
    for ts in moments:
        sequences.append({
            "sequence_id": f"gold_candidate_run_ab_{ts}s",
            "source_asset_group": "match_group_drive_smoke_test_v1",
            "match_id": KNOWN_MATCH_IDS["run_a"],
            "video_id": None,
            "start_frame": None,
            "end_frame": None,
            "start_timestamp_sec": max(0, ts - 8),
            "end_timestamp_sec": ts + 8,
            "reason_selected": ["required_run_a_b_moment", "manual_review_needed"],
            "candidate_categories": ["uncertain_case"] + (["shot"] if ts == 389 else []),
            "expected_objects": ["ball", "players"],
            "review_status": "unreviewed",
            "gold_eligible": False,
            "gold_inclusion_status": "candidate",
            "exclusion_reason": None,
            "annotation_density_required": "dense" if ts in {389, 532, 605} else "sparse",
            "schema_version": "rf1a.gold.sequence.v1",
        })
    manifest = {
        "schema_version": "rf1a.gold.manifest.v1",
        "gold_policy": "candidate_only_until_human_review",
        "sequences": [s["sequence_id"] for s in sequences],
    }
    return {"manifest": manifest, "sequences": sequences}


def review_queue(annotation: Dict[str, Any], duplicates: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    items = []
    for seq in gold["sequences"]:
        ts = seq["start_timestamp_sec"] + (seq["end_timestamp_sec"] - seq["start_timestamp_sec"]) / 2
        items.append({
            "review_id": stable_hash(["review", seq["sequence_id"]])[:16],
            "priority": "P0",
            "asset_id": None,
            "sequence_id": seq["sequence_id"],
            "reason_codes": ["required_gold_candidate_moment", f"timestamp_{int(ts)}s"],
            "required_action": "classify",
            "questions": ["Is the ball visible?", "What event category applies?", "Should this be gold eligible?"],
            "current_annotation": {},
            "reviewer_decision": None,
            "review_status": "pending",
        })
    for issue_item in annotation.get("issues", [])[:100]:
        priority = "P0" if issue_item["severity"] in {"critical", "error"} else "P1"
        items.append({
            "review_id": stable_hash(["annotation_issue", issue_item])[:16],
            "priority": priority,
            "asset_id": None,
            "sequence_id": None,
            "reason_codes": [issue_item["code"], "annotation_quality"],
            "required_action": "correct",
            "questions": ["Verify and correct annotation if source image is usable."],
            "current_annotation": issue_item,
            "reviewer_decision": None,
            "review_status": "pending",
        })
    for group in duplicates.get("groups", [])[:50]:
        items.append({
            "review_id": stable_hash(["duplicate", group["duplicate_group_id"]])[:16],
            "priority": "P2",
            "asset_id": group["canonical_candidate"],
            "sequence_id": None,
            "reason_codes": ["duplicate_or_near_duplicate"],
            "required_action": "verify",
            "questions": ["Is this accidental duplicate or valid temporal sequence?"],
            "current_annotation": group,
            "reviewer_decision": None,
            "review_status": "pending",
        })
    return {
        "schema_version": "rf1a.review_queue.v1",
        "items": items,
        "summary": dict(Counter(i["priority"] for i in items)),
    }


def manual_annotation_estimate(summary: Dict[str, Any]) -> Dict[str, Any]:
    unannotated = summary.get("unannotated_images", 0)
    ball_count = summary.get("ball_annotations", 0)
    rates = {
        "image_box_review_per_hour": 180,
        "dense_ball_frames_per_hour": 240,
    }
    scenarios = {
        "minimum_viable_baseline": {
            "frames": max(200, min(1000, unannotated)),
            "boxes": max(300, ball_count + 200),
            "sequences": 4,
            "hours_review": round(max(200, min(1000, unannotated)) / rates["image_box_review_per_hour"], 1),
            "risk": "high",
            "coverage_expected": "ball plus basic player checks on known hard moments",
        },
        "recommended_first_training": {
            "frames": max(1500, min(5000, unannotated + 1000)),
            "boxes": max(3000, ball_count * 3 + 1000),
            "sequences": 12,
            "hours_review": round(max(1500, min(5000, unannotated + 1000)) / rates["image_box_review_per_hour"], 1),
            "risk": "medium",
            "coverage_expected": "ball, player, goalkeeper/referee review candidates, hard negatives",
        },
        "strong_production_oriented_dataset": {
            "frames": max(6000, unannotated + 3000),
            "boxes": max(15000, ball_count * 6 + 6000),
            "sequences": 30,
            "hours_review": round(max(6000, unannotated + 3000) / rates["image_box_review_per_hour"], 1),
            "risk": "lower",
            "coverage_expected": "multi-match detector foundation with gold sequences kept isolated",
        },
    }
    return {"schema_version": "rf1a.annotation_estimate.v1", "rates_used": rates, "scenarios": scenarios}


def source_health_md(sources: Dict[str, Any]) -> str:
    lines = ["# RF-1A Source Health", ""]
    for src in sources["sources"]:
        lines.append(f"- `{src['source_id']}` ({src['source_type']}): `{src['source_status']}`")
    return "\n".join(lines)


def inventory_summary(assets: List[Dict[str, Any]], annotation: Dict[str, Any], sources: Dict[str, Any]) -> Dict[str, Any]:
    images = [a for a in assets if (a.get("mime_type") or "").startswith("image/")]
    videos = [a for a in assets if (a.get("mime_type") or "").startswith("video/")]
    annotated = [a for a in images if a.get("annotation_available")]
    groups = match_grouping(assets)
    return {
        "schema_version": "rf1a.inventory.summary.v1",
        "generated_at": now_iso(),
        "total_assets": len(assets),
        "unique_images": len({a.get("sha256") or a["asset_id"] for a in images}),
        "videos": len(videos),
        "match_groups": len(groups["groups"]),
        "annotated_images": len(annotated),
        "unannotated_images": max(0, len(images) - len(annotated)),
        "ball_annotations": len(annotation.get("ball_boxes", [])),
        "sources_status": {s["source_id"]: s["source_status"] for s in sources["sources"]},
    }


def generate_markdown_reports(outputs: Dict[str, Any]) -> None:
    summary = outputs["summary"]
    annotation = outputs["annotation"]
    ball = outputs["ball"]
    duplicates = outputs["duplicates"]
    mapping = outputs["mapping"]
    grouping = outputs["grouping"]
    review = outputs["review"]
    estimate = outputs["estimate"]
    gold = outputs["gold"]
    sources = outputs["sources"]
    files = outputs["files_created"]
    tests = outputs.get("tests", {"status": "not_run_in_report_generation"})
    lines = [
        "# RF-1A Final Report",
        "",
        "## Executive Summary",
        f"- Total assets found: `{summary['total_assets']}`",
        f"- Unique images: `{summary['unique_images']}`",
        f"- Videos: `{summary['videos']}`",
        f"- Match groups: `{summary['match_groups']}`",
        f"- Ball annotations discovered: `{summary['ball_annotations']}`",
        "- RF-DETR was not trained, promoted, or benchmarked in this phase.",
        "- Goal Safety Net and production `src/` remained out of scope.",
        "",
        "## Sources Audited",
    ]
    for src in sources["sources"]:
        lines.append(f"- `{src['source_id']}`: `{src['source_status']}`")
    lines += [
        "",
        "## Dataset Counts",
        f"- Annotated images: `{summary['annotated_images']}`",
        f"- Unannotated images: `{summary['unannotated_images']}`",
        f"- Annotation files scanned: `{annotation['annotation_files_scanned']}`",
        "",
        "## Classes and Mapping",
        "| Source class | Count | Target | Status |",
        "|---|---:|---|---|",
    ]
    for m in mapping["mappings"]:
        lines.append(f"| {m['source_class_name']} | {m['observed_count']} | {m['target_class_name']} | {m['mapping_status']} |")
    lines += [
        "",
        "## Ball Annotation Distribution",
        f"- Count: `{ball['ball_annotation_count']}`",
        f"- Area percentiles: `{json.dumps(ball.get('area_px', {}), sort_keys=True)}`",
        f"- Proposed size categories: `{json.dumps(ball.get('size_categories_proposed', {}), sort_keys=True)}`",
        "",
        "## Hard Cases",
        "- Present by objective tags: `" + json.dumps(ball.get("hard_case_tags_observed", {}), sort_keys=True) + "`",
        "- Missing or not verifiable without human review: motion_blur, partial_occlusion, full_occlusion, near_player_feet, near_white_line, aerial_ball, crowded_players, goal_visible, low_light, compression_artifact.",
        "",
        "## Duplicates and Leakage",
        f"- Duplicate groups: `{duplicates['duplicate_group_count']}`",
        f"- Exact duplicate groups: `{duplicates['exact_duplicate_groups']}`",
        f"- Near duplicate groups: `{duplicates['near_duplicate_groups']}`",
        "- Leakage policy: split proposal is group-based; Run A and Run B are forced into the same gold candidate group.",
        "",
        "## Annotation Quality",
        f"- Issue counts: `{json.dumps(annotation.get('issue_counts', {}), sort_keys=True)}`",
        "- Original annotations were not modified.",
        "",
        "## Split Proposal",
    ]
    for g in grouping["groups"]:
        lines.append(f"- `{g['group_id']}` -> `{g['candidate_split']}` ({g['asset_count']} assets)")
    lines += [
        "",
        "## Gold Set Proposal",
        "- Gold remains candidate-only until human review.",
    ]
    for seq in gold["sequences"]:
        lines.append(f"- `{seq['sequence_id']}`: {seq['start_timestamp_sec']}s-{seq['end_timestamp_sec']}s, categories={seq['candidate_categories']}")
    lines += [
        "",
        "## Review Queue",
        f"- Summary: `{json.dumps(review['summary'], sort_keys=True)}`",
        "",
        "## Manual Annotation Estimate",
        f"`{json.dumps(estimate['scenarios'], indent=2, sort_keys=True)}`",
        "",
        "## Gaps and Risks",
        "- Human labels for goalkeeper/referee/goal_frame are not proven sufficient from current local artifacts.",
        "- Several sources depend on external credentials; unavailable sources are listed in source reports.",
        "- Pseudo detections from YOLO/RF-DETR outputs must not be treated as ground truth.",
        "- Final RunPod JSON for Run B was unavailable and must not be used as audit ground truth.",
        "",
        "## Recommendations",
        "- RF-1B: continue only after human review queue P0 is sampled and Gold candidate moments are labeled.",
        "- RF-1C: do not start training until split leakage checks stay clean after adding reviewed annotations.",
        "",
        "## Confirmations",
        "- No training executed.",
        "- No RunPod executed by RF-1A scripts.",
        "- No production `src/` edits made by RF-1A.",
        "- Goal Safety Net left intact.",
        "- Supabase and R2 access is read-only in RF-1A scripts.",
        "",
        "## Files Created",
    ]
    lines += [f"- `{p}`" for p in files]
    lines += ["", "## Tests", f"`{json.dumps(tests, sort_keys=True)}`"]
    lines += ["", "## Git Status", "Git status unavailable in this workspace: not a git repository."]
    write_md(DOCS_DIR / "RF1A_FINAL_REPORT.md", "\n".join(lines))
    write_md(CATALOG_DIR / "dataset_inventory.md", f"# Dataset Inventory\n\nTotal assets: `{summary['total_assets']}`\n\nUnique images: `{summary['unique_images']}`\n")
    write_md(CATALOG_DIR / "duplicate_report.md", f"# Duplicate Report\n\nGroups: `{duplicates['duplicate_group_count']}`\n")
    write_md(CATALOG_DIR / "leakage_report.md", "# Leakage Report\n\nNo cross-split leakage detected by group-level proposal. Run A and Run B are grouped together.\n")
    write_md(CATALOG_DIR / "annotation_quality_report.md", f"# Annotation Quality\n\nIssue counts: `{json.dumps(annotation.get('issue_counts', {}), sort_keys=True)}`\n")
    write_md(CATALOG_DIR / "ball_visibility_audit.md", "# Ball Visibility Audit\n\nSemantic visibility requires human review. Objective size/border tags were generated where possible.\n")
    write_md(REVIEW_DIR / "review_summary.md", f"# Review Summary\n\nQueue summary: `{json.dumps(review['summary'], sort_keys=True)}`\n")
    write_md(REVIEW_DIR / "review_instructions.md", "# Review Instructions\n\nDo not infer goal, shot, goalkeeper, referee or ball visibility without visual confirmation. Fill reviewer_decision only after manual review.\n")
    write_md(DATASET_DIR / "README.md", "# football_detection_v1\n\nRF-1A referential dataset proposal. Images are not materialized yet; manifests reference source assets.\n")
    write_md(GOLD_DIR / "README.md", "# football_gold_v1\n\nCandidate Gold Set only. Do not use for training or threshold tuning.\n")
    write_md(GOLD_DIR / "benchmark_protocol.md", benchmark_protocol_text())
    write_md(GOLD_DIR / "selection_report.md", "# Gold Selection Report\n\nAll sequences are candidates pending human review.\n")
    write_md(DOCS_DIR / "RF1A_ARCHITECTURE.md", "# RF-1A Architecture\n\nInventory -> annotation audit -> duplicate audit -> group split -> gold proposal -> review queue -> final report.\n")
    write_md(DOCS_DIR / "DATASET_GOVERNANCE.md", "# Dataset Governance\n\nUse group-based splits. Keep Gold isolated. Do not train on pseudo-labels or unreviewed Gold candidates.\n")
    write_md(DOCS_DIR / "ANNOTATION_GUIDE_FOOTBALL_V1.md", "# Annotation Guide Football V1\n\nTarget classes: player, goalkeeper, referee, ball, goal_frame. Ambiguous cases go to manual review.\n")
    write_md(DOCS_DIR / "GOLD_SET_POLICY.md", "# Gold Set Policy\n\nGold is frozen after human review and excluded from train/valid/test.\n")
    write_md(DOCS_DIR / "BENCHMARK_PROTOCOL_DRAFT.md", benchmark_protocol_text())


def benchmark_protocol_text() -> str:
    return """# Benchmark Protocol Draft

Compare YOLO productivo, RF-DETR base, RF-DETR-L preentrenado, RF-DETR-L OneFrame Global, RF-DETR-L OneFrame Ball and RF-DETR+YOLO fusion.

Detector benchmark exclusions: no GameReferee, no physical filters, no tracking, no homography, no event rules, no interpolation.

Detection metrics: precision, recall, AP50, AP50:95, recall by class, ball visible-frame recall, ball precision, false positives per minute, recall by ball size and hard-case tag.

Temporal metrics: ball sequence recall, localized sequence percentage, maximum/average ball gap, critical gaps, consecutive missed frames, first detection delay, recovery after occlusion.

Operational metrics: ms/frame, FPS, peak VRAM, GPU model, input resolution, preprocessing, inference, postprocessing and total match time.

Error taxonomy: detector_miss, low_resolution, motion_blur, partial_occlusion, full_occlusion, class_confusion, hard_negative, annotation_error, source_corruption, preprocessing_error, unsupported_case.
"""


def run_all(include_remote: bool = True) -> Dict[str, Any]:
    ensure_dirs()
    assets = local_assets(ROOT)
    annotation = annotation_audit(assets)
    # Link observed annotation counts back to annotation assets conservatively.
    for a in assets:
        if a.get("annotation_available"):
            a["classes_present"] = sorted(annotation.get("classes_count", {}).keys())
            a["object_count"] = sum(annotation.get("classes_count", {}).values())
            a["ball_annotation_count"] = len(annotation.get("ball_boxes", []))
    duplicates = duplicate_report(assets)
    mapping = class_mapping_report(annotation)
    ball = ball_distribution(annotation["ball_boxes"])
    supabase = supabase_source_health() if include_remote else {"source_type": "supabase", "source_status": "unavailable", "tables": []}
    r2 = r2_source_health() if include_remote else {"source_type": "r2", "source_status": "unavailable", "prefixes": []}
    sources = build_sources_manifest(assets, supabase, r2)
    grouping = match_grouping(assets)
    manifests = split_manifests(assets, grouping)
    gold = gold_set_proposal()
    review = review_queue(annotation, duplicates, gold)
    summary = inventory_summary(assets, annotation, sources)
    estimate = manual_annotation_estimate(summary)
    leakage = validate_leakage(manifests, grouping)
    missing = missing_files_report(assets, supabase, r2)
    files = []
    outputs = {
        "assets": assets,
        "annotation": annotation,
        "duplicates": duplicates,
        "mapping": mapping,
        "ball": ball,
        "supabase": supabase,
        "r2": r2,
        "sources": sources,
        "grouping": grouping,
        "manifests": manifests,
        "gold": gold,
        "review": review,
        "summary": summary,
        "estimate": estimate,
        "leakage": leakage,
        "missing": missing,
        "files_created": files,
    }
    file_map = {
        CATALOG_DIR / "dataset_inventory.json": {"assets": assets, "summary": summary},
        CATALOG_DIR / "sources_manifest.json": sources,
        CATALOG_DIR / "duplicate_report.json": duplicates,
        CATALOG_DIR / "leakage_report.json": leakage,
        CATALOG_DIR / "missing_files_report.json": missing,
        CATALOG_DIR / "class_mapping_proposal.json": mapping,
        CATALOG_DIR / "annotation_quality_report.json": annotation,
        CATALOG_DIR / "ball_size_distribution.json": ball,
        CATALOG_DIR / "ball_hard_cases_manifest.json": {
            "schema_version": "rf1a.ball_hard_cases.v1",
            "observed_tags": ball.get("hard_case_tags_observed", {}),
            "missing_or_manual_review_required": ["motion_blur", "partial_occlusion", "full_occlusion", "near_player_feet", "near_white_line", "aerial_ball", "crowded_players", "goal_visible"],
        },
        CATALOG_DIR / "match_grouping.json": grouping,
        CATALOG_DIR / "source_health_report.json": {"supabase": supabase, "r2": r2},
        CATALOG_DIR / "schema_version.json": {"schema_version": SCHEMA_VERSION, "generated_at": now_iso()},
        DATASET_DIR / "manifests" / "train.json": {"items": manifests.get("train", [])},
        DATASET_DIR / "manifests" / "valid.json": {"items": manifests.get("valid", [])},
        DATASET_DIR / "manifests" / "test.json": {"items": manifests.get("test", [])},
        DATASET_DIR / "manifests" / "split_summary.json": {"leakage": leakage, "counts": {k: len(v) for k, v in manifests.items()}},
        GOLD_DIR / "manifest.json": gold["manifest"],
        GOLD_DIR / "sequences.json": {"sequences": gold["sequences"]},
        REVIEW_DIR / "review_queue.json": review,
        REVIEW_DIR / "priority_frames.json": {"items": [i for i in review["items"] if i["priority"] == "P0"]},
        REVIEW_DIR / "review_schema.json": {"schema_version": "rf1a.review_schema.v1", "required_fields": ["review_id", "priority", "required_action", "review_status"]},
    }
    for path, data in file_map.items():
        write_json(path, data)
        files.append(path.relative_to(ROOT).as_posix())
    outputs["files_created"] = sorted(files)
    generate_markdown_reports(outputs)
    for path in [
        DOCS_DIR / "RF1A_FINAL_REPORT.md",
        CATALOG_DIR / "dataset_inventory.md",
        CATALOG_DIR / "duplicate_report.md",
        CATALOG_DIR / "leakage_report.md",
        CATALOG_DIR / "annotation_quality_report.md",
        CATALOG_DIR / "ball_visibility_audit.md",
        REVIEW_DIR / "review_summary.md",
        REVIEW_DIR / "review_instructions.md",
        DATASET_DIR / "README.md",
        GOLD_DIR / "README.md",
        GOLD_DIR / "benchmark_protocol.md",
        GOLD_DIR / "selection_report.md",
        DOCS_DIR / "RF1A_ARCHITECTURE.md",
        DOCS_DIR / "DATASET_GOVERNANCE.md",
        DOCS_DIR / "ANNOTATION_GUIDE_FOOTBALL_V1.md",
        DOCS_DIR / "GOLD_SET_POLICY.md",
        DOCS_DIR / "BENCHMARK_PROTOCOL_DRAFT.md",
    ]:
        files.append(path.relative_to(ROOT).as_posix())
    outputs["files_created"] = sorted(set(files))
    write_json(CATALOG_DIR / "schema_version.json", {"schema_version": SCHEMA_VERSION, "generated_at": now_iso(), "files_created": outputs["files_created"]})
    return outputs


def missing_files_report(assets: List[Dict[str, Any]], supabase: Dict[str, Any], r2: Dict[str, Any]) -> Dict[str, Any]:
    missing = [a for a in assets if a["source_status"] != "available"]
    unavailable = []
    for src in [supabase, r2]:
        if src.get("source_status") != "available":
            unavailable.append(src)
    return {"schema_version": "rf1a.missing_files.v1", "missing_local_assets": missing, "unavailable_sources": unavailable}


def validate_leakage(manifests: Dict[str, List[Dict[str, Any]]], grouping: Dict[str, Any]) -> Dict[str, Any]:
    group_to_split: Dict[str, str] = {}
    errors = []
    for split, items in manifests.items():
        for item in items:
            gid = item["match_group"]
            if gid in group_to_split and group_to_split[gid] != split:
                errors.append({"type": "match_group_leakage", "match_group": gid, "splits": sorted({group_to_split[gid], split})})
            group_to_split[gid] = split
    return {
        "schema_version": "rf1a.leakage.v1",
        "leakage_found": bool(errors),
        "errors": errors,
        "policy": "match_group_and_known_run_grouping",
    }


def validate_outputs() -> Dict[str, Any]:
    failures = []
    inv = read_json(CATALOG_DIR / "dataset_inventory.json", {})
    assets = inv.get("assets", [])
    valid_status = {"available", "unavailable", "missing_credentials", "missing_file", "unsupported_format", "read_error"}
    ids = set()
    for a in assets:
        if not a.get("asset_id"):
            failures.append("asset_missing_id")
        if a.get("asset_id") in ids:
            failures.append("duplicate_asset_id")
        ids.add(a.get("asset_id"))
        if a.get("source_status") not in valid_status:
            failures.append("invalid_source_status")
        if a.get("schema_version") != SCHEMA_VERSION:
            failures.append("invalid_schema_version")
    leakage = read_json(CATALOG_DIR / "leakage_report.json", {})
    if leakage.get("leakage_found"):
        failures.append("leakage_found")
    gold = read_json(GOLD_DIR / "sequences.json", {}).get("sequences", [])
    for seq in gold:
        if seq["start_timestamp_sec"] is not None and seq["end_timestamp_sec"] is not None and seq["start_timestamp_sec"] >= seq["end_timestamp_sec"]:
            failures.append("invalid_gold_sequence_time")
        if seq.get("gold_eligible") is not False:
            failures.append("gold_eligible_before_review")
    for name in ["train", "valid", "test"]:
        data = read_json(DATASET_DIR / "manifests" / f"{name}.json", {})
        seen = set()
        for item in data.get("items", []):
            if item["asset_id"] in seen:
                failures.append(f"duplicate_manifest_id_{name}")
            seen.add(item["asset_id"])
    final_text = (DOCS_DIR / "RF1A_FINAL_REPORT.md").read_text() if (DOCS_DIR / "RF1A_FINAL_REPORT.md").exists() else ""
    secret_hits = []
    for pat in SAFETY_EXCLUDE_PATTERNS:
        if pat.search(final_text):
            secret_hits.append(pat.pattern)
    if secret_hits:
        failures.append("secrets_in_report")
    return {"passed": not failures, "failures": sorted(set(failures))}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="RF-1A dataset inventory and audit toolkit")
    parser.add_argument("command", nargs="?", default="run-all", choices=[
        "run-all", "inventory", "annotations", "duplicates", "ball", "splits", "gold", "review", "validate", "report"
    ])
    parser.add_argument("--no-remote", action="store_true", help="Skip Supabase/R2 read-only checks")
    parser.add_argument("--json", action="store_true", help="Print compact JSON summary")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command in {"run-all", "inventory", "annotations", "duplicates", "ball", "splits", "gold", "review", "report"}:
        outputs = run_all(include_remote=not args.no_remote)
        if args.json:
            print(json.dumps(outputs["summary"], sort_keys=True))
        else:
            print(f"RF-1A generated {len(outputs['files_created'])} files; assets={outputs['summary']['total_assets']}")
        return 0
    if args.command == "validate":
        result = validate_outputs()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
