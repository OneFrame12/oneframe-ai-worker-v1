#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
PE0A3_RUN = AI_WORKER_ROOT / "runs" / "pe0a3_baseline_ccb_ec8836978221c786ed55a0ab_60s_05fps"
if str(AI_WORKER_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(AI_WORKER_ROOT / "scripts"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from run_pe0a3_baseline import (  # noqa: E402
    CANONICAL_BALL,
    CANONICAL_PERSON,
    bbox_iou,
    canonical_class,
    center_from_xyxy,
    clamp_box,
    dedupe_ball_candidates,
    generate_tiles,
    init_rfdetr,
    load_json,
    make_detection_row,
    maybe_write_parquet,
    point_spatial_status,
    polygon_points,
    rfdetr_predict,
    sha256_file,
    stable_id,
    video_meta,
    write_json,
    write_jsonl,
    write_text,
    yolo_predict,
)


PHASE = "PE-0A3R"
CAPTURE_ID = "ccb_ec8836978221c786ed55a0ab"
CALIBRATION_ID_OLD = "vc_a90e53754cb6083389782e25"
MATCH_ID = "test_match_2026-07-15T02-16-23-996Z"
VIDEO_HASH = "885c106cbf89a61b3fd38e9de015aad10decb9f6b47487843711e21115b4f2f9"
FRAME_W = 1920
FRAME_H = 1080
CLASS_BALL_ID = 1


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_points(points: List[List[float]]) -> List[List[float]]:
    return [[round(x / FRAME_W, 8), round(y / FRAME_H, 8)] for x, y in points]


def dict_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def polygon_self_intersects(points: List[List[float]]) -> bool:
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a, b, c, d):
        if a == c or a == d or b == c or b == d:
            return False
        return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0

    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            c, d = points[j], points[(j + 1) % n]
            if intersects(a, b, c, d):
                return True
    return False


def polygon_area(points: List[List[float]]) -> float:
    return abs(cv2.contourArea(np.array(points, dtype=np.float32)))


def full_field_points() -> List[List[float]]:
    # Expanded from PE-0A2C ROI after visual review: top/far boundary moved up,
    # right far goal corridor retained, near field kept tight to playable surface.
    return [
        [354.0, 966.0],
        [4.0, 948.0],
        [4.0, 286.0],
        [540.0, 96.0],
        [1210.0, 108.0],
        [1510.0, 145.0],
        [1918.0, 352.0],
        [1900.0, 1052.0],
        [1578.0, 1078.0],
        [363.0, 1068.0],
    ]


def ensure_dirs(run_dir: Path) -> Dict[str, Path]:
    dirs = {
        "calibration": run_dir / "calibration",
        "preannotations": run_dir / "preannotations",
        "frames": run_dir / "dataset" / "frames",
        "crops": run_dir / "dataset" / "crops",
        "annotations": run_dir / "dataset" / "annotations",
        "manifests": run_dir / "dataset" / "manifests",
        "review": run_dir / "review",
        "overlays": run_dir / "overlays",
        "metrics": run_dir / "metrics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_profile(run_dir: Path, old_profile: Dict[str, Any]) -> Dict[str, Any]:
    old_points = old_profile["detection_roi"]["polygon_pixels_reference"]
    new_points = full_field_points()
    new_profile = json.loads(json.dumps(old_profile))
    new_profile["calibration_id"] = f"vc_full_field_{dict_hash({'points': new_points})[:16]}"
    new_profile["parent_calibration_id"] = old_profile.get("calibration_id")
    new_profile["profile_version"] = int(old_profile.get("profile_version") or 1) + 1
    new_profile["status"] = "draft"
    new_profile["created_at"] = utc_now()
    new_profile["human_review_status"] = "pending"
    new_profile["detection_roi"] = dict(old_profile["detection_roi"])
    new_profile["detection_roi"]["polygon_pixels_reference"] = new_points
    new_profile["detection_roi"]["polygon_normalized"] = norm_points(new_points)
    new_profile["detection_roi"]["point_order"] = list(range(len(new_points)))
    new_profile["detection_roi"]["source"] = "pe0a3r_full_field_roi_correction"
    area_old = polygon_area(old_points)
    area_new = polygon_area(new_points)
    errors = []
    if polygon_self_intersects(new_points):
        errors.append("self_intersection")
    if any(x < 0 or y < 0 or x > FRAME_W or y > FRAME_H for x, y in new_points):
        errors.append("point_out_of_frame")
    validation = {
        "valid": not errors,
        "errors": errors,
        "self_intersecting": polygon_self_intersects(new_points),
        "points_inside_frame": not any(x < 0 or y < 0 or x > FRAME_W or y > FRAME_H for x, y in new_points),
        "area_ratio": round(area_new / (FRAME_W * FRAME_H), 8),
        "area_increase_vs_previous": round((area_new - area_old) / area_old, 6) if area_old else None,
        "far_goal_included_heuristic": min(y for _x, y in new_points) <= 110 and max(x for x, _y in new_points) >= 1910,
        "full_surface_included_heuristic": area_new > area_old and min(y for _x, y in new_points) < min(y for _x, y in old_points),
        "resolution": [FRAME_W, FRAME_H],
    }
    new_profile["detection_roi"]["validation"] = validation
    new_profile["detection_roi"]["valid"] = validation["valid"]

    deprecated = json.loads(json.dumps(old_profile))
    deprecated["status"] = "deprecated_for_full_field_detection"
    deprecated["deprecated_by_calibration_id"] = new_profile["calibration_id"]
    deprecated["deprecation_reason"] = "PE-0A3 visual review found far-field playable surface excluded by original ROI"
    write_json(run_dir / "calibration" / "deprecated_previous_profile_reference.json", deprecated)
    write_json(run_dir / "calibration" / "video_calibration_full_field.json", new_profile)
    write_json(run_dir / "calibration" / "roi_validation.json", validation)
    return new_profile


def draw_roi_overlay(reference_frame: Path, old_profile: Dict[str, Any], new_profile: Dict[str, Any], out_path: Path) -> None:
    frame = cv2.imread(str(reference_frame))
    if frame is None:
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    old = np.array(old_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.int32).reshape((-1, 1, 2))
    new = np.array(new_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [old], True, (0, 255, 255), 3)
    cv2.polylines(frame, [new], True, (0, 255, 0), 3)
    cv2.putText(frame, "yellow=previous ROI, green=full-field ROI", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def validate_before_after(run_dir: Path, old_profile: Dict[str, Any], new_profile: Dict[str, Any], source_video: Path) -> Dict[str, Any]:
    canonical = read_jsonl(PE0A3_RUN / "detections" / "canonical_detections.jsonl")
    old_poly = np.array(old_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.float32)
    new_poly = np.array(new_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.float32)
    person_rows = [row for row in canonical if row.get("canonical_class") == CANONICAL_PERSON]
    changed = []
    for row in person_rows:
        point = tuple(row.get("bottom_center") or row.get("center"))
        before = point_spatial_status(point, old_poly)
        after = point_spatial_status(point, new_poly)
        if before == "outside" and after == "inside":
            changed.append({**row, "before": before, "after": after})
    selected = changed[:5] or person_rows[:5]
    ball_rows = [row for row in canonical if row.get("canonical_class") == CANONICAL_BALL]
    ball_retained = len(ball_rows)
    external_probe_points = [[100, 80], [960, 20], [1850, 120]]
    external_people_status = [
        {"point": p, "status": point_spatial_status(tuple(p), new_poly)} for p in external_probe_points
    ]

    cap = cv2.VideoCapture(str(source_video))
    thumbs = []
    for item in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(item["frame_index"]))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        cv2.polylines(frame, [old_poly.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 255), 3)
        cv2.polylines(frame, [new_poly.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        x1, y1, x2, y2 = [int(round(v)) for v in item["bbox_xyxy"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, f"f={item['frame_index']} {item.get('before')}->{item.get('after')}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        thumbs.append(cv2.resize(frame, (480, 270)))
    cap.release()
    if thumbs:
        sheet = np.zeros((math.ceil(len(thumbs) / 2) * 270, 960, 3), dtype=np.uint8)
        for idx, img in enumerate(thumbs):
            y, x = divmod(idx, 2)
            sheet[y * 270 : y * 270 + 270, x * 480 : x * 480 + 480] = img
        cv2.imwrite(str(run_dir / "calibration" / "roi_before_after.png"), sheet)

    report = {
        "frames_checked": len(selected),
        "players_reclassified_outside_to_inside": len(changed),
        "examples": [
            {
                "frame_index": row["frame_index"],
                "timestamp_sec": row["timestamp_sec"],
                "bbox_xyxy": row["bbox_xyxy"],
                "before": row.get("before"),
                "after": row.get("after"),
            }
            for row in selected
        ],
        "external_probe_people_status": external_people_status,
        "ball_candidates_retained_independent_of_roi": ball_retained,
        "ball_roi_policy": "ROI is not used as a rigid exclusion for ball candidates",
    }
    write_json(run_dir / "calibration" / "spatial_status_before_after.json", report)
    md = [
        "# PE-0A3R ROI Correction Report",
        "",
        f"- previous_profile: `{old_profile.get('calibration_id')}`",
        f"- new_profile: `{new_profile.get('calibration_id')}`",
        f"- far_goal_included: `{new_profile['detection_roi']['validation']['far_goal_included_heuristic']}`",
        f"- players_reclassified_outside_to_inside: `{len(changed)}`",
        f"- external_probe_people_status: `{external_people_status}`",
        f"- ball_policy: `{report['ball_roi_policy']}`",
        f"- overlay: `calibration/roi_before_after.png`",
        "",
    ]
    write_text(run_dir / "calibration" / "roi_correction_report.md", "\n".join(md))
    return report


def select_dense_sequences(fps: float) -> List[Dict[str, Any]]:
    specs = [
        ("dense_normal_passes_01", 94.525, 99.525, "juego normal con pases", ["yolo_only", "candidate_motion"]),
        ("dense_feet_cluster_01", 108.525, 113.525, "balon junto a pies y jugadores agrupados", ["rfdetr_burst", "players_cluster"]),
        ("dense_goal_approach_01", 132.525, 137.525, "aproximacion o accion cerca de arco", ["late_attack", "goal_side"]),
    ]
    splits = {
        "dense_normal_passes_01": "train",
        "dense_feet_cluster_01": "valid",
        "dense_goal_approach_01": "within_video_test_v0",
    }
    return [
        {
            "sequence_id": sid,
            "start_sec": start,
            "end_sec": end,
            "duration_sec": round(end - start, 3),
            "original_frame_start": int(round(start * fps)),
            "original_frame_end": int(round(end * fps)) - 1,
            "fps_original": fps,
            "sampling_fps": 15,
            "motivo": reason,
            "candidate_sources": sources,
            "assigned_split": splits[sid],
            "review_status": "pending",
        }
        for sid, start, end, reason, sources in specs
    ]


def leakage_report(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    overlaps = []
    for i, left in enumerate(sequences):
        for right in sequences[i + 1 :]:
            if not (left["end_sec"] <= right["start_sec"] or right["end_sec"] <= left["start_sec"]):
                overlaps.append({"left": left["sequence_id"], "right": right["sequence_id"]})
    return {"status": "passed" if not overlaps else "failed", "overlaps": overlaps}


def crop_box_512(center: Tuple[float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(width - 512, int(round(center[0] - 256))))
    y1 = max(0, min(height - 512, int(round(center[1] - 256))))
    return x1, y1, x1 + 512, y1 + 512


def agreement_for_candidate(candidate: Dict[str, Any], yolo_rows: List[Dict[str, Any]], rfdetr_rows: List[Dict[str, Any]]) -> str:
    others = rfdetr_rows if candidate["source_model"] == "yolo" else yolo_rows
    for other in others:
        if other["frame_index"] != candidate["frame_index"]:
            continue
        dist = math.hypot(candidate["center"][0] - other["center"][0], candidate["center"][1] - other["center"][1])
        if bbox_iou(candidate["bbox_xyxy"], other["bbox_xyxy"]) >= 0.25 or dist <= 25:
            return "both"
    return f"{candidate['source_model']}_only"


def write_review_assets(run_dir: Path, frames: List[Dict[str, Any]], frame_candidates: Dict[str, List[Dict[str, Any]]]) -> None:
    queue = []
    for frame in frames:
        candidates = frame_candidates.get(frame["frame_id"], [])
        queue.append(
            {
                "frame_id": frame["frame_id"],
                "sequence_id": frame["sequence_id"],
                "split": frame["split"],
                "timestamp_sec": frame["timestamp_sec"],
                "frame_image": frame["image_relpath"],
                "crop_image": frame["review_crop_relpath"],
                "candidate_count": len(candidates),
                "priority": "P0" if candidates else "P2",
                "status": "pending",
            }
        )
    write_json(run_dir / "review" / "review_queue.json", {"items": queue})
    write_json(run_dir / "review" / "review_progress.json", {"total_frames": len(queue), "pending": len(queue), "reviewed_ball": 0, "reviewed_no_ball": 0, "reviewed_uncertain": 0})
    write_jsonl(run_dir / "review" / "review_decisions.jsonl", [])
    write_jsonl(run_dir / "review" / "review_audit_log.jsonl", [])
    write_json(run_dir / "review" / "reviewed_annotations_coco.json", {"images": [], "annotations": [], "categories": [{"id": CLASS_BALL_ID, "name": "ball"}]})
    write_json(run_dir / "review" / "reviewed_dataset_manifest.json", {"status": "pending_human_review", "ground_truth": False})
    write_text(run_dir / "review" / "review_instructions.md", "# PE-0A3R Ball Review\n\nReview every pending frame. Do not train until all validation/test frames and conflicts are reviewed.\n")


def artifact_manifest(run_dir: Path) -> Dict[str, Any]:
    artifacts = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path.name == ".DS_Store":
            continue
        artifacts.append({"relative_path": str(path.relative_to(run_dir)), "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"phase": PHASE, "artifacts": artifacts}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = args.run_id or f"pe0a3r_full_field_ball_data_{utc_now_compact()}"
    run_dir = AI_WORKER_ROOT / "runs" / run_id
    dirs = ensure_dirs(run_dir)
    started = time.time()

    old_profile_path = Path(args.profile)
    old_profile = load_json(old_profile_path)
    source_video = PE0A3_RUN / "source_video.mp4"
    if not source_video.exists():
        raise FileNotFoundError(source_video)
    source_meta = video_meta(source_video)
    if source_meta["sha256"] != VIDEO_HASH:
        raise ValueError("source video hash mismatch")
    reference_frame = Path("/tmp/oneframe_pe0a2c_capture/session_20260715T013505Z/ccb_ec8836978221c786ed55a0ab/reference_frame.png")
    new_profile = write_profile(run_dir, old_profile)
    draw_roi_overlay(reference_frame, old_profile, new_profile, run_dir / "calibration" / "video_calibration_overlay_full_field.png")
    roi_report = validate_before_after(run_dir, old_profile, new_profile, source_video)

    import torch
    import ultralytics

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(str(args.yolo_model))
    rfdetr = init_rfdetr(str(args.rfdetr_model), device=device)
    write_json(
        run_dir / "model_manifest.json",
        {
            "yolo": {"checkpoint": str(args.yolo_model), "sha256": sha256_file(args.yolo_model), "ultralytics_version": ultralytics.__version__, "purpose": "ball_baseline_only"},
            "rfdetr": {"checkpoint": str(args.rfdetr_model), "sha256": sha256_file(args.rfdetr_model), "class": "RFDETRBase", "purpose": "ball_preannotation"},
            "device": device,
        },
    )

    sequences = select_dense_sequences(source_meta["fps"])
    write_json(run_dir / "sequences_manifest.json", {"sequences": sequences})
    split_manifest = {"splits": defaultdict(list)}
    for seq in sequences:
        split_manifest["splits"][seq["assigned_split"]].append(seq["sequence_id"])
    split_manifest["splits"] = dict(split_manifest["splits"])
    write_json(run_dir / "split_manifest.json", split_manifest)
    leakage = leakage_report(sequences)
    write_json(run_dir / "leakage_report.json", leakage)

    cap = cv2.VideoCapture(str(source_video))
    old_poly = np.array(old_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.float32)
    new_poly = np.array(new_profile["detection_roi"]["polygon_pixels_reference"], dtype=np.float32)
    width, height = int(source_meta["width"]), int(source_meta["height"])
    tiles = generate_tiles(width, height, args.tile_size, args.tile_overlap, new_poly)

    frames_manifest = []
    all_candidates = []
    frame_candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    counters = Counter()
    process_every = max(1, int(round(source_meta["fps"] / 15.0)))

    for seq in sequences:
        frame_index = int(round(seq["start_sec"] * source_meta["fps"]))
        end_frame = int(round(seq["end_sec"] * source_meta["fps"]))
        processed_in_seq = 0
        while frame_index < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            timestamp = frame_index / source_meta["fps"]
            frame_id = f"{seq['sequence_id']}_f{frame_index:08d}"
            image_rel = Path("dataset") / "frames" / seq["assigned_split"] / f"{frame_id}.jpg"
            image_path = run_dir / image_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), frame)

            y_raw = yolo_predict(yolo, frame, args.yolo_imgsz, args.confidence, args.iou)
            r_global = rfdetr_predict(rfdetr, frame, args.rfdetr_confidence)
            y_rows = [
                make_detection_row(run_id=run_id, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="yolo", source_pass="global", tile_id=None, raw=raw, width=width, height=height, roi_polygon=new_poly)
                for raw in y_raw
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]
            r_rows = [
                make_detection_row(run_id=run_id, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="rfdetr", source_pass="global", tile_id=None, raw=raw, width=width, height=height, roi_polygon=new_poly)
                for raw in r_global
                if canonical_class(str(raw.get("class_name", ""))) == CANONICAL_BALL
            ]
            counters["rfdetr_global_frames"] += 1 if r_rows else 0
            counters["yolo_frames"] += 1 if y_rows else 0

            for tile in tiles:
                crop = frame[tile["y1"] : tile["y2"], tile["x1"] : tile["x2"]]
                if crop.size == 0:
                    continue
                for raw in rfdetr_predict(rfdetr, crop, args.rfdetr_confidence):
                    if canonical_class(str(raw.get("class_name", ""))) != CANONICAL_BALL:
                        continue
                    raw = dict(raw)
                    raw["bbox_xyxy"] = [raw["bbox_xyxy"][0] + tile["x1"], raw["bbox_xyxy"][1] + tile["y1"], raw["bbox_xyxy"][2] + tile["x1"], raw["bbox_xyxy"][3] + tile["y1"]]
                    r_rows.append(make_detection_row(run_id=run_id, frame_index=frame_index, processed_index=processed_in_seq, timestamp_sec=timestamp, model_name="rfdetr", source_pass="tile", tile_id=tile["tile_id"], raw=raw, width=width, height=height, roi_polygon=new_poly))
            y_rows = dedupe_ball_candidates(y_rows)
            r_rows = dedupe_ball_candidates(r_rows)
            counters["rfdetr_tile_frames"] += 1 if any(row["source_pass"] == "tile" for row in r_rows) else 0

            candidates = []
            for row in y_rows + r_rows:
                row = dict(row)
                row["frame_id"] = frame_id
                row["sequence_id"] = seq["sequence_id"]
                row["split"] = seq["assigned_split"]
                row["agreement"] = agreement_for_candidate(row, y_rows, r_rows)
                row["candidate_status"] = "unreviewed"
                row["pseudo_label"] = True
                row["ground_truth"] = False
                candidates.append(row)
            if len({c["agreement"] for c in candidates}) > 1 or any(c["agreement"] != "both" for c in candidates):
                counters["conflict_frames"] += 1 if candidates else 0
            all_candidates.extend(candidates)
            frame_candidates[frame_id] = candidates

            center = center_from_xyxy(candidates[0]["bbox_xyxy"]) if candidates else (width / 2.0, height / 2.0)
            x1, y1, x2, y2 = crop_box_512(center, width, height)
            crop_rel = Path("dataset") / "crops" / seq["assigned_split"] / f"{frame_id}_review.jpg"
            crop_img = frame[y1:y2, x1:x2]
            crop_path = run_dir / crop_rel
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_path), crop_img)

            overlay = frame.copy()
            for row in candidates:
                bx1, by1, bx2, by2 = [int(round(v)) for v in row["bbox_xyxy"]]
                color = (0, 0, 255) if row["source_model"] == "rfdetr" else (0, 255, 255)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), color, 2)
                cv2.putText(overlay, f"{row['source_model']}:{row['source_pass']} {row['confidence']}", (bx1, max(18, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.polylines(overlay, [new_poly.astype(np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 2)
            overlay_rel = Path("overlays") / f"{frame_id}.jpg"
            cv2.imwrite(str(run_dir / overlay_rel), overlay)

            frames_manifest.append(
                {
                    "frame_id": frame_id,
                    "sequence_id": seq["sequence_id"],
                    "split": seq["assigned_split"],
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp, 6),
                    "image_relpath": str(image_rel),
                    "review_crop_relpath": str(crop_rel),
                    "overlay_relpath": str(overlay_rel),
                    "candidate_count": len(candidates),
                    "status": "pending",
                }
            )
            if len(frames_manifest) % 10 == 0:
                print(
                    json.dumps(
                        {
                            "progress": "pe0a3r_preannotation",
                            "frames": len(frames_manifest),
                            "sequence_id": seq["sequence_id"],
                            "last_frame_index": frame_index,
                            "candidates_total": len(all_candidates),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            frame_index += process_every
            processed_in_seq += 1

    cap.release()
    write_json(run_dir / "source_manifest.json", {"source_video": str(source_video), "video_hash": source_meta["sha256"], "capture_id": CAPTURE_ID, "match_id": MATCH_ID, "fps": source_meta["fps"], "resolution": [width, height]})
    write_json(run_dir / "dataset" / "manifests" / "frames_manifest.json", {"frames": frames_manifest})
    write_jsonl(run_dir / "preannotations" / "ball_candidates.jsonl", all_candidates)
    maybe_write_parquet(run_dir / "preannotations" / "ball_candidates.parquet", all_candidates)
    write_review_assets(run_dir, frames_manifest, frame_candidates)

    frames_with_candidates = len({row["frame_id"] for row in all_candidates})
    no_candidate = len(frames_manifest) - frames_with_candidates
    summary = {
        "phase": PHASE,
        "run_id": run_id,
        "status": "ready_for_ball_review",
        "sequences": len(sequences),
        "duration_total_sec": sum(seq["duration_sec"] for seq in sequences),
        "sampling_fps": 15,
        "frames": len(frames_manifest),
        "rfdetr_global_frames": counters["rfdetr_global_frames"],
        "rfdetr_tile_frames": counters["rfdetr_tile_frames"],
        "yolo_frames": counters["yolo_frames"],
        "candidate_count": len(all_candidates),
        "conflict_frames": counters["conflict_frames"],
        "frames_without_candidate": no_candidate,
        "pending_review": len(frames_manifest),
        "runtime_sec": time.time() - started,
        "roi_profile_old": CALIBRATION_ID_OLD,
        "roi_profile_new": new_profile["calibration_id"],
        "far_goal_included": new_profile["detection_roi"]["validation"]["far_goal_included_heuristic"],
        "players_reclassified": roi_report["players_reclassified_outside_to_inside"],
    }
    write_json(run_dir / "metrics" / "pe0a3r_summary.json", summary)
    write_json(run_dir / "artifact_manifest.json", artifact_manifest(run_dir))
    write_text(
        run_dir / "PE0A3R_FINAL_REPORT.md",
        "# PE-0A3R FULL-FIELD ROI + DENSE BALL DATA\n\n"
        f"- status: `{summary['status']}`\n"
        f"- old_profile: `{summary['roi_profile_old']}`\n"
        f"- new_profile: `{summary['roi_profile_new']}`\n"
        f"- far_goal_included: `{summary['far_goal_included']}`\n"
        f"- players_reclassified: `{summary['players_reclassified']}`\n"
        f"- sequences: `{summary['sequences']}`\n"
        f"- frames: `{summary['frames']}`\n"
        f"- rfdetr_global_frames: `{summary['rfdetr_global_frames']}`\n"
        f"- rfdetr_tile_frames: `{summary['rfdetr_tile_frames']}`\n"
        f"- yolo_frames: `{summary['yolo_frames']}`\n"
        f"- conflict_frames: `{summary['conflict_frames']}`\n"
        f"- frames_without_candidate: `{summary['frames_without_candidate']}`\n\n"
        "Training is blocked until human ball review is complete.\n",
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--profile", default="/tmp/oneframe_pe0a2c_capture/session_20260715T013505Z/materialized/pe0a2c_real_capture_ccb_ec8836978221c786ed55a0ab/video_calibration.json")
    parser.add_argument("--yolo-model", type=Path, default=AI_WORKER_ROOT / "src" / "oneframe_v3_best.pt")
    parser.add_argument("--rfdetr-model", type=Path, default=AI_WORKER_ROOT / "rfdetr_cache" / "rf-detr-base.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--rfdetr-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
