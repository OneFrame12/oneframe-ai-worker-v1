#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


FRAME_W = 1920
FRAME_H = 1080
BALL_SPECIALIST_SHA256 = "9dda20e4e7363a284a9775ff3aac4c10280ecd4c86299127be2c5e77a7b64d55"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024 * 4), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def clamp_box(box: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]


def center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bottom_center(box: List[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, box[3])


def distance_bucket(box: List[float], height: int) -> str:
    _, y = bottom_center(box)
    rel = y / max(height, 1)
    if rel >= 0.66:
        return "near"
    if rel >= 0.38:
        return "mid"
    return "far"


def person_region(box: List[float], geometry: Dict[str, Any]) -> Tuple[bool, str]:
    p = bottom_center(box)
    inside_perception = point_in_polygon(p, geometry["perception_roi"])
    inside_field = point_in_polygon(p, geometry["detection_field_roi"])
    inside_near = point_in_polygon(p, geometry["goal_zones"]["near_goal"])
    inside_far = point_in_polygon(p, geometry["goal_zones"]["far_goal"])
    if not inside_perception:
        return False, "outside_perception_roi"
    if inside_field:
        return True, "detection_field_roi"
    if inside_near:
        return True, "near_goal_zone"
    if inside_far:
        return True, "far_goal_zone"
    return False, "outside_detection_field"


def ball_region(box: List[float], geometry: Dict[str, Any]) -> Tuple[bool, str]:
    p = center(box)
    if point_in_polygon(p, geometry["perception_roi"]):
        return True, "perception_roi"
    return False, "outside_perception_roi"


def draw_poly(frame: np.ndarray, polygon: List[List[float]], color: Tuple[int, int, int], label: str) -> None:
    if len(polygon) < 2:
        return
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    cv2.polylines(frame, [pts], True, color, 3)
    cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_geometry(frame: np.ndarray, geometry: Dict[str, Any]) -> None:
    draw_poly(frame, geometry["perception_roi"], (255, 0, 0), "perception_roi")
    draw_poly(frame, geometry["detection_field_roi"], (0, 230, 0), "detection_field_roi")
    draw_poly(frame, geometry["goal_zones"]["near_goal"], (0, 255, 255), "near_goal")
    draw_poly(frame, geometry["goal_zones"]["far_goal"], (0, 255, 255), "far_goal")


@dataclass
class Det:
    model: str
    cls: str
    conf: float
    box: List[float]
    accepted: bool
    region: str
    source_pass: str = "global"


def draw_det(frame: np.ndarray, det: Det, color: Tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in det.box]
    line = 3 if det.accepted else 2
    draw_color = color if det.accepted else (80, 80, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, line)
    label = (
        f"{det.model} {det.cls} {det.conf:.2f} "
        f"accepted_on_field={str(det.accepted).lower()} {det.region} {distance_bucket(det.box, frame.shape[0])}"
    )
    cv2.putText(frame, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
    if det.cls == "ball":
        c = center(det.box)
        cv2.circle(frame, (int(c[0]), int(c[1])), 5, draw_color, -1)


def draw_header(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (1920, 58), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)


def load_rfdetr_model(device: str, model_size: str = "small", weights: Optional[Path] = None) -> Any:
    import rfdetr

    if model_size == "base":
        klass = getattr(rfdetr, "RFDETRBase")
    else:
        klass = getattr(rfdetr, "RFDETRSmall", None) or getattr(rfdetr, "RFDETRBase")
    kwargs = {"device": device}
    if weights and weights.exists():
        kwargs["pretrain_weights"] = str(weights)
    return klass(**kwargs)


def rfdetr_predict(model: Any, frame: np.ndarray, threshold: float, geometry: Dict[str, Any], wanted: str, model_name: str) -> List[Det]:
    rgb = frame[:, :, ::-1].copy()
    results = model.predict(rgb, threshold=threshold)
    class_names = getattr(model, "class_names", {}) or {}
    out: List[Det] = []
    if results is None or not all(hasattr(results, attr) for attr in ("xyxy", "confidence", "class_id")):
        return out
    for xyxy, conf, cls in zip(results.xyxy, results.confidence, results.class_id):
        class_id = int(cls)
        raw_name = str(class_names.get(class_id, class_id)).lower().replace("_", " ")
        if wanted == "person" and raw_name != "person":
            continue
        if wanted == "ball" and raw_name not in {"sports ball", "ball"}:
            continue
        box = clamp_box([float(v) for v in xyxy], frame.shape[1], frame.shape[0])
        if wanted == "person":
            accepted, region = person_region(box, geometry)
        else:
            accepted, region = ball_region(box, geometry)
        out.append(Det(model_name, wanted, float(conf), box, accepted, region))
    out.sort(key=lambda d: d.conf, reverse=True)
    return out


def yolo_predict(model: Any, frame: np.ndarray, geometry: Dict[str, Any], wanted: str, model_name: str, conf: float) -> List[Det]:
    results = model.predict(frame, imgsz=1280, conf=conf, iou=0.5, verbose=False)
    names = getattr(model, "names", {}) or {}
    out: List[Det] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0].item()) if box.cls is not None else -1
            name = str(names.get(cls, cls)).lower().replace("_", " ")
            if wanted == "person" and name != "person":
                continue
            if wanted == "ball" and name not in {"sports ball", "ball"}:
                continue
            xyxy = clamp_box([float(v) for v in box.xyxy[0].tolist()], frame.shape[1], frame.shape[0])
            confv = float(box.conf[0].item()) if box.conf is not None else 0.0
            if wanted == "person":
                accepted, region = person_region(xyxy, geometry)
            else:
                accepted, region = ball_region(xyxy, geometry)
            out.append(Det(model_name, wanted, confv, xyxy, accepted, region))
    out.sort(key=lambda d: d.conf, reverse=True)
    return out


def writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def windows(duration: float) -> List[Tuple[float, float]]:
    starts = []
    for ratio in (0.10, 0.50, 0.90):
        center_sec = duration * ratio
        starts.append(max(0.0, min(duration - 30.0, center_sec - 15.0)))
    return [(round(s, 3), round(min(duration, s + 30.0), 3)) for s in starts]


def read_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(frames / fps) if fps else 0.0


def frame_iter(path: Path, duration: float, smoke_fps: float) -> Iterable[Tuple[int, float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    for start, end in windows(duration):
        t = start
        while t < end:
            frame_idx = int(round(t * source_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                yield frame_idx, t, frame
            t += 1.0 / smoke_fps
    cap.release()


def render_panel(base: np.ndarray, geometry: Dict[str, Any], detections: List[Det], header: str, color: Tuple[int, int, int]) -> np.ndarray:
    frame = base.copy()
    draw_geometry(frame, geometry)
    for det in detections[:80]:
        draw_det(frame, det, color)
    draw_header(frame, header)
    return frame


def stack_side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.hstack([
        cv2.resize(left, (960, 540), interpolation=cv2.INTER_AREA),
        cv2.resize(right, (960, 540), interpolation=cv2.INTER_AREA),
    ])


def stack_three(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    top = np.hstack([
        cv2.resize(a, (960, 540), interpolation=cv2.INTER_AREA),
        cv2.resize(b, (960, 540), interpolation=cv2.INTER_AREA),
    ])
    bottom = cv2.resize(c, (1920, 540), interpolation=cv2.INTER_AREA)
    return np.vstack([top, bottom])


def run_video(
    *,
    video_id: str,
    video_path: Path,
    profile_path: Path,
    out_dir: Path,
    models: Dict[str, Any],
    smoke_fps: float,
) -> Dict[str, Any]:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    geometry = profile["geometry"]
    duration = read_duration(video_path)
    video_out = out_dir / video_id
    persons_rfdetr_path = video_out / "persons_rfdetr_base_smoke.mp4"
    persons_yolo_path = video_out / "persons_yolo_baseline_smoke.mp4"
    persons_cmp_path = video_out / "persons_rfdetr_vs_yolo_smoke.mp4"
    ball_cmp_path = video_out / "ball_three_way_smoke.mp4"
    combined_path = video_out / "combined_perception_smoke.mp4"

    writers = {
        "persons_rfdetr": writer(persons_rfdetr_path, smoke_fps, 1920, 1080),
        "persons_yolo": writer(persons_yolo_path, smoke_fps, 1920, 1080),
        "persons_cmp": writer(persons_cmp_path, smoke_fps, 1920, 540),
        "ball_cmp": writer(ball_cmp_path, smoke_fps, 1920, 1080),
        "combined": writer(combined_path, smoke_fps, 1920, 1080),
    }
    counters = Counter()
    errors: List[str] = []
    t_start = time.perf_counter()
    for frame_idx, ts, frame in frame_iter(video_path, duration, smoke_fps):
        counters["frames_processed"] += 1
        try:
            rfdetr_people = rfdetr_predict(models["rfdetr_person"], frame, 0.35, geometry, "person", "RFDETR-S")
        except Exception as exc:
            rfdetr_people = []
            errors.append(f"rfdetr_person:{frame_idx}:{exc}")
        try:
            yolo_people = yolo_predict(models["yolo_person"], frame, geometry, "person", "YOLO", 0.25)
        except Exception as exc:
            yolo_people = []
            errors.append(f"yolo_person:{frame_idx}:{exc}")
        try:
            yolo_ball = yolo_predict(models["yolo_ball"], frame, geometry, "ball", "YOLO-ball", 0.20)
        except Exception as exc:
            yolo_ball = []
            errors.append(f"yolo_ball:{frame_idx}:{exc}")
        try:
            rfdetr_ball = rfdetr_predict(models["rfdetr_ball"], frame, 0.20, geometry, "ball", "RFDETR-S")
        except Exception as exc:
            rfdetr_ball = []
            errors.append(f"rfdetr_ball:{frame_idx}:{exc}")
        try:
            specialist_ball = rfdetr_predict(models["ball_specialist"], frame, 0.20, geometry, "ball", "BallSpecialistV0")
        except Exception as exc:
            specialist_ball = []
            errors.append(f"ball_specialist:{frame_idx}:{exc}")

        counters["rfdetr_person_detections"] += len(rfdetr_people)
        counters["yolo_person_detections"] += len(yolo_people)
        counters["yolo_ball_frames"] += int(bool(yolo_ball))
        counters["rfdetr_ball_frames"] += int(bool(rfdetr_ball))
        counters["specialist_ball_frames"] += int(bool(specialist_ball))
        counters["person_disagreement_frames"] += int(bool(rfdetr_people) != bool(yolo_people))
        counters["ball_disagreement_frames"] += int(len({bool(yolo_ball), bool(rfdetr_ball), bool(specialist_ball)}) > 1)

        base_header = f"{video_id} frame={frame_idx} t={ts:.2f}s smoke_fps={smoke_fps} errors={len(errors)}"
        rf_panel = render_panel(frame, geometry, rfdetr_people, base_header + f" RF-DETR raw={len(rfdetr_people)}", (40, 220, 40))
        yo_panel = render_panel(frame, geometry, yolo_people, base_header + f" YOLO raw={len(yolo_people)}", (0, 200, 255))
        ball_panel = frame.copy()
        draw_geometry(ball_panel, geometry)
        for det in yolo_ball[:20]:
            draw_det(ball_panel, det, (0, 200, 255))
        for det in rfdetr_ball[:20]:
            draw_det(ball_panel, det, (40, 220, 40))
        for det in specialist_ball[:20]:
            draw_det(ball_panel, det, (255, 0, 255))
        draw_header(ball_panel, base_header + f" ball candidates YOLO={len(yolo_ball)} RFDETR={len(rfdetr_ball)} SPEC={len(specialist_ball)}")
        combined = frame.copy()
        draw_geometry(combined, geometry)
        for det in rfdetr_people[:80]:
            draw_det(combined, det, (40, 220, 40))
        for det in specialist_ball[:20]:
            draw_det(combined, det, (255, 0, 255))
        draw_header(combined, base_header + f" combined RFDETR people={len(rfdetr_people)} specialist ball={len(specialist_ball)}")

        writers["persons_rfdetr"].write(rf_panel)
        writers["persons_yolo"].write(yo_panel)
        writers["persons_cmp"].write(stack_side_by_side(rf_panel, yo_panel))
        writers["ball_cmp"].write(stack_three(
            render_panel(frame, geometry, yolo_ball, base_header + f" YOLO ball={len(yolo_ball)}", (0, 200, 255)),
            render_panel(frame, geometry, rfdetr_ball, base_header + f" RF-DETR ball={len(rfdetr_ball)}", (40, 220, 40)),
            render_panel(frame, geometry, specialist_ball, base_header + f" Specialist ball={len(specialist_ball)}", (255, 0, 255)),
        ))
        writers["combined"].write(combined)

    for w in writers.values():
        w.release()
    outputs = {
        "persons_rfdetr_base_smoke": persons_rfdetr_path,
        "persons_yolo_baseline_smoke": persons_yolo_path,
        "persons_rfdetr_vs_yolo_smoke": persons_cmp_path,
        "ball_three_way_smoke": ball_cmp_path,
        "combined_perception_smoke": combined_path,
    }
    return {
        "video_id": video_id,
        "source_video": str(video_path),
        "windows": windows(duration),
        "duration_sec": duration,
        "runtime_sec": round(time.perf_counter() - t_start, 3),
        "counters": dict(counters),
        "runtime_errors": errors[:100],
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PE-0 multivideo visual smoke MP4s.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-fps", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ultralytics import YOLO

    import torch

    input_root = args.input_root
    models_root = input_root / "models"
    videos_root = input_root / "videos"
    profiles_root = input_root / "roi_profiles"
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ball_specialist = models_root / "ball_specialist_v0_best.pth"
    if sha256_file(ball_specialist) != BALL_SPECIALIST_SHA256:
        raise RuntimeError("Ball Specialist v0 SHA256 mismatch")

    yolo_ball = YOLO(str(models_root / "oneframe_v3_best.pt"))
    yolo_person = YOLO("yolo11n.pt")
    rfdetr_person = load_rfdetr_model(args.device, "small", None)
    rfdetr_ball = rfdetr_person
    ball_specialist_model = load_rfdetr_model(args.device, "small", ball_specialist)
    models = {
        "yolo_ball": yolo_ball,
        "yolo_person": yolo_person,
        "rfdetr_person": rfdetr_person,
        "rfdetr_ball": rfdetr_ball,
        "ball_specialist": ball_specialist_model,
    }
    results = []
    for video_path in sorted(videos_root.glob("video_*.mp4")):
        stem = video_path.stem
        profile_matches = list(profiles_root.glob(f"*{stem}*/roi_manual_v3_1_profile.json"))
        if not profile_matches:
            profile_matches = list(profiles_root.glob(f"*{stem}*.json"))
        if not profile_matches:
            raise FileNotFoundError(f"Missing ROI profile for {stem}")
        video_id = profile_matches[0].parent.name
        results.append(
            run_video(
                video_id=video_id,
                video_path=video_path,
                profile_path=profile_matches[0],
                out_dir=out_dir,
                models=models,
                smoke_fps=args.smoke_fps,
            )
        )
    summary = {
        "phase": "PE-0 MULTIVIDEO VISUAL SMOKE",
        "created_at": utc_now(),
        "device": args.device,
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "smoke_fps": args.smoke_fps,
        "videos": results,
        "restrictions": {
            "tracking": False,
            "interpolation": False,
            "identity": False,
            "teams": False,
            "possession": False,
            "passes": False,
            "shots": False,
            "events": False,
            "tp_fp_fn_without_gold_set": False,
        },
    }
    write_json(out_dir / "visual_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
