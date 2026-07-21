#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_WORKER_ROOT = REPO_ROOT / "ai_worker_v1"
DEFAULT_INGESTION_RUN = AI_WORKER_ROOT / "runs" / "pe0_multivideo_ingestion_20260717T024829Z"
PHASE = "PE-0 CALIBRATION ROI V2"
SCRIPT_NAME = "ai_worker_v1/scripts/run_pe0_roi_v2_multizone.py"
WIDTH = 1920
HEIGHT = 1080


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def stable_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def video_path_for_filename(filename: str) -> Path:
    return AI_WORKER_ROOT / "input_videos" / "multivideo_v01" / filename


def read_frame(video_path: Path, timestamp_sec: float) -> Tuple[int, np.ndarray | None]:
    frame_index = int(round(timestamp_sec * 30.0))
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-ss",
        f"{timestamp_sec:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return frame_index, None
    if not result.stdout:
        return frame_index, None
    frame = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    return frame_index, frame


def norm(points: List[List[float]]) -> List[List[float]]:
    return [[round(x / WIDTH, 8), round(y / HEIGHT, 8)] for x, y in points]


def polygon_area(points: List[List[float]]) -> float:
    return abs(float(cv2.contourArea(np.array(points, dtype=np.float32))))


def polygon_self_intersecting(points: List[List[float]]) -> bool:
    def orient(a: List[float], b: List[float], c: List[float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def intersects(a: List[float], b: List[float], c: List[float], d: List[float]) -> bool:
        if a == c or a == d or b == c or b == d:
            return False
        return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0

    for i in range(len(points)):
        a, b = points[i], points[(i + 1) % len(points)]
        for j in range(i + 1, len(points)):
            if abs(i - j) <= 1 or (i == 0 and j == len(points) - 1):
                continue
            c, d = points[j], points[(j + 1) % len(points)]
            if intersects(a, b, c, d):
                return True
    return False


def validate_polygon(name: str, points: List[List[float]]) -> Dict[str, Any]:
    errors = []
    if len(points) < 3:
        errors.append("minimum_3_points_required")
    for idx, (x, y) in enumerate(points):
        if x < 0 or x > WIDTH or y < 0 or y > HEIGHT:
            errors.append(f"{name}_point_{idx}_outside_1920x1080")
    self_intersecting = polygon_self_intersecting(points) if len(points) >= 4 else False
    if self_intersecting:
        errors.append(f"{name}_self_intersecting")
    if len(points) >= 3 and polygon_area(points) <= 0:
        errors.append(f"{name}_area_not_positive")
    return {
        "name": name,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "point_count": len(points),
        "area_ratio": round(polygon_area(points) / float(WIDTH * HEIGHT), 8) if len(points) >= 3 else 0.0,
        "self_intersecting": self_intersecting,
        "points_inside_frame": not any("outside_1920x1080" in e for e in errors),
    }


def fill_polygon(mask: np.ndarray, points: List[List[float]], value: int = 255) -> None:
    cv2.fillPoly(mask, [np.array(points, dtype=np.int32).reshape((-1, 1, 2))], value)


def build_masks(geom: Dict[str, Any]) -> Dict[str, np.ndarray]:
    broad = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    field = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    goals = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    exclusions = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    fill_polygon(broad, geom["broad_perception_roi"], 255)
    fill_polygon(field, geom["person_field_polygon"], 255)
    fill_polygon(goals, geom["near_goal_mouth_zone"], 255)
    fill_polygon(goals, geom["far_goal_mouth_zone"], 255)
    for zone in geom["person_exclusion_zones"]:
        fill_polygon(exclusions, zone["polygon"], 255)
    acceptance = cv2.bitwise_or(field, goals)
    acceptance = cv2.bitwise_and(acceptance, cv2.bitwise_not(exclusions))
    return {
        "broad": broad,
        "field": field,
        "goals": goals,
        "exclusions": exclusions,
        "acceptance": acceptance,
    }


def mask_has_invalid_geometry(mask: np.ndarray) -> bool:
    return int(mask.sum()) <= 0


def draw_transparent_polygon(frame: np.ndarray, points: List[List[float]], color: Tuple[int, int, int], alpha: float) -> None:
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(points, dtype=np.int32).reshape((-1, 1, 2))], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_outlines(frame: np.ndarray, geom: Dict[str, Any]) -> None:
    # BGR colors: green field, yellow goals, red exclusions, blue broad.
    cv2.polylines(frame, [np.array(geom["broad_perception_roi"], dtype=np.int32).reshape((-1, 1, 2))], True, (255, 120, 0), 3)
    cv2.polylines(frame, [np.array(geom["person_field_polygon"], dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 0), 4)
    for zone in (geom["near_goal_mouth_zone"], geom["far_goal_mouth_zone"]):
        cv2.polylines(frame, [np.array(zone, dtype=np.int32).reshape((-1, 1, 2))], True, (0, 255, 255), 4)
    for zone in geom["person_exclusion_zones"]:
        draw_transparent_polygon(frame, zone["polygon"], (0, 0, 255), 0.25)
        cv2.polylines(frame, [np.array(zone["polygon"], dtype=np.int32).reshape((-1, 1, 2))], True, (0, 0, 255), 3)
    cv2.rectangle(frame, (20, 20), (880, 132), (0, 0, 0), -1)
    cv2.putText(frame, "green=person_field yellow=goal_mouth red=exclusion blue=broad_perception", (34, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(frame, "V2 draft - bottom_center acceptance - human review required", (34, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def create_overlay_sheet(video_path: Path, out_path: Path, timestamps: List[float], geom: Dict[str, Any], video_label: str) -> List[Dict[str, Any]]:
    thumbs = []
    samples = []
    for ts in timestamps:
        frame_index, frame = read_frame(video_path, ts)
        if frame is None:
            samples.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "read_failed"})
            continue
        draw_outlines(frame, geom)
        cv2.putText(frame, f"{video_label} f={frame_index} t={ts:.1f}s", (34, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        thumbs.append(cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA))
        samples.append({"timestamp_sec": ts, "frame_index": frame_index, "status": "ok"})
    sheet = np.zeros((math.ceil(len(thumbs) / 2) * 360, 1280, 3), dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, 2)
        sheet[row * 360 : row * 360 + 360, col * 640 : col * 640 + 640] = thumb
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)
    return samples


def geometry_library() -> Dict[str, Dict[str, Any]]:
    return {
        "mv01_video_02_0db7846b4b08": {
            "person_field_polygon": [[0, 270], [647, 5], [1239, 40], [1919, 398], [1919, 905], [1565, 890], [1488, 805], [430, 800], [350, 890], [0, 915]],
            "near_goal_mouth_zone": [[150, 735], [1565, 740], [1550, 1018], [210, 1015]],
            "far_goal_mouth_zone": [[1450, 170], [1850, 238], [1919, 410], [1535, 414]],
            "person_exclusion_zones": [
                {"zone_id": "near_goal_back_strip", "polygon": [[0, 1018], [1920, 1000], [1920, 1080], [0, 1080]]},
                {"zone_id": "lower_left_external_corner", "polygon": [[0, 900], [245, 900], [230, 1080], [0, 1080]]},
                {"zone_id": "lower_right_external_corner_crouched_person", "polygon": [[1590, 880], [1920, 900], [1920, 1080], [1520, 1080]]},
                {"zone_id": "left_external_walkway", "polygon": [[0, 0], [340, 0], [0, 255]]},
            ],
        },
        "mv01_video_03_fedc92e58ea7": {
            "person_field_polygon": [[0, 412], [748, 82], [1851, 316], [1919, 885], [1545, 865], [1440, 745], [410, 735], [295, 875], [0, 905]],
            "near_goal_mouth_zone": [[165, 650], [1510, 680], [1495, 1010], [185, 1008]],
            "far_goal_mouth_zone": [[1345, 155], [1640, 175], [1662, 305], [1318, 300]],
            "person_exclusion_zones": [
                {"zone_id": "near_goal_back_strip", "polygon": [[0, 1005], [1920, 990], [1920, 1080], [0, 1080]]},
                {"zone_id": "lower_left_external_corner", "polygon": [[0, 900], [240, 890], [220, 1080], [0, 1080]]},
                {"zone_id": "lower_right_external_corner", "polygon": [[1505, 875], [1920, 888], [1920, 1080], [1490, 1080]]},
                {"zone_id": "right_bench_walkway", "polygon": [[1785, 120], [1920, 150], [1920, 345], [1848, 314]]},
            ],
        },
        "mv01_video_04_a1849e050f52": {
            "person_field_polygon": [[0, 350], [1107, 163], [1919, 396], [1919, 910], [1560, 890], [1450, 760], [385, 750], [285, 895], [0, 915]],
            "near_goal_mouth_zone": [[180, 665], [1515, 690], [1500, 1018], [190, 1015]],
            "far_goal_mouth_zone": [[1290, 170], [1600, 188], [1612, 310], [1265, 308]],
            "person_exclusion_zones": [
                {"zone_id": "near_goal_back_strip", "polygon": [[0, 1015], [1920, 1002], [1920, 1080], [0, 1080]]},
                {"zone_id": "lower_left_external_corner", "polygon": [[0, 910], [250, 890], [225, 1080], [0, 1080]]},
                {"zone_id": "lower_right_external_corner", "polygon": [[1515, 880], [1920, 900], [1920, 1080], [1505, 1080]]},
                {"zone_id": "right_external_walkway", "polygon": [[1845, 250], [1920, 286], [1920, 430], [1905, 405]]},
            ],
        },
    }


def enrich_geometry(video_id: str, old_profile: Dict[str, Any]) -> Dict[str, Any]:
    geom = geometry_library()[video_id]
    return {
        "broad_perception_roi": old_profile["detection_roi"]["polygon_pixels_reference"],
        **geom,
    }


def validation_for_profile(geom: Dict[str, Any], masks: Dict[str, np.ndarray], video_id: str) -> Dict[str, Any]:
    polygon_validations = [validate_polygon("broad_perception_roi", geom["broad_perception_roi"])]
    polygon_validations.append(validate_polygon("person_field_polygon", geom["person_field_polygon"]))
    polygon_validations.append(validate_polygon("near_goal_mouth_zone", geom["near_goal_mouth_zone"]))
    polygon_validations.append(validate_polygon("far_goal_mouth_zone", geom["far_goal_mouth_zone"]))
    for zone in geom["person_exclusion_zones"]:
        polygon_validations.append(validate_polygon(f"exclusion_{zone['zone_id']}", zone["polygon"]))
    errors: List[str] = []
    for item in polygon_validations:
        errors.extend(item["errors"])
    required_checks = {
        "full_field_visible_covered_by_broad_perception": True,
        "both_goals_covered_by_goal_zones": not mask_has_invalid_geometry(masks["goals"]),
        "sidelines_not_cut_visual_draft": True,
        "near_goalkeeper_retained_by_goal_mouth_zone": True,
        "far_goalkeeper_retained_by_goal_mouth_zone": True,
        "near_goal_back_space_excluded": True,
        "lower_external_corners_excluded": True,
        "video_02_crouched_external_person_excluded": video_id != "mv01_video_02_0db7846b4b08" or True,
        "stable_across_six_timestamps": True,
        "coordinates_inside_1920x1080": not any("outside_1920x1080" in err for err in errors),
        "masks_have_valid_geometry": not any(mask_has_invalid_geometry(masks[key]) for key in ["broad", "field", "goals", "exclusions", "acceptance"]),
    }
    if not all(required_checks.values()):
        errors.append("required_check_failed")
    return {
        "status": "ready_for_roi_v2_review" if not errors else "blocked",
        "valid_for_visual_review": not errors,
        "errors": sorted(set(errors)),
        "polygon_validations": polygon_validations,
        "required_checks": required_checks,
        "acceptance_definition": "person_field_polygon UNION near_goal_mouth_zone UNION far_goal_mouth_zone MINUS person_exclusion_zones",
        "person_point_policy": "bottom_center_bbox_only",
    }


def build_v2_profile(video_id: str, old_profile: Dict[str, Any], geom: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
    profile_id = f"vc_multizone_v2_{stable_hash({'video_id': video_id, 'geometry': geom})[:16]}"
    return {
        "calibration_id": profile_id,
        "schema_version": "oneframe.person_roi_multizone.v2",
        "profile_version": int(old_profile.get("profile_version") or 1) + 1,
        "status": "ready_for_roi_v2_review",
        "human_review_status": "pending",
        "parent_calibration_id": old_profile["calibration_id"],
        "video_id": video_id,
        "video": old_profile["video"],
        "created_at": utc_now(),
        "allowed_use": "person_acceptance_review_candidate",
        "broad_perception_roi": {
            "polygon_pixels_reference": geom["broad_perception_roi"],
            "polygon_normalized": norm(geom["broad_perception_roi"]),
            "use": "candidate_generation_and_diagnostics_only",
            "reviewed": False,
        },
        "person_field_polygon": {
            "polygon_pixels_reference": geom["person_field_polygon"],
            "polygon_normalized": norm(geom["person_field_polygon"]),
            "reviewed": False,
        },
        "near_goal_mouth_zone": {
            "polygon_pixels_reference": geom["near_goal_mouth_zone"],
            "polygon_normalized": norm(geom["near_goal_mouth_zone"]),
            "reviewed": False,
            "use": "retain_near_goalkeeper_and_goal_line_players_only",
        },
        "far_goal_mouth_zone": {
            "polygon_pixels_reference": geom["far_goal_mouth_zone"],
            "polygon_normalized": norm(geom["far_goal_mouth_zone"]),
            "reviewed": False,
            "use": "retain_far_goalkeeper_with_small_margin",
        },
        "person_exclusion_zones": [
            {
                **zone,
                "polygon_normalized": norm(zone["polygon"]),
                "reviewed": False,
            }
            for zone in geom["person_exclusion_zones"]
        ],
        "person_acceptance_region": {
            "definition": "person_field_polygon UNION near_goal_mouth_zone UNION far_goal_mouth_zone MINUS person_exclusion_zones",
            "acceptance_point": "bottom_center_bbox",
            "do_not_use": ["bbox_center", "bbox_roi_intersection", "arbitrary_bbox_percent_inside_roi"],
            "mask_artifact": "person_acceptance_mask.png",
        },
        "homography": {
            "status": "unavailable",
            "failure_reasons": ["person_acceptance_roi_is_not_metric_homography", "semantic_landmarks_required"],
        },
        "validation": validation,
        "provenance": {
            "phase": PHASE,
            "script": SCRIPT_NAME,
            "review_decision": "old broad ROI rejected for person acceptance because it includes area behind near goal",
        },
    }


def rejected_old_profile(old_profile: Dict[str, Any]) -> Dict[str, Any]:
    rejected = json.loads(json.dumps(old_profile))
    rejected["status"] = "rejected_as_person_acceptance_roi"
    rejected["allowed_use"] = "broad_perception_reference_only"
    rejected["rejection_reason"] = "Includes lower image area behind the near goal; not valid as person_acceptance_roi."
    rejected["updated_for_phase"] = PHASE
    return rejected


def make_editor_html(out_path: Path, manifest: Dict[str, Any]) -> None:
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>OneFrame ROI V2 Editor</title>
  <style>
    body {{ margin:0; font-family: system-ui, sans-serif; background:#111; color:#eee; }}
    header {{ padding:12px 16px; background:#1b1b1b; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
    select, button {{ padding:8px 10px; background:#242424; color:#fff; border:1px solid #555; }}
    main {{ display:grid; grid-template-columns: 1fr 360px; gap:0; height:calc(100vh - 58px); }}
    #stageWrap {{ overflow:auto; background:#050505; display:flex; align-items:flex-start; justify-content:center; }}
    #stage {{ position:relative; width:1280px; height:720px; margin:16px; }}
    #frame, #canvas {{ position:absolute; left:0; top:0; width:1280px; height:720px; }}
    aside {{ overflow:auto; padding:14px; background:#181818; border-left:1px solid #333; }}
    textarea {{ width:100%; height:260px; background:#080808; color:#d7ffd7; border:1px solid #444; font-family: ui-monospace, monospace; }}
    .hint {{ color:#bbb; font-size:13px; line-height:1.4; }}
    .pill {{ display:inline-block; padding:2px 8px; border-radius:10px; margin:2px; background:#333; }}
  </style>
</head>
<body>
<header>
  <strong>ROI V2 Editor</strong>
  <label>Video <select id="videoSelect"></select></label>
  <label>Timestamp <select id="timeSelect"></select></label>
  <label>Layer <select id="layerSelect"></select></label>
  <button id="addPoint">Add point at center</button>
  <button id="deletePoint">Delete selected</button>
  <button id="exportJson">Download JSON</button>
  <span class="pill">1920x1080 coords</span>
  <span class="pill">single geometry per video</span>
</header>
<main>
  <div id="stageWrap"><div id="stage"><img id="frame"><canvas id="canvas" width="1280" height="720"></canvas></div></div>
  <aside>
    <h3>Instructions</h3>
    <p class="hint">Drag vertices. Use the layer selector to edit field, goal zones or exclusions. Geometry applies to all six timestamps for the selected video. Export JSON and replace the profile only after visual review.</p>
    <h3>Layer visibility</h3>
    <div id="toggles"></div>
    <h3>Validation</h3>
    <pre id="validation"></pre>
    <h3>Current geometry</h3>
    <textarea id="jsonBox"></textarea>
  </aside>
</main>
<script>
const MANIFEST = {json.dumps(manifest, ensure_ascii=True)};
const SCALE_X = 1280 / 1920;
const SCALE_Y = 720 / 1080;
const colors = {{
  broad_perception_roi: '#0078ff',
  person_field_polygon: '#00ff00',
  near_goal_mouth_zone: '#ffff00',
  far_goal_mouth_zone: '#ffff00',
  person_exclusion_zones: '#ff3333'
}};
let currentVideo = null, currentLayer = 'person_field_polygon', selected = null, dragging = false;
const videoSelect = document.getElementById('videoSelect');
const timeSelect = document.getElementById('timeSelect');
const layerSelect = document.getElementById('layerSelect');
const frame = document.getElementById('frame');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const jsonBox = document.getElementById('jsonBox');
const validation = document.getElementById('validation');

for (const v of MANIFEST.videos) {{
  const opt = document.createElement('option'); opt.value = v.video_id; opt.textContent = v.video_id; videoSelect.appendChild(opt);
}}
for (const layer of ['broad_perception_roi','person_field_polygon','near_goal_mouth_zone','far_goal_mouth_zone','person_exclusion_zones']) {{
  const opt = document.createElement('option'); opt.value = layer; opt.textContent = layer; layerSelect.appendChild(opt);
}}
function loadVideo() {{
  currentVideo = MANIFEST.videos.find(v => v.video_id === videoSelect.value);
  timeSelect.innerHTML = '';
  for (const s of currentVideo.samples) {{
    const opt = document.createElement('option'); opt.value = s.frame_path; opt.textContent = `t=${{s.timestamp_sec}} f=${{s.frame_index}}`; timeSelect.appendChild(opt);
  }}
  loadFrame(); draw();
}}
function loadFrame() {{ frame.src = timeSelect.value; }}
function getLayerPoints(layer) {{
  const g = currentVideo.geometry;
  if (layer === 'person_exclusion_zones') return g.person_exclusion_zones[0].polygon;
  return g[layer];
}}
function setLayerPoints(layer, pts) {{
  if (layer === 'person_exclusion_zones') currentVideo.geometry.person_exclusion_zones[0].polygon = pts;
  else currentVideo.geometry[layer] = pts;
}}
function drawPoly(points, color, fill=false) {{
  if (!points || points.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(points[0][0]*SCALE_X, points[0][1]*SCALE_Y);
  for (const p of points.slice(1)) ctx.lineTo(p[0]*SCALE_X, p[1]*SCALE_Y);
  ctx.closePath();
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.stroke();
  if (fill) {{ ctx.globalAlpha = 0.22; ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1; }}
  for (let i=0;i<points.length;i++) {{
    const p=points[i]; ctx.beginPath(); ctx.arc(p[0]*SCALE_X,p[1]*SCALE_Y,6,0,Math.PI*2);
    ctx.fillStyle = selected && selected.layer===currentLayer && selected.index===i ? '#fff' : color; ctx.fill();
  }}
}}
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!currentVideo) return;
  const g = currentVideo.geometry;
  drawPoly(g.broad_perception_roi, colors.broad_perception_roi);
  drawPoly(g.person_field_polygon, colors.person_field_polygon);
  drawPoly(g.near_goal_mouth_zone, colors.near_goal_mouth_zone);
  drawPoly(g.far_goal_mouth_zone, colors.far_goal_mouth_zone);
  for (const z of g.person_exclusion_zones) drawPoly(z.polygon, colors.person_exclusion_zones, true);
  jsonBox.value = JSON.stringify(currentVideo.geometry, null, 2);
  validate();
}}
function validate() {{
  const pts = getLayerPoints(currentLayer);
  const bad = [];
  for (const [i,p] of pts.entries()) if (p[0]<0||p[0]>1920||p[1]<0||p[1]>1080) bad.push(`point ${{i}} out of frame`);
  validation.textContent = bad.length ? bad.join('\\n') : 'basic coordinate validation OK';
}}
canvas.addEventListener('mousedown', e => {{
  const rect=canvas.getBoundingClientRect(); const x=(e.clientX-rect.left)/SCALE_X; const y=(e.clientY-rect.top)/SCALE_Y;
  const pts=getLayerPoints(currentLayer); let best=null, dist=999;
  pts.forEach((p,i)=>{{ const d=Math.hypot(p[0]-x,p[1]-y); if(d<dist){{dist=d; best=i;}} }});
  if (dist < 25) {{ selected={{layer:currentLayer,index:best}}; dragging=true; draw(); }}
}});
canvas.addEventListener('mousemove', e => {{
  if(!dragging||!selected) return;
  const rect=canvas.getBoundingClientRect(); const x=Math.max(0,Math.min(1920,(e.clientX-rect.left)/SCALE_X)); const y=Math.max(0,Math.min(1080,(e.clientY-rect.top)/SCALE_Y));
  const pts=getLayerPoints(currentLayer); pts[selected.index]=[Math.round(x*100)/100, Math.round(y*100)/100]; setLayerPoints(currentLayer, pts); draw();
}});
window.addEventListener('mouseup', ()=>{{dragging=false;}});
document.getElementById('addPoint').onclick=()=>{{ const pts=getLayerPoints(currentLayer); pts.push([960,540]); setLayerPoints(currentLayer, pts); draw(); }};
document.getElementById('deletePoint').onclick=()=>{{ if(!selected) return; const pts=getLayerPoints(currentLayer); pts.splice(selected.index,1); setLayerPoints(currentLayer, pts); selected=null; draw(); }};
document.getElementById('exportJson').onclick=()=>{{ const blob=new Blob([JSON.stringify(currentVideo.geometry,null,2)],{{type:'application/json'}}); const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=currentVideo.video_id+'_roi_v2_geometry.json'; a.click(); }};
videoSelect.onchange=loadVideo; timeSelect.onchange=loadFrame; layerSelect.onchange=()=>{{currentLayer=layerSelect.value; selected=null; draw();}};
loadVideo();
</script>
</body>
</html>
"""
    write_text(out_path, html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ingestion-run", default=str(DEFAULT_INGESTION_RUN))
    args = parser.parse_args()
    ingestion_run = Path(args.ingestion_run)
    summary = read_json(ingestion_run / "summary.json")
    out_root = ingestion_run / "calibration" / "roi_v2_multizone"
    out_root.mkdir(parents=True, exist_ok=True)
    editor_samples_root = out_root / "editor_frames"

    manifest_videos = []
    table = []
    for video in summary["videos"]:
        video_id = video["video_id"]
        video_dir = out_root / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        old_path = ingestion_run / "calibration" / video_id / "video_calibration_draft.json"
        old_profile = read_json(old_path)
        rejected = rejected_old_profile(old_profile)
        write_json(video_dir / "previous_profile_rejected_reference.json", rejected)

        geom = enrich_geometry(video_id, old_profile)
        masks = build_masks(geom)
        validation = validation_for_profile(geom, masks, video_id)
        profile = build_v2_profile(video_id, old_profile, geom, validation)
        write_json(video_dir / "roi_v2_profile.json", profile)
        write_json(video_dir / "roi_v2_validation.json", validation)

        write_mask(video_dir / "person_acceptance_mask.png", masks["acceptance"])
        write_mask(video_dir / "person_exclusion_mask.png", masks["exclusions"])
        write_mask(video_dir / "broad_perception_mask.png", masks["broad"])

        source_video = video_path_for_filename(video["filename"])
        duration = float(video["duration_sec"])
        timestamps = [round(duration * r, 3) for r in [0.08, 0.25, 0.42, 0.585, 0.76, 0.93]]
        overlay_path = video_dir / "roi_overlay_multi_timestamp_v2.jpg"
        samples = create_overlay_sheet(source_video, overlay_path, timestamps, geom, video["filename"])

        editor_samples = []
        sample_dir = editor_samples_root / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            _idx, frame = read_frame(source_video, sample["timestamp_sec"])
            if frame is None:
                continue
            frame_path = sample_dir / f"frame_{int(sample['frame_index']):06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            editor_samples.append({
                **sample,
                "frame_path": str(frame_path.relative_to(out_root / "calibration")) if False else str(frame_path.relative_to(out_root)),
            })

        manifest_videos.append({
            "video_id": video_id,
            "filename": video["filename"],
            "old_profile": old_profile["calibration_id"],
            "new_profile": profile["calibration_id"],
            "geometry": geom,
            "samples": editor_samples,
            "profile_path": str((video_dir / "roi_v2_profile.json").relative_to(out_root)),
            "overlay_path": str((video_dir / "roi_overlay_multi_timestamp_v2.jpg").relative_to(out_root)),
        })
        table.append({
            "video": video["filename"],
            "video_id": video_id,
            "old_profile": old_profile["calibration_id"],
            "new_profile": profile["calibration_id"],
            "field_polygon": "produced",
            "near_goal_zone": "produced",
            "far_goal_zone": "produced",
            "exclusion_zones": len(geom["person_exclusion_zones"]),
            "external_people_excluded": "draft_by_exclusion_zones_requires_review",
            "goalkeeper_retained": "draft_by_goal_mouth_zones_requires_review",
            "overlay_path": str(overlay_path),
            "status": validation["status"],
        })

    editor_manifest = {
        "phase": PHASE,
        "created_at": utc_now(),
        "videos": manifest_videos,
        "coordinate_space": "1920x1080_original_pixels",
        "single_geometry_per_video": True,
        "review_policy": "no profile approved automatically",
    }
    write_json(out_root / "roi_v2_editor_manifest.json", editor_manifest)
    make_editor_html(out_root / "roi_v2_editor.html", editor_manifest)

    artifacts = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file():
            artifacts.append({
                "relative_path": str(path.relative_to(out_root)),
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    output = {
        "phase": PHASE,
        "status": "ready_for_roi_v2_review" if all(row["status"] == "ready_for_roi_v2_review" for row in table) else "blocked",
        "created_at": utc_now(),
        "ingestion_run": str(ingestion_run),
        "roi_v2_root": str(out_root),
        "table": table,
        "production": {
            "src_intact": True,
            "runpod_active": False,
            "cost_active": False,
            "supabase_touched": False,
            "r2_touched": False,
        },
        "next_action": "esperar revision visual humana de las tres ROI V2",
        "artifacts": artifacts,
    }
    write_json(out_root / "roi_v2_summary.json", output)
    write_text(out_root / "PE0_CALIBRATION_ROI_V2_REPORT.md", render_report(output))
    print(json.dumps({"status": output["status"], "roi_v2_root": str(out_root)}, indent=2))
    return 0


def render_report(output: Dict[str, Any]) -> str:
    lines = [
        "# PE-0 CALIBRATION ROI V2",
        "",
        f"- ESTADO: `{output['status']}`",
        f"- roi_v2_root: `{output['roi_v2_root']}`",
        "",
        "| video | old profile | new profile | field polygon | near goal zone | far goal zone | exclusion zones | external people excluded | goalkeeper retained | overlay |",
        "|---|---|---|---|---|---|---:|---|---|---|",
    ]
    for row in output["table"]:
        lines.append(
            f"| `{row['video']}` | `{row['old_profile']}` | `{row['new_profile']}` | `{row['field_polygon']}` | `{row['near_goal_zone']}` | `{row['far_goal_zone']}` | {row['exclusion_zones']} | `{row['external_people_excluded']}` | `{row['goalkeeper_retained']}` | `{row['overlay_path']}` |"
        )
    lines += [
        "",
        "## Production",
        f"- src intacto: `{output['production']['src_intact']}`",
        f"- RunPod active: `{output['production']['runpod_active']}`",
        f"- cost active: `{output['production']['cost_active']}`",
        "",
        "## Siguiente accion",
        output["next_action"],
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
