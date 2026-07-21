#!/usr/bin/env python3
"""Extract DF-0C SAM pilot clips and build a transfer package."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DF = ROOT / "ai_worker_v1" / "data_factory"
CLIPS_MANIFEST = DF / "sam3_worker" / "sam3_pilot_clips.json"
VIDEO_DIR = ROOT / "ai_worker_v1" / "input_videos" / "multivideo_v01"
PACKAGE_DIR = DF / "sam3_worker" / "pilot_package"
CLIP_DIR = PACKAGE_DIR / "clips"
ROI_DIR = PACKAGE_DIR / "roi_profiles"

VIDEO_BY_ID = {
    "mv01_video_02_0db7846b4b08": VIDEO_DIR / "video_02.mp4",
    "mv01_video_03_fedc92e58ea7": VIDEO_DIR / "video_03.mp4",
    "mv01_video_04_a1849e050f52": VIDEO_DIR / "video_04.mp4",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        return json.loads(path.read_text())
    return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def extract_clip(src: Path, dst: Path, start: float, duration: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "fps=15",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    clips = read_json(CLIPS_MANIFEST, [])
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    packaged = []
    for clip in clips:
        video_id = clip["video_id"]
        src = VIDEO_BY_ID[video_id]
        if not src.exists():
            raise FileNotFoundError(src)
        dst = CLIP_DIR / f"{clip['clip_id']}.mp4"
        extract_clip(src, dst, float(clip["start_sec"]), float(clip["duration_sec"]))
        roi_src = Path(clip["roi_profile"])
        roi_dst = ROI_DIR / f"{video_id}_roi_manual_v3_1_profile.json"
        shutil.copy2(roi_src, roi_dst)
        row = dict(clip)
        row.update(
            {
                "local_clip_path": rel(dst),
                "local_clip_sha256": sha256_file(dst),
                "local_clip_size": dst.stat().st_size,
                "local_roi_path": rel(roi_dst),
                "local_roi_sha256": sha256_file(roi_dst),
                "source_video_sha256": sha256_file(src),
            }
        )
        packaged.append(row)
    manifest = {
        "schema_version": "df0c.sam3_pilot_package.v0",
        "created_at": now_iso(),
        "package_dir": rel(PACKAGE_DIR),
        "clip_count": len(packaged),
        "clips": packaged,
        "checkpoint_dir": rel(DF / "sam3_worker" / "checkpoints" / "sam3_1"),
        "checkpoint_manifest": rel(
            DF / "sam3_worker" / "checkpoints" / "sam3_1" / "checkpoint_manifest.json"
        ),
        "token_included": False,
    }
    write_json(PACKAGE_DIR / "pilot_package_manifest.json", manifest)
    print(json.dumps({"clip_count": len(packaged), "package_dir": rel(PACKAGE_DIR)}, indent=2))


if __name__ == "__main__":
    main()
