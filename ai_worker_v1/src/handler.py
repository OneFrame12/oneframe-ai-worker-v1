"""
OneFrame V57 — Handler para RunPod Serverless
Punto de entrada principal. Conecta el pipeline completo.
"""

import base64
import csv
import glob
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import quote, unquote, urlparse

sys.path.insert(0, os.path.dirname(__file__))

import boto3
import gdown
import numpy as np
import requests
import ultralytics

try:
    from supabase import create_client, Client
except ImportError:  # pragma: no cover
    create_client = None
    Client = Any

from config import VisionConfig
from engine import VisionEngine, GameReferee, AudioPreScanner, BallDetector, Detection
from detectors import RFDETRDetectorAdapter, YOLODetectorAdapter
from artifacts import (
    REQUIRED_ARTIFACT_NAMES,
    artifact_manifest,
    build_required_artifacts,
    create_produced_artifact,
    utc_now_iso,
)
from learner import EvolutiveLearner

print(f"Ultralytics version: {ultralytics.__version__}")

SHADOW_WORKER_TYPE = os.getenv("AI_WORKER_V1_WORKER_TYPE", "ai_worker_v1")
SEGMENT_WORKER_TYPE = "ai_worker_v1_rfdetr_primary_shadow_segment"
SCHEMA_VERSION = "oneframe.schemas.v1"
ARCHITECTURE_VERSION = "oneframe.architecture_controlled.v1"
ARTIFACT_NOT_IMPLEMENTED_REASON = "not_implemented_in_current_phase"


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("OneFrame.Handler")

MIN_VALID_VIDEO_SIZE_BYTES = 500 * 1024
DEFAULT_FRAME_RATIO = 0.5
REQUEST_TIMEOUT = (10, 120)
HTML_PREVIEW_BYTES = 4096
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


class DownloadError(RuntimeError):
    def __init__(self, message: str, error_code: str = "VIDEO_DOWNLOAD_FAILED"):
        super().__init__(message)
        self.error_code = error_code


class CriticalJobError(RuntimeError):
    """Error que debe marcar el job de RunPod como FAILED."""


def ensure_rfdetr_base_checkpoint() -> str:
    """Materialize the pinned RF-DETR artifact from configured R2."""
    import hashlib
    import tempfile

    target = "/app/rfdetr_cache/rf-detr-base.pth"
    expected = os.getenv("RFDETR_BASE_SHA256")
    object_key = os.getenv("RFDETR_BASE_OBJECT_KEY")
    if not expected or not object_key:
        raise CriticalJobError("RFDETR_BASE_OBJECT_KEY and RFDETR_BASE_SHA256 are required")

    def digest(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    if os.path.isfile(target) and digest(target) == expected:
        return target
    cloud = CloudManager()
    client = cloud.r2_read_client
    if client is None:
        raise CriticalJobError("R2 read client unavailable for RF-DETR artifact")
    bucket = cloud.r2_read_bucket
    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="rf-detr-", suffix=".pth", dir="/tmp")
    os.close(fd)
    try:
        client.download_file(bucket, object_key, tmp)
        if digest(tmp) != expected:
            raise CriticalJobError("RF-DETR artifact SHA256 mismatch")
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return target


def normalize_video_url(url: str) -> str:
    """
    Convierte cualquier formato de URL de Google Drive al formato de descarga directa.

    URLs que no son de Drive se retornan sin cambio.
    """
    if not url or "drive.google.com" not in url:
        return url

    patterns = [
        r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
        r"drive\.google\.com/uc\?.*?id=([a-zA-Z0-9_-]+)",
        r"docs\.google\.com/[^/]+/d/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            file_id = match.group(1)
            normalized = (
                f"https://drive.google.com/uc"
                f"?export=download&id={file_id}&confirm=t"
            )
            if normalized != url:
                logger.info("🔗 URL normalizada: %s → %s", url[:60], normalized)
            return normalized

    return url


def _extract_google_drive_file_id(url: str) -> Optional[str]:
    if not url or "drive.google.com" not in url:
        return None

    match = re.search(r"(?:id=|/d/)([a-zA-Z0-9_-]{25,})", url)
    return match.group(1) if match else None


def _cleanup_partial_download(output_path: str) -> None:
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            pass


def _content_type_is_html(content_type: str) -> bool:
    lowered = (content_type or "").lower()
    return "text/html" in lowered or "application/xhtml+xml" in lowered


def _bytes_look_like_html(content: bytes) -> bool:
    if not content:
        return False

    prefix = content[:HTML_PREVIEW_BYTES].lstrip().lower()
    return (
        prefix.startswith(b"<!doctype html")
        or prefix.startswith(b"<html")
        or prefix.startswith(b"<?xml")
    )


def _ensure_response_contains_video(
    response: requests.Response,
    preview_bytes: bytes,
    source: str,
) -> None:
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    logger.info(
        "📄 %s response status=%s content_type=%s",
        source,
        response.status_code,
        content_type or "unknown",
    )

    if _content_type_is_html(content_type):
        raise RuntimeError(
            f"{source} returned HTML instead of a video file "
            f"(status={response.status_code}, content_type={content_type})."
        )

    if _bytes_look_like_html(preview_bytes):
        raise RuntimeError(
            f"{source} returned HTML content instead of video bytes "
            f"(status={response.status_code}, content_type={content_type or 'unknown'})."
        )


def _download_via_requests(url: str, output_path: str) -> str:
    logger.info(f"🌍 Starting HTTP video download: url={url} output_path={output_path}")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 OneFrame/1.0"})
        response = session.get(url, stream=True, allow_redirects=True, timeout=REQUEST_TIMEOUT)
        preview_bytes = b""
        total_bytes = 0

        try:
            with open(output_path, "wb") as output_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    if not preview_bytes:
                        preview_bytes = chunk[:HTML_PREVIEW_BYTES]
                    output_file.write(chunk)
                    total_bytes += len(chunk)
        except Exception:
            _cleanup_partial_download(output_path)
            raise
        finally:
            response.close()

    try:
        _ensure_response_contains_video(response, preview_bytes, "HTTP download")
    except Exception:
        _cleanup_partial_download(output_path)
        raise

    if total_bytes <= 0 or not os.path.exists(output_path):
        _cleanup_partial_download(output_path)
        raise RuntimeError("Requests fallback created no video file.")

    if os.path.getsize(output_path) <= 0:
        _cleanup_partial_download(output_path)
        raise RuntimeError("HTTP download created an empty file.")

    logger.info(f"✅ HTTP download completed: {output_path} ({total_bytes} bytes)")
    return output_path


def _extract_extension_from_url(url: str) -> str:
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    extension = os.path.splitext(path)[1].lower()
    if extension and 1 < len(extension) <= 8:
        return extension
    return ".mp4"


def _build_download_path(url: str, output_path: str) -> str:
    root, existing_extension = os.path.splitext(output_path)
    base_path = root if existing_extension else output_path
    return f"{base_path}{_extract_extension_from_url(url)}"


def download_video(url: str, output_path: str) -> str:
    drive_file_id = _extract_google_drive_file_id(url)
    if drive_file_id:
        final_output_path = output_path
        logger.info(f"📥 Descargando con gdown: file_id={drive_file_id}")
        logger.info(f"📁 Local output path resolved to: {final_output_path}")

        _cleanup_partial_download(final_output_path)
        try:
            logger.info(f"⚙️ gdown version: {gdown.__version__}")
            logger.info(f"⚙️ llamando gdown.download(id={drive_file_id})")
            downloaded_path = gdown.download(
                id=drive_file_id,
                output=final_output_path,
                quiet=False,
            )
            if downloaded_path and os.path.exists(final_output_path):
                size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
                logger.info(f"✅ gdown OK: {size_mb:.1f} MB")
                return final_output_path
            logger.warning("⚠️ gdown no creó archivo, intentando con requests...")
        except Exception as exc:
            logger.warning(f"⚠️ gdown falló: {exc}")
            logger.warning(f"⚠️ gdown traceback: {traceback.format_exc()}")
            logger.warning("⚠️ intentando con requests...")

        direct_url = f"https://drive.google.com/uc?export=download&id={drive_file_id}&confirm=t"
        _cleanup_partial_download(final_output_path)
        _download_via_requests(direct_url, final_output_path)
        return final_output_path
    else:
        url = normalize_video_url(url)
        final_output_path = _build_download_path(url, output_path)
        logger.info(f"📥 Downloading remote video: url={url}")
        logger.info(f"📁 Local output path resolved to: {final_output_path}")

        _cleanup_partial_download(final_output_path)
        _download_via_requests(url, final_output_path)

    size_bytes = os.path.getsize(final_output_path)
    size_mb = size_bytes / (1024 * 1024)
    logger.info(f"✅ Video descargado: {size_mb:.1f} MB")

    if size_bytes < MIN_VALID_VIDEO_SIZE_BYTES:
        _cleanup_partial_download(final_output_path)
        raise RuntimeError("Downloaded file is too small, likely invalid")

    return final_output_path


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def normalize_frame_ratio(raw_ratio: Any) -> float:
    try:
        ratio = float(raw_ratio)
    except (TypeError, ValueError):
        ratio = DEFAULT_FRAME_RATIO
    return min(max(ratio, 0.05), 0.95)


def extract_frame_base64(video_path: str, frame_ratio: float = DEFAULT_FRAME_RATIO) -> Dict[str, Any]:
    duration = get_video_duration(video_path)
    frame_ratio = normalize_frame_ratio(frame_ratio)
    timestamp = duration * frame_ratio

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "2",
        "-loglevel",
        "error",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)

    if not result.stdout:
        raise RuntimeError("ffmpeg no pudo extraer el frame de calibracion.")

    image_b64 = base64.b64encode(result.stdout).decode("ascii")
    return {
        "status": "success",
        "action": "get_frame",
        "frame_base64": image_b64,
        "mime_type": "image/jpeg",
        "timestamp_sec": round(timestamp, 3),
        "duration_sec": round(duration, 3),
        "frame_ratio": frame_ratio,
    }


def handle_frame_extraction(link_video: str, video_path: str, frame_ratio: float) -> Dict[str, Any]:
    local_video_path = download_video(link_video, video_path)
    return extract_frame_base64(local_video_path, frame_ratio=frame_ratio)


class CloudManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key_name = ""
        self.supabase_key = ""
        for key_name in (
            "AI_WORKER_V1_SUPABASE_ANON_KEY",
            "SUPABASE_ANON_KEY",
            "NEXT_PUBLIC_SUPABASE_ANON_KEY",
        ):
            value = os.getenv(key_name)
            if value:
                self.supabase_key_name = key_name
                self.supabase_key = value
                break
        self.supabase_ready = bool(
            self.supabase_url and self.supabase_key and create_client is not None
        )
        self.client: Optional[Client] = None

        self.r2_endpoint = os.getenv("AI_WORKER_V1_R2_ENDPOINT") or os.getenv("R2_ENDPOINT")
        self.r2_read_bucket = os.getenv("AI_WORKER_V1_R2_INPUT_BUCKET", "one-frame")
        self.r2_write_bucket = os.getenv("AI_WORKER_V1_R2_OUTPUT_BUCKET", "one-frame-shadow")
        self.r2_bucket = self.r2_write_bucket
        self.r2_read_access_key_id = os.getenv("AI_WORKER_V1_R2_READ_ACCESS_KEY_ID")
        self.r2_read_secret_access_key = os.getenv("AI_WORKER_V1_R2_READ_SECRET_ACCESS_KEY")
        self.r2_write_access_key_id = os.getenv("AI_WORKER_V1_R2_WRITE_ACCESS_KEY_ID")
        self.r2_write_secret_access_key = os.getenv("AI_WORKER_V1_R2_WRITE_SECRET_ACCESS_KEY")
        self.r2_public_base_url = os.getenv("R2_PUBLIC_BASE_URL")
        self.r2_read_ready = bool(
            self.r2_read_bucket
            and self.r2_endpoint
            and self.r2_read_access_key_id
            and self.r2_read_secret_access_key
        )
        self.r2_write_ready = bool(
            self.r2_write_bucket
            and self.r2_endpoint
            and self.r2_write_access_key_id
            and self.r2_write_secret_access_key
        )
        self.r2_ready = self.r2_write_ready
        self.r2_read_client = None
        self.r2_write_client = None
        self.r2_client = None

        if self.supabase_ready:
            self.client = create_client(self.supabase_url, self.supabase_key)
            logger.info("☁️ Supabase listo para metadata de clips.")
        else:
            logger.info("ℹ️ Supabase no configurado o librería ausente; se omite inserción de metadata.")

        if self.r2_read_ready:
            self.r2_read_client = boto3.client(
                "s3",
                endpoint_url=self.r2_endpoint,
                aws_access_key_id=self.r2_read_access_key_id,
                aws_secret_access_key=self.r2_read_secret_access_key,
                region_name="auto",
            )
            logger.info("☁️ R2 lectura listo para bucket input.")
        else:
            logger.info("ℹ️ R2 lectura no configurado; se omiten descargas desde bucket input.")

        if self.r2_write_ready:
            self.r2_write_client = boto3.client(
                "s3",
                endpoint_url=self.r2_endpoint,
                aws_access_key_id=self.r2_write_access_key_id,
                aws_secret_access_key=self.r2_write_secret_access_key,
                region_name="auto",
            )
            self.r2_client = self.r2_write_client
            logger.info("☁️ R2 escritura listo para bucket shadow.")
        else:
            logger.info("ℹ️ R2 escritura no configurado; se omiten outputs shadow.")

    def require_supabase_metrics_ready(self) -> None:
        if not self.supabase_url:
            raise CriticalJobError("SUPABASE_URL no configurada para ai_worker_v1.")
        if not self.supabase_key:
            raise CriticalJobError(
                "Falta key de Supabase para ai_worker_v1 "
                "(AI_WORKER_V1_SUPABASE_ANON_KEY, SUPABASE_ANON_KEY o NEXT_PUBLIC_SUPABASE_ANON_KEY)."
            )
        if create_client is None:
            raise CriticalJobError("Libreria supabase no disponible en la imagen ai_worker_v1.")
        if self.client is None:
            raise CriticalJobError("Cliente Supabase no inicializado para ai_worker_v1.")

        try:
            response = (
                self.client.table("match_processing_metrics")
                .select("id")
                .limit(1)
                .execute()
            )
        except Exception as exc:
            raise CriticalJobError(
                f"Preflight Supabase fallo consultando match_processing_metrics: {exc}"
            ) from exc

        if not hasattr(response, "data"):
            raise CriticalJobError(
                "Preflight Supabase no devolvio una respuesta valida para match_processing_metrics."
            )
        logger.info(
            "🧪 Supabase metrics preflight OK usando %s.",
            self.supabase_key_name,
        )

    def build_public_url(self, object_key: str) -> str:
        if not self.r2_public_base_url:
            return ""

        normalized_base = str(self.r2_public_base_url or "").strip().strip('"').strip("'").rstrip("/")
        encoded_key = "/".join(
            quote(segment, safe="") for segment in object_key.split("/") if segment
        )
        return f"{normalized_base}/{encoded_key}"

    def build_signed_url(self, object_key: str, expires_in: int = 86400) -> Optional[str]:
        if not self.r2_client:
            logger.warning("⚠️ R2 no está listo; no se puede firmar %s", object_key)
            return None

        try:
            return self.r2_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={
                    "Bucket": self.r2_write_bucket,
                    "Key": object_key,
                },
                ExpiresIn=expires_in,
            )
        except Exception as exc:
            logger.warning("⚠️ No se pudo generar signed URL R2 (%s): %s", object_key, exc)
            return None

    def persist_clip(
        self,
        local_path: str,
        storage_path: str,
        clip_metadata: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not self.r2_client:
            logger.warning("⚠️ R2 no está listo; no se puede persistir el clip %s", storage_path)
            return None

        try:
            content_type = mimetypes.guess_type(local_path)[0] or "video/mp4"
            with open(local_path, "rb") as file_stream:
                self.r2_client.upload_fileobj(
                    Fileobj=file_stream,
                    Bucket=self.r2_write_bucket,
                    Key=storage_path,
                    ExtraArgs={"ContentType": content_type},
                )
        except Exception as exc:
            logger.warning(f"⚠️ Subida de clip a R2 falló ({storage_path}): {exc}")
            return None

        public_url = self.build_public_url(storage_path)
        if not public_url:
            public_url = self.build_signed_url(storage_path) or ""
        persisted_metadata = {
            **clip_metadata,
            "video_url": public_url,
            "public_url": public_url,
        }

        logger.info("🧪 Shadow mode: omitiendo inserción en tabla clips.")
        return persisted_metadata

    def upload_file(self, local_path: str, storage_path: str) -> Optional[str]:
        if not self.r2_client:
            logger.warning("⚠️ R2 no está listo; no se puede subir %s", storage_path)
            return None
        if not local_path or not os.path.exists(local_path):
            logger.warning("⚠️ Archivo no existe para subir: %s", local_path)
            return None

        try:
            content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
            with open(local_path, "rb") as file_stream:
                self.r2_client.upload_fileobj(
                    Fileobj=file_stream,
                    Bucket=self.r2_write_bucket,
                    Key=storage_path,
                    ExtraArgs={"ContentType": content_type},
                )
        except Exception as exc:
            logger.warning("⚠️ Subida a R2 falló (%s): %s", storage_path, exc)
            return None

        return self.build_public_url(storage_path)

    def persist_training_frame(self, frame_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        local_path = frame_metadata.get("local_path")
        object_key = frame_metadata.get("object_key") or frame_metadata.get("frame_path")
        if not local_path or not object_key:
            return None
        if not self.r2_client:
            logger.warning("⚠️ R2 no está listo; no se puede persistir frame training %s", object_key)
            return None
        if not os.path.exists(local_path):
            logger.warning("⚠️ Frame training no existe: %s", local_path)
            return None

        try:
            with open(local_path, "rb") as file_stream:
                self.r2_client.upload_fileobj(
                    Fileobj=file_stream,
                    Bucket=self.r2_write_bucket,
                    Key=object_key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
        except Exception as exc:
            logger.warning("⚠️ Subida de frame training a R2 falló (%s): %s", object_key, exc)
            return None

        public_url = self.build_public_url(object_key)
        persisted = {
            "match_id": frame_metadata.get("match_id"),
            "frame_path": object_key,
            "label": frame_metadata.get("label", "low_confidence"),
            "confidence": float(frame_metadata.get("confidence", 0.0) or 0.0),
            "frame_number": int(frame_metadata.get("frame_number", 0) or 0),
            "exported": False,
            "annotated": False,
            "public_url": public_url,
        }

        logger.info("🧪 Shadow mode: omitiendo inserción en tabla training_frames.")
        return persisted


def _find_yolo_model(preferred_model: Optional[str] = None) -> str:
    search_dirs = [
        os.path.dirname(__file__),
        os.path.dirname(os.path.dirname(__file__)),
        "/app",
        "/tmp",
        os.path.expanduser("~"),
    ]
    preferred_candidates = [
        os.environ.get("YOLO_MODEL_PATH"),
        preferred_model,
        "yolo_oneframe_v2.pt",
    ]
    for candidate in preferred_candidates:
        if not candidate:
            continue
        if os.path.isabs(candidate) and os.path.exists(candidate):
            logger.info(f"🔍 Encontrado modelo preferido: {candidate}")
            return candidate
        for directory in search_dirs:
            full_path = os.path.join(directory, candidate)
            if os.path.exists(full_path):
                logger.info(f"🔍 Encontrado modelo preferido: {full_path}")
                return full_path

    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            lower_name = filename.lower()
            if not filename.endswith(".pt"):
                continue
            if any(token in lower_name for token in ("yolo", "best", "ball", "custom")):
                full_path = os.path.join(directory, filename)
                logger.info(f"🔍 Encontrado modelo: {full_path}")
                return full_path
    return "yolov8s.pt"


def _download_latest_ai_model(config: VisionConfig, cloud_manager: CloudManager) -> Optional[str]:
    if not cloud_manager.client or not cloud_manager.r2_read_client:
        logger.info("ℹ️ Modelo dinámico omitido: Supabase o R2 no están listos.")
        return None

    try:
        response = (
            cloud_manager.client.table("ai_memory")
            .select("memory_key, memory_value")
            .in_("memory_key", ["model_version", "model_path"])
            .execute()
        )
        memory = {
            row.get("memory_key"): row.get("memory_value")
            for row in (response.data or [])
            if row.get("memory_key")
        }
        model_key = str(memory.get("model_path") or "").strip()
        model_version = str(memory.get("model_version") or "").strip()
        if not model_key or not model_key.endswith(".pt"):
            return None

        current_name = os.path.basename(str(config.yolo.model_path or ""))
        remote_name = os.path.basename(model_key)
        if current_name == remote_name and os.path.exists(config.yolo.model_path):
            logger.info("🧠 Modelo AI Memory ya está local: %s", config.yolo.model_path)
            return config.yolo.model_path

        local_dir = "/tmp/oneframe_models"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, remote_name)
        if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024 * 1024:
            logger.info("⬇️ Descargando modelo AI Memory %s desde R2: %s", model_version, model_key)
            cloud_manager.r2_read_client.download_file(
                cloud_manager.r2_read_bucket,
                model_key,
                local_path,
            )

        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024 * 1024:
            config.yolo.model_path = local_path
            logger.info("✅ Modelo AI Memory activo: %s (%s)", local_path, model_version or model_key)
            return local_path

    except Exception as exc:
        logger.warning("⚠️ No se pudo cargar modelo desde ai_memory/R2: %s", exc)

    return None


def _persist_generated_clips(
    clips: List[Dict[str, Any]],
    match_id: str,
    cloud_manager: CloudManager,
) -> List[Dict[str, Any]]:
    if not clips:
        return []

    persisted = []
    for clip in clips:
        local_path = clip.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue

        clip_number = int(clip.get("clip_number") or (len(persisted) + 1))
        filename = clip.get("filename") or f"clip_{clip_number:03d}.mp4"
        storage_path = f"clips/{match_id}/{filename}"
        score_breakdown = (
            clip.get("score_breakdown")
            if isinstance(clip.get("score_breakdown"), dict)
            else {}
        )
        team_a_count = _first_numeric(
            clip.get("team_a_count"),
            score_breakdown.get("team_a_count"),
            score_breakdown.get("team_a_near"),
            score_breakdown.get("team_a_in_zone_count"),
        )
        team_b_count = _first_numeric(
            clip.get("team_b_count"),
            score_breakdown.get("team_b_count"),
            score_breakdown.get("team_b_near"),
            score_breakdown.get("team_b_in_zone_count"),
        )
        players_in_danger_zone = _first_numeric(
            clip.get("players_in_danger_zone"),
            score_breakdown.get("players_in_danger_zone"),
            score_breakdown.get("person_count"),
        )
        clip_metadata = {
            "match_id": match_id,
            "match_uuid": match_id if UUID_PATTERN.match(str(match_id)) else None,
            "clip_number": clip_number,
            "start_time": float(clip.get("start_time", 0.0)),
            "timestamp_sec": float(clip.get("timestamp_sec", clip.get("start_time", 0.0))),
            "event_type": clip.get("event_type"),
            "review_status": "pending",
            "is_confirmed": False,
            "speed": float(clip.get("speed", 0.0)),
            "speed_kmh": float(clip.get("speed_kmh", 0.0)),
            "score": float(clip.get("score", 0.0)),
            "velocity_class": clip.get("velocity_class", "unknown"),
            "is_approaching": bool(clip.get("is_approaching", False)),
            "has_audio_peak": bool(clip.get("has_audio_peak", False)),
            "approach_angle": clip.get("approach_angle_deg"),
            "in_danger_zone": bool(clip.get("in_danger_zone", False)),
            "score_breakdown": score_breakdown,
            "team_a_count": int(team_a_count) if team_a_count is not None else None,
            "team_b_count": int(team_b_count) if team_b_count is not None else None,
            "players_in_danger_zone": (
                int(players_in_danger_zone)
                if players_in_danger_zone is not None
                else None
            ),
            "player_tracks": clip.get("player_tracks") if isinstance(clip.get("player_tracks"), dict) else None,
        }

        row = cloud_manager.persist_clip(local_path, storage_path, clip_metadata)
        if row is not None:
            persisted.append({**clip, **row})
        else:
            persisted.append({**clip_metadata, **clip})

    return persisted


def _first_numeric(*values: Any) -> Optional[float]:
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        return numeric
    return None


def _required_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise CriticalJobError(f"{field_name} debe ser numerico y no nulo.") from exc
    return numeric


def _required_int(value: Any, field_name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise CriticalJobError(f"{field_name} debe ser entero y no nulo.") from exc
    return numeric


def _compute_gpu_cost_total(results: Dict[str, Any], processing_time: float) -> float:
    explicit_cost = results.get("gpu_cost_total")
    if explicit_cost is not None:
        return _required_float(explicit_cost, "gpu_cost_total")

    raw_cost_per_second = os.getenv("AI_WORKER_V1_GPU_COST_PER_SECOND")
    if raw_cost_per_second in (None, ""):
        raise CriticalJobError(
            "Falta gpu_cost_total o AI_WORKER_V1_GPU_COST_PER_SECOND; "
            "no se puede cerrar match_processing_metrics sin costo."
        )

    cost_per_second = _required_float(raw_cost_per_second, "AI_WORKER_V1_GPU_COST_PER_SECOND")
    if cost_per_second <= 0:
        raise CriticalJobError("AI_WORKER_V1_GPU_COST_PER_SECOND debe ser mayor a 0.")
    return round(processing_time * cost_per_second, 6)


def _persist_training_frames(
    frames: List[Dict[str, Any]],
    cloud_manager: CloudManager,
) -> List[Dict[str, Any]]:
    if not frames:
        return []

    persisted = []
    for frame_metadata in frames:
        row = cloud_manager.persist_training_frame(frame_metadata)
        if row is not None:
            persisted.append(row)

    logger.info(
        "🧪 Training frames persistidos: %s/%s",
        len(persisted),
        len(frames),
    )
    return persisted


def _record_match_processing_metrics(
    match_id: str,
    results: Dict[str, Any],
    cloud_manager: CloudManager,
) -> None:
    if not cloud_manager.client:
        raise CriticalJobError("No se puede registrar match_processing_metrics: Supabase no configurado.")

    processing_time = float(results.get("processing_time_sec") or 0.0)
    if processing_time <= 0:
        raise CriticalJobError("processing_time_seconds debe ser mayor a 0.")

    gpu_cost_total = _compute_gpu_cost_total(results, processing_time)
    if gpu_cost_total < 0:
        raise CriticalJobError("gpu_cost_total debe ser mayor o igual a 0.")

    clips_generated = _required_int(results.get("clips_generated"), "clips_generated")
    clips_marked_good = _required_int(results.get("clips_marked_good", 0), "clips_marked_good")
    worker_type = str(results.get("worker_type") or SHADOW_WORKER_TYPE)
    if not worker_type.startswith("ai_worker_v1"):
        raise CriticalJobError("worker_type debe comenzar con ai_worker_v1.")

    row = {
        "match_id": match_id,
        "worker_type": worker_type,
        "processing_time_seconds": processing_time,
        "gpu_cost_total": gpu_cost_total,
        "ball_recall": None,
        "ball_false_positives": None,
        "clips_generated": clips_generated,
        "clips_marked_good": clips_marked_good,
    }

    try:
        response = cloud_manager.client.table("match_processing_metrics").insert(row).execute()
    except Exception as exc:
        raise CriticalJobError(
            f"No se pudo insertar match_processing_metrics para match_id={match_id}: {exc}"
        ) from exc

    inserted_rows = getattr(response, "data", None)
    if not inserted_rows:
        raise CriticalJobError(
            f"Supabase no confirmo fila insertada en match_processing_metrics para match_id={match_id}."
        )

    inserted_match_id = str((inserted_rows[0] or {}).get("match_id", ""))
    if inserted_match_id != str(match_id):
        raise CriticalJobError(
            "Supabase devolvio una fila de match_processing_metrics con match_id inesperado."
        )

    logger.info("🧪 Shadow metrics registradas para match_id=%s", match_id)


def _serialize_detection(det: Any, threshold: float, timestamp_sec: float, frame_index: int) -> Dict[str, Any]:
    x = round(float(getattr(det, "x", 0.0) or 0.0), 3)
    y = round(float(getattr(det, "y", 0.0) or 0.0), 3)
    w = round(float(getattr(det, "w", 0.0) or 0.0), 3)
    h = round(float(getattr(det, "h", 0.0) or 0.0), 3)
    class_id = int(getattr(det, "class_id", -1) or -1)
    class_name = str(getattr(det, "class_name", "") or "")
    confidence = round(float(getattr(det, "confidence", 0.0) or 0.0), 6)
    frame_idx = int(getattr(det, "frame_index", frame_index) or frame_index)
    timestamp = round(float(getattr(det, "timestamp_sec", timestamp_sec) or timestamp_sec), 3)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "produced",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "bbox": {
            "xywh": [x, y, w, h],
            "xyxy": [
                round(x - w / 2.0, 3),
                round(y - h / 2.0, 3),
                round(x + w / 2.0, 3),
                round(y + h / 2.0, 3),
            ],
        },
        "confidence": confidence,
        "class_id": class_id,
        "class_name": class_name,
        "detector_source": str(getattr(det, "detector_source", "") or ""),
        "detector_mode": str(getattr(det, "detector_mode", "") or ""),
        "threshold": float(getattr(det, "threshold", threshold) or threshold),
        "frame_index": frame_idx,
        "timestamp_sec": timestamp,
    }
    metadata = getattr(det, "metadata", None)
    if isinstance(metadata, dict):
        for key in [
            "raw_status",
            "final_status",
            "acceptance_reason",
            "diagnostic_override_used",
            "motion_state_status",
            "yolo_agreement",
            "yolo_center_distance_px",
            "yolo_iou",
            "dt_since_previous_accepted",
            "jump_px",
            "allowed_jump_px",
            "kalman_used",
            "kalman_distance_px",
        ]:
            if key in metadata:
                payload[key] = metadata[key]
        payload["metadata"] = metadata
    return payload


def _tag_detection(
    det: Any,
    source: str,
    detector_mode: str,
    threshold: float,
    frame_index: int,
    timestamp_sec: float,
) -> Any:
    setattr(det, "detector_source", source)
    setattr(det, "detector_mode", detector_mode)
    setattr(det, "threshold", float(threshold))
    setattr(det, "frame_index", int(frame_index))
    setattr(det, "timestamp_sec", float(timestamp_sec))
    return det


def _safe_file_size(local_path: str) -> Optional[int]:
    try:
        return os.path.getsize(local_path)
    except OSError:
        return None


def _count_by_key(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _produced_artifact(
    run_id: str,
    match_id: str,
    name: str,
    object_key: str,
    local_path: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    content_type, _ = mimetypes.guess_type(name)
    return create_produced_artifact(
        run_id=run_id,
        match_id=match_id,
        name=name,
        path=object_key,
        content_type=content_type or "",
        size_bytes=_safe_file_size(local_path),
        metadata=metadata or {},
    )


def _build_segment_metadata(
    *,
    run_id: str,
    match_id: str,
    detector_mode: str,
    worker_type: str,
    video_source: str,
    start_sec: float,
    duration_sec: float,
    frame_stride: int,
    output_prefix: str,
    created_at: str,
    artifacts: List[Any],
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "run_id": run_id,
        "match_id": match_id,
        "detector_mode": detector_mode,
        "worker_type": worker_type,
        "docker_image_tag": (
            os.getenv("ONEFRAME_IMAGE_TAG")
            or os.getenv("DOCKER_IMAGE_TAG")
            or os.getenv("IMAGE_TAG")
            or ""
        ),
        "video_source": video_source,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "frame_stride": frame_stride,
        "created_at": created_at,
        "output_prefix": output_prefix,
        "risk_notes": [
            "RF-DETR is the target detector architecture, but real-video performance is still under validation.",
            "process_segment_shadow is a segment-level shadow run, not full-match production processing.",
        ],
        "required_artifacts": list(REQUIRED_ARTIFACT_NAMES),
        "artifacts": artifact_manifest(artifacts),
    }


def _build_segment_run_summary_markdown(
    *,
    run_id: str,
    match_id: str,
    detector_mode: str,
    raw_frames_seen: int,
    processed_frames: int,
    metrics: Dict[str, Any],
    discarded_by_reason: Dict[str, int],
    accepted_by_reason: Dict[str, int],
    clips_generated: int,
    artifacts: List[Any],
) -> str:
    produced = [artifact.name for artifact in artifacts if artifact.status == "produced"]
    not_available = [artifact.name for artifact in artifacts if artifact.status == "not_available"]
    rfdetr_accepted = int(metrics.get("rfdetr_ball_frames") or 0)
    rfdetr_discarded = int(metrics.get("ball_detections_discarded_by_physics_filter") or 0)
    discarded_lines = (
        "\n".join(f"- {reason}: {count}" for reason, count in sorted(discarded_by_reason.items()))
        if discarded_by_reason
        else "- none"
    )
    accepted_lines = (
        "\n".join(f"- {reason}: {count}" for reason, count in sorted((accepted_by_reason or {}).items()))
        if accepted_by_reason
        else "- none"
    )
    produced_lines = "\n".join(f"- {name}" for name in produced) if produced else "- none"
    unavailable_lines = "\n".join(f"- {name}" for name in not_available) if not_available else "- none"

    return "\n".join(
        [
            "# OneFrame Segment Shadow Run Summary",
            "",
            f"- match_id: `{match_id}`",
            f"- run_id: `{run_id}`",
            f"- detector_mode: `{detector_mode}`",
            f"- frames_read: `{raw_frames_seen}`",
            f"- frames_processed: `{processed_frames}`",
            f"- RF-DETR frames con balon: `{metrics.get('rfdetr_ball_frames', 0)}`",
            f"- RF-DETR raw frames: `{metrics.get('rfdetr_raw_frames', 0)}`",
            f"- YOLO frames con balon: `{metrics.get('yolo_ball_frames', 0)}`",
            f"- RF-DETR accepted: `{rfdetr_accepted}`",
            f"- RF-DETR discarded: `{rfdetr_discarded}`",
            f"- stale_state_accepted: `{metrics.get('stale_state_accepted', 0)}`",
            f"- yolo_agreement_accepted: `{metrics.get('yolo_agreement_accepted', 0)}`",
            f"- clips_generated: `{clips_generated}`",
            "",
            "## Accepted By Reason",
            "",
            accepted_lines,
            "",
            "## Discarded By Reason",
            "",
            discarded_lines,
            "",
            "## Artifacts Produced",
            "",
            produced_lines,
            "",
            "## Artifacts Not Available",
            "",
            unavailable_lines,
            "",
            "## Riesgos Pendientes",
            "",
            "- Validar visualmente TP/FP de RF-DETR en video real.",
            "- Revisar detecciones descartadas por filtros fisicos antes de promover thresholds.",
            "- Este run no ejecuta tracking, homografia, eventos ni tactica en la fase actual.",
            "",
        ]
    )


def _write_not_available_artifact_file(
    local_dir: str,
    artifact: Any,
) -> Optional[str]:
    name = str(getattr(artifact, "name", "") or "")
    if name.endswith("/") or not (name.endswith(".json") or name.endswith(".md")):
        return None

    local_path = os.path.join(local_dir, name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "artifact_name": name,
        "run_id": artifact.run_id,
        "match_id": artifact.match_id,
        "status": artifact.status,
        "reason": artifact.reason,
        "created_at": utc_now_iso(),
    }
    if name.endswith(".md"):
        with open(local_path, "w", encoding="utf-8") as file_stream:
            file_stream.write(
                "\n".join(
                    [
                        f"# {name}",
                        "",
                        f"- status: `{artifact.status}`",
                        f"- reason: `{artifact.reason}`",
                        f"- run_id: `{artifact.run_id}`",
                        f"- match_id: `{artifact.match_id}`",
                        "",
                    ]
                )
            )
    else:
        with open(local_path, "w", encoding="utf-8") as file_stream:
            json.dump(payload, file_stream, indent=2)
    return local_path


def _draw_detection_overlay(frame, det: Any, label: str, color: tuple, rank: int = 0) -> None:
    import cv2

    x = float(getattr(det, "x", 0.0) or 0.0)
    y = float(getattr(det, "y", 0.0) or 0.0)
    w = float(getattr(det, "w", 0.0) or 0.0)
    h = float(getattr(det, "h", 0.0) or 0.0)
    x1 = int(round(x - w / 2.0))
    y1 = int(round(y - h / 2.0))
    x2 = int(round(x + w / 2.0))
    y2 = int(round(y + h / 2.0))
    frame_h, frame_w = frame.shape[:2]
    x1, x2 = max(0, x1), min(frame_w - 1, x2)
    y1, y2 = max(0, y1), min(frame_h - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text = f"{label} {float(getattr(det, 'confidence', 0.0) or 0.0):.3f}"
    y_text = max(24, y1 - 8 - rank * 24)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
    cv2.rectangle(frame, (x1, y_text - th - 7), (x1 + tw + 8, y_text + 5), color, -1)
    cv2.putText(frame, text, (x1 + 4, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)


class SegmentPhysicsFilter:
    def __init__(self, config: VisionConfig):
        cfg = config.rfdetr
        self.min_size = float(getattr(cfg, "min_ball_size_px", 4.0) or 4.0)
        self.max_size = float(getattr(cfg, "max_ball_size_px", 90.0) or 90.0)
        self.min_ar = float(getattr(cfg, "min_aspect_ratio", 0.45) or 0.45)
        self.max_ar = float(getattr(cfg, "max_aspect_ratio", 2.2) or 2.2)
        self.max_jump = float(getattr(cfg, "max_jump_px", 280.0) or 280.0)
        self.kalman_max_distance = float(getattr(cfg, "kalman_max_distance_px", 320.0) or 320.0)
        self.motion_state_max_age_sec = float(getattr(cfg, "motion_state_max_age_sec", 2.0) or 2.0)
        self.max_jump_px_per_sec = float(getattr(cfg, "max_jump_px_per_sec", 900.0) or 900.0)
        self.min_jump_gate_px = float(getattr(cfg, "min_jump_gate_px", 120.0) or 120.0)
        self.yolo_agreement_center_px = float(getattr(cfg, "yolo_agreement_center_px", 25.0) or 25.0)
        self.yolo_agreement_iou = float(getattr(cfg, "yolo_agreement_iou", 0.25) or 0.25)
        self.stale_high_conf_threshold = float(getattr(cfg, "stale_high_conf_threshold", 0.70) or 0.70)
        self.conf_threshold = float(getattr(cfg, "conf_threshold", 0.25) or 0.25)
        self.previous_center = None
        self.previous_timestamp = None
        self.velocity = None
        self.motion_state_update_reasons = {
            "fresh_motion_gate",
            "diagnostic_yolo_agreement",
            "high_confidence_geometry_when_stale",
        }

    def filter(
        self,
        detections: List[Any],
        frame_index: int,
        timestamp_sec: float,
        yolo_detections: Optional[List[Any]] = None,
        diagnostic_mode: bool = True,
    ) -> tuple[List[Any], List[Dict[str, Any]]]:
        accepted = []
        discarded = []
        for det in sorted(detections, key=lambda item: item.confidence, reverse=True):
            decision = self.evaluate(det, timestamp_sec, yolo_detections or [], diagnostic_mode=diagnostic_mode)
            self._attach_decision_metadata(det, decision)
            if decision["final_status"] == "discarded":
                discarded.append(self._discarded_row(det, frame_index, timestamp_sec, decision))
                continue
            accepted.append(det)

        for det in accepted:
            metadata = getattr(det, "metadata", {}) or {}
            if metadata.get("acceptance_reason") in self.motion_state_update_reasons:
                self._update_motion_state(det, timestamp_sec)
                break
        return accepted, discarded

    def evaluate(
        self,
        det: Any,
        timestamp_sec: float,
        yolo_detections: List[Any],
        diagnostic_mode: bool = True,
    ) -> Dict[str, Any]:
        dt = self._dt_since_previous(timestamp_sec)
        motion_state_status = self._motion_state_status(dt)
        yolo_match = self._best_yolo_agreement(det, yolo_detections)
        yolo_agreement = bool(
            yolo_match
            and (
                yolo_match["center_distance_px"] <= self.yolo_agreement_center_px
                or yolo_match["iou"] >= self.yolo_agreement_iou
            )
        )
        base = {
            "raw_status": "candidate",
            "final_status": "discarded",
            "reason": "",
            "acceptance_or_rejection_reason": "",
            "acceptance_reason": "",
            "diagnostic_override_used": False,
            "motion_state_status": motion_state_status,
            "dt_since_previous_accepted": self._round_or_none(dt),
            "jump_px": None,
            "allowed_jump_px": None,
            "kalman_used": False,
            "kalman_distance_px": None,
            "yolo_same_frame": bool(yolo_detections),
            "yolo_agreement": yolo_agreement,
            "yolo_center_distance_px": self._round_or_none(yolo_match["center_distance_px"] if yolo_match else None),
            "yolo_iou": self._round_or_none(yolo_match["iou"] if yolo_match else None),
        }

        width = float(getattr(det, "w", 0.0) or 0.0)
        height = float(getattr(det, "h", 0.0) or 0.0)
        if min(width, height) < self.min_size or max(width, height) > self.max_size:
            return self._reject(base, "bbox_size")
        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio < self.min_ar or aspect_ratio > self.max_ar:
            return self._reject(base, "aspect_ratio")

        if diagnostic_mode and yolo_agreement:
            base["diagnostic_override_used"] = True
            return self._accept(base, "diagnostic_yolo_agreement")

        confidence = float(getattr(det, "confidence", 0.0) or 0.0)
        if motion_state_status == "uninitialized":
            if confidence >= self.conf_threshold:
                return self._accept(base, "high_confidence_geometry_when_stale")
            return self._reject(base, "stale_motion_low_confidence")

        if motion_state_status == "stale":
            if confidence >= self.stale_high_conf_threshold:
                return self._accept(base, "high_confidence_geometry_when_stale")
            return self._reject(base, "stale_motion_low_confidence")

        if self.previous_center is not None and dt is not None:
            jump = math.hypot(float(det.x) - self.previous_center[0], float(det.y) - self.previous_center[1])
            allowed_jump = max(self.min_jump_gate_px, self.max_jump_px_per_sec * max(dt, 0.0))
            base["jump_px"] = round(float(jump), 3)
            base["allowed_jump_px"] = round(float(allowed_jump), 3)
            if jump > allowed_jump:
                return self._reject(base, "jump_distance")

        if self.previous_center is not None and self.velocity is not None and self.previous_timestamp is not None:
            predicted = (
                self.previous_center[0] + self.velocity[0] * dt,
                self.previous_center[1] + self.velocity[1] * dt,
            )
            distance = math.hypot(float(det.x) - predicted[0], float(det.y) - predicted[1])
            base["kalman_used"] = True
            base["kalman_distance_px"] = round(float(distance), 3)
            if distance > self.kalman_max_distance:
                return self._reject(base, "kalman_incompatibility")
        return self._accept(base, "fresh_motion_gate")

    def reject_reason(self, det: Any, timestamp_sec: float) -> str:
        return self.evaluate(det, timestamp_sec, [], diagnostic_mode=False)["reason"]

    def _dt_since_previous(self, timestamp_sec: float) -> Optional[float]:
        if self.previous_timestamp is None:
            return None
        return max(0.0, float(timestamp_sec) - float(self.previous_timestamp))

    def _motion_state_status(self, dt: Optional[float]) -> str:
        if self.previous_center is None or self.previous_timestamp is None:
            return "uninitialized"
        if dt is None or dt > self.motion_state_max_age_sec:
            return "stale"
        return "fresh"

    def _accept(self, base: Dict[str, Any], reason: str) -> Dict[str, Any]:
        base["final_status"] = "accepted"
        base["reason"] = ""
        base["acceptance_reason"] = reason
        base["acceptance_or_rejection_reason"] = reason
        return base

    def _reject(self, base: Dict[str, Any], reason: str) -> Dict[str, Any]:
        base["final_status"] = "discarded"
        base["reason"] = reason
        base["acceptance_reason"] = ""
        base["acceptance_or_rejection_reason"] = reason
        return base

    @staticmethod
    def _round_or_none(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        return round(float(value), 3)

    def _attach_decision_metadata(self, det: Any, decision: Dict[str, Any]) -> None:
        metadata = getattr(det, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.update(decision)
        setattr(det, "metadata", metadata)

    def _discarded_row(
        self,
        det: Any,
        frame_index: int,
        timestamp_sec: float,
        decision: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "frame_index": int(frame_index),
            "timestamp_sec": round(float(timestamp_sec), 3),
            "reason": str(decision.get("reason") or "unknown"),
            "confidence": round(float(getattr(det, "confidence", 0.0) or 0.0), 6),
            "bbox": [
                round(float(getattr(det, "x", 0.0) or 0.0), 3),
                round(float(getattr(det, "y", 0.0) or 0.0), 3),
                round(float(getattr(det, "w", 0.0) or 0.0), 3),
                round(float(getattr(det, "h", 0.0) or 0.0), 3),
            ],
            "dt_since_previous_accepted": decision.get("dt_since_previous_accepted"),
            "motion_state_status": decision.get("motion_state_status"),
            "jump_px": decision.get("jump_px"),
            "allowed_jump_px": decision.get("allowed_jump_px"),
            "kalman_used": decision.get("kalman_used"),
            "kalman_distance_px": decision.get("kalman_distance_px"),
            "yolo_same_frame": decision.get("yolo_same_frame"),
            "yolo_agreement": decision.get("yolo_agreement"),
            "yolo_center_distance_px": decision.get("yolo_center_distance_px"),
            "yolo_iou": decision.get("yolo_iou"),
            "acceptance_or_rejection_reason": decision.get("acceptance_or_rejection_reason"),
        }

    def _best_yolo_agreement(self, det: Any, yolo_detections: List[Any]) -> Optional[Dict[str, float]]:
        best = None
        for yolo_det in yolo_detections:
            center_distance = math.hypot(float(det.x) - float(yolo_det.x), float(det.y) - float(yolo_det.y))
            iou = self._bbox_iou(det, yolo_det)
            score = min(center_distance, 99999.0) - (iou * 100.0)
            if best is None or score < best["score"]:
                best = {
                    "center_distance_px": float(center_distance),
                    "iou": float(iou),
                    "score": float(score),
                }
        return best

    @staticmethod
    def _bbox_iou(a: Any, b: Any) -> float:
        ax1 = float(a.x) - float(a.w) / 2.0
        ay1 = float(a.y) - float(a.h) / 2.0
        ax2 = float(a.x) + float(a.w) / 2.0
        ay2 = float(a.y) + float(a.h) / 2.0
        bx1 = float(b.x) - float(b.w) / 2.0
        by1 = float(b.y) - float(b.h) / 2.0
        bx2 = float(b.x) + float(b.w) / 2.0
        by2 = float(b.y) + float(b.h) / 2.0
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return inter / denom

    def _update_motion_state(self, det: Any, timestamp_sec: float) -> None:
        center = (float(det.x), float(det.y))
        if self.previous_center is not None and self.previous_timestamp is not None:
            dt = max(1e-3, float(timestamp_sec) - float(self.previous_timestamp))
            self.velocity = (
                (center[0] - self.previous_center[0]) / dt,
                (center[1] - self.previous_center[1]) / dt,
            )
        self.previous_center = center
        self.previous_timestamp = float(timestamp_sec)


def _extract_segment_clip(video_path: str, output_path: str, start_sec: float, duration_sec: float) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def _write_rows_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file_stream:
        writer = csv.DictWriter(file_stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def handle_process_segment_shadow(
    link_video: str,
    video_path: str,
    match_id: str,
    inp: Dict[str, Any],
) -> Dict[str, Any]:
    import cv2

    started_at = time.time()
    detector_mode = str(inp.get("detector_mode") or "rfdetr_primary_shadow")
    if detector_mode != "rfdetr_primary_shadow":
        raise CriticalJobError("process_segment_shadow requiere detector_mode='rfdetr_primary_shadow'.")

    start_sec = _required_float(inp.get("start_sec", 0.0), "start_sec")
    duration_sec = _required_float(inp.get("duration_sec", 120.0), "duration_sec")
    duration_sec = min(max(duration_sec, 1.0), 300.0)
    output_prefix = str(inp.get("output_prefix") or f"rfdetr_shadow/{match_id}").strip().strip("/")
    run_id = str(inp.get("run_id") or uuid.uuid4())
    created_at = utc_now_iso()
    frame_stride_input = inp.get("frame_stride")
    max_overlay_frames = int(inp.get("max_overlay_frames", 240) or 240)

    cloud_manager = CloudManager()
    local_video_path = download_video(link_video, video_path)

    config = VisionConfig.from_json(os.path.join(os.path.dirname(__file__), "config.json"))
    config.detector_mode = "rfdetr_primary_shadow"
    config.rfdetr.conf_threshold = 0.25
    config.rfdetr.yolo_fallback_enabled = bool(inp.get("yolo_fallback_enabled", False))
    model_path = _find_yolo_model(config.yolo.model_path)
    if model_path:
        config.yolo.model_path = model_path

    yolo_detector = YOLODetectorAdapter(BallDetector(config), inference_only=True)
    rfdetr_detector = RFDETRDetectorAdapter(detection_factory=Detection, config=config.rfdetr)
    if not yolo_detector.is_available:
        raise CriticalJobError("YOLO no disponible para personas/comparacion en process_segment_shadow.")
    if rfdetr_detector.detector_status != "available":
        raise CriticalJobError(
            "RF-DETR no disponible para process_segment_shadow: "
            f"{rfdetr_detector.unavailable_reason or rfdetr_detector.load_error}"
        )

    cap = cv2.VideoCapture(local_video_path)
    if not cap.isOpened():
        raise CriticalJobError(f"No se pudo abrir video para segmento: {local_video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
    frame_stride = int(frame_stride_input or max(1, round(fps)))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000.0)

    local_dir = os.path.join("/tmp", "rfdetr_segment_shadow", match_id)
    overlays_dir = os.path.join(local_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    segment_clip_path = os.path.join(local_dir, "source_segment.mp4")
    debug_video_path = os.path.join(local_dir, "overlay_video.mp4")
    metadata_json_path = os.path.join(local_dir, "metadata.json")
    detections_json_path = os.path.join(local_dir, "detections.json")
    detections_csv_path = os.path.join(local_dir, "detections.csv")
    discarded_json_path = os.path.join(local_dir, "rfdetr_discarded.json")
    discarded_csv_path = os.path.join(local_dir, "rfdetr_discarded.csv")
    summary_json_path = os.path.join(local_dir, "summary.json")
    run_summary_md_path = os.path.join(local_dir, "run_summary.md")

    _extract_segment_clip(local_video_path, segment_clip_path, start_sec, duration_sec)

    physics_filter = SegmentPhysicsFilter(config)
    detections_by_frame = []
    discarded_rows = []
    overlay_paths = []
    debug_writer = None
    processed_frames = 0
    raw_frames_seen = 0
    rfdetr_ball_frames = 0
    rfdetr_raw_frames = 0
    rfdetr_raw_detections = 0
    yolo_ball_frames = 0
    rfdetr_only_frames = 0
    yolo_only_frames = 0
    both_detectors_frames = 0
    yolo_latency = []
    rfdetr_latency = []

    try:
        debug_fps = max(1.0, fps / max(frame_stride, 1))
        debug_writer = cv2.VideoWriter(
            debug_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            debug_fps,
            (frame_w, frame_h),
        )

        local_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if timestamp_sec > start_sec + duration_sec:
                break
            abs_frame_idx = int(round(timestamp_sec * fps))
            raw_frames_seen += 1
            if local_idx % frame_stride != 0:
                local_idx += 1
                continue
            local_idx += 1
            processed_frames += 1

            t0 = time.perf_counter()
            yolo_ball, yolo_people = yolo_detector.detect(frame)
            yolo_latency.append((time.perf_counter() - t0) * 1000.0)
            yolo_ball = [
                _tag_detection(det, "yolo_compare", detector_mode, config.yolo.confidence, abs_frame_idx, timestamp_sec)
                for det in sorted(yolo_ball, key=lambda item: item.confidence, reverse=True)
            ]

            t0 = time.perf_counter()
            raw_rfdetr = rfdetr_detector.raw_detections(frame, threshold=config.rfdetr.conf_threshold)
            rfdetr_latency.append((time.perf_counter() - t0) * 1000.0)
            rfdetr_candidates = []
            for raw_det in raw_rfdetr:
                parsed = rfdetr_detector._prediction_to_detection(raw_det, frame_w=frame_w, frame_h=frame_h)
                if parsed is None:
                    continue
                rfdetr_candidates.append(
                    _tag_detection(
                        parsed,
                        "rfdetr_primary",
                        detector_mode,
                        config.rfdetr.conf_threshold,
                        abs_frame_idx,
                        timestamp_sec,
                    )
                )
            if rfdetr_candidates:
                rfdetr_raw_frames += 1
                rfdetr_raw_detections += len(rfdetr_candidates)

            rfdetr_accepted, discarded = physics_filter.filter(
                rfdetr_candidates,
                abs_frame_idx,
                timestamp_sec,
                yolo_detections=yolo_ball,
                diagnostic_mode=bool(inp.get("diagnostic_mode", True)),
            )
            discarded_rows.extend(discarded)

            chosen_ball = rfdetr_accepted
            if not chosen_ball and config.rfdetr.yolo_fallback_enabled and yolo_ball:
                chosen_ball = [
                    _tag_detection(det, "yolo_fallback", detector_mode, config.yolo.confidence, abs_frame_idx, timestamp_sec)
                    for det in yolo_ball
                ]

            yolo_detected = bool(yolo_ball)
            rfdetr_detected = bool(rfdetr_accepted)
            if yolo_detected:
                yolo_ball_frames += 1
            if rfdetr_detected:
                rfdetr_ball_frames += 1
            if yolo_detected and rfdetr_detected:
                both_detectors_frames += 1
            elif yolo_detected:
                yolo_only_frames += 1
            elif rfdetr_detected:
                rfdetr_only_frames += 1

            overlay = frame.copy()
            title = f"t={timestamp_sec:.2f}s frame={abs_frame_idx} mode={detector_mode}"
            cv2.rectangle(overlay, (0, 0), (frame_w, 42), (0, 0, 0), -1)
            cv2.putText(overlay, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            if yolo_ball:
                _draw_detection_overlay(overlay, yolo_ball[0], "YOLO", (0, 220, 255), rank=0)
            if rfdetr_accepted:
                _draw_detection_overlay(overlay, rfdetr_accepted[0], "RFDETR", (60, 220, 60), rank=1)
            if discarded:
                # Draw the first discarded candidate only as red debug context.
                first = discarded[0]
                discard_det = Detection(
                    x=first["bbox"][0],
                    y=first["bbox"][1],
                    w=first["bbox"][2],
                    h=first["bbox"][3],
                    confidence=first["confidence"],
                    class_id=config.rfdetr.sports_ball_class_id,
                    class_name="sports ball",
                )
                _draw_detection_overlay(overlay, discard_det, f"DISCARD:{first['reason']}", (40, 40, 230), rank=2)
            if debug_writer:
                debug_writer.write(overlay)
            if len(overlay_paths) < max_overlay_frames:
                timestamp_token = f"{timestamp_sec:.2f}".replace(".", "_")
                overlay_name = f"overlay_{processed_frames:05d}_t{timestamp_token}.jpg"
                overlay_path = os.path.join(overlays_dir, overlay_name)
                cv2.imwrite(overlay_path, overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                overlay_paths.append(overlay_path)

            detections_by_frame.append(
                {
                    "frame_index": abs_frame_idx,
                    "timestamp_sec": round(float(timestamp_sec), 3),
                    "detector_mode": detector_mode,
                    "selected_detector_source": (
                        str(getattr(chosen_ball[0], "detector_source", "")) if chosen_ball else ""
                    ),
                    "yolo": [_serialize_detection(det, config.yolo.confidence, timestamp_sec, abs_frame_idx) for det in yolo_ball],
                    "rfdetr_accepted": [
                        _serialize_detection(det, config.rfdetr.conf_threshold, timestamp_sec, abs_frame_idx)
                        for det in rfdetr_accepted
                    ],
                    "rfdetr_raw_count": len(rfdetr_candidates),
                    "raw_status": "candidate" if rfdetr_candidates else "none",
                    "final_status": (
                        "accepted" if rfdetr_accepted else ("discarded" if discarded else "none")
                    ),
                    "acceptance_reason": ",".join(
                        sorted(
                            {
                                str((getattr(det, "metadata", {}) or {}).get("acceptance_reason") or "")
                                for det in rfdetr_accepted
                                if str((getattr(det, "metadata", {}) or {}).get("acceptance_reason") or "")
                            }
                        )
                    ),
                    "discarded_reasons": ",".join(
                        sorted({str(row.get("reason") or "") for row in discarded if str(row.get("reason") or "")})
                    ),
                    "diagnostic_override_used": any(
                        bool((getattr(det, "metadata", {}) or {}).get("diagnostic_override_used"))
                        for det in rfdetr_accepted
                    ),
                    "motion_state_status": (
                        str((getattr(rfdetr_accepted[0], "metadata", {}) or {}).get("motion_state_status") or "")
                        if rfdetr_accepted
                        else (str(discarded[0].get("motion_state_status") or "") if discarded else "")
                    ),
                    "yolo_agreement": any(
                        bool((getattr(det, "metadata", {}) or {}).get("yolo_agreement"))
                        for det in rfdetr_accepted
                    )
                    or any(bool(row.get("yolo_agreement")) for row in discarded),
                    "rfdetr_discarded_count": len(discarded),
                    "people_count_yolo": len(yolo_people),
                }
            )
    finally:
        cap.release()
        if debug_writer:
            debug_writer.release()

    avg_rfdetr_latency_ms = round(float(np.mean(rfdetr_latency)), 3) if rfdetr_latency else 0.0
    avg_yolo_latency_ms = round(float(np.mean(yolo_latency)), 3) if yolo_latency else 0.0
    metrics = {
        "rfdetr_raw_frames": rfdetr_raw_frames,
        "rfdetr_raw_detections": rfdetr_raw_detections,
        "rfdetr_ball_frames": rfdetr_ball_frames,
        "yolo_ball_frames": yolo_ball_frames,
        "rfdetr_only_frames": rfdetr_only_frames,
        "yolo_only_frames": yolo_only_frames,
        "both_detectors_frames": both_detectors_frames,
        "ball_detections_discarded_by_physics_filter": len(discarded_rows),
        "clips_generated_by_rfdetr_mode": 1,
        "avg_rfdetr_latency_ms": avg_rfdetr_latency_ms,
        "avg_yolo_latency_ms": avg_yolo_latency_ms,
    }
    accepted_rows = []
    for row in detections_by_frame:
        for det in row.get("rfdetr_accepted", []):
            accepted_rows.append(det)
    accepted_by_reason = _count_by_key(accepted_rows, "acceptance_reason")
    metrics["accepted_by_reason"] = accepted_by_reason
    metrics["stale_state_accepted"] = sum(
        1 for det in accepted_rows if det.get("motion_state_status") == "stale"
    )
    metrics["yolo_agreement_accepted"] = int(accepted_by_reason.get("diagnostic_yolo_agreement", 0))

    with open(detections_json_path, "w", encoding="utf-8") as file_stream:
        json.dump(
            {
                "schema_version": SCHEMA_VERSION,
                "architecture_version": ARCHITECTURE_VERSION,
                "artifact_name": "detections.json",
                "status": "produced",
                "run_id": run_id,
                "match_id": match_id,
                "detector_mode": detector_mode,
                "frames": detections_by_frame,
                "metrics": metrics,
            },
            file_stream,
            indent=2,
        )
    _write_rows_csv(
        detections_csv_path,
        [
            {
                "frame_index": row["frame_index"],
                "timestamp_sec": row["timestamp_sec"],
                "selected_detector_source": row["selected_detector_source"],
                "yolo_count": len(row["yolo"]),
                "rfdetr_accepted_count": len(row["rfdetr_accepted"]),
                "rfdetr_discarded_count": row["rfdetr_discarded_count"],
                "rfdetr_raw_count": row["rfdetr_raw_count"],
                "raw_status": row["raw_status"],
                "final_status": row["final_status"],
                "acceptance_reason": row["acceptance_reason"],
                "discarded_reasons": row["discarded_reasons"],
                "diagnostic_override_used": row["diagnostic_override_used"],
                "motion_state_status": row["motion_state_status"],
                "yolo_agreement": row["yolo_agreement"],
                "people_count_yolo": row["people_count_yolo"],
            }
            for row in detections_by_frame
        ],
        [
            "frame_index",
            "timestamp_sec",
            "selected_detector_source",
            "yolo_count",
            "rfdetr_accepted_count",
            "rfdetr_discarded_count",
            "rfdetr_raw_count",
            "raw_status",
            "final_status",
            "acceptance_reason",
            "discarded_reasons",
            "diagnostic_override_used",
            "motion_state_status",
            "yolo_agreement",
            "people_count_yolo",
        ],
    )
    with open(discarded_json_path, "w", encoding="utf-8") as file_stream:
        json.dump({"discarded": discarded_rows}, file_stream, indent=2)
    _write_rows_csv(
        discarded_csv_path,
        discarded_rows,
        [
            "frame_index",
            "timestamp_sec",
            "reason",
            "confidence",
            "bbox",
            "dt_since_previous_accepted",
            "motion_state_status",
            "jump_px",
            "allowed_jump_px",
            "kalman_used",
            "kalman_distance_px",
            "yolo_same_frame",
            "yolo_agreement",
            "yolo_center_distance_px",
            "yolo_iou",
            "acceptance_or_rejection_reason",
        ],
    )

    uploaded = {}
    upload_specs = {
        "source_segment_clip": segment_clip_path,
        "overlay_video": debug_video_path,
        "detections_json": detections_json_path,
        "detections_csv": detections_csv_path,
        "rfdetr_discarded_json": discarded_json_path,
        "rfdetr_discarded_csv": discarded_csv_path,
    }
    for label, local_path in upload_specs.items():
        object_key = f"{output_prefix}/{os.path.basename(local_path)}"
        url = cloud_manager.upload_file(local_path, object_key)
        uploaded[label] = {
            "object_key": object_key,
            "url": url,
        }

    overlay_uploads = []
    for overlay_path in overlay_paths:
        object_key = f"{output_prefix}/overlays/{os.path.basename(overlay_path)}"
        url = cloud_manager.upload_file(overlay_path, object_key)
        overlay_uploads.append({"object_key": object_key, "url": url})
    uploaded["overlays"] = overlay_uploads

    processing_time_sec = round(time.time() - started_at, 2)
    discarded_by_reason = _count_by_key(discarded_rows, "reason")

    produced_artifacts = {
        "detections.json": _produced_artifact(
            run_id=run_id,
            match_id=match_id,
            name="detections.json",
            object_key=uploaded["detections_json"]["object_key"],
            local_path=detections_json_path,
            metadata={"detector_mode": detector_mode},
        ),
        "metadata.json": _produced_artifact(
            run_id=run_id,
            match_id=match_id,
            name="metadata.json",
            object_key=f"{output_prefix}/metadata.json",
            local_path=metadata_json_path,
        ),
        "run_summary.md": _produced_artifact(
            run_id=run_id,
            match_id=match_id,
            name="run_summary.md",
            object_key=f"{output_prefix}/run_summary.md",
            local_path=run_summary_md_path,
        ),
    }
    if os.path.exists(debug_video_path) and (_safe_file_size(debug_video_path) or 0) > 0:
        produced_artifacts["overlay_video.mp4"] = _produced_artifact(
            run_id=run_id,
            match_id=match_id,
            name="overlay_video.mp4",
            object_key=uploaded["overlay_video"]["object_key"],
            local_path=debug_video_path,
            metadata={"frames_processed": processed_frames},
        )

    artifacts = build_required_artifacts(
        run_id=run_id,
        match_id=match_id,
        produced=produced_artifacts,
        output_prefix=output_prefix,
        default_reason=ARTIFACT_NOT_IMPLEMENTED_REASON,
    )
    artifacts_manifest = artifact_manifest(artifacts)
    artifacts_produced = [artifact["name"] for artifact in artifacts_manifest if artifact["status"] == "produced"]
    artifacts_not_available = [
        artifact["name"] for artifact in artifacts_manifest if artifact["status"] == "not_available"
    ]
    placeholder_uploads = []
    for artifact in artifacts:
        if artifact.status != "not_available":
            continue
        placeholder_path = _write_not_available_artifact_file(local_dir, artifact)
        if not placeholder_path:
            continue
        object_key = f"{output_prefix}/{artifact.name}"
        url = cloud_manager.upload_file(placeholder_path, object_key)
        placeholder_uploads.append(
            {
                "artifact_name": artifact.name,
                "object_key": object_key,
                "url": url,
                "status": artifact.status,
                "reason": artifact.reason,
            }
        )
    uploaded["placeholder_artifacts"] = placeholder_uploads

    metadata = _build_segment_metadata(
        run_id=run_id,
        match_id=match_id,
        detector_mode=detector_mode,
        worker_type=SEGMENT_WORKER_TYPE,
        video_source=link_video,
        start_sec=start_sec,
        duration_sec=duration_sec,
        frame_stride=frame_stride,
        output_prefix=output_prefix,
        created_at=created_at,
        artifacts=artifacts,
    )
    with open(metadata_json_path, "w", encoding="utf-8") as file_stream:
        json.dump(metadata, file_stream, indent=2)

    run_summary_md = _build_segment_run_summary_markdown(
        run_id=run_id,
        match_id=match_id,
        detector_mode=detector_mode,
        raw_frames_seen=raw_frames_seen,
        processed_frames=processed_frames,
        metrics=metrics,
        discarded_by_reason=discarded_by_reason,
        accepted_by_reason=accepted_by_reason,
        clips_generated=1,
        artifacts=artifacts,
    )
    with open(run_summary_md_path, "w", encoding="utf-8") as file_stream:
        file_stream.write(run_summary_md)

    metadata_key = f"{output_prefix}/metadata.json"
    metadata_url = cloud_manager.upload_file(metadata_json_path, metadata_key)
    uploaded["metadata_json"] = {"object_key": metadata_key, "url": metadata_url}

    run_summary_key = f"{output_prefix}/run_summary.md"
    run_summary_url = cloud_manager.upload_file(run_summary_md_path, run_summary_key)
    uploaded["run_summary_md"] = {"object_key": run_summary_key, "url": run_summary_url}

    summary = {
        "status": "success",
        "action": "process_segment_shadow",
        "run_id": run_id,
        "match_id": match_id,
        "worker_type": SEGMENT_WORKER_TYPE,
        "detector_mode": detector_mode,
        "schema_version": SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "created_at": created_at,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "frame_stride": frame_stride,
        "raw_frames_seen": raw_frames_seen,
        "processed_frame_count": processed_frames,
        "output_bucket": cloud_manager.r2_write_bucket,
        "output_prefix": output_prefix,
        "processing_time_sec": processing_time_sec,
        "clips_generated": 1,
        "clips_marked_good": 0,
        "metrics": metrics,
        "rfdetr_raw_frames": rfdetr_raw_frames,
        "rfdetr_accepted_frames": rfdetr_ball_frames,
        "rfdetr_discarded_total": len(discarded_rows),
        "discarded_by_reason": discarded_by_reason,
        "accepted_by_reason": accepted_by_reason,
        "stale_state_accepted": metrics["stale_state_accepted"],
        "yolo_agreement_accepted": metrics["yolo_agreement_accepted"],
        "outputs": uploaded,
        "artifacts": artifacts_manifest,
        "artifacts_produced": artifacts_produced,
        "artifacts_not_available": artifacts_not_available,
        "physics_filters": {
            "min_ball_size_px": config.rfdetr.min_ball_size_px,
            "max_ball_size_px": config.rfdetr.max_ball_size_px,
            "min_aspect_ratio": config.rfdetr.min_aspect_ratio,
            "max_aspect_ratio": config.rfdetr.max_aspect_ratio,
            "max_jump_px": config.rfdetr.max_jump_px,
            "kalman_max_distance_px": config.rfdetr.kalman_max_distance_px,
            "motion_state_max_age_sec": config.rfdetr.motion_state_max_age_sec,
            "max_jump_px_per_sec": config.rfdetr.max_jump_px_per_sec,
            "min_jump_gate_px": config.rfdetr.min_jump_gate_px,
            "yolo_agreement_center_px": config.rfdetr.yolo_agreement_center_px,
            "yolo_agreement_iou": config.rfdetr.yolo_agreement_iou,
            "stale_high_conf_threshold": config.rfdetr.stale_high_conf_threshold,
        },
        "notes": {
            "segment_selection": "fixed_segment_not_event_selected",
            "evolutive_learner": "skipped",
            "retraining": "skipped",
            "artifact_contract": "all_required_artifacts_declared",
        },
    }
    with open(summary_json_path, "w", encoding="utf-8") as file_stream:
        json.dump(summary, file_stream, indent=2)
    summary_key = f"{output_prefix}/summary.json"
    summary_url = cloud_manager.upload_file(summary_json_path, summary_key)
    summary["outputs"]["summary_json"] = {"object_key": summary_key, "url": summary_url}

    if cloud_manager.client:
        metrics_results = {
            "worker_type": SEGMENT_WORKER_TYPE,
            "processing_time_sec": processing_time_sec,
            "clips_generated": 1,
            "clips_marked_good": 0,
        }
        try:
            _record_match_processing_metrics(match_id, metrics_results, cloud_manager)
            summary["match_processing_metrics_inserted"] = True
        except Exception as exc:
            logger.warning("⚠️ No se pudo registrar metrics de segmento: %s", exc)
            summary["match_processing_metrics_inserted"] = False
            summary["match_processing_metrics_error"] = str(exc)
    else:
        summary["match_processing_metrics_inserted"] = False
        summary["match_processing_metrics_error"] = "Supabase no configurado."

    return summary


def handle_metrics_only(inp: Dict[str, Any], match_id: str) -> Dict[str, Any]:
    cloud_manager = CloudManager()
    cloud_manager.require_supabase_metrics_ready()

    results = {
        "worker_type": inp.get("worker_type") or SHADOW_WORKER_TYPE,
        "processing_time_sec": _required_float(
            inp.get("processing_time_seconds", inp.get("processing_time_sec")),
            "processing_time_seconds",
        ),
        "gpu_cost_total": _required_float(inp.get("gpu_cost_total"), "gpu_cost_total"),
        "clips_generated": _required_int(inp.get("clips_generated"), "clips_generated"),
        "clips_marked_good": _required_int(inp.get("clips_marked_good", 0), "clips_marked_good"),
    }
    _record_match_processing_metrics(match_id, results, cloud_manager)
    return {
        "status": "success",
        "action": "metrics_only",
        "match_id": match_id,
        "worker_type": results["worker_type"],
        "processing_time_seconds": results["processing_time_sec"],
        "gpu_cost_total": results["gpu_cost_total"],
        "clips_generated": results["clips_generated"],
        "clips_marked_good": results["clips_marked_good"],
    }


def _read_text_artifacts(paths: List[str]) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for path in paths:
        if os.path.exists(path):
            artifacts[os.path.basename(path)] = open(path, "r", encoding="utf-8").read()
    return artifacts


def handle_pe0a4r_gpu_preannotation(
    *,
    link_video: str,
    video_path: str,
    match_id: str,
    inp: Dict[str, Any],
) -> Dict[str, Any]:
    """Shadow-only GPU preannotation for PE-0A4R supplemental ball review."""
    started_at = time.time()
    ensure_rfdetr_base_checkpoint()
    local_video_path = download_video(link_video, video_path)
    output_dir = os.path.join("/tmp", "pe0a4r_gpu_preannotation", match_id)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from pe0a4r_preannotation import run_gpu_preannotation

    config = VisionConfig.from_json(os.path.join(os.path.dirname(__file__), "config.json"))
    model_path = _find_yolo_model(config.yolo.model_path) or os.getenv("YOLO_MODEL_PATH") or "/app/oneframe_v3_best.pt"
    rfdetr_model_path = (
        inp.get("rfdetr_model_path")
        or os.getenv("AI_WORKER_V1_RFDETR_MODEL_PATH")
        or "/app/rfdetr_cache/rf-detr-base.pth"
    )
    run_id = str(inp.get("run_id") or f"pe0a4r_gpu_{uuid.uuid4()}")
    result = run_gpu_preannotation(
        video_path=local_video_path,
        output_dir=output_dir,
        run_id=run_id,
        yolo_model_path=str(model_path),
        rfdetr_model_path=str(rfdetr_model_path),
        device=str(inp.get("device") or os.getenv("AI_WORKER_V1_RFDETR_DEVICE") or "cuda"),
        confidence=float(inp.get("confidence", 0.25)),
        iou=float(inp.get("iou", 0.45)),
        rfdetr_confidence=float(inp.get("rfdetr_confidence", 0.25)),
        yolo_imgsz=int(inp.get("yolo_imgsz", 640)),
        tile_size=int(inp.get("tile_size", 640)),
        tile_overlap=float(inp.get("tile_overlap", 0.2)),
    )
    pre_dir = result["preannotations_dir"]
    artifact_names = [
        "rfdetr_global_candidates.jsonl",
        "rfdetr_tile_candidates.jsonl",
        "yolo_candidates.jsonl",
        "fused_candidates.jsonl",
        "frame_status.jsonl",
        "preannotation_gpu_validation.json",
        "preannotation_gpu_summary.md",
        "environment_manifest.json",
        "runtime_metrics.json",
        "artifact_manifest.json",
    ]
    artifact_paths = [os.path.join(pre_dir, name) for name in artifact_names]
    return {
        "status": "success" if result["status"] == "completed" else "failed",
        "action": "pe0a4r_gpu_preannotation",
        "match_id": match_id,
        "run_id": run_id,
        "processing_time_sec": round(time.time() - started_at, 3),
        "preannotations_dir": pre_dir,
        "result": result,
        "artifacts": _read_text_artifacts(artifact_paths),
        "notes": {
            "supabase": "not_used",
            "r2": "not_used",
            "clips": "not_generated",
            "training": "not_started",
        },
    }


def _run_evolutive_learning(cloud_manager: CloudManager) -> Dict[str, Any]:
    learner = EvolutiveLearner(cloud_manager.client)
    learner.load_from_ai_memory()
    learner.load_samples()

    result = {
        "samples": learner.sample_count,
        "new_samples": learner.new_sample_count(),
        "evolved": False,
    }
    if not learner.should_evolve(min_new_samples=10):
        logger.info(
            "🧠 EvolutiveLearner omitido: %s muestras, %s nuevas",
            learner.sample_count,
            learner.new_sample_count(),
        )
        return result

    weights = learner.evolve()
    saved = learner.save_to_ai_memory()
    result.update({"evolved": saved, "weights": weights})
    logger.info("🧠 EvolutiveLearner guardado=%s pesos=%s", saved, weights)
    return result


def maybe_trigger_retraining(match_id: str, supabase_client) -> bool:
    """
    Verifica si hay suficientes frames nuevos para re-entrenar.
    Se llama desde el handler al completar cada job.
    """
    if not supabase_client:
        logger.info("⏳ Re-entrenamiento omitido: Supabase no configurado.")
        return False

    try:
        result = (
            supabase_client.table("training_frames")
            .select("id", count="exact")
            .eq("annotated", False)
            .execute()
        )

        pending_count = result.count or 0
        THRESHOLD = 20

        if pending_count < THRESHOLD:
            logger.info(
                f"⏳ Re-entrenamiento: {pending_count}/{THRESHOLD} frames. "
                "No se dispara aún."
            )
            return False

        memory = (
            supabase_client.table("ai_memory")
            .select("memory_value")
            .eq("memory_key", "last_training_triggered_match")
            .execute()
        )

        last_match = memory.data[0]["memory_value"] if memory.data else None
        if last_match == match_id:
            logger.info("⏳ Re-entrenamiento ya disparado para este match.")
            return False

        kaggle_user = os.environ.get("KAGGLE_USERNAME", "oneframe12")
        kaggle_key = os.environ.get("KAGGLE_API_KEY", "")
        notebook_slug = os.environ.get("KAGGLE_NOTEBOOK_SLUG", "notebook013e50cb3f")

        if not kaggle_key:
            logger.warning("⚠️ KAGGLE_API_KEY no configurada en el Docker.")
            return False

        url = f"https://www.kaggle.com/api/v1/kernels/{kaggle_user}/{notebook_slug}/run"
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {kaggle_key}"},
            timeout=15,
        )

        if resp.status_code in (200, 201):
            logger.info(f"🚀 Re-entrenamiento disparado en Kaggle: {notebook_slug}")
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc).isoformat()
            for key, val in [
                ("last_training_triggered", now),
                ("last_training_triggered_match", match_id),
                ("pending_frames", str(pending_count)),
            ]:
                supabase_client.table("ai_memory").upsert(
                    {
                        "memory_key": key,
                        "memory_value": val,
                    },
                    on_conflict="memory_key",
                ).execute()
            return True
        else:
            logger.warning(f"⚠️ Kaggle trigger falló: {resp.status_code} {resp.text[:200]}")
            return False

    except Exception as exc:
        logger.warning(f"⚠️ Error en trigger re-entrenamiento: {exc}")
        return False


def _error_response(
    match_id: str,
    message: str,
    action: str = "process_match",
    error_code: Optional[str] = None,
) -> Dict[str, Any]:
    if error_code:
        logger.error(f"❌ {message} | error_code={error_code}")
    else:
        logger.error(f"❌ {message}")
    return {
        "status": "error",
        "action": action,
        "match_id": match_id,
        "message": message,
        "error_code": error_code,
        "telemetry": [],
        "detection_by_minute": [],
        "rejected_candidates": [],
        "audio_peaks_sec": [],
        "total_video_sec": 0,
        "ball_detected_sec": 0,
        "processing_time_sec": 0,
        "infractions_count": 0,
        "clips": [],
    }


def handle_match_processing(
    link_video: str,
    video_path: str,
    match_id: str,
    roi_points=None,
    danger_zone=None,
    camera_type: Optional[str] = None,
    debug_video: bool = False,
    use_sahi: bool = False,
) -> Dict[str, Any]:
    started_at = time.time()
    clips_dir = "/tmp/clips"

    cloud_manager = CloudManager()
    cloud_manager.require_supabase_metrics_ready()

    local_video_path = download_video(link_video, video_path)

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = VisionConfig.from_json(config_path)

    if camera_type == "gopro":
        config.yolo.lens_distortion_correction = True
        logger.info("📷 GoPro detectado — corrección de distorsión de lente activada")
    if use_sahi:
        config.yolo.use_sahi = True
        logger.info("🔎 SAHI activado por input del job")

    _download_latest_ai_model(config, cloud_manager)
    model_path = _find_yolo_model(config.yolo.model_path)
    if model_path:
        config.yolo.model_path = model_path
        logger.info(f"🔍 Modelo YOLO encontrado: {model_path}")

    logger.info("🎵 Pre-escaneando audio para identificar ventanas candidatas...")
    candidate_windows = None
    try:
        pre_scanner = AudioPreScanner(config)
        candidate_windows = pre_scanner.scan(local_video_path)
        if candidate_windows:
            approx_total = max(w["end"] for w in candidate_windows)
            coverage = pre_scanner.get_coverage_pct(candidate_windows, approx_total)
            logger.info(
                "🎯 Pre-scan: %d ventanas, ~%.1f%% del video a procesar",
                len(candidate_windows), coverage,
            )
        else:
            logger.info("⚠️ Pre-scan sin picos — procesando video completo")
            candidate_windows = None
    except Exception as exc:
        logger.warning("⚠️ AudioPreScanner falló (%s) — procesando video completo", exc)
        candidate_windows = None

    logger.info("🔄 Iniciando VisionEngine...")
    engine = VisionEngine(config)
    results = engine.process_video(
        video_path=local_video_path,
        roi_points=roi_points,
        danger_zone=danger_zone,
        match_id=match_id,
        candidate_windows=candidate_windows,
        debug_video=debug_video,
    )

    logger.info("⚖️ Iniciando GameReferee...")
    referee = GameReferee(config)
    results = referee.process_and_clip(
        engine_results=results,
        video_path=local_video_path,
        output_dir=clips_dir,
    )

    results["clips"] = _persist_generated_clips(results.get("clips", []), match_id, cloud_manager)
    debug_video_path = results.get("debug_video_path")
    if debug_video_path:
        debug_object_key = f"debug/{match_id}/debug.mp4"
        uploaded_debug_url = cloud_manager.upload_file(debug_video_path, debug_object_key)
        signed_debug_url = (
            cloud_manager.build_signed_url(debug_object_key, expires_in=24 * 60 * 60)
            if uploaded_debug_url is not None
            else None
        )
        debug_url = signed_debug_url or uploaded_debug_url
        if debug_url:
            results["debug_video_url"] = debug_url
            results["debug_video_object_key"] = debug_object_key
            results["debug_video_url_expires_in_sec"] = 24 * 60 * 60 if signed_debug_url else None
            logger.info(
                "🧪 Debug video subido a R2: %s (signed=%s)",
                debug_object_key,
                bool(signed_debug_url),
            )
    results["match_id"] = match_id
    results["status"] = "success"
    results["action"] = "process_match"
    results["processing_time_sec"] = round(time.time() - started_at, 2)
    results["clips_generated"] = len(results.get("clips", []))
    results["learner"] = {"shadow_mode": True, "skipped": True}
    results["retraining_triggered"] = False
    _record_match_processing_metrics(match_id, results, cloud_manager)

    return results


def handle_runtime_canary(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal serverless runtime/storage canary; never loads application models."""
    started = time.time()
    run_id = str(inp.get("run_id") or "runtime-canary")
    cuda_available = False
    gpu = "unknown"
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu = str(torch.cuda.get_device_name(0))
        else:
            gpu = "cuda_unavailable"
    except Exception:
        gpu = "torch_unavailable"
    subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    payload = {
        "run_id": run_id,
        "status": "ok",
        "gpu": gpu,
        "cuda_available": cuda_available,
        "python": sys.version.split()[0],
        "torch": __import__("torch").__version__,
        "ffmpeg": "available",
        "duration_seconds": round(time.time() - started, 3),
    }
    path = f"/tmp/{run_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True)
    cloud = CloudManager()
    object_key = f"runs/{run_id}/runtime_canary.json"
    artifact_uri = cloud.upload_file(path, object_key)
    if not artifact_uri:
        raise CriticalJobError("R2 runtime canary upload failed")
    return {
        "run_id": run_id,
        "status": "completed",
        "gpu": gpu,
        "cuda_available": cuda_available,
        "artifact_uri": object_key,
        "duration_seconds": round(time.time() - started, 3),
    }


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    video_path = "/tmp/video_procesar"
    clips_dir = "/tmp/clips"
    match_id = "unknown"
    action = "process_match"

    try:
        inp = job.get("input", {}) or {}
        if not inp:
            return _error_response(match_id, "Missing 'input' in job payload", action=action)

        link_video = inp.get("link_video") or inp.get("drive_link") or inp.get("video_url") or ""
        link_video = normalize_video_url(link_video) if link_video else None
        match_id = inp.get("match_id", "unknown")
        action = str(inp.get("action", "process_match") or "process_match").strip()
        if inp.get("job_type") == "runtime_canary":
            return handle_runtime_canary(inp)
        roi_points = inp.get("roi_points")
        danger_zone = inp.get("danger_zone")
        if danger_zone is None:
            danger_zone = inp.get("penalty_area_1") or inp.get("penalty_area_2")
        camera_type = inp.get("camera_type")
        debug_mode = bool(inp.get("debug_mode", False))
        use_sahi = bool(inp.get("use_sahi", False))

        logger.info(f"🎯 Processing match: {match_id}")
        if link_video:
            logger.info(f"📥 Video link: {link_video[:80]}...")
        logger.info(f"🧭 Handler action received: {action}")

        if action == "metrics_only":
            logger.info("🧪 Entering metrics_only branch.")
            return handle_metrics_only(inp, match_id)

        if not link_video:
            return _error_response(match_id, "Missing 'link_video' in input", action=action)

        if action == "pe0a4r_gpu_preannotation":
            logger.info("🧪 Entering PE-0A4R GPU preannotation branch.")
            return handle_pe0a4r_gpu_preannotation(
                link_video=link_video,
                video_path=video_path,
                match_id=match_id,
                inp=inp,
            )

        if action in {"get_frame", "extract_frame"}:
            logger.info("🪶 Entering lightweight frame extraction branch.")
            frame_ratio = inp.get("frame_ratio", DEFAULT_FRAME_RATIO)
            return handle_frame_extraction(link_video, video_path, frame_ratio)

        if action == "process_segment_shadow":
            logger.info("🧪 Entering process_segment_shadow branch.")
            return handle_process_segment_shadow(
                link_video=link_video,
                video_path=video_path,
                match_id=match_id,
                inp=inp,
            )

        if action == "process_match":
            logger.info("🏟️ Entering full process_match branch.")
            return handle_match_processing(
                link_video=link_video,
                video_path=video_path,
                match_id=match_id,
                roi_points=roi_points,
                danger_zone=danger_zone,
                camera_type=camera_type,
                debug_video=debug_mode,
                use_sahi=use_sahi,
            )

        return _error_response(match_id, f"Unknown action: {action}", action=action)

    except DownloadError as exc:
        logger.error(
            "❌ DownloadError procesando %s | error_code=%s | message=%s",
            match_id,
            exc.error_code,
            str(exc),
            exc_info=True,
        )
        return _error_response(
            match_id,
            str(exc),
            action=action,
            error_code=exc.error_code,
        )

    except CriticalJobError:
        logger.error("❌ Critical job failure; RunPod debe marcar FAILED.", exc_info=True)
        raise

    except Exception as exc:
        logger.error(f"❌ Error procesando: {exc}", exc_info=True)
        return _error_response(match_id, f"Processing failed: {str(exc)}", action=action)

    finally:
        for candidate in glob.glob(f"{video_path}*"):
            if not os.path.isfile(candidate):
                continue
            try:
                os.remove(candidate)
                logger.info(f"🧹 Borrado: {candidate}")
            except Exception:
                pass

        if os.path.isdir(clips_dir):
            try:
                shutil.rmtree(clips_dir, ignore_errors=True)
                logger.info(f"🧹 Borrado: {clips_dir}")
            except Exception:
                pass


def _run_local_test():
    test_job = {
        "input": {
            "link_video": "https://example.com/video.mp4",
            "match_id": "test_local",
        }
    }
    logger.info("⚠️ Para test real, proporciona un link de video válido")
    logger.info("Handler cargado correctamente. Clases: engine ✅ config ✅")
    return test_job


if __name__ == "__main__":
    if os.getenv("ONEFRAME_LOCAL_TEST") == "1":
        logger.info("🏠 Modo desarrollo local explícito")
        _run_local_test()
    else:
        import runpod

        logger.info(
            "🚀 Starting OneFrame Handler on RunPod serverless... ONEFRAME_ENV=%s",
            os.getenv("ONEFRAME_ENV", ""),
        )
        runpod.serverless.start({"handler": handler})
