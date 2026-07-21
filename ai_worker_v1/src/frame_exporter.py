"""
Exporta frames difíciles para el pipeline automático de reentrenamiento.

Los frames quedan en R2 bajo:
  training_frames/{match_id}/{label}_frame_{n}.jpg
  training_frames/{match_id}/missed_{n}.jpg

Y se registra metadata en Supabase:
  training_frames(match_id, frame_path, confidence, frame_number, exported, annotated)
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import boto3
import cv2

try:
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None


logger = logging.getLogger("OneFrame.FrameExporter")


class FrameExporter:
    def __init__(
        self,
        confidence_threshold: float = 0.3,
        max_frames_per_match: int = 50,
        jpeg_quality: int = 92,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_frames_per_match = max_frames_per_match
        self.jpeg_quality = jpeg_quality
        self.export_counts: Dict[str, int] = {}
        self.exported_frames: List[Dict[str, Any]] = []
        self.r2_bucket = os.getenv("AI_WORKER_V1_R2_OUTPUT_BUCKET", "one-frame-shadow")
        self.r2_client = self._build_r2_client()
        self.supabase = self._build_supabase_client()

    def should_export(self, confidence: float) -> bool:
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return False
        return 0 < value < self.confidence_threshold

    def export_frame(
        self,
        frame,
        match_id: str,
        frame_num: int,
        confidence: float,
        label: str = "low_confidence",
        force: bool = False,
        max_frames: Optional[int] = None,
        filename_prefix: Optional[str] = None,
        hard_case_type: Optional[str] = None,
        label_quality: str = "C",
        teacher_confidence: float = 0.0,
        physics_ok: bool = True,
    ) -> Optional[Dict[str, Any]]:
        safe_match_id = self._safe_match_id(match_id)
        safe_label = self._safe_label(label)
        count_key = f"{safe_match_id}:{safe_label}"
        current_count = self.export_counts.get(count_key, 0)
        frame_limit = int(max_frames or self.max_frames_per_match)
        if current_count >= frame_limit:
            return None
        if not force and not self.should_export(confidence):
            return None

        if filename_prefix:
            safe_prefix = self._safe_label(filename_prefix)
            filename = f"{safe_prefix}_{int(frame_num):08d}.jpg"
        else:
            filename = f"{safe_label}_frame_{int(frame_num):08d}.jpg"
        object_key = f"training_frames/{safe_match_id}/{filename}"
        local_dir = os.path.join("/tmp", "training_frames", safe_match_id)
        local_path = os.path.join(local_dir, filename)
        exported_at = datetime.now(timezone.utc).isoformat()

        try:
            os.makedirs(local_dir, exist_ok=True)
            ok = cv2.imwrite(
                local_path,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
            )
            if not ok:
                logger.warning("⚠️ No se pudo escribir frame training: %s", local_path)
                return None
        except Exception as exc:
            logger.warning("⚠️ Error guardando frame training %s: %s", frame_num, exc)
            return None

        if not self._upload_to_r2(local_path, object_key):
            return None

        metadata = {
            "match_id": match_id,
            "frame_path": object_key,
            "label": safe_label,
            "confidence": float(confidence),
            "frame_number": int(frame_num),
            "exported": False,
            "annotated": False,
            "exported_at": exported_at,
            "local_path": local_path,
            "hard_case_type": hard_case_type,
            "label_quality": label_quality,
            "teacher_confidence": float(teacher_confidence),
            "physics_ok": bool(physics_ok),
        }
        self._insert_supabase(metadata)
        self.export_counts[count_key] = current_count + 1
        self.exported_frames.append(metadata)
        return metadata

    def _upload_to_r2(self, local_path: str, object_key: str) -> bool:
        if not self.r2_client:
            logger.warning("⚠️ R2 no configurado; no se exporta frame training %s", object_key)
            return False
        try:
            with open(local_path, "rb") as file_stream:
                self.r2_client.upload_fileobj(
                    Fileobj=file_stream,
                    Bucket=self.r2_bucket,
                    Key=object_key,
                    ExtraArgs={"ContentType": "image/jpeg"},
                )
            return True
        except Exception as exc:
            logger.warning("⚠️ Upload R2 frame training falló (%s): %s", object_key, exc)
            return False

    def _insert_supabase(self, metadata: Dict[str, Any]) -> None:
        logger.info("🧪 Shadow mode: omitiendo inserción en training_frames.")

    def _build_r2_client(self):
        endpoint = os.getenv("AI_WORKER_V1_R2_ENDPOINT") or os.getenv("R2_ENDPOINT")
        access_key = os.getenv("AI_WORKER_V1_R2_WRITE_ACCESS_KEY_ID")
        secret_key = os.getenv("AI_WORKER_V1_R2_WRITE_SECRET_ACCESS_KEY")
        if not endpoint or not access_key or not secret_key:
            return None
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

    def _build_supabase_client(self):
        if create_client is None:
            return None
        url = os.getenv("SUPABASE_URL")
        key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SERVICE_KEY")
            or os.getenv("SUPABASE_KEY")
        )
        if not url or not key:
            return None
        try:
            return create_client(url, key)
        except Exception as exc:
            logger.warning("⚠️ Supabase no disponible para training_frames: %s", exc)
            return None

    def _safe_match_id(self, match_id: str) -> str:
        raw = str(match_id or "unknown")
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)

    def _safe_label(self, label: str) -> str:
        raw = str(label or "low_confidence").strip().lower()
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in raw)
