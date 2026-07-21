"""
Pipeline automático de reentrenamiento para ejecutar en Kaggle.

Flujo:
1. Lee training_frames donde exported=False.
2. Descarga imágenes desde R2.
3. Auto-anota con SAM 2.1 usando prompt de texto "soccer ball".
4. Sube imagen + etiqueta YOLO a Roboflow.
5. Re-entrena YOLO11n con dataset actualizado.
6. Sube best.pt a R2 y actualiza ai_memory.

Config por variables de entorno:
  SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
  R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET
  ROBOFLOW_API_KEY
  ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT
  DATASET_YAML, ROBOFLOW_DATASET_URL o ultima version Roboflow
"""

import argparse
import logging
import os
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import cv2
import numpy as np
import requests

try:
    from supabase import create_client
except ImportError:  # pragma: no cover
    create_client = None


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("OneFrame.AutoTrainer")


DEFAULT_ROBOFLOW_UPLOAD_URL = "https://api.roboflow.com/dataset/ONE-FRAMEV1/upload"
DEFAULT_ROBOFLOW_WORKSPACE = "ones-workspace-uglag"
DEFAULT_ROBOFLOW_PROJECT = "one-framev1"
DEFAULT_ROBOFLOW_DATASET_FORMAT = "yolov11"


class PseudoLabelFilter:
    """
    Filtra pseudo-labels por calidad antes de subir a Roboflow.
    Solo labels A y B entran al dataset de entrenamiento.

    A: label casi seguro (conf > 0.85) -> entra directo
    B: label probable (conf > 0.60) -> entra con peso reducido
    C: label dudoso -> NO entra, marca para revisión humana
    D: basura -> descarta completamente
    """

    THRESHOLDS = {
        "A": 0.85,
        "B": 0.60,
        "C": 0.40,
    }

    def evaluate(
        self,
        teacher_confidence: float,
        physics_ok: bool = True,
        hard_case_type: str = None,
    ) -> str:
        if not physics_ok:
            return "D"
        if teacher_confidence >= self.THRESHOLDS["A"] and physics_ok:
            return "A"
        elif teacher_confidence >= self.THRESHOLDS["B"]:
            return "B"
        elif teacher_confidence >= self.THRESHOLDS["C"]:
            return "C"
        else:
            return "D"

    def should_upload(self, quality: str) -> bool:
        """Solo A y B van a Roboflow."""
        return quality in ("A", "B")


@dataclass
class TrainingFrame:
    id: str
    match_id: str
    frame_path: str
    confidence: float
    frame_number: int
    physics_ok: bool = True
    hard_case_type: Optional[str] = None


class AutoTrainer:
    def __init__(self, work_dir: str = "/kaggle/working/oneframe_auto_training"):
        self.work_dir = Path(work_dir)
        self.frames_dir = self.work_dir / "frames"
        self.labels_dir = self.work_dir / "labels"
        self.models_dir = self.work_dir / "models"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.bucket = os.getenv("R2_BUCKET", "one-frame")
        self.roboflow_upload_url = os.getenv(
            "ROBOFLOW_UPLOAD_URL",
            DEFAULT_ROBOFLOW_UPLOAD_URL,
        )
        self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "")
        self.roboflow_workspace = os.getenv("ROBOFLOW_WORKSPACE", DEFAULT_ROBOFLOW_WORKSPACE)
        self.roboflow_project_id = os.getenv("ROBOFLOW_PROJECT", DEFAULT_ROBOFLOW_PROJECT)
        self.roboflow_dataset_format = os.getenv(
            "ROBOFLOW_DATASET_FORMAT",
            DEFAULT_ROBOFLOW_DATASET_FORMAT,
        )
        self.supabase = self._build_supabase_client()
        self.r2 = self._build_r2_client()
        self.sam_model = None
        self.roboflow_project = None

    def run(self, limit: int = 200, min_new_frames: int = 50, epochs: int = 150) -> Dict[str, Any]:
        frames = self.fetch_new_frames(limit=limit)
        logger.info("📦 Frames nuevos encontrados: %s", len(frames))
        uploaded = self.annotate_and_upload(frames)

        result: Dict[str, Any] = {
            "frames_found": len(frames),
            "uploaded_to_roboflow": uploaded,
            "trained": False,
        }
        if uploaded < min_new_frames:
            logger.info("ℹ️ Se omite training: %s/%s frames nuevos", uploaded, min_new_frames)
            return result

        best_path = self.train_yolo(epochs=epochs)
        model_key, model_version = self.upload_model(best_path)
        self.update_ai_memory(model_key, model_version)
        result.update(
            {
                "trained": True,
                "model_path": model_key,
                "model_version": model_version,
            }
        )
        return result

    def fetch_new_frames(self, limit: int = 200) -> List[TrainingFrame]:
        if not self.supabase:
            raise RuntimeError("Supabase no configurado.")

        response = (
            self.supabase.table("training_frames")
            .select(
                "id, match_id, frame_path, confidence, frame_number, "
                "physics_ok, hard_case_type"
            )
            .eq("exported", False)
            .limit(limit)
            .execute()
        )
        rows = response.data or []
        return [
            TrainingFrame(
                id=str(row["id"]),
                match_id=str(row["match_id"]),
                frame_path=str(row["frame_path"]),
                confidence=float(row.get("confidence") or 0.0),
                frame_number=int(row.get("frame_number") or 0),
                physics_ok=bool(row.get("physics_ok", True)),
                hard_case_type=row.get("hard_case_type"),
            )
            for row in rows
        ]

    def annotate_and_upload(self, frames: List[TrainingFrame]) -> int:
        label_filter = PseudoLabelFilter()
        uploaded_count = 0
        total = len(frames)
        for item in frames:
            image_path = self.download_frame(item)
            if not image_path:
                continue
            annotation_result = self.generate_yolo_annotation_result(image_path)
            if not annotation_result:
                self.mark_frame(
                    item.id,
                    exported=False,
                    annotated=False,
                    label_quality="D",
                    teacher_confidence=0.0,
                )
                logger.info("Sin anotación útil para %s", item.frame_path)
                continue

            annotation, sam_confidence = annotation_result
            quality = label_filter.evaluate(
                teacher_confidence=sam_confidence,
                physics_ok=item.physics_ok,
                hard_case_type=item.hard_case_type,
            )
            self.mark_frame(
                item.id,
                exported=False,
                annotated=True,
                label_quality=quality,
                teacher_confidence=sam_confidence,
            )
            if not annotation:
                logger.info("Sin anotación útil para %s", item.frame_path)
                continue
            label_path = self.write_label(image_path, annotation)
            if label_filter.should_upload(quality):
                if self.upload_to_roboflow(image_path, label_path, item):
                    self.mark_frame(
                        item.id,
                        exported=True,
                        annotated=True,
                        label_quality=quality,
                        teacher_confidence=sam_confidence,
                    )
                    uploaded_count += 1
            else:
                logger.info(
                    "Frame %s descartado por label_quality=%s teacher_confidence=%.3f",
                    item.frame_path,
                    quality,
                    sam_confidence,
                )
        logger.info(f"Frames subidos a Roboflow: {uploaded_count}")
        logger.info(f"Frames descartados (C/D): {total - uploaded_count}")
        return uploaded_count

    def download_frame(self, item: TrainingFrame) -> Optional[Path]:
        if not self.r2:
            raise RuntimeError("R2 no configurado.")
        local_path = self.frames_dir / Path(item.frame_path).name
        try:
            self.r2.download_file(self.bucket, item.frame_path, str(local_path))
            return local_path
        except Exception as exc:
            logger.warning("⚠️ No se pudo descargar %s: %s", item.frame_path, exc)
            return None

    def generate_yolo_annotation(self, image_path: Path) -> Optional[Tuple[float, float, float, float]]:
        result = self.generate_yolo_annotation_result(image_path)
        if result is None:
            return None
        annotation, _confidence = result
        return annotation

    def generate_yolo_annotation_result(
        self,
        image_path: Path,
    ) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
        image = cv2.imread(str(image_path))
        if image is None:
            return None

        sam_result = self._sam_soccer_ball_object(image_path, image)
        if sam_result is not None:
            sam_box, sam_confidence = sam_result
            return self._box_to_yolo(sam_box, image.shape[1], image.shape[0]), sam_confidence

        return None

    def _sam_soccer_ball_object(
        self,
        image_path: Path,
        image: np.ndarray,
    ) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        try:
            model = self._get_sam_model()
            results = model(str(image_path), texts=["soccer ball"], verbose=False)
            candidates = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    for detected_box in boxes:
                        xyxy = getattr(detected_box, "xyxy", None)
                        if xyxy is None:
                            continue
                        x1, y1, x2, y2 = xyxy[0].tolist()
                        box = (
                            int(max(0, x1)),
                            int(max(0, y1)),
                            int(min(image.shape[1] - 1, x2)),
                            int(min(image.shape[0] - 1, y2)),
                        )
                        if self._is_ball_size_box(box):
                            confidence = self._box_confidence(detected_box)
                            candidates.append((box, confidence))
                masks = getattr(result, "masks", None)
                if masks is None or masks.xy is None:
                    continue
                for polygon in masks.xy:
                    box = self._polygon_to_box(polygon)
                    if box and self._is_ball_size_box(box):
                        candidates.append((box, 0.0))
            if candidates:
                return min(
                    candidates,
                    key=lambda candidate: (
                        (candidate[0][2] - candidate[0][0])
                        * (candidate[0][3] - candidate[0][1])
                    ),
                )
        except Exception as exc:
            logger.info("SAM soccer ball no generó anotación usable: %s", exc)
        return None

    def _box_confidence(self, detected_box) -> float:
        conf = getattr(detected_box, "conf", None)
        if conf is None:
            return 0.0
        try:
            if hasattr(conf, "item"):
                return float(conf.item())
            if hasattr(conf, "__len__") and len(conf) > 0:
                first = conf[0]
                return float(first.item() if hasattr(first, "item") else first)
            return float(conf)
        except Exception:
            return 0.0

    def _get_sam_model(self):
        if self.sam_model is None:
            from ultralytics import SAM

            model_name = os.getenv("SAM_MODEL", "sam2.1_b.pt")
            self.sam_model = SAM(model_name)
        return self.sam_model

    def _heuristic_small_circle(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=80,
            param2=12,
            minRadius=1,
            maxRadius=40,
        )
        if circles is None:
            return None
        h, w = image.shape[:2]
        candidates = []
        for x, y, radius in np.round(circles[0, :]).astype("int"):
            size = radius * 2
            if 2 <= size <= 80:
                x1 = max(0, x - radius)
                y1 = max(0, y - radius)
                x2 = min(w - 1, x + radius)
                y2 = min(h - 1, y + radius)
                candidates.append((x1, y1, x2, y2))
        if not candidates:
            return None
        center_x = w / 2
        center_y = h / 2
        return min(
            candidates,
            key=lambda box: abs(((box[0] + box[2]) / 2) - center_x)
            + abs(((box[1] + box[3]) / 2) - center_y),
        )

    def write_label(self, image_path: Path, annotation: Tuple[float, float, float, float]) -> Path:
        label_path = self.labels_dir / f"{image_path.stem}.txt"
        cx, cy, width, height = annotation
        label_path.write_text(f"0 {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")
        return label_path

    def upload_to_roboflow(self, image_path: Path, label_path: Path, item: TrainingFrame) -> bool:
        if not self.roboflow_api_key:
            raise RuntimeError("Falta ROBOFLOW_API_KEY.")
        try:
            project = self._get_roboflow_project()
            project.upload(
                str(image_path),
                annotation_path=str(label_path),
                annotation_labelmap={"0": "ball"},
                split="train",
                batch_name=f"oneframe-{item.match_id}",
                tag_names=[item.match_id, "auto_sam", f"frame_{item.frame_number}"],
                num_retry_uploads=3,
            )
            return True
        except Exception as exc:
            logger.warning(
                "Roboflow SDK upload fallo para %s: %s. Probando REST fallback.",
                image_path,
                exc,
            )
            return self._upload_to_roboflow_rest(image_path, label_path, item)

    def _upload_to_roboflow_rest(self, image_path: Path, label_path: Path, item: TrainingFrame) -> bool:
        params = {
            "api_key": self.roboflow_api_key,
            "name": image_path.name,
            "split": "train",
        }
        try:
            with image_path.open("rb") as image_file:
                upload_response = requests.post(
                    self.roboflow_upload_url,
                    params=params,
                    files={"file": (image_path.name, image_file, "image/jpeg")},
                    timeout=120,
                )
            if upload_response.status_code >= 400:
                logger.warning(
                    "Roboflow REST image upload fallo %s: %s",
                    upload_response.status_code,
                    upload_response.text[:500],
                )
                return False

            payload = upload_response.json()
            image_id = (
                payload.get("id")
                or payload.get("imageId")
                or (payload.get("image") or {}).get("id")
            )
            if not image_id:
                logger.warning("Roboflow REST no devolvio image_id: %s", payload)
                return False

            annotate_url = f"https://api.roboflow.com/dataset/{self.roboflow_project_id}/annotate/{image_id}"
            annotation_response = requests.post(
                annotate_url,
                params={"api_key": self.roboflow_api_key, "name": label_path.name},
                json={
                    "annotationFile": label_path.read_text(),
                    "labelmap": {"0": "ball"},
                    "match_id": item.match_id,
                    "frame_number": item.frame_number,
                },
                timeout=120,
            )
            if annotation_response.status_code >= 400:
                logger.warning(
                    "Roboflow REST annotation upload fallo %s: %s",
                    annotation_response.status_code,
                    annotation_response.text[:500],
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Roboflow REST upload error %s: %s", image_path, exc)
            return False

    def train_yolo(self, epochs: int = 150) -> Path:
        dataset_yaml = self.prepare_dataset()
        try:
            from ultralytics import YOLO

            model = YOLO("yolo11n.pt")
            result = model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                imgsz=640,
                batch=-1,
                device=0,
                project=str(self.work_dir / "runs"),
                name="oneframe_retrain",
                exist_ok=True,
            )
            best_path = Path(result.save_dir) / "weights" / "best.pt"
        except Exception as exc:
            logger.warning("Ultralytics train directo falló, probando CLI: %s", exc)
            cmd = [
                "yolo",
                "detect",
                "train",
                "model=yolo11n.pt",
                f"data={dataset_yaml}",
                f"epochs={epochs}",
                "imgsz=640",
                "batch=-1",
                "device=0",
            ]
            subprocess.run(cmd, check=True)
            best_path = Path("runs/detect/train/weights/best.pt")

        if not best_path.exists():
            raise RuntimeError(f"No se encontró best.pt en {best_path}")
        return best_path

    def prepare_dataset(self) -> Path:
        dataset_yaml = os.getenv("DATASET_YAML")
        if dataset_yaml and Path(dataset_yaml).exists():
            return Path(dataset_yaml)

        dataset_url = os.getenv("ROBOFLOW_DATASET_URL")
        if dataset_url:
            return self._download_dataset_from_url(dataset_url)

        return self._download_latest_roboflow_dataset()

    def _download_dataset_from_url(self, dataset_url: str) -> Path:
        dataset_zip = self.work_dir / "roboflow_dataset.zip"
        response = requests.get(dataset_url, timeout=300)
        response.raise_for_status()
        dataset_zip.write_bytes(response.content)
        if not zipfile.is_zipfile(dataset_zip):
            raise RuntimeError(
                "Dataset de Roboflow invalido/corrupto: la respuesta no es un ZIP usable."
            )
        dataset_dir = self.work_dir / "dataset"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        shutil.unpack_archive(str(dataset_zip), str(dataset_dir))
        yaml_files = list(dataset_dir.rglob("data.yaml"))
        if not yaml_files:
            raise RuntimeError("Dataset de Roboflow no contiene data.yaml.")
        return yaml_files[0]

    def _download_latest_roboflow_dataset(self) -> Path:
        project = self._get_roboflow_project()
        versions = project.versions()
        if not versions:
            raise RuntimeError("Roboflow no tiene versiones de dataset disponibles.")

        latest = versions[0]
        logger.info("Descargando ultima version Roboflow: %s", latest)
        dataset = latest.download(self.roboflow_dataset_format)
        dataset_dir = Path(dataset.location)
        yaml_files = list(dataset_dir.rglob("data.yaml"))
        if not yaml_files:
            raise RuntimeError("Ultima version de Roboflow no contiene data.yaml.")
        return yaml_files[0]

    def upload_model(self, best_path: Path) -> Tuple[str, str]:
        if not self.r2:
            raise RuntimeError("R2 no configurado.")
        timestamp = int(time.time())
        model_version = f"oneframe_v{timestamp}"
        model_key = f"models/{model_version}.pt"
        self.r2.upload_file(str(best_path), self.bucket, model_key, ExtraArgs={"ContentType": "application/octet-stream"})
        logger.info("✅ Modelo subido a R2: %s", model_key)
        return model_key, model_version

    def update_ai_memory(self, model_key: str, model_version: str) -> None:
        if not self.supabase:
            raise RuntimeError("Supabase no configurado.")
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rows = [
            {
                "memory_key": "model_version",
                "memory_value": model_version,
                "description": "Última versión YOLO reentrenada automáticamente",
                "updated_at": now,
            },
            {
                "memory_key": "model_path",
                "memory_value": model_key,
                "description": "Ruta R2 del último best.pt aprobado",
                "updated_at": now,
            },
        ]
        for row in rows:
            self.supabase.table("ai_memory").upsert(row, on_conflict="memory_key").execute()

    def mark_frame(
        self,
        frame_id: str,
        exported: bool,
        annotated: bool,
        label_quality: Optional[str] = None,
        teacher_confidence: Optional[float] = None,
    ) -> None:
        if not self.supabase:
            return
        update = {"exported": exported, "annotated": annotated}
        if label_quality is not None:
            update["label_quality"] = label_quality
        if teacher_confidence is not None:
            update["teacher_confidence"] = float(teacher_confidence)
        self.supabase.table("training_frames").update(
            update
        ).eq("id", frame_id).execute()

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
        return create_client(url, key)

    def _build_r2_client(self):
        endpoint = os.getenv("R2_ENDPOINT")
        access_key = os.getenv("R2_ACCESS_KEY_ID")
        secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
        if not endpoint or not access_key or not secret_key:
            return None
        return boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

    def _get_roboflow_project(self):
        if self.roboflow_project is not None:
            return self.roboflow_project
        if not self.roboflow_api_key:
            raise RuntimeError("Falta ROBOFLOW_API_KEY.")
        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise RuntimeError("Falta instalar roboflow: pip install roboflow") from exc

        rf = Roboflow(api_key=self.roboflow_api_key)
        self.roboflow_project = rf.workspace(self.roboflow_workspace).project(
            self.roboflow_project_id
        )
        return self.roboflow_project

    def _polygon_to_box(self, polygon: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        if polygon is None or len(polygon) == 0:
            return None
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _is_small_circular_box(self, box: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        size = max(width, height)
        if size < 2 or size > 80:
            return False
        ratio = width / max(height, 1)
        return 0.65 <= ratio <= 1.45

    def _is_ball_size_box(self, box: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = box
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        size = max(width, height)
        return 2 <= size <= 80 and min(width, height) >= 1

    def _box_to_yolo(
        self,
        box: Tuple[int, int, int, int],
        image_w: int,
        image_h: int,
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2) / max(image_w, 1)
        cy = ((y1 + y2) / 2) / max(image_h, 1)
        width = (x2 - x1) / max(image_w, 1)
        height = (y2 - y1) / max(image_h, 1)
        return cx, cy, width, height


def main():
    parser = argparse.ArgumentParser(description="OneFrame auto trainer")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-new-frames", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--work-dir", default="/kaggle/working/oneframe_auto_training")
    args = parser.parse_args()

    trainer = AutoTrainer(work_dir=args.work_dir)
    result = trainer.run(
        limit=args.limit,
        min_new_frames=args.min_new_frames,
        epochs=args.epochs,
    )
    logger.info("Resultado: %s", result)


if __name__ == "__main__":
    main()
