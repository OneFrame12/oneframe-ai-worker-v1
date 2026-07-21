"""
OneFrame V57 - Main Orchestrator
Lambda Edition: Integración Lógica Francotirador + JSON Ready + Audio Ultrarrápido
"""
import os
import sys
import logging
import argparse
import subprocess
import time
import queue
import threading
from typing import List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.io import wavfile

# Importar tus módulos locales
from config import (
    vision_cfg, game_cfg, storage_cfg, processing_cfg,
    validate_config, get_clip_timing, calibracion_data
)
from engine import VisionEngine, GameReferee

# ==========================================
# 📝 LOGGING SETUP
# ==========================================

logging.basicConfig(
    level=getattr(logging, processing_cfg.LOG_LEVEL),
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OneFrame.Main")

# ==========================================
# 🎬 CLIP MANAGER (ASINCRONO)
# ==========================================

class ClipGenerationError(Exception):
    pass

class AsyncClipManager:
    def __init__(self):
        self.out_dir = storage_cfg.OUTPUT_CLIPS_DIR
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.queue = queue.Queue()
        self.running = True
        self.error = None 
        self.clip_counter = 0  
        
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"✅ AsyncClipManager started (output: {self.out_dir})")
    
    def request_clip(self, video_path: str, timestamp: float, event_type: str,
                     cloud_manager, remote_path: str):
        if self.error:
            raise ClipGenerationError(f"Previous clip failed: {self.error}")
        
        self.clip_counter += 1
        
        task = {
            'video_path': video_path,
            'timestamp': timestamp,
            'event_type': event_type,
            'clip_index': self.clip_counter, 
            'cloud': cloud_manager,
            'remote': remote_path
        }
        self.queue.put(task)
    
    def _worker_loop(self):
        while self.running:
            try:
                task = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            try:
                self._generate_clip(task)
            except Exception as e:
                self.error = str(e)
                logger.critical(f"💀 CRITICAL: Clip generation failed: {e}")
                self.running = False
            finally:
                self.queue.task_done()
    
    def _generate_clip(self, task: dict):
        video_path = task['video_path']
        timestamp = task['timestamp']
        event_type = task['event_type']
        idx = task['clip_index']
        
        pre_sec, post_sec = get_clip_timing(event_type)
        start_time = max(0, timestamp - pre_sec)
        duration = (timestamp + post_sec) - start_time
        
        segundos_totales = int(timestamp)
        minutos = segundos_totales // 60
        segundos = segundos_totales % 60
        
        filename = os.path.join(self.out_dir, f"min_{minutos:02d}_sec_{segundos:02d}_{event_type}.mp4")
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_time),   
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",        
            "-preset", "ultrafast",   
            "-c:a", "aac",            
            "-loglevel", "error",
            filename
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, check=True, text=True)
            logger.info(f"✅ Clip generated: {os.path.basename(filename)}")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate clip {idx}: {e.stderr}")
        
        if task['cloud'] and task['remote']:
            try:
                remote_full = f"{task['remote']}/{os.path.basename(filename)}"
                task['cloud'].upload_sync(filename, remote_full)
            except Exception as e:
                logger.error(f"☁️ Upload failed: {e}")
    
    def wait_completion(self):
        self.queue.join()
        if self.error:
            raise ClipGenerationError(f"Clip generation failed: {self.error}")
    
    def shutdown(self):
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)

# ==========================================
# ☁️ CLOUD MANAGER & AUDIO ULTRARRÁPIDO
# ==========================================

class CloudManager:
    def __init__(self):
        self.client = None
    def upload_sync(self, local_path: str, cloud_path: str):
        pass

class AudioAnalyzer:
    """
    Radar Acústico de Doble Banda (Versión Ultrarrápida con FFmpeg)
    """
    BANDS = {
        "MULTITUD": (300,  1_200),
        "SILBATO":  (3_000, 4_000),
    }

    @staticmethod
    def _butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 4):
        nyq = fs / 2.0
        low  = max(low_hz  / nyq, 1e-4)
        high = min(high_hz / nyq, 1.0 - 1e-4)
        b, a = butter(order, [low, high], btype="band")
        return b, a

    @staticmethod
    def _sta_lta_ratio(energy: np.ndarray, n_sta: int, n_lta: int) -> np.ndarray:
        sta = np.convolve(energy, np.ones(n_sta) / n_sta, mode="same")
        lta = np.convolve(energy, np.ones(n_lta) / n_lta, mode="same")
        lta[lta < 1e-6] = 1e-6
        return sta / lta

    @classmethod
    def analyze(cls, video_path: str) -> List[Tuple[float, float]]:
        logger.info("🔊 [AUDIO] Iniciando Extracción ultrarrápida con FFmpeg...")
        temp_wav = "temp_audio_analysis.wav"
        
        try:
            cmd = [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "11025", "-ac", "1",
                "-loglevel", "error", temp_wav
            ]
            subprocess.run(cmd, check=True)
            
            fs, arr = wavfile.read(temp_wav)
            arr = arr.astype(np.float64)
            logger.info(f"   📡 Audio cargado: {len(arr)/fs:.1f}s @ {fs} Hz")

            n_sta = max(1, int(game_cfg.STA_WINDOW_SEC * fs))
            n_lta = max(1, int(game_cfg.LTA_WINDOW_SEC * fs))

            mask_global = np.zeros(len(arr), dtype=bool)

            for band_name, (low_hz, high_hz) in cls.BANDS.items():
                b, a = cls._butter_bandpass(low_hz, high_hz, fs)
                arr_filt = filtfilt(b, a, arr)
                energy = arr_filt ** 2
                ratio = cls._sta_lta_ratio(energy, n_sta, n_lta)

                mask_band = ratio > game_cfg.STA_LTA_THRESHOLD
                mask_global |= mask_band

            diffs  = np.diff(mask_global.astype(int))
            starts = np.where(diffs == 1)[0]

            events = [(s / fs, (s / fs) + 1.0) for s in starts]
            logger.info(f"   ✅ [AUDIO] {len(events)} evento(s) detectado(s) en total.")
            return events

        except Exception as e:
            logger.warning(f"⚠️ [AUDIO] Análisis falló: {e}")
            return []
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

# ==========================================
# 🚀 MAIN ORCHESTRATOR
# ==========================================

def process_video(video_path: str, match_id: str, clip_manager: AsyncClipManager, cloud_manager: CloudManager):
    logger.info(f"\n{'='*60}")
    logger.info(f"📹 Processing: {video_path}")
    logger.info(f"{'='*60}\n")
    
    # 1. Analizar Audio
    audio_events = AudioAnalyzer.analyze(video_path)
    
    # 2. Visión y Tracking
    vision = VisionEngine()
    trajectory = vision.process_video(video_path)
    
    # 3. Cerebro Francotirador
    referee = GameReferee(
        resolution=trajectory.resolution,
        fps=trajectory.fps,
        audio_events=audio_events
    )
    
    clip_events, telemetry, detection_table = referee.process_trajectory(
        trajectory,
        vision.raw_detections,
    )

    logger.info(f"\n🎬 Generating {len(clip_events)} clips...")
    logger.info(f"📈 Telemetry points exported: {len(telemetry)}")
    logger.info(f"🩺 Detection table rows: {len(detection_table)}")
    
    # 4. Exportar
    for timestamp, event_type in clip_events:
        clip_manager.request_clip(
            video_path=video_path,
            timestamp=timestamp,
            event_type=event_type,
            cloud_manager=cloud_manager,
            remote_path=f"matches/{match_id}"
        )
    
    clip_manager.wait_completion()
    logger.info(f"✅ Video processing complete: {video_path}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partido", type=str, default=processing_cfg.MATCH_ID)
    parser.add_argument("--match-id", type=str, default=None)  # ← agregar esta línea
    parser.add_argument("--camara", type=str, default=game_cfg.CAMERA_TYPE)
    args = parser.parse_args()
    
    # --match-id tiene prioridad sobre --partido si se pasa
    processing_cfg.MATCH_ID = args.match_id if args.match_id else args.partido
    game_cfg.CAMERA_TYPE = args.camara
    
    logger.info("🚀 OneFrame V57 - Lógica Francotirador & JSON Ready")
    if calibracion_data:
        logger.info("🗺️  Calibración manual detectada (calibracion.json cargado)")
    else:
        logger.warning("⚠️  No hay calibracion.json. Usando coordenadas por defecto.")
    
    try:
        validate_config()
    except ValueError as e:
        logger.critical(f"💀 Config Error: {e}")
        sys.exit(1)
    
    clip_manager = AsyncClipManager()
    cloud_manager = CloudManager()
    
    try:
        for video_file in processing_cfg.VIDEO_FILES:
            if not os.path.exists(video_file):
                logger.error(f"❌ Video not found: {video_file}")
                continue
            process_video(video_file, args.partido, clip_manager, cloud_manager)
        
        logger.info("🏁 ALL DONE!")
        logger.info(f"📂 TUS CLIPS ESTÁN EN: {storage_cfg.OUTPUT_CLIPS_DIR}")
        
    except Exception as e:
        logger.exception(f"❌ Critical Error: {e}")
        sys.exit(1)
    finally:
        clip_manager.shutdown()

if __name__ == "__main__":
    main()
