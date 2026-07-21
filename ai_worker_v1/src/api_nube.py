"""
api_nube.py - OneFrame Cloud API
Servidor Flask para pre-procesar videos de partidos desde Google Drive.
Corre en Ubuntu con GPU (Lambda Labs) en el puerto 5000.
"""

import os
import io
import base64
import subprocess
import tempfile

import cv2
import gdown
import librosa
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla (headless server)
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, jsonify

# ==========================================
# CONFIGURACIÓN DEL SERVIDORdocker build -f Dockerfile.runpod -t oneframe-handler .

# ==========================================

app = Flask(__name__)

TEMP_DIR = "/tmp/oneframe_temp"
os.makedirs(TEMP_DIR, exist_ok=True)


# ==========================================
# UTILIDADES
# ==========================================

def _limpiar_archivos(*rutas):
    """Elimina archivos temporales del disco de forma segura."""
    for ruta in rutas:
        if ruta and os.path.exists(ruta):
            try:
                os.remove(ruta)
                print(f"🗑️  Eliminado: {ruta}")
            except OSError as e:
                print(f"⚠️ No se pudo eliminar {ruta}: {e}")


def _extraer_frame_b64(ruta_video: str, minuto: int = 20) -> str:
    """
    Usa OpenCV para saltar al minuto indicado y extrae ese frame.
    Codifica la imagen a Base64 directamente en memoria (sin escribir al disco).
    Retorna el string Base64 o lanza una excepción.
    """
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {ruta_video}")

    milisegundos = minuto * 60 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, milisegundos)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError(f"No se pudo extraer el frame en el minuto {minuto}.")

    # Codificación en memoria: frame → JPEG bytes → Base64
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise RuntimeError("Fallo al codificar el frame como JPEG.")

    foto_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return foto_b64


def _analizar_audio_y_grafico(ruta_video: str) -> str:
    """
    Extrae el audio con FFmpeg, analiza la energía RMS con Librosa
    y genera un gráfico en estilo oscuro con un umbral en el percentil 95.
    Devuelve el gráfico como string Base64 (en memoria, sin escribir al disco).
    Maneja el caso donde el video no tiene pista de audio.
    """
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False,
        dir=TEMP_DIR,
        prefix="audio_temp_",
    )
    temp_file.close()
    ruta_audio_temp = temp_file.name

    try:
        # --- Extracción de audio con FFmpeg ---
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            ruta_video,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "11025",
            "-ac",
            "1",
            "-loglevel",
            "error",
            ruta_audio_temp,
        ]
        subprocess.run(cmd, check=True)

        if not os.path.exists(ruta_audio_temp) or os.path.getsize(ruta_audio_temp) == 0:
            raise ValueError("No se pudo extraer una pista de audio valida del video.")

        # --- Análisis con Librosa ---
        # sr=11025 es suficiente para análisis de energía y es rápido
        y, sr = librosa.load(ruta_audio_temp, sr=11025, mono=True)
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        tiempos = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=512
        )
        umbral_pico = np.percentile(rms, 95)

    finally:
        # Siempre limpiar el audio temporal, incluso si falla analytics
        _limpiar_archivos(ruta_audio_temp)

    # --- Gráfico con Matplotlib (estilo oscuro) ---
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    ax.plot(tiempos, rms, color="#00ff88", linewidth=0.8, alpha=0.9)
    ax.fill_between(tiempos, rms, alpha=0.15, color="#00ff88")
    ax.axhline(
        y=umbral_pico,
        color="#ff4444",
        linestyle="--",
        linewidth=1.5,
        label=f"Umbral P95 (Goles/Silbatos): {umbral_pico:.4f}",
    )

    ax.set_xlabel("Tiempo (segundos)", color="white", fontsize=11)
    ax.set_ylabel("Amplitud RMS", color="white", fontsize=11)
    ax.set_title("📊 Análisis de Audio — Picos de Energía del Partido", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_color("#444444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#2a2a2a", edgecolor="#555555", labelcolor="white", fontsize=10)

    plt.tight_layout()

    # Guardar el gráfico en memoria (BytesIO) → Base64
    buffer_grafico = io.BytesIO()
    plt.savefig(buffer_grafico, format="png", facecolor="#1e1e1e", dpi=100)
    plt.close(fig)
    buffer_grafico.seek(0)
    grafico_b64 = base64.b64encode(buffer_grafico.read()).decode("utf-8")

    return grafico_b64


# ==========================================
# ENDPOINT PRINCIPAL
# ==========================================

@app.route("/api/pre-procesar", methods=["POST"])
def pre_procesar():
    """
    POST /api/pre-procesar
    Body JSON: { "link_video": "<Google Drive share URL>" }

    Descarga el video, extrae un frame y analiza el audio.
    Retorna foto_b64 y grafico_b64 como strings Base64.
    Limpia todos los archivos temporales al terminar.
    """
    datos = request.get_json(silent=True)
    if not datos or "link_video" not in datos:
        return jsonify({"status": "error", "mensaje": "Falta la llave 'link_video' en el JSON."}), 400

    link_video = datos["link_video"]
    ruta_video_temp = os.path.join(TEMP_DIR, "video_temp.mp4")

    print(f"\n📥 Solicitud recibida para: {link_video}")

    try:
        # --- 1. Descargar video desde Google Drive ---
        print("⏳ Descargando video de Google Drive con gdown...")
        gdown.download(link_video, ruta_video_temp, quiet=False, fuzzy=True)

        if not os.path.exists(ruta_video_temp) or os.path.getsize(ruta_video_temp) == 0:
            raise FileNotFoundError("gdown no pudo descargar el archivo. Verifica que el link sea público.")

        print(f"✅ Video descargado ({os.path.getsize(ruta_video_temp) / 1e6:.1f} MB)")

        # --- 2. Extraer frame del minuto 20 ---
        print("🖼️  Extrayendo frame del minuto 20...")
        foto_b64 = _extraer_frame_b64(ruta_video_temp, minuto=20)
        print("✅ Frame extraído y codificado en Base64.")

        # --- 3. Analizar audio y generar gráfico ---
        print("🔊 Analizando audio y generando gráfico...")
        try:
            grafico_b64 = _analizar_audio_y_grafico(ruta_video_temp)
            print("✅ Gráfico de audio generado.")
        except ValueError as e:
            # El video no tiene audio: devolvemos string vacío, no es un error fatal
            print(f"⚠️ Sin audio: {e}")
            grafico_b64 = ""

        return jsonify({
            "status": "success",
            "foto_b64": foto_b64,
            "grafico_b64": grafico_b64,
        }), 200

    except FileNotFoundError as e:
        print(f"❌ Error de descarga: {e}")
        return jsonify({"status": "error", "mensaje": str(e)}), 422

    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return jsonify({"status": "error", "mensaje": f"Error interno: {str(e)}"}), 500

    finally:
        # --- 4. Limpieza garantizada del video pesado ---
        _limpiar_archivos(ruta_video_temp)
        print("🧹 Video temporal eliminado del disco.\n")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/api/ping", methods=["GET"])
def ping():
    """Ruta de diagnóstico para confirmar que el servidor está vivo."""
    return jsonify({"status": "ok", "mensaje": "OneFrame Cloud API activa ✅"}), 200


# ==========================================
# ARRANQUE DEL SERVIDOR
# ==========================================

if __name__ == "__main__":
    print("========================================")
    print("   🚀 ONEFRAME CLOUD API — Iniciando   ")
    print("========================================")
    print(f"   📂 Directorio temporal: {TEMP_DIR}")
    print(f"   🌐 Escuchando en: 0.0.0.0:5000")
    print("========================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

