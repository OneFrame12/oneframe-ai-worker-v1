print("🟢 INICIANDO SCRIPT DE DEPURACIÓN (MINUTOS)...")
import os
import sys

# Verificación rápida de archivo
VIDEO_FILE = "Pichanga.Corto.mp4" 

if not os.path.exists(VIDEO_FILE):
    print(f"❌ ERROR: No encuentro el video '{VIDEO_FILE}'")
    print(f"📂 Carpeta actual: {os.getcwd()}")
    sys.exit()

print("📚 Importando librerías...")
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile

from scipy.io import wavfile

# Configuración
STA_WINDOW = 0.5   
LTA_WINDOW = 10.0  
THRESHOLD = 2.5    

def extraer_audio_ffmpeg(video_path, target_sr=44100):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_file.close()
    temp_wav = temp_file.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", str(target_sr), "-ac", "1",
            "-loglevel", "error", temp_wav,
        ]
        subprocess.run(cmd, check=True)
        fs, audio_arr = wavfile.read(temp_wav)
        audio_arr = audio_arr.astype(np.float32) / 32768.0
        return audio_arr, fs
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

def analizar_y_graficar(video_path):
    print(f"🎧 Leyendo audio de: {video_path}...")
    
    try:
        audio_arr, fps = extraer_audio_ffmpeg(video_path)
        step = 10 

        audio_arr = audio_arr[::step]
        fs_real = fps / step
        
        print("📊 Calculando Energía STA/LTA...")
        energy = audio_arr ** 2
        
        n_sta = int(STA_WINDOW * fs_real)
        n_lta = int(LTA_WINDOW * fs_real)
        
        if len(energy) < n_lta:
            print("❌ El video es demasiado corto para el análisis LTA (min 10 seg).")
            return

        sta = np.convolve(energy, np.ones(n_sta)/n_sta, mode='same')
        lta = np.convolve(energy, np.ones(n_lta)/n_lta, mode='same')
        
        lta[lta < 1e-6] = 1e-6 
        ratio = sta / lta
        
        # --- EJE DE TIEMPO EN MINUTOS ---
        duration_min = (len(audio_arr) / max(fs_real, 1)) / 60
        time_axis = np.linspace(0, duration_min, len(ratio))
        # -------------------------------
        
        print("🎨 Generando gráfico...")
        plt.figure(figsize=(12, 6))
        
        # Graficar la señal
        plt.plot(time_axis, ratio, label='Intensidad Emoción (STA/LTA)', color='#1f77b4', linewidth=0.8)
        
        # Graficar el Umbral
        plt.axhline(y=THRESHOLD, color='r', linestyle='--', label=f'Umbral Actual ({THRESHOLD})')
        
        plt.title(f'Análisis de Audio: {video_path}', fontsize=14)
        plt.xlabel('Tiempo (Minutos)', fontsize=12) 
        plt.ylabel('Ratio (Grito / Ambiente)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 10) 
        
        print("✅ ¡Abriendo ventana de gráfico!")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"❌ Ocurrió un error: {e}")

if __name__ == "__main__":
    analizar_y_graficar(VIDEO_FILE)
