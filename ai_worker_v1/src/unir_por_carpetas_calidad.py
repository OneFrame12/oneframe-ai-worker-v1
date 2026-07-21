import glob
import os
import subprocess
import tempfile

# ================= CONFIGURACION =================
BASE_DIR = "PROCESADO"
OUTPUT_FILE = "RESUMEN_FINAL_ESTABLE.mp4"
CARPETAS_PRIORIDAD = ["GOL", "TIRO PELIGROSO", "TIRO AL ARCO", "ATAJADA"]
# =================================================


def _escape_concat_path(path: str) -> str:
    return path.replace("'", "'\\''")


def _build_concat_manifest(video_paths):
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8",
    )
    try:
        for video_path in video_paths:
            temp_file.write(f"file '{_escape_concat_path(os.path.abspath(video_path))}'\n")
    finally:
        temp_file.close()
    return temp_file.name


def main():
    print("🛡️ INICIANDO FUSION SEGURA CON FFMPEG...")

    if not os.path.exists(BASE_DIR):
        print(f"❌ No encuentro la carpeta '{BASE_DIR}'. ¿Descomprimiste el ZIP?")
        return

    clips_para_unir = []

    for carpeta in CARPETAS_PRIORIDAD:
        path = os.path.join(BASE_DIR, carpeta)
        if not os.path.exists(path):
            continue

        archivos = sorted(glob.glob(os.path.join(path, "*.mp4")))
        for archivo in archivos:
            if os.path.exists(archivo) and os.path.getsize(archivo) > 0:
                print(f"  Load: {os.path.basename(archivo)}")
                clips_para_unir.append(archivo)

    if not clips_para_unir:
        print("❌ No encontré videos válidos.")
        return

    print(f"\n🧩 Uniendo {len(clips_para_unir)} clips con FFmpeg...")
    manifest_path = _build_concat_manifest(clips_para_unir)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            manifest_path,
            "-r",
            "30",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            OUTPUT_FILE,
        ]
        subprocess.run(cmd, check=True)
        print(f"\n✅ ¡LISTO! Video guardado: {OUTPUT_FILE}")
    except subprocess.CalledProcessError as exc:
        print(f"❌ Error fatal concatenando clips: {exc}")
    finally:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)


if __name__ == "__main__":
    main()
