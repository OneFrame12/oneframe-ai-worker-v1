import os
import re
from fabric import Connection
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt

# ================= CONFIGURACIÓN FIJA =================
KEY_PATH = "/Users/lucasvalenzuela/Documents/Proyecto ONEFRAME/Copia de controller_oneframe/sshkey.pem"
MODEL_DRIVE_ID = "17mYK-GRKjxrw9qmhq77TI5hAj39qIsVH"

ARCHIVOS_CODIGO = ["main.py", "config.py", "engine.py", "app.py"] 
REMOTE_DIR = "/home/ubuntu"
MODEL_NAME = "yolo_oneframe_v2.pt"
USER = "ubuntu"
# ======================================================

console = Console()

def extraer_id_drive(url):
    patrones = [r'/file/d/([a-zA-Z0-9_-]+)', r'id=([a-zA-Z0-9_-]+)', r'^([a-zA-Z0-9_-]+)$']
    for patron in patrones:
        match = re.search(patron, url)
        if match: return match.group(1)
    return None

def main():
    console.print(Panel.fit("🤖 OneFrame V9.1 - DATA WORKFLOW (FIXED)", style="bold cyan"))

    if not os.path.exists(KEY_PATH):
        console.print(f"[bold red]❌ Error: Llave no encontrada en {KEY_PATH}[/bold red]")
        return

    # --- FASE 1: INTERROGATORIO ---
    host_ip = Prompt.ask("📡 IP del Host (Lambda)")
    num_camaras = IntPrompt.ask("🎥 ¿Cuántas cámaras?", default=1)
    
    lista_videos = []
    for i in range(num_camaras):
        lbl = f"CAM{i+1}"
        link = Prompt.ask(f"🔗 Link Drive {lbl}")
        did = extraer_id_drive(link)
        if did: lista_videos.append((lbl, did))
        else: console.print("[red]❌ ID inválido[/red]")

    # --- FASE 2: PROCESAMIENTO ---
    try:
        console.print(f"[cyan]📡 Conectando a {host_ip}...[/cyan]")
        c = Connection(host=host_ip, user=USER, connect_kwargs={"key_filename": KEY_PATH})
        
        # --- CORRECCIÓN AQUÍ: INSTALACIÓN COMPLETA ---
        console.print("[yellow]🛠️ Inyectando TODAS las librerías necesarias...[/yellow]")
        # Instalamos gdown primero
        c.run("python3 -m pip install -U gdown", hide=True)
        
        # Instalamos el paquete completo: Visión + Datos + Web
        cmd_install = (
            "python3 -m pip install "
            "opencv-python-headless "  # El ojo (cv2)
            "ultralytics "            # El cerebro (YOLO)
            "shapely "                # La geometría (Zonas)
            "scikit-image "           # Utilidades de imagen
            "streamlit "              # La App Web
            "pandas openpyxl "        # Excel
            "\"numpy==1.26.4\" "      # Compatibilidad
            "\"moviepy<2.0.0\""       # Video
        )
        c.run(cmd_install, hide=True)
        console.print("[green]✅ Dependencias instaladas correctamente[/green]")

        # Subir código
        console.print("[yellow]⚡ Subiendo scripts...[/yellow]")
        for f in ARCHIVOS_CODIGO: c.put(f, remote=f"{REMOTE_DIR}/{f}")
        
        # Bajar Modelo IA
        if c.run(f"test -f {REMOTE_DIR}/{MODEL_NAME}", warn=True, hide=True).failed:
            console.print("[blue]🧠 Bajando Modelo...[/blue]")
            c.run(f"python3 -m gdown {MODEL_DRIVE_ID} -O {REMOTE_DIR}/{MODEL_NAME} --fuzzy", pty=True)

        # Limpieza Inicial
        c.run(f"rm -rf {REMOTE_DIR}/clips_out", warn=True)
        c.run(f"mkdir -p {REMOTE_DIR}/clips_out", warn=True)
        c.run(f"rm -rf {REMOTE_DIR}/PROCESADO", warn=True) 

        # Bucle de Cámaras
        for cam_name, drive_id in lista_videos:
            console.print(Panel.fit(f"🎬 PROCESANDO: {cam_name}", style="bold yellow"))
            c.run(f"rm {REMOTE_DIR}/video_analisis.mp4", warn=True)
            console.print(f"  ☁️ Bajando video...")
            c.run(f"python3 -m gdown {drive_id} -O {REMOTE_DIR}/video_analisis.mp4 --fuzzy", pty=True)
            
            console.print(f"  🔥 Analizando...")
            c.run(f"cd {REMOTE_DIR} && python main.py", pty=True)
            
            console.print(f"  🏷️ Etiquetando...")
            cmd = (f"cd {REMOTE_DIR}/clips_out && for f in *.mp4; do "
                   f"if [[ $f != CAM* ]]; then mv \"$f\" \"{cam_name}_$f\"; fi; done")
            c.run(cmd, warn=True)

        # --- FASE 3: CLASIFICACIÓN HUMANA ---
        console.print(Panel.fit(
            f"✅ ¡VIDEOS PROCESADOS!\n"
            f"1. Abre otra terminal.\n"
            f"2. Pega: ssh -i {KEY_PATH} -L 8501:localhost:8501 ubuntu@{host_ip} streamlit run app.py\n"
            f"3. Ve a http://localhost:8501 y clasifica.\n"
            f"4. ¡NO CIERRES ESTA VENTANA! Vuelve aquí cuando termines.",
            style="bold green"
        ))
        
        Prompt.ask("✋ Presiona ENTER cuando hayas terminado de clasificar en la web y guardado el Excel...")

        # --- FASE 4: EXPORTACIÓN ---
        console.print("[yellow]📦 Empaquetando Base de Datos (Videos + Excel)...[/yellow]")
        zip_name = "ENTREGA_PARTIDO.zip"
        # Instalamos ZIP por si acaso no está (en algunas imágenes minimalistas falta)
        c.run("sudo apt-get update && sudo apt-get install -y zip", hide=True)
        
        c.run(f"cd {REMOTE_DIR} && zip -r {zip_name} PROCESADO", pty=True)
        
        console.print("[cyan]📥 Descargando a tu PC...[/cyan]")
        c.get(f"{REMOTE_DIR}/{zip_name}", local=zip_name)
        
        console.print(Panel.fit(f"🎉 ¡ÉXITO! Archivo descargado: {zip_name}", style="bold green"))

    except Exception as e:
        console.print(f"[bold red]💀 Error: {e}[/bold red]")

if __name__ == "__main__":
    main()