import re
import os
import subprocess
import sys
import time
import json
import glob
import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
from datetime import datetime
try:
    from supabase import create_client
except ImportError:
    create_client = None

# ==========================================
# CONFIGURACIÓN
# ==========================================

# Resolución base de config.py (para ROI en píxeles absolutos)
CONFIG_BASE_W = 1920
CONFIG_BASE_H = 1080

# NUEVO: Ruta al archivo JSON en lugar de modificar config.py directamente
CALIBRACION_PATH = os.path.join(os.path.dirname(__file__), "calibracion.json")

st.set_page_config(page_title="Centro de Calibración", layout="wide")
st.title("⚽ OneFrame: Centro de Calibración Inteligente")

# ==========================================
# CONSTANTES DE SERVIDOR
# ==========================================
SSH_KEY = "/Users/lucasvalenzuela/Documents/Proyecto ONEFRAME/Copia de controller_oneframe/sshkey_clean.pem"
REMOTE_VIDEO_PATH = "/tmp/video_oneframe.mp4"

# ==========================================
# 🖥️  CONEXIÓN AL SERVIDOR GPU
# ==========================================
st.markdown("### 🖥️ Conexión al Servidor GPU")

col_ip, col_btn_conn = st.columns([3, 1])
with col_ip:
    server_ip_input = st.text_input(
        "IP del servidor Lambda Labs:",
        value=st.session_state.get("server_ip", ""),
        placeholder="ej. 150.136.32.233",
        key="input_server_ip",
    )
with col_btn_conn:
    st.write("")
    st.write("")
    btn_conectar = st.button("🔌 Conectar", use_container_width=True)

if btn_conectar and server_ip_input:
    resultado_ssh = subprocess.run(
        [
            "ssh",
            "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=6",
            "-o", "BatchMode=yes",
            f"ubuntu@{server_ip_input}",
            "echo ok",
        ],
        capture_output=True,
        text=True,
    )
    if resultado_ssh.returncode == 0:
        st.session_state["server_ip"] = server_ip_input
        st.session_state["server_conectado"] = True
    else:
        st.session_state["server_conectado"] = False

if st.session_state.get("server_conectado") is True:
    st.success(f"🟢 Conectado → `{st.session_state.get('server_ip')}`")
elif st.session_state.get("server_conectado") is False:
    st.error("🔴 Sin conexión — verificá la IP y que el servidor esté encendido.")

st.write("---")

# ==========================================
# 0. SELECTOR DE PARTIDO
# ==========================================
st.markdown("### 🎯 0. Selección del Partido")

modo_partido = st.radio(
    "Modo de análisis:",
    ["🏆 Partido oficial", "🧪 Modo prueba"],
    horizontal=True,
    key="modo_partido",
)

if modo_partido == "🏆 Partido oficial":
    if create_client is None:
        st.error("❌ Librería `supabase` no instalada. Usá Modo prueba o instalá con `pip install supabase`.")
    else:
        @st.cache_resource
        def _init_supabase():
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
            return create_client(url, key)

        @st.cache_data(ttl=600)
        def _get_teams_dict():
            try:
                response = _init_supabase().table("teams").select("id, name").execute()
                return {team["id"]: team["name"] for team in response.data}
            except Exception as e:
                st.error(f"Error conectando a Supabase (Teams): {e}")
                return {}

        @st.cache_data(ttl=60)
        def _get_recent_matches():
            try:
                response = (
                    _init_supabase()
                    .table("matches")
                    .select("id, team_a_id, team_b_id, match_date")
                    .order("created_at", desc=True)
                    .limit(10)
                    .execute()
                )
                return response.data
            except Exception as e:
                st.error(f"Error conectando a Supabase (Matches): {e}")
                return []

        def _format_match(match):
            teams = _get_teams_dict()
            local = teams.get(match["team_a_id"], "Local")
            visita = teams.get(match["team_b_id"], "Visita")
            fecha = match["match_date"].split("T")[0] if match["match_date"] else ""
            return f"{local} vs {visita} ({fecha})"

        _matches = _get_recent_matches()
        if _matches:
            partido = st.selectbox("Seleccioná el partido:", _matches, format_func=_format_match)
            if partido:
                st.session_state["match_id"] = partido["id"]
                st.success(f"✅ Partido seleccionado — match_id: `{partido['id']}`")
        else:
            st.warning("⚠️ No se encontraron partidos en Supabase.")

else:  # Modo prueba
    st.session_state["match_id"] = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    st.info(f"🧪 Modo prueba activo — match_id generado: `{st.session_state['match_id']}`")

st.write("---")


# ==========================================
# NUEVA FUNCIÓN: GUARDADO LIMPIO EN JSON
# ==========================================

def guardar_en_json(tipo_zona: str, coordenadas: list, lado_arco: str = None):
    """
    Guarda las coordenadas limpiamente en un archivo calibracion.json.
    Esto evita romper config.py con expresiones regulares.
    """
    # 1. Leer el JSON si ya existe para no borrar otras zonas
    if os.path.exists(CALIBRACION_PATH):
        with open(CALIBRACION_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # 2. Asegurar que existan los diccionarios base
    if "goal_zones" not in data:
        data["goal_zones"] = {}
    if "danger_zones" not in data:
        data["danger_zones"] = {}

    # 3. Insertar la nueva zona
    if tipo_zona == "ROI":
        data["roi_points"] = coordenadas
    elif tipo_zona == "Gol":
        if not lado_arco: raise ValueError("Lado arco es obligatorio")
        data["goal_zones"][lado_arco] = coordenadas
    elif tipo_zona == "Peligro":
        if not lado_arco: raise ValueError("Lado arco es obligatorio")
        data["danger_zones"][lado_arco] = coordenadas

    # 4. Guardar archivo
    with open(CALIBRACION_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ==========================================
# 1. SUBIDA DE VIDEO AL SERVIDOR
# ==========================================
st.markdown("### 📤 1. Subir video al servidor")

archivo = st.file_uploader(
    "Seleccioná el video del partido:",
    type=["mp4", "mov"],
    key="video_uploader",
)

if archivo is not None:
    size_bytes = archivo.size
    size_gb = size_bytes / (1024 ** 3)
    st.write(f"📁 **{archivo.name}** — {size_gb:.2f} GB")

    btn_subir = st.button("📡 Subir al servidor", type="primary")

    if btn_subir:
        if not st.session_state.get("server_ip"):
            st.error("❌ Primero conectate al servidor en la sección de arriba.")
        else:
            ip = st.session_state["server_ip"]
            local_tmp = f"/tmp/{archivo.name}"

            progress_bar = st.progress(0, text="Preparando archivo...")
            buffer = archivo.getbuffer()
            total = len(buffer)
            chunk = 1024 * 1024  # 1 MB
            with open(local_tmp, "wb") as f:
                written = 0
                while written < total:
                    end = min(written + chunk, total)
                    f.write(buffer[written:end])
                    written = end
                    progress_bar.progress(
                        written / total * 0.3,
                        text=f"Preparando... {written / (1024 ** 2):.0f} / {total / (1024 ** 2):.0f} MB",
                    )

            progress_bar.progress(0.3, text="Transfiriendo al servidor vía SCP...")
            resultado_scp = subprocess.run(
                [
                    "scp",
                    "-i", SSH_KEY,
                    "-o", "StrictHostKeyChecking=no",
                    local_tmp,
                    f"ubuntu@{ip}:{REMOTE_VIDEO_PATH}",
                ],
                capture_output=True,
                text=True,
            )

            if resultado_scp.returncode == 0:
                progress_bar.progress(1.0, text="✅ Transferencia completada")
                st.session_state["video_remoto"] = REMOTE_VIDEO_PATH
                st.success(f"✅ Video subido correctamente → `{REMOTE_VIDEO_PATH}`")

                # Extraer frame de calibración desde el servidor remoto
                try:
                    subprocess.run(
                        [
                            "ssh",
                            "-i", SSH_KEY,
                            "-o", "StrictHostKeyChecking=no",
                            "-o", "BatchMode=yes",
                            f"ubuntu@{ip}",
                            "ffmpeg -i /tmp/video_oneframe.mp4 -ss 00:00:30 -frames:v 1 /tmp/frame_calibracion.jpg -y",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    subprocess.run(
                        [
                            "scp",
                            "-i", SSH_KEY,
                            "-o", "StrictHostKeyChecking=no",
                            f"ubuntu@{ip}:/tmp/frame_calibracion.jpg",
                            "/tmp/frame_calibracion.jpg",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    st.session_state["foto_real"] = Image.open("/tmp/frame_calibracion.jpg")
                    st.image(
                        st.session_state["foto_real"],
                        caption="Frame de calibración (segundo 30)",
                        use_container_width=True,
                    )
                except Exception as e_frame:
                    st.warning(f"⚠️ No se pudo extraer el frame de calibración: {e_frame}")
            else:
                progress_bar.empty()
                st.error(f"❌ Error en SCP: {resultado_scp.stderr}")

elif st.session_state.get("video_remoto"):
    st.info(f"✅ Video ya cargado en sesión: `{st.session_state['video_remoto']}`")

st.write("---")

# ==========================================
# 2. MAPA DE SONIDO (AUDIO)
# ==========================================
if "grafico_real" in st.session_state:
    st.markdown("### 🔊 2. Mapa de Sonido (Picos de Gol/Silbatos)")
    st.image(st.session_state["grafico_real"], use_container_width=True)
    st.info(
        "💡 Usa este gráfico para saber en qué minutos es más probable que haya ocurrido"
        " algo importante en esta cámara."
    )

# ==========================================
# 3. CALIBRACIÓN POR CLICS (FOTO REAL)
# ==========================================
if "foto_real" in st.session_state:
    st.markdown("### 📌 3. Calibración de Zonas (Por Clics)")

    # ── Selector de tipo de zona y lado ─────────────────────────────────────
    col_tipo, col_lado = st.columns([1, 1])
    with col_tipo:
        tipo_zona = st.selectbox(
            "🎯 Qué zona estás calibrando:",
            options=["ROI (Bordes de Cancha)", "Peligro (Area Grande)", "Gol (Arco/Area Chica)"],
            key="sel_tipo_zona",
        )
    with col_lado:
        if tipo_zona != "ROI (Bordes de Cancha)":
            lado_arco = st.radio(
                "🧭 ¿Qué lado?",
                options=["arco_norte", "arco_sur"],
                horizontal=True,
                key="sel_lado_arco",
            )
        else:
            lado_arco = None
            st.markdown("&nbsp;")  # espaciador visual

    # Mapear selector a tipo interno
    TIPO_MAP = {
        "ROI (Bordes de Cancha)": "ROI",
        "Peligro (Area Grande)":  "Peligro",
        "Gol (Arco/Area Chica)":  "Gol",
    }
    tipo_interno = TIPO_MAP[tipo_zona]

    # ── Estado de puntos ───────────────────────────────────────────────────
    estado_key = f"puntos_calibracion_{tipo_interno}_{lado_arco or 'roi'}"
    if estado_key not in st.session_state:
        st.session_state[estado_key] = []
    puntos = st.session_state[estado_key]

    # ── Construir imagen con feedback visual ──────────────────────────────
    foto_base = st.session_state["foto_real"].convert("RGB")
    DISPLAY_W = 1280
    ratio = DISPLAY_W / foto_base.width
    DISPLAY_H = int(foto_base.height * ratio)
    foto_display = foto_base.resize((DISPLAY_W, DISPLAY_H), Image.LANCZOS)

    draw = ImageDraw.Draw(foto_display)
    COLOR_PUNTO = (0, 255, 136)    
    COLOR_LINEA = (0, 255, 136)
    COLOR_CIERRE = (255, 200, 0)   
    R = 6  

    if len(puntos) > 1:
        draw.line(puntos, fill=COLOR_LINEA, width=2)
        draw.line([puntos[-1], puntos[0]], fill=COLOR_CIERRE, width=2)

    for i, (px, py) in enumerate(puntos):
        draw.ellipse((px - R, py - R, px + R, py + R), fill=COLOR_PUNTO, outline="white")
        draw.text((px + R + 2, py - R), str(i + 1), fill="white")

    # ── Captura de clics ───────────────────────────────────────────────────
    st.caption(
        f"ℹ️ Haz **clic** en la imagen para marcar vértices. "
        f"**{len(puntos)} punto(s)** marcado(s). "
        "Cuantos más puntos, más precisa la zona."
    )

    clic = streamlit_image_coordinates(
        foto_display,
        key=f"coords_{estado_key}",
    )

    if clic is not None:
        nuevo_punto = (clic["x"], clic["y"])
        if not puntos or nuevo_punto != puntos[-1]:
            st.session_state[estado_key].append(nuevo_punto)
            st.rerun()

    # ── Controles de edición ───────────────────────────────────────────────
    col_undo, col_clear, col_espacio = st.columns([1, 1, 4])
    with col_undo:
        if st.button("↩️ Deshacer último punto", disabled=len(puntos) == 0):
            st.session_state[estado_key].pop()
            st.rerun()
    with col_clear:
        if st.button("🗑️ Borrar todo", disabled=len(puntos) == 0):
            st.session_state[estado_key] = []
            st.rerun()

    st.write("")

    # ── Preview de coordenadas + botón de guardado ─────────────────────────
    if len(puntos) >= 3:
        if tipo_interno == "ROI":
            escala_x = CONFIG_BASE_W / DISPLAY_W
            escala_y = CONFIG_BASE_H / DISPLAY_H
            coords_config = [(int(x * escala_x), int(y * escala_y)) for x, y in puntos]
            destino_txt = "roi_points"
            st.code(f'"roi_points": {coords_config}')
        else:
            coords_config = [(round(x / DISPLAY_W, 3), round(y / DISPLAY_H, 3)) for x, y in puntos]
            bloque = "goal_zones" if tipo_interno == "Gol" else "danger_zones"
            destino_txt = f"{bloque} -> {lado_arco}"
            st.code(f'"{lado_arco}": {coords_config}')

        st.markdown(f"→ Se guardará en el archivo dinámico `calibracion.json`")

        btn_label = (
            f"💾 Guardar {tipo_zona} en JSON"
            if lado_arco
            else f"💾 Guardar ROI en JSON"
        )
        if st.button(btn_label, type="primary"):
            try:
                guardar_en_json(tipo_interno, coords_config, lado_arco=lado_arco)
                st.success(f"✅ ¡Zona guardada correctamente en `calibracion.json`!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Error inesperado al guardar: {e}")
    else:
        st.info(f"👆 Marca al menos **3 puntos** en la imagen para ver las coordenadas ({len(puntos)}/3).")

elif "foto_real" not in st.session_state:
    st.warning("👈 Primero procesa un video en la nube para activar la calibración.")

# ==========================================
# 4. FASE 4: EJECUCIÓN DEL ANÁLISIS
# ==========================================
st.write("---")
st.markdown("### 🚀 Fase 4: Ejecución de Análisis")
st.write(
    "Una vez calibradas las zonas, lanza el motor YOLO directamente desde aquí. "
    "El análisis correrá en este servidor y podrás seguir el progreso en tiempo real."
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(SCRIPT_DIR, "main.py")
LOG_FILE    = os.path.join(SCRIPT_DIR, "yolo_log.txt")

if "proceso_yolo" not in st.session_state:
    st.session_state["proceso_yolo"] = None

col_btn, col_status = st.columns([2, 3])

with col_btn:
    btn_iniciar = st.button(
        "🔥 Iniciar Análisis del Partido",
        type="primary",
        disabled=(
            not st.session_state.get("server_ip")
            or not st.session_state.get("video_remoto")
            or (st.session_state["proceso_yolo"] is not None
                and st.session_state["proceso_yolo"].poll() is None)
        ),
        help="Requiere: servidor conectado + video subido.",
    )

if btn_iniciar:
    with open(LOG_FILE, "w") as f:
        f.write("")

    with open(LOG_FILE, "w") as log_f:
        proceso = subprocess.Popen(
            [sys.executable, MAIN_SCRIPT,
             "--match-id", st.session_state.get("match_id", "test_sin_id")],
            stdout=log_f,
            stderr=subprocess.STDOUT,  
            cwd=SCRIPT_DIR,
            text=True,
        )
    st.session_state["proceso_yolo"] = proceso
    st.rerun()  

proceso = st.session_state.get("proceso_yolo")

if proceso is not None:
    log_placeholder  = st.empty()
    prog_placeholder = st.empty()

    if proceso.poll() is None:
        prog_placeholder.info("⏳ Análisis en curso... (actualiza cada 2 segundos)")

        while proceso.poll() is None:
            try:
                with open(LOG_FILE, "r", errors="replace") as f:
                    lineas = f.readlines()
                tail = "".join(lineas[-30:])
            except FileNotFoundError:
                tail = "(esperando salida...)"

            log_placeholder.code(tail, language="")  
            time.sleep(2)

        try:
            with open(LOG_FILE, "r", errors="replace") as f:
                log_final = f.read()
        except FileNotFoundError:
            log_final = "(log no encontrado)"

        prog_placeholder.empty()
        log_placeholder.code(log_final[-4000:], language="")  

        codigo_salida = proceso.returncode
        if codigo_salida == 0:
            st.success("✅ ¡Análisis Terminado! Los clips fueron generados en la carpeta `clips_out`.")
            st.balloons()
        else:
            st.error(
                f"❌ El proceso terminó con código de error `{codigo_salida}`. "
                "Revisa el log de arriba para ver el error."
            )

        st.session_state["proceso_yolo"] = None

    else:
        try:
            with open(LOG_FILE, "r", errors="replace") as f:
                log_final = f.read()
        except FileNotFoundError:
            log_final = ""

        if log_final:
            log_placeholder.code(log_final[-4000:], language="")

        st.session_state["proceso_yolo"] = None


# ============================================================
# FASE 5: REVISIÓN Y ETIQUETADO DE CLIPS (Offline)
# ============================================================
st.write("---")
st.markdown("### 🎬 Fase 5: Revisión y Etiquetado de Clips (Modo Offline)")
st.info("💡 **Nota Profesional:** Recuerda que también tienes tu archivo `app.py` que se conecta directamente a Supabase para subir las estadísticas a tu web. Puedes usar esta sección si prefieres una revisión rápida local.")

CLIPS_DIR   = os.path.join(SCRIPT_DIR, "clips_out")
REPORTE_JSON = os.path.join(SCRIPT_DIR, "reporte_partido.json")

ETIQUETAS = ["Pendiente", "Gol", "Tiro al arco", "Tiro peligroso", "Descartar (Falsa Alarma)"]

PREFIJOS = {
    "Gol":           "GOL",
    "Tiro al arco":  "TIRO_ARCO",
    "Tiro peligroso":"TIRO_PELIGROSO",
}

clips_disponibles = sorted(glob.glob(os.path.join(CLIPS_DIR, "*.mp4")))

if not clips_disponibles:
    st.write("⏳ Esperando a que YOLO genere los clips... ")
else:
    if "clasificaciones" not in st.session_state:
        st.session_state["clasificaciones"] = {}

    for clip_path in clips_disponibles:
        nombre = os.path.basename(clip_path)
        if nombre not in st.session_state["clasificaciones"]:
            st.session_state["clasificaciones"][nombre] = "Pendiente"

    clsf = st.session_state["clasificaciones"]

    goles          = sum(1 for v in clsf.values() if v == "Gol")
    tiros_arco     = sum(1 for v in clsf.values() if v == "Tiro al arco")
    tiros_peligros = sum(1 for v in clsf.values() if v == "Tiro peligroso")
    ocasiones      = goles + tiros_arco + tiros_peligros   
    pendientes     = sum(1 for v in clsf.values() if v == "Pendiente")
    descartados    = sum(1 for v in clsf.values() if v == "Descartar (Falsa Alarma)")

    st.markdown("#### 📊 Estadísticas en Vivo")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🥅 Goles",              goles)
    c2.metric("🎯 Tiros al Arco",     tiros_arco)
    c3.metric("⚡ Tiros Peligrosos",  tiros_peligros)
    c4.metric("🔥 Ocasiones Creadas", ocasiones)
    c5.metric("⏳ Pendientes",         pendientes)
    c6.metric("🗑️ Descartados",        descartados)

    st.write("")
    st.markdown("#### 🎞️ Cine Privado")

    for idx, clip_path in enumerate(clips_disponibles, start=1):
        nombre = os.path.basename(clip_path)
        etiqueta_actual = clsf.get(nombre, "Pendiente")

        iconos = {
            "Pendiente":               "⏳",
            "Gol":                     "🥅",
            "Tiro al arco":            "🎯",
            "Tiro peligroso":          "⚡",
            "Descartar (Falsa Alarma)": "🗑️",
        }
        icono = iconos.get(etiqueta_actual, "❓")
        titulo_exp = f"{icono} Jugada {idx} — {nombre}"

        with st.expander(titulo_exp, expanded=(etiqueta_actual == "Pendiente")):
            col_vid, col_label = st.columns([2, 1])

            with col_vid:
                st.video(clip_path)

            with col_label:
                nueva_etiqueta = st.radio(
                    "Clasificación:",
                    options=ETIQUETAS,
                    index=ETIQUETAS.index(etiqueta_actual),
                    key=f"radio_{nombre}",
                )
                if nueva_etiqueta != etiqueta_actual:
                    st.session_state["clasificaciones"][nombre] = nueva_etiqueta
                    st.rerun()  

    st.write("---")
    col_exp, col_info = st.columns([2, 3])

    with col_exp:
        btn_exportar = st.button(
            "💾 Exportar Reporte Local",
            type="primary",
            use_container_width=True,
        )

    if btn_exportar:
        errores_export = []
        renombrados    = []
        eliminados     = []

        for clip_path in clips_disponibles:
            nombre     = os.path.basename(clip_path)
            etiqueta   = clsf.get(nombre, "Pendiente")
            prefijo    = PREFIJOS.get(etiqueta)

            if etiqueta == "Descartar (Falsa Alarma)":
                try:
                    os.remove(clip_path)
                    eliminados.append(nombre)
                except OSError as e:
                    errores_export.append(f"No se pudo eliminar {nombre}: {e}")

            elif prefijo and not nombre.startswith(prefijo):
                nuevo_nombre = f"{prefijo}_{nombre}"
                nuevo_path   = os.path.join(CLIPS_DIR, nuevo_nombre)
                try:
                    os.rename(clip_path, nuevo_path)
                    clsf[nuevo_nombre] = clsf.pop(nombre)
                    renombrados.append((nombre, nuevo_nombre))
                except OSError as e:
                    errores_export.append(f"No se pudo renombrar {nombre}: {e}")

        reporte = {
            "resumen": {
                "goles":           goles,
                "tiros_al_arco":   tiros_arco,
                "tiros_peligrosos":tiros_peligros,
                "ocasiones":       ocasiones,
                "descartados":     descartados,
            },
            "jugadas": {
                etiqueta: [
                    n for n, e in clsf.items() if e == etiqueta
                ]
                for etiqueta in ETIQUETAS if etiqueta != "Pendiente"
            },
        }

        try:
            with open(REPORTE_JSON, "w", encoding="utf-8") as fp:
                json.dump(reporte, fp, indent=2, ensure_ascii=False)

            st.success(f"✅ Reporte guardado en `{REPORTE_JSON}`")
            if renombrados:
                st.info(f"📂 {len(renombrados)} archivo(s) renombrado(s).")
            if eliminados:
                st.warning(f"🗑️ {len(eliminados)} clip(s) falso(s) eliminado(s).")
            if errores_export:
                for err in errores_export:
                    st.error(err)
        except OSError as e:
            st.error(f"❌ No se pudo guardar el reporte JSON: {e}")