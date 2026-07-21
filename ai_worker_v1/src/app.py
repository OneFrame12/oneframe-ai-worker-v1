import streamlit as st
import os
import glob
import shutil
import time
from supabase import create_client, Client

# ==========================================
# 1. CONFIGURACIÓN INICIAL Y SUPABASE
# ==========================================
st.set_page_config(page_title="OneFrame DB Manager", layout="wide")
st.title("🗂️ Centro de Datos y Clasificación")

@st.cache_resource
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_connection()

INPUT_DIR = "/home/ubuntu/clips_out"
OUTPUT_BASE = "/home/ubuntu/PROCESADO"
CATEGORIAS = ["GOL", "TIRO AL ARCO", "TIRO PELIGROSO", "BASURA"]

# SEGURIDAD: Crear carpetas de entrada y salida si no existen (evita crash inicial)
os.makedirs(INPUT_DIR, exist_ok=True)
for cat in CATEGORIAS:
    if cat != "BASURA":
        os.makedirs(os.path.join(OUTPUT_BASE, cat), exist_ok=True)

# ==========================================
# 2. SELECTOR DE PARTIDOS
# ==========================================
st.sidebar.header("⚙️ Configuración del Partido")

@st.cache_data(ttl=600)
def get_teams_dict():
    try:
        response = supabase.table("teams").select("id, name").execute()
        return {team["id"]: team["name"] for team in response.data}
    except Exception as e:
        st.sidebar.error("Error conectando a Supabase (Teams)")
        return {}

teams_dict = get_teams_dict()

@st.cache_data(ttl=60)
def get_recent_matches():
    try:
        response = supabase.table("matches").select("id, team_a_id, team_b_id, match_date").order("created_at", desc=True).limit(10).execute()
        return response.data
    except Exception as e:
        st.sidebar.error("Error conectando a Supabase (Matches)")
        return []

matches = get_recent_matches()

def format_match(match):
    local = teams_dict.get(match["team_a_id"], "Local")
    visita = teams_dict.get(match["team_b_id"], "Visita")
    fecha = match["match_date"].split("T")[0] if match["match_date"] else ""
    return f"{local} vs {visita} ({fecha})"

partido_seleccionado = st.sidebar.selectbox("Selecciona el Partido:", matches, format_func=format_match)

# ==========================================
# 3. LÓGICA DE CLASIFICACIÓN
# ==========================================

# Detectar cambio de partido y resetear la base de datos en memoria.
if partido_seleccionado:
    current_match_id = partido_seleccionado["id"]
    if st.session_state.get("active_match_id") != current_match_id:
        st.session_state.database = []
        st.session_state.active_match_id = current_match_id

if 'database' not in st.session_state:
    st.session_state.database = []

# Buscar clips
clips = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
clips.sort()

if clips and partido_seleccionado:
    current_clip = clips[0]
    clip_name = os.path.basename(current_clip)
    
    total_processed = len(st.session_state.database)
    total_files = len(clips) + total_processed
    
    st.progress(total_processed / total_files if total_files > 0 else 0)
    
    col_video, col_data = st.columns([2, 1])

    with col_video:
        # st.video es nativo, carga rápido
        st.video(current_clip)

    with col_data:
        st.write(f"### 📝 Archivo: `{clip_name}`")
        st.write(f"**Progreso:** {total_processed} de {total_files} revisados")
        
        with st.form("data_form"):
            categoria = st.radio("Clasificación Visual:", CATEGORIAS, index=2)
            equipo = st.radio("¿Qué equipo generó la acción?", ["Local", "Visitante"])
            
            if st.form_submit_button("✅ GUARDAR CLASIFICACIÓN", type="primary", use_container_width=True):
                # Parsear clip_number desde el filename (formato: "clip_003_shot_1427s.mp4")
                match_id = partido_seleccionado["id"]
                clip_number = None
                try:
                    import re as _re
                    m = _re.match(r"clip_(\d+)_", clip_name)
                    if m:
                        clip_number = int(m.group(1))
                except Exception:
                    pass

                if categoria == "BASURA":
                    os.remove(current_clip)
                    # Marcar como rechazado en Supabase para el FeedbackLearner
                    if clip_number is not None:
                        try:
                            supabase.table("clips").update({
                                "is_confirmed": False,
                                "review_status": "rejected"
                            }).eq("match_id", match_id).eq("clip_number", clip_number).execute()
                        except Exception:
                            pass
                else:
                    new_filename = f"{categoria}_{equipo}_{clip_name}"
                    destination = os.path.join(OUTPUT_BASE, categoria, new_filename)
                    shutil.move(current_clip, destination)

                    # Guardar en memoria para las estadísticas
                    st.session_state.database.append({"Categoria": categoria, "Equipo": equipo})

                    # Marcar como aprobado en Supabase para el FeedbackLearner
                    if clip_number is not None:
                        try:
                            supabase.table("clips").update({
                                "is_confirmed": True,
                                "review_status": "approved"
                            }).eq("match_id", match_id).eq("clip_number", clip_number).execute()
                        except Exception:
                            pass

                time.sleep(0.2) # Pequeña pausa para asegurar movimiento de archivos
                st.rerun()

elif not clips and partido_seleccionado and len(st.session_state.database) == 0:
    st.info("⏳ No hay clips en la carpeta de entrada. Esperando a que el motor YOLO genere archivos...")

# ==========================================
# 4. EXPORTACIÓN Y SUBIDA A SUPABASE
# ==========================================
if not clips and partido_seleccionado and len(st.session_state.database) > 0:
    st.success("✅ ¡Todos los clips del partido han sido clasificados!")
    
    st.header("🏁 Paso Final: Enviar Estadísticas a la Web")
    st.write("Calcularemos las métricas y actualizaremos la plataforma inmediatamente.")
    
    if st.button("🚀 SUBIR ESTADÍSTICAS A SUPABASE", type="primary"):
        with st.spinner("Sincronizando con la nube..."):
            try:
                equipo_local_id = partido_seleccionado["team_a_id"]
                equipo_visitante_id = partido_seleccionado["team_b_id"]
                match_id = partido_seleccionado["id"]

                stats = {
                    "Local":     {"team_id": equipo_local_id,     "goals": 0, "shots": 0, "shots_target": 0},
                    "Visitante": {"team_id": equipo_visitante_id, "goals": 0, "shots": 0, "shots_target": 0}
                }
                
                for clip in st.session_state.database:
                    eq = clip["Equipo"]
                    cat = clip["Categoria"]
                    if cat == "GOL":
                        stats[eq]["goals"] += 1
                    elif cat == "TIRO PELIGROSO":
                        stats[eq]["shots"] += 1
                    elif cat == "TIRO AL ARCO":
                        stats[eq]["shots_target"] += 1
                        stats[eq]["shots"] += 1  

                # Borrar registros anteriores para evitar duplicados si se resube
                supabase.table("match_team_stats").delete().eq("match_id", match_id).execute()
                
                # Insertar filas nuevas
                for eq_name, data in stats.items():
                    chances = data["goals"] + data["shots"] 
                    supabase.table("match_team_stats").insert({
                        "match_id": match_id,
                        "team_id": data["team_id"],
                        "goals": data["goals"],
                        "shots_total": data["shots"],
                        "shots_on_target": data["shots_target"],
                        "chances": chances
                    }).execute()
                    
                st.balloons()
                st.success("🎉 Estadísticas subidas con éxito. Plataforma actualizada.")
                
                # Resetear la base de datos local para que no vuelva a pedir subir
                st.session_state.database = []
                
                st.info("""
                **Siguientes pasos:**
                1. Descarga la carpeta `PROCESADO` a tu computador local.
                2. Une los clips en Filmora y sube el video final a YouTube.
                3. Pega el link de YouTube en tu panel de administración web.
                """)
                
            except Exception as e:
                st.error(f"❌ Error al subir a Supabase: {e}")

elif not partido_seleccionado:
    st.warning("👈 Selecciona un partido en el menú lateral para comenzar.")