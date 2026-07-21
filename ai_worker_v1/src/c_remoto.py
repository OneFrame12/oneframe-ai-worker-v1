import cv2
import numpy as np
import requests
import re
import sys

# ==========================================
# 1. HERRAMIENTAS DE CALIBRACIÓN (OPENCV)
# ==========================================

def calibrar_camara(frame, nombre_camara):
    """
    Abre la imagen y permite al usuario dibujar rectángulos con el mouse
    para definir las zonas donde la IA debe prestar atención.
    """
    print(f"\n=============================================")
    print(f"📐 ABRIENDO VENTANA PARA: {nombre_camara}")
    print("=============================================")
    print("INSTRUCCIONES DE DIBUJO:")
    print("1. Haz clic y arrastra el mouse para dibujar un rectángulo.")
    print("2. Presiona 'ENTER' o 'ESPACIO' para confirmar la zona.")
    print("3. Si te equivocas, presiona 'C' para borrar y volver a dibujar.")
    
    # Truco para que la ventana se adapte a tu pantalla
    window_name_1 = f"Paso 1: AREA DE GOL - {nombre_camara}"
    cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
    
    print("\n👉 Dibuja el ÁREA DE GOL (Área grande y portería) y presiona ENTER.")
    bbox_gol = cv2.selectROI(window_name_1, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name_1)
    
    window_name_2 = f"Paso 2: AREA DE PELIGRO - {nombre_camara}"
    cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
    
    print("👉 Dibuja el ÁREA DE PELIGRO (Afuera del área grande) y presiona ENTER.")
    bbox_peligro = cv2.selectROI(window_name_2, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name_2)

    return {
        "GOAL_ZONES_RAW": [bbox_gol], 
        "DANGER_ZONES_RAW": [bbox_peligro]
    }

# ==========================================
# 2. UTILIDADES PARA VIDEOS (GOOGLE DRIVE / OPENCV)
# ==========================================

def obtener_link_directo_drive(url_compartida):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url_compartida)
    if match:
        file_id = match.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url_compartida

def obtener_frame_calibracion(url_video, minuto=20):
    print(f"⏳ Adelantando el video al minuto {minuto} para extraer el frame de calibración...")
    cap = cv2.VideoCapture(url_video)
    
    milisegundos = minuto * 60 * 1000
    cap.set(cv2.CAP_PROP_POS_MSEC, milisegundos)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"⚠️ Advertencia: No se pudo cargar el video (Drive suele bloquear archivos pesados).")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"Error cargando video de Drive", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
    return frame

# ==========================================
# 3. COMUNICACIÓN CON LA NUBE (LAMBDA LABS)
# ==========================================

def disparar_analisis_en_nube(configuracion, ip_lambda, match_id, url_norte, url_sur):
    print(f"\n🚀 Enviando coordenadas e iniciando IA en la nube ({ip_lambda})...")
    api_url = f"http://{ip_lambda}:5000/api/iniciar-oneframe"
    
    payload = {
        "match_id": match_id,
        "video_norte": url_norte,
        "video_sur": url_sur,
        "configuracion_visual": configuracion
    }
    
    try:
        respuesta = requests.post(api_url, json=payload, timeout=10)
        if respuesta.status_code in [200, 202]:
            print("✅ ¡Éxito! La supercomputadora recibió las coordenadas.")
            print("Respuesta de Lambda:", respuesta.json())
        else:
            print(f"❌ Error del servidor (Código {respuesta.status_code}):", respuesta.text)
    except requests.exceptions.RequestException as e:
        print("❌ Error de conexión (Es normal si el servidor en Lambda aún no está encendido).")

# ==========================================
# 4. FLUJO PRINCIPAL DE EJECUCIÓN
# ==========================================

def main():
    print("========================================")
    print("   ⚽ ONEFRAME: CENTRO DE MANDO V2 ⚽  ")
    print("========================================\n")
    
    ip_lambda = input("🌐 Pega la IP de tu máquina en Lambda Labs: ").strip()
    match_id = input("🆔 Escribe el ID del partido (ej. final_colo_colo): ").strip()
    link_norte = input("🔗 Pega el link de Google Drive (Cámara Norte): ").strip()
    link_sur = input("🔗 Pega el link de Google Drive (Cámara Sur): ").strip()
    
    if not ip_lambda or not link_norte:
        print("❌ Faltan datos esenciales. Saliendo del programa.")
        sys.exit(1)

    print("\nPreparando entorno de calibración...")
    direct_norte = obtener_link_directo_drive(link_norte)
    direct_sur = obtener_link_directo_drive(link_sur)

    frame_norte = obtener_frame_calibracion(direct_norte, minuto=20)
    frame_sur = obtener_frame_calibracion(direct_sur, minuto=20)

    print("\n--- 📐 CALIBRANDO CÁMARA NORTE ---")
    config_norte = calibrar_camara(frame_norte, "arco_norte")
    
    print("\n--- 📐 CALIBRANDO CÁMARA SUR ---")
    config_sur = calibrar_camara(frame_sur, "arco_sur")

    configuracion_partido = {
        "GOAL_ZONES": {
            "arco_norte": config_norte["GOAL_ZONES_RAW"],
            "arco_sur": config_sur["GOAL_ZONES_RAW"]
        },
        "DANGER_ZONES": {
            "arco_norte": config_norte["DANGER_ZONES_RAW"],
            "arco_sur": config_sur["DANGER_ZONES_RAW"]
        }
    }

    disparar_analisis_en_nube(configuracion_partido, ip_lambda, match_id, link_norte, link_sur)

if __name__ == "__main__":
    main()