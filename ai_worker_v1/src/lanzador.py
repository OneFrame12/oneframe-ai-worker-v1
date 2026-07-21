"""
lanzador.py - OneFrame: Centro de Comando (Windows)
App de escritorio en Tkinter para orquestar el sistema de visión en Lambda Labs.

INSTRUCCIÓN: Cambia SSH_KEY_PATH con la ruta a tu archivo .pem antes de usar.
"""

import os
import subprocess
import threading
import time
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ============================================================
# ⚙️  CONFIGURACIÓN — EDITA ESTA LÍNEA ANTES DE USAR
# ============================================================
SSH_KEY_PATH = r"C:\Users\narri\Desktop\controller_oneframe\sshkey.pem"

# Rutas remotas donde se subirán los videos en el servidor
REMOTE_VIDEO_NORTE = "/tmp/video_norte.mp4"
REMOTE_VIDEO_SUR   = "/tmp/video_sur.mp4"

# URL de la interfaz Streamlit (expuesta por el túnel)
STREAMLIT_URL = "http://localhost:8501"

# ============================================================
# 🎨  PALETA DE COLORES
# ============================================================
BG_DARK    = "#1a1a2e"
BG_PANEL   = "#16213e"
BG_CARD    = "#0f3460"
ACCENT     = "#e94560"
TEXT_WHITE = "#eaeaea"
TEXT_MUTED = "#8a8aa0"
SUCCESS    = "#00ff88"
WARNING    = "#ffaa00"


# ============================================================
# 🧠  LÓGICA DE FONDO (hilo separado)
# ============================================================

class Orquestador:
    """Encapsula toda la lógica de subir archivos y abrir el túnel SSH."""

    def __init__(self, ip: str,
                 norte_drive: str, norte_local: str,
                 sur_drive: str,   sur_local: str,
                 on_status,        on_error,   on_success):
        self.ip          = ip.strip()
        self.norte_drive = norte_drive.strip()
        self.norte_local = norte_local.strip()
        self.sur_drive   = sur_drive.strip()
        self.sur_local   = sur_local.strip()
        self.on_status   = on_status   # callback(msg: str)
        self.on_error    = on_error    # callback(msg: str)
        self.on_success  = on_success  # callback()
        self._tunel_proc = None

    def ejecutar(self):
        """Punto de entrada para el hilo de trabajo."""
        try:
            # 1 ── Validaciones previas
            if not self.ip:
                raise ValueError("Ingresa la IP de Lambda Labs.")
            if not os.path.isfile(SSH_KEY_PATH):
                raise FileNotFoundError(
                    f"Llave SSH no encontrada:\n{SSH_KEY_PATH}\n"
                    "Edita SSH_KEY_PATH en lanzador.py."
                )

            # 2 ── Subida SCP (solo si el usuario eligió archivos locales)
            if self.norte_local:
                self.on_status("📤 Subiendo Cámara Norte (puede tardar varios minutos)...")
                self._scp_upload(self.norte_local, REMOTE_VIDEO_NORTE)

            if self.sur_local:
                self.on_status("📤 Subiendo Cámara Sur (puede tardar varios minutos)...")
                self._scp_upload(self.sur_local, REMOTE_VIDEO_SUR)

            # 3 ── Abrir túnel SSH bidireccional en segundo plano
            self.on_status("🔌 Estableciendo túnel SSH…")
            self._tunel_proc = subprocess.Popen(
                [
                    "ssh",
                    "-i", SSH_KEY_PATH,
                    "-N",                          # Sin shell remota
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "ServerAliveInterval=60",
                    "-L", "5000:localhost:5000",
                    "-L", "8501:localhost:8501",
                    f"ubuntu@{self.ip}",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # 4 ── Esperar a que el túnel se estabilice
            self.on_status("⏳ Esperando que el túnel se establezca…")
            time.sleep(4)

            # Verificar que el proceso del túnel sigue vivo
            if self._tunel_proc.poll() is not None:
                raise RuntimeError(
                    "El túnel SSH se cerró inesperadamente.\n"
                    "Verifica la IP, la llave .pem y que el servidor esté encendido."
                )

            # 5 ── Abrir navegador apuntando a la UI de Streamlit
            webbrowser.open(STREAMLIT_URL)
            self.on_success()

        except Exception as exc:
            self.on_error(str(exc))

    def _scp_upload(self, local_path: str, remote_path: str):
        """Sube un archivo local al servidor vía SCP. Bloquea hasta terminar."""
        resultado = subprocess.run(
            [
                "scp",
                "-i", SSH_KEY_PATH,
                "-o", "StrictHostKeyChecking=no",
                local_path,
                f"ubuntu@{self.ip}:{remote_path}",
            ],
            capture_output=True,
            text=True,
        )
        if resultado.returncode != 0:
            raise RuntimeError(
                f"Error al subir {os.path.basename(local_path)}:\n{resultado.stderr}"
            )

    def cerrar_tunel(self):
        """Termina el proceso del túnel SSH si sigue vivo."""
        if self._tunel_proc and self._tunel_proc.poll() is None:
            self._tunel_proc.terminate()


# ============================================================
# 🖼️  INTERFAZ GRÁFICA
# ============================================================

class LanzadorApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("⚽ OneFrame — Centro de Comando")
        self.configure(bg=BG_DARK)
        self.resizable(False, False)
        self._orquestador = None

        # Rutas locales seleccionadas con el diálogo de archivo
        self._norte_local = tk.StringVar(value="")
        self._sur_local   = tk.StringVar(value="")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Construcción de la UI ────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 20, "pady": 10}

        # ── Encabezado ──────────────────────────────────────────────────
        header = tk.Frame(self, bg=ACCENT)
        header.pack(fill="x")
        tk.Label(
            header,
            text="⚽  OneFrame  |  Centro de Comando",
            font=("Segoe UI", 16, "bold"),
            bg=ACCENT, fg=TEXT_WHITE,
        ).pack(pady=12)

        # ── Contenedor principal ─────────────────────────────────────────
        main = tk.Frame(self, bg=BG_DARK, padx=24, pady=16)
        main.pack(fill="both")

        # ── IP Lambda Labs ───────────────────────────────────────────────
        self._add_label(main, "🌐  IP de Lambda Labs:")
        self._ip_entry = self._add_entry(main, placeholder="ej. 150.136.32.233")

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=12)

        # ── Cámara Norte ─────────────────────────────────────────────────
        self._add_label(main, "📷  Cámara Norte", subtitle=True)
        self._add_label(main, "Link de Google Drive (opcional):")
        self._norte_drive_entry = self._add_entry(
            main, placeholder="https://drive.google.com/file/d/..."
        )
        self._add_local_row(main, self._norte_local, "norte")

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=12)

        # ── Cámara Sur ───────────────────────────────────────────────────
        self._add_label(main, "📷  Cámara Sur", subtitle=True)
        self._add_label(main, "Link de Google Drive (opcional):")
        self._sur_drive_entry = self._add_entry(
            main, placeholder="https://drive.google.com/file/d/..."
        )
        self._add_local_row(main, self._sur_local, "sur")

        ttk.Separator(main, orient="horizontal").pack(fill="x", pady=12)

        # ── Botón principal ──────────────────────────────────────────────
        self._btn_conectar = tk.Button(
            main,
            text="🚀  Conectar y Preparar Entorno",
            font=("Segoe UI", 13, "bold"),
            bg=ACCENT, fg=TEXT_WHITE,
            activebackground="#c73652",
            activeforeground=TEXT_WHITE,
            bd=0, padx=20, pady=14,
            cursor="hand2",
            command=self._on_conectar,
        )
        self._btn_conectar.pack(fill="x", pady=(4, 0))

        # ── Barra de estado ──────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Esperando configuración…")
        self._status_lbl = tk.Label(
            main,
            textvariable=self._status_var,
            font=("Segoe UI", 10),
            bg=BG_DARK, fg=TEXT_MUTED,
            wraplength=440,
            justify="left",
        )
        self._status_lbl.pack(anchor="w", pady=(10, 4))

        # ── Pie de página ────────────────────────────────────────────────
        tk.Label(
            self,
            text=f"SSH_KEY_PATH: {SSH_KEY_PATH}",
            font=("Segoe UI", 8),
            bg=BG_DARK, fg=TEXT_MUTED,
        ).pack(pady=(0, 8))

    def _add_label(self, parent, text, subtitle=False):
        font = ("Segoe UI", 11, "bold") if subtitle else ("Segoe UI", 9)
        color = TEXT_WHITE if subtitle else TEXT_MUTED
        tk.Label(parent, text=text, font=font, bg=BG_DARK, fg=color).pack(
            anchor="w", pady=(6, 0)
        )

    def _add_entry(self, parent, placeholder=""):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0)
        frame.pack(fill="x", pady=(2, 0))
        entry = tk.Entry(
            frame,
            font=("Segoe UI", 10),
            bg=BG_CARD, fg=TEXT_WHITE,
            insertbackground=TEXT_WHITE,
            relief="flat", bd=8,
        )
        entry.pack(fill="x")
        # Texto de ayuda (placeholder simulado)
        if placeholder:
            entry.insert(0, placeholder)
            entry.config(fg=TEXT_MUTED)
            entry.bind("<FocusIn>",  lambda e, en=entry, ph=placeholder: self._clear_placeholder(e, en, ph))
            entry.bind("<FocusOut>", lambda e, en=entry, ph=placeholder: self._restore_placeholder(e, en, ph))
        return entry

    def _add_local_row(self, parent, local_var: tk.StringVar, camara: str):
        """Fila con campo de ruta local + botón para abrir diálogo de archivo."""
        row = tk.Frame(parent, bg=BG_DARK)
        row.pack(fill="x", pady=(4, 0))

        tk.Label(
            row, text="O archivo local:", font=("Segoe UI", 9),
            bg=BG_DARK, fg=TEXT_MUTED,
        ).pack(side="left")

        path_lbl = tk.Label(
            row,
            textvariable=local_var,
            font=("Segoe UI", 9, "italic"),
            bg=BG_DARK, fg=TEXT_MUTED,
            width=34, anchor="w",
        )
        path_lbl.pack(side="left", padx=6)

        tk.Button(
            row,
            text="📁 Subir Local",
            font=("Segoe UI", 9),
            bg=BG_CARD, fg=TEXT_WHITE,
            activebackground=ACCENT,
            activeforeground=TEXT_WHITE,
            bd=0, padx=8, pady=4,
            cursor="hand2",
            command=lambda v=local_var: self._elegir_archivo(v),
        ).pack(side="right")

    # ── Callbacks de UI ──────────────────────────────────────────────────

    @staticmethod
    def _clear_placeholder(event, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg=TEXT_WHITE)

    @staticmethod
    def _restore_placeholder(event, entry, placeholder):
        if not entry.get():
            entry.insert(0, placeholder)
            entry.config(fg=TEXT_MUTED)

    def _elegir_archivo(self, var: tk.StringVar):
        ruta = filedialog.askopenfilename(
            title="Selecciona el video del partido",
            filetypes=[("Videos MP4", "*.mp4"), ("Todos los archivos", "*.*")],
        )
        if ruta:
            var.set(ruta)

    def _on_conectar(self):
        """Valida los datos y lanza el hilo de trabajo."""
        ip           = self._ip_entry.get()
        norte_drive  = self._norte_drive_entry.get()
        sur_drive    = self._sur_drive_entry.get()
        norte_local  = self._norte_local.get()
        sur_local    = self._sur_local.get()

        # Limpiar placeholders del resultado de get()
        def _clean(val, placeholder):
            return "" if val == placeholder else val

        ip          = _clean(ip,          "ej. 150.136.32.233")
        norte_drive = _clean(norte_drive, "https://drive.google.com/file/d/...")
        sur_drive   = _clean(sur_drive,   "https://drive.google.com/file/d/...")

        if not ip:
            messagebox.showwarning("Falta la IP", "Ingresa la IP de tu servidor en Lambda Labs.")
            return

        # Deshabilitar botón durante la operación
        self._btn_conectar.config(state="disabled", text="⏳  Trabajando…")
        self._set_status("Iniciando…", color=WARNING)

        # Cerrar túnel anterior si existe
        if self._orquestador:
            self._orquestador.cerrar_tunel()

        self._orquestador = Orquestador(
            ip=ip,
            norte_drive=norte_drive, norte_local=norte_local,
            sur_drive=sur_drive,     sur_local=sur_local,
            on_status =lambda msg: self.after(0, self._set_status, msg, WARNING),
            on_error  =lambda msg: self.after(0, self._on_error,   msg),
            on_success=lambda:     self.after(0, self._on_success),
        )

        hilo = threading.Thread(target=self._orquestador.ejecutar, daemon=True)
        hilo.start()

    def _on_error(self, mensaje: str):
        self._set_status(f"❌ Error: {mensaje}", color=ACCENT)
        self._btn_conectar.config(state="normal", text="🚀  Conectar y Preparar Entorno")
        messagebox.showerror("Error de conexión", mensaje)

    def _on_success(self):
        self._set_status("✅ Conectado. Ve a tu navegador.", color=SUCCESS)
        self._status_lbl.config(fg=SUCCESS)
        self._btn_conectar.config(state="normal", text="🔄  Reconectar")

    def _set_status(self, msg: str, color: str = TEXT_MUTED):
        self._status_var.set(msg)
        self._status_lbl.config(fg=color)

    def _on_close(self):
        """Al cerrar la ventana, terminar el túnel SSH si sigue activo."""
        if self._orquestador:
            self._orquestador.cerrar_tunel()
        self.destroy()


# ============================================================
# ▶️  ARRANQUE
# ============================================================

if __name__ == "__main__":
    app = LanzadorApp()

    # Centrar la ventana en la pantalla
    app.update_idletasks()
    w, h = app.winfo_width(), app.winfo_height()
    sw, sh = app.winfo_screenwidth(), app.winfo_screenheight()
    app.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")

    app.mainloop()
