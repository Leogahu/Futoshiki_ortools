import os
import sys
import json
import time
import numpy as np
import cv2
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.append(SRC_DIR)

from core.pipeline import resolver_imagen, get_predictor
from core.visualization.render import render_solution


# =====================================================================
# Configuracion
# =====================================================================
st.set_page_config(
    page_title="Futoshiki Solver",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
    /* ===== importar fuentes de Google ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    /* ===== fondo global azulado ===== */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #f0f7ff 0%, #e0efff 40%, #dbeafe 100%);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }

    /* ===== tipografia ===== */
    html, body, [class*="css"], [data-testid="stAppViewContainer"] * {
        font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0c1e3e;
    }

    /* ===== navbar ===== */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.8rem;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        border: 1px solid rgba(147, 197, 253, 0.45);
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(37, 99, 235, 0.08);
    }
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0c1e3e;
        letter-spacing: -0.02em;
    }
    .navbar-brand .logo {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.35);
    }
    .navbar-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        background: #dcfce7;
        color: #166534;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .navbar-tag .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
    }

    /* ===== hero ===== */
    .hero {
        padding: 2.5rem 2.8rem;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        border-radius: 20px;
        color: white;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(37, 99, 235, 0.3);
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-content {
        position: relative;
        z-index: 1;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: white;
    }
    .hero p {
        margin: 0.8rem 0 0 0;
        font-size: 1.05rem;
        color: rgba(255, 255, 255, 0.95);
        max-width: 700px;
        line-height: 1.65;
        font-weight: 400;
    }
    .hero-stats {
        display: flex;
        gap: 2rem;
        margin-top: 1.5rem;
        position: relative;
        z-index: 1;
    }
    .hero-stat {
        display: flex;
        flex-direction: column;
    }
    .hero-stat .value {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        letter-spacing: -0.02em;
    }
    .hero-stat .label {
        font-size: 0.7rem;
        color: rgba(255, 255, 255, 0.85);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.15rem;
        font-weight: 600;
    }

    /* ===== card ===== */
    .card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border-radius: 16px;
        border: 1px solid rgba(147, 197, 253, 0.35);
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(37, 99, 235, 0.06);
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    .card:hover {
        box-shadow: 0 6px 24px rgba(37, 99, 235, 0.12);
        border-color: rgba(96, 165, 250, 0.55);
    }
    .card-title {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        font-weight: 700;
        color: #0c1e3e;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .card-title .ico {
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
    }

    /* ===== uploader ===== */
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #f8fbff 0%, #eff6ff 100%);
        border: 2px dashed #93c5fd;
        border-radius: 12px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #3b82f6;
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    }

    /* ===== FIX del boton interno del uploader ===== */
    [data-testid="stFileUploaderDropzone"] button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.2rem !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%) !important;
        color: #1d4ed8 !important;
        border: 1.5px solid #93c5fd !important;
        transition: all 0.25s ease !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        white-space: nowrap !important;
        width: auto !important;
        min-width: auto !important;
    }
    [data-testid="stFileUploaderDropzone"] button:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-color: #3b82f6 !important;
        color: #1e40af !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }

    /* ocultar el texto duplicado del boton (deja solo un "Upload") */
    [data-testid="stFileUploaderDropzone"] button > div > span,
    [data-testid="stFileUploaderDropzone"] button > span > span,
    [data-testid="stFileUploaderDropzone"] button [data-testid="stMarkdownContainer"] {
        display: none !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Upload";
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #1d4ed8 !important;
    }

    /* icono SVG del boton */
    [data-testid="stFileUploaderDropzone"] button svg {
        width: 16px !important;
        height: 16px !important;
        flex-shrink: 0 !important;
        fill: currentColor !important;
        stroke: currentColor !important;
    }

    /* texto "200MB per file · PNG, JPG" */
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        color: #2563eb !important;
        font-size: 0.9rem !important;
    }

    /* ===== radio ===== */
    [data-testid="stRadio"] > div {
        display: flex;
        gap: 0.5rem;
    }
    [data-testid="stRadio"] label {
        background: #f8fbff;
        border: 1px solid rgba(147, 197, 253, 0.45);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    [data-testid="stRadio"] label:hover {
        border-color: #3b82f6;
        background: #eff6ff;
    }

    /* ===== botones con degradado animado ===== */
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem 2rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        background-size: 200% 200% !important;
        background-position: 0% 50% !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
        background-size: 200% 200% !important;
        background-position: 0% 50% !important;
        color: white;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 35%, #2563eb 65%, #1e40af 100%) !important;
        background-size: 200% 200% !important;
        background-position: 100% 50% !important;
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.55);
    }
    .stButton > button[kind="primary"]:active {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.5);
    }

    .stDownloadButton > button {
        border-radius: 12px;
        font-weight: 600;
        padding: 0.7rem 1.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
        color: #1d4ed8;
        border: 1.5px solid #93c5fd;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        background-size: 200% 200% !important;
        background-position: 0% 50% !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 50%, #93c5fd 100%) !important;
        background-size: 200% 200% !important;
        background-position: 100% 50% !important;
        color: #1e40af;
        border-color: #3b82f6;
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25);
    }

    /* ===== metricas ===== */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid rgba(147, 197, 253, 0.35);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 12px rgba(37, 99, 235, 0.06);
    }
    [data-testid="stMetricValue"] {
        color: #2563eb !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #1d4ed8 !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ===== tabs ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #eff6ff;
        border-radius: 12px;
        padding: 0.4rem;
        border: 1px solid rgba(147, 197, 253, 0.35);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        color: #2563eb;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(219, 234, 254, 0.6);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%) !important;
        color: #1d4ed8 !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.18);
    }

    /* ===== imagenes ===== */
    [data-testid="stImage"] img {
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(37, 99, 235, 0.12);
    }

    /* ===== placeholder ===== */
    .placeholder {
        text-align: center;
        padding: 3.5rem 2rem;
        background: linear-gradient(135deg, #ffffff 0%, #f8fbff 100%);
        border: 2px dashed #93c5fd;
        border-radius: 16px;
        margin-top: 1rem;
    }
    .placeholder .ico {
        width: 72px;
        height: 72px;
        margin: 0 auto 1.2rem auto;
        border-radius: 20px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }
    .placeholder h3 {
        color: #0c1e3e;
        margin: 0 0 0.4rem 0;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .placeholder p {
        color: #2563eb;
        margin: 0;
        font-size: 0.95rem;
    }

    /* ===== puzzle grid con signos ===== */
    .puzzle-wrap {
        background: linear-gradient(135deg, #f8fbff 0%, #eff6ff 100%);
        border: 1px solid rgba(147, 197, 253, 0.4);
        border-radius: 16px;
        padding: 18px;
        display: flex;
        justify-content: center;
    }
    .puzzle-table {
        border-collapse: separate;
        border-spacing: 6px;
        margin: 0 auto;
    }
    .puzzle-cell {
        width: 56px;
        height: 56px;
        text-align: center;
        vertical-align: middle;
        border-radius: 10px;
        font-size: 1.35rem;
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
        font-weight: 700;
        transition: all 0.15s ease;
    }
    .puzzle-cell.pista {
        background: linear-gradient(135deg, #1e40af 0%, #2563eb 100%);
        color: white;
        box-shadow: 0 3px 10px rgba(30, 64, 175, 0.3);
    }
    .puzzle-cell.solucion {
        background: linear-gradient(135deg, #60a5fa 0%, #93c5fd 100%);
        color: white;
        box-shadow: 0 3px 10px rgba(96, 165, 250, 0.32);
    }
    .puzzle-cell.vacia {
        background: #ffffff;
        color: #93c5fd;
        border: 1px solid rgba(147, 197, 253, 0.55);
    }
    .puzzle-sign {
        width: 20px;
        text-align: center;
        color: #2563eb;
        font-size: 1.2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    }

    /* ===== footer ===== */
    .footer {
        text-align: center;
        color: #2563eb;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding: 1.5rem 1rem;
        border-top: 1px solid rgba(147, 197, 253, 0.45);
    }
    .footer strong {
        color: #1e40af;
        font-weight: 700;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(147, 197, 253, 0.4);
        margin: 1.5rem 0;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Cache
# =====================================================================
@st.cache_resource(show_spinner=False)
def _warmup():
    get_predictor()
    return True


# =====================================================================
# Helpers
# =====================================================================
def _bytes_to_bgr(file_bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _puzzle_grid_with_signs(state, values):
    """
    Genera el puzzle como una tabla HTML con celdas y signos entre ellas.
    """
    n = state["size"]
    pistas = state["grid"]
    h = state["horizontal_constraints"]
    v = state["vertical_constraints"]

    html = '<div class="puzzle-wrap">'
    html += '<table class="puzzle-table">'

    for i in range(n):
        html += "<tr>"
        for j in range(n):
            val = values[i][j]
            es_pista = pistas[i][j] != 0

            if val == 0:
                cls = "vacia"
                val_str = "·"
            elif es_pista:
                cls = "pista"
                val_str = str(val)
            else:
                cls = "solucion"
                val_str = str(val)

            html += f'<td class="puzzle-cell {cls}">{val_str}</td>'

            if j < n - 1:
                sign = h[i][j] if h[i][j] else ""
                html += f'<td class="puzzle-sign">{sign}</td>'

        html += "</tr>"

        if i < n - 1:
            html += "<tr>"
            for j in range(n):
                sign = v[i][j] if v[i][j] else ""
                html += f'<td class="puzzle-sign">{sign}</td>'
                if j < n - 1:
                    html += '<td class="puzzle-sign"></td>'
            html += "</tr>"

    html += "</table>"
    html += "</div>"
    return html


def _stat_card(label, value, icon=""):
    return f"""
    <div class="card" style="padding: 1rem 1.2rem; margin-bottom: 0;">
        <div style="display:flex; align-items:center; gap:0.6rem; color:#2563eb; font-size:0.75rem; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">
            <span>{icon}</span><span>{label}</span>
        </div>
        <div style="font-size:1.4rem; font-weight:700; color:#1d4ed8; margin-top:0.4rem;">
            {value}
        </div>
    </div>
    """


# =====================================================================
# Inicializar
# =====================================================================
_warmup()


# =====================================================================
# Navbar
# =====================================================================
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="logo">🧩</div>
        <span>Futoshiki Solver</span>
    </div>
    <div class="navbar-tag">
        <span class="dot"></span>
        <span>Sistema activo</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# Hero
# =====================================================================
st.markdown("""
<div class="hero">
    <div class="hero-content">
        <h1>Resuelve tu Futoshiki con IA</h1>
        <p>Sube una imagen del puzzle. Nuestro sistema detecta el tablero,
        clasifica los números y signos con redes neuronales convolucionales,
        y lo resuelve en milisegundos con constraint programming.</p>
        <div class="hero-stats">
            <div class="hero-stat">
                <span class="value">4-9</span>
                <span class="label">Tamaños</span>
            </div>
            <div class="hero-stat">
                <span class="value">&lt;0.01s</span>
                <span class="label">Solver</span>
            </div>
            <div class="hero-stat">
                <span class="value">2 CNNs</span>
                <span class="label">Deep Learning</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =====================================================================
# Selector de modo
# =====================================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title"><span class="ico">📥</span>Entrada de imagen</div>', unsafe_allow_html=True)

col_modo, _ = st.columns([2, 3])
with col_modo:
    modo = st.radio(
        "Fuente",
        ["📁 Subir archivo", "📷 Usar cámara"],
        label_visibility="collapsed",
        horizontal=True,
    )

img_bgr = None
if modo == "📁 Subir archivo":
    uploaded = st.file_uploader(
        "Arrastra tu imagen aquí o haz clic para buscar",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded is not None:
        img_bgr = _bytes_to_bgr(uploaded.read())
else:
    camera_photo = st.camera_input("Toma una foto del tablero")
    if camera_photo is not None:
        img_bgr = _bytes_to_bgr(camera_photo.read())

st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# Placeholder
# =====================================================================
if img_bgr is None:
    st.markdown("""
    <div class="placeholder">
        <div class="ico">📸</div>
        <h3>Sube una imagen para empezar</h3>
        <p>Arrastra un archivo o usa la cámara para capturar el puzzle</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# =====================================================================
# Layout
# =====================================================================
st.markdown("")
col_izq, col_der = st.columns([1, 1], gap="medium")

with col_izq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="ico">🖼️</span>Imagen original</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_der:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><span class="ico">✨</span>Resultado</div>', unsafe_allow_html=True)
    resultado_placeholder = st.empty()
    resultado_placeholder.info("Presiona **Resolver puzzle** para procesar la imagen.")
    st.markdown('</div>', unsafe_allow_html=True)


# =====================================================================
# Boton resolver
# =====================================================================
col_l, col_c, col_r = st.columns([1, 1, 1])
with col_c:
    resolver_clicked = st.button(
        "🚀 Resolver puzzle",
        type="primary",
        use_container_width=True,
    )


# =====================================================================
# Procesamiento
# =====================================================================
if resolver_clicked:
    with st.spinner("🔍 Procesando imagen y resolviendo puzzle..."):
        t0 = time.time()
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            result = resolver_imagen(gray, debug=False)
            total_time = time.time() - t0
        except Exception as e:
            st.error(f"❌ Error procesando la imagen: {e}")
            st.stop()

    state = result["state"]
    solution = result["solution"]
    solver_status = result["solver_status"]

    with col_der:
        if solution is None:
            resultado_placeholder.error(
                f"❌ No se encontró solución\n\nStatus: `{solver_status}`"
            )
        else:
            img_sol = render_solution(img_bgr, state, solution)
            resultado_placeholder.empty()
            resultado_placeholder.image(
                cv2.cvtColor(img_sol, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(_stat_card("Tamaño", f"{result['n']}×{result['n']}", "📐"), unsafe_allow_html=True)
    with m2:
        st.markdown(_stat_card("Status", solver_status, "✅"), unsafe_allow_html=True)
    with m3:
        st.markdown(_stat_card("Tiempo solver", f"{result['solver_time']*1000:.1f} ms", "⏱️"), unsafe_allow_html=True)
    with m4:
        st.markdown(_stat_card("Tiempo total", f"{total_time:.2f} s", "⚡"), unsafe_allow_html=True)

    if solution is not None:
        st.markdown("")
        tab1, tab2, tab3 = st.tabs(["📋 Estado inicial", "🎯 Solución", "📄 JSON"])

        with tab1:
            st.markdown("##### Estado inicial detectado")
            st.markdown(_puzzle_grid_with_signs(state, state["grid"]), unsafe_allow_html=True)
            st.caption(
                "Celdas **azul oscuro** = pistas originales. "
                "Signos **< >** entre celdas = restricciones detectadas."
            )

        with tab2:
            st.markdown("##### Solución completa")
            st.markdown(_puzzle_grid_with_signs(state, solution), unsafe_allow_html=True)
            st.caption(
                "Celdas **azul oscuro** = pistas originales. "
                "Celdas **azul claro** = valores calculados por el solver."
            )

        with tab3:
            state_clean = {k: v for k, v in state.items() if k != "_meta"}
            st.json(state_clean)

        st.markdown("")
        st.markdown("#### 💾 Descargas")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            state_clean = {k: v for k, v in state.items() if k != "_meta"}
            output_json = {
                "state": state_clean,
                "solution": solution,
                "solver_status": solver_status,
                "solver_time": result["solver_time"],
                "n": result["n"],
            }
            json_str = json.dumps(output_json, indent=2, ensure_ascii=False)
            st.download_button(
                "⬇️ Descargar JSON",
                data=json_str,
                file_name="futoshiki_resultado.json",
                mime="application/json",
                use_container_width=True,
            )

        with col_dl2:
            img_sol = render_solution(img_bgr, state, solution)
            is_success, buf = cv2.imencode(".png", img_sol)
            if is_success:
                st.download_button(
                    "⬇️ Descargar imagen",
                    data=buf.tobytes(),
                    file_name="futoshiki_solucion.png",
                    mime="image/png",
                    use_container_width=True,
                )


# =====================================================================
# Footer
# =====================================================================
st.markdown("""
<div class="footer">
    <strong>Futoshiki Solver</strong> · Visión Computacional + Constraint Programming<br>
    Desarrollado con PyTorch, OpenCV y OR-Tools
</div>
""", unsafe_allow_html=True)