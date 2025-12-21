"""
Credit Risk Analysis - Streamlit UI Completa
Incluye todas las features necesarias para el modelo.
Excluye: 9 columnas constantes (se rellenan automáticamente) y 2 columnas removidas (PROFESSIONAL_CITY, PROFESSIONAL_BOROUGH).
"""

import streamlit as st
import requests
import os
import json
from pathlib import Path
from datetime import datetime

# API URL - use environment variable or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Cargar opciones de UI desde JSON
UI_OPTIONS_FILE = Path(__file__).parent / "ui_options.json"
try:
    with open(UI_OPTIONS_FILE, "r", encoding="utf-8") as f:
        UI_OPTIONS = json.load(f)
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo {UI_OPTIONS_FILE}. Ejecuta: python -m src.ui.extract_ui_options")
    UI_OPTIONS = {}

# Cargar perfiles predefinidos desde JSON
PROFILES_FILE = Path(__file__).parent / "profiles.json"
try:
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        PROFILES = json.load(f)
except FileNotFoundError:
    st.error(f"Error: No se encontró el archivo {PROFILES_FILE}")
    PROFILES = {}

def get_profile_data(profile_key):
    """Obtiene los datos de un perfil desde el archivo JSON"""
    if profile_key in PROFILES:
        # Convertir None de JSON a None de Python
        profile_data = PROFILES[profile_key]["data"]
        # Convertir valores None (que vienen como null en JSON) a None de Python
        return {k: (None if v is None else v) for k, v in profile_data.items()}
    return None

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== CSS PERSONALIZADO - TEMA OSCURO CON GRADIENTES PÚRPURA/AZUL ==========
st.markdown("""
<style>
    /* Variables de color - Más oscuro y violeta más vibrante */
    :root {
        --bg-primary: #0A0A0F;
        --bg-secondary: #0F0F1A;
        --bg-card: rgba(22, 21, 38, 0.6);
        --bg-input: rgba(15, 52, 96, 0.4);
        --text-primary: #FFFFFF;
        --text-secondary: #CCCCCC;
        --gradient-purple: linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #6A5ACD 100%);
        --gradient-purple-pink: linear-gradient(135deg, #8A2BE2 0%, #BA55D3 50%, #FF00FF 100%);
        --gradient-blue: linear-gradient(135deg, #483D8B 0%, #6A5ACD 50%, #9370DB 100%);
        --accent-purple: #8A2BE2;
        --accent-blue: #6A5ACD;
        --shadow-purple: 0 8px 32px rgba(138, 43, 226, 0.3);
        --shadow-card: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    /* Fondo principal - Más oscuro */
    .stApp {
        background: #0A0A0F !important;
        color: var(--text-primary);
    }
    
    /* Main container - fondo oscuro sólido - sin padding arriba */
    .main .block-container {
        background: #0A0A0F;
        padding-top: 0 !important;
        padding-bottom: 2rem;
        margin-top: 0 !important;
    }
    
    /* Eliminar padding del header de Streamlit */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Eliminar espacio del top */
    #MainMenu {
        visibility: hidden;
    }
    
    /* Eliminar cualquier margen superior */
    .stApp > header {
        display: none !important;
    }
    
    /* Eliminar cualquier padding/margin del elemento principal */
    .stApp > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Eliminar espacio del viewport */
    section[data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    
    /* Asegurar que el primer div (con CSS y título) no tenga padding/margin extra */
    .main .block-container > div[data-testid="stElementContainer"]:first-child:has(h1) {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* ELIMINAR COMPLETAMENTE CUALQUIER DIV VACÍO ANTES DEL TÍTULO */
    .main .block-container > div:first-child:not(:has(h1)):not(:has(*)) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        visibility: hidden !important;
        overflow: hidden !important;
        opacity: 0 !important;
    }
    
    /* Ocultar elementos vacíos */
    .main .block-container > div:empty,
    .main .block-container > div:first-child:empty {
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Ocultar bloques verticales que no contienen el título */
    .main .block-container > div[data-testid="stVerticalBlock"]:first-child:not(:has(h1)) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
    }
    
    /* Asegurar que el título sea el primer elemento visible */
    .main .block-container > div:has(h1):first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Eliminar elementos vacíos antes del contenido */
    .main .block-container > div:empty,
    .main .block-container > div[style*="height"]:empty,
    .element-container:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Forzar que el primer elemento markdown sea el primero visible */
    .main .block-container > div[data-testid="stMarkdownContainer"]:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* Eliminar cualquier espacio antes del primer markdown */
    .main .block-container > *:not([data-testid="stMarkdownContainer"]):first-child {
        display: none !important;
    }
    
    /* Eliminar elementos vacíos que Streamlit pueda insertar */
    .main .block-container > div:first-child:empty,
    .main .block-container > div[class*="st"]:first-child:empty {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ocultar cualquier elemento antes del primer markdown que tenga el título */
    .main .block-container > div:first-child:not([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
    
    /* Ocultar elementos vacíos que Streamlit crea automáticamente */
    .main .block-container > div:first-child[class*="st"],
    .main .block-container > div:first-child[class*="element"],
    .main > div:first-child > div:first-child:empty {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    
    /* Asegurar que solo el primer markdown con el título sea visible */
    /* Si el primer elemento no contiene h1, ocultarlo completamente */
    .main .block-container > div:first-child:not(:has(h1)):not([data-testid="stMarkdownContainer"]) {
        display: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        visibility: hidden !important;
    }
    
    /* Ocultar cualquier div vacío o con solo espacios antes del título */
    .main .block-container > div:first-child:empty,
    .main .block-container > div:first-child:only-child:empty {
        display: none !important;
    }
    
    /* Headers y títulos */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Inter', 'SF Pro', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
    }
    
    h1 {
        background: var(--gradient-purple);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem !important;
    }
    
    /* Texto general */
    p, label, .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }
    
    /* Subtítulos y descripciones */
    .stMarkdown p {
        color: var(--text-secondary) !important;
    }
    
    /* Inputs y selectboxes - Flotantes - TODOS IGUALES */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stTextInput > div > div > input {
        background-color: rgba(15, 52, 96, 0.5) !important;
        color: var(--text-primary) !important;
        border: 1px solid rgba(138, 43, 226, 0.5) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        font-size: 0.95rem !important;
        box-shadow: 0 2px 10px rgba(138, 43, 226, 0.2) !important;
        transition: all 0.3s ease !important;
        height: auto !important;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextInput > div > div > input:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 3px rgba(138, 43, 226, 0.3), 0 4px 15px rgba(138, 43, 226, 0.4) !important;
        background-color: rgba(15, 52, 96, 0.7) !important;
    }
    
    /* Asegurar que text_input tenga el mismo estilo */
    .stTextInput > div > div {
        width: 100% !important;
    }
    
    .stTextInput input {
        width: 100% !important;
    }
    
    /* Labels de inputs */
    label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    /* Botones principales - Flotantes con gradiente vibrante */
    .stButton > button {
        background: linear-gradient(135deg, #8A2BE2 0%, #BA55D3 50%, #FF00FF 100%) !important;
        color: var(--text-primary) !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.85rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 25px rgba(138, 43, 226, 0.5), 0 0 20px rgba(255, 0, 255, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(138, 43, 226, 0.7), 0 0 30px rgba(255, 0, 255, 0.5) !important;
    }
    
    /* Botones secundarios (Martin/Martina) - Flotantes */
    .stButton > button[kind="secondary"] {
        background: rgba(22, 21, 38, 0.8) !important;
        border: 2px solid rgba(138, 43, 226, 0.6) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 4px 15px rgba(138, 43, 226, 0.3) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: rgba(138, 43, 226, 0.3) !important;
        border-color: rgba(138, 43, 226, 0.9) !important;
        box-shadow: 0 6px 20px rgba(138, 43, 226, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Cards y contenedores - Solo fuera del form */
    .element-container {
        background-color: rgba(22, 21, 38, 0.7) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
        border: 1px solid rgba(138, 43, 226, 0.4) !important;
        box-shadow: var(--shadow-card), 0 0 20px rgba(138, 43, 226, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    /* Dentro del form, elementos sin fondo ni borde */
    [data-testid="stForm"] .element-container {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }
    
    /* Subheaders - Completamente integrados */
    .stSubheader {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.25rem !important;
    }
    
    /* Subheaders dentro del form - sin borde, más integrados */
    [data-testid="stForm"] .stSubheader,
    [data-testid="stForm"] h3 {
        margin-top: 0.75rem !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0.25rem !important;
        border-bottom: 1px solid rgba(138, 43, 226, 0.2) !important;
    }
    
    /* Checkboxes */
    .stCheckbox > label {
        color: var(--text-primary) !important;
    }
    
    .stCheckbox > label > div[data-baseweb="checkbox"] {
        background-color: var(--bg-input) !important;
        border-color: var(--accent-purple) !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 200, 0, 0.2) 0%, rgba(0, 150, 0, 0.2) 100%) !important;
        border-left: 4px solid #00C800 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(200, 0, 0, 0.2) 0%, rgba(150, 0, 0, 0.2) 100%) !important;
        border-left: 4px solid #FF4444 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 200, 0, 0.2) 0%, rgba(255, 150, 0, 0.2) 100%) !important;
        border-left: 4px solid #FFC800 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(106, 90, 205, 0.2) 0%, rgba(138, 43, 226, 0.2) 100%) !important;
        border-left: 4px solid var(--accent-purple) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: var(--gradient-purple) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(15, 52, 96, 0.2) !important;
        border-radius: 0 0 12px 12px !important;
    }
    
    /* JSON viewer */
    pre {
        background-color: var(--bg-input) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: var(--text-primary) !important;
    }
    
    /* Dividers - Más sutiles */
    hr {
        border-color: rgba(138, 43, 226, 0.2) !important;
        margin: 1.5rem 0 !important;
        opacity: 0.5 !important;
    }
    
    /* Footer */
    .stCaption {
        color: var(--text-secondary) !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent-purple) transparent transparent transparent !important;
    }
    
    /* Form container - Completamente unificado */
    [data-testid="stForm"] {
        background: rgba(22, 21, 38, 0.8) !important;
        border-radius: 24px !important;
        padding: 2rem !important;
        border: 1px solid rgba(138, 43, 226, 0.4) !important;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(138, 43, 226, 0.2) !important;
        backdrop-filter: blur(15px) !important;
        margin: 1rem 0 !important;
    }
    
    /* Eliminar TODAS las divisiones visuales entre secciones */
    .stSubheader {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Reducir MÁXIMO el espaciado entre elementos del formulario */
    [data-testid="stForm"] .element-container {
        margin-bottom: 0.25rem !important;
        padding: 0.75rem !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    
    /* Columnas completamente unificadas - sin padding */
    [data-testid="column"] {
        padding: 0.1rem !important;
    }
    
    /* Inputs integrados - sin separación visual */
    [data-testid="stForm"] .stNumberInput,
    [data-testid="stForm"] .stSelectbox,
    [data-testid="stForm"] .stTextInput,
    [data-testid="stForm"] .stCheckbox {
        margin-bottom: 0.5rem !important;
    }
    
    /* Eliminar márgenes entre elementos dentro del form */
    [data-testid="stForm"] > div {
        margin-bottom: 0 !important;
    }
    
    /* Hacer que los inputs se vean más integrados */
    [data-testid="stForm"] .stNumberInput > div,
    [data-testid="stForm"] .stSelectbox > div,
    [data-testid="stForm"] .stTextInput > div {
        margin-bottom: 0.25rem !important;
    }
    
    /* Columns spacing */
    [data-testid="column"] {
        padding: 0.5rem !important;
    }
    
</style>
<div style="text-align: center; padding: 0 0 0.5rem 0; margin: 0;">
    <h1 style="background: linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #6A5ACD 100%); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent; 
               background-clip: text; 
               font-size: 3rem; 
               font-weight: 700; 
               margin: 0; padding: 0;">
        CREDIT RISK ANALYSIS
    </h1>
</div>
""", unsafe_allow_html=True)

# Texto descriptivo - Segundo div
st.markdown("""
<div style="text-align: center; padding: 0 0 1rem 0; margin: 0;">
    <p style="color: #B0B0B0; font-size: 1.1rem; margin-top: 0; margin-bottom: 0.5rem; padding: 0;">
        Credit risk evaluation system using Machine Learning
    </p>
    <p style="color: #B0B0B0; font-size: 0.95rem; margin: 0; padding: 0;">
        Complete all available fields for a more accurate evaluation. Fields marked with * are required.
    </p>
</div>
""", unsafe_allow_html=True)

# ========== PERFILES PREDEFINIDOS ==========

# Función para enviar predicción y mostrar resultados
def send_prediction_and_display(payload, profile_name=None):
    """Envía predicción a la API y muestra los resultados"""
    with st.spinner("Evaluando riesgo crediticio..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "unknown")
                probability = result.get("probability", 0.0)
                confidence = result.get("confidence", "low")
                
                # Importar math para calcular coordenadas del gauge
                import math
                
                st.markdown("---")
                
                # Header con gradiente
                if profile_name:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
                        <h2 style="background: linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #6A5ACD 100%); 
                                   -webkit-background-clip: text; 
                                   -webkit-text-fill-color: transparent; 
                                   background-clip: text; 
                                   font-size: 2rem; 
                                   font-weight: 700;">
                               EVALUATION RESULT - {profile_name}
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem;">
                        <h2 style="background: linear-gradient(135deg, #8A2BE2 0%, #9370DB 50%, #6A5ACD 100%); 
                                   -webkit-background-clip: text; 
                                   -webkit-text-fill-color: transparent; 
                                   background-clip: text; 
                                   font-size: 2rem; 
                                   font-weight: 700;">
                               EVALUATION RESULT
                        </h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Top section: Risk Gauge (left) and Approval/Confidence (right)
                col_gauge, col_status = st.columns([1, 1])
                
                with col_gauge:
                    # Determinar color según el nivel de riesgo
                    if probability >= 0.7:
                        gauge_color = "#FF4444"  # Red
                    elif probability >= 0.5:
                        gauge_color = "#FFC800"  # Orange
                    elif probability >= 0.3:
                        gauge_color = "#9370DB"  # Purple
                    else:
                        gauge_color = "#00C800"  # Green
                    
                    # Calcular el ángulo de la aguja para el gauge (0% = 180° izquierda, 100% = 0° derecha)
                    needle_angle = 180 - (probability * 180)
                    needle_angle_rad = math.radians(needle_angle)
                    needle_x = 100 + 70 * math.cos(needle_angle_rad)
                    needle_y = 100 - 70 * math.sin(needle_angle_rad)
                    
                    # Calcular stroke-dasharray para el arco
                    arc_length = 251.33 * probability
                    total_arc = 251.33
                    
                    # Construir el SVG como string
                    svg_content = f'''<svg width="200" height="120" xmlns="http://www.w3.org/2000/svg">
                                <path d="M 20 100 A 80 80 0 0 1 180 100" 
                                      fill="none" 
                                      stroke="rgba(255, 255, 255, 0.2)" 
                                      stroke-width="18" 
                                      stroke-linecap="round"/>
                                <path d="M 20 100 A 80 80 0 0 1 180 100" 
                                      fill="none" 
                                      stroke="{gauge_color}" 
                                      stroke-width="18" 
                                      stroke-linecap="round"
                                      stroke-dasharray="{arc_length} {total_arc}"
                                      style="filter: drop-shadow(0 0 8px {gauge_color});"/>
                                <line x1="100" y1="100" x2="{needle_x:.2f}" y2="{needle_y:.2f}"
                                      stroke="#FFFFFF" 
                                      stroke-width="4" 
                                      stroke-linecap="round"
                                      style="filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.8));"/>
                                <circle cx="100" cy="100" r="8" fill="#FFFFFF" style="filter: drop-shadow(0 0 6px rgba(255, 255, 255, 0.8));"/>
                            </svg>'''
                    
                    # Crear el gauge usando HTML/CSS
                    st.markdown(f"""
                    <div style="background: rgba(15, 52, 96, 0.3);
                                border-radius: 20px;
                                padding: 2rem;
                                text-align: center;
                                border: 1px solid rgba(138, 43, 226, 0.3);
                                height: 280px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;
                                align-items: center;">
                        <div style="position: relative; width: 200px; height: 120px; margin-bottom: 1rem;">
                            {svg_content}
                        </div>
                        <p style="color: #FFFFFF; margin: 0.5rem 0 0 0; font-size: 2.5rem; font-weight: 700;">{probability:.1%}</p>
                        <p style="color: #B0B0B0; margin: 0.25rem 0 0 0; font-size: 0.9rem; font-weight: 500;">RISK</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_status:
                    # Approval Status and Confidence
                    if prediction.lower() == "approved":
                        status_color = "#00C800"
                        status_bg = "rgba(0, 200, 0, 0.2)"
                        status_icon = "✅"
                        status_text = "APPROVED"
                        status_desc = "Credit application was approved"
                    elif prediction.lower() == "rejected":
                        status_color = "#FF4444"
                        status_bg = "rgba(200, 0, 0, 0.2)"
                        status_icon = "❌"
                        status_text = "REJECTED"
                        status_desc = "Credit application was rejected"
                    else:
                        status_color = "#FFC800"
                        status_bg = "rgba(255, 200, 0, 0.2)"
                        status_icon = "⚠️"
                        status_text = "UNKNOWN"
                        status_desc = f"Prediction: {prediction}"
                    
                    confidence_labels = {
                        "high": "High",
                        "medium": "Medium",
                        "low": "Low"
                    }
                    confidence_display = confidence_labels.get(confidence, confidence)
                    
                    st.markdown(f"""<div style="background: linear-gradient(135deg, {status_bg} 0%, rgba(138, 43, 226, 0.1) 100%);
                                border-left: 4px solid {status_color};
                                border-radius: 20px;
                                padding: 2rem;
                                margin-bottom: 1rem;
                                height: 130px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;">
                        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                            <span style="font-size: 2rem; margin-right: 0.75rem;">{status_icon}</span>
                            <h3 style="color: {status_color}; margin: 0; font-size: 1.8rem; font-weight: 700;">{status_text}</h3>
                        </div>
                        <p style="color: #B0B0B0; margin: 0; font-size: 0.95rem;">{status_desc}</p>
                    </div>""", unsafe_allow_html=True)
                    
                    st.markdown(f"""<div style="background: rgba(15, 52, 96, 0.3);
                                border-radius: 20px;
                                padding: 2rem;
                                text-align: center;
                                border: 1px solid rgba(138, 43, 226, 0.3);
                                height: 130px;
                                display: flex;
                                flex-direction: column;
                                justify-content: center;">
                        <p style="color: #B0B0B0; margin: 0 0 0.75rem 0; font-size: 0.9rem; font-weight: 500;">Confidence</p>
                        <p style="color: #FFFFFF; margin: 0; font-size: 2.2rem; font-weight: 700;">{confidence_display}</p>
                    </div>""", unsafe_allow_html=True)
                
                # Mini message section (below top section)
                if probability >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_color = "#FFC800"
                    risk_bg = "rgba(255, 200, 0, 0.2)"
                    risk_icon = "⚠️"
                    risk_message = f"Default probability of {probability:.1%}. Rejection of the application is recommended."
                elif probability >= 0.5:
                    risk_level = "MODERATE-HIGH RISK"
                    risk_color = "#FFC800"
                    risk_bg = "rgba(255, 200, 0, 0.2)"
                    risk_icon = "🔶"
                    risk_message = f"Default probability of {probability:.1%}. Additional review required."
                elif probability >= 0.3:
                    risk_level = "MODERATE RISK"
                    risk_color = "#9370DB"
                    risk_bg = "rgba(106, 90, 205, 0.2)"
                    risk_icon = "🟡"
                    risk_message = f"Default probability of {probability:.1%}. Careful evaluation recommended."
                else:
                    risk_level = "LOW RISK"
                    risk_color = "#00C800"
                    risk_bg = "rgba(0, 200, 0, 0.2)"
                    risk_icon = "✅"
                    risk_message = f"Default probability of {probability:.1%}. Client with good credit profile."
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {risk_bg} 0%, rgba(138, 43, 226, 0.1) 100%);
                            border-left: 4px solid {risk_color};
                            border-radius: 12px;
                            padding: 1.5rem;
                            margin: 1rem 0;">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.5rem; margin-right: 0.5rem;">{risk_icon}</span>
                        <p style="color: {risk_color}; font-weight: 600; margin: 0; font-size: 1.2rem;">{risk_level}</p>
                    </div>
                    <p style="color: #B0B0B0; margin: 0; font-size: 1rem;">{risk_message}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detalles expandibles
                with st.expander("📋 Technical Details of the Query"):
                    st.write("**Payload sent to API:**")
                    st.json(payload)
                    st.write("**Complete API Response:**")
                    st.json(result)
            
            else:
                st.error(f"❌ API Error: Status {response.status_code}")
                st.text(response.text)
        
        except requests.exceptions.ConnectionError:
            st.error(f"❌ **Connection Error** - Could not connect to API at {API_URL}")
            st.info("Make sure the API is running. Verify with: `docker-compose ps`")
        
        except requests.exceptions.Timeout:
            st.error("❌ **Timeout** - The API took too long to respond")
        
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")

# Formulario completo con TODAS las features
with st.form("credit_risk_form"):
    
    # ========== SECTION 1: BASIC INFORMATION AND APPLICATION ==========
    st.markdown("""
    <div style="margin-top: 0.75rem; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
        <h3 style="color: #FFFFFF; font-weight: 600; font-size: 1.3rem; margin: 0;">
            BASIC INFORMATION AND APPLICATION
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        payment_day = st.number_input(
            "Payment Day *",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
            help="Day of the month chosen for payment (1-31)",
            key="payment_day"
        )
        
        application_type = st.selectbox(
            "Application Type *",
            options=["Web", "Carga"],
            help="How the application was submitted",
            key="application_type"
        )
        
        product_options = {
            "Not specified": None,
            "1 - Product A": 1,
            "2 - Product B": 2,
            "3 - Product C": 3,
            "4 - Product D": 4
        }
        product_selection = st.selectbox(
            "Product Type",
            options=list(product_options.keys()),
            help="Product type (optional)",
            key="product_select"
        )
        product = product_options[product_selection]
    
    with col2:
        age = st.number_input(
            "Age *",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Applicant's age",
            key="age"
        )
        
        sex = st.selectbox(
            "Sex *",
            options=["M", "F"],
            help="Applicant's sex",
            key="sex"
        )
        
        quant_dependants = st.number_input(
            "Number of Dependents *",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of people that depend on the applicant",
            key="dependants"
        )
    
    with col3:
        marital_status = st.selectbox(
            "Marital Status",
            options=["", "1 - Single", "2 - Married", "3 - Divorced", "4 - Widowed", "5 - Civil Union", "6 - Separated", "7 - Other"],
            help="Applicant's marital status (optional)",
            key="marital"
        )
        
        # Estado de nacimiento - selectbox con opciones reales
        state_birth_options = UI_OPTIONS.get("STATE_OF_BIRTH", [])
        state_birth_options_with_empty = [""] + state_birth_options
        state_of_birth = st.selectbox(
            "State of Birth",
            options=state_birth_options_with_empty,
            help="State where born (optional). Select from available options in the dataset.",
            key="state_birth"
        )
        state_of_birth = None if state_of_birth == "" else state_of_birth
        
        # Ciudad de nacimiento - text_input (alta cardinalidad: 9,910 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        city_of_birth = st.text_input(
            "City of Birth",
            value="",
            help="City where born (optional). Leave empty if unknown or enter exact name. If not in dataset, will be handled automatically.",
            key="city_birth"
        )
        city_of_birth = None if city_of_birth == "" else city_of_birth
        
        nacionality_options = {
            "Not specified": None,
            "1 - Brazilian": 1,
            "2 - Argentine": 2,
            "3 - Other": 3
        }
        nacionality_selection = st.selectbox(
            "Nationality",
            options=list(nacionality_options.keys()),
            help="Nationality (optional)",
            key="nacionality_select"
        )
        nacionality = nacionality_options[nacionality_selection]
    
    # ========== SECTION 2: FINANCIAL INFORMATION ==========
    st.markdown("""
    <div style="margin-top: 0.75rem; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
        <h3 style="color: #FFFFFF; font-weight: 600; font-size: 1.3rem; margin: 0;">
            FINANCIAL INFORMATION
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Ingreso mensual personal - sin límite superior, con alerta si excede umbral
        income_limits = UI_OPTIONS.get("PERSONAL_MONTHLY_INCOME", {"min": 205.0, "max": 3678.22})
        income_max = float(income_limits.get("max", 3678.22))
        income_threshold = income_max * 1.25  # Alerta después de 25% más que el máximo del dataset
        personal_income = st.number_input(
            "Personal Monthly Income (R$) *",
            min_value=0.0,
            value=float(income_limits.get("median", 1500)),
            step=100.0,
            help=f"Applicant's regular monthly income",
            key="personal_income"
        )
        # Mostrar alerta si excede el umbral
        if personal_income > income_threshold:
            st.warning(f"⚠️ The entered value (R$ {personal_income:,.2f}) is significantly higher than the maximum previously recorded (R$ {income_max:.2f}).")
        
        # Otros ingresos - sin límite superior, con alerta si excede umbral
        other_income_limits = UI_OPTIONS.get("OTHER_INCOMES", {"min": 0.0, "max": 800.0})
        other_income_max = float(other_income_limits.get("max", 800.0))
        other_income_threshold = other_income_max * 1.25  # Alerta después de 25% más que el máximo
        other_incomes_input = st.number_input(
            "Other Monthly Income (R$)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help=f"Additional other income (optional)",
            key="other_incomes"
        )
        # Mostrar alerta si excede el umbral
        if other_incomes_input > other_income_threshold:
            st.warning(f"⚠️ The entered value (R$ {other_incomes_input:,.2f}) is significantly higher than the maximum previously recorded (R$ {other_income_max:.2f}).")
        other_incomes = None if other_incomes_input == 0.0 else other_incomes_input
        
        # Valor de activos personales - sin límite superior, con alerta si excede umbral
        assets_limits = UI_OPTIONS.get("PERSONAL_ASSETS_VALUE", {"min": 0.0, "max": 50000.0})
        assets_max = float(assets_limits.get("max", 50000.0))
        assets_threshold = assets_max * 1.25  # Alerta después de 25% más que el máximo
        assets_input = st.number_input(
            "Personal Assets Value (R$)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help=f"Total value of properties, cars, etc. (optional)",
            key="assets_value"
        )
        # Mostrar alerta si excede el umbral
        if assets_input > assets_threshold:
            st.warning(f"⚠️ The entered value (R$ {assets_input:,.2f}) is significantly higher than the maximum previously recorded (R$ {assets_max:,.2f}).")
        assets_value = None if assets_input == 0.0 else assets_input
    
    with col5:
        # Cuentas bancarias con opciones
        banking_accounts_options = {
            "Not specified": None,
            "0": 0,
            "1": 1,
            "2": 2,
            "3 or more": 3
        }
        banking_accounts_selection = st.selectbox(
            "Number of Banking Accounts",
            options=list(banking_accounts_options.keys()),
            help="Number of banking accounts (optional)",
            key="banking_accounts_select"
        )
        quant_banking_accounts = banking_accounts_options[banking_accounts_selection]
        
        # Cuentas especiales con opciones
        special_accounts_options = {
            "Not specified": None,
            "0": 0,
            "1": 1,
            "2 or more": 2
        }
        special_accounts_selection = st.selectbox(
            "Special Banking Accounts",
            options=list(special_accounts_options.keys()),
            help="Number of special banking accounts (optional)",
            key="special_accounts_select"
        )
        quant_special_banking_accounts = special_accounts_options[special_accounts_selection]
        
        # Autos con opciones
        cars_options = {
            "Not specified": None,
            "0": 0,
            "1": 1,
            "2": 2,
            "3 or more": 3
        }
        cars_selection = st.selectbox(
            "Number of Cars",
            options=list(cars_options.keys()),
            help="Number of vehicles (optional)",
            key="cars_select"
        )
        quant_cars = cars_options[cars_selection]
    
    with col6:
        st.markdown("**Credit Cards**")
        # Primera fila: 3 checkboxes
        cc_col1, cc_col2, cc_col3 = st.columns(3)
        with cc_col1:
            flag_visa = st.checkbox("Visa", key="visa")
        with cc_col2:
            flag_mastercard = st.checkbox("Mastercard", key="mastercard")
        with cc_col3:
            flag_diners = st.checkbox("Diners", key="diners")
        
        # Segunda fila: 3 checkboxes
        cc_col4, cc_col5, cc_col6 = st.columns(3)
        with cc_col4:
            flag_amex = st.checkbox("American Express", key="amex")
        with cc_col5:
            flag_other_cards = st.checkbox("Other Cards", key="other_cards")
        with cc_col6:
            flag_email = st.checkbox("Has Email", key="email")
    
    # ========== SECTION 3: RESIDENCE INFORMATION ==========
    st.markdown("""
    <div style="margin-top: 0.75rem; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
        <h3 style="color: #FFFFFF; font-weight: 600; font-size: 1.3rem; margin: 0;">
            RESIDENCE INFORMATION
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col7, col8, col9 = st.columns(3)
    
    # Columna 1: Información geográfica básica
    with col7:
        # Estado de residencia - selectbox con opciones reales
        res_state_options = UI_OPTIONS.get("RESIDENCIAL_STATE", [])
        res_state_options_with_empty = [""] + res_state_options
        residencial_state = st.selectbox(
            "Residence State",
            options=res_state_options_with_empty,
            help="State where resides (optional). Select from available options.",
            key="res_state"
        )
        residencial_state = None if residencial_state == "" else residencial_state
        
        # Ciudad de residencia - text_input (alta cardinalidad: 3,529 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        residencial_city = st.text_input(
            "Residence City",
            value="",
            help="City where resides (optional). Leave empty if unknown or enter exact name. If not in dataset, will be handled automatically.",
            key="res_city"
        )
        residencial_city = None if residencial_city == "" else residencial_city
        
        # Barrio de residencia - text_input (alta cardinalidad: 14,511 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        residencial_borough = st.text_input(
            "Residence Borough",
            value="",
            help="Borough where resides (optional). Leave empty if unknown or enter exact name. If not in dataset, will be handled automatically.",
            key="res_borough"
        )
        residencial_borough = None if residencial_borough == "" else residencial_borough
    
    # Columna 2: Información de contacto y código postal
    with col8:
        # Código de área teléfono residencial - selectbox (alta cardinalidad: 102 categorías)
        # Usa Frequency Encoding, valores desconocidos se manejan automáticamente
        res_phone_area_options = UI_OPTIONS.get("RESIDENCIAL_PHONE_AREA_CODE", [])
        res_phone_area_options_with_empty = [""] + sorted(res_phone_area_options)
        residencial_phone_area_code = st.selectbox(
            "Residential Phone Area Code",
            options=res_phone_area_options_with_empty,
            help="Area code (optional). Select from available options or leave empty if unknown.",
            key="res_phone_area"
        )
        residencial_phone_area_code = None if residencial_phone_area_code == "" else residencial_phone_area_code
        
        # Código postal - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        res_zip_options = UI_OPTIONS.get("RESIDENCIAL_ZIP_3", [])
        if len(res_zip_options) > 1000:
            # Si hay demasiadas, usar text_input
            residencial_zip_3 = st.text_input(
                "Zip Code (first 3 digits)",
                value="",
                help="Zip code (optional). Leave empty if unknown or enter first 3 digits. If not in dataset, will be handled automatically.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
        else:
            res_zip_options_with_empty = [""] + sorted(res_zip_options)
            residencial_zip_3 = st.selectbox(
                "Zip Code (first 3 digits)",
                options=res_zip_options_with_empty,
                help="Zip code (optional). Select from available options or leave empty if unknown.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
        
        flag_residential_phone = st.selectbox(
            "Residential Phone *",
            options=["Y", "N"],
            help="Has residential phone?",
            key="residential_phone"
        )
    
    # Columna 3: Tipo de residencia y tiempo
    with col9:
        residence_type = st.selectbox(
            "Residence Type",
            options=["", "1 - Owned", "2 - Rented", "3 - Loaned", "4 - With Family", "5 - Other"],
            help="Current residence type (optional)",
            key="residence_type"
        )
        
        # Meses en residencia con opciones
        months_residence_options = {
            "Not specified": None,
            "Less than 6 months": 3,
            "6 months - 1 year": 9,
            "1 - 2 years": 18,
            "2 - 3 years": 30,
            "More than 3 years": 45
        }
        months_residence_selection = st.selectbox(
            "Months in Current Residence",
            options=list(months_residence_options.keys()),
            help="Time living in current residence (optional)",
            key="months_residence_select"
        )
        months_in_residence = months_residence_options[months_residence_selection]
        
        postal_address_options = {
            "Not specified": None,
            "1 - Residential": 1,
            "2 - Commercial": 2,
            "3 - Other": 3
        }
        postal_address_selection = st.selectbox(
            "Postal Address Type",
            options=list(postal_address_options.keys()),
            help="Postal address type (optional)",
            key="postal_address_select"
        )
        postal_address_type = postal_address_options[postal_address_selection]
    
    # ========== SECTION 4: EMPLOYMENT INFORMATION ==========
    st.markdown("""
    <div style="margin-top: 0.75rem; margin-bottom: 0.5rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
        <h3 style="color: #FFFFFF; font-weight: 600; font-size: 1.3rem; margin: 0;">
            EMPLOYMENT INFORMATION
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    col10, col11, col12 = st.columns(3)
    
    with col10:
        company = st.selectbox(
            "Has Company/Formal Employment? *",
            options=["Y", "N"],
            help="Did provide company name where works?",
            key="company"
        )
        
        # Estado profesional - selectbox con opciones reales
        prof_state_options = UI_OPTIONS.get("PROFESSIONAL_STATE", [])
        prof_state_options_with_empty = [""] + prof_state_options
        professional_state = st.selectbox(
            "Professional State",
            options=prof_state_options_with_empty,
            help="State where works (optional). Select from available options.",
            key="prof_state"
        )
        professional_state = None if professional_state == "" else professional_state
        
        # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing
        # porque tienen alta cardinalidad y muchos missing values
    
    with col11:
        flag_professional_phone = st.selectbox(
            "Professional Phone *",
            options=["Y", "N"],
            help="Has professional phone?",
            key="professional_phone"
        )
        
        # Código de área del teléfono profesional - selectbox (alta cardinalidad: 87 categorías)
        # Usa agrupación + OneHot, valores desconocidos se manejan automáticamente
        prof_phone_area_options = UI_OPTIONS.get("PROFESSIONAL_PHONE_AREA_CODE", [])
        prof_phone_area_options_with_empty = [""] + sorted(prof_phone_area_options)
        professional_phone_area_code = st.selectbox(
            "Professional Phone Area Code",
            options=prof_phone_area_options_with_empty,
            help="Area code (optional). Select from available options or leave empty if unknown.",
            key="prof_phone_area"
        )
        professional_phone_area_code = None if professional_phone_area_code == "" else professional_phone_area_code
        
        # Código postal profesional - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        prof_zip_options = UI_OPTIONS.get("PROFESSIONAL_ZIP_3", [])
        if len(prof_zip_options) > 1000:
            professional_zip_3 = st.text_input(
                "Professional Zip Code (first 3 digits)",
                value="",
                help="Zip code (optional). Leave empty if unknown or enter first 3 digits. If not in dataset, will be handled automatically.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        else:
            prof_zip_options_with_empty = [""] + sorted(prof_zip_options)
            professional_zip_3 = st.selectbox(
                "Professional Zip Code (first 3 digits)",
                options=prof_zip_options_with_empty,
                help="Zip code (optional). Select from available options or leave empty if unknown.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        
        # Meses en trabajo con opciones
        months_job_options = {
            "Not specified": None,
            "Less than 6 months": 3,
            "6 months - 1 year": 9,
            "1 - 2 years": 18,
            "2 - 3 years": 30,
            "More than 3 years": 45
        }
        months_job_selection = st.selectbox(
            "Months in Current Job",
            options=list(months_job_options.keys()),
            help="Time working in current job (optional)",
            key="months_job_select"
        )
        months_in_job = months_job_options[months_job_selection]
    
    with col12:
        # Código de profesión con opciones comunes
        profession_code_options = {
            "Not specified": None,
            "1 - Professional": 1,
            "2 - Technical": 2,
            "3 - Administrative": 3,
            "4 - Commercial": 4,
            "5 - Services": 5,
            "6 - Operator": 6,
            "7 - Other": 7
        }
        profession_code_selection = st.selectbox(
            "Profession Code",
            options=list(profession_code_options.keys()),
            help="Profession code (optional)",
            key="profession_select"
        )
        profession_code = profession_code_options[profession_code_selection]
        
        occupation_type = st.selectbox(
            "Occupation Type",
            options=["", "1 - Employee", "2 - Self-employed", "3 - Business Owner", "4 - Unemployed", "5 - Other"],
            help="Occupation type (optional)",
            key="occupation"
        )
        
        mate_profession_options = {
            "No especificado": None,
            "1 - Profesional": 1,
            "2 - Técnico": 2,
            "3 - Administrativo": 3,
            "4 - Comercial": 4,
            "5 - Servicios": 5,
            "6 - Operario": 6,
            "7 - Otro": 7
        }
        mate_profession_selection = st.selectbox(
            "Código de profesión del cónyuge",
            options=list(mate_profession_options.keys()),
            help="Código de profesión del cónyuge (opcional)",
            key="mate_profession_select"
        )
        mate_profession_code = mate_profession_options[mate_profession_selection]
        
        education_level_options = {
            "Not specified": None,
            "1 - Primary": 1,
            "2 - Secondary": 2,
            "3 - Tertiary": 3,
            "4 - University": 4
        }
        education_level_selection = st.selectbox(
            "Spouse Education Level",
            options=list(education_level_options.keys()),
            help="Spouse education level (optional)",
            key="education_level_1_select"
        )
        education_level_1 = education_level_options[education_level_selection]
    
    # Submit button
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Evaluate Credit Risk", type="primary", use_container_width=True)
    
    if submitted:
        # Preparar payload con TODAS las features del formulario
        payload = {
            # Requeridas
            "PAYMENT_DAY": payment_day,
            "APPLICATION_SUBMISSION_TYPE": application_type,
            "SEX": sex,
            "AGE": age,
            "QUANT_DEPENDANTS": quant_dependants,
            "PERSONAL_MONTHLY_INCOME": float(personal_income),
            "FLAG_RESIDENCIAL_PHONE": flag_residential_phone,
            "COMPANY": company,
            "FLAG_PROFESSIONAL_PHONE": flag_professional_phone,
        }
        
        # Opcionales - Información personal
        if marital_status and marital_status != "":
            marital_num = marital_status.split(" -")[0] if " -" in marital_status else marital_status
            payload["MARITAL_STATUS"] = int(marital_num)
        if state_of_birth:
            payload["STATE_OF_BIRTH"] = state_of_birth
        if city_of_birth:
            payload["CITY_OF_BIRTH"] = city_of_birth
        if nacionality is not None:
            payload["NACIONALITY"] = int(nacionality)
        
        # Opcionales - Financieras
        if other_incomes is not None:
            payload["OTHER_INCOMES"] = float(other_incomes)
        if assets_value is not None:
            payload["PERSONAL_ASSETS_VALUE"] = float(assets_value)
        if quant_banking_accounts is not None:
            payload["QUANT_BANKING_ACCOUNTS"] = int(quant_banking_accounts)
        if quant_special_banking_accounts is not None:
            payload["QUANT_SPECIAL_BANKING_ACCOUNTS"] = int(quant_special_banking_accounts)
        if quant_cars is not None:
            payload["QUANT_CARS"] = int(quant_cars)
        
        # Tarjetas
        payload["FLAG_VISA"] = 1 if flag_visa else 0
        payload["FLAG_MASTERCARD"] = 1 if flag_mastercard else 0
        payload["FLAG_DINERS"] = 1 if flag_diners else 0
        payload["FLAG_AMERICAN_EXPRESS"] = 1 if flag_amex else 0
        payload["FLAG_OTHER_CARDS"] = 1 if flag_other_cards else 0
        payload["FLAG_EMAIL"] = 1 if flag_email else 0
        
        # Opcionales - Residencia
        if residencial_state:
            payload["RESIDENCIAL_STATE"] = residencial_state
        if residencial_city:
            payload["RESIDENCIAL_CITY"] = residencial_city
        if residencial_borough:
            payload["RESIDENCIAL_BOROUGH"] = residencial_borough
        if residencial_phone_area_code:
            payload["RESIDENCIAL_PHONE_AREA_CODE"] = residencial_phone_area_code
        if residencial_zip_3:
            payload["RESIDENCIAL_ZIP_3"] = residencial_zip_3
        if residence_type and residence_type != "":
            residence_num = residence_type.split(" -")[0] if " -" in residence_type else residence_type
            payload["RESIDENCE_TYPE"] = int(residence_num)
        if months_in_residence is not None:
            payload["MONTHS_IN_RESIDENCE"] = float(months_in_residence)
        if postal_address_type is not None:
            payload["POSTAL_ADDRESS_TYPE"] = int(postal_address_type)
        
        # Opcionales - Empleo
        if professional_state:
            payload["PROFESSIONAL_STATE"] = professional_state
        # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing
        if professional_phone_area_code:
            payload["PROFESSIONAL_PHONE_AREA_CODE"] = professional_phone_area_code
        if professional_zip_3:
            payload["PROFESSIONAL_ZIP_3"] = professional_zip_3
        if months_in_job is not None:
            payload["MONTHS_IN_THE_JOB"] = float(months_in_job)
        if profession_code is not None:
            payload["PROFESSION_CODE"] = int(profession_code)
        if occupation_type and occupation_type != "":
            occupation_num = occupation_type.split(" -")[0] if " -" in occupation_type else occupation_type
            payload["OCCUPATION_TYPE"] = int(occupation_num)
        if mate_profession_code is not None:
            payload["MATE_PROFESSION_CODE"] = int(mate_profession_code)
        if education_level_1 is not None:
            payload["MATE_EDUCATION_LEVEL"] = int(education_level_1)
        if product is not None:
            payload["PRODUCT"] = int(product)
        
        # Enviar request usando la función helper
        send_prediction_and_display(payload, None)

# ========== SECTION: PRE-LOAD PROFILES ==========
st.markdown("""
<div style="background: rgba(22, 21, 38, 0.8);
            border-radius: 24px;
            padding: 2rem;
            border: 1px solid rgba(138, 43, 226, 0.4);
            box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(138, 43, 226, 0.2);
            backdrop-filter: blur(15px);
            margin: 2rem 0 1rem 0;">
    <h3 style="color: #FFFFFF; font-weight: 600; font-size: 1.3rem; margin: 0 0 1.5rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(138, 43, 226, 0.2);">
        PRE-LOAD PROFILES
    </h3>
""", unsafe_allow_html=True)

# Botones de perfiles
col_prof1, col_prof2, col_prof3 = st.columns(3)

profile_selected = None
profile_payload = None

with col_prof1:
    if st.button("Martin (High Risk)", type="secondary", use_container_width=True, key="btn_profile_martin"):
        profile_data = get_profile_data("martin")
        if profile_data:
            profile_selected = PROFILES["martin"]["name"]
            profile_payload = profile_data

with col_prof2:
    if st.button("Gonzalo (Medium Risk)", type="secondary", use_container_width=True, key="btn_profile_gonzalo"):
        profile_data = get_profile_data("gonzalo")
        if profile_data:
            profile_selected = PROFILES["gonzalo"]["name"]
            profile_payload = profile_data

with col_prof3:
    if st.button("Martina (Low Risk)", type="secondary", use_container_width=True, key="btn_profile_martina"):
        profile_data = get_profile_data("martina")
        if profile_data:
            profile_selected = PROFILES["martina"]["name"]
            profile_payload = profile_data

# Manejar la selección del perfil fuera del contenedor
if profile_selected and profile_payload:
    send_prediction_and_display(profile_payload, profile_selected)

# Footer con estilo mejorado
st.markdown(f"""
<div style="text-align: center; padding: 1.5rem 0; color: #B0B0B0;">
    <p style="margin: 0.5rem 0;">API URL: <code style="background: rgba(138, 43, 226, 0.2); padding: 0.25rem 0.5rem; border-radius: 6px;">{API_URL}</code></p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">Note: Fields marked with * are required. Complete all available fields for a more accurate evaluation.</p>
</div>
""", unsafe_allow_html=True)
