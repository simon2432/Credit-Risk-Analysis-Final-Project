"""
Credit Risk Analysis - Streamlit UI Completa
Incluye todas las features necesarias para el modelo.
Excluye: 9 columnas constantes (se rellenan automáticamente) y 5 columnas removidas en preprocessing 
(CITY_OF_BIRTH, RESIDENCIAL_CITY, RESIDENCIAL_BOROUGH, PROFESSIONAL_CITY, PROFESSIONAL_BOROUGH).
Estas columnas se completan automáticamente con None y luego se eliminan en el preprocessing.
"""

import streamlit as st
import requests
import os
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go

# API URL - use environment variable or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Cargar opciones de UI desde JSON
UI_OPTIONS_FILE = Path(__file__).parent / "ui_options.json"
try:
    with open(UI_OPTIONS_FILE, "r", encoding="utf-8") as f:
        UI_OPTIONS = json.load(f)
except FileNotFoundError:
    st.error(f"Error: File {UI_OPTIONS_FILE} not found. Run: python -m src.ui.extract_ui_options")
    UI_OPTIONS = {}

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for dark theme with neon violet borders
st.markdown("""
<style>
    /* Dark theme background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    /* Main title styling */
    h1 {
        color: #b794f6 !important;
        text-shadow: 0 0 10px rgba(183, 148, 246, 0.5);
        border-bottom: 2px solid #b794f6;
        padding-bottom: 10px;
    }
    
    /* Subheaders with neon violet borders */
    h2, h3 {
        color: #c4b5fd !important;
        border-left: 4px solid #b794f6;
        padding-left: 15px;
        margin-top: 20px;
    }
    
    /* Form containers with neon borders */
    .stForm {
        background: rgba(30, 30, 50, 0.8);
        border: 1px solid #b794f6;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(183, 148, 246, 0.3);
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: rgba(20, 20, 40, 0.9) !important;
        color: #e0e0e0 !important;
        border: 1px solid #7c3aed !important;
        border-radius: 4px;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border: 2px solid #b794f6 !important;
        box-shadow: 0 0 10px rgba(183, 148, 246, 0.5) !important;
    }
    
    /* Buttons with neon violet */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #b794f6 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(183, 148, 246, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(183, 148, 246, 0.6);
        transform: translateY(-2px);
    }
    
    /* Primary button */
    button[kind="primary"] {
        background: linear-gradient(135deg, #b794f6 0%, #7c3aed 100%) !important;
        box-shadow: 0 0 20px rgba(183, 148, 246, 0.6) !important;
    }
    
    /* Success/Error/Info boxes */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10b981;
        border-radius: 4px;
    }
    
    .stError {
        background-color: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #ef4444;
        border-radius: 4px;
    }
    
    .stWarning {
        background-color: rgba(245, 158, 11, 0.2);
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
    }
    
    .stInfo {
        background-color: rgba(59, 130, 246, 0.2);
        border-left: 4px solid #3b82f6;
        border-radius: 4px;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #b794f6 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #c4b5fd !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #7c3aed 0%, #b794f6 100%);
    }
    
    /* Checkboxes */
    .stCheckbox {
        color: #e0e0e0;
    }
    
    /* Labels */
    label {
        color: #c4b5fd !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #9ca3af !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(30, 30, 50, 0.6);
        color: #c4b5fd !important;
        border: 1px solid #7c3aed;
        border-radius: 4px;
    }
    
    /* Divider */
    hr {
        border-color: #7c3aed;
        opacity: 0.5;
    }
    
    /* JSON viewer */
    .stJson {
        background-color: rgba(20, 20, 40, 0.9);
        border: 1px solid #7c3aed;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Credit Risk Analysis")
st.markdown("**Credit risk assessment system using Machine Learning**")
st.markdown("Complete all available fields for a more accurate evaluation. Fields marked with * are required.")

st.markdown("---")

# ========== FUNCIÓN PARA CREAR GAUGE ==========
def create_risk_gauge(probability):
    """Creates a circular gauge to display risk"""
    # Convert probability to percentage (0-100)
    risk_percentage = probability * 100
    
    # Crear el gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_percentage,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "RISK", 'font': {'size': 24, 'color': '#ffffff'}},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#ffffff"},
            'bar': {'color': "#8B5CF6"},  # Violeta neon
            'bgcolor': "rgba(0,0,0,0)",  # Transparente
            'borderwidth': 2,
            'bordercolor': "#8B5CF6",  # Violeta neon
            'steps': [
                {'range': [0, 30], 'color': '#10b981'},  # Green (low risk)
                {'range': [30, 50], 'color': '#f59e0b'},  # Yellow (moderate risk)
                {'range': [50, 70], 'color': '#f97316'},  # Orange (moderate-high risk)
                {'range': [70, 100], 'color': '#ef4444'}  # Red (high risk)
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 4},
                'thickness': 0.75,
                'value': risk_percentage
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",  # Transparente
        plot_bgcolor="rgba(0,0,0,0)",  # Transparente
        font={'color': "#ffffff"},
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ========== PREDEFINED PROFILES ==========
def get_martin_profile():
    """High risk profile - should be rejected"""
    return {
        "PAYMENT_DAY": 25,
        "APPLICATION_SUBMISSION_TYPE": "Carga",
        "SEX": "M",
        "AGE": 25,
        "QUANT_DEPENDANTS": 4,
        "PERSONAL_MONTHLY_INCOME": 800.0,
        "FLAG_RESIDENCIAL_PHONE": "N",
        "COMPANY": "N",
        "FLAG_PROFESSIONAL_PHONE": "N",
        "MARITAL_STATUS": 1,
        "STATE_OF_BIRTH": "SP",
        "NACIONALITY": 1,
        "OTHER_INCOMES": None,
        "PERSONAL_ASSETS_VALUE": None,
        "QUANT_BANKING_ACCOUNTS": 0,
        "QUANT_SPECIAL_BANKING_ACCOUNTS": 0,
        "QUANT_CARS": 0,
        "FLAG_VISA": 0,
        "FLAG_MASTERCARD": 0,
        "FLAG_DINERS": 0,
        "FLAG_AMERICAN_EXPRESS": 0,
        "FLAG_OTHER_CARDS": 0,
        "FLAG_EMAIL": 0,
        "RESIDENCIAL_STATE": "SP",
        "RESIDENCE_TYPE": 2,
        "MONTHS_IN_RESIDENCE": 3.0,
        "PROFESSION_CODE": 6,
        "OCCUPATION_TYPE": 2,
        "MONTHS_IN_THE_JOB": 3.0,
    }

def get_martina_profile():
    """Low risk profile - should be approved"""
    return {
        "PAYMENT_DAY": 5,
        "APPLICATION_SUBMISSION_TYPE": "Web",
        "SEX": "F",
        "AGE": 38,
        "QUANT_DEPENDANTS": 1,
        "PERSONAL_MONTHLY_INCOME": 3500.0,
        "FLAG_RESIDENCIAL_PHONE": "Y",
        "COMPANY": "Y",
        "FLAG_PROFESSIONAL_PHONE": "Y",
        "MARITAL_STATUS": 2,
        "STATE_OF_BIRTH": "SP",
        "NACIONALITY": 1,
        "OTHER_INCOMES": 500.0,
        "PERSONAL_ASSETS_VALUE": 45000.0,
        "QUANT_BANKING_ACCOUNTS": 2,
        "QUANT_SPECIAL_BANKING_ACCOUNTS": 1,
        "QUANT_CARS": 2,
        "FLAG_VISA": 1,
        "FLAG_MASTERCARD": 1,
        "FLAG_DINERS": 0,
        "FLAG_AMERICAN_EXPRESS": 1,
        "FLAG_OTHER_CARDS": 0,
        "FLAG_EMAIL": 1,
        "RESIDENCIAL_STATE": "SP",
        "RESIDENCIAL_PHONE_AREA_CODE": "11",
        "RESIDENCE_TYPE": 1,
        "MONTHS_IN_RESIDENCE": 48.0,
        "POSTAL_ADDRESS_TYPE": 1,
        "PROFESSIONAL_STATE": "SP",
        "PROFESSIONAL_PHONE_AREA_CODE": "11",
        "MONTHS_IN_THE_JOB": 48.0,
        "PROFESSION_CODE": 1,
        "OCCUPATION_TYPE": 1,
        "MATE_PROFESSION_CODE": 1,
        "MATE_EDUCATION_LEVEL": 4,
        "PRODUCT": 2,
    }

def get_gonzalo_profile():
    """Intermediate risk profile - borderline case"""
    return {
        "PAYMENT_DAY": 15,
        "APPLICATION_SUBMISSION_TYPE": "Web",
        "SEX": "M",
        "AGE": 32,
        "QUANT_DEPENDANTS": 2,
        "PERSONAL_MONTHLY_INCOME": 2000.0,
        "FLAG_RESIDENCIAL_PHONE": "Y",
        "COMPANY": "Y",
        "FLAG_PROFESSIONAL_PHONE": "Y",
        "MARITAL_STATUS": 2,
        "STATE_OF_BIRTH": "SP",
        "NACIONALITY": 1,
        "OTHER_INCOMES": 200.0,
        "PERSONAL_ASSETS_VALUE": 15000.0,
        "QUANT_BANKING_ACCOUNTS": 1,
        "QUANT_SPECIAL_BANKING_ACCOUNTS": 0,
        "QUANT_CARS": 1,
        "FLAG_VISA": 1,
        "FLAG_MASTERCARD": 0,
        "FLAG_DINERS": 0,
        "FLAG_AMERICAN_EXPRESS": 0,
        "FLAG_OTHER_CARDS": 0,
        "FLAG_EMAIL": 1,
        "RESIDENCIAL_STATE": "SP",
        "RESIDENCIAL_PHONE_AREA_CODE": "11",
        "RESIDENCE_TYPE": 1,
        "MONTHS_IN_RESIDENCE": 24.0,
        "POSTAL_ADDRESS_TYPE": 1,
        "PROFESSIONAL_STATE": "SP",
        "PROFESSIONAL_PHONE_AREA_CODE": "11",
        "MONTHS_IN_THE_JOB": 18.0,
        "PROFESSION_CODE": 3,
        "OCCUPATION_TYPE": 1,
        "MATE_PROFESSION_CODE": 4,
        "MATE_EDUCATION_LEVEL": 3,
        "PRODUCT": 1,
    }

# Función para enviar predicción y mostrar resultados
def send_prediction_and_display(payload, profile_name=None, return_result=False):
    """Sends prediction to API and displays results
    
    Args:
        payload: Dictionary with client data
        profile_name: Profile name (optional)
        return_result: If True, returns the result instead of displaying it
    
    Returns:
        If return_result=True, returns dict with profile_name and result. Otherwise, returns None.
    """
    with st.spinner("Evaluating credit risk..."):
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction", "unknown")
                probability = result.get("probability", 0.0)
                confidence = result.get("confidence", "low")
                
                # If result return is requested (for pre-loaded profiles)
                if return_result:
                    return {
                        "profile_name": profile_name,
                        "result": result,
                        "payload": payload
                    }
                
                # Mostrar resultado normalmente (para formulario manual)
                st.markdown("---")
                if profile_name:
                    st.subheader(f"Evaluation Result - {profile_name}")
                else:
                    st.subheader("Evaluation Result")
                
                # Layout con gauge a la izquierda y cajas de información a la derecha
                col_gauge, col_info = st.columns([1, 1])
                
                with col_gauge:
                    # Mostrar el gauge
                    fig = create_risk_gauge(probability)
                    st.plotly_chart(fig, use_container_width=True, key=f"gauge_form_{profile_name or 'manual'}")
                    st.markdown('<p style="text-align: center; color: #ffffff; margin-top: -20px;">PERCENTAGE</p>', unsafe_allow_html=True)
                
                with col_info:
                    # Caja de aprobación/rechazo
                    if prediction.lower() == "approved":
                        st.markdown("""
                        <div style="background-color: #10b981; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <span style="font-size: 24px;">✓</span>
                                <h3 style="color: #ffffff; margin: 0;">APPROVED</h3>
                            </div>
                            <p style="color: #ffffff; margin: 0;">Credit application was approved</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prediction.lower() == "rejected":
                        st.markdown("""
                        <div style="background-color: #ef4444; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <span style="font-size: 24px;">✗</span>
                                <h3 style="color: #ffffff; margin: 0;">REJECTED</h3>
                            </div>
                            <p style="color: #ffffff; margin: 0;">Credit application was rejected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #f59e0b; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                <span style="font-size: 24px;">?</span>
                                <h3 style="color: #ffffff; margin: 0;">UNKNOWN</h3>
                            </div>
                            <p style="color: #ffffff; margin: 0;">Prediction: {}</p>
                        </div>
                        """.format(prediction), unsafe_allow_html=True)
                    
                    # Confianza
                    confidence_labels = {
                        "high": "High",
                        "medium": "Medium",
                        "low": "Low"
                    }
                    confidence_display = confidence_labels.get(confidence, confidence.capitalize())
                    st.markdown(f"""
                    <div style="margin-bottom: 15px;">
                        <p style="color: #ffffff; margin-bottom: 5px; font-size: 14px;">Confidence <span style="color: #8B5CF6;">?</span></p>
                        <h2 style="color: #ffffff; margin: 0; font-size: 32px;">{confidence_display}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk interpretation box (smaller and subtle)
                    if probability >= 0.7:
                        risk_level = "HIGH RISK"
                        risk_color_bg = "rgba(239, 68, 68, 0.15)"  # Rojo más transparente
                        risk_color_border = "rgba(239, 68, 68, 0.4)"
                        interpretation_text = f"Default probability of {probability:.1%}. It is recommended to reject the application."
                    elif probability >= 0.5:
                        risk_level = "MODERATE-HIGH RISK"
                        risk_color_bg = "rgba(249, 115, 22, 0.15)"  # Naranja más transparente
                        risk_color_border = "rgba(249, 115, 22, 0.4)"
                        interpretation_text = f"Default probability of {probability:.1%}. Additional review required."
                    elif probability >= 0.3:
                        risk_level = "MODERATE RISK"
                        risk_color_bg = "rgba(245, 158, 11, 0.15)"  # Amarillo más transparente
                        risk_color_border = "rgba(245, 158, 11, 0.4)"
                        interpretation_text = f"Default probability of {probability:.1%}. Careful evaluation recommended."
                    else:
                        risk_level = "LOW RISK"
                        risk_color_bg = "rgba(16, 185, 129, 0.15)"  # Verde más transparente
                        risk_color_border = "rgba(16, 185, 129, 0.4)"
                        interpretation_text = f"Default probability of {probability:.1%}. Client with good credit profile."
                    
                    st.markdown(f"""
                    <div style="background-color: {risk_color_bg}; border: 1px solid {risk_color_border}; border-radius: 8px; padding: 12px; margin-top: 10px;">
                        <h4 style="color: #c4b5fd; margin: 0 0 6px 0; font-size: 14px; font-weight: 600;">{risk_level}</h4>
                        <p style="color: #e0e0e0; margin: 0; font-size: 12px;">{interpretation_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Expandable details
                with st.expander("Technical Details of the Query"):
                    st.write("**Payload sent to API:**")
                    st.json(payload)
                    st.write("**Complete API response:**")
                    st.json(result)
            
            else:
                if return_result:
                    return None
                st.error(f"Error de API: Status {response.status_code}")
                st.text(response.text)
        
        except requests.exceptions.ConnectionError:
            if return_result:
                return None
            st.error(f"**Connection Error** - Could not connect to API at {API_URL}")
            st.info("Make sure the API is running. Check with: `docker-compose ps`")
        
        except requests.exceptions.Timeout:
            if return_result:
                return None
            st.error("**Timeout** - The API took too long to respond")
        
        except Exception as e:
            if return_result:
                return None
            st.error(f"**Error**: {str(e)}")
    
    return None

# Formulario completo con TODAS las features
with st.form("credit_risk_form"):
    
    # ========== SECTION 1: BASIC INFORMATION AND APPLICATION ==========
    st.subheader("Basic Information and Application")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        payment_day = st.number_input(
            "Payment day *",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
            help="Day of the month chosen for payment (1-31)",
            key="payment_day"
        )
        
        application_type = st.selectbox(
            "Application type *",
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
            "Product type",
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
            "Number of dependents *",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Number of people who depend on the applicant",
            key="dependants"
        )
    
    with col3:
        marital_status = st.selectbox(
            "Marital status",
            options=["", "1 - Single", "2 - Married", "3 - Divorced", "4 - Widowed", "5 - Common-law", "6 - Separated", "7 - Other"],
            help="Applicant's marital status (optional)",
            key="marital"
        )
        
        # State of birth - selectbox with real options
        state_birth_options = UI_OPTIONS.get("STATE_OF_BIRTH", [])
        state_birth_options_with_empty = [""] + state_birth_options
        state_of_birth = st.selectbox(
            "State of birth",
            options=state_birth_options_with_empty,
            help="State where born (optional). Select from available options in the dataset.",
            key="state_birth"
        )
        state_of_birth = None if state_of_birth == "" else state_of_birth
        
        # NOTA: CITY_OF_BIRTH se elimina en preprocessing (alta cardinalidad: 9,910 categorías)
        # Se completa automáticamente con None en el feature_mapper
        
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
    
    st.markdown("---")
    
    # ========== SECTION 2: FINANCIAL INFORMATION ==========
    st.subheader("Financial Information")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Ingreso mensual personal - sin límite superior, con alerta si excede umbral
        income_limits = UI_OPTIONS.get("PERSONAL_MONTHLY_INCOME", {"min": 205.0, "max": 3678.22})
        income_max = float(income_limits.get("max", 3678.22))
        income_threshold = income_max * 1.25  # Alerta después de 25% más que el máximo del dataset
        personal_income = st.number_input(
            "Personal monthly income (R$) *",
            min_value=0.0,
            value=float(income_limits.get("median", 1500)),
            step=100.0,
            help=f"Applicant's regular monthly income",
            key="personal_income"
        )
        # Show alert if exceeds threshold
        if personal_income > income_threshold:
            st.warning(f"⚠️ The entered value (R$ {personal_income:,.2f}) is significantly higher than the maximum previously recorded (R$ {income_max:.2f}).")
        
        # Otros ingresos - sin límite superior, con alerta si excede umbral
        other_income_limits = UI_OPTIONS.get("OTHER_INCOMES", {"min": 0.0, "max": 800.0})
        other_income_max = float(other_income_limits.get("max", 800.0))
        other_income_threshold = other_income_max * 1.25  # Alerta después de 25% más que el máximo
        other_incomes_input = st.number_input(
            "Other monthly income (R$)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help=f"Additional other income (optional)",
            key="other_incomes"
        )
        # Show alert if exceeds threshold
        if other_incomes_input > other_income_threshold:
            st.warning(f"⚠️ The entered value (R$ {other_incomes_input:,.2f}) is significantly higher than the maximum previously recorded (R$ {other_income_max:.2f}).")
        other_incomes = None if other_incomes_input == 0.0 else other_incomes_input
        
        # Valor de activos personales - sin límite superior, con alerta si excede umbral
        assets_limits = UI_OPTIONS.get("PERSONAL_ASSETS_VALUE", {"min": 0.0, "max": 50000.0})
        assets_max = float(assets_limits.get("max", 50000.0))
        assets_threshold = assets_max * 1.25  # Alerta después de 25% más que el máximo
        assets_input = st.number_input(
            "Personal assets value (R$)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help=f"Total value of properties, cars, etc. (optional)",
            key="assets_value"
        )
        # Show alert if exceeds threshold
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
            "Number of banking accounts",
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
            "Special banking accounts",
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
            "Number of cars",
            options=list(cars_options.keys()),
            help="Number of vehicles (optional)",
            key="cars_select"
        )
        quant_cars = cars_options[cars_selection]
    
    with col6:
        st.markdown("**Credit Cards**")
        col_cards1, col_cards2 = st.columns(2)
        with col_cards1:
            flag_visa = st.checkbox("Visa", key="visa")
            flag_mastercard = st.checkbox("Mastercard", key="mastercard")
            flag_diners = st.checkbox("Diners", key="diners")
        with col_cards2:
            flag_amex = st.checkbox("American Express", key="amex")
            flag_other_cards = st.checkbox("Other cards", key="other_cards")
            flag_email = st.checkbox("Has email", key="email")
    
    st.markdown("---")
    
    # ========== SECTION 3: RESIDENCE ==========
    st.subheader("Residence Information")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        # Estado de residencia - selectbox con opciones reales
        res_state_options = UI_OPTIONS.get("RESIDENCIAL_STATE", [])
        res_state_options_with_empty = [""] + res_state_options
        residencial_state = st.selectbox(
            "State of residence",
            options=res_state_options_with_empty,
            help="State where resides (optional). Select from available options.",
            key="res_state"
        )
        residencial_state = None if residencial_state == "" else residencial_state
        
        # NOTA: RESIDENCIAL_CITY y RESIDENCIAL_BOROUGH se eliminan en preprocessing 
        # (alta cardinalidad: 3,529 y 14,511 categorías respectivamente)
        # Se completan automáticamente con None en el feature_mapper
        
        flag_residential_phone = st.selectbox(
            "Residential phone *",
            options=["Y", "N"],
            help="Does the applicant have a residential phone?",
            key="residential_phone"
        )
        
        residence_type = st.selectbox(
            "Residence type",
            options=["", "1 - Owned", "2 - Rented", "3 - Loaned", "4 - With family", "5 - Other"],
            help="Current residence type (optional)",
            key="residence_type"
        )
    
    with col8:
        # Código de área teléfono residencial - selectbox (alta cardinalidad: 102 categorías)
        # Usa Frequency Encoding, valores desconocidos se manejan automáticamente
        res_phone_area_options = UI_OPTIONS.get("RESIDENCIAL_PHONE_AREA_CODE", [])
        res_phone_area_options_with_empty = [""] + sorted(res_phone_area_options)
        residencial_phone_area_code = st.selectbox(
            "Residential phone area code",
            options=res_phone_area_options_with_empty,
            help="Area code (optional). Select from available options or leave empty if unknown.",
            key="res_phone_area"
        )
        residencial_phone_area_code = None if residencial_phone_area_code == "" else residencial_phone_area_code
        
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
            "Months in current residence",
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
            "Postal address type",
            options=list(postal_address_options.keys()),
            help="Postal address type (optional)",
            key="postal_address_select"
        )
        postal_address_type = postal_address_options[postal_address_selection]
    
    with col9:
        # Código postal - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        res_zip_options = UI_OPTIONS.get("RESIDENCIAL_ZIP_3", [])
        if len(res_zip_options) > 1000:
            # Si hay demasiadas, usar text_input
            residencial_zip_3 = st.text_input(
                "Postal code (first 3 digits)",
                value="",
                help="Postal code (optional). Leave empty if unknown or enter the first 3 digits. If it doesn't exist in the dataset, it will be handled automatically.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
        else:
            res_zip_options_with_empty = [""] + sorted(res_zip_options)
            residencial_zip_3 = st.selectbox(
                "Postal code (first 3 digits)",
                options=res_zip_options_with_empty,
                help="Postal code (optional). Select from available options or leave empty if unknown.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
    
    st.markdown("---")
    
    # ========== SECTION 4: EMPLOYMENT ==========
    st.subheader("Employment Information")
    
    col10, col11, col12 = st.columns(3)
    
    with col10:
        company = st.selectbox(
            "Has company/formal employment? *",
            options=["Y", "N"],
            help="Did the applicant provide a company name where they work?",
            key="company"
        )
        
        # Professional state - selectbox with real options
        prof_state_options = UI_OPTIONS.get("PROFESSIONAL_STATE", [])
        prof_state_options_with_empty = [""] + prof_state_options
        professional_state = st.selectbox(
            "Professional state",
            options=prof_state_options_with_empty,
            help="State where works (optional). Select from available options.",
            key="prof_state"
        )
        professional_state = None if professional_state == "" else professional_state
        
        # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing
        # porque tienen alta cardinalidad y muchos missing values
    
    with col11:
        flag_professional_phone = st.selectbox(
            "Professional phone *",
            options=["Y", "N"],
            help="Does the applicant have a professional phone?",
            key="professional_phone"
        )
        
        # Professional phone area code - selectbox (high cardinality: 87 categories)
        # Uses grouping + OneHot, unknown values are handled automatically
        prof_phone_area_options = UI_OPTIONS.get("PROFESSIONAL_PHONE_AREA_CODE", [])
        prof_phone_area_options_with_empty = [""] + sorted(prof_phone_area_options)
        professional_phone_area_code = st.selectbox(
            "Professional phone area code",
            options=prof_phone_area_options_with_empty,
            help="Area code (optional). Select from available options or leave empty if unknown.",
            key="prof_phone_area"
        )
        professional_phone_area_code = None if professional_phone_area_code == "" else professional_phone_area_code
        
        # Código postal profesional - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        prof_zip_options = UI_OPTIONS.get("PROFESSIONAL_ZIP_3", [])
        if len(prof_zip_options) > 1000:
            professional_zip_3 = st.text_input(
                "Professional postal code (first 3 digits)",
                value="",
                help="Postal code (optional). Leave empty if unknown or enter the first 3 digits. If it doesn't exist in the dataset, it will be handled automatically.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        else:
            prof_zip_options_with_empty = [""] + sorted(prof_zip_options)
            professional_zip_3 = st.selectbox(
                "Professional postal code (first 3 digits)",
                options=prof_zip_options_with_empty,
                help="Postal code (optional). Select from available options or leave empty if unknown.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        
        # Months in job with options
        months_job_options = {
            "Not specified": None,
            "Less than 6 months": 3,
            "6 months - 1 year": 9,
            "1 - 2 years": 18,
            "2 - 3 years": 30,
            "More than 3 years": 45
        }
        months_job_selection = st.selectbox(
            "Months in current job",
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
            "Profession code",
            options=list(profession_code_options.keys()),
            help="Profession code (optional)",
            key="profession_select"
        )
        profession_code = profession_code_options[profession_code_selection]
        
        occupation_type = st.selectbox(
            "Occupation type",
            options=["", "1 - Employee", "2 - Self-employed", "3 - Business owner", "4 - Unemployed", "5 - Other"],
            help="Type of occupation (optional)",
            key="occupation"
        )
        
        mate_profession_options = {
            "Not specified": None,
            "1 - Professional": 1,
            "2 - Technical": 2,
            "3 - Administrative": 3,
            "4 - Commercial": 4,
            "5 - Services": 5,
            "6 - Operator": 6,
            "7 - Other": 7
        }
        mate_profession_selection = st.selectbox(
            "Spouse profession code",
            options=list(mate_profession_options.keys()),
            help="Spouse profession code (optional)",
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
            "Spouse education level",
            options=list(education_level_options.keys()),
            help="Spouse education level (optional)",
            key="education_level_1_select"
        )
        education_level_1 = education_level_options[education_level_selection]
    
    st.markdown("---")
    
    # Submit button
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
        # NOTA: CITY_OF_BIRTH se elimina en preprocessing, se completa automáticamente con None
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
        # NOTA: RESIDENCIAL_CITY y RESIDENCIAL_BOROUGH se eliminan en preprocessing, se completan automáticamente con None
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

# Pre-loaded Profiles Section
st.markdown("---")
st.subheader("Pre-loaded Profiles")

st.markdown("Test the system with predefined profiles to see how credit risk evaluation works.")

# Inicializar session_state para almacenar resultados
if "profile_results" not in st.session_state:
    st.session_state.profile_results = []

col_profile1, col_profile2, col_profile3 = st.columns(3)

with col_profile1:
    st.markdown("**Martin - High Risk Profile**")
    st.markdown("Client profile with characteristics indicating high default risk. Should be rejected.")
    if st.button("Evaluate Profile: Martin", key="btn_martin", use_container_width=True):
        payload = get_martin_profile()
        result = send_prediction_and_display(payload, "Martin (High Risk)", return_result=True)
        if result:
            st.session_state.profile_results.append(result)
            st.rerun()

with col_profile2:
    st.markdown("**Gonzalo - Intermediate Risk Profile**")
    st.markdown("Client profile with characteristics indicating moderate risk. Borderline case for evaluation.")
    if st.button("Evaluate Profile: Gonzalo", key="btn_gonzalo", use_container_width=True):
        payload = get_gonzalo_profile()
        result = send_prediction_and_display(payload, "Gonzalo (Intermediate Risk)", return_result=True)
        if result:
            st.session_state.profile_results.append(result)
            st.rerun()

with col_profile3:
    st.markdown("**Martina - Low Risk Profile**")
    st.markdown("Client profile with characteristics indicating low default risk. Should be approved.")
    if st.button("Evaluate Profile: Martina", key="btn_martina", use_container_width=True):
        payload = get_martina_profile()
        result = send_prediction_and_display(payload, "Martina (Low Risk)", return_result=True)
        if result:
            st.session_state.profile_results.append(result)
            st.rerun()

# Mostrar todos los resultados acumulados
if st.session_state.profile_results:
    st.markdown("---")
    st.subheader("Profile Evaluation Results")
    
    for idx, result_data in enumerate(st.session_state.profile_results):
        profile_name = result_data.get("profile_name", f"Profile {idx + 1}")
        result = result_data.get("result")
        payload = result_data.get("payload", {})
        
        with st.expander(f"{profile_name} - {result.get('prediction', 'unknown').upper()}", expanded=True):
            prediction = result.get("prediction", "unknown")
            probability = result.get("probability", 0.0)
            confidence = result.get("confidence", "low")
            
            # Layout con gauge a la izquierda y cajas de información a la derecha
            col_gauge, col_info = st.columns([1, 1])
            
            with col_gauge:
                # Mostrar el gauge
                fig = create_risk_gauge(probability)
                st.plotly_chart(fig, use_container_width=True, key=f"gauge_profile_{idx}_{profile_name}")
                st.markdown('<p style="text-align: center; color: #ffffff; margin-top: -20px;">PERCENTAGE</p>', unsafe_allow_html=True)
            
            with col_info:
                # Caja de aprobación/rechazo
                if prediction.lower() == "approved":
                    st.markdown("""
                    <div style="background-color: #10b981; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                            <span style="font-size: 24px;">✓</span>
                            <h3 style="color: #ffffff; margin: 0;">APPROVED</h3>
                        </div>
                        <p style="color: #ffffff; margin: 0;">Credit application was approved</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif prediction.lower() == "rejected":
                    st.markdown("""
                    <div style="background-color: #ef4444; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                            <span style="font-size: 24px;">✗</span>
                            <h3 style="color: #ffffff; margin: 0;">REJECTED</h3>
                        </div>
                        <p style="color: #ffffff; margin: 0;">Credit application was rejected</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background-color: #f59e0b; border: 2px solid #8B5CF6; border-radius: 10px; padding: 20px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(139, 92, 246, 0.3);">
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                            <span style="font-size: 24px;">?</span>
                            <h3 style="color: #ffffff; margin: 0;">UNKNOWN</h3>
                        </div>
                        <p style="color: #ffffff; margin: 0;">Prediction: {}</p>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)
                
                # Confianza
                confidence_labels = {
                    "high": "High",
                    "medium": "Medium",
                    "low": "Low"
                }
                confidence_display = confidence_labels.get(confidence, confidence.capitalize())
                st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <p style="color: #ffffff; margin-bottom: 5px; font-size: 14px;">Confidence <span style="color: #8B5CF6;">?</span></p>
                    <h2 style="color: #ffffff; margin: 0; font-size: 32px;">{confidence_display}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Caja de interpretación del riesgo (más pequeña y sutil)
                if probability >= 0.7:
                    risk_level = "HIGH RISK"
                    risk_color_bg = "rgba(239, 68, 68, 0.15)"  # Rojo más transparente
                    risk_color_border = "rgba(239, 68, 68, 0.4)"
                    interpretation_text = f"Default probability of {probability:.1%}. It is recommended to reject the application."
                elif probability >= 0.5:
                    risk_level = "MODERATE-HIGH RISK"
                    risk_color_bg = "rgba(249, 115, 22, 0.15)"  # Naranja más transparente
                    risk_color_border = "rgba(249, 115, 22, 0.4)"
                    interpretation_text = f"Default probability of {probability:.1%}. Additional review required."
                elif probability >= 0.3:
                    risk_level = "MODERATE RISK"
                    risk_color_bg = "rgba(245, 158, 11, 0.15)"  # Amarillo más transparente
                    risk_color_border = "rgba(245, 158, 11, 0.4)"
                    interpretation_text = f"Default probability of {probability:.1%}. Careful evaluation recommended."
                else:
                    risk_level = "LOW RISK"
                    risk_color_bg = "rgba(16, 185, 129, 0.15)"  # Verde más transparente
                    risk_color_border = "rgba(16, 185, 129, 0.4)"
                    interpretation_text = f"Default probability of {probability:.1%}. Client with good credit profile."
                
                st.markdown(f"""
                <div style="background-color: {risk_color_bg}; border: 1px solid {risk_color_border}; border-radius: 8px; padding: 12px; margin-top: 10px;">
                    <h4 style="color: #c4b5fd; margin: 0 0 6px 0; font-size: 14px; font-weight: 600;">{risk_level}</h4>
                    <p style="color: #e0e0e0; margin: 0; font-size: 12px;">{interpretation_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Expandable details
            with st.expander("Technical Details of the Query"):
                st.write("**Payload sent to API:**")
                st.json(payload)
                st.write("**Complete API response:**")
                st.json(result)
    
    # Button to clear results
    if st.button("Clear Results", key="btn_clear_results"):
        st.session_state.profile_results = []
        st.rerun()

# Footer
st.markdown("---")
st.caption(f"API URL: {API_URL}")
st.caption("Note: Fields marked with * are required. Complete all available fields for a more accurate evaluation.")
