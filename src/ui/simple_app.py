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

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Analysis")
st.markdown("**Sistema de evaluación de riesgo crediticio usando Machine Learning**")
st.markdown("Complete todos los campos disponibles para una evaluación más precisa. Los campos marcados con * son requeridos.")

st.markdown("---")

# ========== PERFILES PREDEFINIDOS ==========
def get_martin_profile():
    """Perfil de alto riesgo - debería ser rechazado"""
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
        "CITY_OF_BIRTH": "Sao Paulo",
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
        "RESIDENCIAL_CITY": "Sao Paulo",
        "RESIDENCE_TYPE": 2,
        "MONTHS_IN_RESIDENCE": 3.0,
        "PROFESSION_CODE": 6,
        "OCCUPATION_TYPE": 2,
        "MONTHS_IN_THE_JOB": 3.0,
    }

def get_martina_profile():
    """Perfil de bajo riesgo - debería ser aprobado"""
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
        "CITY_OF_BIRTH": "Sao Paulo",
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
        "RESIDENCIAL_CITY": "Sao Paulo",
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
                
                st.markdown("---")
                if profile_name:
                    st.subheader(f"📊 Resultado de la Evaluación - {profile_name}")
                else:
                    st.subheader("📊 Resultado de la Evaluación")
                
                # Mostrar predicción con mejor formato
                col_pred1, col_pred2 = st.columns([2, 1])
                with col_pred1:
                    if prediction.lower() == "approved":
                        st.success(f"✅ **APROBADO** - La solicitud de crédito fue aprobada")
                    elif prediction.lower() == "rejected":
                        st.error(f"❌ **RECHAZADO** - La solicitud de crédito fue rechazada")
                    else:
                        st.warning(f"⚠️ **DESCONOCIDO** - Predicción: {prediction}")
                with col_pred2:
                    # Barra de probabilidad visual
                    st.progress(probability)
                    st.caption(f"{probability:.1%} riesgo")
                
                # Mostrar probabilidad y confianza
                col_result1, col_result2 = st.columns(2)
                with col_result1:
                    st.metric(
                        "Probabilidad de Default",
                        f"{probability:.1%}",
                        help="Probabilidad de que el cliente no pague (0% = muy seguro, 100% = muy riesgoso)"
                    )
                with col_result2:
                    confidence_labels = {
                        "high": "Alta",
                        "medium": "Media",
                        "low": "Baja"
                    }
                    st.metric(
                        "Confianza",
                        confidence_labels.get(confidence, confidence),
                        help="Nivel de confianza de la predicción"
                    )
                
                # Interpretación mejorada
                if probability >= 0.7:
                    interpretation = f"**⚠️ Alto Riesgo:** Probabilidad de default de {probability:.1%}. Se recomienda rechazar la solicitud."
                    st.warning(interpretation)
                elif probability >= 0.5:
                    interpretation = f"**🔶 Riesgo Moderado-Alto:** Probabilidad de default de {probability:.1%}. Se requiere revisión adicional."
                    st.warning(interpretation)
                elif probability >= 0.3:
                    interpretation = f"**🟡 Riesgo Moderado:** Probabilidad de default de {probability:.1%}. Evaluación cuidadosa recomendada."
                    st.info(interpretation)
                else:
                    interpretation = f"**✅ Bajo Riesgo:** Probabilidad de default de {probability:.1%}. Cliente con buen perfil crediticio."
                    st.success(interpretation)
                
                # Detalles expandibles
                with st.expander("📋 Detalles Técnicos de la Consulta"):
                    st.write("**Payload enviado a la API:**")
                    st.json(payload)
                    st.write("**Respuesta completa de la API:**")
                    st.json(result)
            
            else:
                st.error(f"❌ Error de API: Status {response.status_code}")
                st.text(response.text)
        
        except requests.exceptions.ConnectionError:
            st.error(f"❌ **Error de Conexión** - No se pudo conectar a la API en {API_URL}")
            st.info("Asegúrate de que la API esté corriendo. Verifica con: `docker-compose ps`")
        
        except requests.exceptions.Timeout:
            st.error("❌ **Timeout** - La API tardó demasiado en responder")
        
        except Exception as e:
            st.error(f"❌ **Error**: {str(e)}")

# Formulario completo con TODAS las features
with st.form("credit_risk_form"):
    
    # ========== SECCIÓN 1: INFORMACIÓN BÁSICA Y APLICACIÓN ==========
    st.subheader("📋 Información Básica y Aplicación")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        payment_day = st.number_input(
            "Día de pago *",
            min_value=1,
            max_value=31,
            value=15,
            step=1,
            help="Día del mes elegido para el pago (1-31)",
            key="payment_day"
        )
        
        application_type = st.selectbox(
            "Tipo de aplicación *",
            options=["Web", "Carga"],
            help="Cómo se envió la solicitud",
            key="application_type"
        )
        
        product_options = {
            "No especificado": None,
            "1 - Producto A": 1,
            "2 - Producto B": 2,
            "3 - Producto C": 3,
            "4 - Producto D": 4
        }
        product_selection = st.selectbox(
            "Tipo de producto",
            options=list(product_options.keys()),
            help="Tipo de producto (opcional)",
            key="product_select"
        )
        product = product_options[product_selection]
    
    with col2:
        age = st.number_input(
            "Edad *",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
            help="Edad del solicitante",
            key="age"
        )
        
        sex = st.selectbox(
            "Sexo *",
            options=["M", "F"],
            help="Sexo del solicitante",
            key="sex"
        )
        
        quant_dependants = st.number_input(
            "Cantidad de dependientes *",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            help="Número de personas que dependen del solicitante",
            key="dependants"
        )
    
    with col3:
        marital_status = st.selectbox(
            "Estado civil",
            options=["", "1 - Soltero", "2 - Casado", "3 - Divorciado", "4 - Viudo", "5 - Unión libre", "6 - Separado", "7 - Otro"],
            help="Estado civil del solicitante (opcional)",
            key="marital"
        )
        
        # Estado de nacimiento - selectbox con opciones reales
        state_birth_options = UI_OPTIONS.get("STATE_OF_BIRTH", [])
        state_birth_options_with_empty = [""] + state_birth_options
        state_of_birth = st.selectbox(
            "Estado de nacimiento",
            options=state_birth_options_with_empty,
            help="Estado donde nació (opcional). Seleccione de las opciones disponibles en el dataset.",
            key="state_birth"
        )
        state_of_birth = None if state_of_birth == "" else state_of_birth
        
        # Ciudad de nacimiento - text_input (alta cardinalidad: 9,910 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        city_of_birth = st.text_input(
            "Ciudad de nacimiento",
            value="",
            help="Ciudad donde nació (opcional). Deje vacío si desconoce o ingrese el nombre exacto. Si no existe en el dataset, se manejará automáticamente.",
            key="city_birth"
        )
        city_of_birth = None if city_of_birth == "" else city_of_birth
        
        nacionality_options = {
            "No especificado": None,
            "1 - Brasileño": 1,
            "2 - Argentino": 2,
            "3 - Otro": 3
        }
        nacionality_selection = st.selectbox(
            "Nacionalidad",
            options=list(nacionality_options.keys()),
            help="Nacionalidad (opcional)",
            key="nacionality_select"
        )
        nacionality = nacionality_options[nacionality_selection]
    
    st.markdown("---")
    
    # ========== SECCIÓN 2: INFORMACIÓN FINANCIERA ==========
    st.subheader("💰 Información Financiera")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        # Ingreso mensual personal - sin límite superior, con alerta si excede umbral
        income_limits = UI_OPTIONS.get("PERSONAL_MONTHLY_INCOME", {"min": 205.0, "max": 3678.22})
        income_max = float(income_limits.get("max", 3678.22))
        income_threshold = income_max * 1.25  # Alerta después de 25% más que el máximo del dataset
        personal_income = st.number_input(
            "Ingreso mensual personal (R$) *",
            min_value=0.0,
            value=float(income_limits.get("median", 1500)),
            step=100.0,
            help=f"Ingreso mensual regular del solicitante",
            key="personal_income"
        )
        # Mostrar alerta si excede el umbral
        if personal_income > income_threshold:
            st.warning(f"⚠️ El valor ingresado (R$ {personal_income:,.2f}) es significativamente mayor al máximo registrado anteriormente (R$ {income_max:.2f}).")
        
        # Otros ingresos - sin límite superior, con alerta si excede umbral
        other_income_limits = UI_OPTIONS.get("OTHER_INCOMES", {"min": 0.0, "max": 800.0})
        other_income_max = float(other_income_limits.get("max", 800.0))
        other_income_threshold = other_income_max * 1.25  # Alerta después de 25% más que el máximo
        other_incomes_input = st.number_input(
            "Otros ingresos mensuales (R$)",
            min_value=0.0,
            value=0.0,
            step=50.0,
            help=f"Otros ingresos adicionales (opcional)",
            key="other_incomes"
        )
        # Mostrar alerta si excede el umbral
        if other_incomes_input > other_income_threshold:
            st.warning(f"⚠️ El valor ingresado (R$ {other_incomes_input:,.2f}) es significativamente mayor al máximo registrado anteriormente (R$ {other_income_max:.2f}).")
        other_incomes = None if other_incomes_input == 0.0 else other_incomes_input
        
        # Valor de activos personales - sin límite superior, con alerta si excede umbral
        assets_limits = UI_OPTIONS.get("PERSONAL_ASSETS_VALUE", {"min": 0.0, "max": 50000.0})
        assets_max = float(assets_limits.get("max", 50000.0))
        assets_threshold = assets_max * 1.25  # Alerta después de 25% más que el máximo
        assets_input = st.number_input(
            "Valor de activos personales (R$)",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            help=f"Valor total de propiedades, autos, etc. (opcional)",
            key="assets_value"
        )
        # Mostrar alerta si excede el umbral
        if assets_input > assets_threshold:
            st.warning(f"⚠️ El valor ingresado (R$ {assets_input:,.2f}) es significativamente mayor al máximo registrado anteriormente (R$ {assets_max:,.2f}).")
        assets_value = None if assets_input == 0.0 else assets_input
    
    with col5:
        # Cuentas bancarias con opciones
        banking_accounts_options = {
            "No especificado": None,
            "0": 0,
            "1": 1,
            "2": 2,
            "3 o más": 3
        }
        banking_accounts_selection = st.selectbox(
            "Cantidad de cuentas bancarias",
            options=list(banking_accounts_options.keys()),
            help="Número de cuentas bancarias (opcional)",
            key="banking_accounts_select"
        )
        quant_banking_accounts = banking_accounts_options[banking_accounts_selection]
        
        # Cuentas especiales con opciones
        special_accounts_options = {
            "No especificado": None,
            "0": 0,
            "1": 1,
            "2 o más": 2
        }
        special_accounts_selection = st.selectbox(
            "Cuentas bancarias especiales",
            options=list(special_accounts_options.keys()),
            help="Cantidad de cuentas bancarias especiales (opcional)",
            key="special_accounts_select"
        )
        quant_special_banking_accounts = special_accounts_options[special_accounts_selection]
        
        # Autos con opciones
        cars_options = {
            "No especificado": None,
            "0": 0,
            "1": 1,
            "2": 2,
            "3 o más": 3
        }
        cars_selection = st.selectbox(
            "Cantidad de autos",
            options=list(cars_options.keys()),
            help="Número de vehículos (opcional)",
            key="cars_select"
        )
        quant_cars = cars_options[cars_selection]
    
    with col6:
        st.markdown("**Tarjetas de Crédito**")
        flag_visa = st.checkbox("Visa", key="visa")
        flag_mastercard = st.checkbox("Mastercard", key="mastercard")
        flag_diners = st.checkbox("Diners", key="diners")
        flag_amex = st.checkbox("American Express", key="amex")
        flag_other_cards = st.checkbox("Otras tarjetas", key="other_cards")
        flag_email = st.checkbox("Tiene email", key="email")
    
    st.markdown("---")
    
    # ========== SECCIÓN 3: RESIDENCIA ==========
    st.subheader("🏠 Información de Residencia")
    
    col7, col8, col9 = st.columns(3)
    
    with col7:
        # Estado de residencia - selectbox con opciones reales
        res_state_options = UI_OPTIONS.get("RESIDENCIAL_STATE", [])
        res_state_options_with_empty = [""] + res_state_options
        residencial_state = st.selectbox(
            "Estado de residencia",
            options=res_state_options_with_empty,
            help="Estado donde reside (opcional). Seleccione de las opciones disponibles.",
            key="res_state"
        )
        residencial_state = None if residencial_state == "" else residencial_state
        
        # Ciudad de residencia - text_input (alta cardinalidad: 3,529 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        residencial_city = st.text_input(
            "Ciudad de residencia",
            value="",
            help="Ciudad donde reside (opcional). Deje vacío si desconoce o ingrese el nombre exacto. Si no existe en el dataset, se manejará automáticamente.",
            key="res_city"
        )
        residencial_city = None if residencial_city == "" else residencial_city
        
        # Barrio de residencia - text_input (alta cardinalidad: 14,511 categorías)
        # El usuario puede dejar vacío (None) o ingresar un valor
        # Si el valor no existe en el dataset, Frequency Encoding usará frecuencia mínima
        residencial_borough = st.text_input(
            "Barrio de residencia",
            value="",
            help="Barrio donde reside (opcional). Deje vacío si desconoce o ingrese el nombre exacto. Si no existe en el dataset, se manejará automáticamente.",
            key="res_borough"
        )
        residencial_borough = None if residencial_borough == "" else residencial_borough
        
        # Código de área teléfono residencial - selectbox (alta cardinalidad: 102 categorías)
        # Usa Frequency Encoding, valores desconocidos se manejan automáticamente
        res_phone_area_options = UI_OPTIONS.get("RESIDENCIAL_PHONE_AREA_CODE", [])
        res_phone_area_options_with_empty = [""] + sorted(res_phone_area_options)
        residencial_phone_area_code = st.selectbox(
            "Código de área teléfono residencial",
            options=res_phone_area_options_with_empty,
            help="Código de área (opcional). Seleccione de las opciones disponibles o deje vacío si desconoce.",
            key="res_phone_area"
        )
        residencial_phone_area_code = None if residencial_phone_area_code == "" else residencial_phone_area_code
        
        # Código postal - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        res_zip_options = UI_OPTIONS.get("RESIDENCIAL_ZIP_3", [])
        if len(res_zip_options) > 1000:
            # Si hay demasiadas, usar text_input
            residencial_zip_3 = st.text_input(
                "Código postal (primeros 3 dígitos)",
                value="",
                help="Código postal (opcional). Deje vacío si desconoce o ingrese los primeros 3 dígitos. Si no existe en el dataset, se manejará automáticamente.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
        else:
            res_zip_options_with_empty = [""] + sorted(res_zip_options)
            residencial_zip_3 = st.selectbox(
                "Código postal (primeros 3 dígitos)",
                options=res_zip_options_with_empty,
                help="Código postal (opcional). Seleccione de las opciones disponibles o deje vacío si desconoce.",
                key="res_zip"
            )
            residencial_zip_3 = None if residencial_zip_3 == "" else residencial_zip_3
    
    with col8:
        flag_residential_phone = st.selectbox(
            "Teléfono residencial *",
            options=["Y", "N"],
            help="¿Tiene teléfono residencial?",
            key="residential_phone"
        )
        
        residence_type = st.selectbox(
            "Tipo de residencia",
            options=["", "1 - Propia", "2 - Alquilada", "3 - Cedida", "4 - Con familiares", "5 - Otro"],
            help="Tipo de residencia actual (opcional)",
            key="residence_type"
        )
        
        # Meses en residencia con opciones
        months_residence_options = {
            "No especificado": None,
            "Menos de 6 meses": 3,
            "6 meses - 1 año": 9,
            "1 - 2 años": 18,
            "2 - 3 años": 30,
            "Más de 3 años": 45
        }
        months_residence_selection = st.selectbox(
            "Meses en residencia actual",
            options=list(months_residence_options.keys()),
            help="Tiempo viviendo en la residencia actual (opcional)",
            key="months_residence_select"
        )
        months_in_residence = months_residence_options[months_residence_selection]
        
        postal_address_options = {
            "No especificado": None,
            "1 - Residencial": 1,
            "2 - Comercial": 2,
            "3 - Otro": 3
        }
        postal_address_selection = st.selectbox(
            "Tipo de dirección postal",
            options=list(postal_address_options.keys()),
            help="Tipo de dirección postal (opcional)",
            key="postal_address_select"
        )
        postal_address_type = postal_address_options[postal_address_selection]
    
    st.markdown("---")
    
    # ========== SECCIÓN 4: EMPLEO ==========
    st.subheader("💼 Información de Empleo")
    
    col10, col11, col12 = st.columns(3)
    
    with col10:
        company = st.selectbox(
            "¿Tiene compañía/empleo formal? *",
            options=["Y", "N"],
            help="¿Proporcionó nombre de compañía donde trabaja?",
            key="company"
        )
        
        # Estado profesional - selectbox con opciones reales
        prof_state_options = UI_OPTIONS.get("PROFESSIONAL_STATE", [])
        prof_state_options_with_empty = [""] + prof_state_options
        professional_state = st.selectbox(
            "Estado profesional",
            options=prof_state_options_with_empty,
            help="Estado donde trabaja (opcional). Seleccione de las opciones disponibles.",
            key="prof_state"
        )
        professional_state = None if professional_state == "" else professional_state
        
        # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing
        # porque tienen alta cardinalidad y muchos missing values
    
    with col11:
        flag_professional_phone = st.selectbox(
            "Teléfono profesional *",
            options=["Y", "N"],
            help="¿Tiene teléfono profesional?",
            key="professional_phone"
        )
        
        # Código de área del teléfono profesional - selectbox (alta cardinalidad: 87 categorías)
        # Usa agrupación + OneHot, valores desconocidos se manejan automáticamente
        prof_phone_area_options = UI_OPTIONS.get("PROFESSIONAL_PHONE_AREA_CODE", [])
        prof_phone_area_options_with_empty = [""] + sorted(prof_phone_area_options)
        professional_phone_area_code = st.selectbox(
            "Código de área teléfono profesional",
            options=prof_phone_area_options_with_empty,
            help="Código de área (opcional). Seleccione de las opciones disponibles o deje vacío si desconoce.",
            key="prof_phone_area"
        )
        professional_phone_area_code = None if professional_phone_area_code == "" else professional_phone_area_code
        
        # Código postal profesional - selectbox o text_input según cantidad (alta cardinalidad: 794 categorías)
        prof_zip_options = UI_OPTIONS.get("PROFESSIONAL_ZIP_3", [])
        if len(prof_zip_options) > 1000:
            professional_zip_3 = st.text_input(
                "Código postal profesional (primeros 3 dígitos)",
                value="",
                help="Código postal (opcional). Deje vacío si desconoce o ingrese los primeros 3 dígitos. Si no existe en el dataset, se manejará automáticamente.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        else:
            prof_zip_options_with_empty = [""] + sorted(prof_zip_options)
            professional_zip_3 = st.selectbox(
                "Código postal profesional (primeros 3 dígitos)",
                options=prof_zip_options_with_empty,
                help="Código postal (opcional). Seleccione de las opciones disponibles o deje vacío si desconoce.",
                key="prof_zip"
            )
            professional_zip_3 = None if professional_zip_3 == "" else professional_zip_3
        
        # Meses en trabajo con opciones
        months_job_options = {
            "No especificado": None,
            "Menos de 6 meses": 3,
            "6 meses - 1 año": 9,
            "1 - 2 años": 18,
            "2 - 3 años": 30,
            "Más de 3 años": 45
        }
        months_job_selection = st.selectbox(
            "Meses en trabajo actual",
            options=list(months_job_options.keys()),
            help="Tiempo trabajando en el empleo actual (opcional)",
            key="months_job_select"
        )
        months_in_job = months_job_options[months_job_selection]
    
    with col12:
        # Código de profesión con opciones comunes
        profession_code_options = {
            "No especificado": None,
            "1 - Profesional": 1,
            "2 - Técnico": 2,
            "3 - Administrativo": 3,
            "4 - Comercial": 4,
            "5 - Servicios": 5,
            "6 - Operario": 6,
            "7 - Otro": 7
        }
        profession_code_selection = st.selectbox(
            "Código de profesión",
            options=list(profession_code_options.keys()),
            help="Código de profesión (opcional)",
            key="profession_select"
        )
        profession_code = profession_code_options[profession_code_selection]
        
        occupation_type = st.selectbox(
            "Tipo de ocupación",
            options=["", "1 - Empleado", "2 - Autónomo", "3 - Empresario", "4 - Desempleado", "5 - Otro"],
            help="Tipo de ocupación laboral (opcional)",
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
            "No especificado": None,
            "1 - Primario": 1,
            "2 - Secundario": 2,
            "3 - Terciario": 3,
            "4 - Universitario": 4
        }
        education_level_selection = st.selectbox(
            "Nivel educativo del cónyuge",
            options=list(education_level_options.keys()),
            help="Nivel educativo del cónyuge (opcional)",
            key="education_level_1_select"
        )
        education_level_1 = education_level_options[education_level_selection]
    
    st.markdown("---")
    
    # Submit buttons
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    with col_btn1:
        submitted = st.form_submit_button("🔍 Evaluar Riesgo Crediticio", type="primary", use_container_width=True)
    with col_btn2:
        martin_clicked = st.form_submit_button("👤 Martin (Alto Riesgo)", use_container_width=True)
    with col_btn3:
        martina_clicked = st.form_submit_button("👤 Martina (Bajo Riesgo)", use_container_width=True)
    
    # Verificar qué botón fue presionado
    use_martin = martin_clicked
    use_martina = martina_clicked
    
    if submitted or use_martin or use_martina:
        # Si se usó un perfil predefinido, usar ese payload directamente
        if use_martin:
            payload = get_martin_profile()
        elif use_martina:
            payload = get_martina_profile()
        else:
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
        profile_name = None
        if use_martin:
            profile_name = "Martin (Alto Riesgo)"
        elif use_martina:
            profile_name = "Martina (Bajo Riesgo)"
        
        send_prediction_and_display(payload, profile_name)

# Footer
st.markdown("---")
st.caption(f"🌐 API URL: {API_URL}")
st.caption("💡 Nota: Los campos marcados con * son requeridos. Complete todos los campos disponibles para una evaluación más precisa.")
