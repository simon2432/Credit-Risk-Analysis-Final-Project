"""
Credit Risk Analysis - Streamlit UI
Formulario para evaluación de riesgo crediticio de clientes
"""

import streamlit as st
import json


def create_form_section_personal():
    """Sección 1: Datos personales"""
    st.subheader("📋 Personal Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=90,
            value=30,
            step=1,
            help="Client's age"
        )
    
    with col2:
        sex = st.selectbox(
            "Sex",
            options=["F", "M"],
            help="Client's gender"
        )
    
    marital_status = st.selectbox(
        "Marital Status",
        options=["single", "married", "divorced", "widowed"],
        help="Client's marital status"
    )
    
    education_level = st.selectbox(
        "Education Level",
        options=["primary", "secondary", "university", "postgraduate"],
        help="Client's education level"
    )
    
    quant_dependants = st.number_input(
        "Number of Dependants",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
        help="Number of people who depend on the client"
    )
    
    return {
        "age": int(age),
        "sex": sex,
        "marital_status": marital_status,
        "education_level": education_level,
        "quant_dependants": int(quant_dependants)
    }


def create_form_section_residence():
    """Sección 2: Residencia"""
    st.subheader("🏠 Residence")
    
    residence_type = st.selectbox(
        "Residence Type",
        options=["owner", "renter", "family", "other"],
        help="Type of residence"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        months_in_residence = st.number_input(
            "Months in Current Residence",
            min_value=0,
            max_value=600,
            value=24,
            step=1,
            help="Number of months living in current residence"
        )
    
    with col2:
        residencial_state = st.text_input(
            "Residential State",
            value="RN",
            placeholder="RN",
            help="State where the client resides"
        )
    
    return {
        "residence_type": residence_type,
        "months_in_residence": int(months_in_residence),
        "residencial_state": residencial_state
    }


def create_form_section_employment():
    """Sección 3: Empleo"""
    st.subheader("💼 Employment")
    
    occupation_type = st.selectbox(
        "Occupation Type",
        options=["employee", "self_employed", "unemployed", "student", "retired"],
        help="Client's occupation type"
    )
    
    months_in_the_job = st.number_input(
        "Months in Current Job",
        min_value=0,
        max_value=600,
        value=12,
        step=1,
        help="Number of months in current job"
    )
    
    return {
        "occupation_type": occupation_type,
        "months_in_the_job": int(months_in_the_job)
    }


def create_form_section_income():
    """Sección 4: Ingresos"""
    st.subheader("💰 Income")
    
    col1, col2 = st.columns(2)
    
    with col1:
        personal_monthly_income = st.number_input(
            "Personal Monthly Income",
            min_value=0.0,
            value=3000.0,
            step=100.0,
            format="%.2f",
            help="Client's personal monthly income"
        )
    
    with col2:
        other_incomes = st.number_input(
            "Other Monthly Incomes",
            min_value=0.0,
            value=0.0,
            step=100.0,
            format="%.2f",
            help="Other sources of monthly income"
        )
    
    return {
        "personal_monthly_income": float(personal_monthly_income),
        "other_incomes": float(other_incomes)
    }


def create_form_section_banking():
    """Sección 5: Bancos y contacto"""
    st.subheader("🏦 Banking & Contact")
    
    quant_banking_accounts = st.number_input(
        "Number of Banking Accounts",
        min_value=0,
        max_value=20,
        value=1,
        step=1,
        help="Number of banking accounts the client has"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        flag_mobile_phone = st.checkbox(
            "Has Mobile Phone?",
            value=True,
            help="Whether the client has a mobile phone"
        )
    
    with col2:
        flag_email = st.checkbox(
            "Has Email?",
            value=True,
            help="Whether the client has an email address"
        )
    
    return {
        "quant_banking_accounts": int(quant_banking_accounts),
        "flag_mobile_phone": flag_mobile_phone,
        "flag_email": flag_email
    }


def build_payload(form_data):
    """Construye el payload JSON con todos los datos del formulario"""
    payload = {}
    
    # Combinar todos los datos de las secciones
    for section_data in form_data:
        payload.update(section_data)
    
    return payload


def main():
    """Función principal de la aplicación"""
    # Configuración de la página
    st.set_page_config(
        page_title="Credit Risk Analysis",
        page_icon="💳",
        layout="wide"
    )
    
    # Título principal
    st.title("💳 Credit Risk – Client Evaluation")
    st.markdown("Fill the form below to evaluate the credit risk of a client.")
    
    st.markdown("---")
    
    # Inicializar lista para almacenar datos de las secciones
    form_sections_data = []
    
    # Sección 1: Datos personales
    personal_data = create_form_section_personal()
    form_sections_data.append(personal_data)
    
    st.markdown("---")
    
    # Sección 2: Residencia
    residence_data = create_form_section_residence()
    form_sections_data.append(residence_data)
    
    st.markdown("---")
    
    # Sección 3: Empleo
    employment_data = create_form_section_employment()
    form_sections_data.append(employment_data)
    
    st.markdown("---")
    
    # Sección 4: Ingresos
    income_data = create_form_section_income()
    form_sections_data.append(income_data)
    
    st.markdown("---")
    
    # Sección 5: Bancos y contacto
    banking_data = create_form_section_banking()
    form_sections_data.append(banking_data)
    
    st.markdown("---")
    
    # Botón para evaluar
    if st.button("🔍 Evaluate Risk", type="primary", use_container_width=True):
        # Construir el payload
        payload = build_payload(form_sections_data)
        
        # Mostrar mensaje informativo
        st.info("This is the payload that will be sent to the `/predict` API endpoint.")
        
        # Mostrar el JSON
        st.json(payload)
        
        # Opcional: mostrar también un resumen
        with st.expander("📊 Payload Summary"):
            st.write(f"**Total fields:** {len(payload)}")
            st.write("**Fields included:**")
            for key, value in payload.items():
                st.write(f"- `{key}`: {value}")


if __name__ == "__main__":
    main()
