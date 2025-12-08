"""
Credit Risk Analysis - Simple Streamlit UI
Basic interface to test API connection and prediction
"""

import streamlit as st
import requests
import os
from datetime import datetime

# API URL - use environment variable or default
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Credit Risk Analysis",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Analysis - Simple Demo")
st.markdown("Fill the form below to evaluate credit risk")

st.markdown("---")

# Form fields matching API requirements
with st.form("credit_risk_form"):
    st.subheader("📋 Client Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Name", value="John Doe", help="Client's full name")
        birth_date = st.date_input(
            "Birth Date",
            value=datetime(1990, 1, 1),
            min_value=datetime(1920, 1, 1),
            max_value=datetime.now(),
            help="Client's date of birth"
        )
        gender = st.selectbox(
            "Gender",
            options=["M", "F"],
            help="Client's gender"
        )
        education = st.selectbox(
            "Education",
            options=["Graduate", "Not Graduate"],
            help="Education level"
        )
    
    with col2:
        employeed = st.checkbox("Employed", value=True, help="Is the client employed?")
        marital_status = st.selectbox(
            "Marital Status",
            options=["Yes", "No"],
            help="Marital status (Yes = Married, No = Single)"
        )
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of dependents"
        )
        property_area = st.selectbox(
            "Property Area",
            options=["Urban", "Semiurban", "Rural"],
            help="Property area type"
        )
    
    st.markdown("---")
    st.subheader("💰 Financial Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        income = st.number_input(
            "Applicant Income",
            min_value=0,
            value=5000,
            step=100,
            help="Monthly applicant income"
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income",
            min_value=0,
            value=0,
            step=100,
            help="Monthly co-applicant income"
        )
    
    with col4:
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=100000,
            step=1000,
            help="Requested loan amount"
        )
        loan_term = st.number_input(
            "Loan Term (days)",
            min_value=1,
            value=360,
            step=30,
            help="Loan term in days"
        )
    
    credit_history = st.selectbox(
        "Credit History",
        options=["Yes", "No"],
        help="Has credit history? (Yes = Good, No = Bad)"
    )
    
    st.markdown("---")
    
    # Submit button
    submitted = st.form_submit_button("🔍 Evaluate Risk", type="primary", use_container_width=True)
    
    if submitted:
        # Prepare payload
        payload = {
            "name": name,
            "birth_date": birth_date.strftime("%Y-%m-%d"),
            "gender": gender,
            "education": education,
            "employeed": employeed,
            "marital_status": marital_status,
            "dependents": dependents,
            "property_area": property_area,
            "income": income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "credit_history": credit_history
        }
        
        # Show loading
        with st.spinner("Sending request to API..."):
            try:
                # Send request to API
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result.get("prediction", "unknown")
                    
                    st.markdown("---")
                    st.subheader("📊 Prediction Result")
                    
                    if prediction.lower() == "approved":
                        st.success(f"✅ **APPROVED** - Credit application approved!")
                    elif prediction.lower() == "rejected":
                        st.error(f"❌ **REJECTED** - Credit application rejected")
                    else:
                        st.warning(f"⚠️ **UNKNOWN** - Prediction: {prediction}")
                    
                    # Show details
                    with st.expander("📋 Request Details"):
                        st.json(payload)
                        st.json(result)
                
                else:
                    st.error(f"❌ API Error: Status {response.status_code}")
                    st.text(response.text)
            
            except requests.exceptions.ConnectionError:
                st.error(f"❌ **Connection Error** - Could not connect to API at {API_URL}")
                st.info("Make sure the API is running. Check Docker containers with: `docker-compose ps`")
            
            except requests.exceptions.Timeout:
                st.error("❌ **Timeout** - API took too long to respond")
            
            except Exception as e:
                st.error(f"❌ **Error**: {str(e)}")

# Footer
st.markdown("---")
st.caption(f"API URL: {API_URL}")

