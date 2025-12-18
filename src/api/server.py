"""
FastAPI server para Credit Risk Analysis.
Carga el modelo entrenado y el pipeline de preprocessing para hacer predicciones.
"""

import uvicorn
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os

# Cargar modelo y pipeline al iniciar
from src.config import MODEL_FILE, PREPROCESSOR_FILE, MODEL_DIR
from src.preprocessing import PreprocessingPipeline
from src.api.feature_mapper import create_full_feature_dict

app = FastAPI(title="Credit Risk Analysis API", version="1.0.0")

# Cargar modelo y pipeline
model = None
preprocessor = None
optimal_threshold = 0.5  # Default threshold

def load_model_and_preprocessor():
    """Carga el modelo, pipeline y threshold óptimo."""
    global model, preprocessor, optimal_threshold
    try:
        model = joblib.load(MODEL_FILE)
        preprocessor = PreprocessingPipeline.load()
        
        # Intentar cargar threshold óptimo
        threshold_path = MODEL_DIR / "optimal_threshold.txt"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                optimal_threshold = float(f.read().strip())
            print(f"[OK] Optimal threshold loaded: {optimal_threshold:.4f}")
        else:
            print(f"[WARNING] Optimal threshold file not found, using default: {optimal_threshold}")
        
        print(f"[OK] Model loaded from: {MODEL_FILE}")
        print(f"[OK] Preprocessor loaded from: {PREPROCESSOR_FILE}")
        print(f"[DEBUG] PREPROCESSOR_FILE path: {PREPROCESSOR_FILE}")
        print(f"[DEBUG] File exists: {os.path.exists(PREPROCESSOR_FILE)}")
        
        # Verificar que el preprocessor tenga los métodos de protección
        if hasattr(preprocessor, '__setstate__'):
            print("[OK] Preprocessor has __setstate__ method")
        if hasattr(preprocessor, '__getattribute__'):
            print("[OK] Preprocessor has __getattribute__ method")
        
        # Verificar atributos obsoletos
        if hasattr(preprocessor, 'outlier_limits'):
            print(f"[INFO] outlier_limits exists, type: {type(getattr(preprocessor, 'outlier_limits', None))}")
        else:
            print("[WARNING] outlier_limits does not exist (may cause issues)")
        return True
    except Exception as e:
        print(f"[ERROR] Error loading model/preprocessor: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"   MODEL_FILE: {MODEL_FILE}")
        print(f"   PREPROCESSOR_FILE: {PREPROCESSOR_FILE}")
        return False

# Intentar cargar al iniciar
@app.on_event("startup")
async def startup_event():
    """Cargar modelo y pipeline al iniciar la API."""
    success = load_model_and_preprocessor()
    if not success:
        print("⚠️  API started but model/preprocessor could not be loaded!")
        print("   Make sure you have trained the model first by running: python -m src.train_model")

# Esquema completo - Acepta TODAS las 44 features (sin las 9 constantes)
# Las 9 columnas constantes NO se piden (se rellenan automáticamente)
# Todas las demás features son opcionales pero se pueden proporcionar para mejor predicción
class PredictRequest(BaseModel):
    """
    Request schema completo con TODAS las features necesarias.
    
    Todas las features son opcionales (excepto las más críticas marcadas como requeridas).
    Si no se proporcionan, se usan valores por defecto razonables.
    
    NOTA: Las 9 columnas constantes (CLERK_TYPE, FLAG_ACSP_RECORD, etc.)
    NO necesitan ser proporcionadas porque se eliminan en el preprocessing.
    """
    
    # Features esenciales (requeridas) - Las más importantes para credit risk
    PAYMENT_DAY: int = Field(..., ge=1, le=31, description="Día del mes para pago (1-31)")
    APPLICATION_SUBMISSION_TYPE: str = Field(..., description="Tipo de aplicación: 'Web' o 'Carga'")
    SEX: str = Field(..., description="Sexo: 'M' o 'F'")
    AGE: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    QUANT_DEPENDANTS: int = Field(..., ge=0, description="Cantidad de dependientes")
    PERSONAL_MONTHLY_INCOME: float = Field(..., ge=0, description="Ingreso mensual personal")
    FLAG_RESIDENCIAL_PHONE: str = Field(..., description="Teléfono residencial: 'Y' o 'N'")
    COMPANY: str = Field(..., description="Tiene compañía: 'Y' o 'N'")
    FLAG_PROFESSIONAL_PHONE: str = Field(..., description="Teléfono profesional: 'Y' o 'N'")
    
    # Features opcionales - Información personal y demográfica
    MARITAL_STATUS: Optional[int] = Field(None, ge=1, le=7, description="Estado civil: 1-7")
    STATE_OF_BIRTH: Optional[str] = Field(None, description="Estado de nacimiento")
    CITY_OF_BIRTH: Optional[str] = Field(None, description="Ciudad de nacimiento")
    NACIONALITY: Optional[int] = Field(None, description="Nacionalidad")
    
    # Features opcionales - Residencia
    RESIDENCIAL_STATE: Optional[str] = Field(None, description="Estado de residencia")
    RESIDENCIAL_CITY: Optional[str] = Field(None, description="Ciudad de residencia")
    RESIDENCIAL_BOROUGH: Optional[str] = Field(None, description="Barrio de residencia")
    RESIDENCIAL_PHONE_AREA_CODE: Optional[str] = Field(None, description="Código de área teléfono residencial")
    RESIDENCE_TYPE: Optional[int] = Field(None, ge=1, le=5, description="Tipo de residencia: 1-5")
    MONTHS_IN_RESIDENCE: Optional[float] = Field(None, ge=0, description="Meses en residencia actual")
    RESIDENCIAL_ZIP_3: Optional[str] = Field(None, description="Código postal (primeros 3 dígitos)")
    POSTAL_ADDRESS_TYPE: Optional[int] = Field(None, description="Tipo de dirección postal")
    
    # Features opcionales - Financieras
    OTHER_INCOMES: Optional[float] = Field(None, ge=0, description="Otros ingresos mensuales")
    PERSONAL_ASSETS_VALUE: Optional[float] = Field(None, ge=0, description="Valor de activos personales")
    QUANT_BANKING_ACCOUNTS: Optional[int] = Field(None, ge=0, description="Cantidad de cuentas bancarias")
    QUANT_SPECIAL_BANKING_ACCOUNTS: Optional[int] = Field(None, ge=0, description="Cantidad de cuentas bancarias especiales")
    QUANT_CARS: Optional[int] = Field(None, ge=0, description="Cantidad de autos")
    
    # Features opcionales - Tarjetas
    FLAG_VISA: Optional[int] = Field(None, ge=0, le=1, description="Tiene tarjeta Visa: 0 o 1")
    FLAG_MASTERCARD: Optional[int] = Field(None, ge=0, le=1, description="Tiene tarjeta Mastercard: 0 o 1")
    FLAG_DINERS: Optional[int] = Field(None, ge=0, le=1, description="Tiene tarjeta Diners: 0 o 1")
    FLAG_AMERICAN_EXPRESS: Optional[int] = Field(None, ge=0, le=1, description="Tiene tarjeta American Express: 0 o 1")
    FLAG_OTHER_CARDS: Optional[int] = Field(None, ge=0, le=1, description="Tiene otras tarjetas: 0 o 1")
    FLAG_EMAIL: Optional[int] = Field(None, ge=0, le=1, description="Tiene email: 0 o 1")
    
    # Features opcionales - Empleo
    PROFESSIONAL_STATE: Optional[str] = Field(None, description="Estado profesional")
    # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing (alta cardinalidad + muchos missing)
    PROFESSIONAL_PHONE_AREA_CODE: Optional[str] = Field(None, description="Código de área teléfono profesional")
    PROFESSIONAL_ZIP_3: Optional[str] = Field(None, description="Código postal profesional (primeros 3 dígitos)")
    MONTHS_IN_THE_JOB: Optional[float] = Field(None, ge=0, description="Meses en trabajo actual")
    PROFESSION_CODE: Optional[int] = Field(None, description="Código de profesión")
    OCCUPATION_TYPE: Optional[int] = Field(None, ge=1, le=5, description="Tipo de ocupación: 1-5")
    MATE_PROFESSION_CODE: Optional[int] = Field(None, description="Código de profesión del cónyuge")
    MATE_EDUCATION_LEVEL: Optional[int] = Field(None, description="Nivel educativo del cónyuge")
    
    # Features opcionales - Otros
    PRODUCT: Optional[int] = Field(None, description="Tipo de producto")


class PredictResponse(BaseModel):
    """Response schema con la predicción."""
    prediction: str = Field(..., description="Predicción: 'approved' o 'rejected'")
    probability: float = Field(..., ge=0, le=1, description="Probabilidad de default (BAD=1)")
    confidence: str = Field(..., description="Nivel de confianza: 'high', 'medium', 'low'")


@app.get("/")
def read_root():
    """Endpoint raíz."""
    return {
        "message": "Credit Risk Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Make prediction (POST)",
            "/model_info": "Model information"
        },
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "message": "API is running",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.get("/model_info")
def model_info():
    """Información sobre el modelo cargado."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")
    
    return {
        "model_type": type(model).__name__,
        "preprocessor_type": type(preprocessor).__name__,
        "model_file": MODEL_FILE,
        "preprocessor_file": PREPROCESSOR_FILE,
        "status": "ready"
    }


@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    """
    Endpoint para hacer predicciones de riesgo crediticio.
    
    Recibe features esenciales y automáticamente rellena las constantes y opcionales
    para crear el dataset completo necesario para el preprocessing.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Please train the model first."
        )
    
    try:
        # Convertir request simplificado a diccionario completo
        simplified_input = data.dict(exclude_unset=True, exclude_none=False)
        full_input_dict = create_full_feature_dict(simplified_input)
        
        # Crear DataFrame con una sola fila (todas las features del dataset)
        input_df = pd.DataFrame([full_input_dict])
        
        # Aplicar preprocessing usando el pipeline guardado
        processed_data = preprocessor.transform(input_df)
        
        # Verificar que los datos procesados no tengan NaN o Inf
        if np.isnan(processed_data).any() or np.isinf(processed_data).any():
            print(f"⚠️  WARNING: NaN or Inf values in processed data!")
            print(f"   NaN count: {np.isnan(processed_data).sum()}")
            print(f"   Inf count: {np.isinf(processed_data).sum()}")
        
        # Hacer predicción
        probability = model.predict_proba(processed_data)[0, 1]  # Probabilidad de default (BAD=1)
        
        # DEBUG: Log información importante
        print(f"📊 Prediction Debug:")
        print(f"   Input features provided: {len(simplified_input)}")
        print(f"   Full features after mapping: {len(full_input_dict)}")
        print(f"   Processed features shape: {processed_data.shape}")
        print(f"   Probability (default): {probability:.4f}")
        print(f"   Key inputs: INCOME={simplified_input.get('PERSONAL_MONTHLY_INCOME', 'N/A')}, "
              f"ASSETS={simplified_input.get('PERSONAL_ASSETS_VALUE', 'N/A')}, "
              f"AGE={simplified_input.get('AGE', 'N/A')}")
        
        # Usar threshold óptimo calculado durante el entrenamiento
        # Este threshold fue calculado usando Youden's J statistic en la curva ROC
        # para maximizar TPR - FPR (mejor balance entre recall y precision)
        THRESHOLD = optimal_threshold
        
        # Decisión basada en probabilidad y threshold óptimo
        if probability >= THRESHOLD:
            prediction_str = "rejected"
        else:
            prediction_str = "approved"
        
        # Determinar nivel de confianza
        if probability >= 0.7 or probability <= 0.3:
            confidence = "high"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictResponse(
            prediction=prediction_str,
            probability=float(probability),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )


# uvicorn.run is handled by Docker CMD
# uvicorn.run(app, host="0.0.0.0", port=8000)
