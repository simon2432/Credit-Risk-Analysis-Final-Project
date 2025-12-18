"""
Módulo para mapear features simplificadas a las completas del dataset.
Solo incluye las features que realmente se usan (excluye constantes).
"""

from typing import Dict, Any, Optional, List

# Cache para el orden de columnas (se calcula una vez)
_COLUMN_ORDER_CACHE: Optional[List[str]] = None

# Columnas CONSTANTES que se eliminan automáticamente en el preprocessing
# Estas NO necesitan ser proporcionadas - se rellenan automáticamente
CONSTANT_COLUMNS_REMOVED = {
    "CLERK_TYPE": "C",
    "QUANT_ADDITIONAL_CARDS": 0,
    "EDUCATION_LEVEL": 0,
    "FLAG_MOBILE_PHONE": "N",
    "FLAG_HOME_ADDRESS_DOCUMENT": 0,
    "FLAG_RG": 0,
    "FLAG_CPF": 0,
    "FLAG_INCOME_PROOF": 0,
    "FLAG_ACSP_RECORD": "N",
}

# Valores por defecto razonables para campos opcionales/missing
# Estos se usan si el usuario no los proporciona
DEFAULT_OPTIONAL_VALUES = {
    # Variables de Aplicación
    "POSTAL_ADDRESS_TYPE": 1,
    
    # Variables Demográficas
    "MARITAL_STATUS": 1,  # Valor por defecto más común
    "STATE_OF_BIRTH": None,  # Se manejará como missing
    "CITY_OF_BIRTH": None,  # Se manejará como missing
    "NACIONALITY": 1,  # 1 = Brasil (probablemente)
    
    # Variables de Residencia
    "RESIDENCIAL_STATE": None,  # Se manejará como missing
    "RESIDENCIAL_CITY": None,
    "RESIDENCIAL_BOROUGH": None,
    "RESIDENCIAL_PHONE_AREA_CODE": None,
    "RESIDENCE_TYPE": 1,  # 1 = propia (asumido)
    "MONTHS_IN_RESIDENCE": None,  # Se manejará como missing
    "RESIDENCIAL_ZIP_3": None,
    
    # Variables Financieras
    "OTHER_INCOMES": None,  # Se manejará como missing (más realista que 0)
    "PERSONAL_ASSETS_VALUE": None,  # Se manejará como missing (más realista que 0)
    "QUANT_BANKING_ACCOUNTS": None,  # Se manejará como missing
    "QUANT_SPECIAL_BANKING_ACCOUNTS": None,  # Se manejará como missing
    "QUANT_CARS": None,  # Se manejará como missing
    
    # Variables de Tarjetas (0 = no tiene, es un valor válido, no missing)
    "FLAG_VISA": 0,  # 0 es un valor válido (no tiene tarjeta)
    "FLAG_MASTERCARD": 0,
    "FLAG_DINERS": 0,
    "FLAG_AMERICAN_EXPRESS": 0,
    "FLAG_OTHER_CARDS": 0,
    "FLAG_EMAIL": 0,
    
    # Variables de Empleo
    "PROFESSIONAL_STATE": None,
    # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH fueron removidas del preprocessing (alta cardinalidad + muchos missing)
    "PROFESSIONAL_PHONE_AREA_CODE": None,
    "MONTHS_IN_THE_JOB": None,
    "PROFESSION_CODE": None,
    "OCCUPATION_TYPE": None,
    "MATE_PROFESSION_CODE": None,
    "PROFESSIONAL_ZIP_3": None,
    
    # Otras
    "MATE_EDUCATION_LEVEL": None,  # Nivel educativo del cónyuge
    "PRODUCT": 1,  # Tipo de producto más común
}
    
# Features que DEBEMOS pedir al usuario (importantes para el modelo)
REQUIRED_ESSENTIAL_FEATURES = [
    "PAYMENT_DAY",
    "APPLICATION_SUBMISSION_TYPE",
    "SEX",
    "AGE",
    "QUANT_DEPENDANTS",
    "PERSONAL_MONTHLY_INCOME",
    "FLAG_RESIDENCIAL_PHONE",
    "COMPANY",
    "FLAG_PROFESSIONAL_PHONE",
]

# Features opcionales pero útiles (se pueden agregar a la UI si se quiere)
OPTIONAL_USEFUL_FEATURES = [
    "OTHER_INCOMES",
    "PERSONAL_ASSETS_VALUE",
    "MONTHS_IN_RESIDENCE",
    "MONTHS_IN_THE_JOB",
    "PROFESSION_CODE",
    "MARITAL_STATUS",
    "RESIDENCE_TYPE",
    "OCCUPATION_TYPE",
]


def _get_column_order() -> List[str]:
    """
    Obtiene el orden de columnas del dataset original.
    Cachea el resultado para evitar leer el dataset múltiples veces.
    """
    global _COLUMN_ORDER_CACHE
    
    if _COLUMN_ORDER_CACHE is None:
        from src.data_utils import get_datasets
        train, _, _ = get_datasets()
        _COLUMN_ORDER_CACHE = [c for c in train.columns if c not in ['ID_CLIENT', 'TARGET_LABEL_BAD=1']]
    
    return _COLUMN_ORDER_CACHE


def create_full_feature_dict(simplified_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convierte un diccionario con features simplificadas a uno completo
    con TODAS las features originales del dataset en el orden correcto.
    
    IMPORTANTE: 
    - Las 9 columnas constantes se rellenan automáticamente porque se eliminan en el preprocessing.
    - PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH se incluyen con None y se remueven en preprocessing paso 1.
    
    Args:
        simplified_input: Diccionario con solo las features que el usuario proporcionó
        
    Returns:
        Diccionario completo con todas las features del dataset original en el orden correcto
    """
    # Obtener el orden correcto de columnas (cached)
    required_cols = _get_column_order()
    
    # Crear diccionario base con valores por defecto en el orden correcto
    full_dict = {}
    
    # 1. Primero, agregar valores por defecto para constantes
    full_dict.update(CONSTANT_COLUMNS_REMOVED)
    
    # 2. Agregar valores por defecto para campos opcionales
    full_dict.update(DEFAULT_OPTIONAL_VALUES)
    
    # 3. Sobrescribir con valores proporcionados por el usuario
    full_dict.update(simplified_input)
    
    # 4. Crear diccionario en el orden correcto
    # NOTA: PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH se remueven en preprocessing paso 1
    # pero están en el dataset original, así que las incluimos con None para que el preprocessing las remueva
    ordered_dict = {}
    for col in required_cols:
        if col in full_dict:
            ordered_dict[col] = full_dict[col]
        else:
            # Si falta alguna columna, usar None (se manejará como missing)
            # Para PROFESSIONAL_CITY y PROFESSIONAL_BOROUGH, se usarán None y luego se removerán en preprocessing
            ordered_dict[col] = None
    
    return ordered_dict


def get_required_features_list() -> list:
    """
    Retorna la lista de todas las features requeridas (44 features).
    Excluye las 9 constantes que se eliminan automáticamente.
    """
    # Lista completa de 44 features requeridas (sin constantes)
    # Basado en el output de test_required_features.py
    return [
        "AGE",
        "APPLICATION_SUBMISSION_TYPE",
        "CITY_OF_BIRTH",
        "COMPANY",
        "MATE_EDUCATION_LEVEL",
        "FLAG_AMERICAN_EXPRESS",
        "FLAG_DINERS",
        "FLAG_EMAIL",
        "FLAG_MASTERCARD",
        "FLAG_OTHER_CARDS",
        "FLAG_PROFESSIONAL_PHONE",
        "FLAG_RESIDENCIAL_PHONE",
        "FLAG_VISA",
        "MARITAL_STATUS",
        "MATE_PROFESSION_CODE",
        "MONTHS_IN_RESIDENCE",
        "MONTHS_IN_THE_JOB",
        "NACIONALITY",
        "OCCUPATION_TYPE",
        "OTHER_INCOMES",
        "PAYMENT_DAY",
        "PERSONAL_ASSETS_VALUE",
        "PERSONAL_MONTHLY_INCOME",
        "POSTAL_ADDRESS_TYPE",
        "PRODUCT",
        # NOTA: PROFESSIONAL_BOROUGH y PROFESSIONAL_CITY fueron removidas del preprocessing (alta cardinalidad + muchos missing)
        "PROFESSIONAL_PHONE_AREA_CODE",
        "PROFESSIONAL_STATE",
        "PROFESSIONAL_ZIP_3",
        "PROFESSION_CODE",
        "QUANT_BANKING_ACCOUNTS",
        "QUANT_CARS",
        "QUANT_DEPENDANTS",
        "QUANT_SPECIAL_BANKING_ACCOUNTS",
        "RESIDENCE_TYPE",
        "RESIDENCIAL_BOROUGH",
        "RESIDENCIAL_CITY",
        "RESIDENCIAL_PHONE_AREA_CODE",
        "RESIDENCIAL_STATE",
        "RESIDENCIAL_ZIP_3",
        "SEX",
        "STATE_OF_BIRTH",
]
