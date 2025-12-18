"""
Pipeline completo de preprocessing para Credit Risk Analysis.
Incluye: limpieza, feature engineering, missing values, encoding y escalado.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import PREPROCESSOR_FILE

# ============================================================================
# CONSTANTES Y CONFIGURACIÓN
# ============================================================================

ID_COL = "ID_CLIENT"
TARGET_COL = "TARGET_LABEL_BAD=1"

# Columnas Y/N: se normalizan a "Y"/"N" y se preservan NaN como categoría distinta en encoding
YN_COLUMNS = [
    "FLAG_RESIDENCIAL_PHONE",
    "FLAG_MOBILE_PHONE",
    "COMPANY",
    "FLAG_PROFESSIONAL_PHONE",
    "FLAG_ACSP_RECORD",
]

# Columnas removidas: alta cardinalidad + muchos missing (no aportan información útil)
HIGH_CARDINALITY_MANY_MISSING_COLS = [
    "PROFESSIONAL_CITY",
    "PROFESSIONAL_BOROUGH",
]

# Variables para crear indicadores de missing (1 si falta, 0 si existe)
# El missing puede ser informativo (ej: sin cónyuge, sin trabajo formal)
MISSING_INDICATOR_COLS = [
    "PROFESSION_CODE",
    "MONTHS_IN_RESIDENCE",
    "MATE_PROFESSION_CODE",
    "MATE_EDUCATION_LEVEL",
    "RESIDENCE_TYPE",
    "OCCUPATION_TYPE",
]

# Umbrales para estrategias de encoding según cardinalidad
GROUPING_THRESHOLD = 100  # Columnas con >100 categorías: Frequency Encoding (sin orden artificial)
MIN_FREQUENCY_FOR_GROUPING = 10  # Categorías con <10 ocurrencias se agrupan en "OTROS" (para media cardinalidad)


class PreprocessingPipeline:
    """
    Pipeline completo de preprocessing reutilizable.
    Guarda todos los transformadores para aplicar en nuevos datos.
    """

    def __init__(self, low_cardinality_threshold: int = 20):
        """
        Inicializa el pipeline.

        Args:
            low_cardinality_threshold: Umbral para considerar baja cardinalidad (default: 20)
        """
        self.low_cardinality_threshold = low_cardinality_threshold
        self.is_fitted = False

        # Almacenar transformadores y configuraciones
        self.constant_columns_removed: List[str] = []
        self.high_cardinality_many_missing_removed: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.binary_cat_columns: List[str] = []
        self.binary_cat_columns_for_imputation: List[str] = []
        self.high_cardinality_cols_for_imputation: List[str] = []  # Alta cardinalidad excluidas de imputación
        self.ohe_cat_columns: List[str] = []
        self.frequency_encoding_columns: List[str] = []  # Columnas con Frequency Encoding
        self.frequency_encoders: Dict[str, Dict[str, float]] = {}  # Mapeo categoría → frecuencia
        self.rare_categories_map: Dict[str, List[str]] = {}  # Mapeo de categorías raras para agrupar
        self.feature_engineering_features: List[str] = []

        # Transformadores
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.binary_encoder: Optional[OrdinalEncoder] = None
        self.yn_encoder: Optional[OrdinalEncoder] = None  # Encoder especial para columnas Y/N
        self.ohe_encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[MinMaxScaler] = None

    def __setstate__(self, state: dict):
        """
        Se llama automáticamente cuando joblib deserializa el objeto.
        Limpia atributos obsoletos de versiones anteriores (outlier_limits, outlier_cols).
        """
        # Limpiar atributos obsoletos de versiones anteriores
        obsolete_attrs = ['outlier_limits', 'outlier_cols']
        for attr in obsolete_attrs:
            if attr in state:
                del state[attr]
        
        # Restaurar el estado limpio
        self.__dict__.update(state)

    def _step1_initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 1: Limpieza inicial
        - Remover ID_CLIENT
        - Normalizar columnas Y/N (Y/y→Y, N/n→N, mantener NaN para encoding posterior)
        - Remover columnas constantes
        - Remover columnas de alta cardinalidad con muchos missing values
        """
        df = df.copy()

        # Remover ID_CLIENT si existe
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])

        # Normalizar columnas Y/N: Y/y→Y, N/n→N, mantener NaN
        # (La conversión a numérico se hace en encoding para preservar NaN como categoría distinta)
        for col in YN_COLUMNS:
            if col in df.columns:
                df[col] = df[col].replace({"Y": "Y", "y": "Y", "N": "N", "n": "N", 1: "Y", 0: "N"})
                df[col] = df[col].astype(object)  # Mantener como object para encoding posterior

        # Identificar y remover columnas constantes y de alta cardinalidad con muchos missing
        if not self.is_fitted:
            # Detectar columnas constantes: sin varianza (todos los valores iguales)
            constant_cols = []
            for col in df.columns:
                if col == ID_COL:
                    continue
                unique_count = df[col].nunique(dropna=True)
                if unique_count == 0 or unique_count == 1:
                    constant_cols.append(col)
                elif df[col].dtype in ["int64", "float64", "int32", "float32"]:
                    if df[col].std() == 0 or pd.isna(df[col].std()):
                        constant_cols.append(col)

            self.constant_columns_removed = constant_cols
            if constant_cols:
                print(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            
            # Identificar columnas de alta cardinalidad con muchos missing
            high_card_missing_cols = [
                col for col in HIGH_CARDINALITY_MANY_MISSING_COLS if col in df.columns
            ]
            self.high_cardinality_many_missing_removed = high_card_missing_cols
            if high_card_missing_cols:
                print(f"Removing {len(high_card_missing_cols)} high cardinality + many missing columns: {high_card_missing_cols}")

        # Remover columnas identificadas
        cols_to_remove = []
        if self.constant_columns_removed:
            cols_to_remove.extend([col for col in self.constant_columns_removed if col in df.columns])
        if hasattr(self, 'high_cardinality_many_missing_removed') and self.high_cardinality_many_missing_removed:
            cols_to_remove.extend([col for col in self.high_cardinality_many_missing_removed if col in df.columns])
        
        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)

        return df

    def _step2_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 2: Manejo de outliers
        
        No se aplica Winsorization. Basado en el EDA, el porcentaje de outliers
        es bajo (~2% máximo) y los valores extremos son informativos para credit risk.
        """
        return df.copy()

    def _step3_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 3: Feature Engineering
        Crea 8 tipos de features combinadas según el plan.
        """
        df = df.copy()

        # 1. Features Financieras: ratios y agregaciones que capturan capacidad de pago
        if "PERSONAL_MONTHLY_INCOME" in df.columns and "OTHER_INCOMES" in df.columns:
            df["TOTAL_MONTHLY_INCOME"] = (
                df["PERSONAL_MONTHLY_INCOME"].fillna(0) + df["OTHER_INCOMES"].fillna(0)
            )

        if "PERSONAL_MONTHLY_INCOME" in df.columns and "PERSONAL_ASSETS_VALUE" in df.columns:
            df["INCOME_TO_ASSETS_RATIO"] = (
                df["PERSONAL_MONTHLY_INCOME"] / (df["PERSONAL_ASSETS_VALUE"] + 1)
            )

        if "TOTAL_MONTHLY_INCOME" in df.columns and "QUANT_DEPENDANTS" in df.columns:
            df["INCOME_PER_DEPENDANT"] = (
                df["TOTAL_MONTHLY_INCOME"] / (df["QUANT_DEPENDANTS"] + 1)
            )

        if "OTHER_INCOMES" in df.columns and "PERSONAL_MONTHLY_INCOME" in df.columns:
            df["INCOME_RATIO"] = (
                df["OTHER_INCOMES"] / (df["PERSONAL_MONTHLY_INCOME"] + 1e-6)
            )

        if "PERSONAL_ASSETS_VALUE" in df.columns and "QUANT_DEPENDANTS" in df.columns:
            df["ASSETS_PER_DEPENDANT"] = (
                df["PERSONAL_ASSETS_VALUE"] / (df["QUANT_DEPENDANTS"] + 1)
            )

        # 2. Features de Estabilidad: indicadores de permanencia (reducen riesgo)
        if "MONTHS_IN_RESIDENCE" in df.columns:
            df["YEARS_IN_RESIDENCE"] = df["MONTHS_IN_RESIDENCE"] / 12

        if "MONTHS_IN_THE_JOB" in df.columns:
            df["YEARS_IN_JOB"] = df["MONTHS_IN_THE_JOB"] / 12

        if "MONTHS_IN_RESIDENCE" in df.columns and "MONTHS_IN_THE_JOB" in df.columns:
            df["STABILITY_SCORE"] = (
                df["MONTHS_IN_RESIDENCE"].fillna(0) + df["MONTHS_IN_THE_JOB"].fillna(0)
            ) / 24

        # 3. Features de Contacto: más métodos = más confiabilidad
        # NOTA: FLAG_MOBILE_PHONE se elimina (constante), solo usamos FLAG_RESIDENCIAL_PHONE y FLAG_EMAIL
        contact_cols = ["FLAG_RESIDENCIAL_PHONE", "FLAG_EMAIL"]
        if all(col in df.columns for col in contact_cols):
            df["CONTACT_METHODS_COUNT"] = (
                (df["FLAG_RESIDENCIAL_PHONE"] == "Y").astype(int).fillna(0)
                + df["FLAG_EMAIL"].fillna(0)
            )

        # 4. Features de Tarjetas: cantidad total y presencia de tarjetas principales
        card_cols = [
            "FLAG_VISA",
            "FLAG_MASTERCARD",
            "FLAG_DINERS",
            "FLAG_AMERICAN_EXPRESS",
            "FLAG_OTHER_CARDS",
        ]
        if all(col in df.columns for col in card_cols):
            df["TOTAL_CARDS"] = (
                df["FLAG_VISA"].fillna(0)
                + df["FLAG_MASTERCARD"].fillna(0)
                + df["FLAG_DINERS"].fillna(0)
                + df["FLAG_AMERICAN_EXPRESS"].fillna(0)
                + df["FLAG_OTHER_CARDS"].fillna(0)
            )
            # NOTA: QUANT_ADDITIONAL_CARDS se elimina (constante 0), no se usa

            df["HAS_MAJOR_CARDS"] = (
                (df["FLAG_VISA"].fillna(0) + df["FLAG_MASTERCARD"].fillna(0)) > 0
            ).astype(int)

        # 5. Features Geográficas: coincidencias entre ubicaciones (indican estabilidad)
        if "RESIDENCIAL_STATE" in df.columns and "PROFESSIONAL_STATE" in df.columns:
            df["SAME_STATE_RES_PROF"] = (
                df["RESIDENCIAL_STATE"] == df["PROFESSIONAL_STATE"]
            ).astype(int)

        if "RESIDENCIAL_ZIP_3" in df.columns and "PROFESSIONAL_ZIP_3" in df.columns:
            df["SAME_ZIP_RES_PROF"] = (
                df["RESIDENCIAL_ZIP_3"] == df["PROFESSIONAL_ZIP_3"]
            ).astype(int)

        if "STATE_OF_BIRTH" in df.columns and "RESIDENCIAL_STATE" in df.columns:
            df["BORN_IN_RESIDENCE_STATE"] = (
                df["STATE_OF_BIRTH"] == df["RESIDENCIAL_STATE"]
            ).astype(int)

        # 6. Features de Cuentas Bancarias: total y presencia de cuentas especiales
        if "QUANT_BANKING_ACCOUNTS" in df.columns and "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["TOTAL_BANKING_ACCOUNTS"] = (
                df["QUANT_BANKING_ACCOUNTS"].fillna(0)
                + df["QUANT_SPECIAL_BANKING_ACCOUNTS"].fillna(0)
            )

        if "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["HAS_SPECIAL_ACCOUNTS"] = (df["QUANT_SPECIAL_BANKING_ACCOUNTS"] > 0).astype(int)

        # 7. Features de Edad: edad al cuadrado (captura relaciones no lineales)
        if "AGE" in df.columns:
            df["AGE_SQUARED"] = df["AGE"] ** 2

        # 8. Features de Missing Values: indicadores binarios (se crean en paso 4)

        # Guardar lista de features creadas (para referencia)
        if not self.is_fitted:
            self.feature_engineering_features = [
                "TOTAL_MONTHLY_INCOME",
                "INCOME_TO_ASSETS_RATIO",
                "INCOME_PER_DEPENDANT",
                "INCOME_RATIO",
                "ASSETS_PER_DEPENDANT",
                "YEARS_IN_RESIDENCE",
                "YEARS_IN_JOB",
                "STABILITY_SCORE",
                "CONTACT_METHODS_COUNT",
                "TOTAL_CARDS",
                "HAS_MAJOR_CARDS",
                "SAME_STATE_RES_PROF",
                "SAME_ZIP_RES_PROF",
                "BORN_IN_RESIDENCE_STATE",
                "TOTAL_BANKING_ACCOUNTS",
                "HAS_SPECIAL_ACCOUNTS",
                "AGE_SQUARED",
            ]

        return df

    def _step4_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 4: Manejo de Missing Values
        - Crear indicadores de missing
        - Imputar categóricas con moda (excluyendo Y/N, binarias y alta cardinalidad)
        - Imputar numéricas con mediana
        - Las columnas de alta cardinalidad preservan NaN para Frequency Encoding
        """
        df = df.copy()

        # Crear indicadores de missing ANTES de imputar (el missing puede ser informativo)
        for col in MISSING_INDICATOR_COLS:
            if col in df.columns:
                indicator_col = f"MISSING_{col}"
                df[indicator_col] = df[col].isna().astype(int)

        # Separar categóricas y numéricas para imputación
        # Y/N, binarias y alta cardinalidad se excluyen: preservamos NaN como categoría distinta en encoding
        yn_cols_in_data = [col for col in YN_COLUMNS if col in df.columns]
        
        if not self.is_fitted:
            all_categorical_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            
            # Identificar binarias (2 valores únicos, excluyendo Y/N)
            potential_binary_cols = [
                col for col in all_categorical_columns
                if col not in yn_cols_in_data and df[col].nunique(dropna=True) == 2
    ]
            
            # Identificar columnas de alta cardinalidad (>100) que usarán Frequency Encoding
            # Estas se excluyen de imputación para preservar NaN como "MISSING" en encoding
            high_card_cols_for_imputation = [
                col for col in all_categorical_columns
                if col not in yn_cols_in_data 
                and col not in potential_binary_cols
                and df[col].nunique(dropna=True) > GROUPING_THRESHOLD
            ]

            # Excluir Y/N, binarias y alta cardinalidad del imputer (preservarán NaN para encoding)
            cols_to_exclude = yn_cols_in_data + potential_binary_cols + high_card_cols_for_imputation
            self.categorical_columns = [col for col in all_categorical_columns if col not in cols_to_exclude]
            self.binary_cat_columns_for_imputation = potential_binary_cols
            self.high_cardinality_cols_for_imputation = high_card_cols_for_imputation
            
            if high_card_cols_for_imputation:
                print(f"Columnas de alta cardinalidad excluidas de imputación (NaN→'MISSING' en encoding): {high_card_cols_for_imputation}")
            
            self.numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

            # Remover target de numéricas si existe
            if TARGET_COL in self.numeric_columns:
                self.numeric_columns.remove(TARGET_COL)
        else:
            # En transformación, usar las columnas guardadas
            if hasattr(self, 'binary_cat_columns_for_imputation'):
                binary_cols = self.binary_cat_columns_for_imputation
            else:
                binary_cols = []
            
            if hasattr(self, 'high_cardinality_cols_for_imputation'):
                high_card_cols = self.high_cardinality_cols_for_imputation
            else:
                high_card_cols = []
            
            cols_to_exclude = yn_cols_in_data + binary_cols + high_card_cols

        # Imputar categóricas con moda (valor más frecuente)
        cat_cols_to_impute = [col for col in self.categorical_columns if col in df.columns]
        if cat_cols_to_impute:
            if not self.is_fitted:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                self.categorical_imputer.fit(df[cat_cols_to_impute])
            df[cat_cols_to_impute] = self.categorical_imputer.transform(
                df[cat_cols_to_impute]
            )

        # Imputar numéricas con mediana (robusta a outliers)
        numeric_cols_to_impute = [
            col for col in self.numeric_columns if col in df.columns
        ]
        if numeric_cols_to_impute:
            if not self.is_fitted:
                self.numeric_imputer = SimpleImputer(strategy="median")
                self.numeric_imputer.fit(df[numeric_cols_to_impute])
            df[numeric_cols_to_impute] = self.numeric_imputer.transform(
                df[numeric_cols_to_impute]
            )
    
        # Crear grupos de edad después de imputar AGE (para encoding categórico)
        if "AGE" in df.columns:
            df["AGE_GROUP"] = pd.cut(
                df["AGE"],
                bins=[0, 30, 40, 50, 60, 100],
                labels=["<30", "30-40", "40-50", "50-60", "60+"],
            )
            # Convertir a string para encoding
            df["AGE_GROUP"] = df["AGE_GROUP"].astype(str)

        return df

    def _step5_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 5: Encoding
        - Binarias: OrdinalEncoder (preserva NaN)
        - Y/N: OrdinalEncoder (preserva NaN)
        - Baja cardinalidad (≤20): OneHotEncoder
        - Media cardinalidad (21-100): Agrupar poco frecuentes + OneHotEncoder
        - Alta cardinalidad (>100): Frequency Encoding (sin orden artificial)
        """
        df = df.copy()

        # Identificar columnas categóricas (después de feature engineering)
        cat_cols = [
            col
            for col in df.columns
            if col in self.categorical_columns or df[col].dtype == "object"
        ]

        # Y/N se tratan especialmente: NaN se convierte en "MISSING" para preservarlo como categoría
        yn_cols_in_data = [col for col in YN_COLUMNS if col in df.columns]
        
        if not self.is_fitted:
            # Usar binarias identificadas en imputación, o identificarlas aquí
            if hasattr(self, 'binary_cat_columns_for_imputation') and self.binary_cat_columns_for_imputation:
                self.binary_cat_columns = [col for col in self.binary_cat_columns_for_imputation if col in df.columns]
            else:
                self.binary_cat_columns = [
                    col for col in cat_cols
                    if col not in yn_cols_in_data and df[col].nunique(dropna=True) == 2
                ]
            multi_cat_columns = [
                col for col in cat_cols
                if col not in self.binary_cat_columns and col not in yn_cols_in_data
            ]

            # Separar por cardinalidad: baja (OneHot), media (agrupar+OneHot), alta (Frequency)
            low_card_cols = [
        col for col in multi_cat_columns
                if df[col].nunique(dropna=True) <= self.low_cardinality_threshold
    ]
            
            medium_card_cols = [
                col for col in multi_cat_columns
                if self.low_cardinality_threshold < df[col].nunique(dropna=True) <= GROUPING_THRESHOLD
            ]
            
            high_card_cols = [
                col for col in multi_cat_columns
                if df[col].nunique(dropna=True) > GROUPING_THRESHOLD
            ]
            
            if high_card_cols:
                print(f"Columnas de alta cardinalidad (>100) usarán Frequency Encoding: {high_card_cols}")
            
            self.ohe_cat_columns = low_card_cols + medium_card_cols  # Ambas usan OneHot (media después de agrupar)
            self.frequency_encoding_columns = high_card_cols

        # Encoding binarias: OrdinalEncoder (NaN → "MISSING" → categoría numérica)
        if self.binary_cat_columns:
            binary_cols = [col for col in self.binary_cat_columns if col in df.columns]
            if binary_cols:
                binary_df = df[binary_cols].copy().fillna("MISSING")
                
                if not self.is_fitted:
                    self.binary_encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    self.binary_encoder.fit(binary_df)
                
                encoded_binary = self.binary_encoder.transform(binary_df)
                for i, col in enumerate(binary_cols):
                    df[col] = encoded_binary[:, i]
        
        # Encoding Y/N: OrdinalEncoder (Y=0, N=1, MISSING=2)
        if yn_cols_in_data:
            yn_df = df[yn_cols_in_data].copy().fillna("MISSING")
            
            if not self.is_fitted:
                self.yn_encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                self.yn_encoder.fit(yn_df)
            
            encoded_yn = self.yn_encoder.transform(yn_df)
            for i, col in enumerate(yn_cols_in_data):
                df[col] = encoded_yn[:, i]

        # Agrupar categorías poco frecuentes en columnas de media cardinalidad (antes de OneHot)
        # Identificar columnas de media cardinalidad que necesitan agrupación
        if not self.is_fitted:
            medium_card_cols_to_group = [
                col for col in self.ohe_cat_columns
                if col in df.columns and self.low_cardinality_threshold < df[col].nunique(dropna=True) <= GROUPING_THRESHOLD
            ]
            
            # Guardar mapeo de categorías raras para aplicar en transformación
            for col in medium_card_cols_to_group:
                value_counts = df[col].value_counts()
                rare_categories = value_counts[value_counts < MIN_FREQUENCY_FOR_GROUPING].index.tolist()
                if rare_categories:
                    self.rare_categories_map[col] = rare_categories
                    df[col] = df[col].replace(rare_categories, "OTROS")
                    print(f"Agrupadas {len(rare_categories)} categorías poco frecuentes en '{col}' como 'OTROS'")
        else:
            # En transformación: aplicar agrupación usando mapeo guardado
            for col, rare_cats in self.rare_categories_map.items():
                if col in df.columns:
                    df[col] = df[col].replace(rare_cats, "OTROS")
        
        # Encoding OneHot: baja y media cardinalidad (después de agrupar)
        ohe_cols = [col for col in self.ohe_cat_columns if col in df.columns]
        if ohe_cols:
            if not self.is_fitted:
                self.ohe_encoder = OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                )
                self.ohe_encoder.fit(df[ohe_cols])

            ohe_array = self.ohe_encoder.transform(df[ohe_cols])
            ohe_df = pd.DataFrame(
                ohe_array,
                columns=self.ohe_encoder.get_feature_names_out(ohe_cols),
                index=df.index,
            )
            df = df.drop(columns=ohe_cols)
            df = pd.concat([df, ohe_df], axis=1)

        # Encoding Frequency: alta cardinalidad (>100) - codifica por frecuencia (sin orden artificial)
        # Las categorías más frecuentes tienen valores más altos, sin orden arbitrario
        freq_cols = [col for col in self.frequency_encoding_columns if col in df.columns]
        if freq_cols:
            for col in freq_cols:
                if not self.is_fitted:
                    # Calcular frecuencias relativas (proporción de aparición en el dataset)
                    value_counts = df[col].value_counts()
                    total = len(df[col].dropna())
                    # Mapeo: categoría → frecuencia relativa (0-1)
                    freq_map = (value_counts / total).to_dict()
                    # Para NaN, usar frecuencia promedio de categorías raras o valor mínimo
                    if pd.isna(df[col]).any():
                        rare_freq = value_counts[value_counts < MIN_FREQUENCY_FOR_GROUPING].sum() / total
                        freq_map["MISSING"] = rare_freq if rare_freq > 0 else 0.001
                    self.frequency_encoders[col] = freq_map
                    print(f"Frequency Encoding aplicado a '{col}': {len(freq_map)} categorías mapeadas")
                
                # Aplicar encoding: reemplazar categoría por su frecuencia relativa
                df[col] = df[col].fillna("MISSING").map(self.frequency_encoders[col])
                # Si hay categorías nuevas (unknown), usar frecuencia mínima
                if df[col].isna().any():
                    min_freq = min(self.frequency_encoders[col].values())
                    df[col] = df[col].fillna(min_freq)
                # Convertir a numérico (ahora es una feature numérica continua)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.001)

        return df

    def _step6_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Paso 6: Escalado con MinMaxScaler
        
        Normaliza todas las features numéricas al rango [0, 1] para que modelos
        como Logistic Regression y Neural Networks converjan mejor.
        """
        df = df.copy()

        # Obtener columnas numéricas finales (después de encoding)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if TARGET_COL in numeric_cols:
            numeric_cols.remove(TARGET_COL)

        if numeric_cols:
            if not self.is_fitted:
                self.scaler = MinMaxScaler()
                self.scaler.fit(df[numeric_cols])

            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        return df

    def fit_transform(
        self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Ajusta el pipeline con datos de entrenamiento y transforma train/val/test.

        Args:
            train_df: DataFrame de entrenamiento
            val_df: DataFrame de validación (opcional)
            test_df: DataFrame de test (opcional)

        Returns:
            Tupla con arrays numpy transformados (train, val, test)
        """
        self.is_fitted = False

        # Procesar train: aplicar los 6 pasos del pipeline
        train_processed = self._step1_initial_cleaning(train_df)
        train_processed = self._step2_handle_outliers(train_processed)
        train_processed = self._step3_feature_engineering(train_processed)
        train_processed = self._step4_missing_values(train_processed)
        train_processed = self._step5_encoding(train_processed)
        train_processed = self._step6_scaling(train_processed)

        self.is_fitted = True

        # Procesar val y test con el mismo pipeline ajustado
        val_processed = None
        test_processed = None

        if val_df is not None:
            val_processed = self._step1_initial_cleaning(val_df)
            val_processed = self._step2_handle_outliers(val_processed)
            val_processed = self._step3_feature_engineering(val_processed)
            val_processed = self._step4_missing_values(val_processed)
            val_processed = self._step5_encoding(val_processed)
            val_processed = self._step6_scaling(val_processed)

        if test_df is not None:
            test_processed = self._step1_initial_cleaning(test_df)
            test_processed = self._step2_handle_outliers(test_processed)
            test_processed = self._step3_feature_engineering(test_processed)
            test_processed = self._step4_missing_values(test_processed)
            test_processed = self._step5_encoding(test_processed)
            test_processed = self._step6_scaling(test_processed)

        # Convertir a numpy arrays para entrenamiento del modelo
        train_array = train_processed.values
        val_array = val_processed.values if val_processed is not None else None
        test_array = test_processed.values if test_processed is not None else None

        return train_array, val_array, test_array

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforma nuevos datos usando el pipeline ajustado.

        Args:
            df: DataFrame a transformar

        Returns:
            Array numpy transformado
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform()")


        # Ejecutar pasos de preprocessing
        # Los atributos obsoletos ya están manejados por __getattribute__ y __setstate__
        df_processed = self._step1_initial_cleaning(df)
        df_processed = self._step2_handle_outliers(df_processed)
        df_processed = self._step3_feature_engineering(df_processed)
        df_processed = self._step4_missing_values(df_processed)
        df_processed = self._step5_encoding(df_processed)
        df_processed = self._step6_scaling(df_processed)

        return df_processed.values

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Guarda el pipeline completo usando joblib.
        Asegura que no se guarden atributos obsoletos.

        Args:
            filepath: Ruta donde guardar. Si None, usa PREPROCESSOR_FILE de config.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        filepath = filepath or PREPROCESSOR_FILE
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Limpiar atributos obsoletos antes de guardar (compatibilidad con versiones anteriores)
        obsolete_attrs = ['outlier_limits', 'outlier_cols']
        for attr in obsolete_attrs:
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except:
                    pass

        joblib.dump(self, filepath)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "PreprocessingPipeline":
        """
        Carga un pipeline guardado.
        Limpia atributos obsoletos de versiones anteriores para compatibilidad.

        Args:
            filepath: Ruta del pipeline. Si None, usa PREPROCESSOR_FILE de config.

        Returns:
            Pipeline cargado
        """
        filepath = filepath or PREPROCESSOR_FILE
        pipeline = joblib.load(filepath)
        
        # Limpiar atributos obsoletos de versiones anteriores (compatibilidad)
        obsolete_attrs = ['outlier_limits', 'outlier_cols']
        for attr in obsolete_attrs:
            if hasattr(pipeline, attr):
                try:
                    delattr(pipeline, attr)
                except:
                    pass
        
        return pipeline


# Función de compatibilidad con código existente
def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    low_cardinality_threshold: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Función de compatibilidad con código existente.
    Crea un pipeline, lo ajusta y transforma los datos.
    """
    pipeline = PreprocessingPipeline(low_cardinality_threshold=low_cardinality_threshold)
    train, val, test = pipeline.fit_transform(train_df, val_df, test_df)
    return train, val, test
