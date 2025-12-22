"""
Complete Preprocessing Pipeline for Credit Risk Analysis.
Includes: cleaning, feature engineering, missing values handling, encoding, and scaling.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from src.config import PREPROCESSOR_FILE

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

ID_COL = "ID_CLIENT"
TARGET_COL = "TARGET_LABEL_BAD=1"

# Y/N Columns: Normalized to "Y"/"N", NaN preserved as a distinct category for encoding
YN_COLUMNS = [
    "FLAG_RESIDENCIAL_PHONE",
    "FLAG_MOBILE_PHONE",
    "COMPANY",
    "FLAG_PROFESSIONAL_PHONE",
    "FLAG_ACSP_RECORD",
]

# Columns to remove: High cardinality + many missing values (provide no useful info)
HIGH_CARDINALITY_MANY_MISSING_COLS = [
    "PROFESSIONAL_CITY",
    "PROFESSIONAL_BOROUGH",
]

# Variables to create missing indicators for (1 if missing, 0 if present)
# Missingness can be informative (e.g., no spouse, no formal job)
MISSING_INDICATOR_COLS = [
    "PROFESSION_CODE",
    "MONTHS_IN_RESIDENCE",
    "MATE_PROFESSION_CODE",
    "MATE_EDUCATION_LEVEL",
    "RESIDENCE_TYPE",
    "OCCUPATION_TYPE",
]

# Thresholds for encoding strategies based on cardinality
GROUPING_THRESHOLD = 100  # Columns with >100 categories: Frequency Encoding (no artificial order)
MIN_FREQUENCY_FOR_GROUPING = 10  # Categories with <10 occurrences grouped into "OTHER" (medium cardinality)


class PreprocessingPipeline:
    """
    Reusable complete preprocessing pipeline.
    Saves all transformers to apply to new data.
    """

    def __init__(self, low_cardinality_threshold: int = 20):
        """
        Initializes the pipeline.

        Args:
            low_cardinality_threshold: Threshold to consider low cardinality (default: 20)
        """
        self.low_cardinality_threshold = low_cardinality_threshold
        self.is_fitted = False

        # Store transformers and configurations
        self.constant_columns_removed: List[str] = []
        self.high_cardinality_many_missing_removed: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.binary_cat_columns: List[str] = []
        self.binary_cat_columns_for_imputation: List[str] = []
        self.high_cardinality_cols_for_imputation: List[str] = []  # High cardinality excluded from imputation
        self.ohe_cat_columns: List[str] = []
        self.frequency_encoding_columns: List[str] = []  # Columns using Frequency Encoding
        self.frequency_encoders: Dict[str, Dict[str, float]] = {}  # Map category -> frequency
        self.rare_categories_map: Dict[str, List[str]] = {}  # Map of rare categories to group
        self.feature_engineering_features: List[str] = []

        # Transformers
        self.categorical_imputer: Optional[SimpleImputer] = None
        self.numeric_imputer: Optional[SimpleImputer] = None
        self.binary_encoder: Optional[OrdinalEncoder] = None
        self.yn_encoder: Optional[OrdinalEncoder] = None  # Special encoder for Y/N columns
        self.ohe_encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[MinMaxScaler] = None

    def __setstate__(self, state: dict):
        """
        Automatically called when joblib deserializes the object.
        Cleans obsolete attributes from previous versions (outlier_limits, outlier_cols).
        """
        # Clean obsolete attributes from previous versions
        obsolete_attrs = ['outlier_limits', 'outlier_cols']
        for attr in obsolete_attrs:
            if attr in state:
                del state[attr]
        
        # Restore clean state
        self.__dict__.update(state)

    def _step1_initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Initial Cleaning
        - Remove ID_CLIENT
        - Normalize Y/N columns (Y/y->Y, N/n->N, keep NaN for later encoding)
        - Remove constant columns
        - Remove high cardinality columns with many missing values
        """
        df = df.copy()

        # Remove ID_CLIENT if exists
        if ID_COL in df.columns:
            df = df.drop(columns=[ID_COL])

        # Normalize Y/N columns: Y/y->Y, N/n->N, keep NaN
        # (Numeric conversion happens in encoding to preserve NaN as distinct category)
        for col in YN_COLUMNS:
            if col in df.columns:
                df[col] = df[col].replace({"Y": "Y", "y": "Y", "N": "N", "n": "N", 1: "Y", 0: "N"})
                df[col] = df[col].astype(object)  # Keep as object for later encoding

        # Identify and remove constant columns and high cardinality columns with many missing values
        if not self.is_fitted:
            # Detect constant columns: zero variance (all values equal)
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
            
            # Identify high cardinality columns with many missing values
            high_card_missing_cols = [
                col for col in HIGH_CARDINALITY_MANY_MISSING_COLS if col in df.columns
            ]
            self.high_cardinality_many_missing_removed = high_card_missing_cols
            if high_card_missing_cols:
                print(f"Removing {len(high_card_missing_cols)} high cardinality + many missing columns: {high_card_missing_cols}")

        # Remove identified columns
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
        Step 2: Handle Outliers
        
        Winsorization is NOT applied. Based on EDA, outlier percentage is low (~2% max) 
        and extreme values are informative for credit risk.
        """
        return df.copy()

    def _step3_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Feature Engineering
        Creates 8 types of combined features according to the plan.
        """
        df = df.copy()

        # 1. Financial Features: ratios and aggregations capturing payment capacity
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

        # 2. Stability Features: persistence indicators (reduce risk)
        if "MONTHS_IN_RESIDENCE" in df.columns:
            df["YEARS_IN_RESIDENCE"] = df["MONTHS_IN_RESIDENCE"] / 12

        if "MONTHS_IN_THE_JOB" in df.columns:
            df["YEARS_IN_JOB"] = df["MONTHS_IN_THE_JOB"] / 12

        if "MONTHS_IN_RESIDENCE" in df.columns and "MONTHS_IN_THE_JOB" in df.columns:
            df["STABILITY_SCORE"] = (
                df["MONTHS_IN_RESIDENCE"].fillna(0) + df["MONTHS_IN_THE_JOB"].fillna(0)
            ) / 24

        # 3. Contact Features: more methods = higher reliability
        # NOTE: FLAG_MOBILE_PHONE is removed (constant), only use FLAG_RESIDENCIAL_PHONE and FLAG_EMAIL
        contact_cols = ["FLAG_RESIDENCIAL_PHONE", "FLAG_EMAIL"]
        if all(col in df.columns for col in contact_cols):
            df["CONTACT_METHODS_COUNT"] = (
                (df["FLAG_RESIDENCIAL_PHONE"] == "Y").astype(int).fillna(0)
                + df["FLAG_EMAIL"].fillna(0)
            )

        # 4. Card Features: total quantity and presence of major cards
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
            # NOTE: QUANT_ADDITIONAL_CARDS is removed (constant 0), not used

            df["HAS_MAJOR_CARDS"] = (
                (df["FLAG_VISA"].fillna(0) + df["FLAG_MASTERCARD"].fillna(0)) > 0
            ).astype(int)

        # 5. Geographic Features: matches between locations (indicate stability)
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

        # 6. Banking Account Features: total and presence of special accounts
        if "QUANT_BANKING_ACCOUNTS" in df.columns and "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["TOTAL_BANKING_ACCOUNTS"] = (
                df["QUANT_BANKING_ACCOUNTS"].fillna(0)
                + df["QUANT_SPECIAL_BANKING_ACCOUNTS"].fillna(0)
            )

        if "QUANT_SPECIAL_BANKING_ACCOUNTS" in df.columns:
            df["HAS_SPECIAL_ACCOUNTS"] = (df["QUANT_SPECIAL_BANKING_ACCOUNTS"] > 0).astype(int)

        # 7. Age Features: age squared (captures non-linear relationships)
        if "AGE" in df.columns:
            df["AGE_SQUARED"] = df["AGE"] ** 2

        # 8. Missing Value Features: binary indicators (created in step 4)

        # Save list of created features (for reference)
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
        Step 4: Handling Missing Values
        - Create missing indicators
        - Impute categoricals with Mode (excluding Y/N, binary, and high cardinality)
        - Impute numericals with Median
        - High cardinality columns preserve NaN for Frequency Encoding
        """
        df = df.copy()

        # Create missing indicators BEFORE imputation (missingness can be informative)
        for col in MISSING_INDICATOR_COLS:
            if col in df.columns:
                indicator_col = f"MISSING_{col}"
                df[indicator_col] = df[col].isna().astype(int)

        # Separate categorical and numerical columns for imputation
        # Y/N, binary, and high cardinality are excluded: preserve NaN for encoding
        yn_cols_in_data = [col for col in YN_COLUMNS if col in df.columns]
        
        if not self.is_fitted:
            all_categorical_columns = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            
            # Identify binary columns (2 unique values, excluding Y/N)
            potential_binary_cols = [
                col for col in all_categorical_columns
                if col not in yn_cols_in_data and df[col].nunique(dropna=True) == 2
    ]
            
            # Identify high cardinality columns (>100) that will use Frequency Encoding
            # These are excluded from imputation to preserve NaN as "MISSING" in encoding
            high_card_cols_for_imputation = [
                col for col in all_categorical_columns
                if col not in yn_cols_in_data 
                and col not in potential_binary_cols
                and df[col].nunique(dropna=True) > GROUPING_THRESHOLD
            ]

            # Exclude Y/N, binary, and high cardinality from imputer (preserve NaN for encoding)
            cols_to_exclude = yn_cols_in_data + potential_binary_cols + high_card_cols_for_imputation
            self.categorical_columns = [col for col in all_categorical_columns if col not in cols_to_exclude]
            self.binary_cat_columns_for_imputation = potential_binary_cols
            self.high_cardinality_cols_for_imputation = high_card_cols_for_imputation
            
            if high_card_cols_for_imputation:
                print(f"High cardinality columns excluded from imputation (NaN->'MISSING' in encoding): {high_card_cols_for_imputation}")
            
            self.numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

            # Remove target from numericals if exists
            if TARGET_COL in self.numeric_columns:
                self.numeric_columns.remove(TARGET_COL)
        else:
            # During transformation, use saved columns
            if hasattr(self, 'binary_cat_columns_for_imputation'):
                binary_cols = self.binary_cat_columns_for_imputation
            else:
                binary_cols = []
            
            if hasattr(self, 'high_cardinality_cols_for_imputation'):
                high_card_cols = self.high_cardinality_cols_for_imputation
            else:
                high_card_cols = []
            
            cols_to_exclude = yn_cols_in_data + binary_cols + high_card_cols

        # Impute categoricals with Mode (most frequent value)
        cat_cols_to_impute = [col for col in self.categorical_columns if col in df.columns]
        if cat_cols_to_impute:
            if not self.is_fitted:
                self.categorical_imputer = SimpleImputer(strategy="most_frequent")
                self.categorical_imputer.fit(df[cat_cols_to_impute])
            df[cat_cols_to_impute] = self.categorical_imputer.transform(
                df[cat_cols_to_impute]
            )

        # Impute numericals with Median (robust to outliers)
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
    
        # Create Age Groups after imputing AGE (for categorical encoding)
        if "AGE" in df.columns:
            df["AGE_GROUP"] = pd.cut(
                df["AGE"],
                bins=[0, 30, 40, 50, 60, 100],
                labels=["<30", "30-40", "40-50", "50-60", "60+"],
            )
            # Convert to string for encoding
            df["AGE_GROUP"] = df["AGE_GROUP"].astype(str)

        return df

    def _step5_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5: Encoding
        - Binary: OrdinalEncoder (preserves NaN)
        - Y/N: OrdinalEncoder (preserves NaN)
        - Low Cardinality (<=20): OneHotEncoder
        - Medium Cardinality (21-100): Group rare + OneHotEncoder
        - High Cardinality (>100): Frequency Encoding (no artificial order)
        """
        df = df.copy()

        # Identify categorical columns (after feature engineering)
        cat_cols = [
            col
            for col in df.columns
            if col in self.categorical_columns or df[col].dtype == "object"
        ]

        # Y/N treated specially: NaN converts to "MISSING" to preserve as category
        yn_cols_in_data = [col for col in YN_COLUMNS if col in df.columns]
        
        if not self.is_fitted:
            # Use binary identified in imputation, or identify here
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

            # Separate by cardinality: low (OneHot), medium (group+OneHot), high (Frequency)
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
                print(f"High cardinality columns (>100) using Frequency Encoding: {high_card_cols}")
            
            self.ohe_cat_columns = low_card_cols + medium_card_cols  # Both use OneHot (medium after grouping)
            self.frequency_encoding_columns = high_card_cols

        # Encoding Binary: OrdinalEncoder (NaN -> "MISSING" -> numeric category)
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

        # Group rare categories in medium cardinality columns (before OneHot)
        # Identify medium cardinality columns needing grouping
        if not self.is_fitted:
            medium_card_cols_to_group = [
                col for col in self.ohe_cat_columns
                if col in df.columns and self.low_cardinality_threshold < df[col].nunique(dropna=True) <= GROUPING_THRESHOLD
            ]
            
            # Save mapping of rare categories to apply in transformation
            for col in medium_card_cols_to_group:
                value_counts = df[col].value_counts()
                rare_categories = value_counts[value_counts < MIN_FREQUENCY_FOR_GROUPING].index.tolist()
                if rare_categories:
                    self.rare_categories_map[col] = rare_categories
                    df[col] = df[col].replace(rare_categories, "OTHER")
                    print(f"Grouped {len(rare_categories)} rare categories in '{col}' as 'OTHER'")
        else:
            # During transformation: apply grouping using saved mapping
            for col, rare_cats in self.rare_categories_map.items():
                if col in df.columns:
                    df[col] = df[col].replace(rare_cats, "OTHER")
        
        # Encoding OneHot: low and medium cardinality (after grouping)
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

        # Encoding Frequency: high cardinality (>100) - encodes by frequency (no artificial order)
        # More frequent categories have higher values
        freq_cols = [col for col in self.frequency_encoding_columns if col in df.columns]
        if freq_cols:
            for col in freq_cols:
                if not self.is_fitted:
                    # Calculate relative frequencies (proportion of appearance in dataset)
                    value_counts = df[col].value_counts()
                    total = len(df[col].dropna())
                    # Map: category -> relative frequency (0-1)
                    freq_map = (value_counts / total).to_dict()
                    # For NaN, use average frequency of rare categories or minimum value
                    if pd.isna(df[col]).any():
                        rare_freq = value_counts[value_counts < MIN_FREQUENCY_FOR_GROUPING].sum() / total
                        freq_map["MISSING"] = rare_freq if rare_freq > 0 else 0.001
                    self.frequency_encoders[col] = freq_map
                    print(f"Frequency Encoding applied to '{col}': {len(freq_map)} mapped categories")
                
                # Apply encoding: replace category with its relative frequency
                df[col] = df[col].fillna("MISSING").map(self.frequency_encoders[col])
                # If new categories (unknown), use minimum frequency
                if df[col].isna().any():
                    min_freq = min(self.frequency_encoders[col].values())
                    df[col] = df[col].fillna(min_freq)
                # Convert to numeric (now it's a continuous numerical feature)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.001)

        return df

    def _step6_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 6: Scaling with MinMaxScaler
        
        Normalizes all numerical features to [0, 1] range so models like 
        Logistic Regression and Neural Networks converge better.
        """
        df = df.copy()

        # Get numeric columns (after encoding)
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
        Fits the pipeline with training data and transforms train/val/test.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame (optional)
            test_df: Test DataFrame (optional)

        Returns:
            Tuple with transformed numpy arrays (train, val, test)
        """
        self.is_fitted = False

        # Process train: apply 6 pipeline steps
        train_processed = self._step1_initial_cleaning(train_df)
        train_processed = self._step2_handle_outliers(train_processed)
        train_processed = self._step3_feature_engineering(train_processed)
        train_processed = self._step4_missing_values(train_processed)
        train_processed = self._step5_encoding(train_processed)
        train_processed = self._step6_scaling(train_processed)

        self.is_fitted = True

        # Process val and test with same fitted pipeline
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

        # Convert to numpy arrays for model training
        train_array = train_processed.values
        val_array = val_processed.values if val_processed is not None else None
        test_array = test_processed.values if test_processed is not None else None

        return train_array, val_array, test_array

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms new data using the fitted pipeline.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed numpy array
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before calling transform()")


        # Execute preprocessing steps
        df_processed = self._step1_initial_cleaning(df)
        df_processed = self._step2_handle_outliers(df_processed)
        df_processed = self._step3_feature_engineering(df_processed)
        df_processed = self._step4_missing_values(df_processed)
        df_processed = self._step5_encoding(df_processed)
        df_processed = self._step6_scaling(df_processed)

        return df_processed.values

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Saves the complete pipeline using joblib.
        Ensures obsolete attributes are not saved.

        Args:
            filepath: Path to save. If None, uses PREPROCESSOR_FILE from config.
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")

        filepath = filepath or PREPROCESSOR_FILE
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Clean obsolete attributes before saving
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
        Loads a saved pipeline.
        Cleans obsolete attributes from previous versions for compatibility.

        Args:
            filepath: Pipeline path. If None, uses PREPROCESSOR_FILE from config.

        Returns:
            Loaded pipeline
        """
        filepath = filepath or PREPROCESSOR_FILE
        pipeline = joblib.load(filepath)
        
        # Clean obsolete attributes
        obsolete_attrs = ['outlier_limits', 'outlier_cols']
        for attr in obsolete_attrs:
            if hasattr(pipeline, attr):
                try:
                    delattr(pipeline, attr)
                except:
                    pass
        
        return pipeline


# Compatibility function with existing code
def preprocess_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    low_cardinality_threshold: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compatibility function.
    Creates, fits, and transforms data using the pipeline.
    """
    pipeline = PreprocessingPipeline(low_cardinality_threshold=low_cardinality_threshold)
    train, val, test = pipeline.fit_transform(train_df, val_df, test_df)
    return train, val, test
