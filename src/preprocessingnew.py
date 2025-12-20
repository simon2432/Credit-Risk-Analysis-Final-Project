# Preprocessing pipeline aligned with EDA_FINDINGS:
# - Drop ID + constant columns
# - Convert placeholder strings like "NULL", "NULL.1"... to missing
# - Drop selected very-high-cardinality columns (configurable)
# - Add missing indicators for selected columns (informative missingness)
# - Cap outliers (winsorization) using train quantiles
# - Feature engineering (income ratios, stability, counts, cyclical payment day)
# - ColumnTransformer with:
#     * numeric: median imputation + scaling
#     * categorical: most_frequent imputation + OneHot with infrequent grouping
#

from __future__ import annotations
import re 
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    """All knobs for the preprocessing pipeline."""
    # Drop columns (ID + columns you decided to remove due to huge cardinality + missingness)
    drop_cols: tuple[str, ...] = (
        "ID_CLIENT",
        "CITY_OF_BIRTH",
        "RESIDENCIAL_CITY",
        "RESIDENCIAL_BOROUGH",
        "PROFESSIONAL_CITY",
        "PROFESSIONAL_BOROUGH",
    )

    # Missing indicators (informative missingness from EDA)
    missing_indicator_cols: tuple[str, ...] = (
        "PROFESSION_CODE",
        "MONTHS_IN_RESIDENCE",
        "MATE_PROFESSIONAL_CODE",
        "MATE_EDUCATION_LEVEL",
        "RESINDENCE_TYPE",
        "OCCUPATION_TYPE",
    )

    # Outlier capping (winsorization) – based on your EDA outlier list
    winsorize_cols: tuple[str, ...] = (
        "PERSONAL_MONTHLY_INCOME",
        "OTHER_INCOMES",
        "PERSONAL_ASSETS_VALUE",
        "AGE",
        "MONTHS_IN_RESIDENCE",
        "MONTHS_IN_THE_JOB",
        "PROFESSION_CODE",
        "MATE_PROFESSION_CODE",
        "MARITAL_STATUS",
        "QUANT_DEPENDANTS",
    )

    winsorize_low_q: float = 0.01
    winsorize_high_q: float = 0.99

    # OneHotEncoder infrequent handling (helps with high-cardinality geo columns if kept)
    ohe_min_frequency: float = 0.01 # group categories < 1% as "infrequent" (tune)

# -----------------------------
# Transformers
# -----------------------------

class CleanNullLikeStrings(BaseEstimator, TransformerMixin):
    """Replace placeholders "NULL", "NULL.1", "NULL.2"... and empty strings with NaN.

    Analogy: it's like replacing fake ingredients (placeholders) with "missing" so the recipe
    (the model) doesn't learn nonsense.
    """

    _null_re = re.compile(r"^NULL([.][0-9]+)?$", re.IGNORECASE)

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        obj_cols = X.select_dtypes(include=["object", "string"]).columns
        for col in obj_cols:
            s = X[col].astype("string").str.strip()
            s = s.replace("", np.nan)
            s = s.where(~s.str.match(self._null_re, na=False), np.nan)

            s_obj = s.astype(object)
            X[col] = s_obj.where(pd.notna(s_obj), np.nan)
        
        return X
    

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Iterable[str]):
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=list(self.columns), errors="ignore")

class DropConstantColumns(BaseEstimator, TransformerMixin):
    """Drop columns with no variance (nunique<=1)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.constant_col_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.constant_col_, errors="ignore")

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """Create MISSING_* indicator columns for selected features."""

    def __init__(self, columns: Iterable[str]):
        self.columns = tuple(columns)

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = [c for c in self.columns if c in X.columns]
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.columns_:
            X[f"MISSING_{c}"] = X[c].isna().astype(int)
        return X

class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap numeric outliers using train quantiles (winsorization).

    Variables:
    - **X**: features DataFrame
    - **q_low/q_high**: percentiles used as caps (learned on train)
    """
    def __init__(self, columns: Iterable[str], q_low: float = 0.01, q_high: float = 0.99):
         self.columns = tuple(columns)
         self.q_low = q_low
         self.q_high = q_high
     
    def fit(self, X: pd.DataFrame, y=None):
        self.bounds_ = {}
        for c in self.columns:
            if c not in X.columns:
                continue
            s = pd.to_numeric(X[c], errors="coerce")
            if s.notna().sum() == 0:
                continue
            self.bounds_[c] = (float(s.quantile(self.q_low)), float(s.quantile(self.q_high)))
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c, (lo, hi) in self.bounds_.items():
            s = pd.to_numeric(X[c], errors="coerce")
            X[c] = s.clip(lower=lo, upper=hi)
        return X
    
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering based on EDA insights.

    Creates:
    - INCOME_TOTAL, INCOME_RATIO, HAS_OTHER_INCOME
    - YEARS_IN_RESIDENCE, YEARS_IN_JOB, STABILITY_SCORE
    - CARDS_COUNT, HAS_ANY_CARD
    - DOCS_COUNT
    - PAYMENT_DAY_SIN, PAYMENT_DAY_COS (cyclical)
    """

    def __init__(self, eps: float = 1e-6, payment_period: float = 30.0):
        self.eps = eps
        self.payment_period = payment_period

    def fit(self, X: pd.DataFrame, y=None):
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        def num(col: str) -> pd.Series:
            if col not in X.columns:
                return pd.Series(np.nan, index=X.index)
            return pd.to_numeric(X[col], errors="coerce")
        
        pm_income = num("PERSONAL_MONTHLY_INCOME")
        other_incomme = num("OTHER_INCOMES")

        if ("PERSONAL_MONTHLY_INCOME" in X.columns) or ("OTHER_INCOMES" in X.columns):
            X["INCOME_TOTAL"] = pm_income.fillna(0) + other_incomme.fillna(0)
            X["INCOME_RATIO"] = other_incomme / (pm_income + self.eps)
            X["HAS_OTHER_INCOME"] = (other_incomme.fillna(0) > 0).astype(int)
        
        months_res = num("MONTHS_IN_RESIDENCE")
        months_job = num("MONTHS_IN_THE_JOB")

        if "MONTHS_IN_RESIDENCE" in X.columns:
            X["YEARS_IN_RESIDENCE"] = months_res / 12.0
        if "MONTHS_IN_THE_JOB" in X.columns:
            X["YEARS_IN_JOB"] = months_job / 12.0
        
        if ("TEARS_IN_RESIDENCE" in X.columns) or ("YEARS_IN_JOB" in X.columns):
            yrs_res = pd.to_numeric(X.get("YEARS_IN_RESIDENCE"), errors="coerce")
            yrs_job = pd.to_numeric(X.get("YEARS_IN_JOB"), errors="coerce")
            X["STABILITY_SCORE"] = yrs_res.fillna(0) + yrs_job.fillna(0)

        # Cards aggregation
        card_flags = [
            "FLAG_VISA",
            "FLAG_MASTERCARD",
            "FLAG_DINERS",
            "FLAG_AMERICAN_EXPRESS",
            "FLAG_OTHER_CARDS",
        ]
        existing_cards = [c for c in card_flags if c in X.columns]
        if existing_cards:
            cards_mat = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce").fillna(0) for c in existing_cards})
            X["CARDS_COUNT"] = cards_mat.sum(axis=1)
            X["HAS_ANY_CARD"] = (X["CARDS_COUNT"] > 0).astype(int)
        
        # Documents aggregation
        doc_flags = [
            "FLAG_HOME_ADDRESS_DOCUMENT",
            "FLAG_RG",
            "FLAG_CPF",
            "FLAG_INCOME_PROOF",
        ]      
        existing_docs = [c for c in doc_flags if c in X.columns]
        if existing_docs:
            docs_mat = pd.DataFrame({c: pd.to_numeric(X[c], errors="coerce").fillna(0) for c in existing_docs})
            X["DOCS_COUNT"] = docs_mat.sum(axis=1)
        
        # PAYMENT_DAY cyclical encoding
        if "PAYMENT_DAY" in X.columns:
            day = pd.to_numeric(X["PAYMENT_DAY"], errors="coerce")
            angle = 2.0 * np.pi * (day / self.payment_period)
            X["PAYMENT_DAY_SIN"] = np.sin(angle)
            X["PAYMENT_DAY_COS"] = np.cos(angle)
        
        return X

# -----------------------------
# Builder
# -----------------------------

def build_preprocessing_pipeline(cfg: Optional[PreprocessConfig] = None) -> Pipeline:
    """Build the full preprocessing pipeline.

    Variables:
    - **X**: input features (DataFrame)
    - **y**: target (TARGET_LABEL_BAD=1)

    Why Pipeline helps: every learned step (imputer/ohe/scaler/outlier caps)
    is fit only on train, then reused consistently on val/test/production.
    """

    cfg = cfg or PreprocessConfig()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=cfg.ohe_min_frequency,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipe, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("clean_nulls", CleanNullLikeStrings()),
            ("drop_cols", DropColumns(cfg.drop_cols)),
            ("drop_constants", DropConstantColumns()),
            ("missing_indicators", MissingIndicatorAdder(cfg.missing_indicator_cols)),
            (
                "winsorize",
                Winsorizer(cfg.winsorize_cols, cfg.winsorize_low_q, cfg.winsorize_high_q),
            ),
            ("feature_engineering", FeatureEngineer()),
            ("preprocess", preprocessor),
        ]
    )