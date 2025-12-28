from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from src.preprocessing import build_preprocessing_pipeline

POS_WEIGHT = 0.7392 / 0.2600   # ~2.83

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def make_logreg_pipeline():
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        )),
    ])

def make_hgb_pipeline():
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=42
        ))
    ])

def make_gb_pipeline():
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", GradientBoostingClassifier(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.015,
            random_state=42,
            subsample=0.65,
            min_samples_split=40,
            min_samples_leaf=20,
            max_features="sqrt",
        )),
    ])

def make_xgb_pipeline():
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", XGBClassifier(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.2,
            reg_lambda=1.5,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
        )),
    ])

def make_lgbm_pipeline():
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed")
    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", LGBMClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.02,
            num_leaves=15,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_samples=30,
            reg_alpha=0.3,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        )),
    ])

def make_catboost_pipeline():
    # CatBoost is external, optional import 
    from catboost import CatBoostClassifier

    return Pipeline([
        ("prep", build_preprocessing_pipeline()),
        ("model", CatBoostClassifier(
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            scale_pos_weight=float(POS_WEIGHT),
            allow_writing_files=False,
        )),
    ])

def get_podium():
    podium = {
        "logreg_balanced": make_logreg_pipeline(),
        "hgb_balanced": make_hgb_pipeline(),
        "gb_sample_weight": make_gb_pipeline(),
        "catboost_pos_weight": make_catboost_pipeline(),
        "xgboost_sample_weight": make_xgb_pipeline(),
        "lightgbm_sample_weight": make_lgbm_pipeline(),
    }

    return podium

def get_tuning_candidates():
    """
    Returns:
      dict[name] = {"pipe": Pipeline, "params": dict, "n_iter": int}
    """
    common_preproc = {
        "prep__preprocess__cat__ohe__min_frequency": [0.002, 0.005, 0.01, 0.02],
        "prep__winsorize__q_low": [0.005, 0.01, 0.02],
        "prep__winsorize__q_high": [0.98, 0.99, 0.995],
    }

    candidates = {
        "logreg": {
            "pipe": make_logreg_pipeline(),
            "params": {
                **common_preproc,
                "model__C": np.logspace(-3.5, 1.5, 10),
                "model__class_weight": [None, "balanced"],
                "model__solver": ["lbfgs", "saga"],
                "model__penalty": ["l2"],
            },
            "n_iter": 30,
        },
        "hgb": {
            "pipe": make_hgb_pipeline(),
            "params": {
                **common_preproc,
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
                "model__max_leaf_nodes": [31, 63, 127],
                "model__min_samples_leaf": [40, 80, 120, 200],
                "model__max_depth": [None, 3, 5],
                "model__l2_regularization": np.logspace(-2, 1, 10),
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 50,
        },
        "gb": {
            "pipe": make_gb_pipeline(),
            "params": {
                **common_preproc,
                "model__n_estimators": [300, 500, 800],
                "model__learning_rate": [0.005, 0.01, 0.02],
                "model__max_depth": [2, 3],
                "model__subsample": [0.6, 0.75],
                "model__min_samples_split": [40, 60, 80],
                "model__min_samples_leaf": [20, 40, 60],
                "model__max_features": ["sqrt", "log2", None],
            },
            "n_iter": 50,
        },
        "catboost": {
            "pipe": make_catboost_pipeline(),
            "params": {
                **common_preproc,
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.02, 0.03, 0.05],
                "model__l2_leaf_reg": [3, 5, 10, 20],
                "model__iterations": [400, 800, 1200],
                "model__scale_pos_weight": [2.0, float(POS_WEIGHT), 3.0, 3.5],
            },
            "n_iter": 35,
        },
        "xgboost": {
            "pipe": make_xgb_pipeline(),
            "params": {
                **common_preproc,
                "model__n_estimators": [300, 500, 800],
                "model__learning_rate": [0.01, 0.02, 0.05],
                "model__max_depth": [3, 4, 5],
                "model__min_child_weight": [3, 5, 8, 12],
                "model__subsample": [0.6, 0.75],
                "model__colsample_bytree": [0.6, 0.75],
                "model__gamma": [0.1, 0.2, 0.4],
                "model__reg_alpha": [0.1, 0.3, 0.5],
                "model__reg_lambda": [1.5, 2.0, 3.0],
            },
            "n_iter": 50,
        },
        "lightgbm": {
            "pipe": make_lgbm_pipeline(),
            "params": {
                **common_preproc,
                "model__n_estimators": [400, 600, 900],
                "model__learning_rate": [0.01, 0.02, 0.05],
                "model__num_leaves": [15, 31],
                "model__max_depth": [3, 4, 6],
                "model__min_child_samples": [30, 60, 100],
                "model__subsample": [0.6, 0.75],
                "model__colsample_bytree": [0.6, 0.75],
                "model__reg_alpha": [0.1, 0.3, 0.5],
                "model__reg_lambda": [2.0, 3.0, 5.0],
            },
            "n_iter": 50,
        },
    }
    return candidates
