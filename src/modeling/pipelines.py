from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.preprocessing import build_preprocessing_pipeline

POS_WEIGHT = 0.7392 / 0.2600   # ~2.83

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
        )),
    ])

def get_podium():
    return{
        "logreg_balanced": make_logreg_pipeline(),
        "hight_balanced": make_hgb_pipeline(),
        "catboost_pos_weight": make_catboost_pipeline(),
    }