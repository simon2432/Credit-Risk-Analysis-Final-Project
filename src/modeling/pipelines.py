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
        "hgb_balanced": make_hgb_pipeline(),
        "catboost_pos_weight": make_catboost_pipeline(),
    }

def get_tuning_candidates():
    """
    Returns:
      dict[name] = {"pipe": Pipeline, "params": dict, "n_iter": int}
    """
    common_preproc = {
        "prep__preprocess__cat__ohe__min_frequency": [0.005, 0.01, 0.02],
        "prep__winsorize__q_low": [0.005, 0.01],
        "prep__winsorize__q_high": [0.99, 0.995],
    }

    candidates = {
        "logreg": {
            "pipe": make_logreg_pipeline(),
            "params": {
                **common_preproc,
                "model__C": np.logspace(-3, 2, 10),
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 25,
        },
        "hgb": {
            "pipe": make_hgb_pipeline(),
            "params": {
                **common_preproc,
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [31, 63, 127],
                "model__min_samples_leaf": [20, 50, 100],
                "model__max_depth": [None, 3, 5],
                "model__l2_regularization": np.logspace(-3, 1, 8),
                "model__class_weight": [None, "balanced"],
            },
            "n_iter": 30,
        },
        "catboost": {
            "pipe": make_catboost_pipeline(),
            "params": {
                **common_preproc,
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 10],
                "model__iterations": [500, 800, 1200],
                "model__scale_pos_weight": [2.0, float(POS_WEIGHT), 3.5],
            },
            "n_iter": 25,
        },
    }
    return candidates