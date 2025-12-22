"""
Model configuration for training.
This file allows easy addition of new models for comparison.

To add a new model, simply add an entry to the dictionary returned by get_models_config().

Current Models:
- Logistic Regression: Linear baseline (interpretable)
- Random Forest: Bagging ensemble (stable)
- Gradient Boosting (sklearn): Gradient boosting baseline
- XGBoost: Optimized gradient boosting, popular in credit risk
- LightGBM: Fast and efficient gradient boosting
- CatBoost: Best for categorical data (if available)
"""

from typing import Dict, Any, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Conditional imports for optional libraries
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def get_models_config() -> Dict[str, Dict[str, Any]]:
    """
    Returns the configuration for all models to be trained.
    
    Each model must have:
    - "class": The model class (not instantiated)
    - "params": Dictionary with constructor hyperparameters
    - "class_weight": Balancing strategy ("balanced", "sample_weight", or None)
      - "balanced": Uses class_weight='balanced' in constructor
      - "sample_weight": Calculates sample_weight manually (for models that don't support class_weight param)
      - None: No balancing
    
    Returns:
        Dictionary with model configurations
    """
    
    models = {
        # 1. Linear Baseline
        "Logistic Regression": {
            "class": LogisticRegression,
            "params": {
                "C": 1.0,
                "solver": "liblinear", # Robust for small/medium datasets and binary classification
                "max_iter": 1000,
                "random_state": 42,
            },
            "class_weight": "balanced",
        },

        # 2. Bagging Ensemble (Stable)
        "Random Forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 300,
                "max_depth": 10, # Limited depth to prevent overfitting
                "min_samples_leaf": 10,
                "random_state": 42,
                "n_jobs": -1,
            },
            "class_weight": "balanced",
        },

        # 3. Boosting Baseline (Sklearn)
        "Gradient Boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.05,
                "random_state": 42,
                "subsample": 0.8,
                "min_samples_split": 20,
                "max_features": "sqrt",
            },
            "class_weight": "sample_weight",
        },
    }
    
    # Add XGBoost if available
    if XGBClassifier is not None:
        models["XGBoost"] = {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 500,
                "max_depth": 3,
                "learning_rate": 0.025,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "min_child_weight": 5,
                "gamma": 0.2,
                "reg_alpha": 0.2, 
                "reg_lambda": 1.5,
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",
                "eval_metric": "logloss",
            },
            "class_weight": "sample_weight",
        }
    
    # Add LightGBM if available
    if LGBMClassifier is not None:
        models["LightGBM"] = {
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 600,
                "max_depth": 4, 
                "learning_rate": 0.02,
                "num_leaves": 15,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_samples": 30,
                "reg_alpha": 0.3,
                "reg_lambda": 2.0,
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
                "force_col_wise": True,
            },
            "class_weight": "sample_weight",
        }

    # Add CatBoost if available (Great for categoricals)
    if CatBoostClassifier is not None:
        models["CatBoost"] = {
            "class": CatBoostClassifier,
            "params": {
                "iterations": 800,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 3,
                "border_count": 128,
                "random_state": 42,
                "allow_writing_files": False,
                "verbose": False,
                "auto_class_weights": "Balanced", # CatBoost specific balancing
            },
            "class_weight": None, # Handled by auto_class_weights
        }
    
    return models


def create_model_instance(model_config: Dict[str, Any]) -> Any:
    """
    Creates a model instance from configuration.
    
    Args:
        model_config: Model configuration (from get_models_config)
    
    Returns:
        Configured model instance
    """
    model_class = model_config["class"]
    params = model_config["params"].copy()
    
    # Add standard sklearn class_weight if configured
    if model_config.get("class_weight") == "balanced":
        params["class_weight"] = "balanced"
    
    return model_class(**params)
