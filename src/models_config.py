"""
Configuraci?n de modelos para entrenamiento.
Este archivo permite agregar nuevos modelos f?cilmente para compararlos.

Para agregar un nuevo modelo, simplemente a?ade una entrada al diccionario retornado
por get_models_config().

Modelos actuales:
- Gradient Boosting (sklearn): Baseline de gradient boosting
- XGBoost: Gradient boosting optimizado, muy popular en credit risk
- LightGBM: Gradient boosting r?pido y eficiente, maneja bien datasets grandes
- CatBoost: Gradient boosting con manejo nativo de categor?as, excelente para credit risk
"""

from typing import Dict, Any, Callable
from sklearn.ensemble import GradientBoostingClassifier
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
    Retorna la configuraci?n de todos los modelos a entrenar.
    
    Cada modelo debe tener:
    - "class": La clase del modelo (sin instanciar)
    - "params": Diccionario con hiperpar?metros para el constructor
    - "class_weight": Estrategia de balanceo ("balanced", None, o una funci?n)
      - Si es "balanced", se usa class_weight='balanced' en el constructor
      - Si es "sample_weight", se calcula sample_weight manualmente
      - Si es None, no se usa balanceo
    
    Returns:
        Diccionario con configuraci?n de modelos
    """
    
    models = {
        "Gradient Boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 1500,  # Reducido ligeramente para balance
                "max_depth": 5,  # Reducido (de 6 a 5) para reducir overfitting
                "learning_rate": 0.01,  # Aumentado ligeramente (de 0.008 a 0.01) para mejor balance
                "random_state": 42,
                "subsample": 0.8,  # Reducido (de 0.9 a 0.8) para m?s regularizaci?n
                "min_samples_split": 50,  # Aumentado (de 20 a 50) para m?s regularizaci?n
                "min_samples_leaf": 25,  # Aumentado (de 10 a 25) para m?s regularizaci?n
                "max_features": "sqrt",
            },
            "class_weight": "sample_weight",
        },
    }
    
    # Agregar XGBoost si est? disponible
    if XGBClassifier is not None:
        models["XGBoost"] = {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 1200,  # Reducido ligeramente
                "max_depth": 5,  # Reducido (de 6 a 5) para reducir overfitting
                "learning_rate": 0.02,  # Aumentado (de 0.015 a 0.02) para mejor balance
                "subsample": 0.8,  # Reducido (de 0.9 a 0.8) para m?s regularizaci?n
                "colsample_bytree": 0.8,  # Reducido (de 0.9 a 0.8) para m?s regularizaci?n
                "min_child_weight": 5,  # Aumentado (de 2 a 5) para m?s regularizaci?n
                "gamma": 0.2,  # Aumentado (de 0.05 a 0.2) para m?s regularizaci?n en splits
                "reg_alpha": 0.2,  # Aumentado (de 0.05 a 0.2) para m?s L1 regularization
                "reg_lambda": 1.5,  # Aumentado (de 0.8 a 1.5) para m?s L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",  # M?s r?pido para datasets grandes
                "eval_metric": "logloss",
            },
            "class_weight": "sample_weight",
        }
    
    # Agregar LightGBM si est? disponible
    if LGBMClassifier is not None:
        models["LightGBM"] = {
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 1500,  # Reducido ligeramente
                "max_depth": 6,  # Reducido (de 7 a 6) para reducir overfitting
                "learning_rate": 0.015,  # Aumentado (de 0.01 a 0.015) para mejor balance
                "num_leaves": 31,  # Reducido (de 63 a 31 = 2^5-1) para m?s regularizaci?n
                "subsample": 0.8,  # Reducido (de 0.9 a 0.8) para m?s regularizaci?n
                "colsample_bytree": 0.8,  # Reducido (de 0.9 a 0.8) para m?s regularizaci?n
                "min_child_samples": 30,  # Aumentado (de 15 a 30) para m?s regularizaci?n
                "reg_alpha": 0.2,  # Aumentado (de 0.05 a 0.2) para m?s L1 regularization
                "reg_lambda": 1.5,  # Aumentado (de 0.8 a 1.5) para m?s L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,  # Silenciar outputs
                "force_col_wise": True,  # Para evitar warnings
            },
            "class_weight": "sample_weight",
        }
    
    # Agregar CatBoost si est? disponible
    if CatBoostClassifier is not None:
        models["CatBoost"] = {
            "class": CatBoostClassifier,
            "params": {
                "iterations": 1200,  # Reducido ligeramente
                "depth": 6,  # Reducido (de 7 a 6) para reducir overfitting
                "learning_rate": 0.02,  # Aumentado (de 0.015 a 0.02) para mejor balance
                "l2_leaf_reg": 5,  # Aumentado (de 3 a 5) para más regularización L2
                "bootstrap_type": "Bernoulli",  # Tipo de bootstrap
                "subsample": 0.8,  # Reducido (de 0.9 a 0.8) para más regularización
                "loss_function": "Logloss",  # Funci?n de p?rdida
                "eval_metric": "AUC",  # M?trica de evaluaci?n
                "random_seed": 42,
                "verbose": False,  # Silenciar outputs
                "task_type": "CPU",
            },
            "class_weight": "sample_weight",
        }
    
    return models


def create_model_instance(model_config: Dict[str, Any]) -> Any:
    """
    Crea una instancia del modelo seg?n la configuraci?n.
    
    Args:
        model_config: Configuraci?n del modelo (retornada por get_models_config)
    
    Returns:
        Instancia del modelo configurado
    """
    model_class = model_config["class"]
    params = model_config["params"].copy()
    
    # Agregar class_weight si es necesario
    if model_config.get("class_weight") == "balanced":
        params["class_weight"] = "balanced"
    
    return model_class(**params)
