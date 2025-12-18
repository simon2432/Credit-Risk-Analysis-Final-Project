"""
Configuración de modelos para entrenamiento.
Este archivo permite agregar nuevos modelos fácilmente para compararlos.

Para agregar un nuevo modelo, simplemente añade una entrada al diccionario retornado
por get_models_config().

Modelos actuales:
- Gradient Boosting (sklearn): Baseline de gradient boosting
- XGBoost: Gradient boosting optimizado, muy popular en credit risk
- LightGBM: Gradient boosting rápido y eficiente, maneja bien datasets grandes
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


def get_models_config() -> Dict[str, Dict[str, Any]]:
    """
    Retorna la configuración de todos los modelos a entrenar.
    
    Cada modelo debe tener:
    - "class": La clase del modelo (sin instanciar)
    - "params": Diccionario con hiperparámetros para el constructor
    - "class_weight": Estrategia de balanceo ("balanced", None, o una función)
      - Si es "balanced", se usa class_weight='balanced' en el constructor
      - Si es "sample_weight", se calcula sample_weight manualmente
      - Si es None, no se usa balanceo
    
    Returns:
        Diccionario con configuración de modelos
    """
    
    models = {
        "Gradient Boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 800,  # Aumentado para más capacidad (con learning_rate bajo)
                "max_depth": 3,  # Mantener profundidad limitada para regularización
                "learning_rate": 0.015,  # Reducido para mejor generalización con más árboles
                "random_state": 42,
                "subsample": 0.65,  # Más regularización (reducido de 0.7)
                "min_samples_split": 40,  # Aumentado para más regularización (era 30)
                "min_samples_leaf": 20,  # Aumentado para más regularización (era 15)
                "max_features": "sqrt",
            },
            "class_weight": "sample_weight",
        },
    }
    
    # Agregar XGBoost si está disponible
    if XGBClassifier is not None:
        models["XGBoost"] = {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 500,  # Aumentado de 400 para más capacidad
                "max_depth": 3,  # Reducido de 4 para más regularización (reduce overfitting)
                "learning_rate": 0.025,  # Ligeramente reducido para mejor generalización
                "subsample": 0.75,  # Reducido de 0.8 para más regularización
                "colsample_bytree": 0.75,  # Reducido de 0.8 para más regularización
                "min_child_weight": 5,  # Aumentado de 3 para más regularización
                "gamma": 0.2,  # Aumentado de 0.1 para más regularización (min split loss)
                "reg_alpha": 0.2,  # Aumentado de 0.1 para más L1 regularization
                "reg_lambda": 1.5,  # Aumentado de 1.0 para más L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "tree_method": "hist",  # Más rápido para datasets grandes
                "eval_metric": "logloss",
            },
            "class_weight": "sample_weight",
        }
    
    # Agregar LightGBM si está disponible
    if LGBMClassifier is not None:
        models["LightGBM"] = {
            "class": LGBMClassifier,
            "params": {
                "n_estimators": 600,  # Aumentado de 500 (con learning_rate más bajo)
                "max_depth": 4,  # Reducido de 5 para reducir overfitting significativamente
                "learning_rate": 0.02,  # Reducido de 0.03 para mejor generalización
                "num_leaves": 15,  # REDUCIDO significativamente de 31 para reducir overfitting (era 2^5-1, ahora más restrictivo)
                "subsample": 0.7,  # Reducido de 0.8 para más regularización
                "colsample_bytree": 0.7,  # Reducido de 0.8 para más regularización
                "min_child_samples": 30,  # Aumentado de 20 para más regularización
                "reg_alpha": 0.3,  # Aumentado significativamente de 0.1 para más L1 regularization
                "reg_lambda": 2.0,  # Aumentado significativamente de 1.0 para más L2 regularization
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,  # Silenciar outputs
                "force_col_wise": True,  # Para evitar warnings
            },
            "class_weight": "sample_weight",
        }
    
    return models


def create_model_instance(model_config: Dict[str, Any]) -> Any:
    """
    Crea una instancia del modelo según la configuración.
    
    Args:
        model_config: Configuración del modelo (retornada por get_models_config)
    
    Returns:
        Instancia del modelo configurado
    """
    model_class = model_config["class"]
    params = model_config["params"].copy()
    
    # Agregar class_weight si es necesario
    if model_config.get("class_weight") == "balanced":
        params["class_weight"] = "balanced"
    
    return model_class(**params)
