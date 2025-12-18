"""
Configuración de modelos para entrenamiento.
Este archivo permite agregar nuevos modelos fácilmente para compararlos.

Para agregar un nuevo modelo, simplemente añade una entrada al diccionario retornado
por get_models_config().
"""

from typing import Dict, Any, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


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
    
    return {
        "Logistic Regression": {
            "class": LogisticRegression,
            "params": {
                "random_state": 42,
                "max_iter": 5000,
                "solver": "lbfgs",
                "C": 0.3,  # Reducido de 0.5 para más regularización
                "tol": 1e-4,
            },
            "class_weight": "balanced",
        },
        
        "Random Forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 300,  # Aumentado para más capacidad
                "max_depth": 8,  # Reducido de 10 para más regularización (reduce overfitting)
                "min_samples_split": 30,  # Aumentado de 20 para más regularización
                "min_samples_leaf": 15,  # Aumentado de 10 para más regularización
                "random_state": 42,
                "n_jobs": -1,
                "max_features": "sqrt",
                "max_samples": 0.7,  # Reducido de 0.8 para más regularización (reduce overfitting)
            },
            "class_weight": "balanced",
        },
        
        "Gradient Boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 600,  # Aumentado de 400 para más capacidad
                "max_depth": 3,  # Reducido de 4 para más regularización (reduce overfitting)
                "learning_rate": 0.02,  # Reducido de 0.025 (más conservador, mejor con más n_estimators)
                "random_state": 42,
                "subsample": 0.7,  # Reducido de 0.75 para más regularización
                "min_samples_split": 30,  # Aumentado de 25 para más regularización
                "min_samples_leaf": 15,  # Aumentado de 12 para más regularización
                "max_features": "sqrt",
            },
            "class_weight": "sample_weight",
        },
    }


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
