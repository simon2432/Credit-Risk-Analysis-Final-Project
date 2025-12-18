"""
Script completo para entrenar y evaluar modelos de credit risk.
Usa el PreprocessingPipeline y entrena múltiples modelos comparándolos.

Este script:
- Usa SOLO el dataset de Modeling (PAKDD2010_Modeling_Data.txt)
- Divide en 70% train, 15% validation, 15% test
- Entrena y evalúa modelos solo en train y validation
- Guarda test set en memoria pero NO lo usa (reservado para evaluación final)
- Calcula threshold óptimo en VALIDATION para TODOS los modelos
- Guarda un historial detallado de cada entrenamiento en JSON
- El archivo de predicción (PAKDD2010_Prediction_Data.txt) NO se usa
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODEL_FILE, MODEL_DIR, PREPROCESSOR_FILE
from src.data_utils import get_datasets, get_feature_target, get_train_val_sets
from src.preprocessing import PreprocessingPipeline, TARGET_COL
from src.models_config import get_models_config, create_model_instance

# Configuración
RANDOM_STATE = 42
TRAINING_HISTORY_DIR = MODEL_DIR.parent / "training_history"


def calculate_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calcula el threshold óptimo usando Youden's J statistic.
    
    Args:
        y_true: Valores reales del target
        y_proba: Probabilidades predichas
    
    Returns:
        Diccionario con threshold óptimo y métricas en ese threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predicciones con threshold óptimo
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    return {
        "optimal_threshold": float(optimal_threshold),
        "tpr_at_threshold": float(tpr[optimal_idx]),
        "fpr_at_threshold": float(fpr[optimal_idx]),
        "youden_j": float(youden_j[optimal_idx]),
        "metrics_at_optimal": {
            "f1": float(f1_score(y_true, y_pred_optimal)),
            "precision": float(precision_score(y_true, y_pred_optimal)),
            "recall": float(recall_score(y_true, y_pred_optimal)),
        }
    }


def evaluate_model_comprehensive(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    """
    Evalúa un modelo con múltiples métricas en train y validation, incluyendo threshold óptimo.
    El threshold óptimo se calcula en validation.
    
    Args:
        model: Modelo entrenado
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        model_name: Nombre del modelo
    
    Returns:
        Diccionario completo con todas las métricas (train y validation)
    """
    # Predicciones en train y validation
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Métricas básicas (threshold=0.5)
    train_metrics = {
        "roc_auc": float(roc_auc_score(y_train, y_train_proba)),
        "f1": float(f1_score(y_train, y_train_pred)),
        "precision": float(precision_score(y_train, y_train_pred)),
        "recall": float(recall_score(y_train, y_train_pred)),
    }
    
    val_metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_val_proba)),
        "f1": float(f1_score(y_val, y_val_pred)),
        "precision": float(precision_score(y_val, y_val_pred)),
        "recall": float(recall_score(y_val, y_val_pred)),
    }
    
    # Calcular threshold óptimo en VALIDATION (no en test, test se guarda pero no se usa)
    threshold_info = calculate_optimal_threshold(y_val, y_val_proba)
    
    return {
        "model_name": model_name,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "optimal_threshold_info": threshold_info,
    }


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: PreprocessingPipeline,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], float]:
    """
    Entrena múltiples modelos y los evalúa en train y validation.
    El conjunto de test se guarda en memoria pero NO se usa para evaluación
    (se reserva para evaluación final del modelo seleccionado).
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        X_test: Features de test (guardado en memoria, no usado)
        y_test: Target de test (guardado en memoria, no usado)
        pipeline: Pipeline de preprocessing
    
    Returns:
        Tupla con (mejor modelo, mejores métricas, lista de resultados de todos los modelos)
    """
    # Aplicar preprocessing (procesamos test también para guardarlo, aunque no lo usemos)
    print("\n" + "=" * 60)
    print("APPLYING PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Train input: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"Validation input: {X_val.shape[0]:,} rows × {X_val.shape[1]} features")
    print(f"Test input: {X_test.shape[0]:,} rows × {X_test.shape[1]} features (guardado en memoria, NO usado para evaluación)")
    print(f"\n[INFO] Applying ALL preprocessing steps: cleaning -> outliers -> feature engineering -> encoding -> scaling...")
    
    preprocessing_start = time.time()
    X_train_processed, X_val_processed, X_test_processed = pipeline.fit_transform(
        X_train, X_val, X_test
    )
    preprocessing_time = time.time() - preprocessing_start
    
    print(f"\n[OK] Preprocessing completed in {preprocessing_time:.2f} seconds ({preprocessing_time/60:.2f} minutes)")
    print(f"\nAfter preprocessing:")
    print(f"  Train: {X_train_processed.shape[0]:,} rows × {X_train_processed.shape[1]} features")
    print(f"  Validation: {X_val_processed.shape[0]:,} rows × {X_val_processed.shape[1]} features")
    print(f"  Test: {X_test_processed.shape[0]:,} rows × {X_test_processed.shape[1]} features (guardado, NO evaluado)")
    
    # Verificar calidad de datos procesados
    print(f"\nData quality check:")
    print(f"  Train stats - Min: {X_train_processed.min():.4f}, Max: {X_train_processed.max():.4f}")
    print(f"                Mean: {X_train_processed.mean():.4f}, Std: {X_train_processed.std():.4f}")
    print(f"  NaN count: {np.isnan(X_train_processed).sum()}")
    print(f"  Inf count: {np.isinf(X_train_processed).sum()}")
    
    if np.isnan(X_train_processed).sum() > 0 or np.isinf(X_train_processed).sum() > 0:
        print("  [WARNING] NaN or Inf values detected!")
    else:
        print("  [OK] No NaN or Inf values - data is clean")

    # Convertir targets a numpy (solo train y val se usan para entrenamiento/evaluación)
    y_train_np = y_train.values
    y_val_np = y_val.values
    
    # Test se guarda pero no se usa (se reserva para evaluación final del modelo seleccionado)
    # No necesitamos procesarlo ni validarlo aquí
    
    # Calcular sample_weight para modelos que lo necesiten
    sample_weights = compute_sample_weight('balanced', y_train_np)
    
    # Obtener configuración de modelos
    models_config = get_models_config()
    
    # Lista para guardar resultados de todos los modelos
    all_results = []
    results = {}
    best_model = None
    best_score = 0
    best_model_name = None

    print("\n" + "=" * 60)
    print(f"Training and evaluating {len(models_config)} models...")
    print("=" * 60)

    for model_name, model_config in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        print(f"  Training on: {X_train_processed.shape[0]:,} samples, {X_train_processed.shape[1]} features")
        print(f"  Target distribution: Class 0: {np.sum(y_train_np==0):,}, Class 1: {np.sum(y_train_np==1):,}")
        
        try:
            train_start = time.time()
            print(f"  Starting training...")
            
            # Crear instancia del modelo
            model = create_model_instance(model_config)
            
            # Entrenar con o sin sample_weight según configuración
            if model_config.get("class_weight") == "sample_weight":
                model.fit(X_train_processed, y_train_np, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train_np)
            
            train_time = time.time() - train_start
            print(f"  [OK] Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
            
            # Evaluar modelo completamente (solo train y validation, test no se usa)
            metrics = evaluate_model_comprehensive(
                model, X_train_processed, y_train_np, X_val_processed, y_val_np, model_name
            )
            
            # Agregar información de entrenamiento
            metrics["training_time_seconds"] = float(train_time)
            metrics["hyperparameters"] = model_config["params"].copy()
            metrics["class_weight_strategy"] = model_config.get("class_weight", None)
            
            results[model_name] = {"model": model, "metrics": metrics}
            all_results.append(metrics)

            # Mostrar métricas
            val_roc_auc = metrics["val_metrics"]["roc_auc"]
            print(f"  Train ROC-AUC: {metrics['train_metrics']['roc_auc']:.4f}")
            print(f"  Train F1: {metrics['train_metrics']['f1']:.4f}")
            print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
            print(f"  Validation F1: {metrics['val_metrics']['f1']:.4f}")
            print(f"  Optimal Threshold (validation): {metrics['optimal_threshold_info']['optimal_threshold']:.4f}")
            print(f"  Validation F1 (optimal threshold): {metrics['optimal_threshold_info']['metrics_at_optimal']['f1']:.4f}")

            # Seleccionar mejor modelo por ROC-AUC en validation
            if val_roc_auc > best_score:
                best_score = val_roc_auc
                best_model = model
                best_model_name = model_name

        except Exception as e:
            print(f"  [ERROR] Error training {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    if best_model is None:
        raise ValueError("No model was successfully trained")

    # Obtener métricas del mejor modelo
    best_metrics = results[best_model_name]["metrics"]
    
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Train ROC-AUC: {best_metrics['train_metrics']['roc_auc']:.4f}")
    print(f"Validation ROC-AUC: {best_metrics['val_metrics']['roc_auc']:.4f}")
    print(f"Optimal Threshold (validation): {best_metrics['optimal_threshold_info']['optimal_threshold']:.4f}")
    print("=" * 60)
    print("\n[INFO] Test set guardado en memoria pero NO usado para evaluación.")
    print("       Se reserva para evaluación final del modelo seleccionado.")

    return best_model, best_metrics, all_results, preprocessing_time


def save_training_history(
    all_results: List[Dict[str, Any]],
    best_model_name: str,
    dataset_info: Dict[str, Any],
    preprocessing_time: float,
) -> Path:
    """
    Guarda el historial completo del entrenamiento en un archivo JSON.
    
    Args:
        all_results: Lista con resultados de todos los modelos
        best_model_name: Nombre del mejor modelo
        dataset_info: Información sobre el dataset usado
        preprocessing_time: Tiempo de preprocessing
    
    Returns:
        Path del archivo guardado
    """
    # Crear directorio si no existe
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    # Crear timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_filename = f"training_history_{timestamp}.json"
    history_path = TRAINING_HISTORY_DIR / history_filename
    
    # Construir el historial completo
    history = {
        "timestamp": datetime.now().isoformat(),
        "timestamp_formatted": timestamp,
        "best_model": best_model_name,
        "dataset_info": dataset_info,
        "preprocessing_time_seconds": float(preprocessing_time),
        "models": all_results,
    }
    
    # Guardar JSON
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining history saved to: {history_path}")
    return history_path


def save_model_and_pipeline(
    model: Any, pipeline: PreprocessingPipeline, model_name: str, metrics: Dict[str, Any]
) -> None:
    """
    Guarda el modelo, pipeline y métricas.
    
    Args:
        model: Modelo entrenado
        pipeline: Pipeline de preprocessing
        model_name: Nombre del modelo
        metrics: Métricas del modelo
    """
    # Guardar modelo
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Guardar pipeline en la nueva ubicación: models/preprocessor/preprocessor.joblib
    pipeline.save(PREPROCESSOR_FILE)
    print(f"Pipeline saved to: {PREPROCESSOR_FILE}")

    # Guardar métricas en formato legible
    metrics_path = MODEL_DIR / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Best Model: {model_name}\n\n")
        f.write("Train Metrics (threshold=0.5):\n")
        for metric, value in metrics["train_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nValidation Metrics (threshold=0.5):\n")
        for metric, value in metrics["val_metrics"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nValidation Metrics (optimal threshold):\n")
        for metric, value in metrics["optimal_threshold_info"]["metrics_at_optimal"].items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nOptimal Threshold (calculated on validation): {metrics['optimal_threshold_info']['optimal_threshold']:.4f}\n")
        f.write("\n[NOTE] Test set guardado en memoria pero NO usado para evaluación.\n")
        f.write("       Se reserva para evaluación final del modelo seleccionado.\n")
    print(f"Metrics saved to: {metrics_path}")
    
    # Guardar threshold óptimo en un archivo separado para la API
    threshold_path = MODEL_DIR / "optimal_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(metrics["optimal_threshold_info"]["optimal_threshold"]))
    print(f"Optimal threshold saved to: {threshold_path}")


def main():
    """
    Función principal de entrenamiento.
    
    Estrategia:
    - Usa SOLO el dataset de Modeling (PAKDD2010_Modeling_Data.txt)
    - Divide en 70% train, 15% validation, 15% test
    - Evalúa modelos solo en train y validation
    - Guarda test set en memoria (NO usado para evaluación)
    - El archivo de predicción (PAKDD2010_Prediction_Data.txt) NO se usa
    """
    print("=" * 60)
    print("Credit Risk Model Training")
    print("=" * 60)
    print("\n[INFO] Estrategia: Usar solo dataset de Modeling")
    print("       División: 70% train, 15% validation, 15% test")
    print("       Evaluación: Solo train y validation")
    print("       Test: Guardado en memoria, NO usado (reservado para evaluación final)")

    # 1. Cargar datos (solo dataset de Modeling)
    print("\n1. Loading dataset de Modeling...")
    app_train, _, column_descriptions = get_datasets()
    
    print(f"   Dataset shape: {app_train.shape}")
    print(f"   [INFO] Archivo de predicción NO se usa para entrenamiento/evaluación")
    
    # 2. Separar features y target del dataset completo
    print("\n2. Separating features and target...")
    X_full = app_train.drop(columns=['TARGET_LABEL_BAD=1'])
    y_full = app_train['TARGET_LABEL_BAD=1']
    print(f"   X_full shape: {X_full.shape}")
    print(f"   y_full distribution:\n{y_full.value_counts(normalize=True)}")
    
    # 3. Dividir en 70% train, 15% validation, 15% test
    print("\n3. Splitting dataset into train/validation/test (70/15/15)...")
    from sklearn.model_selection import train_test_split
    
    # Primero dividir en train (70%) y temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full, test_size=0.3, random_state=RANDOM_STATE, shuffle=True, stratify=y_full
    )
    
    # Luego dividir temp en val (15%) y test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, shuffle=True, stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]:,} muestras ({X_train.shape[0]/len(X_full)*100:.1f}%)")
    print(f"   Validation: {X_val.shape[0]:,} muestras ({X_val.shape[0]/len(X_full)*100:.1f}%)")
    print(f"   Test: {X_test.shape[0]:,} muestras ({X_test.shape[0]/len(X_full)*100:.1f}%) - GUARDADO EN MEMORIA, NO USADO")
    print(f"   Total: {len(X_full):,} muestras")

    # 4. Información del dataset para el historial
    dataset_info = {
        "dataset_source": "PAKDD2010_Modeling_Data.txt only",
        "train_size": int(X_train.shape[0]),
        "validation_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
        "train_features": int(X_train.shape[1]),
        "target_distribution_train": {
            "class_0": int(y_train.value_counts().get(0, 0)),
            "class_1": int(y_train.value_counts().get(1, 0)),
        },
        "test_usage": "stored_in_memory_not_used_for_evaluation",
        "split_ratio": "70/15/15",
    }

    # 5. Crear pipeline
    print("\n5. Creating preprocessing pipeline...")
    pipeline = PreprocessingPipeline(low_cardinality_threshold=20)

    # 6. Entrenar modelos (el preprocessing se hace dentro de train_models)
    print("\n6. Training models...")
    best_model, best_metrics, all_results, preprocessing_time = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test, pipeline
    )

    # 7. Guardar historial de entrenamiento
    print("\n7. Saving training history...")
    history_path = save_training_history(
        all_results, best_metrics["model_name"], dataset_info, preprocessing_time
    )

    # 8. Guardar mejor modelo y pipeline
    print("\n8. Saving best model and pipeline...")
    save_model_and_pipeline(best_model, pipeline, best_metrics["model_name"], best_metrics)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best model: {best_metrics['model_name']}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
