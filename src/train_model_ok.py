"""
Complete script to train and evaluate credit risk models.
Uses PreprocessingPipeline and trains multiple models for comparison.

This script:
- Uses ONLY the Modeling dataset (PAKDD2010_Modeling_Data.txt)
- Splits into 70% train, 15% validation, 15% test
- Trains and evaluates models on train and validation only
- Keeps test set in memory but does NOT use it (reserved for final evaluation)
- Calculates optimal threshold on VALIDATION for ALL models
- Saves detailed training history in JSON
- The prediction file (PAKDD2010_Prediction_Data.txt) is NOT used
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

# IMPORTING OPTIMIZED MODULES (_ok)
from src.preprocessing_ok import PreprocessingPipeline, TARGET_COL
from src.models_config_ok import get_models_config, create_model_instance

# Configuration
RANDOM_STATE = 42
TRAINING_HISTORY_DIR = MODEL_DIR.parent / "training_history"


def calculate_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """
    Calculates optimal threshold using Youden's J statistic.
    
    Args:
        y_true: True target values
        y_proba: Predicted probabilities
    
    Returns:
        Dictionary with optimal threshold and metrics at that threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions with optimal threshold
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
    Evaluates a model with multiple metrics on train and validation, including optimal threshold.
    Optimal threshold is calculated on validation set.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        model_name: Model name
    
    Returns:
        Dictionary with full metrics (train and validation)
    """
    # Predictions on train and validation
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    # Basic metrics (threshold=0.5)
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
    
    # Calculate optimal threshold on VALIDATION
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
    Trains multiple models and evaluates them on train and validation.
    Test set is kept in memory but NOT used for evaluation.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features (stored, not used)
        y_test: Test target (stored, not used)
        pipeline: Preprocessing pipeline
    
    Returns:
        Tuple (best model, best metrics, all results, preprocessing time)
    """
    # Apply preprocessing
    print("\n" + "=" * 60)
    print("APPLYING PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Train input: {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"Validation input: {X_val.shape[0]:,} rows × {X_val.shape[1]} features")
    print(f"Test input: {X_test.shape[0]:,} rows × {X_test.shape[1]} features (stored, NOT used)")
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
    print(f"  Test: {X_test_processed.shape[0]:,} rows × {X_test_processed.shape[1]} features (stored, NOT evaluated)")
    
    # Check data quality
    print(f"\nData quality check:")
    print(f"  Train stats - Min: {X_train_processed.min():.4f}, Max: {X_train_processed.max():.4f}")
    print(f"                Mean: {X_train_processed.mean():.4f}, Std: {X_train_processed.std():.4f}")
    print(f"  NaN count: {np.isnan(X_train_processed).sum()}")
    print(f"  Inf count: {np.isinf(X_train_processed).sum()}")
    
    if np.isnan(X_train_processed).sum() > 0 or np.isinf(X_train_processed).sum() > 0:
        print("  [WARNING] NaN or Inf values detected!")
    else:
        print("  [OK] No NaN or Inf values - data is clean")

    # Convert targets to numpy
    y_train_np = y_train.values
    y_val_np = y_val.values
    
    # Calculate sample_weight for models that need it
    sample_weights = compute_sample_weight('balanced', y_train_np)
    
    # Get model configuration
    models_config = get_models_config()
    
    # List to store results
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
            
            # Create model instance
            model = create_model_instance(model_config)
            
            # Train with or without sample_weight depending on config
            if model_config.get("class_weight") == "sample_weight":
                model.fit(X_train_processed, y_train_np, sample_weight=sample_weights)
            else:
                model.fit(X_train_processed, y_train_np)
            
            train_time = time.time() - train_start
            print(f"  [OK] Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
            
            # Evaluate model
            metrics = evaluate_model_comprehensive(
                model, X_train_processed, y_train_np, X_val_processed, y_val_np, model_name
            )
            
            # Add training info
            metrics["training_time_seconds"] = float(train_time)
            metrics["hyperparameters"] = model_config["params"].copy()
            metrics["class_weight_strategy"] = model_config.get("class_weight", None)
            
            results[model_name] = {"model": model, "metrics": metrics}
            all_results.append(metrics)

            # Show metrics
            val_roc_auc = metrics["val_metrics"]["roc_auc"]
            print(f"  Train ROC-AUC: {metrics['train_metrics']['roc_auc']:.4f}")
            print(f"  Train F1: {metrics['train_metrics']['f1']:.4f}")
            print(f"  Validation ROC-AUC: {val_roc_auc:.4f}")
            print(f"  Validation F1: {metrics['val_metrics']['f1']:.4f}")
            print(f"  Optimal Threshold (validation): {metrics['optimal_threshold_info']['optimal_threshold']:.4f}")
            print(f"  Validation F1 (optimal threshold): {metrics['optimal_threshold_info']['metrics_at_optimal']['f1']:.4f}")

            # Select best model by ROC-AUC
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

    # Get best metrics
    best_metrics = results[best_model_name]["metrics"]
    
    print("\n" + "=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Train ROC-AUC: {best_metrics['train_metrics']['roc_auc']:.4f}")
    print(f"Validation ROC-AUC: {best_metrics['val_metrics']['roc_auc']:.4f}")
    print(f"Optimal Threshold (validation): {best_metrics['optimal_threshold_info']['optimal_threshold']:.4f}")
    print("=" * 60)
    print("\n[INFO] Test set stored in memory but NOT used for evaluation.")
    print("       Reserved for final evaluation of selected model.")

    return best_model, best_metrics, all_results, preprocessing_time


def save_training_history(
    all_results: List[Dict[str, Any]],
    best_model_name: str,
    dataset_info: Dict[str, Any],
    preprocessing_time: float,
) -> Path:
    """
    Saves complete training history to JSON.
    
    Args:
        all_results: Results from all models
        best_model_name: Name of best model
        dataset_info: Dataset information
        preprocessing_time: Preprocessing time
    
    Returns:
        Path to saved file
    """
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_filename = f"training_history_{timestamp}.json"
    history_path = TRAINING_HISTORY_DIR / history_filename
    
    history = {
        "timestamp": datetime.now().isoformat(),
        "timestamp_formatted": timestamp,
        "best_model": best_model_name,
        "dataset_info": dataset_info,
        "preprocessing_time_seconds": float(preprocessing_time),
        "models": all_results,
    }
    
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining history saved to: {history_path}")
    return history_path


def save_model_and_pipeline(
    model: Any, pipeline: PreprocessingPipeline, model_name: str, metrics: Dict[str, Any]
) -> None:
    """
    Saves model, pipeline and metrics.
    
    Args:
        model: Trained model
        pipeline: Preprocessing pipeline
        model_name: Model name
        metrics: Model metrics
    """
    # Save model
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save pipeline
    pipeline.save(PREPROCESSOR_FILE)
    print(f"Pipeline saved to: {PREPROCESSOR_FILE}")

    # Save readable metrics
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
        f.write("\n[NOTE] Test set stored in memory but NOT used for evaluation.\n")
        f.write("       Reserved for final evaluation of selected model.\n")
    print(f"Metrics saved to: {metrics_path}")
    
    # Save optimal threshold for API
    threshold_path = MODEL_DIR / "optimal_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(metrics["optimal_threshold_info"]["optimal_threshold"]))
    print(f"Optimal threshold saved to: {threshold_path}")


def main():
    """
    Main training function.
    
    Strategy:
    - Use ONLY Modeling dataset
    - Split: 70% train, 15% validation, 15% test
    - Evaluate: Train and validation only
    - Test: Stored in memory, NOT used (reserved for final evaluation)
    """
    print("=" * 60)
    print("Credit Risk Model Training (Optimized English Pipeline)")
    print("=" * 60)
    print("\n[INFO] Strategy: Use only Modeling dataset")
    print("       Split: 70% train, 15% validation, 15% test")
    print("       Evaluation: Train and validation only")
    print("       Test: Stored in memory, NOT used (reserved for final evaluation)")

    # 1. Load data
    print("\n1. Loading Modeling dataset...")
    app_train, _, column_descriptions = get_datasets()
    
    print(f"   Dataset shape: {app_train.shape}")
    print(f"   [INFO] Prediction file NOT used for training/evaluation")
    
    # 2. Separate features and target
    print("\n2. Separating features and target...")
    X_full = app_train.drop(columns=['TARGET_LABEL_BAD=1'])
    y_full = app_train['TARGET_LABEL_BAD=1']
    print(f"   X_full shape: {X_full.shape}")
    print(f"   y_full distribution:\n{y_full.value_counts(normalize=True)}")
    
    # 3. Split into train/val/test
    print("\n3. Splitting dataset into train/validation/test (70/15/15)...")
    from sklearn.model_selection import train_test_split
    
    # First split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full, y_full, test_size=0.3, random_state=RANDOM_STATE, shuffle=True, stratify=y_full
    )
    
    # Then split temp into val (15%) and test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, shuffle=True, stratify=y_temp
    )
    
    print(f"   Train: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X_full)*100:.1f}%)")
    print(f"   Validation: {X_val.shape[0]:,} samples ({X_val.shape[0]/len(X_full)*100:.1f}%)")
    print(f"   Test: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X_full)*100:.1f}%) - STORED, NOT USED")
    print(f"   Total: {len(X_full):,} samples")

    # 4. Dataset info for history
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

    # 5. Create pipeline
    print("\n5. Creating preprocessing pipeline...")
    pipeline = PreprocessingPipeline(low_cardinality_threshold=20)

    # 6. Train models
    print("\n6. Training models...")
    best_model, best_metrics, all_results, preprocessing_time = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test, pipeline
    )

    # 7. Save history
    print("\n7. Saving training history...")
    history_path = save_training_history(
        all_results, best_metrics["model_name"], dataset_info, preprocessing_time
    )

    # 8. Save best model
    print("\n8. Saving best model and pipeline...")
    save_model_and_pipeline(best_model, pipeline, best_metrics["model_name"], best_metrics)

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best model: {best_metrics['model_name']}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
