from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

from src.config import MODEL_DIR, PREPROCESSOR_FILE
from src.data_utils import get_datasets
from src.modeling.pipelines import get_podium, get_tuning_candidates

# -----------------------------
# Types / configs
# -----------------------------

DEFAULT_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}

TRAINING_HISTORY_DIR = MODEL_DIR.parent / "training_history"
RANDOM_STATE = 42


@dataclass(frozen=True)
class TrainArtifacts:
    model_name: str
    created_at_utc: str
    sklearn_version: str
    threshold: float
    cv_metric_path: Optional[str] = None
    val_metrics_path: Optional[str] = None
    model_path: Optional[str] = None
    preprocessor_path: Optional[str] = None


# -----------------------------
# Helpers: file IO
# -----------------------------


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_json(obj: Dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# -----------------------------
# Helpers: threshold selection
# -----------------------------


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    objective: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    default: float = 0.5,
) -> float:
    """
    Choose a decision threshold using the Precision-Recall curve.

    objective:
      - "f1": pick threshold maximizing F1
      - "precision_at_recall": maximize precision s.t. recall >= min_recall
      - "recall_at_precision": maximize recall s.t. precision >= min_precision
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asanyarray(y_score).astype(float)

    if len(np.unique(y_true)) < 2:
        return float(default)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        return float(default)

    p = precisions[:-1]
    r = recalls[:-1]
    t = thresholds

    if objective == "f1":
        f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
        idx = int(np.nanargmax(f1))
        return float(t[idx])

    if objective == "precision_at_recall":
        if min_recall is None:
            raise ValueError("min_recall is required for precision_at_recall")
        mask = r >= min_recall
        if not np.any(mask):
            return float(default)
        idx = int(np.nanargmax(p[mask]))
        return float(t[np.where(mask)[0][idx]])

    if objective == "recall_at_precision":
        if min_precision is None:
            raise ValueError("min_precision is required for recall_at_precision")
        mask = p >= min_precision
        if not np.any(mask):
            return float(default)
        idx = int(np.nanargmax(r[mask]))
        return float(t[np.where(mask)[0][idx]])

    raise ValueError(f"Unknown objective: {objective}")


# -----------------------------
# Part A: CV comparison
# -----------------------------


def evaluate_podium_cv(
    podium: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
    scoring: Optional[Dict[str, str]] = None,
    n_jobs: int = -1,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare models via stratified CV and return aggregate metrics.
    """
    scoring = scoring or DEFAULT_SCORING
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    results: Dict[str, Dict[str, float]] = {}

    for name, pipe in podium.items():
        scores = {
            "test_roc_auc": [],
            "test_pr_auc": [],
            "train_roc_auc": [],
            "train_pr_auc": [],
        }

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            fold_pipe = clone(pipe)

            if _uses_sample_weight(name):
                sw = compute_sample_weight("balanced", np.asarray(y_train).astype(int))
                fold_pipe.fit(X_train, y_train, model__sample_weight=sw)
            else:
                fold_pipe.fit(X_train, y_train)

            y_train_score = _get_proba(fold_pipe, X_train)
            y_test_score = _get_proba(fold_pipe, X_test)

            scores["train_roc_auc"].append(roc_auc_score(y_train, y_train_score))
            scores["train_pr_auc"].append(average_precision_score(y_train, y_train_score))
            scores["test_roc_auc"].append(roc_auc_score(y_test, y_test_score))
            scores["test_pr_auc"].append(average_precision_score(y_test, y_test_score))

        summary = {}
        for k, arr in scores.items():
            arr = np.asarray(arr, dtype=float)
            summary[f"{k}_mean"] = float(np.mean(arr))
            summary[f"{k}_std"] = float(np.std(arr))

        results[name] = summary

    if save_path:
        _save_json(results, save_path)

    return results


def pick_best_model(
    cv_metrics: Dict[str, Dict[str, float]],
    *,
    primary: str = "test_pr_auc_mean",
    secondary: str = "test_roc_auc_mean",
) -> str:
    """
    Pick best model name from CV metrics.
    Defaults: maximize PR-AUC, tie-break with ROC-AUC.
    """
    best_name = None
    best_tuple = None

    for name, m in cv_metrics.items():
        a = m.get(primary, float("-inf"))
        b = m.get(secondary, float("-inf"))
        tpl = (a, b)
        if best_tuple is None or tpl > best_tuple:
            best_tuple = tpl
            best_name = name

    if best_name is None:
        raise ValueError("Could not pick best model; cv_metrics is empty?")
    return best_name


# -----------------------------
# Part B: Train/val metrics + save bundle
# -----------------------------


def _get_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        return np.asarray(proba)[:, 1]
    raise TypeError("Model does not support predict_proba().")

def _uses_sample_weight(model_name: str) -> bool:
    return "sample_weight" in model_name


def _metrics_block(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }

def _prediction_stats(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "pred_counts": {
            "0": int((y_pred == 0).sum()),
            "1": int((y_pred == 1).sum()),
        },
        "pred_rate": float(np.mean(y_pred)),
        "actual_rate": float(np.mean(y_true)),
    }

def evaluate_train_val(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    *,
    threshold_objective: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    default_threshold: float = 0.5,
) -> Dict[str, Any]:
    y_train_arr = np.asarray(y_train).astype(int)
    y_val_arr = np.asarray(y_val).astype(int)

    train_score = _get_proba(pipe, X_train)
    val_score = _get_proba(pipe, X_val)

    train_metrics = _metrics_block(y_train_arr, train_score, threshold=default_threshold)
    val_metrics = _metrics_block(y_val_arr, val_score, threshold=default_threshold)

    optimal_threshold = choose_threshold(
        y_true=y_val_arr,
        y_score=val_score,
        objective=threshold_objective,
        min_precision=min_precision,
        min_recall=min_recall,
        default=default_threshold,
    )

    val_metrics_opt = _metrics_block(y_val_arr, val_score, threshold=optimal_threshold)

    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "optimal_threshold_info": {
            "optimal_threshold": float(optimal_threshold),
            "metrics_at_optimal": val_metrics_opt,
            "objective": threshold_objective,
            "min_precision": None if min_precision is None else float(min_precision),
            "min_recall": None if min_recall is None else float(min_recall),
        },
        "prediction_stats": {
            "train": {
                "threshold_0_5": _prediction_stats(y_train_arr, train_score, default_threshold),
                "threshold_opt": _prediction_stats(y_train_arr, train_score, optimal_threshold),
            },
            "val": {
                "threshold_0_5": _prediction_stats(y_val_arr, val_score, default_threshold),
                "threshold_opt": _prediction_stats(y_val_arr, val_score, optimal_threshold),
            },
        },
    }


def train_and_evaluate_models(
    podium: Dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    *,
    threshold_objective: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    default_threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Pipeline]]:
    all_results = []
    best_name = None
    best_score = float("-inf")
    best_pipe = None

    sample_weights = compute_sample_weight("balanced", np.asarray(y_train).astype(int))

    for name, pipe in podium.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        try:
            start = time.time()
            if _uses_sample_weight(name):
                pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
            else:
                pipe.fit(X_train, y_train)
            train_time = time.time() - start

            metrics = evaluate_train_val(
                pipe,
                X_train,
                y_train,
                X_val,
                y_val,
                threshold_objective=threshold_objective,
                min_precision=min_precision,
                min_recall=min_recall,
                default_threshold=default_threshold,
            )

            model_params = pipe.named_steps["model"].get_params()
            metrics["training_time_seconds"] = float(train_time)
            metrics["model_name"] = name
            metrics["hyperparameters"] = model_params

            all_results.append(metrics)

            val_pr_auc = metrics["val_metrics"]["pr_auc"]
            if val_pr_auc > best_score:
                best_score = val_pr_auc
                best_name = name
                best_pipe = pipe

            print(f"  Train ROC-AUC: {metrics['train_metrics']['roc_auc']:.4f}")
            print(f"  Train F1: {metrics['train_metrics']['f1']:.4f}")
            print(f"  Val ROC-AUC: {metrics['val_metrics']['roc_auc']:.4f}")
            print(f"  Val PR-AUC: {metrics['val_metrics']['pr_auc']:.4f}")
            print(f"  Val F1 (0.5): {metrics['val_metrics']['f1']:.4f}")
            print(
                f"  Optimal Threshold (val): "
                f"{metrics['optimal_threshold_info']['optimal_threshold']:.4f}"
            )
            print(
                "  Val F1 (optimal): "
                f"{metrics['optimal_threshold_info']['metrics_at_optimal']['f1']:.4f}"
            )

        except Exception as exc:
            print(f"  [ERROR] Error training {name}: {exc}")
            import traceback
            traceback.print_exc()
            continue

    if best_name is None:
        raise ValueError("No model was successfully trained")

    best_metrics = next(m for m in all_results if m["model_name"] == best_name)
    return best_metrics, all_results, best_pipe


def evaluate_on_test(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    *,
    default_threshold: float,
    optimal_threshold: float,
) -> Dict[str, Any]:
    y_test_arr = np.asarray(y_test).astype(int)
    y_test_score = _get_proba(pipe, X_test)
    return {
        "metrics": _metrics_block(y_test_arr, y_test_score, threshold=default_threshold),
        "prediction_stats": {
            "threshold_0_5": _prediction_stats(y_test_arr, y_test_score, default_threshold),
            "threshold_opt": _prediction_stats(y_test_arr, y_test_score, optimal_threshold),
        },
    }


def save_training_history(
    all_results: list[Dict[str, Any]],
    best_model_name: str,
    dataset_info: Dict[str, Any],
    cv_metrics: Dict[str, Dict[str, float]],
    *,
    artifacts: Optional[Dict[str, Any]] = None,
    test_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    TRAINING_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = TRAINING_HISTORY_DIR / f"training_history_{timestamp}.json"

    history = {
        "timestamp": datetime.now().isoformat(),
        "timestamp_formatted": timestamp,
        "best_model": best_model_name,
        "dataset_info": dataset_info,
        "cv_metrics": cv_metrics,
        "models": all_results,
        "artifacts": artifacts or {},
        "test_metrics": test_summary or {},
    }

    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nTraining history saved to: {history_path}")
    return history_path


def save_prediction_stats(
    all_results: list[Dict[str, Any]],
    *,
    model_dir: Path,
    test_summary: Optional[Dict[str, Any]] = None,
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_path = model_dir / f"prediction_stats_{timestamp}.json"
    payload = {
        "models": [
            {
                "model_name": r["model_name"],
                "prediction_stats": r.get("prediction_stats", {}),
            }
            for r in all_results
        ],
        "test_metrics": test_summary or {},
    }
    stats_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPrediction stats saved to: {stats_path}")
    return stats_path


def save_model_and_preprocessor(
    model_bundle: Dict[str, Any],
    *,
    metrics: Dict[str, Any],
    model_name: str,
    model_dir: Path,
    preprocessor_path: str,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"\nModel saved to: {model_path}")

    pipeline = model_bundle["pipeline"]
    if "prep" in pipeline.named_steps:
        joblib.dump(pipeline.named_steps["prep"], preprocessor_path)
        print(f"Preprocessor saved to: {preprocessor_path}")

    metrics_path = model_dir / "metrics.txt"
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
        f.write(
            f"\nOptimal Threshold (calculated on validation): "
            f"{metrics['optimal_threshold_info']['optimal_threshold']:.4f}\n"
        )
    print(f"Metrics saved to: {metrics_path}")

    threshold_path = model_dir / "optimal_threshold.txt"
    with open(threshold_path, "w", encoding="utf-8") as f:
        f.write(str(metrics["optimal_threshold_info"]["optimal_threshold"]))
    print(f"Optimal threshold saved to: {threshold_path}")


def fit_and_save_final(
    podium: Dict[str, Pipeline],
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    *,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series | np.ndarray] = None,
    threshold_objective: str = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    default_threshold: float = 0.5,
    artifacts_dir: str = "artifacts",
    model_filename: str = "model.joblib",
    val_metrics_filename: str = "val_metrics.json",
) -> Tuple[TrainArtifacts, Dict[str, Any], Dict[str, Any]]:
    if model_name not in podium:
        raise KeyError(f"Unknown model_name={model_name}. Available: {list(podium.keys())}")

    _ensure_dir(artifacts_dir)

    pipe = podium[model_name]
    if _uses_sample_weight(model_name):
        sample_weights = compute_sample_weight("balanced", np.asarray(y_train).astype(int))
        pipe.fit(X_train, y_train, model__sample_weight=sample_weights)
    else:
        pipe.fit(X_train, y_train)

    threshold = float(default_threshold)
    metrics: Dict[str, Any] = {}

    if X_val is not None and y_val is not None:
        y_val_arr = np.asarray(y_val).astype(int)
        y_score = _get_proba(pipe, X_val)

        threshold = choose_threshold(
            y_true=y_val_arr,
            y_score=y_score,
            objective=threshold_objective,
            min_precision=min_precision,
            min_recall=min_recall,
            default=default_threshold,
        )

        metrics = {
            "roc_auc": float(roc_auc_score(y_val_arr, y_score)),
            "pr_auc": float(average_precision_score(y_val_arr, y_score)),
            "threshold": float(threshold),
            "threshold_objective": threshold_objective,
            "min_precision": None if min_precision is None else float(min_precision),
            "min_recall": None if min_recall is None else float(min_recall),
        }

        val_metrics_path = _save_json(metrics, Path(artifacts_dir) / val_metrics_filename)
    else:
        val_metrics_path = None

    created_at = datetime.now(timezone.utc).isoformat()

    bundle = {
        "pipeline": pipe,
        "threshold": float(threshold),
        "model_name": model_name,
        "created_at_utc": created_at,
        "sklearn_version": sklearn.__version__,
    }

    model_path = str(Path(artifacts_dir) / model_filename)
    joblib.dump(bundle, model_path)

    artifacts = TrainArtifacts(
        model_name=model_name,
        created_at_utc=created_at,
        sklearn_version=sklearn.__version__,
        threshold=float(threshold),
        val_metrics_path=val_metrics_path,
        model_path=model_path,
        preprocessor_path=PREPROCESSOR_FILE,
    )

    return artifacts, metrics, bundle


# -----------------------------
# Optional: tuning helpers
# -----------------------------


def tune_pipeline_random(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    param_distributions: Dict[str, Any],
    *,
    n_iter: int = 25,
    seed: int = 42,
    n_splits: int = 5,
    n_jobs: int = -1,
    refit_metric: str = "pr_auc",
):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def save_search_results(search, out_dir: str, prefix: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / f"{prefix}_best_params.json").write_text(
        json.dumps(search.best_params_, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    df = pd.DataFrame(search.cv_results_)
    if "mean_test_pr_auc" in df.columns:
        df = df.sort_values("mean_test_pr_auc", ascending=False)
    df.to_csv(out / f"{prefix}_cv_results.csv", index=False)

    summary = {
        "best_score_pr_auc": float(search.best_score_),
        "best_params": search.best_params_,
    }
    (out / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# -----------------------------
# Runner
# -----------------------------


def main():
    print("=" * 60)
    print("Credit Risk Model Training (Unified)")
    print("=" * 60)
    print("\n[INFO] Strategy: CV compare + holdout validation threshold")
    print("       Split: 70% train, 15% validation, 15% test")
    print("       Test: reserved in memory (not used for evaluation)")

    print("\n1. Loading modeling dataset...")
    app_train, _, _ = get_datasets()
    print(f"   Dataset shape: {app_train.shape}")

    X_full = app_train.drop(columns=["TARGET_LABEL_BAD=1"])
    y_full = app_train["TARGET_LABEL_BAD=1"]

    print("\n2. Splitting dataset...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_full,
        y_full,
        test_size=0.3,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_full,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y_temp,
    )

    print(f"   Train: {X_train.shape[0]:,} rows")
    print(f"   Validation: {X_val.shape[0]:,} rows")
    print(f"   Test: {X_test.shape[0]:,} rows (reserved)")

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

    print("\n3. Building model podium...")
    podium = get_podium()

    print("\n4. Cross-validation comparison...")
    cv_metrics = evaluate_podium_cv(
        podium,
        X_full,
        y_full,
        save_path=str(Path(MODEL_DIR) / "metrics_cv.json"),
    )
    best_cv_model = pick_best_model(cv_metrics)
    print(f"\nBest by CV (PR-AUC): {best_cv_model}")

    print("\n5. Train/val evaluation for all models...")
    best_metrics, all_results, _ = train_and_evaluate_models(
        podium,
        X_train,
        y_train,
        X_val,
        y_val,
        threshold_objective="f1",
    )
    best_model_name = best_metrics["model_name"]
    print(f"\nBest by holdout PR-AUC: {best_model_name}")

    best_cv_metrics = next(
        (m for m in all_results if m["model_name"] == best_cv_model),
        best_metrics,
    )

    print("\n6. Fitting final model and saving artifacts...")
    artifacts, val_metrics, bundle = fit_and_save_final(
        podium,
        best_cv_model,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        threshold_objective="f1",
        artifacts_dir=str(MODEL_DIR),
    )

    # Informative: evaluate on reserved test set (no model selection)
    best_pipe = bundle["pipeline"]
    best_threshold = float(artifacts.threshold)
    test_summary = evaluate_on_test(
        best_pipe,
        X_test,
        y_test,
        default_threshold=0.5,
        optimal_threshold=best_threshold,
    )
    _save_json(test_summary, Path(MODEL_DIR) / "test_metrics.json")
    print("\n[INFO] Reserved test set evaluation saved (informational only).")

    save_model_and_preprocessor(
        bundle,
        metrics=best_cv_metrics,
        model_name=best_cv_model,
        model_dir=MODEL_DIR,
        preprocessor_path=PREPROCESSOR_FILE,
    )

    stats_path = save_prediction_stats(all_results, model_dir=MODEL_DIR, test_summary=test_summary)
    history_path = save_training_history(
        all_results,
        best_cv_model,
        dataset_info,
        cv_metrics,
        artifacts=asdict(artifacts),
        test_summary=test_summary,
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best model (CV): {best_cv_model}")
    print(f"Prediction stats: {stats_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
