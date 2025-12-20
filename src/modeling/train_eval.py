from __future__ import annotations

import json
import math
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
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

# -----------------------------
# Types / configs
# -----------------------------

DEFAULT_SCORING = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",   # PR-AUC
}

@dataclass(frozen=True)
class TrainArtifacts:
    model_name: str
    created_at_utc: str
    sklearn_version: str
    threshold: float
    cv_metric_path: Optional[str] = None
    val_metrics_path: Optional[str] = None
    model_path: Optional[str] = None

# -----------------------------
# Helpers: file IO
# -----------------------------

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_json(obj: Dict[str,Any], path: str | Path) -> str:
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

    Variables:
    - **y_true**: ground truth labels (0/1)
    - **y_score**: predicted probability for class 1 (bad = 1)
    - **threshold**: cutoff so pred = 1 if score >= threshold

    objective:
      - "f1": pick threshold maximizing F1
      - "precision_at_recall": maximize precision s.t. recall >= min_recall
      - "recall_at_precision": maximize recall s.t. precision >= min_precision
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asanyarray(y_score).astype(float)

    # Need both classes present; otherwise threshold is meaningless
    if len(np.unique(y_true)) < 2:
        return float(default)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns thresholds of length n-1
    # We'll align by using thresholds indices.
    if thresholds.size == 0:
        return float(default)
    
    # Safe: precissions/recalls length = threshold
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

        # Convert idx back to original indices
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
        save_path: Optional[str] = "artifacts/metrics_cv.json",
) -> Dict[str, Dict[str, float]]:
    """
    Compare models fairly via stratified CV and return aggregate metrics.

    Returns a dict: model_name -> metrics summary.
    Optionally saves it to artifacts/metrics_cv.json.
    """    
    scoring = scoring or DEFAULT_SCORING
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    results: Dict[str, Dict[str, float]] = {}
    
    for name, pipe in podium.items():
        out = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )

        # Convert keys like test_roc_auc, train_pr_auc, etc.
        summary = {}
        for k, arr in out.items():
            if not isinstance(arr, np.ndarray):
                continue
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
        secondary: str = "test_roc_auc_mean"
) -> str:
    """
    Pick the best model name from CV metrics.
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
# Part B: Fit + eval holdout + save bundle
# -----------------------------

def _get_proba(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return probability of positive class (1).
    Assumes classifier supports predict_proba.
    """
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        return np.asarray(proba)[:, 1]
    raise TypeError("Model does not support predict_proba().")

def evaluate_on_val(
        pipe: Pipeline,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate ROC-AUC and PR-AUC on validation set using predicted probabilities.
    """
    y_val_arr = np.asarray(y_val).astype(int)
    y_score = _get_proba(pipe, X_val)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val_arr, y_score)),
        "pr_auc": float(average_precision_score(y_val_arr, y_score)),
    }
    return metrics

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
) -> Tuple[TrainArtifacts, Dict[str, Any]]:
    """
    Fit the chosen model, optionally tune a threshold on val, and save everything.

    Saves:
      - artifacts/model.joblib  (dict with pipeline + threshold + metadata)
      - artifacts/val_metrics.json (if val provided)

    Returns:
      (artifacts_metadata, metrics_dict)
    """
    if model_name not in podium:
        raise KeyError(f"Unknown model_name={model_name}.Available: {list(podium.key())}")

    _ensure_dir(artifacts_dir)

    pipe = podium[model_name]
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

        metrics = evaluate_on_val(pipe, X_val, y_val)
        metrics.update(
            {
                "threshold": float(threshold),
                "threshold_objective": threshold_objective,
                "min_precision": None if min_precision is None else float(min_precision),
                "min_recall": None if min_recall is None else float(min_recall)
            }
        )

        val_metrics_path = _save_json(metrics, Path(artifacts_dir) / val_metrics_filename)
    else:
        val_metrics_path = None

    created_at = datetime.now(timezone.utc).isoformat()

    bundle = {
        "pipeline": pipe,
        "threshold": float(threshold),
        "model_name": model_name,
        "created_at_utc": created_at,
        "sklean_version": sklearn.__version__,
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
    )

    return artifacts, metrics

# -----------------------------
# Optional: convenience runner (call from notebooks or scripts)
# -----------------------------

def run_full_workflow(
        podium: Dict[str, Pipeline],
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        *,
        X_train: pd.DataFrame,
        y_train:pd.Series | np.ndarray,
        X_val: pd.DataFrame,
        y_val: pd.Series | np.ndarray,
        artifacts_dir: str = "artifacts",
) -> Dict[str, Any]:
    
    """
    1) CV compare podium (save metrics_cv.json)
    2) Pick best by PR-AUC
    3) Fit+save final bundle using holdout val for threshold
    """

    cv_metrics = evaluate_podium_cv(podium, X, y, save_path=str(Path(artifacts_dir) / "metrics_cv.json"))
    best = pick_best_model(cv_metrics)

    artifacts, val_metrics = fit_and_save_final(
        podium,
        best,
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        threshold_objective="f1",
        artifacts_dir=artifacts_dir,
    )

    return {
        "best_model": best,
        "cv_metrics": cv_metrics,
        "artifacts": asdict(artifacts),
        "val_metrics": val_metrics,
    }


def save_metrics(metrics: dict, path="artifacts/metrics.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")