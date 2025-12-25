from sklearn.model_selection import train_test_split

from src.data_utils import get_datasets
from src.modeling.pipelines import get_tuning_candidates
from src.modeling.train_eval import (
    tune_pipeline_random,
    save_search_results,
    fit_and_save_final
)

import os
import json
from pathlib import Path

model_version = os.getenv("MODEL_VERSION", "production")
out_dir = f"models/{model_version}"

def main():
    tuning_dir = out_dir / "tunig"
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_dir.mkdir(parents=True, exist_ok=True)

    print("[train_final] starting...")

    train_df, _, _ = get_datasets()
    print("[train_final] dataset loaded:", train_df.shape)

    y = train_df["TARGET_LABEL_BAD=1"].astype(int)
    X = train_df.drop(columns=["TARGET_LABEL_BAD=1"])
    print("[train_final] X, y ready:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[train_final] split done:", X_train.shape, X_val.shape, y_train.mean(), y_val.mean())

    candidates = get_tuning_candidates()

    searches = {}
    leaderboard = {}

    # 1) Tune per model
    for name, cfg in candidates.items():
        print(f"\n[tuning] {name} ...")
        search = tune_pipeline_random(
            cfg["pipe"],
            X_train,
            y_train,
            cfg["params"],
            n_iter = cfg.get("n_iter", 25),
            n_splits=5,
            n_jobs=-1,
        )
        save_search_results(search, str(tuning_dir), prefix=name)
        searches[name] = search
        leaderboard[name] = {
            "best_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
        }

    (tuning_dir / "leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 2) Compare winners
    best_name = max(searches.keys(), key=lambda k: searches[k].best_score_)
    best_pipe = searches[best_name].best_estimator_
    print("\n[train_final] BEST tuned model:", best_name, "PR-AUC:", searches[best_name].best_score_)

    # 3) Fit final + threshold on validation and save model bundle
    tuned_podium = {best_name: best_pipe}
    artifacts, val_metrics = fit_and_save_final(
        podium=tuned_podium,
        model_name=best_name,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        threshold_objective="f1",
        artifacts_dir=str(out_dir),
        model_filename="model.joblib",
        val_metrics_filename="val_metrics.json",
    )

    print("Saved model to:", artifacts.model_path)
    print("Validation metrics:", val_metrics)

if __name__ == "__main__":
    main()