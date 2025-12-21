import os
from pathlib import Path

from src.data_utils import get_datasets
from src.modeling.pipelines import get_podium
from src.modeling.train_eval import evaluate_podium_cv, pick_best_model

def main():
    model_version = os.getenv("MODEL_VERSION", "production")
    out_dir = Path("models") / model_version
    out_dir.mkdir(parents=None, exist_ok=True)

    train_df, _, _ = get_datasets()
    y = train_df["TARGET_LABEL_BAD=1"]
    X = train_df.drop(columns=["TARGET_LABEL_BAD=1"])

    podium = get_podium()
    cv_metrics = evaluate_podium_cv(podium, X, y, save_path=str(out_dir / "metrics_cv.json"))

    best = pick_best_model(cv_metrics)
    print("Best model by PR-AUC", best)


if __name__ == "__main__":
    main()