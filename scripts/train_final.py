from sklearn.model_selection import train_test_split

from src.data_utils import get_datasets
from src.modeling.pipelines import get_podium
from src.modeling.train_eval import evaluate_podium_cv, pick_best_model, fit_and_save_final

def main():
    print("[train_final] starting...")

    train_df, _, _ = get_datasets()
    print("[train_final] dataset loaded:", train_df.shape)

    y = train_df["TARGET_LABEL_BAD=1"]
    X = train_df.drop(columns=["TARGET_LABEL_BAD=1"])
    print("[train_final] X, y ready:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("[train_final] split done:", X_train.shape, X_val.shape, y_train.mean(), y_val.mean())

    podium = get_podium()

    cv_metrics = evaluate_podium_cv(podium, X, y, save_path="artifacts/metrics_cv.json")
    best = pick_best_model(cv_metrics)
    print("[train_final] best model:", best)

    artifacts, val_metrics = fit_and_save_final(
        podium=podium,
        model_name=best,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        threshold_objective="f1",
        artifacts_dir="artifacts",
    )

    print("Saved model to:", artifacts.model_path)
    print("Validation metrics:", val_metrics)

if __name__ == "__main__":
    main()