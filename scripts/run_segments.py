from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_DIR, MODEL_FILE
from src.data_utils import get_datasets
from src.modeling.payer_segments import (
    build_top_payers_scenarios,
    save_risk_by_age_plot,
    save_risk_by_tenure_plot,
    save_top_segments_rate_plot,
    save_top_segments_amount_plot,
)


def main():
    # Load training data (modeling dataset)
    train_df, _, _ = get_datasets()
    y = train_df["TARGET_LABEL_BAD=1"]
    X = train_df.drop(columns=["TARGET_LABEL_BAD=1"])

    # Load model (bundle or estimator)
    loaded = joblib.load(MODEL_FILE)
    if isinstance(loaded, dict) and "pipeline" in loaded:
        model = loaded["pipeline"]
    else:
        model = loaded

    pred_proba = model.predict_proba(X)[:, 1]

    out_dir = Path(MODEL_DIR) / "segments"
    plots_dir = out_dir / "plots"
    build_top_payers_scenarios(
        X,
        pred_proba,
        top_pcts=[5, 10, 20],
        amounts=[1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000],
        rates=[0.05, 0.10, 0.15, 0.20, 0.30],
        out_dir=out_dir,
    )
    save_risk_by_age_plot(X, pred_proba, out_dir=plots_dir)
    save_risk_by_tenure_plot(X, pred_proba, out_dir=plots_dir)
    save_top_segments_rate_plot(
        X,
        pred_proba,
        top_pcts=[5, 10, 20],
        amount=1000,
        rates=[0.05, 0.10, 0.15, 0.20, 0.30],
        out_dir=plots_dir,
    )
    save_top_segments_amount_plot(
        X,
        pred_proba,
        top_pcts=[5, 10, 20],
        amounts=[1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000],
        rate=0.10,
        out_dir=plots_dir,
    )

    print(f"[OK] Segments saved to: {out_dir}")


if __name__ == "__main__":
    main()
