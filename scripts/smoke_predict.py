import joblib
import pandas as pd
from src.data_utils import get_datasets

def main():
    # 1) Charge trained model
    bundle = joblib.load("models/production/model.joblib")
    pipe = bundle["pipeline"]
    thr = float(bundle.get("threshold", 0.5))

    # 2) Take a real train line as an example
    train_df, _, _ = get_datasets()
    X = train_df.drop(columns=["TARGET_LABEL_BAD=1"])
    row = X.iloc[0].to_dict()

    # 3) Predict
    X_one = pd.DataFrame([row])
    print("X_one columns (first 20):", list(X_one.columns)[:20])
    proba = float(pipe.predict_proba(X_one)[:, 1][0])
    pred = int(proba >= thr)

    prep = bundle["pipeline"].named_steps["prep"]
    print("Prep steps:", prep.named_steps.keys())

    print("prob_bad:", proba)
    print("pred_bad:", pred)
    print("threshold:", thr)
    print("model_name:", bundle.get("model_name"))

if __name__ == "__main__":
    main()