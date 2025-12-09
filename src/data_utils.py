from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

def _deduplicate(names: List[str]) -> List[str]:
    seen = {}
    unique: List[str] = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            unique.append(name)
        else:
            seen[name] +=1
            unique.append(f"{name}_{seen[name]}")
    return unique    

def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    column_descriptions = pd.read_excel(config.DATASET_DESCRIPTION)
    col_names = _deduplicate(column_descriptions["Var_Title"].tolist())

    app_train = pd.read_csv(
        config.DATASET_TRAIN,
        sep="\t",
        header=None,
        names=col_names,
        na_values=["NULL"],
        encoding="latin1",
        low_memory=False
    )
    app_test = pd.read_csv(
        config.DATASET_TEST,
        sep="\t",
        header=None,
        encoding="latin1",
        low_memory=False
    )

    return app_train, app_test, column_descriptions

def get_feature_target(
        app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    X_train = app_train.drop(columns=['TARGET_LABEL_BAD=1'])
    y_train = app_train['TARGERT_LABEL_BAD=1']
    X_test = app_test.drop(columns=['TARGET_LABEL_BAD=1'])
    y_test = app_test['TARGET_LABEL_BAD=1']

    return X_train, y_train, X_test, y_test

def get_train_val_sets(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train
    )

    return X_train, X_val, y_train, y_val