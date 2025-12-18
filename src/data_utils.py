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
    """
    Carga los datasets. Si DATASET_TEST no existe, retorna None para test.
    """
    import os
    
    # Intentar leer con xlrd primero (para .XLS), luego openpyxl si es necesario
    try:
        # Para archivos .XLS antiguos, usar xlrd
        column_descriptions = pd.read_excel(config.DATASET_DESCRIPTION, engine='xlrd')
    except:
        try:
            # Para archivos .XLSX modernos, usar openpyxl
            column_descriptions = pd.read_excel(config.DATASET_DESCRIPTION, engine='openpyxl')
        except:
            # Si ambos fallan, intentar sin especificar engine
            column_descriptions = pd.read_excel(config.DATASET_DESCRIPTION)
    
    # Obtener los VALORES de la columna Var_Title (usar .values para evitar problemas con índices)
    # Luego convertir a lista para asegurar que obtenemos los valores reales, no el índice
    col_names_raw = column_descriptions["Var_Title"].values.tolist()
    
    # CORRECCIÓN: La columna 43 tiene EDUCATION_LEVEL pero debería ser MATE_EDUCATION_LEVEL
    # (la columna 9 es EDUCATION_LEVEL del solicitante, la 43 es del cónyuge)
    if len(col_names_raw) > 43 and col_names_raw[43] == "EDUCATION_LEVEL":
        col_names_raw[43] = "MATE_EDUCATION_LEVEL"
    
    col_names = _deduplicate(col_names_raw)

    app_train = pd.read_csv(
        config.DATASET_TRAIN,
        sep="\t",
        header=None,
        names=col_names,
        na_values=["NULL"],
        encoding="latin1",
        low_memory=False
    )
    
    # Si el archivo de test no existe, retornar None
    app_test = None
    if os.path.exists(config.DATASET_TEST):
        app_test = pd.read_csv(
            config.DATASET_TEST,
            sep="\t",
            header=None,
            names=col_names,
            na_values=["NULL"],
            encoding="latin1",
            low_memory=False
        )

    return app_train, app_test, column_descriptions

def get_feature_target(
        app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    X_train = app_train.drop(columns=['TARGET_LABEL_BAD=1'])
    y_train = app_train['TARGET_LABEL_BAD=1']
    X_test = app_test.drop(columns=['TARGET_LABEL_BAD=1'])
    y_test = app_test['TARGET_LABEL_BAD=1']

    return X_train, y_train, X_test, y_test

def get_train_val_sets(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train
    )

    return X_train, X_val, y_train, y_val