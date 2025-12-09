from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

TARGET_COL = "TARGET_LABEL_BAD_1"
ID_COL = "ID_CLIENT"

def preprocess_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, low_cardinality_threshold: int = 20,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling:
        - Remove ``ID_CLIENT``
        - Detect categorical variables such as ``object`` columns or numeric variables with few categories (<= low_cardinality_threshold). 
        - Binary categorical variables -> OrdinalEncoder
            - Categories with more categories:
        * Low cardinality -> One-Hot 
        * High cardinality -> OrdinalEncode (to avoid column explosion)
        - Imputes median to everything and scales with MinMaxScaler
        - Imputes missing values with the nearest neighbor
    """

    print("Input train data shape: ", train_df)
    print("Input validation data shape: ", val_df)
    print("Input test data shape: ", test_df)

    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    for df in (working_train_df, working_val_df, working_test_df):
        if ID_COL in df.columns:
            df.drop(columns=[ID_COL], inplace=True)
    
    # Detect categoricals
    cat_columns: List[str] = working_train_df.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns: list[str] = working_train_df.select_dtypes(include=["number"]).columns.tolist()

    num_low_card = [
        col for col in numeric_columns
        if working_train_df[col].nunique(dropna=True) <= low_cardinality_threshold
    ]
    cat_columns = list(dict.fromkeys(cat_columns + num_low_card)) # remove duplicates
    numeric_columns = [col for col in numeric_columns if col not in num_low_card]

    # Separate binary vs multi-category
    binary_cat_columns = [
        col for col in cat_columns if working_train_df[col].nunique(dropna=True) == 2
    ]
    multi_cat_columns = [col for col in cat_columns if col not in binary_cat_columns]

    # Impute missing values for categorical columns with mode (using only training data)
    for col in cat_columns:
        mode = working_train_df[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Missing"
        for df in (working_train_df, working_val_df, working_test_df):
            df[col] = df[col].fillna(fill_value)
    
    # Binaries -> OrdinalEncoder (0/1)
    if binary_cat_columns:
        binary_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", 
            unknown_value=-1,
        )
        binary_encoder.fit(working_train_df[binary_cat_columns])
        for df in (working_train_df, working_val_df, working_test_df):
            df[binary_cat_columns] = binary_encoder.transform(df[binary_cat_columns])

    # Multi-category -> One-Hot; High -> Ordinal
    ohe_cat_columns = [
        col for col in multi_cat_columns
        if working_train_df[col].nunique(dropna=True) <= low_cardinality_threshold
    ]
    ordinal_cat_columns = [col for col in multi_cat_columns if col not in ohe_cat_columns]

    if ohe_cat_columns:
        ohe_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        ohe_encoder.fit(working_train_df[ohe_cat_columns])

        def _ohe(df: pd.DataFrame) -> pd.DataFrame:
            ohe_array = ohe_encoder.transform(df[ohe_cat_columns])
            cols = ohe_encoder.get_feature_names_out(ohe_cat_columns)
            return pd.DataFrame(ohe_array, columns=cols, index=df.index)
        
        train_ohe = _ohe(working_train_df)
        val_ohe = _ohe(working_val_df)
        test_ohe = _ohe(working_test_df)

        working_train_df.drop(columns=ohe_cat_columns, inplace=True)
        working_val_df.drop(columns=ohe_cat_columns, inplace=True)
        working_test_df.drop(columns=ohe_cat_columns, inplace=True)

        working_train_df = pd.concat([working_train_df, train_ohe], axis=1)
        working_val_df = pd.concat([working_val_df, val_ohe], axis=1)
        working_test_df = pd.concat([working_test_df, test_ohe], axis=1)

        if ordinal_cat_columns:
            multi_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            multi_encoder.fit(working_train_df[ordinal_cat_columns])
            for df in (working_train_df, working_val_df, working_test_df):
                df[ordinal_cat_columns] = multi_encoder.transform(df[ordinal_cat_columns])

    # Impute numeric columns with median for all datasets and scale with MinMaxScaler
    imputer = SimpleImputer(strategy="median")
    imputer.fit(working_train_df[numeric_columns])

    train_imputed = pd.DataFrame(
        imputer.transform(working_train_df),
        columns=working_train_df.columns,
        index=working_train_df.index,
    ) 
    val_imputed = pd.DataFrame(
        imputer.transform(working_val_df),
        columns=working_val_df.columns,
        index=working_val_df.index,
    )
    test_imputed = pd.DataFrame(
        imputer.transform(working_test_df),
        columns=working_test_df.columns,
        index=working_test_df.index,
    )

    scaler = MinMaxScaler()
    scaler.fit(train_imputed)

    train = scaler.transform(train_imputed)
    val = scaler.transform(val_imputed)
    test = scaler.transform(test_imputed)

    return train, val, test
    

