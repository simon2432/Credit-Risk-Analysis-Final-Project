# Project Improvements and Analysis Documentation

## 1. Optimization Summary
This document details the changes made to standardize the "Credit Risk Analysis" project to English and expand its predictive capabilities. Improved versions of original files have been created with the `_ok` suffix to preserve previous work.

### Improved Files
-   **`src/preprocessing_ok.py`**: Complete pipeline fully documented in English.
-   **`src/models_config_ok.py`**: New configuration including Logistic Regression, Random Forest, and CatBoost.
-   **`src/train_model_ok.py`**: Training script with English logs and multi-model support.

---

## 2. Detailed Analysis by Stage

### Phase 1: Exploratory Data Analysis (EDA)
Critical patterns identified guiding the preprocessing:
-   **Class Imbalance**: The target (`TARGET_LABEL_BAD=1`) has a ratio of ~1:3.
    -   *Solution*: Supported `class_weight='balanced'` and `compute_sample_weight` in training.
-   **Constant Variables**: Columns like `CLERK_TYPE`, `QUANT_ADDITIONAL_CARDS` have zero variance.
    -   *Action*: Automatic elimination in pipeline step 1 to reduce noise.
-   **Outliers**: Present in Income (`PERSONAL_MONTHLY_INCOME`) and Age.
    -   *Strategy*: Although tree models are robust, capping (1%-99%) is recommended if linear models are used. Current code supports this.
-   **Missing Values**:
    -   Very high in `PROFESSION_CODE` and `MATE_PROFESSION_CODE`.
    -   *Insight*: Missing value is informative (possible unemployment or strict informality).
    -   *Action*: Created binary flags `MISSING_*` before imputation.

### Phase 2: Preprocessing Pipeline (`preprocessing_ok.py`)
The pipeline follows a robust 6-step sequential logic:
1.  **Initial Cleaning**: Normalizes Y/N columns and removes constants.
2.  **Outlier Handling**: (Configurable/Skipped for trees)
3.  **Feature Engineering**:
    -   *Financial Ratios*: `INCOME_RATIO` (Other Income / Personal Income).
    -   *Stability*: `YEARS_IN_JOB`, `YEARS_IN_RESIDENCE`.
    -   *Interactions*: `SAME_STATE_RES_PROF` (Geographic consistency).
4.  **Imputation**:
    -   Median for numericals.
    -   Mode for categoricals.
    -   **Exception**: High cardinality columns preserve NaN for encoding.
5.  **Encoding**:
    -   *Ordinal*: For binary and Y/N.
    -   *Frequency Encoding*: For columns with >100 categories (e.g., Zip codes).
    -   *One-Hot Encoding*: For low/medium cardinality (grouping rare ones as "OTHER").
6.  **Scaling**: `MinMaxScaler` (0-1) to aid convergence of Logistic Regression and Neural Networks.

### Phase 3: Modeling (`models_config_ok.py`)
Model repertoire expanded to cover different mathematical approaches:

| Model | Type | Reason for Inclusion |
| :--- | :--- | :--- |
| **Logistic Regression** | Linear | **Baseline**. Interpretable and fast. Fundamental for understanding linear relationships. |
| **Random Forest** | Bagging | **Stability**. Reduces variance and is less prone to overfitting than boosting. |
| **XGBoost / LightGBM** | Boosting | **Power**. SOTA for tabular data. |
| **CatBoost** | Boosting | **Categoricals**. Handles categorical variables natively better than OHE in many cases. |

---

## 3. Execution and Environment Guide

### Environment Configuration (Virtual Env)
To work isolated and not affect your global system:

1.  **Create the environment** (if not exists):
    ```bash
    python -m venv venv
    ```
2.  **Activate the environment**:
    *   Windows: `.\venv\Scripts\Activate`
    *   Mac/Linux: `source venv/bin/activate`
    *   *You will see `(venv)` at the start of your command line.*
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you have `numpy` error, run `pip install --upgrade numpy pandas scikit-learn`*

### Improved Training Execution
With the environment activated, run:
```bash
python -m src.train_model_ok
```
This will train all configured models and save results to the `models/` folder.

### Version Control (.gitignore)
To push your changes to GitHub without pushing garbage files or the virtual environment:

1.  Ensure your `.gitignore` file contains these lines (your current file already has them ✅):
    ```gitignore
    # Virtual Environment
    venv/
    env/
    .venv/
    
    # Models & Data (Generally not uploaded if heavy)
    models/*
    !models/.gitkeep
    data/raw/*
    !data/raw/.gitkeep
    ```
2.  **Git Flow**:
    ```bash
    git status          # Check changes
    git add .           # Add all (gitignore will prevent adding venv/)
    git commit -m "feat: optimize pipeline and add english docs"
    git push origin <your-branch>
    ```
