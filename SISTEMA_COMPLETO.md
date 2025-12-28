# Complete Credit Risk Analysis System

## Executive Summary

This system evaluates customer credit risk using Machine Learning. The complete flow is:

1. **Training**: The dataset is processed, models are trained, and the best one is saved.
2. **Prediction**: User completes a form in the UI → API processes the data → Model predicts → Result is displayed.

---

## Complete System Flow

### 1. **Training Phase** (`src/modeling/train_eval.py`)

```
Original Dataset (50,000 rows × 53 columns)
    ↓
Split into Train/Validation/Test (70% / 15% / 15%)
    ↓
Cross-Validation (5 folds) to compare models
    ↓
Training of multiple models:
  - Logistic Regression (balanced)
  - HistGradientBoostingClassifier (balanced)
  - Gradient Boosting (sample_weight)
  - XGBoost (sample_weight)
  - LightGBM (sample_weight)
  - CatBoost (scale_pos_weight)
    ↓
Selection of best model by PR-AUC in CV
    ↓
Final training of best model on complete train set
    ↓
Preprocessing Pipeline (7 steps):
  1. NULL string cleaning
  2. Column removal (ID + high cardinality)
  3. Constant column removal
  4. Missing indicator creation (6 features)
  5. Outlier winsorization (1% and 99% percentiles)
  6. Feature Engineering (11 new features)
  7. ColumnTransformer:
     - Numeric: median imputation + StandardScaler
     - Categorical: mode imputation + OneHotEncoder
    ↓
Processed Dataset (~100-200 final features)
    ↓
Optimal threshold calculation (F1 maximization on validation)
    ↓
Saving best model:
  ✓ models/production/model.joblib (bundle: pipeline + model + threshold)
  ✓ models/preprocessor/preprocessor.joblib (separate pipeline)
  ✓ models/production/optimal_threshold.txt (optimal threshold)
  ✓ models/production/metrics.txt (performance metrics)
  ✓ models/production/metrics_cv.json (cross-validation metrics)
  ✓ models/training_history/ (complete training history)
```

**Important note:** Preprocessing does NOT save processed CSV files. Instead, it saves the **trained pipeline** (`preprocessor.joblib`) that can be reused for any new data. The model is saved as a bundle that includes the complete pipeline.

---

### 2. **Prediction Phase** (API + UI)

```
User completes form in UI (Streamlit)
    ↓ Only provides essential fields (others are optional)
UI builds JSON request with basic features
    ↓
UI sends POST request to API (FastAPI)
    ↓
API receives simplified request
    ↓
Feature Mapper completes missing features:
  - Adds the 9 constant columns (default values)
  - Fills optional fields with default values or None
  - Orders columns in the correct order of the original dataset
    ↓
API creates DataFrame with all original columns
    ↓
Pipeline.transform() (uses saved pipeline)
  - Applies ALL saved transformations:
    1. NULL cleaning
    2. Column removal
    3. Constant removal
    4. Missing indicators
    5. Winsorization
    6. Feature engineering
    7. Imputation + Encoding + Scaling
  - Same processing as during training
  - Result: ~100-200 final numeric features
    ↓
Model.predict_proba() → Gets default probability (0-1)
    ↓
API compares probability with optimal_threshold (calculated during training):
  - If probability ≥ threshold → REJECTED
  - If probability < threshold → APPROVED
    ↓
API returns JSON response:
  {
    "prediction": "approved" or "rejected",
    "probability": 0.XX,
    "confidence": "high/medium/low"
  }
    ↓
UI displays result to user with explanation
```

---

## System Components

### **Preprocessing Pipeline** (`src/preprocessing.py`)

Reusable pipeline that transforms raw data into a format the model understands. Consists of **7 sequential steps**:

1. **NULL String Cleaning**

   - Converts placeholders "NULL", "NULL.1", "NULL.2", etc. to real NaN values
   - Converts empty strings to NaN

2. **Column Removal**

   - Removes `ID_CLIENT` (unique identifier)
   - Removes high cardinality columns: `CITY_OF_BIRTH`, `RESIDENCIAL_CITY`, `RESIDENCIAL_BOROUGH`, `PROFESSIONAL_CITY`, `PROFESSIONAL_BOROUGH`

3. **Constant Column Removal**

   - Automatically detects and removes columns without variance (all have the same value)
   - Typically removes 9 constant columns (CLERK_TYPE, FLAG_MOBILE_PHONE, etc.)

4. **Missing Indicators**

   - Creates **6 binary features** that indicate if important variables have missing values
   - Variables: PROFESSION_CODE, MONTHS_IN_RESIDENCE, MATE_PROFESSION_CODE, MATE_EDUCATION_LEVEL, RESIDENCE_TYPE, OCCUPATION_TYPE

5. **Winsorization (Outlier Capping)**

   - Limits extreme values using 1% and 99% percentiles from the training set
   - Applied to 10 numeric variables (income, assets, age, etc.)

6. **Feature Engineering**

   - Creates **11 new features**:
     - **Financial (3):** INCOME_TOTAL, INCOME_RATIO, HAS_OTHER_INCOME
     - **Stability (3):** YEARS_IN_RESIDENCE, YEARS_IN_JOB, STABILITY_SCORE
     - **Cards (2):** CARDS_COUNT, HAS_ANY_CARD
     - **Documentation (1):** DOCS_COUNT
     - **Cyclical (2):** PAYMENT_DAY_SIN, PAYMENT_DAY_COS

7. **ColumnTransformer (Imputation + Encoding + Scaling)**

   - **Numeric:** Median imputation + StandardScaler (mean=0, std=1)
   - **Categorical:** Mode imputation + OneHotEncoder (with grouping of infrequent categories <1%)
   - **Result:** ~100-200 final numeric features (depends on unique categories)

**Why we save the pipeline and not processed data:**

- Reusable for new data
- Less space (only saves transformers, not data)
- Guaranteed consistency (same preprocessing always)

---

### **Feature Mapper** (`src/api/feature_mapper.py`)

Converts simplified input from the UI to the complete format required by the model:

- Completes the **9 constant columns** (which are removed later but must be present)
- Fills optional fields with default values or `None`
- Orders columns in the correct order of the original dataset
- Ensures the DataFrame has exactly 53 columns before preprocessing

---

### **Model** (`models/production/model.joblib`)

**Models available for training:**

- **Gradient Boosting** (sklearn): Gradient boosting baseline
- **XGBoost**: Optimized gradient boosting, very popular in credit risk
- **LightGBM**: Fast and efficient gradient boosting
- **CatBoost**: Gradient boosting with automatic categorical handling
- **Logistic Regression**: Balanced linear model
- **HistGradientBoostingClassifier**: Optimized gradient boosting from sklearn

**Model selection:**

- All models are compared using **Cross-Validation (5 folds)**
- Best model is selected by **PR-AUC** (Precision-Recall AUC) in CV
- Best model is trained on the complete training set

**Optimal threshold:** Calculated on the validation set by maximizing **F1-score**. This threshold is saved and used in production to make approval/rejection decisions.

---

## File Format: Joblib

**Why do we use `.joblib` instead of `.pkl`?**

- Specialized for sklearn models and NumPy arrays
- More efficient with large objects
- Used by default in sklearn
- Better compatibility between versions

**Saved files:**

- `models/production/model.joblib`: Bundle with complete pipeline + model + threshold + metadata
- `models/preprocessor/preprocessor.joblib`: Preprocessing pipeline (may be embedded in bundle or separate)
- `models/production/optimal_threshold.txt`: Calculated optimal threshold
- `models/production/metrics.txt`: Performance metrics in text
- `models/production/metrics_cv.json`: Cross-validation metrics in JSON
- `models/production/val_metrics.json`: Validation metrics
- `models/training_history/training_history_*.json`: Complete history of each training
- `models/production/curves/`: ROC, CAP, PD distribution plots
- `models/production/calibration_table_*.csv`: Calibration tables
- `models/production/discriminatory_stats_*.csv`: Discriminatory statistics

All files are portable and can be shared with other developers.

---

## Quick Start Guide

### **For someone downloading the project for the first time**

This guide will help you set up the dataset, start Docker, train the model, and test it.

---

### **Prerequisites**

1. **Docker and Docker Compose** installed and working
2. **Dataset files** ready to place in `data/raw/`

---

### **Step 1: Prepare the Dataset**

Make sure you have the dataset files in the `data/raw/` folder:

```bash
data/raw/
  ├── PAKDD2010_Modeling_Data.txt
  └── PAKDD2010_VariablesList.XLS
```

**Important:** Files must be in this location before training the model.

---

### **Step 2: Start the System with Docker**

Build and start all services (UI, API and Model):

```bash
# First time (builds the images)
docker-compose up --build

# Subsequent times (faster, uses existing images)
docker-compose up
```

**What does this do?**

- Builds Docker images for UI, API and Model
- Starts the 3 services in separate containers
- Configures internal network between services
- Mounts necessary volumes (data, models, etc.)

**Available services:**

- **UI:** http://localhost:8501 (Streamlit)
- **API:** http://localhost:8000 (FastAPI)
- **API Docs:** http://localhost:8000/docs (Swagger UI)

**Note:** The first time may take several minutes to download and install dependencies.

---

### **Step 3: Train the Model**

With Docker running, execute training inside the API container (which has access to all data):

```bash
# Execute training inside API container
docker-compose exec api python -m src.modeling.train_eval
```

**Or if you prefer to train locally** (with Python installed on your machine):

```bash
# Install dependencies locally (only if not using Docker)
pip install -r requirements.txt

# Execute training
python -m src.modeling.train_eval
```

**What does this command do?**

1. Loads dataset from `data/raw/PAKDD2010_Modeling_Data.txt`
2. Splits into Train/Validation/Test (70% / 15% / 15%)
3. Executes Cross-Validation (5 folds) to compare models
4. Trains multiple models (Gradient Boosting, XGBoost, LightGBM, CatBoost, Logistic Regression, HistGBM)
5. Selects best model by PR-AUC in CV
6. Trains best model on complete train set
7. Evaluates on validation and calculates optimal threshold (F1 maximization)
8. Informatively evaluates on test (not used for selection)
9. Saves everything in:
   - `models/production/model.joblib` (complete bundle)
   - `models/preprocessor/preprocessor.joblib` (pipeline)
   - `models/production/optimal_threshold.txt`
   - `models/production/metrics.txt`
   - `models/production/metrics_cv.json`
   - `models/production/val_metrics.json`
   - `models/training_history/training_history_*.json`
   - `models/production/curves/` (plots)
   - `models/production/calibration_table_*.csv`
   - `models/production/discriminatory_stats_*.csv`

**Estimated time:** 5-15 minutes (depends on hardware and number of models)

**When finished you will see:**

- Cross-Validation metrics for each model
- Selected best model
- Metrics on train, validation and test
- Calculated optimal threshold
- Confirmation of saved files

**Important:** After training, restart the API service to load the new model:

```bash
docker-compose restart api
```

---

### **Step 4: Test the System with the UI**

#### **Option A: Use the UI (Recommended)**

1. Open your browser at: **http://localhost:8501**
2. Complete the form with customer data
3. Click "Evaluate Credit Risk"
4. You will see the result: **APPROVED** or **REJECTED** with the probability

#### **Option B: Use the API directly**

```bash
# Example request using curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "PAYMENT_DAY": 15,
    "APPLICATION_SUBMISSION_TYPE": "Web",
    "SEX": "M",
    "AGE": 35,
    "QUANT_DEPENDANTS": 1,
    "PERSONAL_MONTHLY_INCOME": 5000.0,
    "FLAG_RESIDENCIAL_PHONE": "Y",
    "COMPANY": "Y",
    "FLAG_PROFESSIONAL_PHONE": "Y"
  }'
```

**Expected response:**

```json
{
  "prediction": "approved",
  "probability": 0.4231,
  "confidence": "medium"
}
```

#### **Option C: Use interactive documentation**

1. Open: **http://localhost:8000/docs**
2. Expand the `/predict` endpoint
3. Click "Try it out"
4. Complete the example JSON
5. Click "Execute"
6. You will see the response directly in the browser

---

### **Useful Commands**

```bash
# View logs from all services
docker-compose logs -f

# View logs from a specific service
docker-compose logs -f api
docker-compose logs -f ui

# Stop services
docker-compose down

# Stop and remove volumes (cleans everything)
docker-compose down -v

# Rebuild a specific service
docker-compose build --no-cache api
docker-compose up api
```

---

### **Verify Everything Works**

1. **API health check:**

   ```bash
   curl http://localhost:8000/health
   ```

   Should return: `{"status":"ok","model_loaded":true,"preprocessor_loaded":true}`

2. **API model info:**

   ```bash
   curl http://localhost:8000/model_info
   ```

   Should show information about the loaded model

3. **UI loads correctly:** http://localhost:8501 shows the form

---

### **Troubleshooting Common Problems**

**Problem:** `FileNotFoundError: data/raw/PAKDD2010_Modeling_Data.txt`

- **Solution:** Verify that dataset files are in `data/raw/`

**Problem:** `Model or preprocessor not loaded`

- **Solution:** Make sure you have executed `python -m src.modeling.train_eval` first and then restart the API with `docker-compose restart api`

**Problem:** API returns error 500

- **Solution:** Check logs: `docker-compose logs api`
- Verify that `scikit-learn==1.6.1` is installed (version must match)
- Verify that the model is saved in `models/production/model.joblib`
- Verify that the preprocessor is in `models/preprocessor/preprocessor.joblib`

**Problem:** UI shows error loading `ui_options.json`

- **Solution:** Verify that the file `src/ui/ui_options.json` exists. If missing, the UI will work the same but with limited options.

## Complete Flow Summary

```
1. Prepare dataset → data/raw/
2. Start Docker → docker-compose up --build
3. Train model → docker-compose exec api python -m src.modeling.train_eval
4. Restart API → docker-compose restart api
5. Test system → http://localhost:8501
```

Ready! You now have the complete system working.
