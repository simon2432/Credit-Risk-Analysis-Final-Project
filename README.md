# Credit Risk Analysis - Final Project

## Project Description

This project aims to develop a complete credit risk analysis system using Machine Learning techniques. The system will evaluate the probability of customer default and make informed decisions about credit approval or rejection.

**Dataset**: PAKDD2010 - Credit Risk Analysis Dataset

## Objectives

1. **EDA (Exploratory Data Analysis)**: Perform a complete exploratory analysis of the customer and credit dataset.
2. **Preprocessing Pipeline**: Design a standard preprocessing pipeline for the entire team.
3. **Model Training**: Train and compare various ML models for credit risk (logistic regression, decision trees, ensembles, etc.).
4. **Model Selection**: Choose a final model based on evaluation metrics.
5. **API Deployment**: Expose the model through a REST API using FastAPI.
6. **UI Demo**: Build a simple interface (Streamlit) to demonstrate how a "bank" would use the model to approve/reject credits.

## Project Structure

```
Credit-Risk-Analysis-Final-Project/
├── data/
│   ├── raw/              # Original PAKDD2010 dataset unprocessed
│   │   ├── PAKDD2010_Modeling_Data.txt
│   │   ├── PAKDD2010_Prediction_Data.txt
│   │   └── PAKDD2010_VariablesList.XLS
│   └── processed/        # Processed data directory (not used, pipeline is saved instead)
├── models/               # Saved trained models and artifacts
│   ├── preprocessor/     # Preprocessing pipeline
│   ├── production/       # Production model and metrics
│   └── training_history/ # Training history logs
├── notebooks/
│   └── EDA/              # Exploratory data analysis notebooks
│       ├── maria_EDA.ipynb
│       └── simon_EDA_ordered.ipynb
├── scripts/              # Utility scripts for training, CV, and analysis
├── src/                  # Python source code
│   ├── __init__.py
│   ├── preprocessing.py  # Preprocessing pipeline implementation
│   ├── data_utils.py     # Data loading utilities
│   ├── models_config.py  # Model configurations
│   ├── config.py         # Configuration paths and settings
│   ├── api/              # FastAPI service
│   │   ├── server.py     # API endpoints
│   │   └── feature_mapper.py  # Feature mapping utilities
│   ├── ui/               # Streamlit UI
│   │   ├── simple_app.py  # Main UI application
│   │   └── ui_options.json  # UI dropdown options
│   └── modeling/         # Model training and evaluation
│       ├── train_eval.py    # Main training and evaluation system
│       ├── pipelines.py     # Pipeline factory functions
│       └── payer_segments.py  # Customer segmentation utilities
├── docker-compose.yml    # Docker Compose orchestration (3 services)
├── Dockerfile.api        # Dockerfile for API service
├── Dockerfile.ui         # Dockerfile for UI service
├── Dockerfile.model      # Dockerfile for model service
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Installation and Setup

The system uses Docker Compose for easy deployment:

```bash
# Build and start all services
docker-compose up --build

# Services will be available at:
# - UI: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

For detailed setup and usage instructions, see **`SISTEMA_COMPLETO.md`**.

## Current Project Status

The project is **fully functional** and ready for use:

- **Folder structure**: Complete project organization
- **Dataset**: PAKDD2010 data processing implemented
- **EDA**: Exploratory analysis completed (`notebooks/EDA/`)
- **Preprocessing**: Complete preprocessing pipeline implemented (`src/preprocessing.py`)
  - 7-step pipeline: cleaning, column removal, missing indicators, winsorization, feature engineering, imputation, encoding, scaling
  - Creates 11 new features + 6 missing indicators
- **Models**: Training and evaluation system implemented (`src/modeling/train_eval.py`)
  - Multiple models: Gradient Boosting, XGBoost, LightGBM, CatBoost, Logistic Regression, HistGBM
  - Cross-validation (5 folds) for model comparison
  - Automatic model selection based on PR-AUC
  - Optimal threshold calculation (F1 maximization)
- **API**: FastAPI service implemented (`src/api/server.py`)
  - `/predict` endpoint for credit risk evaluation
  - Automatic model and preprocessor loading
  - Feature mapping from simplified input to full dataset format
- **UI**: Streamlit interface implemented (`src/ui/simple_app.py`)
  - Complete form with all necessary features
  - Real-time predictions via API
  - User-friendly visualization of results
- **Docker**: Complete containerization with Docker Compose
  - 3 services: API, UI, and Model service
  - Easy deployment and scaling

## Quick Start

For detailed instructions, see:

- **`SISTEMA_COMPLETO.md`**: Complete system documentation with quick start guide
- **`DOCKER_QUICK.md`**: Docker setup and usage guide
- **`MODELOS_PERSONALIZADOS.md`**: Guide to add custom models
- **`PREPROCESSING_PLAN.md`**: Detailed preprocessing documentation

**Quick setup:**

1. Place dataset files in `data/raw/`
2. Run `docker-compose up --build`
3. Train models: `python -m src.modeling.train_eval`
4. Restart API: `docker-compose restart api`
5. Use UI: http://localhost:8501
