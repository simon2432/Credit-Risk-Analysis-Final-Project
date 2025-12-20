# Credit Risk Analysis - Final Project

## 📋 Project Description

This project aims to develop a complete credit risk analysis system using Machine Learning techniques. The system will evaluate the probability of customer default and make informed decisions about credit approval or rejection.

**Dataset**: PAKDD2010 - Credit Risk Analysis Dataset

## 🎯 Objectives

1. **EDA (Exploratory Data Analysis)**: Perform a complete exploratory analysis of the customer and credit dataset.
2. **Preprocessing Pipeline**: Design a standard preprocessing pipeline for the entire team.
3. **Model Training**: Train and compare various ML models for credit risk (logistic regression, decision trees, ensembles, etc.).
4. **Model Selection**: Choose a final model based on evaluation metrics.
5. **API Deployment**: Expose the model through a REST API using FastAPI.
6. **UI Demo**: Build a simple interface (Streamlit) to demonstrate how a "bank" would use the model to approve/reject credits.

## 📁 Project Structure

```
Credit-Risk-Analysis-Final-Project/
├── data/
│   ├── raw/              # Original PAKDD2010 dataset unprocessed
│      ├── PAKDD2010_Modeling_Data.txt
│      ├── PAKDD2010_Prediction_Data.txt
│      ├── PAKDD2010_VariablesList.XLS
│      └── ...
│
├── notebooks/
│   └── EDA/              # Exploratory data analysis notebooks
│       ├── maria_EDA.ipynb
│       └── simon_EDA_ordered.ipynb
├── src/                  # Python source code
│   ├── __init__.py
│   ├── preprocessing.py  # Preprocessing pipeline
│   ├── train_model.py    # Model training script
│   ├── data_utils.py     # Data loading utilities
│   ├── models_config.py  # Model configurations
│   ├── config.py         # Configuration paths
│   ├── api/              # FastAPI service
│   │   ├── server.py
│   │   └── feature_mapper.py
│   └── ui/               # Streamlit UI
│       ├── simple_app.py
│       └── ui_options.json
├── models/               # Saved trained models (gitignored, except .gitkeep)
├── data/
│   ├── raw/              # Original dataset (gitignored, except .gitkeep)
│   └── processed/        # Processed data (gitignored, except .gitkeep)
├── Dockerfile            # Docker for API + model
├── docker-compose.yml    # Service orchestration
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## 🚀 Installation and Setup

### 1. Create virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify installation

```bash
python -c "import pandas, sklearn, fastapi; print('Dependencies installed correctly')"
```

## 📝 Current Project Status

The project is **fully functional** and ready for use:

- ✅ **Folder structure**: Complete project organization
- ✅ **Dataset**: PAKDD2010 data processing implemented
- ✅ **EDA**: Exploratory analysis completed (`notebooks/EDA/`)
- ✅ **Preprocessing**: Complete preprocessing pipeline implemented (`src/preprocessing.py`)
- ✅ **Models**: Training and comparison implemented (`src/train_model.py`)
  - Logistic Regression, Random Forest, Gradient Boosting
  - Automatic model selection based on ROC-AUC
  - Optimal threshold calculation
- ✅ **API**: FastAPI service implemented (`src/api/server.py`)
  - `/predict` endpoint for credit risk evaluation
  - Automatic model and preprocessor loading
- ✅ **UI**: Streamlit interface implemented (`src/ui/simple_app.py`)
  - User-friendly form for credit evaluation
  - Real-time predictions via API

## 📝 Quick Start

For detailed instructions, see:

- **`SISTEMA_COMPLETO.md`**: Complete system documentation with quick start guide
- **`DOCKER_QUICK.md`**: Docker setup and usage guide
- **`MODELOS_PERSONALIZADOS.md`**: Guide to add custom models
- **`PREPROCESSING_PLAN.md`**: Detailed preprocessing documentation

**Quick setup:**

1. Place dataset files in `data/raw/`
2. Run `docker-compose up --build`
3. Train models: `python -m src.train_model`
4. Use UI: http://localhost:8501

## 👥 Team

This project is being developed by a team of 6 people.
