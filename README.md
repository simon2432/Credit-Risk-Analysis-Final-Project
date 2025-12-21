# Credit Risk Analysis - Final Project

## 📋 Project Overview

A complete **Credit Risk Analysis System** using Machine Learning to evaluate the probability of customer default and make automated loan approval/rejection decisions. The system consists of a trained ML model (Gradient Boosting, XGBoost, LightGBM), a FastAPI REST API, and a Streamlit web interface for real-time credit risk evaluation.

**Dataset**: PAKDD2010 - Credit Risk Analysis Dataset

## 🏗️ Architecture

The system is built using **Docker Compose** with 3 microservices:

- **Model Service**: Data container storing trained models and preprocessor
- **API Service** (FastAPI): REST API on port `8000` for predictions
- **UI Service** (Streamlit): Web interface on port `8501` for credit evaluation

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Dataset files placed in `data/raw/`:
  - `PAKDD2010_Modeling_Data.txt`
  - `PAKDD2010_VariablesList.XLS`

### 1. Train the Model

First, train the model and generate the preprocessor:

```bash
python -m src.train_model
```

This will:

- Load and preprocess the dataset
- Train multiple models (Gradient Boosting, XGBoost, LightGBM)
- Select the best model based on ROC-AUC
- Save the model to `models/production/model.joblib`
- Save the preprocessor to `models/preprocessor/preprocessor.joblib`

### 2. Start the System

Start all services with Docker Compose:

```bash
# First time (builds Docker images)
docker-compose up --build

# Subsequent times (faster)
docker-compose up
```

Or run in background:

```bash
docker-compose up -d
```

### 3. Access the Application

- **UI Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## 📁 Project Structure

```
Credit-Risk-Analysis-Final-Project/
├── data/
│   └── raw/                    # Dataset files (PAKDD2010)
├── notebooks/
│   └── EDA/                    # Exploratory data analysis
├── src/
│   ├── preprocessing.py        # Preprocessing pipeline
│   ├── train_model.py          # Model training script
│   ├── models_config.py        # Model configurations
│   ├── api/
│   │   └── server.py           # FastAPI service
│   └── ui/
│       └── simple_app.py       # Streamlit UI
├── models/
│   ├── production/             # Trained model (generated)
│   └── preprocessor/           # Preprocessor (generated)
├── docker-compose.yml          # Service orchestration
└── requirements.txt            # Python dependencies
```

## 📚 Documentation

For detailed information, see:

- **`SISTEMA_COMPLETO.md`**: Complete system documentation
- **`PREPROCESSING_PLAN.md`**: Preprocessing pipeline details
- **`MODELOS_PERSONALIZADOS.md`**: Guide to add custom models
- **`DOCKER_QUICK.md`**: Docker usage guide

## 🔧 Useful Commands

```bash
# View logs
docker-compose logs -f          # All services
docker-compose logs -f api      # API only
docker-compose logs -f ui       # UI only

# Stop services
docker-compose down

# Rebuild after code changes
docker-compose up --build
```
