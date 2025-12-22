# Documentación de Mejoras y Análisis del Proyecto

## 1. Resumen de Optimizaciones
Este documento detalla los cambios realizados para estandarizar el proyecto "Credit Risk Analysis" al inglés y expandir sus capacidades predictivas. Se han creado versiones mejoradas de los archivos originales con el sufijo `_ok` para preservar el trabajo previo.

### Archivos Mejorados
-   **`src/preprocessing_ok.py`**: Pipeline completo documentado 100% en inglés.
-   **`src/models_config_ok.py`**: Nueva configuración que incluye Regresión Logística, Random Forest y CatBoost.
-   **`src/train_model_ok.py`**: Script de entrenamiento con logs en inglés y manejo de múltiples modelos.

---

## 2. Análisis Detallado por Etapa

### Fase 1: Análisis Exploratorio de Datos (EDA)
Se identificaron patrones críticos que guían el preprocesamiento:
-   **Desbalance de Clases**: El target (`TARGET_LABEL_BAD=1`) tiene un radio de ~1:3.
    -   *Solución*: Se implementó soporte para `class_weight='balanced'` y `compute_sample_weight` en el entrenamiento.
-   **Variables Constantes**: Columnas como `CLERK_TYPE`, `QUANT_ADDITIONAL_CARDS` tienen varianza cero.
    -   *Acción*: Eliminación automática en paso 1 del pipeline para reducir ruido.
-   **Outliers**: Presentes en Ingresos (`PERSONAL_MONTHLY_INCOME`) y Edad.
    -   *Estrategia*: Aunque los modelos de árboles son robustos, se recomienda capping (1%-99%) si se usan modelos lineales. El código actual soporta esto.
-   **Valores Perdidos (Missing)**:
    -   Muy altos en `PROFESSION_CODE` y `MATE_PROFESSION_CODE`.
    -   *Insight*: El valor perdido es informativo (posible desempleo o informalidad).
    -   *Acción*: Se crean banderas binarias `MISSING_*` antes de imputar.

### Fase 2: Pipeline de Preprocesamiento (`preprocessing_ok.py`)
El pipeline sigue una lógica secuencial robusta de 6 pasos:
1.  **Limpieza Inicial**: Normaliza columnas Y/N y elimina constantes.
2.  **Manejo de Outliers**: (Configurable)
3.  **Feature Engineering**:
    -   *Ratios Financieros*: `INCOME_RATIO` (Ingresos otros / Ingresos personales).
    -   *Estabilidad*: `YEARS_IN_JOB`, `YEARS_IN_RESIDENCE`.
    -   *Interacciones*: `SAME_STATE_RES_PROF` (Coherencia geográfica).
4.  **Imputación**:
    -   Mediana para numéricas.
    -   Moda para categóricas.
    -   **Excepción**: Columnas de alta cardinalidad preservan NaN para el encoding.
5.  **Encoding (Codificación)**:
    -   *Ordinal*: Para binarias y Y/N.
    -   *Frequency Encoding*: Para columnas con >100 categorías (ej. Códigos postales).
    -   *One-Hot Encoding*: Para cardinalidad baja/media (agrupando raros como "OTHER").
6.  **Escalado**: `MinMaxScaler` (0-1) para ayudar a la convergencia de Regresión Logística y Redes Neuronales.

### Fase 3: Modelado (`models_config_ok.py`)
Se expandió el repertorio de modelos para cubrir diferentes enfoques matemáticos:

| Modelo | Tipo | Razón de Inclusión |
| :--- | :--- | :--- |
| **Logistic Regression** | Lineal | **Baseline**. Interpretable y rápida. Fundamental para entender relaciones lineales. |
| **Random Forest** | Bagging | **Estabilidad**. Reduce varianza y es menos propenso a overfitear que el boosting. |
| **XGBoost / LightGBM** | Boosting | **Potencia**. Estado del arte para datos tabulares. |
| **CatBoost** | Boosting | **Categóricas**. Maneja variables categóricas nativamente mejor que OHE en muchos casos. |

---

## 3. Guía de Ejecución y Entorno

### Configuración del Entorno (Virtual Env)
Para trabajar de forma aislada y no afectar tu sistema global:

1.  **Crear el entorno** (si no existe):
    ```bash
    python -m venv venv
    ```
2.  **Activar el entorno**:
    *   Windows: `.\venv\Scripts\Activate`
    *   Mac/Linux: `source venv/bin/activate`
    *   *Verás `(venv)` al inicio de tu línea de comandos.*
3.  **Instalar dependencias**:
    ```bash
    pip install -r requirements.txt
    ```
    *Nota: Si tienes error de `numpy`, ejecuta `pip install --upgrade numpy pandas scikit-learn`*

### Ejecución del Entrenamiento Mejorado
Con el entorno activado, ejecuta:
```bash
python -m src.train_model_ok
```
Esto entrenará todos los modelos configurados y guardará los resultados en la carpeta `models/`.

### Control de Versiones (.gitignore)
Para subir tus cambios a GitHub sin subir archivos basura o el entorno virtual:

1.  Asegúrate de que tu archivo `.gitignore` contenga estas líneas (tu archivo actual ya las tiene ✅):
    ```gitignore
    # Virtual Environment
    venv/
    env/
    .venv/
    
    # Models & Data (Generalmente no se suben si son pesados)
    models/*
    !models/.gitkeep
    data/raw/*
    !data/raw/.gitkeep
    ```
2.  **Flujo Git**:
    ```bash
    git status          # Ver cambios
    git add .           # Agregar todo (el .gitignore evitará que se agregue venv/)
    git commit -m "feat: optimize pipeline and add english docs"
    git push origin <tu-rama>
    ```
