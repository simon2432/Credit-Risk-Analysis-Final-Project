# 📚 Sistema Completo de Credit Risk Analysis

## 🎯 Resumen Ejecutivo

Este sistema evalúa el riesgo crediticio de clientes usando Machine Learning. El flujo completo es:

1. **Entrenamiento**: Se procesa el dataset, se entrenan modelos y se guarda el mejor.
2. **Predicción**: El usuario completa un formulario en la UI → API procesa los datos → Modelo predice → Se muestra el resultado.

---

## 🔄 Flujo Completo del Sistema

### 1️⃣ **Fase de Entrenamiento** (`src/train_model.py`)

```
Dataset Original (50,000 filas × 53 columnas)
    ↓
PreprocessingPipeline.fit_transform()
    ↓ Transformación automática:
      - Limpieza (remueve constantes y columnas alta cardinalidad+missing, normaliza Y/N)
      - Feature Engineering (crea 17 nuevas features)
      - Manejo de missing values (imputación + 6 indicadores)
      - Encoding (OneHot para baja cardinalidad, Ordinal para alta)
      - Scaling (MinMaxScaler 0-1)
    ↓
Dataset Procesado (50,000 filas × 286 features numéricas)
    ↓
Entrenamiento de 3 modelos:
  - Logistic Regression (class_weight='balanced')
  - Random Forest (class_weight='balanced')
  - Gradient Boosting (sample_weight='balanced')
    ↓
Evaluación en conjunto de validación
    ↓
Cálculo de threshold óptimo (Youden's J statistic)
    ↓
Guardado del mejor modelo:
  ✓ models/production/model.joblib (modelo)
  ✓ models/preprocessor/preprocessor.joblib (pipeline)
  ✓ models/production/optimal_threshold.txt (threshold óptimo)
  ✓ models/production/metrics.txt (métricas de rendimiento)
```

**Nota importante:** El preprocessing NO guarda archivos CSV procesados. En su lugar, guarda el **pipeline entrenado** (`preprocessor.joblib`) que puede reutilizarse para cualquier dato nuevo.

---

### 2️⃣ **Fase de Predicción** (API + UI)

```
Usuario completa formulario en UI (Streamlit)
    ↓ Solo proporciona campos esenciales (otros son opcionales)
UI construye request JSON con features básicas
    ↓
UI envía POST request a API (FastAPI)
    ↓
API recibe request simplificado
    ↓
Feature Mapper completa features faltantes:
  - Agrega las 9 columnas constantes (valores por defecto)
  - Completa campos opcionales con valores por defecto o None
  - Ordena columnas en el orden correcto del dataset original
    ↓
API crea DataFrame con todas las 53 columnas originales
    ↓
PreprocessingPipeline.transform() (usa pipeline guardado)
  - Aplica TODAS las transformaciones guardadas
  - Mismo procesamiento que durante entrenamiento
  - Resultado: 286 features numéricas finales
    ↓
Modelo.predict_proba() → Obtiene probabilidad de default (0-1)
    ↓
API compara probabilidad con optimal_threshold (0.5059):
  - Si probabilidad ≥ 0.5059 → RECHAZADO
  - Si probabilidad < 0.5059 → APROBADO
    ↓
API retorna respuesta JSON:
  {
    "prediction": "approved" o "rejected",
    "probability": 0.XX,
    "confidence": "high/medium/low"
  }
    ↓
UI muestra resultado al usuario con explicación
```

---

## 🔧 Componentes del Sistema

### **PreprocessingPipeline** (`src/preprocessing.py`)

Pipeline reutilizable que transforma datos raw en formato que el modelo entiende. Consta de **6 pasos secuenciales**:

1. **Limpieza Inicial**

   - Remueve `ID_CLIENT`
   - Convierte flags Y/N → 0/1
   - Identifica y remueve **9 columnas constantes**

2. **Manejo de Outliers**

   - No se aplica Winsorization
   - Basado en el EDA, el porcentaje de outliers es bajo (~2% máximo)
   - Los valores extremos son informativos para credit risk

3. **Feature Engineering**

   - Crea **17 nuevas features**:
     - Ratios financieros (ingresos/activos, ingresos por dependiente, etc.)
     - Scores de estabilidad (años en residencia/trabajo)
     - Conteos (tarjetas, métodos de contacto)
     - Comparaciones geográficas (estado residencia vs nacimiento, mismo ZIP, etc.)
     - Features de cuentas bancarias
     - Edad al cuadrado (relaciones no lineales)
   - **Nota:** Features de documentos fueron removidas (usaban columnas constantes)

4. **Manejo de Missing Values**

   - Crea **6 indicadores binarios** para missing importantes
   - Imputa: moda para categóricas, mediana para numéricas
   - **Resultado:** ~62 → ~68 columnas

5. **Encoding**

   - **Binarias (2 valores):** OrdinalEncoder
   - **Baja cardinalidad (≤20 categorías):** OneHotEncoder
   - **Alta cardinalidad (>20 categorías):** OrdinalEncoder
   - **Resultado:** 286 features numéricas finales

6. **Scaling**
   - MinMaxScaler (normaliza todas las features a rango 0-1)

**Por qué guardamos el pipeline y no los datos procesados:**

- ✅ Reutilizable para nuevos datos
- ✅ Menos espacio (solo guarda transformadores, no datos)
- ✅ Consistencia garantizada (mismo preprocessing siempre)

---

### **Feature Mapper** (`src/api/feature_mapper.py`)

Convierte el input simplificado de la UI al formato completo que requiere el modelo:

- Completa las **9 columnas constantes** (que se eliminan después pero deben estar presentes)
- Rellena campos opcionales con valores por defecto o `None`
- Ordena las columnas en el orden correcto del dataset original
- Garantiza que el DataFrame tenga exactamente 53 columnas antes del preprocessing

---

### **Modelo** (`models/production/model.joblib`)

**Modelo seleccionado:** Gradient Boosting Classifier

**Métricas actuales:**

- **ROC-AUC:** 0.64 (capacidad de distinguir entre clases)
- **F1:** 0.44 (balance entre precisión y recall)
- **Precision:** 0.35 (cuando predice "riesgoso", ¿cuántas veces tiene razón?)
- **Recall:** 0.58 (¿qué % de riesgosos detecta?)

**Threshold óptimo:** 0.5059 (calculado dinámicamente usando Youden's J statistic)

---

## 💾 Formato de Archivos: Joblib

**¿Por qué usamos `.joblib` en vez de `.pkl`?**

- ✅ Especializado para modelos sklearn y arrays NumPy
- ✅ Más eficiente con objetos grandes
- ✅ Usado por defecto en sklearn
- ✅ Mejor compatibilidad entre versiones

**Archivos guardados:**

- `model.joblib`: Modelo entrenado (Gradient Boosting)
- `preprocessor.joblib`: Pipeline de preprocessing completo
- Ambos archivos son portables y pueden compartirse con otros desarrolladores

---

## 📊 Estado Actual y Mejoras Aplicadas

### **Rendimiento del Modelo**

El modelo mejoró significativamente con las optimizaciones aplicadas:

| Métrica     | Antes | Ahora | Mejora         |
| ----------- | ----- | ----- | -------------- |
| **Recall**  | 0.08  | 0.58  | **7x mejor**   |
| **F1**      | 0.13  | 0.44  | **3.4x mejor** |
| **ROC-AUC** | 0.63  | 0.64  | Estable        |

### **Optimizaciones Implementadas**

1. ✅ **Threshold óptimo calculado dinámicamente** (se calcula automáticamente para cada modelo)
2. ✅ **Balanceo de clases** en todos los modelos
3. ✅ **17 nuevas features** de feature engineering
4. ✅ **6 indicadores de missing** para capturar información faltante
5. ✅ **Hiperparámetros optimizados** (más árboles, profundidad controlada)
6. ✅ **UI mejorada** con selectboxes descriptivos y opciones realistas
7. ✅ **Campos opcionales manejan `None`** correctamente (no `0`)

---

## 🚀 Guía de Inicio Rápido

### **Para alguien que descarga el proyecto por primera vez**

Esta guía te ayudará a configurar el dataset, levantar Docker, entrenar el modelo y probarlo.

---

### **Prerequisitos**

1. **Docker y Docker Compose** instalados y funcionando
2. **Datos del dataset** listos para colocar en `data/raw/`

---

### **Paso 1: Preparar el Dataset**

Asegúrate de tener los archivos del dataset en la carpeta `data/raw/`:

```bash
data/raw/
  ├── PAKDD2010_Modeling_Data.txt
  └── PAKDD2010_VariablesList.XLS
```

**Importante:** Los archivos deben estar en esta ubicación antes de entrenar el modelo.

---

### **Paso 2: Levantar el Sistema con Docker**

Construye y levanta todos los servicios (UI, API y Model):

```bash
# Primera vez (construye las imágenes)
docker-compose up --build

# Siguientes veces (más rápido, usa imágenes existentes)
docker-compose up
```

**¿Qué hace esto?**

- ✅ Construye las imágenes Docker para UI, API y Model
- ✅ Levanta los 3 servicios en contenedores separados
- ✅ Configura la red interna entre servicios
- ✅ Monta los volúmenes necesarios (datos, modelos, etc.)

**Servicios disponibles:**

- 🌐 **UI:** http://localhost:8501 (Streamlit)
- 🔌 **API:** http://localhost:8000 (FastAPI)
- 📊 **API Docs:** http://localhost:8000/docs (Swagger UI)

**Nota:** La primera vez puede tardar varios minutos en descargar e instalar dependencias.

---

### **Paso 3: Entrenar el Modelo**

Con Docker levantado, ejecuta el entrenamiento dentro del contenedor de la API (que tiene acceso a todos los datos):

```bash
# Ejecutar entrenamiento dentro del contenedor API
docker-compose exec api python -m src.train_model
```

**O si prefieres entrenar localmente** (con Python instalado en tu máquina):

```bash
# Instalar dependencias localmente (solo si no usas Docker)
pip install -r requirements.txt

# Ejecutar entrenamiento
python -m src.train_model
```

**¿Qué hace este comando?**

1. ✅ Carga el dataset desde `data/raw/`
2. ✅ Ejecuta el preprocessing completo (6 pasos)
3. ✅ Entrena 3 modelos (Logistic Regression, Random Forest, Gradient Boosting)
4. ✅ Evalúa y selecciona el mejor modelo
5. ✅ Calcula el threshold óptimo
6. ✅ Guarda todo en:
   - `models/production/model.joblib`
   - `models/preprocessor/preprocessor.joblib` (nueva ubicación)
   - `models/production/optimal_threshold.txt`
   - `models/production/metrics.txt`

**Tiempo estimado:** 1-3 minutos (depende del hardware)

**Al finalizar verás:**

- Métricas de cada modelo
- Modelo seleccionado como mejor
- Confirmación de archivos guardados

**Importante:** Después de entrenar, reinicia el servicio API para que cargue el nuevo modelo:

```bash
docker-compose restart api
```

---

### **Paso 4: Probar el Sistema con la UI**

#### **Opción A: Usar la UI (Recomendado)**

1. Abre tu navegador en: **http://localhost:8501**
2. Completa el formulario con los datos de un cliente
3. Haz clic en "Evaluar Riesgo Crediticio"
4. Verás el resultado: **APROBADO** o **RECHAZADO** con la probabilidad

#### **Opción B: Usar la API directamente**

```bash
# Ejemplo de request usando curl
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

**Respuesta esperada:**

```json
{
  "prediction": "approved",
  "probability": 0.4231,
  "confidence": "medium"
}
```

#### **Opción C: Usar la documentación interactiva**

1. Abre: **http://localhost:8000/docs**
2. Expande el endpoint `/predict`
3. Haz clic en "Try it out"
4. Completa el JSON de ejemplo
5. Haz clic en "Execute"
6. Verás la respuesta directamente en el navegador

---

### **Comandos Útiles**

```bash
# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f api
docker-compose logs -f ui

# Detener servicios
docker-compose down

# Detener y eliminar volúmenes (limpia todo)
docker-compose down -v

# Reconstruir un servicio específico
docker-compose build --no-cache api
docker-compose up api
```

---

### **Verificar que Todo Funciona**

1. ✅ **API health check:**

   ```bash
   curl http://localhost:8000/health
   ```

   Debe retornar: `{"status":"ok","model_loaded":true,"preprocessor_loaded":true}`

2. ✅ **API model info:**

   ```bash
   curl http://localhost:8000/model_info
   ```

   Debe mostrar información del modelo cargado

3. ✅ **UI carga correctamente:** http://localhost:8501 muestra el formulario

---

### **Solucionar Problemas Comunes**

**Problema:** `FileNotFoundError: data/raw/PAKDD2010_Modeling_Data.txt`

- **Solución:** Verifica que los archivos del dataset estén en `data/raw/`

**Problema:** `Model or preprocessor not loaded`

- **Solución:** Asegúrate de haber ejecutado `python -m src.train_model` primero

**Problema:** API retorna error 500

- **Solución:** Revisa los logs: `docker-compose logs api`
- Verifica que `scikit-learn==1.6.1` esté instalado (versión debe coincidir)

**Problema:** UI muestra error al cargar `ui_options.json`

- **Solución:** Verifica que el archivo `src/ui/ui_options.json` exista. Si falta, la UI funcionará igual pero con opciones limitadas.

---

## 📝 Resumen del Flujo Completo

```
1. Preparar dataset → data/raw/
2. Levantar Docker → docker-compose up --build
3. Entrenar modelo → docker-compose exec api python -m src.train_model
4. Probar sistema → http://localhost:8501
```

¡Listo! Ya tienes el sistema completo funcionando. 🎉
