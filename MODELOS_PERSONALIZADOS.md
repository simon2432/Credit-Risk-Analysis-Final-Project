# 🔧 Guía para Probar Modelos Personalizados

Esta guía te explica **paso a paso** cómo agregar y probar tus propios modelos en el sistema de Credit Risk Analysis.

---

## 🎯 Resumen Rápido

Para agregar un modelo nuevo, solo necesitas:

1. **Abrir** `src/models_config.py`
2. **Agregar** tu modelo al diccionario (3 campos: `class`, `params`, `class_weight`)
3. **Ejecutar** `python -m src.train_model`
4. **Listo** - Tu modelo se entrenará y comparará automáticamente con los demás

---

## 📋 Prerequisitos

Antes de empezar, asegúrate de tener:

- ✅ Python instalado
- ✅ El proyecto configurado (dependencias instaladas)
- ✅ El dataset en `data/raw/PAKDD2010_Modeling_Data.txt`
- ✅ Conocimiento básico de scikit-learn (opcional, pero útil)

**No necesitas** modificar código complejo, solo editar un archivo de configuración.

---

## 📝 Paso a Paso: Agregar un Nuevo Modelo

### Paso 1: Abrir el archivo de configuración

Abre `src/models_config.py` en tu editor. Verás algo así:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models_config() -> Dict[str, Dict[str, Any]]:
    return {
        "Logistic Regression": { ... },
        "Random Forest": { ... },
        "Gradient Boosting": { ... },
    }
```

### Paso 2: Importar tu modelo (si es necesario)

Si tu modelo no está en los imports, agrégalo arriba. Por ejemplo, para XGBoost:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier  # <-- Agregar aquí
```

**Nota**: Si usas un modelo de scikit-learn que ya está importado, puedes saltarte este paso.

### Paso 3: Agregar tu modelo al diccionario

Dentro de `get_models_config()`, agrega una nueva entrada. **Ejemplo con XGBoost**:

```python
def get_models_config() -> Dict[str, Dict[str, Any]]:
    return {
        "Logistic Regression": { ... },
        "Random Forest": { ... },
        "Gradient Boosting": { ... },

        # Tu nuevo modelo aquí:
        "XGBoost": {
            "class": XGBClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "class_weight": "sample_weight",
        },
    }
```

### Paso 4: Ejecutar el entrenamiento

Guarda el archivo y ejecuta:

```bash
python -m src.train_model
```

**¡Eso es todo!** Tu modelo se entrenará automáticamente y se comparará con los demás. Verás los resultados en la consola y en `models/training_history/`.

---

## 🔍 Explicación de los 3 Campos Necesarios

Cada modelo necesita **exactamente 3 campos**:

### 1. `"class"` - La clase del modelo

Es la clase del modelo de scikit-learn (sin paréntesis). Ejemplos:

```python
"class": LogisticRegression
"class": RandomForestClassifier
"class": XGBClassifier  # Si usas XGBoost
```

**⚠️ Importante**: El modelo DEBE tener `predict_proba()`. Algunos modelos necesitan configuración especial:

- **SVC**: Agrega `"probability": True` en `params`
- La mayoría de modelos de scikit-learn ya lo tienen

### 2. `"params"` - Los hiperparámetros

Diccionario con los parámetros del modelo. Ejemplos:

```python
"params": {
    "random_state": 42,      # Siempre incluir para reproducibilidad
    "n_estimators": 200,     # Parámetros específicos del modelo
    "max_depth": 6,
    "learning_rate": 0.1,
}
```

**💡 Tip**: Puedes usar los valores por defecto de scikit-learn o ajustarlos según tu experiencia. Siempre incluye `"random_state": 42` si el modelo lo soporta.

### 3. `"class_weight"` - Estrategia de balanceo

Cómo manejar el desbalanceo de clases (tenemos ~74% clase 0, ~26% clase 1). **3 opciones**:

#### Opción A: `"balanced"` (recomendado si el modelo lo soporta)

```python
"class_weight": "balanced"
```

**Usa esto para**: LogisticRegression, RandomForest, SVC, etc.

#### Opción B: `"sample_weight"` (para modelos sin class_weight)

```python
"class_weight": "sample_weight"
```

**Usa esto para**: GradientBoosting, XGBoost, LightGBM, etc.

#### Opción C: `None` (no recomendado)

```python
"class_weight": None
```

**No recomendado** porque el dataset está desbalanceado.

**¿Cómo saber cuál usar?**

- Si el modelo tiene el parámetro `class_weight` → usa `"balanced"`
- Si NO tiene `class_weight` → usa `"sample_weight"`
- Consulta la documentación de scikit-learn si tienes dudas

---

## 📊 Ejemplo Completo: Agregar XGBoost

Vamos a agregar XGBoost paso a paso:

### Paso 1: Instalar XGBoost

```bash
pip install xgboost
```

### Paso 2: Modificar `src/models_config.py`

**Antes:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models_config() -> Dict[str, Dict[str, Any]]:
    return {
        "Logistic Regression": { ... },
        "Random Forest": { ... },
        "Gradient Boosting": { ... },
    }
```

**Después:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier  # <-- Agregar import

def get_models_config() -> Dict[str, Dict[str, Any]]:
    return {
        "Logistic Regression": { ... },
        "Random Forest": { ... },
        "Gradient Boosting": { ... },

        "XGBoost": {  # <-- Agregar modelo nuevo
            "class": XGBClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
            },
            "class_weight": "sample_weight",  # XGBoost no tiene class_weight
        },
    }
```

### Paso 3: Ejecutar

```bash
python -m src.train_model
```

**Resultado**: XGBoost se entrenará junto con los otros modelos y verás sus métricas en la consola.

---

## 📈 Métricas que se Calculan Automáticamente

Para cada modelo, el sistema calcula y guarda:

### Métricas en Train (threshold=0.5)

- ROC-AUC
- F1-Score
- Precision
- Recall

### Métricas en Validation (threshold=0.5)

- ROC-AUC
- F1-Score
- Precision
- Recall

### Métricas en Validation (threshold óptimo)

- F1-Score
- Precision
- Recall
- **Threshold óptimo** (calculado con Youden's J statistic)

### Información Adicional

- ⏱️ Tiempo de entrenamiento
- ⚙️ Hiperparámetros usados
- ⚖️ Estrategia de balanceo

**Nota**: El conjunto de test se guarda en memoria pero NO se usa para evaluación (se reserva para evaluación final del modelo seleccionado).

---

## 📁 Archivos que se Generan

Después de ejecutar `python -m src.train_model`, se crean:

1. **`models/production/model.joblib`**

   - El mejor modelo (seleccionado por ROC-AUC en validation)
   - Se usa automáticamente por la API

2. **`models/production/preprocessor.joblib`**

   - Pipeline de preprocessing guardado
   - Se usa automáticamente por la API

3. **`models/production/metrics.txt`**

   - Métricas del mejor modelo en formato legible
   - Fácil de leer y compartir

4. **`models/production/optimal_threshold.txt`**

   - Threshold óptimo del mejor modelo
   - Se usa automáticamente por la API para decisiones

5. **`models/training_history/training_history_YYYYMMDD_HHMMSS.json`**
   - Historial completo de TODOS los modelos entrenados
   - Incluye métricas, hiperparámetros, tiempos, etc.
   - Útil para comparar modelos

---

## 🎯 Selección del Mejor Modelo

El sistema selecciona el mejor modelo basándose en:

- **ROC-AUC en el conjunto de Validation**

Si quieres cambiar este criterio, modifica la función `train_models()` en `src/train_model.py`, específicamente esta línea:

```python
if val_roc_auc > best_score:
    best_score = val_roc_auc
    best_model = model
    best_model_name = model_name
```

Puedes cambiarlo a otra métrica (F1, Precision, Recall, etc.).

---

## ⚠️ Consideraciones Importantes

### 1. Método `predict_proba()`

**IMPORTANTE**: Tu modelo DEBE tener el método `predict_proba()`. Algunos modelos requieren configuración especial:

- **SVC**: Agrega `"probability": True` en params
- **Otros modelos**: Consulta la documentación de scikit-learn

### 2. Balanceo de Clases

Este dataset está desbalanceado (muchos más casos clase 0 que clase 1). Por eso es importante usar balanceo:

- Usa `"class_weight": "balanced"` si el modelo lo soporta
- Usa `"sample_weight": "sample_weight"` si el modelo NO tiene `class_weight`

### 3. Reproducibilidad

Siempre incluye `"random_state": 42` en los params si tu modelo lo soporta para tener resultados reproducibles.

### 4. Hiperparámetros

Ajusta los hiperparámetros según tu conocimiento del modelo. Puedes usar:

- Valores por defecto de scikit-learn
- Valores encontrados en la literatura
- Optimización de hiperparámetros (GridSearchCV, etc.)

---

## ✅ Verificar que Todo Funciona

Después de agregar tu modelo:

1. **Ejecuta**: `python -m src.train_model`
2. **Revisa la consola**: Deberías ver las métricas de tu modelo impresas
3. **Revisa `models/training_history/`**: Abre el JSON más reciente para ver todos los detalles
4. **Compara métricas**: ¿Tu modelo es mejor que los demás? Revisa el ROC-AUC en validation

**Si hay errores:**

- Verifica que el import esté correcto
- Verifica que el modelo tenga `predict_proba()`
- Revisa que los hiperparámetros sean válidos para ese modelo

---

## 📚 Recursos Útiles

- [Documentación de scikit-learn](https://scikit-learn.org/stable/)
- [Guía de clasificación desbalanceada](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Tuning de hiperparámetros](https://scikit-learn.org/stable/modules/grid_search.html)

---

## ❓ Preguntas Frecuentes

### ¿Puedo usar modelos de otras librerías (TensorFlow, PyTorch)?

Actualmente el sistema está diseñado para modelos de **scikit-learn** que tienen la interfaz estándar (`fit()`, `predict()`, `predict_proba()`).

Para otros frameworks necesitarías crear un wrapper que implemente esta interfaz. Si necesitas ayuda con esto, consulta con el equipo.

### ¿Dónde veo los resultados de todos los modelos?

Revisa los archivos JSON en `models/training_history/`. Cada archivo contiene:

- Métricas de todos los modelos
- Hiperparámetros usados
- Tiempos de entrenamiento
- Threshold óptimo de cada modelo

**Ejemplo**: Abre `training_history_20251218_043613.json` y busca tu modelo por nombre.

### ¿Puedo entrenar solo algunos modelos?

Sí, simplemente **comenta o elimina** las entradas que no quieras en `get_models_config()`:

```python
return {
    "Logistic Regression": { ... },
    # "Random Forest": { ... },  # <-- Comentado, no se entrenará
    "Gradient Boosting": { ... },
}
```

### ¿Qué pasa si mi modelo da error?

1. Verifica que el import esté correcto
2. Verifica que los hiperparámetros sean válidos
3. Verifica que el modelo tenga `predict_proba()`
4. Revisa el mensaje de error en la consola

El sistema continuará entrenando los otros modelos aunque uno falle.

---

## 🚀 Resumen: Pasos Rápidos

1. **Abrir** `src/models_config.py`
2. **Agregar import** (si es necesario)
3. **Agregar modelo** al diccionario con 3 campos:
   - `"class"`: La clase del modelo
   - `"params"`: Hiperparámetros
   - `"class_weight"`: `"balanced"` o `"sample_weight"`
4. **Ejecutar** `python -m src.train_model`
5. **Revisar resultados** en consola y `models/training_history/`

---

## 💡 Tips Finales

- **Empieza simple**: Usa valores por defecto o valores comunes de la literatura
- **Experimenta**: Prueba diferentes hiperparámetros y compara resultados
- **Revisa el historial**: Los JSON tienen toda la información para comparar modelos
- **No te preocupes por errores**: El sistema continuará con otros modelos si uno falla

**¡Éxito con tus experimentos!** 🎉
