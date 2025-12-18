# 📊 Hallazgos del EDA - Resumen Completo

## 🎯 Target Variable

- **Variable:** `TARGET_LABEL_BAD=1`
- **Distribución:**
  - NO (0): ~73.92% (36,959 casos)
  - YES (1): ~26.08% (13,041 casos)
- **Conclusión:** Dataset desbalanceado (3:1 ratio)

---

## 🔍 Columnas Constantes (Sin Varianza)

Según el EDA, se encontraron **9 columnas constantes** que se detectan y remueven automáticamente durante el preprocessing:

1. **`CLERK_TYPE`** - Todos valores "C"
2. **`QUANT_ADDITIONAL_CARDS`** - Todos valores 0
3. **`EDUCATION_LEVEL`** - Todos valores 0
4. **`FLAG_MOBILE_PHONE`** - Todos valores "N"
5. **`FLAG_HOME_ADDRESS_DOCUMENT`** - Todos valores 0
6. **`FLAG_RG`** - Todos valores 0
7. **`FLAG_CPF`** - Todos valores 0
8. **`FLAG_INCOME_PROOF`** - Todos valores 0
9. **`FLAG_ACSP_RECORD`** - Todos valores "N"

**Nota:** El preprocessing detecta automáticamente columnas constantes usando `nunique() == 1` o `std() == 0`. Estas columnas se eliminan antes de cualquier procesamiento adicional ya que no aportan información para el modelado.

---

## 📉 Missing Values

### Variables con Muchos Missing:

Basado en el análisis, las variables con más missing values son:

1. **`PROFESSIONAL_CITY`** - Muchos missing (NaN) ❌ **REMOVIDA** (alta cardinalidad + muchos missing)
2. **`PROFESSIONAL_BOROUGH`** - Muchos missing (NaN) ❌ **REMOVIDA** (alta cardinalidad + muchos missing)
3. **`MATE_PROFESSION_CODE`** - Muchos missing (NaN) ⭐ (tiene indicador de missing)
4. **`MATE_EDUCATION_LEVEL`** - Muchos missing (NaN) ⭐ (tiene indicador de missing)
5. **`PROFESSION_CODE`** - Algunos missing ⭐ (tiene indicador de missing)
6. **`MONTHS_IN_RESIDENCE`** - Algunos missing (NULL) ⭐ (tiene indicador de missing)
7. **`RESIDENCE_TYPE`** - Algunos missing (NULL) ⭐ (tiene indicador de missing)
8. **`OCCUPATION_TYPE`** - Algunos missing (NULL) ⭐ (tiene indicador de missing)

**Variables con missing pero sin indicador:**

- `PROFESSIONAL_PHONE_AREA_CODE` - Puede tener algunos missing (se imputa pero no se crea indicador)

**Nota:** Para las 6 variables marcadas con ⭐ (ya no 8, porque PROFESSIONAL*CITY y PROFESSIONAL_BOROUGH fueron removidas), el preprocessing crea indicadores binarios (`MISSING*\*`) antes de imputar. Los missing en variables profesionales pueden ser informativos (indica que no tiene trabajo formal).

---

## 📊 Outliers Detectados

Según el análisis de maria_EDA, se encontraron outliers en las siguientes variables (proporción fuera de percentiles 1%-99%):

1. **`PERSONAL_MONTHLY_INCOME`** - 2% de outliers
2. **`MONTHS_IN_RESIDENCE`** - 0.85% de outliers
3. **`OTHER_INCOMES`** - 0.92% de outliers
4. **`PERSONAL_ASSETS_VALUE`** - 0.96% de outliers
5. **`AGE`** - 0.88% de outliers
6. **`MONTHS_IN_THE_JOB`** - 0.19% de outliers
7. **`PROFESSION_CODE`** - 0.85% de outliers
8. **`MATE_PROFESSION_CODE`** - 0.43% de outliers
9. **`MARITAL_STATUS`** - 0.45% de outliers
10. **`QUANT_DEPENDANTS`** - 0.61% de outliers

**Variables con más outliers:**

- `PERSONAL_MONTHLY_INCOME` (2%) - Variable crítica
- `OTHER_INCOMES` (0.92%)
- `PERSONAL_ASSETS_VALUE` (0.96%)

---

## 🔗 Correlaciones y Variables Importantes

### Análisis de Correlación con Target:

El EDA muestra análisis de correlación usando:

- **Numéricas:** Correlación de Pearson
- **Categóricas:** Cramer's V (chi-squared)

### Variables Clave Identificadas:

1. ~~**`FLAG_ACSP_RECORD`**~~ - Variable identificada como importante en EDA original, pero resultó ser constante (todos "N") en el dataset actual - **removida**
2. **`PAYMENT_DAY`** - Analizado en relación con target
3. **`PERSONAL_MONTHLY_INCOME`** - Variable financiera crítica
4. **`AGE`** - Variable demográfica importante
5. **`MONTHS_IN_THE_JOB`** - Estabilidad laboral
6. **`MONTHS_IN_RESIDENCE`** - Estabilidad residencial

---

## 💡 Feature Engineering Ya Realizado en EDA

En el notebook de maria se encontró un ejemplo de feature engineering:

```python
# Income Ratio (ejemplo encontrado)
INCOME_RATIO = OTHER_INCOMES / (PERSONAL_MONTHLY_INCOME + 1e-6)
```

Esta feature muestra diferencia entre grupos del target, lo que sugiere que ratios financieros son útiles.

---

## 📋 Resumen de Variables por Tipo

### Variables Numéricas Continuas (con outliers):

- `PERSONAL_MONTHLY_INCOME` ⚠️ (2% outliers)
- `OTHER_INCOMES` ⚠️ (0.92% outliers)
- `PERSONAL_ASSETS_VALUE` ⚠️ (0.96% outliers)
- `AGE` ⚠️ (0.88% outliers)
- `MONTHS_IN_RESIDENCE` ⚠️ (0.85% outliers, algunos missing)
- `MONTHS_IN_THE_JOB` ⚠️ (0.19% outliers)

### Variables Numéricas Discretas:

- `PAYMENT_DAY` (1, 5, 10, 15, 20, 25)
- `QUANT_DEPENDANTS` ⚠️ (0.61% outliers)
- `QUANT_BANKING_ACCOUNTS` (0, 1, 2)
- `QUANT_SPECIAL_BANKING_ACCOUNTS` (0, 1, 2)
- `QUANT_CARS`
- `QUANT_ADDITIONAL_CARDS` (1, 2, NULL) ⚠️ (constante - todos 0 - removida)

### Variables Categóricas Binarias (Y/N):

- `FLAG_RESIDENCIAL_PHONE` (Y/N)
- `FLAG_MOBILE_PHONE` (Y/N) ⚠️ (constante - todos "N" - removida)
- `COMPANY` (Y/N)
- `FLAG_PROFESSIONAL_PHONE` (Y/N)
- `FLAG_ACSP_RECORD` (Y/N) ⚠️ (constante - todos "N" - removida)

### Variables Categóricas Binarias (0/1):

- `FLAG_VISA` (0/1)
- `FLAG_MASTERCARD` (0/1)
- `FLAG_DINERS` (0/1)
- `FLAG_AMERICAN_EXPRESS` (0/1)
- `FLAG_OTHER_CARDS` (0, 1, NULL)
- `FLAG_EMAIL` (0/1)
- `FLAG_HOME_ADDRESS_DOCUMENT` (0/1)
- `FLAG_RG` (0/1)
- `FLAG_CPF` (0/1)
- `FLAG_INCOME_PROOF` (0/1)

### Variables Categóricas con Baja Cardinalidad:

- `SEX` (M, F)
- `APPLICATION_SUBMISSION_TYPE` (Web, Carga)
- `EDUCATION_LEVEL` (1, 2, 3, 4, 5)
- `MARITAL_STATUS` (1, 2, 3, 4, 5, 6, 7) ⚠️ (0.45% outliers)
- `RESIDENCE_TYPE` (1, 2, 3, 4, 5, NULL)
- `OCCUPATION_TYPE` (1, 2, 3, 4, 5, NULL)
- `PRODUCT` (1, 2, 7)
- `NACIONALITY` (0, 1, 2)
- `POSTAL_ADDRESS_TYPE` (1, 2)

### Variables Categóricas con Alta Cardinalidad:

- `RESIDENCIAL_STATE` - Estados brasileños
- `RESIDENCIAL_CITY` - Muchas ciudades
- `RESIDENCIAL_BOROUGH` - Muchos barrios
- `CITY_OF_BIRTH` - Muchas ciudades
- `STATE_OF_BIRTH` - Estados brasileños
- `PROFESSIONAL_STATE` - Estados brasileños
- `PROFESSIONAL_CITY` ⚠️ - Muchas ciudades + muchos missing
- `PROFESSIONAL_BOROUGH` ⚠️ - Muchos barrios + muchos missing
- `RESIDENCIAL_PHONE_AREA_CODE` - Códigos de área
- `PROFESSIONAL_PHONE_AREA_CODE` - Códigos de área
- `RESIDENCIAL_ZIP_3` - Códigos postales
- `PROFESSIONAL_ZIP_3` - Códigos postales
- `PROFESSION_CODE` - Muchos códigos ⚠️ (0.85% outliers)
- `MATE_PROFESSION_CODE` - Muchos códigos + muchos missing ⚠️

---

## ⚠️ Consideraciones Críticas para Preprocessing

### 1. **Columnas Constantes:**

- **Remover inmediatamente** antes de cualquier procesamiento
- Identificar automáticamente: `nunique() == 1` o `std() == 0`
- Guardar lista para aplicar en test

### 2. **Outliers:**

- **Prioridad alta:** `PERSONAL_MONTHLY_INCOME` (2% outliers)
- Usar **capping con IQR** o **Winsorization** (percentiles 1%-99%)
- Considerar transformaciones log para variables financieras sesgadas

### 3. **Missing Values:**

- **Variables profesionales:** Missing puede ser informativo (no tiene trabajo formal)
- **6 indicadores de missing creados** para variables importantes:

  - `MISSING_PROFESSION_CODE`
  - `MISSING_MONTHS_IN_RESIDENCE`
  - `MISSING_MATE_PROFESSION_CODE`
  - `MISSING_MATE_EDUCATION_LEVEL`
  - `MISSING_RESIDENCE_TYPE`
  - `MISSING_OCCUPATION_TYPE`

  **Nota:** `MISSING_PROFESSIONAL_CITY` y `MISSING_PROFESSIONAL_BOROUGH` NO se crean porque estas columnas fueron removidas en el Paso 1 (alta cardinalidad + muchos missing).

- Imputar con mediana (numéricas) / moda (categóricas) según tipo

### 4. **Alta Cardinalidad:**

- Variables geográficas tienen muchas categorías
- Considerar:
  - Agrupar categorías poco frecuentes
  - Target Encoding (si está disponible)
  - Usar ZIP codes para agrupar

### 5. **Feature Engineering Priorizado:**

- **Ratios financieros** (ej: INCOME_RATIO) mostraron ser útiles
- **Estabilidad** (meses en trabajo/residencia)
- **Agregaciones** (total de tarjetas, documentos, etc.)

---

## 📝 Notas Adicionales

1. ✅ **`CLERK_TYPE`** confirmado como constante (todos "C") - removida automáticamente
2. ⚠️ **Nota importante:** `FLAG_ACSP_RECORD` está en la lista de columnas constantes en el código actual (todos valores "N"). Si el dataset completo realmente tiene esta variable como constante, entonces es correcto removerla. Sin embargo, según el EDA original, esta variable era muy importante para riesgo crediticio, por lo que debería verificarse que efectivamente sea constante en todos los datos.
3. Variables de **estabilidad** (meses en trabajo/residencia) son importantes y se usan en feature engineering (YEARS_IN_RESIDENCE, YEARS_IN_JOB, STABILITY_SCORE)
4. **Missing en variables importantes** se usa como feature informativa - se crean 6 indicadores de missing para estas variables antes de imputar

---

**Próximo paso:** Usar esta información para refinar el pipeline de preprocessing.
