# Implemented Preprocessing - Complete Documentation

## Objective

This document describes **exactly how the preprocessing pipeline is implemented** in the current code (`src/preprocessing.py`).

The pipeline processes data in **7 sequential steps**, transforming original columns into normalized final features ready for the model. It uses a `sklearn.Pipeline` with custom transformers and a `ColumnTransformer` to handle numeric and categorical features separately.

---

## Dataset Structure (54 columns)

### Variables con Descripción Completa:

#### **Identificadores:**

- **`ID_CLIENT`** (Var_Id: 1)
  - **Descripción:** Número secuencial para el solicitante (usar como clave)
  - **Valores:** 1-50000 (train), 50001-70000 (test), 70001-90000 (prediction)
  - **Acción:** Remover antes del preprocessing

#### **Variables de Aplicación:**

- **`CLERK_TYPE`** (Var_Id: 2)

  - **Descripción:** Tipo de empleado/clerk (no informado)
  - **Valores:** C
  - **Tipo:** Categórica

- **`PAYMENT_DAY`** (Var_Id: 3)

  - **Descripción:** Día del mes elegido por el solicitante para el pago de la factura
  - **Valores:** 1, 5, 10, 15, 20, 25
  - **Tipo:** Numérica discreta

- **`APPLICATION_SUBMISSION_TYPE`** (Var_Id: 4)

  - **Descripción:** Indica si la aplicación fue enviada vía internet o en persona/por correo
  - **Valores:** Web, Carga
  - **Tipo:** Categórica binaria

- **`QUANT_ADDITIONAL_CARDS`** (Var_Id: 5)

  - **Descripción:** Cantidad de tarjetas adicionales solicitadas en el mismo formulario
  - **Valores:** 1, 2, NULL
  - **Tipo:** Numérica discreta

- **`POSTAL_ADDRESS_TYPE`** (Var_Id: 6)
  - **Descripción:** Indica si la dirección postal es la del hogar u otra. Encoding no informado
  - **Valores:** 1, 2
  - **Tipo:** Numérica/categórica

#### **Variables Demográficas:**

- **`SEX`** (Var_Id: 7)

  - **Descripción:** Sexo del solicitante
  - **Valores:** M=Male, F=Female
  - **Tipo:** Categórica binaria

- **`MARITAL_STATUS`** (Var_Id: 8)

  - **Descripción:** Estado civil. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, 6, 7
  - **Tipo:** Numérica/categórica ordinal

- **`QUANT_DEPENDANTS`** (Var_Id: 9)

  - **Descripción:** Cantidad de dependientes
  - **Valores:** 0, 1, 2, ...
  - **Tipo:** Numérica discreta

- **`EDUCATION_LEVEL`** (Var_Id: 10)

  - **Descripción:** Nivel educativo en orden gradual. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5
  - **Tipo:** Numérica/categórica ordinal

- **`STATE_OF_BIRTH`** (Var_Id: 11)

  - **Descripción:** Estado de nacimiento
  - **Valores:** Estados brasileños, XX, missing
  - **Tipo:** Categórica

- **`CITY_OF_BIRTH`** (Var_Id: 12)

  - **Descripción:** Ciudad de nacimiento
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`NACIONALITY`** (Var_Id: 13)
  - **Descripción:** País de nacimiento. Encoding no informado pero Brasil probablemente es 1
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica/categórica

#### **Variables de Residencia:**

- **`RESIDENCIAL_STATE`** (Var_Id: 14)

  - **Descripción:** Estado de residencia
  - **Valores:** Estados brasileños
  - **Tipo:** Categórica

- **`RESIDENCIAL_CITY`** (Var_Id: 15)

  - **Descripción:** Ciudad de residencia
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`RESIDENCIAL_BOROUGH`** (Var_Id: 16)

  - **Descripción:** Barrio de residencia
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad)

- **`FLAG_RESIDENCIAL_PHONE`** (Var_Id: 17)

  - **Descripción:** Indica si el solicitante posee teléfono residencial
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`RESIDENCIAL_PHONE_AREA_CODE`** (Var_Id: 18)

  - **Descripción:** Código de área de tres dígitos (pseudo-código)
  - **Valores:** Códigos de área
  - **Tipo:** Categórica

- **`RESIDENCE_TYPE`** (Var_Id: 19)

  - **Descripción:** Tipo de residencia. Encoding no informado. Generalmente: propia, hipoteca, alquilada, padres, familia, etc.
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica

- **`MONTHS_IN_RESIDENCE`** (Var_Id: 20)

  - **Descripción:** Tiempo en la residencia actual en meses
  - **Valores:** 1, 2, ..., NULL
  - **Tipo:** Numérica continua

- **`RESIDENCIAL_ZIP_3`** (Var_Id: 52)
  - **Descripción:** Tres dígitos más significativos del código postal real del hogar
  - **Valores:** Códigos postales
  - **Tipo:** Numérica/categórica

#### **Variables Financieras:**

- **`PERSONAL_MONTHLY_INCOME`** (Var_Id: 23)

  - **Descripción:** Ingreso mensual regular personal del solicitante en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Variable crítica, puede tener outliers

- **`OTHER_INCOMES`** (Var_Id: 24)

  - **Descripción:** Otros ingresos del solicitante promediados mensualmente en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua

- **`PERSONAL_ASSETS_VALUE`** (Var_Id: 32)

  - **Descripción:** Valor total de posesiones personales como casas, autos, etc. en moneda brasileña (R$)
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Puede tener outliers extremos

- **`QUANT_BANKING_ACCOUNTS`** (Var_Id: 30)

  - **Descripción:** Cantidad de cuentas bancarias
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica discreta

- **`QUANT_SPECIAL_BANKING_ACCOUNTS`** (Var_Id: 31)
  - **Descripción:** Cantidad de cuentas bancarias especiales
  - **Valores:** 0, 1, 2
  - **Tipo:** Numérica discreta

#### **Variables de Tarjetas:**

- **`FLAG_VISA`** (Var_Id: 25)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta VISA
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_MASTERCARD`** (Var_Id: 26)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta MASTERCARD
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_DINERS`** (Var_Id: 27)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta DINERS
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_AMERICAN_EXPRESS`** (Var_Id: 28)

  - **Descripción:** Flag indicando si el solicitante es titular de tarjeta AMERICAN EXPRESS
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_OTHER_CARDS`** (Var_Id: 29)
  - **Descripción:** A pesar de ser "FLAG", este campo presenta tres valores no explicados
  - **Valores:** 0, 1, NULL
  - **Tipo:** Numérica/categórica

#### **Variables de Empleo:**

- **`COMPANY`** (Var_Id: 34)

  - **Descripción:** Si el solicitante ha proporcionado el nombre de la compañía donde trabaja formalmente
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`PROFESSIONAL_STATE`** (Var_Id: 35)

  - **Descripción:** Estado donde trabaja el solicitante
  - **Valores:** Estados brasileños
  - **Tipo:** Categórica

- **`PROFESSIONAL_CITY`** (Var_Id: 36)

  - **Descripción:** Ciudad donde trabaja el solicitante
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad, muchos missing)

- **`PROFESSIONAL_BOROUGH`** (Var_Id: 37)

  - **Descripción:** Barrio donde trabaja el solicitante
  - **Valores:** Varios
  - **Tipo:** Categórica (alta cardinalidad, muchos missing)

- **`FLAG_PROFESSIONAL_PHONE`** (Var_Id: 38)

  - **Descripción:** Indica si se proporcionó el número de teléfono profesional
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`PROFESSIONAL_PHONE_AREA_CODE`** (Var_Id: 39)

  - **Descripción:** Código de área de tres dígitos (pseudo-código)
  - **Valores:** Códigos de área
  - **Tipo:** Categórica

- **`MONTHS_IN_THE_JOB`** (Var_Id: 40)

  - **Descripción:** Tiempo en el trabajo actual en meses
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua

- **`PROFESSION_CODE`** (Var_Id: 41)

  - **Descripción:** Código de profesión del solicitante. Encoding no informado
  - **Valores:** 1, 2, 3, ...
  - **Tipo:** Numérica/categórica

- **`OCCUPATION_TYPE`** (Var_Id: 42)

  - **Descripción:** Tipo de ocupación. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica

- **`MATE_PROFESSION_CODE`** (Var_Id: 43)

  - **Descripción:** Código de profesión del cónyuge. Encoding no informado
  - **Valores:** 1, 2, 3, ..., NULL
  - **Tipo:** Numérica/categórica (muchos missing)

- **`PROFESSIONAL_ZIP_3`** (Var_Id: 53)
  - **Descripción:** Tres dígitos más significativos del código postal real del trabajo
  - **Valores:** Códigos postales
  - **Tipo:** Numérica/categórica

#### **Variables de Contacto:**

- **`FLAG_MOBILE_PHONE`** (Var_Id: 21)

  - **Descripción:** Indica si el solicitante posee teléfono móvil
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria

- **`FLAG_EMAIL`** (Var_Id: 22)
  - **Descripción:** Indica si el solicitante posee dirección de email
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

#### **Variables de Documentación:**

- **`FLAG_HOME_ADDRESS_DOCUMENT`** (Var_Id: 45)

  - **Descripción:** Flag indicando confirmación documental de dirección del hogar
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_RG`** (Var_Id: 46)

  - **Descripción:** Flag indicando confirmación documental del número de cédula de ciudadanía
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_CPF`** (Var_Id: 47)

  - **Descripción:** Flag indicando confirmación documental del estado de contribuyente
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

- **`FLAG_INCOME_PROOF`** (Var_Id: 48)
  - **Descripción:** Flag indicando confirmación documental de ingresos
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria

#### **Otras Variables:**

- **`QUANT_CARS`** (Var_Id: 33)

  - **Descripción:** Cantidad de autos que posee el solicitante
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica discreta

- **`MATE_EDUCATION_LEVEL`** (Var_Id: 44)

  - **Descripción:** Nivel educativo del cónyuge en orden gradual. Encoding no informado
  - **Valores:** 1, 2, 3, 4, 5, NULL
  - **Tipo:** Numérica/categórica ordinal (muchos missing)

- **`PRODUCT`** (Var_Id: 49)

  - **Descripción:** Tipo de producto de crédito solicitado. Encoding no informado
  - **Valores:** 1, 2, 7
  - **Tipo:** Numérica/categórica

- **`FLAG_ACSP_RECORD`** (Var_Id: 50)

  - **Descripción:** Flag indicando si el solicitante tiene algún registro previo de morosidad crediticia
  - **Valores:** Y, N
  - **Tipo:** Categórica binaria
  - **Nota:** Variable muy importante para riesgo crediticio

- **`AGE`** (Var_Id: 51)
  - **Descripción:** Edad del solicitante al momento de la solicitud
  - **Valores:** Valores numéricos
  - **Tipo:** Numérica continua
  - **Nota:** Variable importante, puede tener outliers (edades muy altas o muy bajas)

#### **Target:**

- **`TARGET_LABEL_BAD=1`** (Var_Id: 54)
  - **Descripción:** Variable objetivo: BAD=1 (default), GOOD=0 (no default)
  - **Valores:** 0, 1
  - **Tipo:** Numérica binaria
  - **Distribución:** ~74% NO (0), ~26% YES (1) - **Desbalanceado**

---

## Implemented Feature Engineering

The pipeline creates **11 new features** grouped into 5 categories:

### 1. **Financial Features** (3 features)

- **INCOME_TOTAL**: Sum of personal and other monthly income
- **INCOME_RATIO**: Ratio of other income to main income (uses a very small value to avoid division by zero)
- **HAS_OTHER_INCOME**: Binary flag indicating if the person has other income besides the main one

### 2. **Stability Features** (3 features)

- **YEARS_IN_RESIDENCE**: Conversion of months in residence to years (divides by 12)
- **YEARS_IN_JOB**: Conversion of months in job to years (divides by 12)
- **STABILITY_SCORE**: Sum of years in residence and years in job (indicates overall applicant stability)

### 3. **Card Features** (2 features)

- **CARDS_COUNT**: Total credit cards owned (sum of all card flags: Visa, Mastercard, Diners, American Express, Others)
- **HAS_ANY_CARD**: Binary flag indicating if they have at least one credit card

### 4. **Documentation Features** (1 feature)

- **DOCS_COUNT**: Total documents provided (sum of document flags: address, RG, CPF, income proof)

### 5. **Cyclical Features** (2 features)

- **PAYMENT_DAY_SIN**: Cyclical encoding of payment day using sine
- **PAYMENT_DAY_COS**: Cyclical encoding of payment day using cosine

**Reason:** Cyclical encoding allows the model to understand that day 31 and day 1 are temporally close, which is important for capturing temporal patterns.

### 6. **Missing Values Features (Indicators)** (6 features)

Binary features (0/1) are created that indicate if an important variable has missing values:

- **MISSING_PROFESSION_CODE**: Indicates if profession code is missing
- **MISSING_MONTHS_IN_RESIDENCE**: Indicates if time in residence is missing
- **MISSING_MATE_PROFESSION_CODE**: Indicates if spouse profession code is missing
- **MISSING_MATE_EDUCATION_LEVEL**: Indicates if spouse education level is missing
- **MISSING_RESIDENCE_TYPE**: Indicates if residence type is missing
- **MISSING_OCCUPATION_TYPE**: Indicates if occupation type is missing

**Reason:** Missing can be informative. For example, if `PROFESSION_CODE` is missing, it may indicate that the person does not have formal employment, which is relevant for credit risk.

**Total features created:** 11 new features + 6 missing indicators = **17 new columns**

---

## Preprocessing Pipeline - Current Implementation

The pipeline executes in **7 sequential steps** using a `sklearn.Pipeline` with custom transformers:

### **Step 1: NULL String Cleaning**

**What is done:** Converts placeholders like "NULL", "NULL.1", "NULL.2", etc. and empty strings to real NaN values.

**How it is done:** All text columns are identified and patterns matching "NULL" (with or without numeric suffixes) are searched. These values are converted to NaN, as well as empty strings.

**Why it is done:** The original dataset uses "NULL" as a placeholder for missing values, but for processing we need real NaN values that can be correctly handled by imputers.

---

### **Step 2: Column Removal**

**What is done:** Columns that are not useful for the model are removed.

**Removed columns:**

- **ID_CLIENT**: Unique identifier, does not provide information to predict risk
- **CITY_OF_BIRTH**: High cardinality (9,910 different categories)
- **RESIDENCIAL_CITY**: High cardinality (3,529 categories)
- **RESIDENCIAL_BOROUGH**: High cardinality (14,511 categories)
- **PROFESSIONAL_CITY**: High cardinality (2,236 categories) and many missing values (67.6%)
- **PROFESSIONAL_BOROUGH**: High cardinality (5,057 categories) and many missing values (67.6%)

**Why it is done:** High cardinality geographic columns do not provide useful information and create dimensionality problems. Lower cardinality columns like states or postal codes are preferred as they are more informative and manageable.

---

### **Step 3: Constant Column Removal**

**What is done:** Columns that have all equal values (no variance) are automatically detected and removed.

**How it is done:** The number of unique values in each column is checked. If a column has only one unique value (or all are NaN), it is removed.

**Typically removed columns:**

- **CLERK_TYPE**: All values are "C"
- **QUANT_ADDITIONAL_CARDS**: All values are 0 or 1
- **EDUCATION_LEVEL**: All values are 0 or 1
- **FLAG_MOBILE_PHONE**: All values are "N"
- **FLAG_HOME_ADDRESS_DOCUMENT**: All values are 0
- **FLAG_RG**: All values are 0
- **FLAG_CPF**: All values are 0
- **FLAG_INCOME_PROOF**: All values are 0
- **FLAG_ACSP_RECORD**: All values are "N"

**Why it is done:** Constant columns do not provide information to the model because there is no variability that the model can learn. Additionally, they can cause numerical problems in some algorithms.

---

### **Step 4: Missing Indicators**

**What is done:** New binary features (0 or 1) are created that indicate if an important variable has missing values.

**Variables with created indicators:**

- **MISSING_PROFESSION_CODE**: Indicates if profession code is missing
- **MISSING_MONTHS_IN_RESIDENCE**: Indicates if time in residence is missing
- **MISSING_MATE_PROFESSION_CODE**: Indicates if spouse profession code is missing
- **MISSING_MATE_EDUCATION_LEVEL**: Indicates if spouse education level is missing
- **MISSING_RESIDENCE_TYPE**: Indicates if residence type is missing
- **MISSING_OCCUPATION_TYPE**: Indicates if occupation type is missing

**How it is done:** For each selected variable, a new column is created that equals 1 if the original variable has NaN, and 0 if it has a value.

**Why it is done:** Missing can be informative. For example, if `PROFESSION_CODE` is missing, it may indicate that the person does not have formal employment, which is very relevant for evaluating credit risk. These indicators allow the model to learn from the absence of information.

---

### **Step 5: Winsorization (Outlier Capping)**

**What is done:** Extreme values (outliers) are limited using the 1% and 99% percentiles calculated on the training set.

**Variables to which winsorization is applied:**

- **PERSONAL_MONTHLY_INCOME**: Personal income
- **OTHER_INCOMES**: Other income
- **PERSONAL_ASSETS_VALUE**: Asset value
- **AGE**: Age
- **MONTHS_IN_RESIDENCE**: Months in residence
- **MONTHS_IN_THE_JOB**: Months in job
- **PROFESSION_CODE**: Profession code
- **MATE_PROFESSION_CODE**: Spouse profession code
- **MARITAL_STATUS**: Marital status
- **QUANT_DEPENDANTS**: Number of dependents

**How it is done:**

1. During training, the 1% and 99% percentiles of each variable are calculated on the training set
2. During transformation, any value below the 1% percentile is replaced by that percentile, and any value above the 99% percentile is replaced by that percentile

**Why it is done:** Extreme values can negatively affect model training. Winsorization preserves information from normal values but limits the impact of anomalous values that could be data errors or very rare cases.

---

### **Step 6: Feature Engineering**

**What is done:** New derived features are created based on insights from exploratory data analysis (EDA).

**Created features (11 new):**

1. **INCOME_TOTAL**: Sum of personal and other income
2. **INCOME_RATIO**: Ratio of other income to main income
3. **HAS_OTHER_INCOME**: Binary flag indicating if they have other income
4. **YEARS_IN_RESIDENCE**: Conversion of months in residence to years
5. **YEARS_IN_JOB**: Conversion of months in job to years
6. **STABILITY_SCORE**: Sum of years in residence and years in job (indicates stability)
7. **CARDS_COUNT**: Total credit cards owned
8. **HAS_ANY_CARD**: Binary flag indicating if they have at least one card
9. **DOCS_COUNT**: Total documents provided
10. **PAYMENT_DAY_SIN**: Cyclical encoding of payment day using sine
11. **PAYMENT_DAY_COS**: Cyclical encoding of payment day using cosine

**Why it is done:** These derived features capture important relationships between variables that the model may not easily learn with original variables. For example, income ratio or overall stability are more informative concepts than individual variables.

---

### **Step 7: Imputation, Encoding and Scaling**

**What is done:** Numeric and categorical columns are processed separately, applying missing value imputation, encoding (conversion to numbers) and scaling.

#### **For Numeric Columns:**

1. **Imputation:** Missing values are replaced with the median calculated on the training set
2. **Scaling:** Values are normalized using StandardScaler (mean=0, standard deviation=1)

**Why:** Numeric columns need to have all values complete (no NaN) and be on a similar scale for models to work correctly.

#### **For Categorical Columns:**

1. **Imputation:** Missing values are replaced with the mode (most frequent value) calculated on the training set
2. **OneHot Encoding:** Categories are converted to binary columns (0 or 1)
   - Each unique category becomes a new column
   - Categories with frequency less than 1% are grouped as "infrequent"
   - If a new category appears in production that was not seen in training, it is handled as "infrequent"

**Why:** Machine learning models need numbers, not text. OneHot Encoding is the most common strategy to convert categories to numbers and works well with tree-based models.

**Final result:** The number of final features depends on how many unique categories there are in each categorical column. Typically 100-200 final features are generated ready for the model.

---

## Transformation Summary

### **Column Transformation:**

```
Original columns (53)
    ↓ Step 1: CleanNullLikeStrings
    - Converts "NULL", "NULL.1", etc. to NaN
    - Does not change number of columns
    ↓ Step 2: DropColumns
    - Removes ID_CLIENT + 5 high cardinality columns
    = ~47 columns
    ↓ Step 3: DropConstantColumns
    - Removes columns without variance (automatically detected)
    = ~38-42 columns (depends on detected constants)
    ↓ Step 4: MissingIndicatorAdder
    - Creates 6 missing indicators
    = ~44-48 columns
    ↓ Step 5: Winsorizer
    - Caps outliers using 1% and 99% percentiles
    - Does not change number of columns
    ↓ Step 6: FeatureEngineer
    - Creates 11 new features
    = ~55-59 columns
    ↓ Step 7: ColumnTransformer
    - Numeric: median imputation + StandardScaler
    - Categorical: most_frequent imputation + OneHotEncoder
    - OneHot expands categoricals (1 → multiple columns)
    = ~100-200 final features (depends on unique categories)
```

**Note:** The exact number of final features varies according to unique categories in each categorical column. OneHotEncoder creates one column per unique category (plus one column for "infrequent" if there are infrequent categories).

---

## Pipeline Saving

**File:** `models/preprocessor/preprocessor.joblib`

**What is saved:** The complete pipeline with all transformers and their parameters learned during training:

- Winsorization limits (1% and 99% percentiles of each variable)
- Detected and removed constant columns
- Medians for imputing missing values in numeric columns
- Modes for imputing missing values in categorical columns
- Learned categories for OneHotEncoder
- Scaling parameters (mean and standard deviation of each numeric variable)

**Typical size:** Between 1-5 MB (depends on the number of unique categories in categorical variables)

**Why it is important:** By saving the pipeline, we guarantee that in production exactly the same transformations used in training are applied, using the same learned parameters (medians, modes, limits, etc.).

---

## Important Considerations

1. **Step Order is Critical:**

   The order of steps is very important because each step depends on the previous ones:

   - First NULLs are cleaned to have real NaN values
   - Then non-useful columns are removed
   - Then missing indicators are created (before imputing)
   - Winsorization is applied before creating new features
   - New derived features are created
   - Finally missing values are imputed, categoricals are encoded and numerics are scaled

2. **Handling Unknown Values in Production:**

   If a new category appears in production that was not seen in training, it is automatically handled by grouping it as "infrequent". This allows the model to work correctly with new data.

3. **Winsorization:**

   Winsorization limits are calculated only once during training using percentiles from the training set. These same limits are reused in production to guarantee consistency.

4. **Informative Missing Values:**

   Missing indicators are created before imputing missing values. This allows the model to learn from the absence of information, which can be very relevant (for example, if profession code is missing it may indicate unemployment).

5. **Scaling:**

   Scaling normalizes all numeric variables to the same scale (mean=0, standard deviation=1). This is important for models to work correctly, especially those that use distances or regularization.

6. **Infrequent Category Grouping:**

   Categories that appear very few times (less than 1%) are grouped as "infrequent". This reduces the number of final features and improves the model's ability to generalize to new data.

---

## References

- **EDA Findings:** See `EDA_FINDINGS.md` for complete details
- **Implementation:** See `src/preprocessing.py` for complete code
- **Removed columns:** ID_CLIENT + 5 high cardinality columns (configurable)
- **Constant columns:** Automatically removed by `DropConstantColumns`
- **Outliers:** Winsorization applied using 1% and 99% percentiles (configurable)
- **Feature Engineering:** 11 new features implemented according to EDA findings
- **Encoding:** OneHotEncoder with infrequent category grouping (<1% frequency)

---

---

**Status:** Implemented and working. The pipeline is completely implemented in `src/preprocessing.py` and is saved in `models/preprocessor/preprocessor.joblib` after training.
