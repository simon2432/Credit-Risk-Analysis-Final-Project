# 🐳 Docker - Guía Rápida

## 📋 Estructura de Servicios

El proyecto usa **Docker Compose** con 3 servicios (arquitectura de microservicios):

1. **Model** (`credit-risk-model`): Contenedor de datos

   - Contiene el modelo entrenado (`model.joblib`)
   - Contiene el preprocesador (`preprocessor.joblib`)
   - Mantiene los archivos disponibles para otros servicios
   - No expone puertos (solo volúmenes compartidos)

2. **API** (`credit-risk-api`): Servicio FastAPI en puerto 8000

   - Expone endpoints para predicción
   - Carga modelos desde el servicio `model` (volumen compartido)
   - Health check automático
   - Depende del servicio `model`

3. **UI** (`credit-risk-ui`): Servicio Streamlit en puerto 8501
   - Interfaz web para evaluación de crédito
   - Se comunica con el servicio API
   - Espera a que la API esté saludable antes de iniciar

**Volúmenes compartidos:**

- `./models` → `/app/models:ro` (solo lectura)
  - Modelo: `models/production/model.joblib`
  - Preprocesador: `models/preprocessor/preprocessor.joblib`
  - Compartido entre: `model` y `api`
- `./data/raw` → `/app/data/raw:ro` (solo lectura, solo en API)
  - Datos de entrenamiento si se necesitan en runtime

---

## 🚀 Levantar los servicios

### Primera vez (construir imágenes):

```bash
docker-compose up --build
```

### Siguientes veces (más rápido, sin reconstruir):

```bash
docker-compose up
```

### En segundo plano (terminal libre):

```bash
docker-compose up -d
```

---

## 💻 Manejar la terminal

### Cuando usas `docker-compose up` (sin `-d`):

- **Ves los logs en tiempo real** de los 3 servicios
- **Para detener**: Presiona `CTRL+C`
- **Si se congela**: Presiona `CTRL+C` varias veces

### Ver logs después (si usaste `-d`):

```bash
docker-compose logs -f          # Todos los servicios
docker-compose logs -f model    # Solo Model
docker-compose logs -f api       # Solo API
docker-compose logs -f ui        # Solo UI
```

(Presiona `CTRL+C` para salir de los logs)

---

## 🛑 Detener

### Si está corriendo en la terminal:

Presiona `CTRL+C`

### Si está en segundo plano (`-d`):

```bash
docker-compose down
```

### Detener y eliminar volúmenes (⚠️ cuidado, borra datos):

```bash
docker-compose down -v
```

---

## 🌐 Acceder

- **UI (Streamlit)**: http://localhost:8501
- **API Health Check**: http://localhost:8000/health
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

---

## ⚡ Comandos útiles

```bash
# Ver qué está corriendo
docker-compose ps

# Reiniciar un servicio
docker-compose restart model
docker-compose restart api
docker-compose restart ui

# Ver logs de un servicio
docker-compose logs -f model
docker-compose logs -f api
docker-compose logs -f ui

# Entrar a un contenedor (para debugging)
docker-compose exec model bash
docker-compose exec api bash
docker-compose exec ui bash

# Reconstruir solo un servicio
docker-compose build api
docker-compose up -d api

# Ver estado de health checks
docker-compose ps
```

---

## 🔧 Troubleshooting

### Error: "Cannot connect to Docker daemon"

**Solución**: Abre Docker Desktop en Windows

### Error: "Port already in use"

**Solución**:

```bash
# Ver qué está usando el puerto (Windows PowerShell)
netstat -ano | findstr :8000
netstat -ano | findstr :8501

# O cambiar los puertos en docker-compose.yml
```

### Los modelos no se cargan

**Verificar**:

1. El servicio `model` está corriendo: `docker-compose ps`
2. Los archivos están en `./models/` en tu máquina local
3. Los archivos tienen nombres correctos: `model.joblib`, `preprocessor.joblib`
4. El preprocesador está en `./data/processed/` si se guardó ahí
5. Revisar logs: `docker-compose logs model` y `docker-compose logs api`
6. Verificar que los volúmenes están montados correctamente: `docker-compose exec model ls -la /app/models`

### La UI no se conecta a la API

**Verificar**:

1. La API está saludable: http://localhost:8000/health
2. Variable de entorno `API_URL=http://api:8000` está configurada
3. Revisar logs: `docker-compose logs ui`

### Limpiar todo y empezar de nuevo

```bash
# Detener y eliminar contenedores, imágenes y volúmenes
docker-compose down -v --rmi all

# Reconstruir desde cero
docker-compose up --build
```

---

## ⚠️ Nota importante

**Antes de levantar Docker, asegúrate de que Docker Desktop esté corriendo en Windows.**

Si ves el error `The system cannot find the file specified` → Abre Docker Desktop primero.
