# 🐳 Docker - Guía Rápida

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
- **Ves los logs en tiempo real** de ambos servicios
- **Para detener**: Presiona `CTRL+C`
- **Si se congela**: Presiona `CTRL+C` varias veces

### Ver logs después (si usaste `-d`):
```bash
docker-compose logs -f          # Ambos servicios
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

---

## 🌐 Acceder

- **UI**: http://localhost:8501
- **API**: http://localhost:8000/health

---

## ⚡ Comandos útiles

```bash
# Ver qué está corriendo
docker-compose ps

# Reiniciar un servicio
docker-compose restart api

# Ver logs de un servicio
docker-compose logs -f ui
```

---

## ⚠️ Nota importante

**Antes de levantar Docker, asegúrate de que Docker Desktop esté corriendo en Windows.**

Si ves el error `The system cannot find the file specified` → Abre Docker Desktop primero.

