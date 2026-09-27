# Gemelo Digital de Evacuación Minera

Sistema de investigación: gemelo digital con intervención humana para la
evacuación de emergencia en minas subterráneas. Construido por fases,
verificado en cada una (tests automatizados + validación visual real con
Playwright + servidores reales corriendo).

## Qué incluye

| Capa | Estado | Ubicación |
|---|---|---|
| Motor IA (WESAD, EDA, entrenamiento, CV, tuning, Model Registry, estadística, reportes PDF) | ✅ Completo y probado | `backend/training/`, `backend/evaluation/`, `backend/statistics/`, `backend/reports/` |
| Gemelo digital + ABM (grafo, Mesa, propagación de peligro, enrutamiento) | ✅ Completo y probado | `backend/digital_twin/`, `backend/simulation/` |
| Backend de producción FastAPI (WebSocket + REST) | ✅ Completo y probado | `backend/api/`, `backend/services/` |
| Frontend Next.js + React Three Fiber | ✅ Completo, compilado y validado con navegador real | `frontend/` |
| Cliente VR Unity | ❌ Fuera de alcance (decisión conjunta, ver hilo de la conversación) | — |

**64/64 tests automatizados del backend en verde.** El frontend fue
compilado en producción (`npm run build`, cero errores TypeScript) y
**verificado con un navegador real (Playwright + Chromium) contra un
backend real corriendo**: creación de sesión, WebSocket en tiempo real
(streaming continuo confirmado paso a paso), panel de estrés en vivo,
propagación de peligro visible en el grafo 3D, panel de Motor IA con EDA
real y modelos reales, historial de sesiones, descarga de reportes PDF,
tema claro/oscuro e i18n — todo con cero errores de consola.

## Arquitectura y principio rector

El entrenamiento vive en el backend (`training/`). La inferencia y el
gemelo digital se sirven desde el backend (`api/`). El frontend **solo
consume la API FastAPI** — no entrena, no simula, no accede al sistema de
archivos. Ver `backend/utils/config.py` para toda la configuración
centralizada (nada hardcodeado).

## Arrancar el backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages
cp ../.env.example ../.env   # ajustar si hace falta
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
# (ejecutar desde la RAÍZ del proyecto, no desde dentro de backend/,
#  por los imports absolutos `backend.xxx`)
```

Desde la raíz del proyecto:
```bash
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

Documentación interactiva: `http://localhost:8000/docs`

### Streamlit (EDA / calibración — independiente de producción)
```bash
streamlit run backend/training/streamlit_app.py
```

### Tests
```bash
python3 -m pytest backend/tests/ -v
```

## Arrancar el frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000
```

Si el backend corre en un host/puerto distinto del default
(`http://localhost:8000`), configura en `frontend/.env.local`:
```
NEXT_PUBLIC_API_BASE_URL=http://tu-host:puerto
NEXT_PUBLIC_WS_BASE_URL=ws://tu-host:puerto
```
y en el backend (`.env`), añade el origen del frontend a `CORS_ORIGINS`
(por defecto solo permite `http://localhost:3000`, el puerto estándar de
Next.js — si usas otro puerto, esto es lo primero a revisar).

## Activar el dataset real WESAD

Este entorno de desarrollo no tuvo acceso de red al servidor de WESAD, así
que el pipeline se validó con un dataset **sintético** con el mismo
esquema exacto de canales (`wesad_synthetic_demo`, siempre marcado
explícitamente como tal — nunca usado para cifras reales). Para activar el
dataset real:

1. Descarga el dataset (~2 GB) desde `https://uni-siegen.sciebo.de/s/HGdUkoNlTOJOtwZ/download`
   (o busca "WESAD dataset Schmidt" en Kaggle como espejo alternativo).
2. Coloca el `.zip` en `backend/data/raw/WESAD.zip`, o extrae su
   contenido directamente en `backend/data/raw/WESAD/` (debe quedar
   `WESAD/S2/S2.pkl`, `WESAD/S3/S3.pkl`, etc.).
3. Desde la raíz del proyecto:
   ```bash
   python3 -m backend.training.download_wesad
   ```
4. `settings.DATASETS` ya es `["wesad"]` por defecto — el pipeline entero
   (EDA, entrenamiento, CV) funciona igual, sin cambiar código, gracias a
   la interfaz `DatasetLoader`.

## Metodología de desarrollo seguida

El sistema se construyó fase por fase (ver historial de la conversación
para la justificación técnica de cada decisión), verificando cada fase
antes de avanzar:

1. Estructura base + configuración
2. Grafo del gemelo digital + generador de layouts
3. Motor ABM (Mesa): agentes, pánico, propagación de peligro
4. Enrutamiento dinámico: A* adaptativo vs. Q-learning
5. Motor IA — WESAD (real + sintético), preprocesamiento, EDA
6. Motor IA — arquitecturas, CV (LOSO), tuning (Optuna), Model Registry
7. Evaluación — Monte Carlo, Wilcoxon/Friedman/Nemenyi
8. Reportes PDF automáticos
9. API FastAPI — WebSocket tiempo real + REST completo
10. Frontend Next.js + React Three Fiber

**Bugs reales encontrados y corregidos durante el desarrollo** (no solo
código que "se ve bien" — cada uno se detectó ejecutando el sistema de
verdad):
- Velocidad de propagación de incendio mal calibrada (bloqueaba toda la
  mina en 46 pasos, sin dar tiempo a evacuar).
- Lógica de decisión de refugio ambigua (agentes quedaban flotando sin
  avanzar, contados incorrectamente como "perdidos").
- **Bug crítico de aislamiento de estado**: `DigitalTwinState` no clonaba
  el layout, así que instancias distintas sobre el mismo grafo compartían
  objetos mutables — corrompía silenciosamente el entrenamiento offline
  del Q-learning.
- Colisión de nombres (`cm` como matriz de confusión vs. unidad de
  centímetros de reportlab) que rompía la generación de imágenes en los
  PDFs.
- Snapshot inicial de sesión sin agentes hasta el primer `step()`.
- CORS bloqueando silenciosamente el frontend cuando corre en un puerto
  distinto al configurado.

## Limitaciones conocidas

- **Sin cliente VR/Unity** (decisión conjunta): el control humano y la
  intervención se hacen desde el propio frontend (clic para seguir a un
  agente, panel de estrés simulado), no desde un headset.
- El dataset primario de investigación (WESAD) requiere descarga manual
  fuera de este entorno (ver arriba).
- Next.js está en la última versión parche de la rama 14.x
  (`14.2.35`); persiste un CVE crítico (`CVE-2025-59471`) que solo aplica
  a despliegues con `images.remotePatterns` configurado — **este proyecto
  no lo usa**, por lo que no es explotable aquí, pero si en el futuro se
  habilita optimización de imágenes remotas, debe migrarse a Next.js 15/16.
