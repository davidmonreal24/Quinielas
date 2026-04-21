# Proyecto Quinielas — Pipeline, Base de Datos y Automatización

> Última actualización: 2026-04-17  
> Entorno: Windows 11 · Python 3.11 · venv local

---

## 1. Arquitectura general

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FUENTES EXTERNAS                            │
│  Sofascore API  │  The Odds API  │  API-Football  │  Altenar/Playdoit│
└────────┬────────┴───────┬────────┴───────┬────────┴────────┬────────┘
         │                │                │                 │
         ▼                ▼                ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE RECOLECCIÓN                            │
│   collect_sofascore.py        │        odds_client.py               │
│   (partidos + resultados)     │        (momios en tiempo real)      │
└─────────────────────┬─────────┴──────────────────┬─────────────────┘
                      │                             │
                      ▼                             ▼
         ┌────────────────────────┐    ┌────────────────────────┐
         │  data/sofascore_       │    │  data/ligamx_          │
         │  events.parquet        │    │  odds_cache.json        │
         └────────────┬───────────┘    └────────────┬───────────┘
                      │                             │
                      ▼                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE PREDICCIÓN                             │
│   predict_ligamx.py                                                 │
│     ├─ context_enricher.py  (Dixon-Coles, motivación, goleadores)   │
│     └─ lineup_watcher.py    (convocados via API-Football)           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
         ┌────────────────────────┐    ┌────────────────────────┐
         │  data/ligamx_          │    │  data/ligamx_          │
         │  predicciones.csv      │    │  metodologia.docx       │
         └────────────┬───────────┘    └────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CAPA DE PRESENTACIÓN                           │
│   api.py (FastAPI :8000)  ←→  dashboard.html / dashboard.py        │
│   _reporte_playdoit.py  (reporte terminal con valor esperado)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Scripts del proyecto

| Script | Propósito | Frecuencia recomendada |
|--------|-----------|------------------------|
| `collect_sofascore.py` | Descarga partidos+resultados de todas las ligas via Sofascore API. Genera `sofascore_events.parquet` | **Diaria** — 08:00 CDT |
| `predict_ligamx.py` | Calcula ratings Dixon-Coles, λ, probabilidades, value bets Liga MX. Genera CSV + DOCX | **Por jornada** — viernes 10:00 CDT |
| `context_enricher.py` | Módulo de apoyo: corrección Dixon-Coles, goleadores, motivación, corners | Importado por `predict_ligamx.py` |
| `odds_client.py` | Obtiene momios The Odds API / Altenar en tiempo real | Llamado desde `predict_ligamx.py` |
| `lineup_watcher.py` | Monitorea convocados via API-Football (endpoint `/fixtures/lineups`) | Llamado desde `predict_ligamx.py` |
| `_reporte_playdoit.py` | Imprime reporte formateado en terminal con value bets | Manual, antes de cada jornada |
| `analisis_j15.py` | Backtesting: compara predicciones vs resultados reales | Manual, post-jornada |
| `api.py` | Backend FastAPI que sirve `ligamx_predicciones.csv` al dashboard | Siempre activo (puerto 8000) |
| `dashboard.py` | Genera `dashboard.html` estático desde el CSV | Después de cada ejecución de `predict_ligamx.py` |
| `predict_ucl_v2.py` | Predicciones UCL (Champions League) | Por jornada UCL |
| `predict_simple.py` | Predicciones Poisson puro para 5 ligas europeas | Opcional / análisis |

---

## 3. Orden de ejecución del pipeline

### Pipeline diario (automatizado)
```bash
# 1. Actualizar historial de partidos
python collect_sofascore.py --days 7 --history-days 150

# 2. Generar predicciones Liga MX
python predict_ligamx.py --odds-key $ODDS_API_KEY

# 3. Regenerar dashboard HTML
python dashboard.py
```

### Pipeline de jornada (manual, viernes)
```bash
# 1. Recolección con más días hacia adelante
python collect_sofascore.py --days 15 --history-days 150

# 2. Predicciones con momios
python predict_ligamx.py --odds-key $ODDS_API_KEY

# 3. Ver reporte en terminal
python _reporte_playdoit.py

# 4. Actualizar dashboard
python dashboard.py
```

### Post-jornada (lunes, backtesting)
```bash
python analisis_j15.py
```

---

## 4. Configuración de APIs

### 4.1 Sofascore API (RapidAPI)
| Parámetro | Valor |
|-----------|-------|
| Host | `sofascore6.p.rapidapi.com` |
| Key | `c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d` |
| Endpoint | `GET /api/sofascore/v1/match/list?sport_slug=football&date=YYYY-MM-DD` |
| Rate limit | 0.8 s entre peticiones (evitar 429) |
| Plan | RapidAPI free tier |

**IDs de torneos usados:**
| Liga | ID |
|------|----|
| UEFA Champions League | 7 |
| Premier League | 17 |
| La Liga | 8 |
| Bundesliga | 35 |
| Serie A | 23 |
| Ligue 1 | 34 |
| Liga MX (Clausura) | 11620 |

### 4.2 The Odds API
| Parámetro | Valor |
|-----------|-------|
| Base URL | `https://api.the-odds-api.com/v4` |
| Key | Ver `memory/reference_apis.md` |
| Quota | 500 requests/mes (plan gratuito) |
| Uso | Momios Pinnacle para value bets Liga MX |
| Sport key | `soccer_mexico_ligamx` |

### 4.3 API-Football (RapidAPI)
| Parámetro | Valor |
|-----------|-------|
| Host | `api-football-v1.p.rapidapi.com` |
| Key | Ver `memory/reference_apis.md` |
| Quota | 100 requests/día (plan gratuito) |
| Uso | Stats de equipos por tarjetas, corners; convocados |
| Liga MX ID | 262 · Season | 2024 |

### 4.4 Altenar / Playdoit (momios locales)
| Parámetro | Valor |
|-----------|-------|
| Endpoint | Ver `odds_client.py` |
| Uso | Momios alternativos Playdoit para comparar vs The Odds API |

---

## 5. Variables de entorno (recomendado `.env`)

Crear archivo `.env` en la raíz del proyecto:

```env
# Sofascore
SOFASCORE_KEY=c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d

# The Odds API
ODDS_API_KEY=<tu_clave_odds_api>

# API-Football
API_FOOTBALL_KEY=<tu_clave_api_football>

# Configuración general
TIMEZONE=America/Mexico_City
LIGAMX_SEASON=2024
HISTORY_DAYS=150
```

Instalar `python-dotenv` y cargar en cada script:
```python
from dotenv import load_dotenv
import os
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
```

---

## 6. Base de datos (SQLite)

Actualmente el proyecto usa archivos **Parquet + JSON + CSV**. Para automatización robusta se recomienda migrar a **SQLite** (`data/quinielas.db`).

### 6.1 Schema propuesto

```sql
-- Partidos históricos (reemplaza sofascore_events.parquet)
CREATE TABLE IF NOT EXISTS matches (
    event_id        INTEGER PRIMARY KEY,
    date            TEXT NOT NULL,          -- YYYY-MM-DD en CDT
    tournament_id   INTEGER NOT NULL,
    tournament_name TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_score      INTEGER,                -- NULL si no jugado
    away_score      INTEGER,
    status          TEXT DEFAULT 'notstarted', -- notstarted | inprogress | finished
    timestamp_utc   INTEGER,                -- UNIX timestamp para timezone
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_matches_date ON matches(date);
CREATE INDEX idx_matches_tournament ON matches(tournament_id);
CREATE INDEX idx_matches_teams ON matches(home_team, away_team);

-- Predicciones generadas (reemplaza ligamx_predicciones.csv)
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date        TEXT NOT NULL,          -- fecha en que se generó la predicción
    match_date      TEXT NOT NULL,
    jornada         TEXT,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    prediccion      TEXT NOT NULL,          -- Local | Empate | Visitante
    p_local         REAL,
    p_empate        REAL,
    p_visit         REAL,
    lambda_h        REAL,
    lambda_a        REAL,
    att_h           REAL,
    def_h           REAL,
    att_a           REAL,
    def_a           REAL,
    n_h             INTEGER,
    n_a             INTEGER,
    forma_h         TEXT,
    forma_a         TEXT,
    pos_h           INTEGER,
    pts_h           INTEGER,
    pos_a           INTEGER,
    pts_a           INTEGER,
    confianza       TEXT,                   -- ALTA | MEDIA | BAJA
    ev_local        REAL,
    ev_empate       REAL,
    ev_visit        REAL,
    momio_local     REAL,
    momio_empate    REAL,
    momio_visit     REAL,
    corners_h       REAL,
    corners_a       REAL,
    amarillas_h     REAL,
    amarillas_a     REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_predictions_run_date ON predictions(run_date);
CREATE INDEX idx_predictions_match_date ON predictions(match_date);

-- Resultados reales (para backtesting automático)
CREATE TABLE IF NOT EXISTS results (
    event_id        INTEGER PRIMARY KEY REFERENCES matches(event_id),
    home_score      INTEGER NOT NULL,
    away_score      INTEGER NOT NULL,
    result          TEXT NOT NULL,          -- Local | Empate | Visitante
    recorded_at     TEXT DEFAULT (datetime('now'))
);

-- Backtesting por jornada
CREATE TABLE IF NOT EXISTS backtests (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    jornada         TEXT NOT NULL,
    run_date        TEXT NOT NULL,
    n_games         INTEGER,
    accuracy        REAL,
    brier_score     REAL,
    log_loss        REAL,
    lambda_mae      REAL,
    lambda_rmse     REAL,
    bias_local      REAL,
    bias_empate     REAL,
    bias_visit      REAL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Caché de momios (reemplaza ligamx_odds_cache.json)
CREATE TABLE IF NOT EXISTS odds_cache (
    match_key       TEXT PRIMARY KEY,       -- "home_team vs away_team YYYY-MM-DD"
    momio_local     REAL,
    momio_empate    REAL,
    momio_visit     REAL,
    n_bookmakers    INTEGER,
    casas           TEXT,
    fetched_at      TEXT DEFAULT (datetime('now'))
);

-- Log de ejecuciones del pipeline
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    script          TEXT NOT NULL,
    status          TEXT NOT NULL,          -- success | error | partial
    duration_secs   REAL,
    records_written INTEGER,
    error_msg       TEXT,
    run_at          TEXT DEFAULT (datetime('now'))
);
```

### 6.2 Script de inicialización de la BD
```bash
python -c "
import sqlite3
from pathlib import Path
sql = Path('PIPELINE_CONFIG.md').read_text()
# O ejecutar directamente el schema
conn = sqlite3.connect('data/quinielas.db')
# ... ejecutar schema
conn.close()
"
```

---

## 7. Automatización con Windows Task Scheduler

### 7.1 Script wrapper de automatización

Crear `run_pipeline.bat` en la raíz del proyecto:

```batch
@echo off
cd /d "C:\Users\USER\Documents\Proyecto Quinielas"
call venv\Scripts\activate.bat

echo [%date% %time%] Iniciando pipeline diario >> logs\pipeline.log

python collect_sofascore.py --days 7 --history-days 150 >> logs\collect.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] ERROR en collect_sofascore.py >> logs\pipeline.log
    exit /b 1
)

python predict_ligamx.py >> logs\predict.log 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] ERROR en predict_ligamx.py >> logs\pipeline.log
    exit /b 1
)

python dashboard.py >> logs\dashboard.log 2>&1

echo [%date% %time%] Pipeline completado exitosamente >> logs\pipeline.log
```

Crear directorio de logs:
```bash
mkdir logs
```

### 7.2 Crear la tarea programada

Abrir **Programador de tareas** (`taskschd.msc`) y configurar:

| Campo | Valor |
|-------|-------|
| Nombre | `Quinielas - Pipeline Diario` |
| Disparador | Diariamente a las **08:00 CDT** |
| Acción | `C:\Users\USER\Documents\Proyecto Quinielas\run_pipeline.bat` |
| Iniciar en | `C:\Users\USER\Documents\Proyecto Quinielas` |
| Ejecutar con permisos más altos | Sí |
| Condición: red | Requerir conexión de red disponible |

**Comando alternativo via CLI (PowerShell como Admin):**
```powershell
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\USER\Documents\Proyecto Quinielas\venv\Scripts\python.exe" `
    -Argument "C:\Users\USER\Documents\Proyecto Quinielas\collect_sofascore.py --days 7 --history-days 150" `
    -WorkingDirectory "C:\Users\USER\Documents\Proyecto Quinielas"

$trigger = New-ScheduledTaskTrigger -Daily -At "08:00AM"

Register-ScheduledTask `
    -TaskName "Quinielas - Collect Sofascore" `
    -Action $action `
    -Trigger $trigger `
    -RunLevel Highest `
    -Force
```

### 7.3 Tarea adicional: refresco de momios (2h antes del partido)

Para La Liga MX, los partidos suelen ser viernes 19:00, sábado 17:00–19:00, domingo 12:00–19:00 CDT:

```powershell
# Ejecutar viernes 17:00 CDT para capturar momios más ajustados
$trigger2 = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "5:00PM"
```

---

## 8. Estructura de directorios

```
Proyecto Quinielas/
├── collect_sofascore.py        # Recolección de datos
├── predict_ligamx.py           # Predicciones Liga MX (script principal)
├── predict_ucl_v2.py           # Predicciones Champions League
├── predict_simple.py           # Predicciones Poisson simple (5 ligas EU)
├── context_enricher.py         # Módulo de enriquecimiento
├── lineup_watcher.py           # Monitor de convocados
├── odds_client.py              # Cliente de momios
├── api.py                      # Backend FastAPI
├── dashboard.py                # Generador de dashboard HTML
├── dashboard.html              # Dashboard estático generado
├── _reporte_playdoit.py        # Reporte terminal (value bets)
├── _reporte_ligamx.py          # Reporte Liga MX extendido
├── analisis_j15.py             # Backtesting post-jornada
├── collect_data.py             # Recolección FBref/Understat (ligas EU)
├── feature_engineering.py      # Pipeline de features para ML
├── train_model.py              # Entrenamiento modelos XGBoost
├── requirements.txt
├── run_pipeline.bat            # Wrapper de automatización (crear)
├── .env                        # Variables de entorno (crear, NO subir a git)
├── .gitignore
├── soccer-project.code-workspace
│
├── data/
│   ├── sofascore_events.parquet        # Historial completo de partidos
│   ├── ligamx_predicciones.csv         # Predicciones activas
│   ├── ligamx_odds_cache.json          # Caché de momios
│   ├── ligamx_metodologia.docx         # Documento Word generado
│   ├── predicciones_sofascore.csv      # Predicciones 7 ligas
│   ├── quinielas.db                    # Base de datos SQLite (propuesto)
│   ├── _sofascore_cache/               # JSONs cacheados por fecha
│   │   └── matches_YYYY-MM-DD.json
│   ├── _fbref_cache/                   # Caché FBref
│   ├── processed/                      # Parquets de datos EU
│   └── models/                         # Modelos ML entrenados
│
├── logs/                               # Logs de automatización (crear)
│   ├── pipeline.log
│   ├── collect.log
│   └── predict.log
│
└── memory/                             # Memoria persistente Claude
    ├── MEMORY.md
    └── *.md
```

---

## 9. Setup en entorno nuevo

### 9.1 Requisitos previos
- Python 3.11+
- Git
- Windows 11 (o adaptar scripts .bat a Linux/macOS)

### 9.2 Instalación
```bash
# 1. Clonar repositorio
git clone <repo-url>
cd "Proyecto Quinielas"

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
pip install fastapi uvicorn python-dotenv python-docx

# 4. Crear archivo .env
copy .env.example .env       # Editar con tus claves API

# 5. Crear directorios necesarios
mkdir logs
mkdir data\_sofascore_cache
mkdir data\_fbref_cache

# 6. Primera ejecución
python collect_sofascore.py --history-days 150
python predict_ligamx.py
python dashboard.py
```

### 9.3 Iniciar API + Dashboard
```bash
# Terminal 1: Backend
uvicorn api:app --reload --port 8000

# Terminal 2: Abrir dashboard
start dashboard.html
# O con servidor local:
python -m http.server 3000
```

---

## 10. Parámetros del modelo

### Constantes calibradas (predict_ligamx.py)

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `SHRINK_K` | 4 | Partidos virtuales para shrinkage Bayesiano |
| `EWM_ALPHA` | 0.7 | Decaimiento exponencial (partido 8 semanas = 8.2% peso) |
| `LAMBDA_MAX` | 3.2 | Cap de goles esperados (ningún equipo LigaMX > 3.2) |
| `DRAW_FLOOR` | 0.15 | Probabilidad mínima de empate |
| `MIN_GAMES` | 3 | Mínimo de partidos para usar ratings propios |
| `WINDOW` | 8 | Ventana de partidos para calcular forma |
| `DC_RHO` | -0.13 | Parámetro Dixon-Coles (correlación resultados bajos) |
| `HOME_ADV` | 1.10 | Factor de ventaja de local (~+10%) |

### Regla de predicción de empate
```
Si P(Empate) >= 0.25 Y max(P_local, P_visit) - P(Empate) < 0.10:
    predecir "Empate"
Sino:
    predecir el outcome con mayor probabilidad
```

### Fórmulas clave
```
λ_local  = att_h × def_a_rival × μ_home × factor_motivacion × form_mult
λ_visit  = att_a × def_h_rival × μ_away × factor_motivacion × form_mult

att_h    = shrink(EWM(goles_marcados_casa) / μ_home, n_partidos)
def_h    = shrink(EWM(goles_recibidos_casa) / μ_away, n_partidos)

shrink   = (n × rating + K × 1.0) / (n + K)
form_mult= 0.92 + (pts_últimos_3 / pts_max) × 0.16   # rango [0.92, 1.08]
EV       = P_modelo × momio_decimal − 1
```

---

## 11. Métricas de calidad (referencia backtesting J15)

| Métrica | J15 (antes de mejoras) | Target |
|---------|------------------------|--------|
| Accuracy | 22.2% (2/9) | > 40% |
| Brier Score | 0.7929 | < 0.55 |
| Log Loss | 1.2933 | < 1.00 |
| λ RMSE | 1.445 goles | < 0.80 |
| Bias Local | +14 pp | ±5 pp |
| Bias Empate | -7.8 pp | ±5 pp |

---

## 12. Mantenimiento periódico

### Semanal
- [ ] Verificar que `collect_sofascore.py` actualizó `sofascore_events.parquet`
- [ ] Revisar `logs/pipeline.log` por errores
- [ ] Actualizar `CLAUSURA_2026_SCORERS` en `context_enricher.py` con goleadores de la jornada

### Por jornada (viernes)
- [ ] Ejecutar pipeline completo
- [ ] Revisar lambdas (rango esperado: 0.7–2.7)
- [ ] Revisar % de empates predichos (esperado: 3–6 de 9 partidos)
- [ ] Ejecutar `_reporte_playdoit.py` para reporte final

### Post-jornada (lunes)
- [ ] Ejecutar `analisis_j15.py` con resultados reales
- [ ] Registrar accuracy en tabla de métricas
- [ ] Actualizar equipos eliminados en `ligamx_situation()` si aplica

### Inicio de torneo (enero / julio)
- [ ] Actualizar `LIGAMX_TOURNAMENT_ID` si cambia (Apertura vs Clausura)
- [ ] Resetear cache: borrar `data/_sofascore_cache/*.json`
- [ ] Actualizar `CLAUSURA_2026_SCORERS` → `APERTURA_2026_SCORERS`
- [ ] Verificar equipos que ascendieron/descendieron
