# Informe Técnico — Soccer Prediction Project
**Fecha:** Marzo 2026
**Entorno:** Python 3.12, Windows 11, venv en `/venv`
**Workspace:** `soccer-project.code-workspace`

---

## 1. Visión General

Sistema de predicción estadística de resultados de fútbol orientado a la Champions League UEFA y ligas europeas top-5. El proyecto combina múltiples fuentes de datos históricos con modelos probabilísticos clásicos (Poisson, Dixon-Coles) y aprendizaje automático (XGBoost), complementados con odds de mercado en tiempo real para identificar discrepancias entre el modelo y el consenso del mercado.

**El sistema no es un sistema de apuestas automatizado.** Es una herramienta de análisis que produce probabilidades calibradas y compara contra líneas de mercado (Pinnacle) para detectar posibles value bets que el usuario evalúa manualmente.

---

## 2. Arquitectura del Sistema

```
FUENTES DE DATOS
├── football-data.org (FDorg)     → Resultados UCL históricos, fixtures
├── Understat                     → xG por partido (5 ligas, 9 temporadas)
├── FBref / soccerdata            → Estadísticas de jugadores por temporada
├── Sofascore6 (RapidAPI)         → Eventos multi-liga en tiempo real
├── The Odds API                  → Odds Pinnacle + 40 bookmakers (tiempo real)
└── API-Football                  → Lineups e injuries (temporadas 2022-2024)

PIPELINE DE DATOS
├── collect_data.py               → Descarga y cachea datos históricos
├── collect_sofascore.py          → Barrido diario Sofascore (histórico + próximos)
└── feature_engineering.py        → Genera features.parquet para ML

MODELOS
├── train_model.py                → Entrena XGBoost / RF / LogReg, guarda pkl
├── predict_simple.py             → Poisson puro con xG de Understat
├── predict_ucl_v2.py             → Dixon-Coles UCL (modelo principal)
├── predict_upcoming.py           → XGBoost pre-partido con features completas
└── predict_ligamx.py             → Modelo adaptado Liga MX

ANÁLISIS EN TIEMPO REAL
├── odds_client.py                → Módulo The Odds API (importable)
└── lineup_watcher.py             → Predicción + odds + ajuste por alineación
```

---

## 3. Scripts Principales

### 3.1 `collect_data.py` — Recolección de datos históricos

**Fuentes:** FBref (vía `soccerdata`), Understat, football-data.org
**Output:**
- `data/processed/schedule_xg.parquet` — partidos con xG (Understat)
- `data/processed/player_season_stats.parquet` — stats de jugadores
- `data/league_{1-5}_season_{2017-2025}.json` — caché JSON por liga/temporada

**Cobertura:**
- 5 ligas: Premier League, La Liga, Bundesliga, Serie A, Ligue 1
- 9 temporadas: 2017-18 a 2025-26 (formato FBref: `1718` a `2526`)
- Jugadores: 24,471 registros × 23 columnas (npxG, xA, xG_chain, minutos, etc.)

---

### 3.2 `feature_engineering.py` — Pipeline de features para ML

**Input:** `schedule_xg.parquet`, `player_season_stats.parquet`
**Output:** `data/processed/features.parquet`

**Features generadas:**

| Feature | Descripción | Data leakage |
|---|---|---|
| `resultado` | Target: 1=Local, 0=Empate, -1=Visitante | — |
| `home_xg`, `away_xg` | xG del partido | Sí (in-match) |
| `xg_diff`, `xg_total` | Diferencia y suma de xG | Sí |
| `home_roll_xg_for/against` | Rolling xG últimos 5 partidos (shift-1) | No |
| `form_xg_diff` | Diferencial de forma | No |
| `home/away_squad_np_xg` | Calidad del plantel (temporada) | No |
| `home/away_table_pos` | Posición en tabla acumulada | No |
| `h2h_*` | Head-to-head features | No |
| `es_local` | Feature binaria ventaja local | No |

> **Nota crítica:** Las features `home_xg`, `away_xg`, `xg_diff`, `xg_total` son **in-match** (no disponibles pre-partido). El modelo `pre_match_model.pkl` las excluye; los modelos baseline las incluyen solo para análisis retrospectivo.

**Implementación técnica:**
- Rolling form vectorizado con `merge_asof` + `shift(1)` (evita data leakage por fila)
- Table position usando `cumsum` acumulado antes de cada partido
- Squad quality: media ponderada de npxG + xA por posición del plantel

---

### 3.3 `train_model.py` — Entrenamiento ML

**Algoritmos:** Logistic Regression, Random Forest (n=200), XGBoost (n=200, 300)
**Split temporal:** train = temporadas antiguas, test = 2526 + 2425
**Baseline ingenuo:** siempre predice Local (~44% accuracy)

**Modelos guardados en `data/models/`:**

| Archivo | Descripción |
|---|---|
| `best_model.pkl` | Mejor de los 3 modelos (con features in-match, para análisis) |
| `pre_match_model.pkl` | XGBoost 300 árboles, solo features pre-partido |
| `pre_match_feature_names.pkl` | Lista de features del modelo pre-partido |
| `scaler.pkl`, `label_encoder.pkl` | Preprocesadores |

---

### 3.4 `predict_ucl_v2.py` — Modelo principal UCL (Dixon-Coles)

**El script más avanzado del proyecto.** Predice resultados de Champions League aplicando:

#### Modelo estadístico
**Dixon-Coles (1997)** con corrección τ para baja puntuación:

```
P(X=h, Y=a) = τ(h,a,λh,λa) × Poisson(h|λh) × Poisson(a|λa)

τ(0,0) = 1 - λh·λa·ρ
τ(1,0) = 1 + λa·ρ
τ(0,1) = 1 + λh·ρ
τ(1,1) = 1 - ρ
τ(·,·) = 1  (en cualquier otro caso)

ρ = -0.13 (fijo, del paper original)
```

#### Modelo de ratings (lambdas multiplicativas)
```
λh = att_h × def_a_rival × μh × home_adv
λa = att_a × def_h_rival × μa

att_h = prom_goles_marcados_casa / μh
def_a = prom_goles_recibidos_fuera / μh
```

- **Fuente UCL:** football-data.org, temporadas 2023, 2024, 2025 (482 partidos)
- **Blend UCL + Doméstico:** peso UCL = min(n_partidos/6, 1.0) × 80%; el resto de doméstico (Understat xG)
- **Home advantage UCL calibrado:** 1.257× (vs 1.15 asumido en versiones anteriores)
- **Draw floor dinámico:** proporcional a λ promedio, mínimo 15%
- **Pesos temporales:** media ponderada reciente (N=más reciente, 1=más antiguo)

#### Funcionalidades adicionales
- **P(clasificación):** Para partidos de vuelta, busca resultado de ida vía fuzzy matching y simula todos los scorelines posibles → P(clasifica local), P(prórroga), P(clasifica visitante)
- **Bet slips automáticos:** 3 tipos por partido (pata simple, SGP 2 patas, SGP soñador 3 patas)
- **Mercados calculados desde la matriz DC:** 1X2, Over/Under 1.5/2.5/3.5, BTTS, marcador más probable, top-5 scorelines
- **Flag `--refresh`:** Invalida caché de la temporada en curso para obtener resultados recientes

**Output:** `data/predicciones_ucl_v2.csv` (60 columnas × N partidos)

**Uso:**
```bash
python predict_ucl_v2.py --days 3 --refresh
python predict_ucl_v2.py --days 7 --rho -0.10
```

---

### 3.5 `collect_sofascore.py` — Pipeline multi-liga en tiempo real

**Fuente:** `sofascore6.p.rapidapi.com`
**Ligas:** UCL (7), Premier League (17), La Liga (8), Bundesliga (35), Serie A (23), Ligue 1 (34), Liga MX (11620)

**Arquitectura:**
- Barrido único: descarga histórico (150 días atrás) + próximos (15 días adelante) en una sola ejecución
- Caché permanente para fechas pasadas; siempre fresco para hoy y futuro
- Ratings multiplicativos Dixon-Coles adaptados (sin corrección τ, más rápido)
- Blend UCL (hasta 80%) + Doméstico cuando el partido es de UCL

**Output:** `data/predicciones_sofascore.csv` (36 columnas con nombres descriptivos)

**Uso:**
```bash
python collect_sofascore.py --days 15 --history-days 150
```

---

### 3.6 `odds_client.py` — The Odds API (módulo importable)

**API Key:** `306f7fec9f210e1c341292af655dd0d0`
**Cuota free:** 500 requests/mes
**Bookmakers disponibles:** 41 por partido UCL (incluye Pinnacle, Betfair, Marathonbet)

#### Funciones principales

```python
from odds_client import fetch_odds, find_match, calc_edge, ev_pct, format_edge_label

# Obtener odds (caché 30 min)
odds = fetch_odds("UCL", ttl_minutes=30)
# → list[{home, away, date, preferred, avg_market, n_bookmakers, bookmakers}]

# Cascade: Pinnacle > Betfair > Marathonbet > Nordicbet > Betsson
# preferred = {bookmaker_key, o_h, o_d, o_a, p_h, p_d, p_a, overround_pct}

# Probabilidades sin vig (remove bookmaker margin)
p_h, p_d, p_a = no_vig(o_h=1.62, o_d=4.60, o_a=5.15)
# → (0.600, 0.211, 0.189)  [suman 1.0]

# Edge y EV
edge = calc_edge(our_prob_pct=43.1, market_prob_pct=19.9)  # +23.2%
ev   = ev_pct(our_prob=0.431, decimal_odds=4.73)            # +103.8%
```

#### Matching de partidos
Usa `SequenceMatcher` + diccionario de aliases para emparejar nombres de equipos entre sistemas (e.g., "FC Barcelona" → "Barcelona" en The Odds API).

**Sports disponibles:** UCL, Europa, EPL, LaLiga, Bundesliga, SerieA, Ligue1, LigaMX

**Uso standalone:**
```bash
python odds_client.py --sport UCL --ttl 0   # forzar descarga fresca
python odds_client.py --sport EPL
```

---

### 3.7 `lineup_watcher.py` — Análisis integral pre-partido

Script principal de análisis día de partido. Combina tres capas:

```
[1] Predicción base (predicciones_ucl_v2.csv)
         ↓
[2] Ajuste por alineación (player_season_stats.parquet)
         ↓
[3] Comparación con odds Pinnacle → Edge + EV
```

#### Modelo de fuerza de jugadores

```python
# Temporadas recientes: pesos 2526=3, 2425=2, 2324=1
p90             = minutes / 90
goal_cont_p90   = (np_xg + xa) / p90        # aportación directa al gol
chain_p90       = xg_chain / p90             # participación en cadenas de gol
strength        = 0.6 × goal_cont_p90 + 0.4 × chain_p90

# Media ponderada por temporada para cada (player, team)
# Cobertura: 4,620 jugadores en 119 equipos (5 ligas top europeas)
```

**Baseline del equipo:** `squad_mean_strength × 11`
(no top-11 ofensivo, para evitar sesgo hacia atacantes vs. porteros/defensas)

**Factor de ajuste:**
```python
factor = xi_announced_strength / expected_xi_strength
factor = clamp(factor, 0.65, 1.35)

lambda_h_adj = lambda_h × factor_home
lambda_a_adj = lambda_a × factor_away
```

**Matching de jugadores:** 3 estrategias en cascada:
1. Exacto (sin tildes, Unicode normalizado)
2. Apellido como sufijo: `"salah"` → `"mohamed salah"`, `"konate"` → `"ibrahima konate"`
3. Fuzzy SequenceMatcher (umbral 0.72)

**Limitaciones conocidas:**
- Equipos sin datos en parquet (Galatasaray, Sporting CP, Bodø/Glimt, equipos no-top5): factor = 1.0
- API-Football free tier **no accede a la temporada actual (2025-26)** → lineups deben ingresarse manualmente via archivo JSON

#### Archivo de alineaciones (JSON)

```json
{
  "matches": [{
    "home": "Barcelona",
    "away": "Newcastle United",
    "home_formation": "4-3-3",
    "away_formation": "4-3-3",
    "home_xi": ["Ter Stegen", "Kounde", "Cubarsí", ...],
    "away_xi":  ["Pope", "Trippier", "Schar", ...]
  }]
}
```

**Uso:**
```bash
# Sin alineaciones (solo predicción base + odds)
python lineup_watcher.py

# Con alineaciones confirmadas
python lineup_watcher.py --lineup data/lineups/hoy.json

# Liga doméstica
python lineup_watcher.py --sport EPL --csv data/predicciones_sofascore.csv
```

---

## 4. Flujo de Trabajo Estándar

### Día previo al partido
```bash
# 1. Actualizar predicciones UCL con datos recientes
python predict_ucl_v2.py --days 3 --refresh

# 2. Ver predicciones base + odds Pinnacle
python lineup_watcher.py
```

### ~75 minutos antes del kickoff (cuando anuncian XI)
```bash
# 3. Ingresar alineaciones en JSON
#    → editar data/lineups/hoy.json con los XI confirmados

# 4. Ejecutar análisis completo con ajuste de lineup
python lineup_watcher.py --lineup data/lineups/hoy.json --ttl 5
```

### Para ligas domésticas (EPL, LaLiga, etc.)
```bash
# 1. Recolectar datos Sofascore
python collect_sofascore.py --days 10 --history-days 150

# 2. Ver odds + predicciones
python lineup_watcher.py --sport EPL --csv data/predicciones_sofascore.csv
```

---

## 5. APIs y Credenciales

| Servicio | Clave | Plan | Límite | Uso en proyecto |
|---|---|---|---|---|
| **The Odds API** | `306f7fec9f210e1c341292af655dd0d0` | Free | 500 req/mes | `odds_client.py` |
| **API-Football** | `5cf3eb50762eeb4e9cf15173bae1cb65` | Free | 100 req/día | Lineups históricos (2022-2024) |
| **Sofascore6** (RapidAPI) | `c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d` | Free | — | `collect_sofascore.py` |
| **football-data.org** | En `predict_ucl_v2.py` | Free | Rate limited | Fixtures + resultados UCL |

> **Limitación API-Football:** El plan gratuito no tiene acceso a la temporada actual (2025-26). Solo disponibles temporadas 2022-2024. Para lineups de la temporada actual, se requiere entrada manual vía JSON o upgrade a plan Pro (~$10/mes).

---

## 6. Estructura de Datos

### Inputs (data/processed/)

| Archivo | Descripción | Filas | Columnas |
|---|---|---|---|
| `schedule_xg.parquet` | Partidos con xG Understat | ~16,000 | ~15 |
| `player_season_stats.parquet` | Stats jugadores por temporada | 24,471 | 23 |
| `features.parquet` | Features ML completas | ~16,000 | ~30 |
| `fdorg_cards.parquet` | Tarjetas Football-Data.org | — | — |
| `sportsapi_match_stats.parquet` | Stats SportsAPI | — | — |

### Outputs principales

| Archivo | Generado por | Descripción |
|---|---|---|
| `data/predicciones_ucl_v2.csv` | `predict_ucl_v2.py` | 60 cols: probabilidades, lambdas, bet slips UCL |
| `data/predicciones_sofascore.csv` | `collect_sofascore.py` | 36 cols: 7 ligas, multi-día |
| `data/predicciones_simple.csv` | `predict_simple.py` | Poisson puro con xG |
| `data/predicciones_proximas.csv` | `predict_upcoming.py` | XGBoost pre-partido |
| `data/models/pre_match_model.pkl` | `train_model.py` | XGBoost 300 árboles |

### Caché (data/_*_cache/)

| Directorio | Contenido | TTL |
|---|---|---|
| `_fdorg_cache/` | JSON por endpoint football-data.org | Permanente (terminados) / fresco (próximos) |
| `_sofascore_cache/` | JSON por fecha Sofascore | Permanente para pasado |
| `_odds_cache/` | JSON por sport_key Odds API | 30 min |
| `_ucl_cache/` | JSON UCL histórico por temporada | Permanente / --refresh para actual |

---

## 7. Dependencias

```
soccerdata>=1.8.8     # FBref scraping
pandas>=2.0
numpy>=1.26
pyarrow>=14           # formato parquet
scikit-learn>=1.4
xgboost>=2.0
matplotlib>=3.8
seaborn>=0.13
requests              # HTTP clients
```

**Instalación:**
```bash
python -m venv venv
source venv/Scripts/activate    # Windows
pip install -r requirements.txt
```

---

## 8. Limitaciones y Deuda Técnica

### Limitaciones del modelo estadístico

| Limitación | Impacto | Estado |
|---|---|---|
| `rho = -0.13` hardcodeado (Dixon-Coles) | Debería calibrarse con MLE desde datos UCL propios | Pendiente |
| Sin decaimiento temporal exponencial | Partidos de hace 40 jornadas pesan demasiado | Pendiente |
| Ratings UCL y doméstico en escalas distintas | El blend asume equivalencia de escalas | Pendiente |
| BTTS × resultado no son independientes (SGP) | Las probabilidades SGP están sobreestimadas | Pendiente |
| Sin ajuste táctico por contexto de eliminatoria | Edge artificial en partidos de vuelta con resultado global decisivo (e.g., Tottenham 82% pero Atletico ganó 5-2) | Documentado |

### Limitaciones de datos

| Limitación | Causa | Mitigación |
|---|---|---|
| Sin lineups actuales automáticos | API-Football free tier no cubre temporada 2025-26 | Entrada manual JSON |
| Sin datos de equipos non-top5 (Galatasaray, Sporting, Bodo/Glimt) | `player_season_stats.parquet` solo cubre 5 ligas | Factor = 1.0 |
| Sofascore6 sin endpoints de detalle | Plan RapidAPI actual solo expone `match/list` | Sin solución actual |

---

## 9. Interpretación del Edge

El `edge` reportado por `lineup_watcher.py` **no es una señal directa de apuesta.** Es la diferencia entre la probabilidad del modelo y la probabilidad de mercado sin vig:

```
edge = modelo_% - pinnacle_no_vig_%
EV%  = (prob_modelo × (odds - 1) - (1 - prob_modelo)) × 100
```

**Cómo interpretar:**
- `edge > +5%` → Nuestro modelo es significativamente más optimista que el mercado para ese resultado. Investigar por qué: ¿nuestro modelo ignora algo? ¿o el mercado tiene sesgo?
- `EV > 0` → Si el modelo es correcto, la apuesta tiene valor esperado positivo.
- Edge en partidos de vuelta de eliminatoria: **verificar siempre** si el edge es real o es un artefacto de no modelar el marcador global (el mercado lo incorpora, nuestro modelo solo ve el partido de 90 min).

---

## 10. Roadmap de Mejoras Pendientes

### Alta prioridad
- [ ] Calibrar ρ (rho) Dixon-Coles por MLE desde los 482 partidos UCL propios
- [ ] Decaimiento exponencial en ratings (half-life ~20 partidos)
- [ ] Curva de calibración (Brier score) para verificar que probabilities son fieles
- [ ] Ajuste táctico para partidos de vuelta: penalizar lambda del equipo que va ganando en el global

### Media prioridad
- [ ] Lineups automáticos: scraping de SofaScore.com o upgrade API-Football Pro
- [ ] Ajuste por calendario/fatiga: días desde último partido
- [ ] Weather API (Open-Meteo, gratuita): reducción de goles con lluvia intensa (~8%)
- [ ] Normalización de ratings UCL vs. doméstico antes del blend

### Baja prioridad
- [ ] Árbitro como feature: algunos árbitros producen significativamente más/menos goles
- [ ] Ajuste por factor táctico (equipo que parquea el bus vs. presión alta)
- [ ] Exportación a Google Sheets para seguimiento de resultados vs. predicciones
