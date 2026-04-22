#!/usr/bin/env python3
"""
api.py — Data Analysis Picks / Backend API
==========================================
FastAPI que sirve predicciones Liga MX + UCL al dashboard React.

Uso:
  python api.py                          → http://0.0.0.0:8000
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  GET /api/picks        → predicciones completas (74 campos)
  GET /api/summary      → KPIs y estadísticas agregadas
  GET /api/value-bets   → solo picks con EV positivo
  GET /health           → estado del servicio
  GET /                 → Dashboard React (dashboard/dist/)
"""
import math
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ── Asegurar que utils/ y el root estén en el path ───────────────────────────
_ROOT = Path(__file__).resolve().parent
for _p in (str(_ROOT / "utils"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.config import PATHS
from utils.logger import get_logger

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

log = get_logger("api")
CDT = timezone(timedelta(hours=-6))

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Data Analysis Picks API",
    version="2.0",
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_PATH   = PATHS["ligamx_csv"]
REACT_DIST = _ROOT / "dashboard" / "dist"

# ─────────────────────────────────────────────────────────────────────────────
# Mapeo: columna larga CSV → clave corta JSON
# ─────────────────────────────────────────────────────────────────────────────
COL_MAP = {
    "Fecha":                                                            "fecha",
    "Fase / Jornada":                                                   "fase",
    "Equipo Local":                                                     "local",
    "Equipo Visitante":                                                 "visitante",
    "Prediccion":                                                       "prediccion",
    "Probabilidad Local % (p_local)":                                   "p_local",
    "Probabilidad Empate % (p_empate)":                                 "p_empate",
    "Probabilidad Visitante % (p_visit)":                               "p_visit",
    "Goles Esperados Local (lambda_h)":                                 "lambda_h",
    "Goles Esperados Visitante (lambda_a)":                             "lambda_a",
    "Margen Goles Esperado (ventaja = lambda_h - lambda_a)":            "ventaja",
    "Ratio Ataque Local (att_h, >1 = mejor que promedio)":              "att_h",
    "Ratio Defensa Local (def_h, <1 = mejor que promedio)":             "def_h",
    "Ratio Ataque Visitante (att_a)":                                   "att_a",
    "Ratio Defensa Visitante (def_a)":                                  "def_a",
    "Partidos Base Local (n_h)":                                        "n_h",
    "Partidos Base Visitante (n_a)":                                    "n_a",
    "Forma Reciente Local W/D/L (forma_h)":                             "forma_h",
    "Forma Reciente Visitante W/D/L (forma_a)":                         "forma_a",
    "Puntos Forma Local (pts_h, W=3 D=1 L=0)":                         "pts_forma_h",
    "Puntos Forma Visitante (pts_a, W=3 D=1 L=0)":                     "pts_forma_a",
    "Posicion Tabla Local (pos_h)":                                     "pos_h",
    "Puntos en Tabla Local (pts_tabla_h)":                              "pts_tabla_h",
    "Partidos Jugados Local (pj_h)":                                    "pj_h",
    "Posicion Tabla Visitante (pos_v)":                                 "pos_v",
    "Puntos en Tabla Visitante (pts_tabla_v)":                          "pts_tabla_v",
    "Partidos Jugados Visitante (pj_v)":                                "pj_v",
    "H2H Partidos Analizados (h2h_n)":                                  "h2h_n",
    "H2H Victorias Local (h2h_w_h)":                                    "h2h_w_h",
    "H2H Empates (h2h_d)":                                              "h2h_d",
    "H2H Victorias Visitante (h2h_w_a)":                                "h2h_w_a",
    "H2H Goles Favor Local (h2h_gf_h)":                                 "h2h_gf_h",
    "H2H Goles Favor Visitante (h2h_gf_a)":                             "h2h_gf_a",
    "Media Goles Local Liga (mu_h)":                                    "mu_h",
    "Media Goles Visitante Liga (mu_a)":                                "mu_a",
    "Factor Ventaja Local (home_adv)":                                  "home_adv",
    "Prob Local % sin suavizar (p_local_raw)":                          "p_local_raw",
    "Prob Empate % sin suavizar (p_empate_raw)":                        "p_empate_raw",
    "Prob Visitante % sin suavizar (p_visit_raw)":                      "p_visit_raw",
    "Nivel de Confianza (ALTA / MEDIA / BAJA)":                         "confianza",
    "Momio Referencia Local (momio_ref_local)":                         "momio_local",
    "Momio Referencia Empate (momio_ref_empate)":                       "momio_empate",
    "Momio Referencia Visitante (momio_ref_visit)":                     "momio_visit",
    "Probabilidad Implicita Local % (p_imp_local, ajust. vig)":         "p_imp_local",
    "Probabilidad Implicita Empate % (p_imp_empate)":                   "p_imp_empate",
    "Probabilidad Implicita Visitante % (p_imp_visit)":                 "p_imp_visit",
    "Numero de Casas con Odds (n_bookmakers)":                          "n_bookmakers",
    "Casas de Apuestas Referencia (casas_referencia)":                  "casas",
    "Expected Value Local (ev_local = P_modelo*momio - 1)":             "ev_local",
    "Expected Value Empate (ev_empate)":                                "ev_empate",
    "Expected Value Visitante (ev_visitante)":                          "ev_visit",
    "Es Value Bet (ev > 0)":                                            "value_bet",
    "Goleadores Local (goleadores_local)":                              "goleadores_local",
    "Goleadores Visitante (goleadores_visita)":                         "goleadores_visita",
    "Corners Local Predichos (corners_h_pred)":                         "corners_local",
    "Corners Visitante Predichos (corners_a_pred)":                     "corners_visita",
    "Corners Total Predichos (corners_total_pred)":                     "corners_total",
    "Amarillas Local (amarillas_local)":                                "amarillas_local",
    "Amarillas Visitante (amarillas_visita)":                           "amarillas_visita",
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _clean(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    except Exception:
        return None


def _load_csv() -> list[dict]:
    if not CSV_PATH.exists():
        log.warning("CSV no encontrado: %s", CSV_PATH)
        return []
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        records = [{k: _clean(v) for k, v in row.items()} for _, row in df.iterrows()]
        log.debug("CSV cargado: %d picks", len(records))
        return records
    except Exception as exc:
        log.error("Error cargando CSV: %s", exc)
        return []


def _best_ev(pick: dict) -> float:
    pred    = pick.get("prediccion", "")
    mapping = {"Local": "ev_local", "Visitante": "ev_visit", "Empate": "ev_empate"}
    return pick.get(mapping.get(pred, ""), None) or -999.0


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — deben definirse ANTES del mount de StaticFiles
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    csv_ok   = CSV_PATH.exists()
    react_ok = (REACT_DIST / "index.html").exists()
    return {
        "status":     "ok" if csv_ok else "degraded",
        "csv":        str(CSV_PATH) if csv_ok else "missing",
        "react_dist": str(REACT_DIST) if react_ok else "missing — cd dashboard && npm run build",
        "updated":    datetime.now(CDT).isoformat(),
    }


@app.get("/api/picks")
def get_picks():
    picks = _load_csv()
    return {"picks": picks, "total": len(picks)}


@app.get("/api/summary")
def get_summary():
    picks = _load_csv()
    if not picks:
        raise HTTPException(status_code=503, detail="Sin datos — ejecutar predict_ligamx.py")

    confianza = {"ALTA": 0, "MEDIA": 0, "BAJA": 0}
    pred_dist = {"Local": 0, "Empate": 0, "Visitante": 0}
    pred_probs, evs = [], []

    for p in picks:
        conf = p.get("confianza", "BAJA")
        confianza[conf] = confianza.get(conf, 0) + 1
        pred = p.get("prediccion", "")
        pred_dist[pred] = pred_dist.get(pred, 0) + 1
        prob_key = {"Local": "p_local", "Visitante": "p_visit", "Empate": "p_empate"}.get(pred)
        if prob_key and (prob := p.get(prob_key)) is not None:
            pred_probs.append(prob)
        ev = _best_ev(p)
        if ev > -999:
            evs.append(ev)

    fechas = sorted({p["fecha"] for p in picks if p.get("fecha")})

    return {
        "total_picks":   len(picks),
        **confianza,
        "value_bets":    sum(1 for p in picks if p.get("value_bet") is True),
        "local_preds":   pred_dist.get("Local", 0),
        "empate_preds":  pred_dist.get("Empate", 0),
        "visita_preds":  pred_dist.get("Visitante", 0),
        "avg_ev":        round(sum(evs) / len(evs), 4) if evs else None,
        "avg_pred_prob": round(sum(pred_probs) / len(pred_probs), 1) if pred_probs else None,
        "has_odds":      any(p.get("momio_local") is not None for p in picks),
        "fecha_min":     fechas[0] if fechas else None,
        "fecha_max":     fechas[-1] if fechas else None,
        "fechas":        fechas,
    }


@app.get("/api/value-bets")
def get_value_bets():
    picks = _load_csv()
    vb = sorted(
        [p for p in picks if p.get("value_bet") is True],
        key=_best_ev,
        reverse=True,
    )
    return {"value_bets": vb, "total": len(vb)}


# ─────────────────────────────────────────────────────────────────────────────
# REACT DASHBOARD — montado DESPUÉS de todos los endpoints /api/*
# StaticFiles con html=True sirve index.html para cualquier ruta no coincidente
# ─────────────────────────────────────────────────────────────────────────────
if REACT_DIST.exists():
    app.mount("/", StaticFiles(directory=str(REACT_DIST), html=True), name="react")
    log.info("React dashboard montado desde %s", REACT_DIST)
else:
    log.warning("dashboard/dist/ no encontrado — ejecutar: cd dashboard && npm run build")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    log.info("Iniciando servidor en http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False, workers=2)
