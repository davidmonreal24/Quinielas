#!/usr/bin/env python3
"""
api.py  —  Data Analysis Picks  /  Backend API
===============================================
FastAPI que sirve los datos de predict_ligamx.py al dashboard React.

Uso:
  python api.py               → inicia en http://localhost:8000
  uvicorn api:app --reload    → con hot-reload
"""

import math
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import pandas as pd

app = FastAPI(title="Data Analysis Picks API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR  = Path("data")
CSV_PATH  = DATA_DIR / "ligamx_predicciones.csv"

# Mapeo: columna larga CSV → clave corta JSON
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
}


def _clean(v):
    """Convierte NaN/inf a None para JSON serialization."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    except Exception:
        return None


def load_picks() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})
        records = []
        for _, row in df.iterrows():
            record = {k: _clean(v) for k, v in row.items()}
            records.append(record)
        return records
    except Exception as e:
        print(f"Error cargando CSV: {e}")
        return []


@app.get("/api/picks")
def get_picks():
    """Lista completa de predicciones Liga MX."""
    picks = load_picks()
    return {"picks": picks, "total": len(picks)}


@app.get("/api/summary")
def get_summary():
    """KPIs y estadísticas agregadas del modelo."""
    picks = load_picks()
    if not picks:
        return {"error": "Sin datos"}

    total      = len(picks)
    alta       = sum(1 for p in picks if p.get("confianza") == "ALTA")
    media      = sum(1 for p in picks if p.get("confianza") == "MEDIA")
    baja       = sum(1 for p in picks if p.get("confianza") == "BAJA")
    value_bets = sum(1 for p in picks if p.get("value_bet") is True)

    local_preds    = sum(1 for p in picks if p.get("prediccion") == "Local")
    empate_preds   = sum(1 for p in picks if p.get("prediccion") == "Empate")
    visita_preds   = sum(1 for p in picks if p.get("prediccion") == "Visitante")

    # EV stats (sólo partidos con EV)
    evs = []
    for p in picks:
        pred = p.get("prediccion", "")
        if pred == "Local"     and p.get("ev_local")  is not None:
            evs.append(p["ev_local"])
        elif pred == "Visitante" and p.get("ev_visit") is not None:
            evs.append(p["ev_visit"])
        elif pred == "Empate"  and p.get("ev_empate") is not None:
            evs.append(p["ev_empate"])
    avg_ev = round(sum(evs) / len(evs), 4) if evs else None

    # Avg probabilidad de la predicción ganadora
    pred_probs = []
    for p in picks:
        pred = p.get("prediccion", "")
        if pred == "Local":    pred_probs.append(p.get("p_local") or 0)
        elif pred == "Visitante": pred_probs.append(p.get("p_visit") or 0)
        elif pred == "Empate":    pred_probs.append(p.get("p_empate") or 0)
    avg_prob = round(sum(pred_probs) / len(pred_probs), 1) if pred_probs else None

    # Fechas
    fechas = sorted({p["fecha"] for p in picks if p.get("fecha")})

    return {
        "total_picks":     total,
        "alta":            alta,
        "media":           media,
        "baja":            baja,
        "value_bets":      value_bets,
        "local_preds":     local_preds,
        "empate_preds":    empate_preds,
        "visita_preds":    visita_preds,
        "avg_ev":          avg_ev,
        "avg_pred_prob":   avg_prob,
        "fechas":          fechas,
        "fecha_min":       fechas[0] if fechas else None,
        "fecha_max":       fechas[-1] if fechas else None,
        "has_odds":        any(p.get("momio_local") is not None for p in picks),
    }


@app.get("/api/value-bets")
def get_value_bets():
    """Solo los picks con EV positivo (Value Bets)."""
    picks = load_picks()
    vb = [p for p in picks if p.get("value_bet") is True]
    # Sort by best EV
    def best_ev(p):
        pred = p.get("prediccion", "")
        if pred == "Local":      return p.get("ev_local") or -999
        if pred == "Visitante":  return p.get("ev_visit") or -999
        if pred == "Empate":     return p.get("ev_empate") or -999
        return -999
    vb.sort(key=best_ev, reverse=True)
    return {"value_bets": vb, "total": len(vb)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
