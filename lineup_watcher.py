"""
=====================================================================
  lineup_watcher.py — Predicciones con Alineaciones + Odds de Mercado
=====================================================================

DESCRIPCIÓN:
  Combina tres fuentes de información para cada partido UCL:
    1. Predicción base (predict_ucl_v2.py → predicciones_ucl_v2.csv)
    2. Odds en tiempo real (The Odds API — Pinnacle / Betfair)
    3. Ajuste por alineación (FBref player_season_stats.parquet)

MODOS DE USO:

  Modo 1 — Solo odds + predicción base (sin alineación):
    python lineup_watcher.py

  Modo 2 — Con archivo de alineación:
    python lineup_watcher.py --lineup data/lineups/hoy.json

  Modo 3 — Ligas domésticas (EPL, LaLiga, etc.):
    python lineup_watcher.py --sport EPL

FORMATO DEL ARCHIVO DE ALINEACIÓN (JSON):
  Ver data/lineups/ejemplo.json

  Campo opcional "playdoit" por partido (odds manuales de Playdoit):
    "playdoit": {"local": 1.70, "empate": 3.50, "visitante": 4.50}
  Si se incluye, Playdoit es la referencia principal del EV.
  Pinnacle/Betfair (Odds API) se muestra como línea de referencia secundaria.

FLUJO PREVIO:
  Antes de correr este script, ejecutar:
    python predict_ucl_v2.py --days 3 --refresh
  Esto genera data/predicciones_ucl_v2.csv que este script utiliza.
=====================================================================
"""

import argparse
import json
import math
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import unicodedata

import numpy as np
import pandas as pd

from odds_client import (
    fetch_odds,
    find_match,
    no_vig,
    calc_edge,
    ev_pct,
    format_edge_label,
    overround_pct,
    SPORT_KEYS,
)

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

PREDS_UCL  = Path("data/predicciones_ucl_v2.csv")
PLAYER_PKL = Path("data/processed/player_season_stats.parquet")
LINEUPS_DIR = Path("data/lineups")
LINEUPS_DIR.mkdir(parents=True, exist_ok=True)

# Seasons recientes para modelo de fuerza de jugadores (mayor peso a más reciente)
RECENT_SEASONS = ["2526", "2425", "2324"]

# Factor de ajuste mínimo/máximo (evitar extremos por plantel roto)
FACTOR_MIN = 0.65
FACTOR_MAX = 1.35

# Normalización de equipos (nuestro nombre → nombre en parquet)
TEAM_MAP = {
    "fc barcelona":               "Barcelona",
    "barcelona":                  "Barcelona",
    "real madrid cf":             "Real Madrid",
    "real madrid":                "Real Madrid",
    "manchester city fc":         "Manchester City",
    "manchester city":            "Manchester City",
    "arsenal fc":                 "Arsenal",
    "arsenal":                    "Arsenal",
    "chelsea fc":                 "Chelsea",
    "chelsea":                    "Chelsea",
    "liverpool fc":               "Liverpool",
    "liverpool":                  "Liverpool",
    "tottenham hotspur fc":       "Tottenham",
    "tottenham hotspur":          "Tottenham",
    "tottenham":                  "Tottenham",
    "newcastle united fc":        "Newcastle United",
    "newcastle united":           "Newcastle United",
    "newcastle":                  "Newcastle United",
    "atletico madrid":            "Atletico Madrid",
    "atletico de madrid":         "Atletico Madrid",
    "club atletico de madrid":    "Atletico Madrid",
    "fc bayern münchen":          "Bayern Munich",
    "fc bayern munchen":          "Bayern Munich",
    "bayern munich":              "Bayern Munich",
    "bayer 04 leverkusen":        "Bayer Leverkusen",
    "bayer leverkusen":           "Bayer Leverkusen",
    "atalanta bc":                "Atalanta",
    "atalanta":                   "Atalanta",
    "paris saint-germain fc":     "PSG",
    "paris saint-germain":        "PSG",
    "psg":                        "PSG",
    # Equipos sin datos en parquet (solo ligas europeas top 5)
    "galatasaray sk":             None,
    "galatasaray":                None,
    "sporting clube de portugal": None,
    "sporting cp":                None,
    "fk bodo/glimt":             None,
    "bodo/glimt":                 None,
}


def _norm_team(name: str) -> str:
    """Normaliza nombre de equipo para búsqueda en TEAM_MAP."""
    s = str(name).lower().strip()
    s = re.sub(r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk)\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def resolve_team(name: str) -> str | None:
    """Retorna nombre en el parquet, o None si no hay datos domésticos."""
    key = _norm_team(name)
    if key in TEAM_MAP:
        return TEAM_MAP[key]
    # Fuzzy fallback
    best, best_score = None, 0.75
    for k, v in TEAM_MAP.items():
        r = SequenceMatcher(None, key, k).ratio()
        if r > best_score:
            best_score, best = r, v  # correcto: guardar ratio, no el valor
    return best


# ─────────────────────────────────────────────
# MODELO DE FUERZA DE JUGADORES
# ─────────────────────────────────────────────

def build_player_strength_model() -> dict:
    """
    Construye un diccionario {player_name_lower: strength_score} y
    {team_name: {expected_xi_strength, player_dict}} desde FBref.

    Fórmula de fuerza:
      goal_contribution_p90 = (np_xg + xa) / (minutes / 90)
      chain_p90             = xg_chain / (minutes / 90)
      strength = 0.6 * goal_contribution_p90 + 0.4 * chain_p90
      (ambos capturan aportación directa e indirecta al juego)

    Se usa la media ponderada de temporadas recientes:
      2526 → peso 3, 2425 → peso 2, 2324 → peso 1
    """
    if not PLAYER_PKL.exists():
        print("  [lineup] AVISO: player_season_stats.parquet no encontrado.")
        return {}

    df = pd.read_parquet(PLAYER_PKL)
    df = df[df["season"].isin(RECENT_SEASONS)].copy()
    df = df[df["minutes"] >= 90].copy()  # mínimo 1 partido completo

    season_weight = {"2526": 3, "2425": 2, "2324": 1}
    df["w"] = df["season"].map(season_weight).fillna(1)

    # Stats por 90
    df["p90"] = df["minutes"] / 90.0
    df["goal_cont_p90"] = (df["np_xg"] + df["xa"]) / df["p90"]
    df["chain_p90"]     = df["xg_chain"] / df["p90"]
    df["strength_raw"]  = 0.6 * df["goal_cont_p90"] + 0.4 * df["chain_p90"]

    # Media ponderada por temporada para cada jugador+equipo
    def _agg_fn(g):
        return pd.Series({
            "strength":  np.average(g["strength_raw"].values, weights=g["w"].values),
            "total_min": g["minutes"].sum(),
            "seasons":   ",".join(sorted(g["season"].unique())),
        })

    agg = (
        df.groupby(["player", "team"])[
            ["strength_raw", "w", "minutes", "season"]
        ]
        .apply(_agg_fn)
        .reset_index()
    )

    # Estructura del modelo: por equipo
    # BASELINE: media del squad × 11 (no top-11 ofensivo, que sesga contra defensas/porteros)
    model = {}
    for team, grp in agg.groupby("team"):
        grp = grp.sort_values("strength", ascending=False)
        all_pl = {
            row["player"].lower(): round(float(row["strength"]), 4)
            for _, row in grp.iterrows()
        }
        # Squad mean × 11 = línea base neutra (un XI de calidad promedio del equipo = factor 1.0)
        squad_mean = float(grp["strength"].mean())
        expected_xi_strength = squad_mean * 11
        model[team] = {
            "expected_xi_strength": expected_xi_strength,
            "squad_mean":           squad_mean,
            "players":              all_pl,
        }

    print(f"  [lineup] Modelo de fuerza: {len(agg)} jugadores en {len(model)} equipos")
    return model


def _deaccent(s: str) -> str:
    """Elimina tildes y diacríticos: 'Konaté' → 'konate'."""
    return "".join(
        c for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def _find_player_strength(name: str, team_model: dict) -> float | None:
    """
    Busca la fuerza de un jugador en el modelo del equipo.
    Estrategias (en orden de prioridad):
      1. Exacto (normalizado + sin tildes)
      2. Apellido como sufijo del nombre completo en el modelo
      3. Fuzzy matching SequenceMatcher
    """
    name_l  = _deaccent(name.lower().strip())
    players = team_model.get("players", {})

    # players dict normalizado sin tildes para comparación
    players_norm = {_deaccent(k): v for k, v in players.items()}

    # 1. Exacto (sin tildes)
    if name_l in players_norm:
        return players_norm[name_l]

    # 2. Apellido como sufijo: "salah" → "mohamed salah", "konate" → "ibrahima konate"
    #    o como substring completo: "van dijk" → "virgil van dijk"
    parts = name_l.split()
    for pname_n, strength in players_norm.items():
        for part in parts:
            if len(part) >= 4 and pname_n.endswith(part):
                return strength
        if name_l in pname_n:
            return strength

    # 3. Fuzzy SequenceMatcher (sin tildes en ambos lados)
    best_score, best_val = 0.72, None
    for pname_n, strength in players_norm.items():
        r = SequenceMatcher(None, name_l, pname_n).ratio()
        if r > best_score:
            best_score, best_val = r, strength
    return best_val


def compute_lineup_factor(xi: list, team_name: str, model: dict) -> dict:
    """
    Calcula el factor de ajuste de lambda para un equipo dado su XI.

    Retorna:
      {
        "team": str,
        "factor": float,           # lambda *= factor
        "xi_strength": float,      # fuerza del XI anunciado
        "expected_strength": float,# fuerza del XI típico top-11
        "found": int,              # jugadores encontrados en el modelo
        "not_found": list,         # jugadores no encontrados
      }
    """
    parquet_team = resolve_team(team_name)
    default = {
        "team":               team_name,
        "factor":             1.0,
        "xi_strength":        None,
        "expected_strength":  None,
        "found":              0,
        "not_found":          xi,
        "note":               "sin datos en parquet",
    }

    if parquet_team is None or parquet_team not in model:
        return default

    tm = model[parquet_team]
    expected_xi = tm["expected_xi_strength"]
    if expected_xi <= 0:
        return {**default, "note": "expected_xi=0"}

    xi_strengths = []
    found, not_found = 0, []

    for player in xi:
        s = _find_player_strength(player, tm)
        if s is not None:
            xi_strengths.append(s)
            found += 1
        else:
            not_found.append(player)
            # Usamos la media del equipo como proxy para jugadores desconocidos
            team_avg = sum(tm["players"].values()) / len(tm["players"]) if tm["players"] else 0
            xi_strengths.append(team_avg * 0.85)  # ligero descuento por incertidumbre

    xi_total = sum(xi_strengths)
    raw_factor = xi_total / expected_xi
    factor = max(FACTOR_MIN, min(FACTOR_MAX, raw_factor))

    return {
        "team":               parquet_team,
        "factor":             round(factor, 4),
        "xi_strength":        round(xi_total, 3),
        "expected_strength":  round(expected_xi, 3),
        "found":              found,
        "not_found":          not_found,
        "note":               f"{found}/{len(xi)} jugadores en modelo",
    }


# ─────────────────────────────────────────────
# POISSON / PROBABILIDADES (standalone, sin importar predict_ucl_v2)
# ─────────────────────────────────────────────

def _ppmf(k: int, lam: float) -> float:
    if lam <= 0 or math.isnan(lam):
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))


def poisson_1x2(lh: float, la: float, max_g: int = 9) -> tuple:
    pl = pd_ = pv = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _ppmf(h, lh) * _ppmf(a, la)
            if h > a:    pl  += p
            elif h == a: pd_ += p
            else:        pv  += p
    total = pl + pd_ + pv
    return (pl / total, pd_ / total, pv / total) if total else (1/3, 1/3, 1/3)


def smooth_draw(pl: float, pd_: float, pv: float, floor: float = 0.15) -> tuple:
    if pd_ >= floor:
        return pl, pd_, pv
    deficit = floor - pd_
    lv = pl + pv
    if lv <= 0:
        return pl, floor, pv
    pl -= deficit * (pl / lv)
    pv -= deficit * (pv / lv)
    return pl, floor, pv


# ─────────────────────────────────────────────
# FORMATO DE SALIDA
# ─────────────────────────────────────────────

W = 72


def _bar(prob_pct: float, width: int = 20) -> str:
    """Barra de probabilidad."""
    filled = round(prob_pct / 100 * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _parse_playdoit(raw: dict | None) -> dict | None:
    """
    Normaliza odds de Playdoit desde el JSON de alineaciones.
    Acepta cualquiera de estos formatos:
      {"local": 1.70, "empate": 3.50, "visitante": 4.50}
      {"home": 1.70, "draw": 3.50, "away": 4.50}
    Retorna dict con {o_h, o_d, o_a, p_h, p_d, p_a, overround_pct} o None.
    """
    if not raw:
        return None
    try:
        o_h = float(raw.get("local") or raw.get("home") or 0)
        o_d = float(raw.get("empate") or raw.get("draw") or 0)
        o_a = float(raw.get("visitante") or raw.get("away") or 0)
        if not all(o > 1.0 for o in [o_h, o_d, o_a]):
            return None
        p_h, p_d, p_a = no_vig(o_h, o_d, o_a)
        return {
            "o_h": o_h, "o_d": o_d, "o_a": o_a,
            "p_h": round(p_h * 100, 1),
            "p_d": round(p_d * 100, 1),
            "p_a": round(p_a * 100, 1),
            "overround_pct": overround_pct(o_h, o_d, o_a),
        }
    except (TypeError, ValueError):
        return None


def print_match_report(row: pd.Series, odds_item: dict | None,
                       home_lf: dict | None, away_lf: dict | None,
                       playdoit_odds: dict | None = None):
    """Imprime el reporte completo de un partido."""
    home    = row["local"]
    away    = row["visitante"]
    date    = row["fecha"]
    fase    = row.get("fase", "")
    ph_base = float(row["p_local_%"])
    pd_base = float(row["p_empate_%"])
    pv_base = float(row["p_visitante_%"])
    lh_base = float(row["lambda_local"])
    la_base = float(row["lambda_visitante"])

    print(f"\n{'─'*W}")
    print(f"  {date}  [{fase}]")
    print(f"  {home:<28} vs  {away}")
    print()

    # ── Predicción base ──
    print(f"  PREDICCION BASE (Dixon-Coles)")
    print(f"  lambda = {lh_base:.3f} vs {la_base:.3f}")
    print(f"  Local {ph_base:.1f}%  {_bar(ph_base)}  "
          f"Empate {pd_base:.1f}%  Visitante {pv_base:.1f}%")

    # ── Clasificación (si hay ──
    p_clf_h = row.get("p_clasif_local_%")
    p_clf_v = row.get("p_clasif_visit_%")
    if pd.notna(p_clf_h) and pd.notna(p_clf_v) and float(p_clf_h) + float(p_clf_v) > 0:
        ida     = row.get("resultado_ida", "")
        fav_clf = row.get("favorito_clasificacion", "")
        print(f"  P(clasifica): {home} {p_clf_h:.1f}%  |  {away} {p_clf_v:.1f}%  "
              f"← ida: {ida} | favorito: {fav_clf}")

    # ── Ajuste por alineación ──
    has_lineup = (home_lf is not None or away_lf is not None)
    lh_adj = lh_base
    la_adj = la_base

    if has_lineup:
        print()
        print(f"  AJUSTE POR ALINEACION:")

        if home_lf:
            fh = home_lf["factor"]
            lh_adj = lh_base * fh
            note_h = home_lf["note"]
            delta_h = (fh - 1) * 100
            print(f"  {home:<24} factor={fh:.3f} ({delta_h:+.1f}%)  [{note_h}]")
            if home_lf.get("not_found"):
                print(f"    No encontrados: {', '.join(home_lf['not_found'][:5])}")

        if away_lf:
            fa = away_lf["factor"]
            la_adj = la_base * fa
            note_a = away_lf["note"]
            delta_a = (fa - 1) * 100
            print(f"  {away:<24} factor={fa:.3f} ({delta_a:+.1f}%)  [{note_a}]")
            if away_lf.get("not_found"):
                print(f"    No encontrados: {', '.join(away_lf['not_found'][:5])}")

        # Probabilidades ajustadas
        ph_adj, pd_adj, pv_adj = smooth_draw(*poisson_1x2(lh_adj, la_adj))
        ph_adj *= 100; pd_adj *= 100; pv_adj *= 100

        print()
        print(f"  PREDICCION AJUSTADA:")
        print(f"  lambda = {lh_adj:.3f} vs {la_adj:.3f}")
        print(f"  Local {ph_adj:.1f}%  {_bar(ph_adj)}  "
              f"Empate {pd_adj:.1f}%  Visitante {pv_adj:.1f}%")

        # Usar probabilidades ajustadas para el edge
        ph_use, pd_use, pv_use = ph_adj, pd_adj, pv_adj
    else:
        ph_use, pd_use, pv_use = ph_base, pd_base, pv_base

    # ── Odds de mercado + edge ──
    # Prioridad: Playdoit (ingresado manualmente) > Pinnacle/Betfair (The Odds API)
    primary     = None   # casa donde se apuesta → base del EV
    primary_name = ""
    pinnacle    = None   # referencia de mercado eficiente
    avg         = None
    n_bk        = 0

    if playdoit_odds:
        primary      = playdoit_odds
        primary_name = "PLAYDOIT"

    if odds_item:
        pref = odds_item.get("preferred")
        avg  = odds_item.get("avg_market")
        n_bk = odds_item.get("n_bookmakers", 0)
        if pref:
            pinnacle = pref
            if primary is None:
                # Sin Playdoit: usar Pinnacle como primario
                primary      = pref
                primary_name = pref.get("bookmaker_key", "?").upper()

    if primary:
        print()
        header = f"  ODDS — {primary_name}" if primary_name else "  MERCADO"
        if n_bk:
            header += f"  (+ referencia {n_bk} casas via Odds API)"
        print(header)

        vig   = primary.get("overround_pct", 0)
        edge_h = calc_edge(ph_use, primary["p_h"])
        edge_d = calc_edge(pd_use, primary["p_d"])
        edge_v = calc_edge(pv_use, primary["p_a"])
        ev_h   = ev_pct(ph_use / 100, primary["o_h"])
        ev_d   = ev_pct(pd_use / 100, primary["o_d"])
        ev_v   = ev_pct(pv_use / 100, primary["o_a"])

        print(f"  {primary_name:<12} vig={vig:.1f}%")
        print(f"  Local     {primary['o_h']:.2f}  ({primary['p_h']:.1f}%)  "
              f"modelo={ph_use:.1f}%  {format_edge_label(edge_h)}  EV={ev_h:+.1f}%")
        print(f"  Empate    {primary['o_d']:.2f}  ({primary['p_d']:.1f}%)  "
              f"modelo={pd_use:.1f}%  {format_edge_label(edge_d)}  EV={ev_d:+.1f}%")
        print(f"  Visitante {primary['o_a']:.2f}  ({primary['p_a']:.1f}%)  "
              f"modelo={pv_use:.1f}%  {format_edge_label(edge_v)}  EV={ev_v:+.1f}%")

        # Referencia Pinnacle (solo si Playdoit fue el primario)
        if playdoit_odds and pinnacle:
            vig_pin = pinnacle.get("overround_pct", 0)
            bk_pin  = pinnacle.get("bookmaker_key", "?").upper()
            print(f"  {bk_pin:<12} vig={vig_pin:.1f}%  "
                  f"(ref)  L:{pinnacle['o_h']:.2f}({pinnacle['p_h']:.1f}%)  "
                  f"E:{pinnacle['o_d']:.2f}({pinnacle['p_d']:.1f}%)  "
                  f"V:{pinnacle['o_a']:.2f}({pinnacle['p_a']:.1f}%)")
            # Comparar Playdoit vs Pinnacle (¿qué tan buena es la línea?)
            diff_h = primary["p_h"] - pinnacle["p_h"]
            diff_v = primary["p_a"] - pinnacle["p_a"]
            overcut = []
            if diff_h > 3:
                overcut.append(f"Local sobrevaluado en Playdoit (+{diff_h:.1f}% vs Pinnacle)")
            if diff_v > 3:
                overcut.append(f"Visitante sobrevaluado en Playdoit (+{diff_v:.1f}% vs Pinnacle)")
            if overcut:
                for o in overcut:
                    print(f"  !! {o}")

        # Resaltar value bets
        value_bets = []
        if edge_h >= 5:
            value_bets.append(
                f"Local @ {primary['o_h']:.2f} {primary_name}  "
                f"(edge={edge_h:+.1f}%, EV={ev_h:+.1f}%)"
            )
        if edge_d >= 5:
            value_bets.append(
                f"Empate @ {primary['o_d']:.2f} {primary_name}  "
                f"(edge={edge_d:+.1f}%, EV={ev_d:+.1f}%)"
            )
        if edge_v >= 5:
            value_bets.append(
                f"Visitante @ {primary['o_a']:.2f} {primary_name}  "
                f"(edge={edge_v:+.1f}%, EV={ev_v:+.1f}%)"
            )

        if value_bets:
            print()
            print(f"  *** APUESTAS CON VALOR POTENCIAL ({primary_name}) ***")
            for vb in value_bets:
                print(f"    -> {vb}")
        else:
            print(f"  (sin edge significativo en 1X2)")

        if avg and not playdoit_odds:
            print(f"  Mercado medio ({n_bk} casas): "
                  f"L {avg['p_h']:.1f}%  E {avg['p_d']:.1f}%  V {avg['p_a']:.1f}%")
    else:
        print(f"  [sin odds — agrega odds de Playdoit en el JSON o verifica Odds API]")


# ─────────────────────────────────────────────
# CARGA DE ARCHIVO DE ALINEACIONES
# ─────────────────────────────────────────────

def load_lineup_file(path: str) -> dict:
    """
    Carga un archivo JSON de alineaciones.
    Formato esperado:
    {
      "matches": [
        {
          "home": "Barcelona",
          "away": "Newcastle United",
          "home_formation": "4-3-3",   (opcional)
          "away_formation": "4-3-3",   (opcional)
          "home_xi": ["Ter Stegen", "Kounde", ...],
          "away_xi":  ["Pope", "Trippier", ...]
        },
        ...
      ]
    }
    Retorna dict keyed by (home_lower, away_lower) → match_dict.
    """
    p = Path(path)
    if not p.exists():
        print(f"  ERROR: archivo de alineaciones no encontrado: {path}")
        sys.exit(1)

    data = json.loads(p.read_bytes().decode("utf-8"))
    matches = data if isinstance(data, list) else data.get("matches", [])

    lineup_map = {}
    for m in matches:
        key = (_norm_key(m.get("home", "")), _norm_key(m.get("away", "")))
        lineup_map[key] = m

    print(f"  [lineup] Alineaciones cargadas: {len(lineup_map)} partidos")
    return lineup_map


def _norm_key(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"\b(fc|cf|sc|ac|fk|sk)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _find_lineup(home: str, away: str, lineup_map: dict) -> dict | None:
    """Busca la alineación para (home, away) usando fuzzy matching."""
    key_h = _norm_key(home)
    key_a = _norm_key(away)

    # Exacto
    if (key_h, key_a) in lineup_map:
        return lineup_map[(key_h, key_a)]

    # Fuzzy
    best, best_score = None, 0.70
    for (h, a), m in lineup_map.items():
        sh = SequenceMatcher(None, key_h, h).ratio()
        sa = SequenceMatcher(None, key_a, a).ratio()
        score = (sh + sa) / 2
        if score > best_score:
            best_score, best = score, m
    return best


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(sport: str = "UCL", lineup_file: str | None = None,
         preds_csv: str | None = None, odds_ttl: int = 30):
    W_LINE = "=" * W
    print(f"\n{W_LINE}")
    print(f"  LINEUP WATCHER — {sport}")
    print(f"  Prediccion base + Odds en vivo + Ajuste por alineacion")
    print(f"{W_LINE}")

    # ── Cargar predicciones base ──
    csv_path = Path(preds_csv) if preds_csv else PREDS_UCL
    if not csv_path.exists():
        print(f"\n  ERROR: No se encontro {csv_path}")
        print(f"  Ejecuta primero: python predict_ucl_v2.py --days 3 --refresh")
        sys.exit(1)

    preds = pd.read_csv(csv_path)
    print(f"\n  Predicciones cargadas: {len(preds)} partidos desde {csv_path.name}")

    # ── Construir modelo de fuerza de jugadores ──
    print("\n  Construyendo modelo de fuerza de jugadores...")
    player_model = build_player_strength_model()

    # ── Cargar alineaciones (si hay) ──
    lineup_map = {}
    if lineup_file:
        print(f"\n  Cargando alineaciones desde {lineup_file}...")
        lineup_map = load_lineup_file(lineup_file)

    # ── Cargar odds ──
    print(f"\n  Descargando odds ({sport})...")
    odds_list = []
    try:
        odds_list = fetch_odds(sport, ttl_minutes=odds_ttl)
        print(f"  Odds obtenidas: {len(odds_list)} partidos")
    except Exception as e:
        print(f"  AVISO: No se pudieron obtener odds — {e}")

    # ── Procesar cada partido ──
    print(f"\n{W_LINE}")
    print(f"  ANALISIS POR PARTIDO")
    print(f"{W_LINE}")

    for _, row in preds.iterrows():
        home = str(row["local"])
        away = str(row["visitante"])

        # Odds
        odds_item = find_match(home, away, odds_list) if odds_list else None

        # Alineaciones
        lineup_data = _find_lineup(home, away, lineup_map) if lineup_map else None
        home_lf = away_lf = None

        playdoit_odds = None
        if lineup_data:
            home_xi = lineup_data.get("home_xi", [])
            away_xi = lineup_data.get("away_xi", [])
            home_form = lineup_data.get("home_formation", "")
            away_form = lineup_data.get("away_formation", "")

            # Odds manuales de Playdoit (opcionales, por partido)
            playdoit_odds = _parse_playdoit(lineup_data.get("playdoit"))

            if home_xi:
                home_lf = compute_lineup_factor(home_xi, home, player_model)
                fmt_xi  = " | ".join(home_xi)
                print(f"\n  {home} ({home_form}): {fmt_xi}")
            if away_xi:
                away_lf = compute_lineup_factor(away_xi, away, player_model)
                fmt_xi  = " | ".join(away_xi)
                print(f"  {away} ({away_form}): {fmt_xi}")

        print_match_report(row, odds_item, home_lf, away_lf, playdoit_odds=playdoit_odds)

    print(f"\n{'─'*W}")
    print(f"  NOTAS:")
    print(f"  - Edge = nuestro modelo % - mercado no-vig %")
    print(f"  - EV > 0 significa apuesta con valor esperado positivo")
    print(f"  - El edge NO garantiza ganancia; indica dónde nuestro modelo")
    print(f"    difiere del mercado. Verifica con contexto (lesiones, rotación).")
    if not lineup_file:
        print(f"  - Para ajuste por alineacion: --lineup data/lineups/hoy.json")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Lineup Watcher — Predicciones + Odds + Alineaciones"
    )
    ap.add_argument("--sport",   default="UCL",
                    choices=list(SPORT_KEYS.keys()),
                    help="Competición (default: UCL)")
    ap.add_argument("--lineup",  default=None,
                    help="Ruta al archivo JSON de alineaciones")
    ap.add_argument("--csv",     default=None,
                    help="Ruta al CSV de predicciones (default: data/predicciones_ucl_v2.csv)")
    ap.add_argument("--ttl",     type=int, default=30,
                    help="TTL del cache de odds en minutos (0=forzar descarga, default=30)")
    args = ap.parse_args()
    main(sport=args.sport, lineup_file=args.lineup,
         preds_csv=args.csv, odds_ttl=args.ttl)
