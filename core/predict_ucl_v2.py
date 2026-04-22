"""
=====================================================================
  Predicciones UCL v2 — Dixon-Coles + Clasificación + Bet Slips
=====================================================================

MEJORAS SOBRE predict_ucl.py:

  1. DIXON-COLES τ correction
     Corrige correlación negativa en 0-0/1-0/0-1/1-1 (Dixon & Coles 1997).
     El Poisson puro sobreestima 0-0 y subestima 1-1.

  2. PRIMER PARTIDO → PROBABILIDAD DE CLASIFICACIÓN
     Para partidos de vuelta, busca resultado de ida y calcula P(clasifica)
     via simulación discreta de todos los scorelines posibles.

  3. LAMBDA MULTIPLICATIVO
     lambda_h = att_h × def_a × mu_ucl × home_adv
     (antes: lambda = (att + def_rival)/2 ← incorrecto)

  4. HOME ADVANTAGE UCL CALIBRADO desde histórico UCL.

  5. DRAW FLOOR DINÁMICO (proporcional al lambda promedio).

  6. CONFIANZA BASADA EN EVIDENCIA (margen + cantidad de datos UCL).

  7. TRES MODOS DE APUESTA por partido:
     ─ PATA SIMPLE (segura):  mejor selección única del partido
     ─ SGP 2 PATAS  (media):  resultado + mercado secundario (BTTS/O-U)
     ─ SGP SOÑADOR  (riesgo): resultado + BTTS + marcador exacto más prob.

     Mercados calculados desde la matriz DC:
       · 1X2 (Local/Empate/Visitante)
       · Over/Under 1.5, 2.5, 3.5 goles totales
       · BTTS (ambos anotan) — P(lh>=1) × P(la>=1)
       · Marcador exacto más probable

Uso:
  python predict_ucl_v2.py [--days 3] [--rho -0.13]
=====================================================================
"""

import argparse
import json
import math
import re
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

FDORG_TOKEN   = "1f7a12b6f18f418caa88a0f59884e80a"
FDORG_BASE    = "https://api.football-data.org/v4"
FDORG_RATE    = 6.5   # segundos entre peticiones (free tier = 10 req/min)

DATA_DIR  = Path("data/processed")
CACHE_DIR = Path("data/_ucl_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UCL_SEASONS = [2023, 2024, 2025]
DOM_SEASONS = ["2324", "2425", "2526"]

# Dixon-Coles rho — correlación baja entre goles local y visitante
# Valor estándar literatura: -0.13 (Dixon & Coles 1997, English football)
# Para UCL ligeramente menos extremo: -0.10 a -0.13
DC_RHO_DEFAULT = -0.13

# Home advantage UCL knockout (calculado abajo desde histórico)
# Si no hay suficientes datos, fallback a este valor conservador
UCL_HOME_ADV_FALLBACK = 1.18

# Media goles UCL por partido
UCL_MU_HOME = 1.65   # goles locales promedio UCL
UCL_MU_AWAY = 1.20   # goles visitantes promedio UCL

# Ventana de partidos para calcular ratings
WINDOW = 10          # últimos N partidos UCL por equipo

# Draw floor dinámico: solo se activa si p_draw < floor calculado
# Floor = DRAW_K / (1 + lambda_promedio) — decae con más goles esperados
DRAW_K = 0.20        # calibrado para UCL (~0.20 a 0.22 en knockouts)

# Penaltis: si empate en doble vuelta se juega AET+penaltis
# P(clasificación para equipo de mayor calidad) ≈ 50/50 ajustado por ELO.
# Simplificación: 50/50 es suficiente sin datos de penaltis.
P_PENS_HOME = 0.50


# ─────────────────────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES
# ─────────────────────────────────────────────────────────────

_TEAM_ALIASES = {
    "fc barcelona":                "barcelona",
    "real madrid cf":              "real madrid",
    "club atletico de madrid":     "atletico madrid",
    "atletico de madrid":          "atletico madrid",
    "fc bayern munchen":           "bayern munich",
    "fc bayern münchen":           "bayern munich",
    "bayer 04 leverkusen":         "bayer leverkusen",
    "borussia dortmund":           "dortmund",
    "paris saint-germain fc":      "paris saint-germain",
    "arsenal fc":                  "arsenal",
    "chelsea fc":                  "chelsea",
    "liverpool fc":                "liverpool",
    "manchester city fc":          "manchester city",
    "newcastle united fc":         "newcastle united",
    "tottenham hotspur fc":        "tottenham",
    "atalanta bc":                 "atalanta",
    "galatasaray sk":              "galatasaray",
    "galatasaray a.s.":            "galatasaray",
    "sporting clube de portugal":  "sporting cp",
    "fk bodo/glimt":               "bodo/glimt",
    "fk bodoe/glimt":              "bodo/glimt",
    "inter milan":                 "inter",
    "internazionale":              "inter",
    "ac milan":                    "milan",
    "juventus fc":                 "juventus",
    "as monaco fc":                "monaco",
    "olympique lyonnais":          "lyon",
    "olympique de marseille":      "marseille",
    "rb leipzig":                  "leipzig",
    "rasenballsport leipzig":      "leipzig",
    "aston villa fc":              "aston villa",
    "celtic fc":                   "celtic",
    "sl benfica":                  "benfica",
    "fc porto":                    "porto",
    "psv eindhoven":               "psv",
    "feyenoord":                   "feyenoord",
    "club brugge kv":              "brugge",
    "dinamo zagreb":               "dinamo zagreb",
    "shakhtar donetsk":            "shakhtar",
}


def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(
        r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd|sad|sae|a\.s\.|s\.a\.|kv|sl|rb)\b",
        "", name,
    )
    name = re.sub(r"\s+", " ", name).strip()
    return _TEAM_ALIASES.get(name, name)


def _best_match(target: str, candidates, threshold: float = 0.68):
    t = _norm(target)
    best_name, best_ratio = None, threshold
    for c in candidates:
        r = SequenceMatcher(None, t, _norm(str(c))).ratio()
        if r > best_ratio:
            best_ratio, best_name = r, c
    return best_name


# ─────────────────────────────────────────────────────────────
# HTTP — football-data.org
# ─────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-Auth-Token": FDORG_TOKEN, "Accept": "application/json"})
    return s


def _get(session, url: str, cache: Path = None, no_cache: bool = False):
    if cache and cache.exists() and not no_cache:
        return json.loads(cache.read_text(encoding="utf-8"))
    time.sleep(FDORG_RATE)
    r = session.get(url, timeout=25)
    r.raise_for_status()
    data = r.json()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


# ─────────────────────────────────────────────────────────────
# CARGA DATOS UCL
# ─────────────────────────────────────────────────────────────

def load_ucl_history(session, refresh_current: bool = False) -> pd.DataFrame:
    """Carga todos los partidos UCL terminados de las temporadas configuradas.
    refresh_current=True fuerza re-descarga de la temporada más reciente."""
    records = []
    for season in UCL_SEASONS:
        path = CACHE_DIR / f"ucl_finished_{season}.json"
        # Siempre re-descarga la temporada actual (en curso)
        is_current = (season == max(UCL_SEASONS))
        no_cache   = refresh_current and is_current
        url  = f"{FDORG_BASE}/competitions/CL/matches?status=FINISHED&season={season}"
        try:
            data = _get(session, url, path, no_cache=no_cache)
        except Exception as e:
            print(f"  [warn] UCL {season}: {e}")
            continue
        for m in data.get("matches", []):
            ft = m.get("score", {}).get("fullTime", {})
            hg, ag = ft.get("home"), ft.get("away")
            if hg is None or ag is None:
                continue
            records.append({
                "season":     season,
                "date":       m["utcDate"][:10],
                "stage":      m.get("stage", ""),
                "matchday":   m.get("matchday"),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
                "home_goals": int(hg),
                "away_goals": int(ag),
            })

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  UCL histórico: {len(df)} partidos ({UCL_SEASONS[0]}–{UCL_SEASONS[-1]})")
    return df


def load_ucl_upcoming(session, days_ahead: int = 7) -> pd.DataFrame:
    """Carga fixtures UCL programados en los próximos days_ahead días."""
    today_str = datetime.now().strftime("%Y%m%d")
    path = CACHE_DIR / f"ucl_scheduled_{today_str}.json"
    url  = f"{FDORG_BASE}/competitions/CL/matches?status=SCHEDULED"
    try:
        data = _get(session, url, path)
    except Exception as e:
        print(f"  [error] Fixtures: {e}")
        return pd.DataFrame()

    today    = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    rows = []
    for m in data.get("matches", []):
        d = datetime.fromisoformat(m["utcDate"][:10]).date()
        h = m["homeTeam"]["name"]
        a = m["awayTeam"]["name"]
        # Omitir fixtures sin equipos confirmados (draw pendiente)
        if h is None or a is None:
            continue
        if today <= d <= end_date:
            rows.append({
                "date":      d,
                "stage":     m.get("stage", ""),
                "matchday":  m.get("matchday"),
                "home_team": h,
                "away_team": a,
            })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# PRIMER PARTIDO (para partidos de vuelta)
# ─────────────────────────────────────────────────────────────

def find_first_leg(ucl_hist: pd.DataFrame, second_home: str, second_away: str,
                   second_date) -> dict | None:
    """
    Busca el partido de ida (primer partido) dado el de vuelta.
    El primer partido tiene roles invertidos: second_away jugó en casa.
    Devuelve dict con 'home_goals'/'away_goals' del primer partido,
    y los nombres de 'home_team'/'away_team' como se jugó en ida.
    """
    if ucl_hist.empty:
        return None

    all_teams = pd.concat([ucl_hist["home_team"], ucl_hist["away_team"]]).unique()
    h_match = _best_match(second_home, all_teams)
    a_match = _best_match(second_away, all_teams)
    if h_match is None or a_match is None:
        return None

    ts = pd.Timestamp(second_date)
    # Ventana de búsqueda: 7 a 60 días antes
    window_start = ts - timedelta(days=60)
    window_end   = ts - timedelta(days=7)

    # El primer partido: second_away jugó en CASA, second_home jugó FUERA
    mask = (
        (ucl_hist["date"] >= window_start)
        & (ucl_hist["date"] <= window_end)
        & (ucl_hist["home_team"] == a_match)
        & (ucl_hist["away_team"] == h_match)
    )
    candidates = ucl_hist[mask]

    if candidates.empty:
        # También buscar al revés (por si acaso)
        mask2 = (
            (ucl_hist["date"] >= window_start)
            & (ucl_hist["date"] <= window_end)
            & (ucl_hist["home_team"] == h_match)
            & (ucl_hist["away_team"] == a_match)
        )
        candidates = ucl_hist[mask2]
        if candidates.empty:
            return None
        row = candidates.iloc[-1]
        return {
            "date":              row["date"].strftime("%Y-%m-%d"),
            "first_home":        row["home_team"],   # = second_home
            "first_away":        row["away_team"],   # = second_away
            "first_home_goals":  int(row["home_goals"]),
            "first_away_goals":  int(row["away_goals"]),
            # Desde perspectiva del partido de VUELTA:
            # second_home necesita superar el déficit/ventaja del primer partido
            "goals_favor_second_home": int(row["away_goals"]),  # segundo local marcó estos
            "goals_favor_second_away": int(row["home_goals"]),  # segundo visitante marcó estos
        }

    row = candidates.iloc[-1]  # el más reciente en el rango
    return {
        "date":              row["date"].strftime("%Y-%m-%d"),
        "first_home":        row["home_team"],   # = second_away
        "first_away":        row["away_team"],   # = second_home
        "first_home_goals":  int(row["home_goals"]),
        "first_away_goals":  int(row["away_goals"]),
        # En la vuelta, second_home fue el visitante de la ida
        "goals_favor_second_home": int(row["away_goals"]),  # marcados como visitante en ida
        "goals_favor_second_away": int(row["home_goals"]),  # marcados como local en ida
    }


# ─────────────────────────────────────────────────────────────
# RATINGS MULTIPLICATIVOS — Dixon-Coles style
# ─────────────────────────────────────────────────────────────

def compute_home_advantage(ucl_hist: pd.DataFrame) -> float:
    """
    Calibra ventaja local desde histórico UCL.
    alpha = media(goles_local) / media(goles_visitante), solo knockouts.
    """
    if ucl_hist.empty:
        return UCL_HOME_ADV_FALLBACK

    ko = ucl_hist[ucl_hist["stage"].isin(
        {"ROUND_OF_16", "LAST_16", "QUARTER_FINALS", "SEMI_FINALS", "FINAL"}
    )]
    if len(ko) < 20:
        ko = ucl_hist  # fallback a todos si pocos knockouts

    mu_home = ko["home_goals"].mean()
    mu_away = ko["away_goals"].mean()
    if mu_away <= 0:
        return UCL_HOME_ADV_FALLBACK
    alpha = round(mu_home / mu_away, 3)
    print(f"  Home advantage UCL calibrado: {alpha:.3f} "
          f"(mu_casa={mu_home:.2f}, mu_visit={mu_away:.2f}, n={len(ko)})")
    return max(1.0, min(alpha, 1.50))


def compute_ratings(ucl_hist: pd.DataFrame, window: int = WINDOW) -> dict:
    """
    Calcula ratings multiplicativos att/def por equipo desde histórico UCL.

    att_h = weighted_mean(home_goals_scored)  / mu_home_ucl
    def_h = weighted_mean(home_goals_conceded)/ mu_away_ucl
    att_a = weighted_mean(away_goals_scored)  / mu_away_ucl
    def_a = weighted_mean(away_goals_conceded)/ mu_home_ucl

    Retorna dict: { team_name: { att_h, def_h, att_a, def_a, n_h, n_a } }
    """
    if ucl_hist.empty:
        return {}

    # Últimas 3 temporadas — mismo peso por ahora
    recent = ucl_hist[ucl_hist["season"].isin(UCL_SEASONS)].copy()

    mu_home = recent["home_goals"].mean()
    mu_away = recent["away_goals"].mean()

    teams = set(recent["home_team"]) | set(recent["away_team"])
    ratings = {}

    def _wmean(series: pd.Series) -> float:
        """Media ponderada reciente > antiguo."""
        n = len(series)
        if n == 0:
            return np.nan
        w = np.arange(1, n + 1, dtype=float)
        return float(np.average(series.values, weights=w))

    for team in teams:
        as_home = (recent[recent["home_team"] == team]
                   .sort_values("date")
                   .tail(window))
        as_away = (recent[recent["away_team"] == team]
                   .sort_values("date")
                   .tail(window))

        n_h = len(as_home)
        n_a = len(as_away)

        att_h = (_wmean(as_home["home_goals"]) / mu_home) if n_h >= 2 else 1.0
        def_h = (_wmean(as_home["away_goals"]) / mu_away) if n_h >= 2 else 1.0
        att_a = (_wmean(as_away["away_goals"]) / mu_away) if n_a >= 2 else 1.0
        def_a = (_wmean(as_away["home_goals"]) / mu_home) if n_a >= 2 else 1.0

        # Forma reciente (últimos 3 resultados)
        all_games = pd.concat([
            as_home[["date", "home_goals", "away_goals"]].rename(
                columns={"home_goals": "gf", "away_goals": "ga"}),
            as_away[["date", "away_goals", "home_goals"]].rename(
                columns={"away_goals": "gf", "home_goals": "ga"}),
        ]).sort_values("date").tail(3)
        forma = "".join(
            "W" if r["gf"] > r["ga"] else ("D" if r["gf"] == r["ga"] else "L")
            for _, r in all_games.iterrows()
        )

        ratings[team] = {
            "att_h": round(att_h, 3), "def_h": round(def_h, 3), "n_h": n_h,
            "att_a": round(att_a, 3), "def_a": round(def_a, 3), "n_a": n_a,
            "forma": forma,
        }

    return ratings, mu_home, mu_away


def get_team_rating(ratings: dict, team_raw: str, fallback: float = 1.0) -> dict:
    """Busca rating con fuzzy matching."""
    matched = _best_match(team_raw, list(ratings.keys()))
    if matched:
        return ratings[matched], matched
    return {
        "att_h": fallback, "def_h": fallback,
        "att_a": fallback, "def_a": fallback,
        "n_h": 0, "n_a": 0, "forma": "",
    }, None


# ─────────────────────────────────────────────────────────────
# FORMA DOMÉSTICA (fallback/blend)
# ─────────────────────────────────────────────────────────────

def _wmean_series(s: pd.Series) -> float:
    n = len(s)
    if n == 0:
        return np.nan
    return float(np.average(s.values, weights=np.arange(1, n + 1)))


def get_domestic_form(schedule: pd.DataFrame, team_raw: str, before_date,
                      window: int = 6) -> tuple:
    """
    Retorna (xg_att, xg_def, n, liga) para blend con UCL.
    Usa xG doméstico ponderado.
    """
    if schedule.empty:
        return 1.0, 1.0, 0, ""

    all_teams = pd.concat([schedule["home_team"], schedule["away_team"]]).unique()
    matched   = _best_match(team_raw, all_teams)
    if not matched:
        return 1.0, 1.0, 0, ""

    sub = schedule[(schedule["home_team"] == matched) | (schedule["away_team"] == matched)]
    liga = sub.iloc[-1]["league"] if not sub.empty else ""

    as_h = sub[sub["home_team"] == matched][["date", "home_xg", "away_xg"]].rename(
        columns={"home_xg": "xg_for", "away_xg": "xg_against"})
    as_a = sub[sub["away_team"] == matched][["date", "away_xg", "home_xg"]].rename(
        columns={"away_xg": "xg_for", "home_xg": "xg_against"})

    recent = (pd.concat([as_h, as_a])
              .sort_values("date")
              .loc[lambda d: d["date"] < pd.Timestamp(before_date)]
              .tail(window))

    if recent.empty:
        return 1.0, 1.0, 0, liga

    return (
        round(_wmean_series(recent["xg_for"]),     3),
        round(_wmean_series(recent["xg_against"]), 3),
        len(recent),
        liga,
    )


# ─────────────────────────────────────────────────────────────
# LAMBDA MULTIPLICATIVO + BLEND UCL/DOMÉSTICO
# ─────────────────────────────────────────────────────────────

def calc_lambda(
    att_team: float, def_rival: float,
    mu_ucl: float, home_adv: float,
    dom_att: float, dom_def_rival: float,
    ucl_n: int,
    is_home: bool,
) -> float:
    """
    lambda = blend(UCL, domestic) con peso UCL proporcional a datos disponibles.

    UCL: att_team × def_rival × mu_ucl × home_adv  (si es local)
         att_team × def_rival × mu_ucl               (si es visitante)
    DOM: dom_att / dom_def_rival × mu_ucl × home_adv (escalado)
    """
    ha = home_adv if is_home else 1.0

    # Lambda UCL multiplicativo
    lam_ucl = att_team * def_rival * mu_ucl * ha
    lam_ucl = max(0.3, min(lam_ucl, 6.0))

    # Lambda doméstico (escalado a mu_ucl)
    if dom_def_rival > 0 and dom_att > 0:
        dom_ratio = dom_att / dom_def_rival
        lam_dom = dom_ratio * mu_ucl * ha
        lam_dom = max(0.3, min(lam_dom, 6.0))
    else:
        lam_dom = mu_ucl * ha

    # Peso UCL: crece linealmente de 0 a MAX_UCL_WEIGHT con los partidos
    MAX_UCL_W = 0.80
    ucl_w = min(ucl_n / 6.0, 1.0) * MAX_UCL_W
    dom_w = 1.0 - ucl_w

    return round(ucl_w * lam_ucl + dom_w * lam_dom, 4)


# ─────────────────────────────────────────────────────────────
# DIXON-COLES τ CORRECTION
# ─────────────────────────────────────────────────────────────

def tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """
    Función de corrección Dixon-Coles para scorelines bajos.
    Corrige la sobreestimación de 0-0 y subestimación de 1-1 del Poisson puro.
    """
    if x == 0 and y == 0:
        return 1.0 - lh * la * rho
    elif x == 1 and y == 0:
        return 1.0 + la * rho
    elif x == 0 and y == 1:
        return 1.0 + lh * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _pmf(k: int, lam: float) -> float:
    if lam <= 0 or not np.isfinite(lam):
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))


def dc_joint_matrix(lh: float, la: float, rho: float = DC_RHO_DEFAULT,
                    max_g: int = 9) -> np.ndarray:
    """
    Calcula la matriz de probabilidades conjuntas (h_goals × a_goals)
    con corrección Dixon-Coles.
    Devuelve matriz [max_g+1, max_g+1] normalizada.
    """
    mat = np.zeros((max_g + 1, max_g + 1))
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            mat[h, a] = tau(h, a, lh, la, rho) * _pmf(h, lh) * _pmf(a, la)
    # Renormalizar (tau puede generar pequeña desviación de 1.0)
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def probs_90min(mat: np.ndarray) -> tuple:
    """Suma la matriz para obtener (p_home, p_draw, p_away)."""
    p_home = float(np.tril(mat, -1).sum())   # home > away
    p_draw = float(np.diag(mat).sum())        # home == away
    p_away = float(np.triu(mat, 1).sum())     # away > home
    total  = p_home + p_draw + p_away
    if total > 0:
        return p_home / total, p_draw / total, p_away / total
    return 1/3, 1/3, 1/3


def draw_floor_dynamic(p_h: float, p_d: float, p_a: float,
                       lh: float, la: float) -> tuple:
    """
    Floor de empate dinámico: a menor lambda promedio, mayor floor.
    Floor = DRAW_K / (1 + (lh + la) / 2)
    Esto refleja que en partidos de baja intensidad ofensiva los empates
    son más probables.
    """
    mu_avg = (lh + la) / 2.0
    floor  = DRAW_K / (1.0 + mu_avg)
    floor  = max(0.10, min(floor, 0.22))  # límites razonables

    if p_d >= floor:
        return p_h, p_d, p_a

    deficit = floor - p_d
    total_ha = p_h + p_a
    if total_ha <= 0:
        return p_h, floor, p_a
    p_h -= deficit * (p_h / total_ha)
    p_a -= deficit * (p_a / total_ha)
    return max(p_h, 0.0), floor, max(p_a, 0.0)


# ─────────────────────────────────────────────────────────────
# SIMULACIÓN DE CLASIFICACIÓN (doble vuelta)
# ─────────────────────────────────────────────────────────────

def simulate_qualification(
    lh2: float, la2: float,
    goals_favor_second_home: int,
    goals_favor_second_away: int,
    rho: float = DC_RHO_DEFAULT,
    max_g: int = 9,
) -> tuple:
    """
    Calcula P(clasificación) para el partido de vuelta dado el resultado de ida.

    args:
      lh2/la2 — lambdas del partido de vuelta
      goals_favor_second_home — goles marcados por el local del partido de vuelta
                                 en el partido de ida (= goles del visitante de ida)
      goals_favor_second_away — goles marcados por el visitante del partido de vuelta
                                 en el partido de ida (= goles del local de ida)
    Retorna:
      (p_clasifica_local_vuelta, p_empate_global, p_clasifica_visitante_vuelta)
    """
    mat = dc_joint_matrix(lh2, la2, rho, max_g)

    p_home_adv   = 0.0  # local de vuelta clasifica en 90 min
    p_away_adv   = 0.0  # visitante de vuelta clasifica en 90 min
    p_aet        = 0.0  # empate en global (va a prórroga)

    g1h = goals_favor_second_home  # goles del local de vuelta en el partido de ida
    g1a = goals_favor_second_away  # goles del visitante de vuelta en el partido de ida

    for h2 in range(max_g + 1):
        for a2 in range(max_g + 1):
            p = mat[h2, a2]
            if p < 1e-10:
                continue
            total_home = g1h + h2  # acumulado local de vuelta
            total_away = g1a + a2  # acumulado visitante de vuelta

            if total_home > total_away:
                p_home_adv += p
            elif total_away > total_home:
                p_away_adv += p
            else:
                # Empate global → prórroga + penaltis
                # Desde 2021 UEFA eliminó gol de visitante doble
                p_aet += p

    # En prórroga: 50/50 ajustado por ventaja local de vuelta
    # (ligera ventaja para el que juega en casa)
    p_home_via_aet = p_aet * (P_PENS_HOME + 0.02)  # leve ventaja local
    p_away_via_aet = p_aet * (1.0 - P_PENS_HOME - 0.02)

    p_home_total = p_home_adv + p_home_via_aet
    p_away_total = p_away_adv + p_away_via_aet

    return round(p_home_total, 4), round(p_aet, 4), round(p_away_total, 4)


# ─────────────────────────────────────────────────────────────
# CONFIANZA
# ─────────────────────────────────────────────────────────────

def semaforo(p1: float, p2: float, p3: float, n_ucl: int) -> str:
    """
    ALTA:  margen > 20% Y >= 5 partidos UCL de contexto
    MEDIA: margen 10-20% O datos escasos (3-4 partidos)
    BAJA:  margen < 10% O sin datos UCL
    """
    probs  = sorted([p1, p2, p3], reverse=True)
    margin = probs[0] - probs[1]
    if margin > 0.20 and n_ucl >= 5:
        return "ALTA"
    elif margin > 0.10 and n_ucl >= 3:
        return "MEDIA"
    return "BAJA"


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

_STAGE_MAP = {
    "LAST_16":         "Octavos",
    "ROUND_OF_16":     "Octavos",
    "QUARTER_FINALS":  "Cuartos",
    "SEMI_FINALS":     "Semis",
    "FINAL":           "Final",
    "GROUP_STAGE":     "Grupos",
    "LEAGUE_PHASE":    "Fase de Liga",
}

def _fmt_stage(s: str) -> str:
    return _STAGE_MAP.get(s, s or "UCL")

def _pct(v, dec=1):
    try:
        return round(float(v) * 100, dec) if v is not None else None
    except Exception:
        return None

def _r(v, dec=3):
    try:
        return round(float(v), dec) if v is not None else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# MERCADOS DE APUESTA — calculados desde la matriz DC
# ─────────────────────────────────────────────────────────────

def p_over(mat: np.ndarray, line: float) -> float:
    """P(total goles > line) desde la matriz de scorelines."""
    total = 0.0
    n = mat.shape[0]
    for h in range(n):
        for a in range(n):
            if (h + a) > line:
                total += mat[h, a]
    return round(total, 4)


def p_btts(lh: float, la: float) -> float:
    """P(ambos equipos anotan >= 1 gol) = P(lh>=1) × P(la>=1)."""
    p_h_scores = 1.0 - _pmf(0, lh)
    p_a_scores = 1.0 - _pmf(0, la)
    return round(p_h_scores * p_a_scores, 4)


def most_likely_score(mat: np.ndarray, top_n: int = 3) -> list:
    """Devuelve los top_n scorelines más probables como [(h, a, prob), ...]."""
    flat = [(h, a, mat[h, a]) for h in range(mat.shape[0]) for a in range(mat.shape[1])]
    flat.sort(key=lambda x: x[2], reverse=True)
    return [(h, a, round(p * 100, 1)) for h, a, p in flat[:top_n]]


def build_markets(lh: float, la: float, mat: np.ndarray,
                  p_h: float, p_d: float, p_a: float) -> dict:
    """
    Calcula todos los mercados disponibles para un partido.
    Retorna dict con probabilidades y texto de selección.
    """
    ov15 = p_over(mat, 1.5)
    ov25 = p_over(mat, 2.5)
    ov35 = p_over(mat, 3.5)
    un15 = 1.0 - ov15
    un25 = 1.0 - ov25
    un35 = 1.0 - ov35
    btts = p_btts(lh, la)
    nbts = 1.0 - btts
    scores = most_likely_score(mat, 5)

    # 1X2 con etiqueta
    m_1x2 = {"Local": p_h, "Empate": p_d, "Visitante": p_a}
    best_1x2 = max(m_1x2, key=m_1x2.get)

    # Over/Under — escoger la línea con más certeza
    ou_lines = [
        ("Over 1.5",  ov15),
        ("Under 1.5", un15),
        ("Over 2.5",  ov25),
        ("Under 2.5", un25),
        ("Over 3.5",  ov35),
        ("Under 3.5", un35),
    ]
    best_ou = max(ou_lines, key=lambda x: x[1])

    # BTTS
    btts_sel = ("BTTS Sí", btts) if btts > 0.5 else ("BTTS No", nbts)

    # Marcador más probable
    top_score = scores[0]

    return {
        "best_1x2":   (best_1x2,   round(m_1x2[best_1x2], 4)),
        "best_ou":    best_ou,
        "btts":       btts_sel,
        "top_score":  top_score,  # (h, a, prob%)
        "scores_top5": scores,
        "ov15": ov15, "un15": un15,
        "ov25": ov25, "un25": un25,
        "ov35": ov35, "un35": un35,
        "btts_p": btts, "nbts_p": nbts,
    }


def bet_single(markets: dict, home: str, away: str) -> dict:
    """
    PATA SIMPLE: selección única con mayor probabilidad del partido.
    Elige entre 1X2, Over/Under y BTTS la que tenga prob más alta.
    """
    candidates = [
        {"sel": markets["best_1x2"][0],  "prob": markets["best_1x2"][1], "market": "1X2"},
        {"sel": markets["best_ou"][0],    "prob": markets["best_ou"][1],  "market": "O/U"},
        {"sel": markets["btts"][0],       "prob": markets["btts"][1],     "market": "BTTS"},
    ]
    best = max(candidates, key=lambda x: x["prob"])
    return {
        "partido":   f"{home} vs {away}",
        "pata":      best["sel"],
        "mercado":   best["market"],
        "prob_%":    round(best["prob"] * 100, 1),
    }


def bet_sgp(markets: dict, home: str, away: str) -> dict:
    """
    SGP 2 PATAS: resultado + mercado secundario más confiable.
    Selecciona el Over/Under o BTTS con prob más alta como segunda pata.
    Solo combina si ambas patas tienen prob > 50%.
    """
    r_sel, r_prob = markets["best_1x2"]

    # Segunda pata: elegir entre OU y BTTS
    second_candidates = [
        {"sel": markets["best_ou"][0], "prob": markets["best_ou"][1], "market": "O/U"},
        {"sel": markets["btts"][0],    "prob": markets["btts"][1],    "market": "BTTS"},
    ]
    second = max(second_candidates, key=lambda x: x["prob"])

    # Probabilidad combinada (asumiendo independencia aproximada)
    joint_prob = r_prob * second["prob"]

    return {
        "partido":      f"{home} vs {away}",
        "pata_1":       r_sel,
        "mercado_1":    "1X2",
        "prob_1_%":     round(r_prob * 100, 1),
        "pata_2":       second["sel"],
        "mercado_2":    second["market"],
        "prob_2_%":     round(second["prob"] * 100, 1),
        "prob_sgp_%":   round(joint_prob * 100, 1),
        "advertencia":  "bajo" if joint_prob < 0.35 else ("medio" if joint_prob < 0.50 else "confiable"),
    }


def bet_dreamer(markets: dict, home: str, away: str) -> dict:
    """
    SGP SOÑADOR: 3 patas — resultado + BTTS + marcador exacto.
    Alta varianza, alta ganancia potencial.
    """
    r_sel, r_prob  = markets["best_1x2"]
    btts_sel, btts_p = markets["btts"]
    h_sc, a_sc, sc_pct = markets["top_score"]
    sc_sel   = f"{h_sc}–{a_sc}"
    sc_prob  = sc_pct / 100.0

    four_way_prob = r_prob * btts_p * sc_prob

    return {
        "partido":          f"{home} vs {away}",
        "pata_1":           r_sel,
        "prob_1_%":         round(r_prob  * 100, 1),
        "pata_2":           btts_sel,
        "prob_2_%":         round(btts_p  * 100, 1),
        "pata_3":           f"Marcador {sc_sel}",
        "prob_3_%":         sc_pct,
        "top5_marcadores":  ", ".join(f"{h}–{a}({p}%)" for h, a, p in markets["scores_top5"]),
        "prob_sgp_%":       round(four_way_prob * 100, 2),
        "advertencia":      "SOÑADOR — riesgo extremo, multiplicador muy alto",
    }


def print_bet_slips(singles: list, sgps: list, dreamers: list):
    """Imprime los bet slips en formato legible."""
    W = 72
    print("\n" + "═" * W)
    print("  🎯  PATA SIMPLE — UNA SELECCIÓN POR PARTIDO (Más segura)")
    print("=" * W)
    for i, s in enumerate(singles, 1):
        print(f"  {i}. {s['partido']:<42}  →  {s['pata']:<20} ({s['prob_%']:.1f}%)")

    print("\n" + "═" * W)
    print("  ⚡  SGP 2 PATAS — RESULTADO + MERCADO SECUNDARIO (Media)")
    print("=" * W)
    for i, s in enumerate(sgps, 1):
        print(f"  {i}. {s['partido']}")
        print(f"     Pata 1: {s['pata_1']:<20} ({s['prob_1_%']:.1f}%)")
        print(f"     Pata 2: {s['pata_2']:<20} ({s['prob_2_%']:.1f}%)")
        print(f"     SGP prob combinada: {s['prob_sgp_%']:.1f}%  "
              f"[confianza: {s['advertencia']}]")

    print("\n" + "═" * W)
    print("  🚀  SGP SOÑADOR — 3 PATAS (Alto riesgo / Alta recompensa)")
    print("=" * W)
    for i, d in enumerate(dreamers, 1):
        print(f"  {i}. {d['partido']}")
        print(f"     Pata 1: {d['pata_1']:<20} ({d['prob_1_%']:.1f}%)")
        print(f"     Pata 2: {d['pata_2']:<20} ({d['prob_2_%']:.1f}%)")
        print(f"     Pata 3: {d['pata_3']:<20} ({d['prob_3_%']:.1f}%)")
        print(f"     Top marcadores: {d['top5_marcadores']}")
        print(f"     Prob combinada:  {d['prob_sgp_%']:.2f}%  ← {d['advertencia']}")
    print("=" * W)
    print("  ⚠️  Las probabilidades son del modelo (sin ajuste de vig/margen casa).")
    print("     Compara con los momios reales para encontrar value.")
    print("=" * W)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main(days_ahead: int = 7, rho: float = DC_RHO_DEFAULT, refresh: bool = False):
    print("=" * 72)
    print("  SOCCER PROJECT — UCL v2 (Dixon-Coles + Clasificación)")
    print("=" * 72)

    session = _session()

    # ── 1. Histórico UCL ──────────────────────────────────────
    print("\n[1/5] Cargando histórico UCL...")
    ucl_hist = load_ucl_history(session, refresh_current=refresh)

    # ── 2. Fixtures próximos ──────────────────────────────────
    print(f"\n[2/5] Cargando fixtures UCL próximos {days_ahead} días...")
    upcoming = load_ucl_upcoming(session, days_ahead)
    if upcoming.empty:
        print("  No hay partidos programados en ese rango.")
        return
    print(f"  Partidos: {len(upcoming)}")

    # ── 3. Ratings UCL + home advantage ───────────────────────
    print("\n[3/5] Calculando ratings UCL multiplicativos...")
    if ucl_hist.empty:
        print("  [warn] Sin histórico UCL, usando defaults.")
        ratings_dict = {}
        mu_home = UCL_MU_HOME
        mu_away = UCL_MU_AWAY
        home_adv = UCL_HOME_ADV_FALLBACK
    else:
        ratings_dict, mu_home, mu_away = compute_ratings(ucl_hist)
        home_adv = compute_home_advantage(ucl_hist)
    mu_ucl = (mu_home + mu_away) / 2.0
    print(f"  mu_UCL = {mu_ucl:.3f} (casa {mu_home:.3f} / fuera {mu_away:.3f})")

    # ── 4. Forma doméstica ────────────────────────────────────
    print("\n[4/5] Cargando xG doméstico...")
    try:
        sched = pd.read_parquet(DATA_DIR / "schedule_xg.parquet").reset_index()
        sched["date"] = pd.to_datetime(sched["date"])
        dom = sched[
            (sched["is_result"] == True) & (sched["season"].isin(DOM_SEASONS))
        ].copy()
        print(f"  Partidos domésticos: {len(dom)}")
    except Exception as e:
        print(f"  [warn] schedule_xg no disponible: {e}")
        dom = pd.DataFrame()

    # ── 5. Predicciones ───────────────────────────────────────
    print(f"\n[5/5] Calculando predicciones con Dixon-Coles (rho={rho})...\n")
    SEP = "-" * 72
    rows    = []
    singles = []
    sgps    = []
    dreamers = []

    for _, match in upcoming.sort_values("date").iterrows():
        home  = match["home_team"]
        away  = match["away_team"]
        date  = match["date"]
        stage = match["stage"]

        # ── Ratings multiplicativos ──
        r_home, _ = get_team_rating(ratings_dict, home)
        r_away, _ = get_team_rating(ratings_dict, away)

        n_ucl_h = r_home["n_h"]  # partidos como local en UCL
        n_ucl_a = r_away["n_a"]  # partidos como visitante en UCL
        n_ucl   = min(n_ucl_h, n_ucl_a)

        # ── Forma doméstica ──
        h_dom_att, h_dom_def, h_dom_n, h_liga = get_domestic_form(dom, home, date)
        a_dom_att, a_dom_def, a_dom_n, a_liga = get_domestic_form(dom, away, date)

        # ── Lambda multiplicativo + blend UCL/dom ──
        # Local (home): usa att_h × def_a_rival_en_UCL_fuera × mu × home_adv
        lh = calc_lambda(
            att_team=r_home["att_h"], def_rival=r_away["def_a"],
            mu_ucl=mu_ucl, home_adv=home_adv,
            dom_att=h_dom_att, dom_def_rival=a_dom_def,
            ucl_n=n_ucl_h,
            is_home=True,
        )
        # Visitante (away): usa att_a × def_h_rival_en_UCL_casa × mu
        la = calc_lambda(
            att_team=r_away["att_a"], def_rival=r_home["def_h"],
            mu_ucl=mu_ucl, home_adv=home_adv,
            dom_att=a_dom_att, dom_def_rival=h_dom_def,
            ucl_n=n_ucl_a,
            is_home=False,
        )

        # ── Dixon-Coles joint matrix ──
        mat = dc_joint_matrix(lh, la, rho)
        mat_raw = dc_joint_matrix(lh, la, rho=0.0)  # Poisson puro para comparar

        p_h_raw, p_d_raw, p_a_raw = probs_90min(mat_raw)
        p_h_dc,  p_d_dc,  p_a_dc  = probs_90min(mat)

        # Draw floor dinámico
        p_h, p_d, p_a = draw_floor_dynamic(p_h_dc, p_d_dc, p_a_dc, lh, la)

        # Predicción 90 min
        if p_h >= p_d and p_h >= p_a:
            pred_90 = "Local"
        elif p_d >= p_h and p_d >= p_a:
            pred_90 = "Empate"
        else:
            pred_90 = "Visitante"

        # ── Primer partido ──
        first_leg = find_first_leg(ucl_hist, home, away, date)

        # ── Clasificación ──
        if first_leg:
            g1h = first_leg["goals_favor_second_home"]
            g1a = first_leg["goals_favor_second_away"]
            p_clasif_home, p_aet, p_clasif_away = simulate_qualification(
                lh, la, g1h, g1a, rho
            )
            if p_clasif_home > p_clasif_away:
                clasif_favorito = home
            else:
                clasif_favorito = away
        else:
            g1h, g1a = None, None
            p_clasif_home = p_clasif_away = p_aet = None
            clasif_favorito = "N/D (sin resultado de ida)"

        # Confianza
        conf = semaforo(p_h, p_d, p_a, n_ucl)

        # ── Consola ──
        print(SEP)
        print(f"  {date.strftime('%d/%m/%y')}  [{_fmt_stage(stage)}]")
        print(f"  {home:28s} vs  {away}")
        print(f"  λ = {lh:.3f} vs {la:.3f}   "
              f"{'Dixon-Coles' if abs(rho) > 0 else 'Poisson'}")
        print(f"  90min: Local {p_h*100:.1f}%  /  Empate {p_d*100:.1f}%  "
              f"/  Visitante {p_a*100:.1f}%   [{pred_90}] [{conf}]")

        if first_leg:
            print(f"  Ida ({first_leg['date']}): "
                  f"{first_leg['first_home']} {first_leg['first_home_goals']}"
                  f" – {first_leg['first_away_goals']} {first_leg['first_away']}")
            print(f"  Global acumulado a favor de {home}: {g1h} goles | {away}: {g1a} goles")
            print(f"  CLASIFICACIÓN → {clasif_favorito}")
            print(f"    {home}: {p_clasif_home*100:.1f}%  "
                  f"| Prórroga: {p_aet*100:.1f}%  "
                  f"| {away}: {p_clasif_away*100:.1f}%")
        else:
            print(f"  [sin resultado de ida — solo probabilidades 90 min]")

        dc_diff = (p_d_dc - p_d_raw) * 100
        print(f"  DC vs Raw: empate +{dc_diff:.1f}%  "
              f"(ratings UCL: att_h={r_home['att_h']:.2f} def_a={r_away['def_a']:.2f})")

        # ── Mercados de apuesta ──
        mkts = build_markets(lh, la, mat, p_h, p_d, p_a)
        singles.append(bet_single(mkts, home, away))
        sgps.append(bet_sgp(mkts, home, away))
        dreamers.append(bet_dreamer(mkts, home, away))

        # Resumen de mercados en consola
        print(f"  Mercados:  1X2 → {mkts['best_1x2'][0]} ({mkts['best_1x2'][1]*100:.0f}%)  "
              f"| {mkts['best_ou'][0]} ({mkts['best_ou'][1]*100:.0f}%)  "
              f"| {mkts['btts'][0]} ({mkts['btts'][1]*100:.0f}%)  "
              f"| Score más prob: {mkts['top_score'][0]}–{mkts['top_score'][1]} ({mkts['top_score'][2]}%)")

        # ── Row CSV ──
        rows.append({
            # Identificación
            "fecha":                      date.strftime("%Y-%m-%d"),
            "fase":                       _fmt_stage(stage),
            "local":                      home,
            "visitante":                  away,
            "liga_local":                 h_liga,
            "liga_visitante":             a_liga,

            # Predicción 90 min (Dixon-Coles)
            "prediccion_90min":           pred_90,
            "p_local_%":                  _pct(p_h),
            "p_empate_%":                 _pct(p_d),
            "p_visitante_%":              _pct(p_a),
            "confianza":                  conf,

            # Lambdas multiplicativos
            "lambda_local":               _r(lh, 3),
            "lambda_visitante":           _r(la, 3),
            "ventaja_goles":              _r(lh - la, 3),
            "home_adv_ucl":               _r(home_adv, 3),

            # Ratings UCL multiplicativos
            "att_h":                      r_home["att_h"],
            "def_h":                      r_home["def_h"],
            "att_a":                      r_away["att_a"],
            "def_a":                      r_away["def_a"],
            "n_ucl_local_casa":           n_ucl_h,
            "n_ucl_visit_fuera":          n_ucl_a,
            "forma_reciente_local":       r_home.get("forma", ""),
            "forma_reciente_visit":       r_away.get("forma", ""),

            # xG doméstico
            "dom_xg_att_local":           _r(h_dom_att),
            "dom_xg_def_local":           _r(h_dom_def),
            "dom_n_local":                h_dom_n,
            "dom_xg_att_visit":           _r(a_dom_att),
            "dom_xg_def_visit":           _r(a_dom_def),
            "dom_n_visit":                a_dom_n,

            # Primer partido (ida)
            "resultado_ida":              (f"{g1h}–{g1a}" if g1h is not None else None),
            "goles_acum_local_vuelta":    g1h,
            "goles_acum_visit_vuelta":    g1a,
            "fecha_ida":                  first_leg["date"] if first_leg else None,

            # Clasificación (doble vuelta)
            "p_clasif_local_%":           _pct(p_clasif_home),
            "p_prorroga_%":               _pct(p_aet),
            "p_clasif_visit_%":           _pct(p_clasif_away),
            "favorito_clasificacion":     clasif_favorito,

            # Comparación DC vs Poisson puro
            "p_local_%_raw":              _pct(p_h_raw),
            "p_empate_%_raw":             _pct(p_d_raw),
            "p_visitante_%_raw":          _pct(p_a_raw),
            "dc_rho":                     rho,

            # Mercados de apuesta
            "ov15_%":                     round(mkts["ov15"] * 100, 1),
            "un15_%":                     round(mkts["un15"] * 100, 1),
            "ov25_%":                     round(mkts["ov25"] * 100, 1),
            "un25_%":                     round(mkts["un25"] * 100, 1),
            "ov35_%":                     round(mkts["ov35"] * 100, 1),
            "un35_%":                     round(mkts["un35"] * 100, 1),
            "btts_%":                     round(mkts["btts_p"] * 100, 1),
            "score_mas_probable":         f"{mkts['top_score'][0]}–{mkts['top_score'][1]}",
            "prob_score_%":               mkts["top_score"][2],
            "top5_scores":                "; ".join(
                f"{h}–{a}({p}%)" for h, a, p in mkts["scores_top5"]
            ),

            # Bet slips
            "pata_simple":                singles[-1]["pata"],
            "pata_simple_prob_%":         singles[-1]["prob_%"],
            "sgp_pata1":                  sgps[-1]["pata_1"],
            "sgp_pata2":                  sgps[-1]["pata_2"],
            "sgp_prob_%":                 sgps[-1]["prob_sgp_%"],
            "dream_pata1":                dreamers[-1]["pata_1"],
            "dream_pata2":                dreamers[-1]["pata_2"],
            "dream_pata3":                dreamers[-1]["pata_3"],
            "dream_prob_%":               dreamers[-1]["prob_sgp_%"],
        })

    print(SEP)
    print(f"\n  Modelo: Dixon-Coles (rho={rho}) + Lambda Multiplicativo")
    print(f"  UCL histórico: {len(ucl_hist)} partidos")
    print(f"  Home advantage calibrado: {home_adv:.3f}x")
    print(f"  Draw floor dinámico: DRAW_K={DRAW_K}")

    # ── Bet Slips ──
    print_bet_slips(singles, sgps, dreamers)

    # ── Guardar CSV ──
    out = pd.DataFrame(rows)
    out_path = Path("data/predicciones_ucl_v2.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV guardado: {out_path}")
    print(f"  {len(out)} partidos × {len(out.columns)} columnas\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="UCL Predictor v2 — Dixon-Coles")
    ap.add_argument("--days",  type=int,   default=7,          help="Días hacia adelante")
    ap.add_argument("--rho",     type=float, default=DC_RHO_DEFAULT,
                    help=f"Dixon-Coles rho (default {DC_RHO_DEFAULT})")
    ap.add_argument("--refresh", action="store_true",
                    help="Forzar re-descarga de resultados de la temporada actual")
    args = ap.parse_args()
    main(days_ahead=args.days, rho=args.rho, refresh=args.refresh)
