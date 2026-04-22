"""
=====================================================================
  Predicciones UCL (UEFA Champions League) — Soccer Project
  Uso:    python predict_ucl.py [--days 14]
  Output: data/predicciones_ucl.csv

  Fuentes de datos:
    1. football-data.org — fixtures y resultados historicos UCL
       (temporadas 2023, 2024 y 2025/26)
    2. Understat schedule_xg — forma domestica de cada equipo
       (ultimas 3 temporadas: 2324, 2425, 2526)
    3. Google Trends — senal social de interes publico por equipo
       (señal informativa, no determinante del lambda)

  Como se calculan las probabilidades:
    1. FORMA UCL: goles/xG en los ultimos N partidos de UCL,
       SEPARADOS por rol (como local vs como visitante).
       La media ponderada da mas peso a partidos recientes.
    2. FORMA DOMESTICA: xG ofensivo/defensivo rolling (ultimos 5
       partidos de liga, temporadas 2023+), media ponderada.
    3. LAMBDA combinado:
         Si equipo tiene >= 5 partidos UCL: 70% UCL + 30% domestico
         Si tiene 1-4 partidos UCL:         blend proporcional
         Sin datos UCL:                     100% domestico
       El equipo local usa su tasa de ataque EN CASA (UCL),
       el visitante usa su tasa de ataque FUERA (UCL).
    4. Poisson independiente + suavizado minimo 15% empate.
    5. Columnas de desglose explican cada paso del calculo.

  Cache: data/_ucl_cache/ — borra para forzar actualizacion
  Nota Twitter: twitter135.p.rapidapi.com no esta suscrito con la
                clave actual. Se usa Google Trends como alternativa.
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

# ─────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────

FDORG_TOKEN = "1f7a12b6f18f418caa88a0f59884e80a"
FDORG_BASE  = "https://api.football-data.org/v4"
FDORG_RATE  = 6.5   # s entre peticiones

DATA_DIR   = Path("data/processed")
CACHE_DIR  = Path("data/_ucl_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TREND_CACHE_DIR = Path("data/_trend_cache")
TREND_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Temporadas UCL disponibles en football-data.org (free tier)
UCL_SEASONS = [2023, 2024, 2025]

# Temporadas domesticas a considerar (ultimas 3)
DOM_SEASONS = ["2324", "2425", "2526"]

# Peso UCL vs domestico segun cuantos partidos UCL tiene el equipo
UCL_WEIGHT_MAX = 0.7

WINDOW_UCL = 8   # ultimos N partidos UCL para calcular forma
WINDOW_DOM = 5   # ultimos N partidos domesticos para rolling xG

UCL_AVG_GOALS = 1.4   # fallback cuando no hay datos
DRAW_FLOOR    = 0.15  # probabilidad minima de empate en 90 min


# ─────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES
# ─────────────────────────────────────────────

_TEAM_ALIASES = {
    "fc barcelona":           "barcelona",
    "real madrid cf":         "real madrid",
    "club atletico de madrid":"atletico madrid",
    "atletico de madrid":     "atletico madrid",
    "atletico madrid":        "atletico madrid",
    "fc bayern munchen":      "bayern munich",
    "fc bayern münchen":      "bayern munich",
    "bayer 04 leverkusen":    "bayer leverkusen",
    "borussia dortmund":      "dortmund",
    "bvb":                    "dortmund",
    "paris saint-germain fc": "paris saint-germain",
    "psg":                    "paris saint-germain",
    "arsenal fc":             "arsenal",
    "chelsea fc":             "chelsea",
    "liverpool fc":           "liverpool",
    "manchester city fc":     "manchester city",
    "newcastle united fc":    "newcastle united",
    "tottenham hotspur fc":   "tottenham",
    "atalanta bc":            "atalanta",
    "galatasaray sk":         "galatasaray",
    "galatasaray a.s.":       "galatasaray",
    "sporting clube de portugal": "sporting cp",
    "fk bodo/glimt":          "bodo/glimt",
    "fk bodoe/glimt":         "bodo/glimt",
}


def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(
        r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd|sad|sae|a\.s\.|s\.a\.)\b",
        "", name,
    )
    name = re.sub(r"\s+", " ", name).strip()
    return _TEAM_ALIASES.get(name, name)


def _best_match(target: str, candidates, threshold: float = 0.70):
    t = _norm(target)
    best_name, best_ratio = None, threshold
    for c in candidates:
        ratio = SequenceMatcher(None, t, _norm(str(c))).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, c
    return best_name


# ─────────────────────────────────────────────
# CLIENTE football-data.org
# ─────────────────────────────────────────────

def _fdorg_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"X-Auth-Token": FDORG_TOKEN, "Accept": "application/json"})
    return s


def _fdorg_get(session: requests.Session, url: str, cache_path: Path = None):
    if cache_path and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)
    time.sleep(FDORG_RATE)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return data


# ─────────────────────────────────────────────
# CARGA DE DATOS UCL
# ─────────────────────────────────────────────

def load_ucl_matches(session: requests.Session) -> pd.DataFrame:
    records = []
    for season in UCL_SEASONS:
        path  = CACHE_DIR / f"ucl_finished_{season}.json"
        url   = f"{FDORG_BASE}/competitions/CL/matches?status=FINISHED&season={season}"
        try:
            data    = _fdorg_get(session, url, path)
            matches = data.get("matches", [])
        except Exception as exc:
            print(f"  Aviso: no se pudo cargar UCL {season}: {exc}")
            continue

        for m in matches:
            ft = m.get("score", {}).get("fullTime", {})
            hg = ft.get("home")
            ag = ft.get("away")
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

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["resultado"] = np.where(
            df["home_goals"] > df["away_goals"],  1,
            np.where(df["home_goals"] < df["away_goals"], -1, 0),
        )
    print(f"  UCL historico cargado: {len(df)} partidos ({UCL_SEASONS[0]}-{UCL_SEASONS[-1]})")
    return df


def load_ucl_upcoming(session: requests.Session, days_ahead: int = 14) -> pd.DataFrame:
    path  = CACHE_DIR / f"ucl_scheduled_{datetime.now().strftime('%Y%m%d')}.json"
    url   = f"{FDORG_BASE}/competitions/CL/matches?status=SCHEDULED"
    try:
        data    = _fdorg_get(session, url, path)
        matches = data.get("matches", [])
    except Exception as exc:
        print(f"  Error cargando partidos programados UCL: {exc}")
        return pd.DataFrame()

    today    = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    records  = []
    for m in matches:
        d = datetime.fromisoformat(m["utcDate"][:10]).date()
        if today <= d <= end_date:
            records.append({
                "date":       d,
                "stage":      m.get("stage", ""),
                "home_team":  m["homeTeam"]["name"],
                "away_team":  m["awayTeam"]["name"],
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# FORMA UCL — SPLIT LOCAL/VISITANTE + PONDERADA
# ─────────────────────────────────────────────

def _weighted_mean(series: pd.Series) -> float:
    """Media ponderada: partido mas reciente = peso N, mas antiguo = peso 1."""
    n = len(series)
    if n == 0:
        return UCL_AVG_GOALS
    weights = list(range(1, n + 1))
    return float(np.average(series.values, weights=weights))


def _result_char(gf: float, ga: float) -> str:
    if gf > ga:  return "W"
    if gf == ga: return "D"
    return "L"


def get_ucl_form_split(ucl_hist: pd.DataFrame, schedule_xg: pd.DataFrame,
                       team_raw: str, before_date, window: int = WINDOW_UCL):
    """
    MEJORA 1 + 2 + 3:
      - Separa estadisticas UCL como LOCAL y como VISITANTE
      - Cuando hay datos xG en schedule_xg de UCL, los usa; si no, usa goles reales
      - Aplica media ponderada (reciente > antiguo)
      - Devuelve string de forma reciente "WWD" (ultimos 3 resultados)

    Retorna:
      home_att, home_def, home_n,   (stats como local)
      away_att, away_def, away_n,   (stats como visitante)
      total_n,                       (total partidos UCL)
      recent_str                     (ultimos 3: "WWD")
    """
    EMPTY = (UCL_AVG_GOALS, UCL_AVG_GOALS, 0,
             UCL_AVG_GOALS, UCL_AVG_GOALS, 0, 0, "")

    if ucl_hist.empty:
        return EMPTY

    all_teams = pd.concat([ucl_hist["home_team"], ucl_hist["away_team"]]).unique()
    matched   = _best_match(team_raw, all_teams)
    if matched is None:
        return EMPTY

    before_ts = pd.Timestamp(before_date)

    # ── Mejora 2: Intentar xG de schedule_xg para UCL ──
    # schedule_xg actualmente cubre solo ligas domesticas, pero si en el futuro
    # incluye UCL (league contiene "Champions" / "CL"), se usara xG en vez de goles.
    ucl_xg_sched = pd.DataFrame()
    if not schedule_xg.empty:
        ucl_mask = schedule_xg["league"].str.contains(
            r"Champions|UCL|CL\b", case=False, na=False, regex=True
        )
        ucl_xg_sched = schedule_xg[ucl_mask & (schedule_xg["is_result"] == True)]

    use_xg = not ucl_xg_sched.empty

    if use_xg:
        # Usar xG desde schedule_xg
        xg_all = pd.concat([ucl_xg_sched["home_team"], ucl_xg_sched["away_team"]]).unique()
        xg_matched = _best_match(team_raw, xg_all)
        if xg_matched:
            xg_home = ucl_xg_sched[
                (ucl_xg_sched["home_team"] == xg_matched) & (ucl_xg_sched["date"] < before_ts)
            ][["date", "home_xg", "away_xg"]].rename(columns={"home_xg": "gf", "away_xg": "ga"})
            xg_away = ucl_xg_sched[
                (ucl_xg_sched["away_team"] == xg_matched) & (ucl_xg_sched["date"] < before_ts)
            ][["date", "away_xg", "home_xg"]].rename(columns={"away_xg": "gf", "home_xg": "ga"})
            as_home_raw = xg_home.sort_values("date").tail(window)
            as_away_raw = xg_away.sort_values("date").tail(window)
        else:
            use_xg = False

    if not use_xg:
        # Usar goles reales desde ucl_hist
        as_home_raw = (ucl_hist[
            (ucl_hist["home_team"] == matched) & (ucl_hist["date"] < before_ts)
        ][["date", "home_goals", "away_goals"]]
            .rename(columns={"home_goals": "gf", "away_goals": "ga"})
            .sort_values("date").tail(window))

        as_away_raw = (ucl_hist[
            (ucl_hist["away_team"] == matched) & (ucl_hist["date"] < before_ts)
        ][["date", "away_goals", "home_goals"]]
            .rename(columns={"away_goals": "gf", "home_goals": "ga"})
            .sort_values("date").tail(window))

    # ── Media ponderada por rol ──
    home_att = _weighted_mean(as_home_raw["gf"]) if len(as_home_raw) else UCL_AVG_GOALS
    home_def = _weighted_mean(as_home_raw["ga"]) if len(as_home_raw) else UCL_AVG_GOALS
    home_n   = len(as_home_raw)

    away_att = _weighted_mean(as_away_raw["gf"]) if len(as_away_raw) else UCL_AVG_GOALS
    away_def = _weighted_mean(as_away_raw["ga"]) if len(as_away_raw) else UCL_AVG_GOALS
    away_n   = len(as_away_raw)

    total_n  = home_n + away_n

    # ── String de forma reciente (ultimos 3 partidos UCL) ──
    all_games = pd.concat([as_home_raw, as_away_raw]).sort_values("date").tail(3)
    recent_str = "".join(
        _result_char(float(r["gf"]), float(r["ga"]))
        for _, r in all_games.iterrows()
    )

    return home_att, home_def, home_n, away_att, away_def, away_n, total_n, recent_str


# ─────────────────────────────────────────────
# H2H EN UCL
# ─────────────────────────────────────────────

def get_ucl_h2h(ucl_hist: pd.DataFrame, home_raw: str, away_raw: str,
                before_date, window: int = 6):
    """
    Historial directo en UCL. Devuelve:
      (n, p_home_wins, p_draw, p_away_wins, avg_goals_by_home_in_h2h)
    """
    if ucl_hist.empty:
        return 0, np.nan, np.nan, np.nan, np.nan

    all_teams = pd.concat([ucl_hist["home_team"], ucl_hist["away_team"]]).unique()
    h_match   = _best_match(home_raw, all_teams)
    a_match   = _best_match(away_raw, all_teams)
    if h_match is None or a_match is None:
        return 0, np.nan, np.nan, np.nan, np.nan

    before_ts = pd.Timestamp(before_date)
    mask = (
        (ucl_hist["date"] < before_ts)
        & (
            ((ucl_hist["home_team"] == h_match) & (ucl_hist["away_team"] == a_match))
            | ((ucl_hist["home_team"] == a_match) & (ucl_hist["away_team"] == h_match))
        )
    )
    h2h = ucl_hist[mask].tail(window)
    n   = len(h2h)
    if n == 0:
        return 0, np.nan, np.nan, np.nan, np.nan

    home_wins = int(
        ((h2h["home_team"] == h_match) & (h2h["resultado"] == 1)).sum()
        + ((h2h["away_team"] == h_match) & (h2h["resultado"] == -1)).sum()
    )
    draws     = int((h2h["resultado"] == 0).sum())
    away_wins = n - home_wins - draws

    # Goles marcados por el equipo "local" en estos H2H
    goles_h_as_home = h2h.loc[h2h["home_team"] == h_match, "home_goals"].sum()
    goles_h_as_away = h2h.loc[h2h["away_team"] == h_match, "away_goals"].sum()
    avg_h2h_goals   = float(goles_h_as_home + goles_h_as_away) / n

    return n, home_wins / n, draws / n, away_wins / n, avg_h2h_goals


# ─────────────────────────────────────────────
# FORMA DOMESTICA — MEDIA PONDERADA
# ─────────────────────────────────────────────

def get_domestic_form(schedule: pd.DataFrame, team_raw: str, before_date,
                      window: int = WINDOW_DOM):
    """
    MEJORA 3: Media ponderada (reciente > antiguo) sobre xG rolling domestico.
    """
    if schedule.empty:
        return UCL_AVG_GOALS, UCL_AVG_GOALS, 0, ""

    all_teams = pd.concat([schedule["home_team"], schedule["away_team"]]).unique()
    matched   = _best_match(team_raw, all_teams)
    if matched is None:
        return UCL_AVG_GOALS, UCL_AVG_GOALS, 0, ""

    league = ""
    sub = schedule[(schedule["home_team"] == matched) | (schedule["away_team"] == matched)]
    if not sub.empty:
        league = sub.iloc[-1]["league"]

    as_home = sub[sub["home_team"] == matched][["date", "home_xg", "away_xg"]].rename(
        columns={"home_xg": "xg_for", "away_xg": "xg_against"})
    as_away = sub[sub["away_team"] == matched][["date", "away_xg", "home_xg"]].rename(
        columns={"away_xg": "xg_for", "home_xg": "xg_against"})

    recent = (pd.concat([as_home, as_away])
              .sort_values("date")
              .loc[lambda d: d["date"] < pd.Timestamp(before_date)]
              .tail(window))

    if recent.empty:
        return UCL_AVG_GOALS, UCL_AVG_GOALS, 0, league

    xg_for     = _weighted_mean(recent["xg_for"])
    xg_against = _weighted_mean(recent["xg_against"])
    return xg_for, xg_against, len(recent), league


# ─────────────────────────────────────────────
# LAMBDA COMBINADO — SEPARADO POR ROL
# ─────────────────────────────────────────────

def combined_lambda(ucl_att: float, ucl_def_rival: float, ucl_n: int,
                    dom_att: float, dom_def_rival: float) -> float:
    """
    Combina UCL y forma domestica. Peso UCL crece con partidos disponibles.
    """
    ucl_w = min(ucl_n / 5, 1.0) * UCL_WEIGHT_MAX
    dom_w = 1.0 - ucl_w
    ucl_lam = (ucl_att + ucl_def_rival) / 2
    dom_lam = (dom_att + dom_def_rival) / 2
    return ucl_w * ucl_lam + dom_w * dom_lam


# ─────────────────────────────────────────────
# POISSON + SUAVIZADO DE EMPATE
# ─────────────────────────────────────────────

def _ppmf(k: int, lam: float) -> float:
    if lam <= 0 or np.isnan(lam):
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))


def poisson_probs(lh: float, la: float, max_g: int = 8):
    p_local = p_draw = p_away = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _ppmf(h, lh) * _ppmf(a, la)
            if h > a:    p_local += p
            elif h == a: p_draw  += p
            else:        p_away  += p
    total = p_local + p_draw + p_away
    if total > 0:
        p_local /= total
        p_draw  /= total
        p_away  /= total
    return p_local, p_draw, p_away


def smooth_draw(p_local: float, p_draw: float, p_away: float,
                floor: float = DRAW_FLOOR):
    """
    MEJORA 5: Si p_draw < floor, eleva el empate al minimo redistribuyendo
    proporcionalmente entre local y visitante. En UCL los empates en 90 min
    son frecuentes y Poisson los subestima con lambdas muy diferentes.
    """
    if p_draw >= floor:
        return p_local, p_draw, p_away
    deficit  = floor - p_draw
    total_lv = p_local + p_away
    if total_lv <= 0:
        return p_local, floor, p_away
    p_local -= deficit * (p_local / total_lv)
    p_away  -= deficit * (p_away  / total_lv)
    p_draw   = floor
    return p_local, p_draw, p_away


# ─────────────────────────────────────────────
# SEÑAL SOCIAL — Google Trends
# (Twitter135 API no disponible con la clave actual)
# ─────────────────────────────────────────────

def get_social_trend(team_name: str) -> float:
    """
    Retorna un score de buzz en Google Trends para el equipo en los
    ultimos 7 dias relativo a la media de los 90 dias previos.
      > 1.0 = mas interes que lo usual (positivo o negativo)
      ~ 1.0 = interes normal
      < 1.0 = menos interes que usual
    Cachea resultado 24h. Falla silenciosamente con 1.0 (neutral).
    """
    safe_name = re.sub(r"[^\w]", "_", team_name.lower())[:30]
    cache_path = TREND_CACHE_DIR / f"trend_{safe_name}.txt"

    # Cache valido 24h
    if cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < 86400:
            try:
                return float(cache_path.read_text())
            except Exception:
                pass

    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        kw = _norm(team_name)[:50]
        pt.build_payload([kw], cat=20, timeframe="today 90-d", geo="", gprop="")
        df = pt.interest_over_time()
        if df.empty or kw not in df.columns:
            return 1.0
        # Ultimo 10% de periodos = "reciente", resto = baseline
        n_rows = len(df)
        split  = max(1, int(n_rows * 0.10))
        baseline = float(df[kw].iloc[:-split].mean())
        recent   = float(df[kw].iloc[-split:].mean())
        score    = round(recent / baseline, 3) if baseline > 0 else 1.0
        cache_path.write_text(str(score))
        return score
    except Exception:
        return 1.0   # neutral si pytrends falla


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

_STAGE_MAP = {
    "LAST_16":        "Octavos",
    "QUARTER_FINALS": "Cuartos",
    "SEMI_FINALS":    "Semis",
    "FINAL":          "Final",
    "GROUP_STAGE":    "Grupos",
}

def _fmt_stage(s: str) -> str:
    return _STAGE_MAP.get(s, s)

def _safe(v, dec=2):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, dec)
    except (TypeError, ValueError):
        return None

def _pct(v):
    f = _safe(v)
    return None if f is None else round(f * 100, 1)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(days_ahead: int = 14):
    print("=" * 70)
    print("  SOCCER PROJECT -- Predicciones UCL (Champions League)")
    print("=" * 70)

    session = _fdorg_session()

    # ── Cargar datos UCL historicos ──
    print("\n[1/4] Cargando historico UCL desde football-data.org...")
    ucl_hist = load_ucl_matches(session)

    # ── Cargar partidos programados ──
    print(f"\n[2/4] Cargando fixtures UCL proximos {days_ahead} dias...")
    ucl_upcoming = load_ucl_upcoming(session, days_ahead)

    if ucl_upcoming.empty:
        print("\n  No hay partidos UCL programados en ese rango.")
        return

    print(f"  Partidos encontrados: {len(ucl_upcoming)}")

    # ── Cargar forma domestica ──
    print("\n[3/4] Cargando forma domestica (schedule_xg)...")
    schedule_full = pd.read_parquet(DATA_DIR / "schedule_xg.parquet").reset_index()
    schedule_full["date"] = pd.to_datetime(schedule_full["date"])

    dom = schedule_full[
        (schedule_full["is_result"] == True)
        & (schedule_full["season"].isin(DOM_SEASONS))
    ].copy()
    print(f"  Partidos domesticos 2023-2026: {len(dom)}")

    # ── Calcular predicciones ──
    print("\n[4/4] Calculando predicciones y senales sociales...")
    SEP = "-" * 70
    output_rows = []

    for _, match in ucl_upcoming.sort_values("date").iterrows():
        home  = match["home_team"]
        away  = match["away_team"]
        date  = match["date"]
        stage = match["stage"]

        # ── MEJORA 1+2+3: Forma UCL con split local/visitante y media ponderada ──
        (h_ucl_home_att, h_ucl_home_def, h_ucl_home_n,
         h_ucl_away_att, h_ucl_away_def, h_ucl_away_n,
         h_ucl_total_n, h_recent_str) = get_ucl_form_split(ucl_hist, schedule_full, home, date)

        (a_ucl_home_att, a_ucl_home_def, a_ucl_home_n,
         a_ucl_away_att, a_ucl_away_def, a_ucl_away_n,
         a_ucl_total_n, a_recent_str) = get_ucl_form_split(ucl_hist, schedule_full, away, date)

        # H2H en UCL
        h2h_n, h2h_loc, h2h_emp, h2h_vis, h2h_avg_goals_local = \
            get_ucl_h2h(ucl_hist, home, away, date)

        # Forma domestica (media ponderada)
        h_dom_att, h_dom_def, h_dom_n, h_league = get_domestic_form(dom, home, date)
        a_dom_att, a_dom_def, a_dom_n, a_league = get_domestic_form(dom, away, date)

        # ── MEJORA 1: Lambda usa ataque/defensa especifico por rol ──
        # Local juega en casa → usamos su tasa UCL como LOCAL
        # Visitante juega fuera → usamos su tasa UCL como VISITANTE
        ucl_n_eff = min(h_ucl_home_n, a_ucl_away_n)
        lh = combined_lambda(
            ucl_att=h_ucl_home_att, ucl_def_rival=a_ucl_away_def, ucl_n=ucl_n_eff,
            dom_att=h_dom_att, dom_def_rival=a_dom_def,
        )
        la = combined_lambda(
            ucl_att=a_ucl_away_att, ucl_def_rival=h_ucl_home_def, ucl_n=ucl_n_eff,
            dom_att=a_dom_att, dom_def_rival=h_dom_def,
        )

        # ── MEJORA 5: Poisson + suavizado empate ──
        p_loc_raw, p_emp_raw, p_vis_raw = poisson_probs(lh, la)
        p_loc, p_emp, p_vis = smooth_draw(p_loc_raw, p_emp_raw, p_vis_raw)

        if p_loc >= p_emp and p_loc >= p_vis:
            prediccion = "Local"
        elif p_emp >= p_loc and p_emp >= p_vis:
            prediccion = "Empate"
        else:
            prediccion = "Visitante"

        ucl_w_eff = round(min(ucl_n_eff / 5, 1.0) * UCL_WEIGHT_MAX * 100)

        # ── Señal social (Google Trends) ──
        h_trend = get_social_trend(home)
        a_trend = get_social_trend(away)

        # ── Registro CSV ──
        # [columnas originales — sin cambios de posicion]
        row = {
            # Identificacion
            "fecha":                      date.strftime("%Y-%m-%d"),
            "fase":                       _fmt_stage(stage),
            "local":                      home,
            "visitante":                  away,
            "liga_local":                 h_league,
            "liga_visitante":             a_league,

            # Prediccion
            "prediccion":                 prediccion,
            "p_local_%":                  round(p_loc * 100, 1),
            "p_empate_%":                 round(p_emp * 100, 1),
            "p_visitante_%":              round(p_vis * 100, 1),

            # Paso 1 — Forma en UCL (ahora separada por rol)
            "ucl_goles_marcados_local":   _safe(h_ucl_home_att),
            "ucl_goles_concedidos_local": _safe(h_ucl_home_def),
            "ucl_partidos_local":         h_ucl_home_n,
            "ucl_goles_marcados_visit":   _safe(a_ucl_away_att),
            "ucl_goles_concedidos_visit": _safe(a_ucl_away_def),
            "ucl_partidos_visit":         a_ucl_away_n,

            # Paso 2 — Forma domestica
            "dom_xg_ataque_local":        _safe(h_dom_att),
            "dom_xg_defensa_local":       _safe(h_dom_def),
            "dom_partidos_local":         h_dom_n,
            "dom_xg_ataque_visit":        _safe(a_dom_att),
            "dom_xg_defensa_visit":       _safe(a_dom_def),
            "dom_partidos_visit":         a_dom_n,

            # Paso 3 — Lambda resultante
            "lambda_local":               round(lh, 2),
            "lambda_visitante":           round(la, 2),
            "peso_ucl_%":                 ucl_w_eff,

            # H2H en UCL (columnas originales)
            "h2h_ucl_partidos":           h2h_n,
            "h2h_ucl_gana_local_%":       _pct(h2h_loc),
            "h2h_ucl_empate_%":           _pct(h2h_emp),
            "h2h_ucl_gana_visit_%":       _pct(h2h_vis),

            # ── MEJORA 4: Columnas nuevas al final ──
            "ventaja_goles":              round(lh - la, 2),
            "h2h_ucl_goles_favor_local":  _safe(h2h_avg_goals_local),
            "forma_reciente_local":       h_recent_str if h_recent_str else "N/D",
            "forma_reciente_visitante":   a_recent_str if a_recent_str else "N/D",
            # UCL stats como visitante (para contexto)
            "ucl_goles_marcados_local_away":   _safe(h_ucl_away_att),
            "ucl_goles_marcados_visit_home":   _safe(a_ucl_home_att),
            "ucl_partidos_total_local":        h_ucl_total_n,
            "ucl_partidos_total_visit":        a_ucl_total_n,
            # Probabilidades brutas antes de suavizado
            "p_local_%_raw":              round(p_loc_raw * 100, 1),
            "p_empate_%_raw":             round(p_emp_raw * 100, 1),
            "p_visitante_%_raw":          round(p_vis_raw * 100, 1),
            # Señal social
            "social_trend_local":         _safe(h_trend, 3),
            "social_trend_visit":         _safe(a_trend, 3),
        }
        output_rows.append(row)

        # ── Consola ──
        print(SEP)
        print(f"  {date.strftime('%d/%m/%y')}  [{_fmt_stage(stage)}]  "
              f"{home[:22]:<22} vs  {away[:22]}")
        print(f"  Goles esp: {lh:.2f} vs {la:.2f}  =>  "
              f"Local {p_loc*100:.1f}%  /  Empate {p_emp*100:.1f}%  /  Visitante {p_vis*100:.1f}%"
              f"  [{prediccion}]")
        print(f"  UCL casa ({h_ucl_home_n}pj): att {h_ucl_home_att:.2f}/def {h_ucl_home_def:.2f}"
              f"  |  UCL fuera ({a_ucl_away_n}pj): att {a_ucl_away_att:.2f}/def {a_ucl_away_def:.2f}")
        if h_dom_n > 0 or a_dom_n > 0:
            print(f"  DOM({h_dom_n}pj)[{h_league.split('-')[-1].strip() if h_league else 'N/D'}]:"
                  f" xG {h_dom_att:.2f}/{h_dom_def:.2f}"
                  f"  |  "
                  f"DOM({a_dom_n}pj)[{a_league.split('-')[-1].strip() if a_league else 'N/D'}]:"
                  f" xG {a_dom_att:.2f}/{a_dom_def:.2f}")
        print(f"  Peso UCL: {ucl_w_eff}%  |  "
              f"Forma: {h_recent_str or 'N/D'} vs {a_recent_str or 'N/D'}  |  "
              f"Trend: {h_trend:.2f}x vs {a_trend:.2f}x")
        if h2h_n > 0:
            print(f"  H2H UCL ({h2h_n} partidos): "
                  f"local {h2h_loc*100:.0f}% / empate {h2h_emp*100:.0f}%"
                  f" / visitante {h2h_vis*100:.0f}%  "
                  f"(local marca {h2h_avg_goals_local:.1f} goles en H2H)")

    print(SEP)
    print(f"\n  Modelo: Poisson independiente con ventaja local UCL + media ponderada")
    print(f"  UCL historico: {len(ucl_hist)} partidos (seasons {UCL_SEASONS})")
    print(f"  Mejoras activas: split local/visitante UCL, media ponderada,")
    print(f"    suavizado empate (min {DRAW_FLOOR*100:.0f}%), Google Trends social signal")
    print(f"  Nota: Twitter API (twitter135.p.rapidapi.com) no suscrito con clave actual.")

    # ── Guardar CSV ──
    out = pd.DataFrame(output_rows)
    out_path = Path("data/predicciones_ucl.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV guardado: {out_path}")
    print(f"  {len(out)} partidos x {len(out.columns)} columnas\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones UCL")
    parser.add_argument("--days", type=int, default=14,
                        help="Dias hacia adelante a cubrir (default: 14)")
    args = parser.parse_args()
    main(days_ahead=args.days)
