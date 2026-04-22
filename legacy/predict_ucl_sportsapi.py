"""
=====================================================================
  Predicciones UCL via SportsAPI (RapidAPI) — Soccer Project
  Uso:    python predict_ucl_sportsapi.py [--days 14]
  Output: data/predicciones_ucl_sportsapi.csv
          data/comparacion_ucl_fdorg_vs_sportsapi.csv

  IMPORTANTE: SportsAPI (plan BASIC free) tiene limite de 50 req/mes.
  Este script verifica cuanto quota hay, usa datos en cache si existen,
  y reporta exactamente que columnas tienen datos y cuales son N/D.

  Fuentes que intenta usar (solo SportsAPI / RapidAPI):
    - UCL historico: /unique-tournament/7/season/{sid}/events/last/{page}
    - UCL proximos:  /unique-tournament/7/season/{sid}/events/next/{page}
    - Forma domestica: /unique-tournament/{id}/season/{sid}/events/last/{page}
    - Stats por partido: /event/{id}/statistics

  Comparacion: carga data/predicciones_ucl.csv (football-data.org)
  y genera diferencias en probabilidades partido por partido.
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
# CONFIGURACION
# ─────────────────────────────────────────────

SPORTSAPI_KEY  = "c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d"
SPORTSAPI_HOST = "sportapi7.p.rapidapi.com"
SPORTSAPI_BASE = f"https://{SPORTSAPI_HOST}/api/v1"
SPORTSAPI_RATE = 2.0   # s entre peticiones

CACHE_DIR = Path("data/_ucl_sportsapi_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DOM_CACHE_DIR = Path("data/_sportsapi_cache")   # cache ya existente (PL)

DATA_DIR  = Path("data/processed")
WINDOW    = 8   # ultimos N partidos para forma

# Tournament IDs (SofaScore / sportapi7)
UCL_ID = 7
DOM_IDS = {
    "ENG-Premier League": 17,
    "ESP-La Liga":         8,
    "GER-Bundesliga":     35,
    "ITA-Serie A":        23,
    "FRA-Ligue 1":        34,
}

# Mapeo nombre SofaScore → liga interna
TEAM_TO_LEAGUE = {
    "Arsenal":          "ENG-Premier League",
    "Liverpool":        "ENG-Premier League",
    "Manchester City":  "ENG-Premier League",
    "Chelsea":          "ENG-Premier League",
    "Newcastle United": "ENG-Premier League",
    "Tottenham Hotspur":"ENG-Premier League",
    "Barcelona":        "ESP-La Liga",
    "Real Madrid":      "ESP-La Liga",
    "Atletico Madrid":  "ESP-La Liga",
    "Bayern Munich":    "GER-Bundesliga",
    "Bayer Leverkusen": "GER-Bundesliga",
    "Atalanta":         "ITA-Serie A",
    "Paris Saint-Germain": "FRA-Ligue 1",
}

UCL_AVG_GOALS = 1.4


# ─────────────────────────────────────────────
# CLIENTE SPORTSAPI
# ─────────────────────────────────────────────

_quota_exhausted  = False
_requests_remaining = None

def _get(url: str, cache_path: Path = None):
    """
    GET con cache JSON. Detecta y registra quota exhausted (429).
    Devuelve (data_dict, from_cache: bool, status_code: int).
    """
    global _quota_exhausted, _requests_remaining

    if cache_path and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f), True, 200

    if _quota_exhausted:
        return {}, False, 429

    time.sleep(SPORTSAPI_RATE)
    try:
        hdrs = {"x-rapidapi-key": SPORTSAPI_KEY, "x-rapidapi-host": SPORTSAPI_HOST}
        r    = requests.get(url, headers=hdrs, timeout=15)

        remaining = r.headers.get("X-RateLimit-Requests-Remaining")
        if remaining is not None:
            _requests_remaining = int(remaining)

        if r.status_code == 429:
            _quota_exhausted = True
            return {}, False, 429

        r.raise_for_status()
        data = r.json()

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)

        return data, False, r.status_code

    except Exception as exc:
        return {"_error": str(exc)}, False, 0


# ─────────────────────────────────────────────
# NORMALIZACIÓN
# ─────────────────────────────────────────────

_ALIASES = {
    "fc barcelona":           "barcelona",
    "real madrid cf":         "real madrid",
    "club atletico de madrid":"atletico madrid",
    "atletico de madrid":     "atletico madrid",
    "paris saint-germain fc": "paris saint-germain",
    "psg":                    "paris saint-germain",
    "arsenal fc":             "arsenal",
    "chelsea fc":             "chelsea",
    "liverpool fc":           "liverpool",
    "manchester city fc":     "manchester city",
    "newcastle united fc":    "newcastle united",
    "tottenham hotspur fc":   "tottenham",
    "bayer 04 leverkusen":    "bayer leverkusen",
    "fc bayern munchen":      "bayern munich",
    "fc bayern münchen":      "bayern munich",
    "atalanta bc":            "atalanta",
    "galatasaray sk":         "galatasaray",
    "galatasaray a.s.":       "galatasaray",
    "sporting clube de portugal": "sporting cp",
    "fk bodo/glimt":          "bodo/glimt",
    "fk bodoe/glimt":         "bodo/glimt",
}

def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd|sad|sae|a\.s\.|s\.a\.)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return _ALIASES.get(name, name)

def _best_match(target, candidates, threshold=0.68):
    t = _norm(target)
    best_name, best_ratio = None, threshold
    for c in candidates:
        r = SequenceMatcher(None, t, _norm(str(c))).ratio()
        if r > best_ratio:
            best_ratio, best_name = r, c
    return best_name


# ─────────────────────────────────────────────
# OBTENER SEASON ID
# ─────────────────────────────────────────────

def get_season_id(t_id: int):
    """
    Devuelve (season_id, from_cache, available).
    Busca en DOM_CACHE_DIR primero (datos ya descargados), luego hace peticion API.
    """
    # Intentar cache existente primero
    for cdir in [CACHE_DIR, DOM_CACHE_DIR]:
        p = cdir / f"seasons_{t_id}.json"
        if p.exists():
            with p.open(encoding="utf-8") as f:
                d = json.load(f)
            seasons = d.get("seasons", [])
            if seasons:
                return seasons[0]["id"], True, True

    url  = f"{SPORTSAPI_BASE}/unique-tournament/{t_id}/seasons"
    path = CACHE_DIR / f"seasons_{t_id}.json"
    data, _, __ = _get(url, path)
    seasons = data.get("seasons", [])
    if seasons:
        return seasons[0]["id"], _, True
    return None, False, False


# ─────────────────────────────────────────────
# CARGAR PARTIDOS UCL (historico + proximos)
# ─────────────────────────────────────────────

def load_ucl_events(season_id, days_ahead: int = 14):
    """
    Carga:
      - Partidos pasados UCL (hasta WINDOW*2 paginas)
      - Proximos partidos UCL en los siguientes days_ahead dias
    Devuelve (ucl_hist_df, upcoming_df, issues_list).
    """
    issues = []

    # ── Pasados ──
    hist_records = []
    for page in range(20):
        url   = f"{SPORTSAPI_BASE}/unique-tournament/{UCL_ID}/season/{season_id}/events/last/{page}"
        cache = CACHE_DIR / f"ucl_last_{season_id}_p{page}.json"
        data, from_cache, status = _get(url, cache)

        if status == 429:
            issues.append(f"UCL historial pagina {page}: quota agotada")
            break
        if not data.get("events"):
            break

        for ev in data["events"]:
            st_type = ev.get("status", {}).get("type", "")
            if st_type != "finished":
                continue
            ft = ev.get("homeScore", {})
            ht = int(ft.get("current", 0)) if ft else 0
            ft2 = ev.get("awayScore", {})
            at = int(ft2.get("current", 0)) if ft2 else 0
            ts = ev.get("startTimestamp", 0)
            hist_records.append({
                "event_id":   ev.get("id"),
                "date":       datetime.fromtimestamp(ts),
                "home_team":  ev.get("homeTeam", {}).get("name", ""),
                "away_team":  ev.get("awayTeam", {}).get("name", ""),
                "home_goals": ht,
                "away_goals": at,
            })

    ucl_hist = pd.DataFrame(hist_records)
    if not ucl_hist.empty:
        ucl_hist["date"] = pd.to_datetime(ucl_hist["date"])
        ucl_hist = ucl_hist.sort_values("date").reset_index(drop=True)
        ucl_hist["resultado"] = np.where(
            ucl_hist["home_goals"] > ucl_hist["away_goals"],  1,
            np.where(ucl_hist["home_goals"] < ucl_hist["away_goals"], -1, 0),
        )

    # ── Proximos ──
    today    = datetime.now()
    end_date = today + timedelta(days=days_ahead)
    next_records = []
    for page in range(5):
        url   = f"{SPORTSAPI_BASE}/unique-tournament/{UCL_ID}/season/{season_id}/events/next/{page}"
        cache = CACHE_DIR / f"ucl_next_{season_id}_p{page}.json"
        data, from_cache, status = _get(url, cache)

        if status == 429:
            issues.append(f"UCL proximos pagina {page}: quota agotada")
            break
        if not data.get("events"):
            break

        for ev in data["events"]:
            ts   = ev.get("startTimestamp", 0)
            edt  = datetime.fromtimestamp(ts)
            if edt > end_date:
                break
            next_records.append({
                "event_id":  ev.get("id"),
                "date":      edt,
                "stage":     ev.get("roundInfo", {}).get("name", ""),
                "home_team": ev.get("homeTeam", {}).get("name", ""),
                "away_team": ev.get("awayTeam", {}).get("name", ""),
            })

    upcoming = pd.DataFrame(next_records)
    if not upcoming.empty:
        upcoming["date"] = pd.to_datetime(upcoming["date"])

    return ucl_hist, upcoming, issues


# ─────────────────────────────────────────────
# FORMA DOMESTIC DESDE SPORTSAPI
# ─────────────────────────────────────────────

def load_dom_events_for_team(team_raw: str, season_id, t_id: int, window: int = WINDOW):
    """
    Descarga o carga desde cache los eventos domesticos recientes del equipo.
    Devuelve (goles_for_avg, goles_against_avg, n_partidos, status_msg).
    """
    records = []
    for page in range(6):
        # Buscar en cache DOM existente primero
        dom_cache = DOM_CACHE_DIR / f"events_{t_id}_{season_id}_p{page}.json"
        ucl_cache = CACHE_DIR    / f"dom_events_{t_id}_{season_id}_p{page}.json"
        cache     = dom_cache if dom_cache.exists() else ucl_cache

        url  = f"{SPORTSAPI_BASE}/unique-tournament/{t_id}/season/{season_id}/events/last/{page}"
        data, from_cache, status = _get(url, cache)

        if status == 429:
            if records:   # ya tenemos datos de paginas anteriores, usarlos
                break
            return None, None, 0, "quota_agotada"
        if not data.get("events"):
            break

        for ev in data["events"]:
            st_type = ev.get("status", {}).get("type", "")
            if st_type != "finished":
                continue
            records.append({
                "date":       datetime.fromtimestamp(ev.get("startTimestamp", 0)),
                "home_team":  ev.get("homeTeam", {}).get("name", ""),
                "away_team":  ev.get("awayTeam", {}).get("name", ""),
                "home_goals": int(ev.get("homeScore", {}).get("current", 0) or 0),
                "away_goals": int(ev.get("awayScore", {}).get("current", 0) or 0),
            })

    if not records:
        return None, None, 0, "sin_datos"

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    all_names  = pd.concat([df["home_team"], df["away_team"]]).unique()
    matched    = _best_match(team_raw, all_names)
    if matched is None:
        return None, None, 0, f"no_match({team_raw})"

    as_home = df[df["home_team"] == matched][["date","home_goals","away_goals"]].rename(
        columns={"home_goals": "gf", "away_goals": "ga"})
    as_away = df[df["away_team"] == matched][["date","away_goals","home_goals"]].rename(
        columns={"away_goals": "gf", "home_goals": "ga"})
    recent  = pd.concat([as_home, as_away]).sort_values("date").tail(window)

    if recent.empty:
        return None, None, 0, "sin_partidos_recientes"

    return float(recent["gf"].mean()), float(recent["ga"].mean()), len(recent), "ok"


# ─────────────────────────────────────────────
# FORMA UCL DEL EQUIPO
# ─────────────────────────────────────────────

def get_ucl_form(ucl_hist: pd.DataFrame, team_raw: str, before_date):
    if ucl_hist is None or ucl_hist.empty:
        return None, None, 0
    all_names = pd.concat([ucl_hist["home_team"], ucl_hist["away_team"]]).unique()
    matched   = _best_match(team_raw, all_names)
    if matched is None:
        return None, None, 0

    as_home = ucl_hist[ucl_hist["home_team"] == matched][["date","home_goals","away_goals"]].rename(
        columns={"home_goals": "gf", "away_goals": "ga"})
    as_away = ucl_hist[ucl_hist["away_team"] == matched][["date","away_goals","home_goals"]].rename(
        columns={"away_goals": "gf", "home_goals": "ga"})
    recent  = (pd.concat([as_home, as_away])
               .sort_values("date")
               .loc[lambda d: d["date"] < pd.Timestamp(before_date)]
               .tail(WINDOW))

    if recent.empty:
        return None, None, 0
    return float(recent["gf"].mean()), float(recent["ga"].mean()), len(recent)


def get_ucl_h2h(ucl_hist: pd.DataFrame, home_raw: str, away_raw: str, before_date):
    if ucl_hist is None or ucl_hist.empty:
        return 0, None, None, None
    all_names = pd.concat([ucl_hist["home_team"], ucl_hist["away_team"]]).unique()
    h_m = _best_match(home_raw, all_names)
    a_m = _best_match(away_raw, all_names)
    if h_m is None or a_m is None:
        return 0, None, None, None

    mask = (
        (ucl_hist["date"] < pd.Timestamp(before_date))
        & (
            ((ucl_hist["home_team"] == h_m) & (ucl_hist["away_team"] == a_m))
            | ((ucl_hist["home_team"] == a_m) & (ucl_hist["away_team"] == h_m))
        )
    )
    h2h = ucl_hist[mask].tail(6)
    n   = len(h2h)
    if n == 0:
        return 0, None, None, None

    hw = int(((h2h["home_team"] == h_m) & (h2h["resultado"] == 1)).sum()
             + ((h2h["away_team"] == h_m) & (h2h["resultado"] == -1)).sum())
    dr = int((h2h["resultado"] == 0).sum())
    aw = n - hw - dr
    return n, hw / n, dr / n, aw / n


# ─────────────────────────────────────────────
# POISSON
# ─────────────────────────────────────────────

def _ppmf(k, lam):
    if not lam or np.isnan(lam) or lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))

def poisson_probs(lh, la, max_g=8):
    pl = pe = pv = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _ppmf(h, lh) * _ppmf(a, la)
            if h > a:    pl += p
            elif h == a: pe += p
            else:        pv += p
    total = pl + pe + pv
    return (pl/total, pe/total, pv/total) if total else (1/3, 1/3, 1/3)


def safe(v, dec=2):
    try:
        f = float(v)
        return None if np.isnan(f) else round(f, dec)
    except (TypeError, ValueError):
        return None


def pct(v):
    f = safe(v)
    return None if f is None else round(f * 100, 1)


# ─────────────────────────────────────────────
# REPORTE DE COMPLETITUD
# ─────────────────────────────────────────────

def print_completeness(rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)
    print("\n  COMPLETITUD DE DATOS SPORTSAPI:")
    print("  " + "-" * 50)
    for col in df.columns:
        if col in ("local", "visitante", "fecha"):
            continue
        n_total = len(df)
        n_ok    = df[col].notna().sum()
        pct_ok  = n_ok / n_total * 100
        bar     = "#" * int(pct_ok / 5)
        print(f"  {col:<32} {n_ok:>2}/{n_total}  {pct_ok:>5.1f}%  [{bar:<20}]")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(days_ahead: int = 14):
    print("=" * 70)
    print("  UCL PREDICTIONS via SportsAPI (RapidAPI)")
    print("=" * 70)

    all_issues = []

    # ────────────────────────────────────────
    # PASO 0: Verificar quota
    # ────────────────────────────────────────
    print("\n[0] Verificando quota SportsAPI...")
    test_url   = f"{SPORTSAPI_BASE}/unique-tournament/{UCL_ID}/seasons"
    test_cache = CACHE_DIR / f"seasons_{UCL_ID}.json"
    data, from_cache, status = _get(test_url, test_cache)

    if status == 429 and not from_cache:
        print("\n  !! QUOTA AGOTADA !!")
        print("  Plan BASIC free tier: 50 requests/mes")
        print("  Requests restantes: 0  (reset: inicio de abril 2026)")
        print("  Solo se usaran datos en cache. Columnas sin cache = N/D")
        all_issues.append("QUOTA_AGOTADA: SportsAPI no puede hacer nuevas peticiones")
    elif from_cache:
        print(f"  Seasons UCL cargadas desde cache ({test_cache.name})")
    elif status == 200:
        print(f"  API disponible. Requests restantes: {_requests_remaining}")

    ucl_seasons = data.get("seasons", [])
    if not ucl_seasons:
        print("  SIN season ID para UCL — revisando alternativa...")
        # Intentar usar cache de eventos PL para verificar estructura
        ucl_season_id = None
        all_issues.append("UCL_SEASON_ID: No disponible — se usara solo cache")
    else:
        ucl_season_id = ucl_seasons[0]["id"]
        print(f"  UCL season ID: {ucl_season_id} ({ucl_seasons[0].get('year', '?')})")

    # ────────────────────────────────────────
    # PASO 1: Cargar historial UCL
    # ────────────────────────────────────────
    print("\n[1] Cargando historico UCL desde SportsAPI...")
    ucl_hist = pd.DataFrame()

    if ucl_season_id:
        ucl_hist, ucl_upcoming_sa, issues_ucl = load_ucl_events(ucl_season_id, days_ahead)
        all_issues.extend(issues_ucl)
        if not ucl_hist.empty:
            print(f"  Partidos UCL en cache: {len(ucl_hist)}")
        else:
            print("  Sin partidos UCL en cache (no fueron descargados previamente)")
            all_issues.append("UCL_HIST: 0 partidos disponibles en cache")
    else:
        ucl_upcoming_sa = pd.DataFrame()
        all_issues.append("UCL_HIST: sin season ID, 0 partidos disponibles")

    # ────────────────────────────────────────
    # PASO 2: Obtener fixtures UCL (de football-data.org como fallback)
    # ────────────────────────────────────────
    print("\n[2] Cargando fixtures UCL programados...")
    fdorg_preds_path = Path("data/predicciones_ucl.csv")

    if not ucl_upcoming_sa.empty:
        upcoming = ucl_upcoming_sa
        upcoming_source = "SportsAPI"
    elif fdorg_preds_path.exists():
        # Usar los mismos partidos del run de football-data.org para comparar
        fdorg_df = pd.read_csv(fdorg_preds_path)
        fdorg_df["date"] = pd.to_datetime(fdorg_df["fecha"])
        today    = pd.Timestamp.now().normalize()
        end_date = today + pd.Timedelta(days=days_ahead)
        fdorg_df = fdorg_df[(fdorg_df["date"] >= today) & (fdorg_df["date"] <= end_date)]
        upcoming = fdorg_df[["date", "fase", "local", "visitante"]].rename(
            columns={"local": "home_team", "visitante": "away_team"})
        upcoming_source = "football-data.org (fallback para comparacion)"
        print(f"  Fixtures desde football-data.org (fallback): {len(upcoming)} partidos")
        all_issues.append("FIXTURES: SportsAPI no pudo devolver proximos partidos — usando fdorg como referencia")
    else:
        print("  Sin fixtures disponibles. Ejecuta predict_ucl.py primero.")
        return

    print(f"  Fuente fixtures: {upcoming_source}")
    print(f"  Partidos a predecir: {len(upcoming)}")

    # ────────────────────────────────────────
    # PASO 3: Season IDs ligas domesticas
    # ────────────────────────────────────────
    print("\n[3] Verificando season IDs de ligas domesticas...")
    dom_season_ids = {}
    for league, t_id in DOM_IDS.items():
        sid, from_cache, available = get_season_id(t_id)
        status_txt = "cache" if from_cache else ("API" if available else "N/D")
        if sid:
            dom_season_ids[league] = sid
            print(f"  {league:<25} season_id={sid} [{status_txt}]")
        else:
            print(f"  {league:<25} N/D [{status_txt}]")
            all_issues.append(f"DOM_SEASON_{league}: season ID no disponible")

    # ────────────────────────────────────────
    # PASO 4: Calcular predicciones
    # ────────────────────────────────────────
    print("\n[4] Calculando predicciones...")
    SEP = "-" * 70
    output_rows   = []
    complete_rows = []   # para reporte de completitud

    for _, match in upcoming.sort_values("date").iterrows():
        home  = str(match["home_team"])
        away  = str(match["away_team"])
        date  = match["date"]
        stage = match.get("fase", match.get("stage", ""))

        # -- Forma UCL --
        h_ucl_att, h_ucl_def, h_ucl_n = get_ucl_form(ucl_hist, home, date)
        a_ucl_att, a_ucl_def, a_ucl_n = get_ucl_form(ucl_hist, away, date)

        # -- H2H UCL --
        h2h_n, h2h_loc, h2h_emp, h2h_vis = get_ucl_h2h(ucl_hist, home, away, date)

        # -- Forma domestica desde SportsAPI --
        h_dom_att = h_dom_def = a_dom_att = a_dom_def = None
        h_dom_n = a_dom_n = 0
        h_dom_status = a_dom_status = "sin_liga"
        h_league = a_league = "N/D"

        # Buscar liga de cada equipo
        for name, league_key in TEAM_TO_LEAGUE.items():
            if SequenceMatcher(None, _norm(home), _norm(name)).ratio() > 0.7:
                h_league = league_key
                break
        for name, league_key in TEAM_TO_LEAGUE.items():
            if SequenceMatcher(None, _norm(away), _norm(name)).ratio() > 0.7:
                a_league = league_key
                break

        if h_league in dom_season_ids:
            h_dom_att, h_dom_def, h_dom_n, h_dom_status = load_dom_events_for_team(
                home, dom_season_ids[h_league], DOM_IDS[h_league])
        if a_league in dom_season_ids:
            a_dom_att, a_dom_def, a_dom_n, a_dom_status = load_dom_events_for_team(
                away, dom_season_ids[a_league], DOM_IDS[a_league])

        # -- Lambda combinado --
        # Solo UCL si hay datos, blend si ambas fuentes disponibles
        ucl_w = min(min(h_ucl_n or 0, a_ucl_n or 0) / 5, 1.0) * 0.7

        lh_ucl = (h_ucl_att + a_ucl_def) / 2 if (h_ucl_att and a_ucl_def) else None
        la_ucl = (a_ucl_att + h_ucl_def) / 2 if (a_ucl_att and h_ucl_def) else None

        # Lambda domestico: acepta datos parciales usando UCL_AVG_GOALS como proxy
        # si solo un equipo tiene datos de su liga
        if h_dom_att is not None and a_dom_def is not None:
            lh_dom = (h_dom_att + a_dom_def) / 2
        elif h_dom_att is not None:
            lh_dom = (h_dom_att + UCL_AVG_GOALS) / 2   # defensa rival = media UCL
        elif a_dom_def is not None:
            lh_dom = (UCL_AVG_GOALS + a_dom_def) / 2   # ataque local = media UCL
        else:
            lh_dom = None

        if a_dom_att is not None and h_dom_def is not None:
            la_dom = (a_dom_att + h_dom_def) / 2
        elif a_dom_att is not None:
            la_dom = (a_dom_att + UCL_AVG_GOALS) / 2
        elif h_dom_def is not None:
            la_dom = (UCL_AVG_GOALS + h_dom_def) / 2
        else:
            la_dom = None

        if lh_ucl is not None and lh_dom is not None:
            lh = ucl_w * lh_ucl + (1 - ucl_w) * lh_dom
            la = ucl_w * la_ucl + (1 - ucl_w) * la_dom
            lam_src = f"blend {int(ucl_w*100)}% UCL"
        elif lh_ucl is not None:
            lh = lh_ucl
            la = la_ucl
            lam_src = "solo UCL"
        elif lh_dom is not None:
            lh = lh_dom
            la = la_dom
            # describir que equipo aportó datos reales
            h_ok = "L" if h_dom_att is not None else "-"
            a_ok = "V" if a_dom_att is not None else "-"
            lam_src = f"DOM({h_ok}/{a_ok})+avg"
        else:
            lh = la = UCL_AVG_GOALS
            lam_src = "fallback"

        # -- Poisson --
        if lh and la:
            p_loc, p_emp, p_vis = poisson_probs(lh, la)
        else:
            p_loc = p_emp = p_vis = 1/3
        pred = "Local" if p_loc >= p_emp and p_loc >= p_vis else \
               ("Empate" if p_emp >= p_loc and p_emp >= p_vis else "Visitante")

        row = {
            "fecha":                   pd.Timestamp(date).strftime("%Y-%m-%d"),
            "fase":                    str(stage),
            "local":                   home,
            "visitante":               away,
            "liga_local":              h_league,
            "liga_visitante":          a_league,
            "prediccion":              pred,
            "p_local_%":               round(p_loc * 100, 1),
            "p_empate_%":              round(p_emp * 100, 1),
            "p_visitante_%":           round(p_vis * 100, 1),
            "lambda_local":            safe(lh),
            "lambda_visitante":        safe(la),
            "fuente_lambda":           lam_src,
            # UCL
            "ucl_att_local":           safe(h_ucl_att),
            "ucl_def_local":           safe(h_ucl_def),
            "ucl_n_local":             h_ucl_n or 0,
            "ucl_att_visit":           safe(a_ucl_att),
            "ucl_def_visit":           safe(a_ucl_def),
            "ucl_n_visit":             a_ucl_n or 0,
            # Domestico
            "dom_att_local":           safe(h_dom_att),
            "dom_def_local":           safe(h_dom_def),
            "dom_n_local":             h_dom_n,
            "dom_att_visit":           safe(a_dom_att),
            "dom_def_visit":           safe(a_dom_def),
            "dom_n_visit":             a_dom_n,
            "dom_status_local":        h_dom_status,
            "dom_status_visit":        a_dom_status,
            # H2H
            "h2h_ucl_n":               h2h_n,
            "h2h_ucl_gana_local_%":    pct(h2h_loc),
            "h2h_ucl_empate_%":        pct(h2h_emp),
            "h2h_ucl_gana_visit_%":    pct(h2h_vis),
        }
        output_rows.append(row)

        # Para reporte completitud (solo los campos de datos, no metadatos)
        complete_rows.append({
            "local":             home,
            "visitante":         away,
            "ucl_att_local":     safe(h_ucl_att),
            "ucl_att_visit":     safe(a_ucl_att),
            "dom_att_local":     safe(h_dom_att),
            "dom_att_visit":     safe(a_dom_att),
            "lambda_local":      safe(lh) if lam_src != "fallback" else None,
            "h2h_ucl_n>0":       h2h_n if h2h_n > 0 else None,
        })

        print(SEP)
        print(f"  {pd.Timestamp(date).strftime('%d/%m/%y')}  [{stage}]  "
              f"{home[:22]:<22} vs  {away[:22]}")
        print(f"  Goles esp: {lh:.2f} vs {la:.2f}  ({lam_src})  =>  "
              f"Local {p_loc*100:.1f}%  /  Empate {p_emp*100:.1f}%  /  Visitante {p_vis*100:.1f}%"
              f"  [{pred}]")
        ucl_l = f"{h_ucl_att:.2f}/{h_ucl_def:.2f} ({h_ucl_n}pj)" if h_ucl_att else "N/D"
        ucl_a = f"{a_ucl_att:.2f}/{a_ucl_def:.2f} ({a_ucl_n}pj)" if a_ucl_att else "N/D"
        dom_l = f"{h_dom_att:.2f}/{h_dom_def:.2f} ({h_dom_n}pj)" if h_dom_att else f"N/D [{h_dom_status}]"
        dom_a = f"{a_dom_att:.2f}/{a_dom_def:.2f} ({a_dom_n}pj)" if a_dom_att else f"N/D [{a_dom_status}]"
        print(f"  UCL: L {ucl_l}  vs  A {ucl_a}")
        print(f"  DOM: L {dom_l}  vs  A {dom_a}")

    print(SEP)

    # ────────────────────────────────────────
    # REPORTE DE COMPLETITUD
    # ────────────────────────────────────────
    print_completeness(complete_rows)

    # ────────────────────────────────────────
    # PROBLEMAS DETECTADOS
    # ────────────────────────────────────────
    print(f"\n  PROBLEMAS / DATOS INCOMPLETOS ({len(all_issues)}):")
    for i, iss in enumerate(all_issues, 1):
        print(f"  {i}. {iss}")

    # ────────────────────────────────────────
    # COMPARACION CON FOOTBALL-DATA.ORG
    # ────────────────────────────────────────
    print("\n  COMPARACION SportsAPI vs Football-Data.org:")
    print("  " + "=" * 64)
    fdorg_loaded = False
    if fdorg_preds_path.exists() and output_rows:
        fdorg = pd.read_csv(fdorg_preds_path)
        fdorg["date"] = pd.to_datetime(fdorg["fecha"])
        today    = pd.Timestamp.now().normalize()
        end_date = today + pd.Timedelta(days=days_ahead)
        fdorg = fdorg[(fdorg["date"] >= today) & (fdorg["date"] <= end_date)]

        comp_rows = []
        out_df    = pd.DataFrame(output_rows)
        out_df["date_dt"] = pd.to_datetime(out_df["fecha"])

        for _, fd_row in fdorg.iterrows():
            # Buscar partido correspondiente en SportsAPI
            sa_match = None
            for _, sa_row in out_df.iterrows():
                if (SequenceMatcher(None, _norm(fd_row["local"]), _norm(sa_row["local"])).ratio() > 0.7
                        and SequenceMatcher(None, _norm(fd_row["visitante"]), _norm(sa_row["visitante"])).ratio() > 0.7):
                    sa_match = sa_row
                    break

            if sa_match is None:
                continue

            fd_pred = fd_row["prediccion"]
            sa_pred = sa_match["prediccion"]
            acuerdo = "SI" if fd_pred == sa_pred else "NO"

            fd_l, fd_e, fd_v = fd_row["p_local_%"], fd_row["p_empate_%"], fd_row["p_visitante_%"]
            sa_l = sa_match["p_local_%"] or 33.3
            sa_e = sa_match["p_empate_%"] or 33.3
            sa_v = sa_match["p_visitante_%"] or 33.3

            comp_rows.append({
                "fecha":          fd_row["fecha"],
                "local":          fd_row["local"],
                "visitante":      fd_row["visitante"],
                "pred_fdorg":     fd_pred,
                "pred_sportsapi": sa_pred,
                "acuerdo":        acuerdo,
                "fdorg_loc_%":    fd_l,
                "sa_loc_%":       sa_l,
                "diff_loc_%":     round(sa_l - fd_l, 1),
                "fdorg_emp_%":    fd_e,
                "sa_emp_%":       sa_e,
                "diff_emp_%":     round(sa_e - fd_e, 1),
                "fdorg_vis_%":    fd_v,
                "sa_vis_%":       sa_v,
                "diff_vis_%":     round(sa_v - fd_v, 1),
                "fuente_sa":      sa_match.get("fuente_lambda", "N/D"),
            })

        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            acuerdo_pct = (comp_df["acuerdo"] == "SI").mean() * 100
            print(f"  Partidos comparados:  {len(comp_df)}")
            print(f"  Predicciones iguales: {(comp_df['acuerdo']=='SI').sum()}/{len(comp_df)} ({acuerdo_pct:.0f}%)")
            print()
            print(f"  {'Partido':<35}  {'fdorg':>10}  {'SportAPI':>10}  {'Acuerdo':>8}")
            print(f"  {'-'*35}  {'-'*10}  {'-'*10}  {'-'*8}")
            for _, r in comp_df.iterrows():
                partido = f"{r['local'][:16]} vs {r['visitante'][:15]}"
                fd_str  = f"{r['pred_fdorg']} ({r['fdorg_loc_%']:.0f}/{r['fdorg_emp_%']:.0f}/{r['fdorg_vis_%']:.0f})"
                sa_str  = f"{r['pred_sportsapi']} ({r['sa_loc_%']:.0f}/{r['sa_emp_%']:.0f}/{r['sa_vis_%']:.0f})"
                print(f"  {partido:<35}  {fd_str:>10}  {sa_str:>10}  {r['acuerdo']:>8}")

            # Guardar comparacion
            comp_path = Path("data/comparacion_ucl_fdorg_vs_sportsapi.csv")
            comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")
            print(f"\n  Comparacion guardada: {comp_path}")
            fdorg_loaded = True

    if not fdorg_loaded:
        print("  (Ejecuta predict_ucl.py primero para generar predicciones football-data.org)")

    # ────────────────────────────────────────
    # GUARDAR CSV PRINCIPAL
    # ────────────────────────────────────────
    out_path = Path("data/predicciones_ucl_sportsapi.csv")
    pd.DataFrame(output_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV SportsAPI guardado: {out_path}")
    print(f"  {len(output_rows)} partidos x {len(pd.DataFrame(output_rows).columns)} columnas\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCL predictions via SportsAPI")
    parser.add_argument("--days", type=int, default=14,
                        help="Dias hacia adelante (default: 14)")
    args = parser.parse_args()
    main(days_ahead=args.days)
