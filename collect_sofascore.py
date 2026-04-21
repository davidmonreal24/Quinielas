"""
=====================================================================
  collect_sofascore.py — Predicciones via SofascoreAPI (RapidAPI)
  Host:  sofascore6.p.rapidapi.com
  Key:   c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d

  Uso:   python collect_sofascore.py [--days 15] [--history-days 150]
  Output:
    data/_sofascore_cache/          — JSONs cacheados por fecha
    data/sofascore_events.parquet   — historial recolectado
    data/predicciones_sofascore.csv — predicciones próximos N días

  Modelo:
    Ratings multiplicativos ataque/defensa (inspirado en Dixon-Coles):
      att_rate = avg_goles_marcados / promedio_liga
      def_rate = avg_goles_recibidos / promedio_liga
      lambda_local  = att_h * def_a_rival * mu_home
      lambda_visit  = att_a * def_h_rival * mu_away
    Blend UCL (peso mayor con más datos) + Doméstico
    Poisson independiente + suavizado de empate (floor 15%)

  Datos: únicamente Sofascore6 API
    Endpoint: GET /api/sofascore/v1/match/list?sport_slug=football&date=YYYY-MM-DD
    Respuesta: array JSON de partidos

  Torneos (uniqueTournament.id):
    UCL=7, PL=17, LaLiga=8, Bundesliga=35, Serie A=23, Ligue1=34
=====================================================================
"""

import argparse
import json
import math
import re
import time
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

SOFASCORE_KEY  = "c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d"
SOFASCORE_HOST = "sofascore6.p.rapidapi.com"
SOFASCORE_BASE = f"https://{SOFASCORE_HOST}/api/sofascore/v1"
SOFASCORE_RATE = 0.8   # segundos entre peticiones

CACHE_DIR = Path("data/_sofascore_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Torneos de interés (uniqueTournament.id → nombre corto)
TOURNAMENTS_OF_INTEREST = {
    7:   "UCL",
    17:  "ENG-Premier League",
    8:   "ESP-La Liga",
    35:  "GER-Bundesliga",
    23:  "ITA-Serie A",
    34:  "FRA-Ligue 1",
    11620: "MEX-Liga MX",
}

UCL_ID     = 7
UCL_MU     = 1.40   # media goles UCL (histórica de referencia)
DRAW_FLOOR = 0.15   # mínimo probabilidad empate
MIN_GAMES  = 3      # mínimo partidos para usar ratings de equipo
WINDOW     = 8      # partidos para forma reciente


# ─────────────────────────────────────────────
# CLIENTE API
# ─────────────────────────────────────────────

def _cache_is_stale(data: list) -> bool:
    """Detecta si la caché tiene partidos que debían haber terminado pero siguen como notstarted."""
    import time as _time
    now_ts = _time.time()
    for ev in data:
        ts = ev.get("timestamp", 0)
        status = ev.get("status", {}).get("type", "")
        # Si un partido estaba programado hace más de 4 horas y sigue como notstarted → caché stale
        if status == "notstarted" and ts > 0 and (now_ts - ts) > 4 * 3600:
            return True
    return False


def _ss_get_date(d: str, use_cache: bool = True) -> list:
    """
    GET /api/sofascore/v1/match/list?sport_slug=football&date=YYYY-MM-DD
    Retorna lista de partidos. Usa caché si use_cache=True.
    Invalida automáticamente cachés con partidos pasados en status notstarted.
    """
    cache = CACHE_DIR / f"matches_{d}.json"
    if use_cache and cache.exists():
        cached = json.loads(cache.read_bytes().decode("utf-8"))
        if not _cache_is_stale(cached):
            return cached
        # Caché stale → eliminar y refrescar
        cache.unlink()
        print(f"  Caché stale detectada y eliminada: {cache.name}")

    time.sleep(SOFASCORE_RATE)
    hdrs = {"x-rapidapi-key": SOFASCORE_KEY, "x-rapidapi-host": SOFASCORE_HOST}
    try:
        r = requests.get(
            f"{SOFASCORE_BASE}/match/list",
            headers=hdrs,
            params={"sport_slug": "football", "date": d},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list):
                cache.write_bytes(json.dumps(data, ensure_ascii=False).encode("utf-8"))
                return data
        print(f"  HTTP {r.status_code} para {d}")
        return []
    except Exception as e:
        print(f"  ERR {d}: {e}")
        return []


# ─────────────────────────────────────────────
# PARSEO DE EVENTOS
# ─────────────────────────────────────────────

def _is_interest(ev: dict) -> bool:
    return ev.get("uniqueTournament", {}).get("id") in TOURNAMENTS_OF_INTEREST


def parse_ss_event(ev: dict) -> dict | None:
    """Normaliza un evento Sofascore a registro interno."""
    try:
        ts     = ev.get("timestamp", 0)
        status = ev.get("status", {}).get("type", "")
        home   = ev.get("homeTeam", {}).get("name", "")
        away   = ev.get("awayTeam", {}).get("name", "")
        if not home or not away:
            return None

        # Scores solo disponibles cuando terminó el partido
        finished = (status == "finished")
        hg = ag = None
        if finished:
            hg = ev.get("homeScore", {}).get("normaltime") or ev.get("homeScore", {}).get("current")
            ag = ev.get("awayScore", {}).get("normaltime") or ev.get("awayScore", {}).get("current")

        uni  = ev.get("uniqueTournament", {})
        tour = ev.get("tournament", {})
        seas = ev.get("season", {})
        rnd  = ev.get("round", {})

        # Convertir a CDT (UTC-6) para que partidos del viernes noche no aparezcan como sábado
        CDT = timezone(timedelta(hours=-6))
        return {
            "event_id":      ev.get("id"),
            "date":          datetime.fromtimestamp(ts, tz=CDT).date().isoformat(),
            "timestamp":     ts,
            "status":        status,
            "is_result":     finished,
            "home_team":     home,
            "away_team":     away,
            "home_id":       ev.get("homeTeam", {}).get("id"),
            "away_id":       ev.get("awayTeam", {}).get("id"),
            "home_goals":    int(hg) if hg is not None else None,
            "away_goals":    int(ag) if ag is not None else None,
            "tournament_id": uni.get("id") or tour.get("id"),
            "tournament_name": uni.get("name") or tour.get("name", ""),
            "season_id":     seas.get("id"),
            "season_year":   seas.get("year", ""),
            "round_name":    rnd.get("name", ""),
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# RECOLECCIÓN: UN SOLO BARRIDO (histórico + próximos)
# ─────────────────────────────────────────────

def collect_all_matches(days_back: int = 150, days_ahead: int = 15) -> tuple:
    """
    Descarga partidos desde (hoy - days_back) hasta (hoy + days_ahead).
    Cache permanente para fechas pasadas; siempre fresco para fechas futuras.
    Retorna: (historical: list[dict], upcoming: list[dict])
    """
    today    = date.today()
    hist_map = {}   # event_id → record (dedup)
    up_map   = {}   # event_id → record (dedup)
    total    = days_back + days_ahead + 1

    print(f"  Barriendo {total} fechas ({days_back} atrás + {days_ahead} adelante)...")
    downloaded = 0

    for offset in range(-days_back, days_ahead + 1):
        d     = today + timedelta(days=offset)
        d_str = d.strftime("%Y-%m-%d")
        # Solo caché para fechas PASADAS (> 1 día atrás); frescos para hoy y futuro
        use_cache = (offset < -1)
        events = _ss_get_date(d_str, use_cache=use_cache)
        if not use_cache:
            downloaded += 1

        for ev in events:
            if not _is_interest(ev):
                continue
            rec = parse_ss_event(ev)
            if not rec:
                continue
            eid = rec.get("event_id")
            if rec["is_result"] and rec["home_goals"] is not None:
                if eid and eid not in hist_map:
                    hist_map[eid] = rec
            elif rec["status"] == "notstarted" and offset > 0:
                if eid and eid not in up_map:
                    up_map[eid] = rec

    historical = list(hist_map.values())
    upcoming   = list(up_map.values())
    print(f"  Partidos terminados: {len(historical)} | Partidos próximos: {len(upcoming)}")
    print(f"  Peticiones nuevas a API: {downloaded}")
    return historical, upcoming


# ─────────────────────────────────────────────
# MODELO DE RATINGS ATAQUE / DEFENSA
# ─────────────────────────────────────────────

def _weighted_mean(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 0.0
    weights = np.arange(1, n + 1, dtype=float)
    return float(np.average(series.values, weights=weights))


def compute_ratings(df: pd.DataFrame) -> dict:
    """
    Calcula ratings multiplicativos de ataque y defensa para cada equipo.

    Modelo:
      mu_h = promedio goles marcados como local en la liga/torneo
      mu_a = promedio goles marcados como visitante en la liga/torneo
      att_h = (prom. goles marcados en casa) / mu_h    → >1 = ataque fuerte
      def_h = (prom. goles recibidos en casa) / mu_a   → <1 = defensa fuerte
      att_a = (prom. goles marcados fuera)   / mu_a    → >1 = ataque fuerte
      def_a = (prom. goles recibidos fuera)  / mu_h    → <1 = defensa fuerte

    lambda_local   = att_h(local)  × def_a(rival)  × mu_h
    lambda_visita  = att_a(rival)  × def_h(local)  × mu_a

    Retorna: {"teams": {...}, "mu_h": float, "mu_a": float, "home_adv": float}
    """
    empty = {"teams": {}, "mu_h": UCL_MU, "mu_a": UCL_MU * 0.86, "home_adv": 1.15}
    if df is None or df.empty:
        return empty

    df = df.dropna(subset=["home_goals", "away_goals"]).copy()
    if len(df) < 5:
        return empty

    mu_h = float(df["home_goals"].mean())
    mu_a = float(df["away_goals"].mean())
    if mu_h <= 0 or mu_a <= 0:
        return empty

    home_adv = mu_h / mu_a if mu_a > 0 else 1.15

    teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    ratings = {}

    for team in teams:
        hg = df[df["home_team"] == team].sort_values("date")
        ag = df[df["away_team"] == team].sort_values("date")
        nh, na = len(hg), len(ag)

        att_h = (_weighted_mean(hg["home_goals"]) / mu_h) if nh >= MIN_GAMES else None
        def_h = (_weighted_mean(hg["away_goals"]) / mu_a) if nh >= MIN_GAMES else None
        att_a = (_weighted_mean(ag["away_goals"]) / mu_a) if na >= MIN_GAMES else None
        def_a = (_weighted_mean(ag["home_goals"]) / mu_h) if na >= MIN_GAMES else None

        # Forma reciente (últimos 5 partidos)
        recent = (pd.concat([
            hg[["date", "home_goals", "away_goals"]].rename(columns={"home_goals": "gf", "away_goals": "ga"}),
            ag[["date", "away_goals", "home_goals"]].rename(columns={"away_goals": "gf", "home_goals": "ga"}),
        ]).sort_values("date").tail(5))

        def _rc(gf, ga):
            return "W" if gf > ga else ("D" if gf == ga else "L")

        forma = "".join(_rc(float(r["gf"]), float(r["ga"])) for _, r in recent.iterrows())

        ratings[team] = {
            "att_h": att_h, "def_h": def_h,
            "att_a": att_a, "def_a": def_a,
            "nh": nh, "na": na,
            "forma": forma,
        }

    return {"teams": ratings, "mu_h": mu_h, "mu_a": mu_a, "home_adv": home_adv}


def _clip(v: float | None, lo=0.30, hi=3.50) -> float:
    if v is None:
        return 1.0
    return max(lo, min(hi, v))


def _get_team(team_raw: str, ratings: dict) -> dict:
    """Busca equipo en ratings con fuzzy matching. Retorna dict de ratings."""
    teams = ratings.get("teams", {})
    if team_raw in teams:
        return teams[team_raw]
    matched = _best_match(team_raw, list(teams.keys()))
    return teams.get(matched, {}) if matched else {}


def get_lambdas(
    home: str, away: str,
    ucl_r: dict, dom_r: dict,
    is_ucl: bool,
) -> tuple:
    """
    Calcula (lh, la, meta) = (lambda_local, lambda_visit, dict_metricas).

    Para UCL: blend de ratings UCL + doméstico ponderado por cantidad de datos.
    Para doméstico: solo ratings domésticos.
    """
    mu_h_ucl = ucl_r.get("mu_h", UCL_MU)
    mu_a_ucl = ucl_r.get("mu_a", UCL_MU * 0.86)
    mu_h_dom = dom_r.get("mu_h", UCL_MU)
    mu_a_dom = dom_r.get("mu_a", UCL_MU * 0.86)

    h_ucl = _get_team(home, ucl_r)
    a_ucl = _get_team(away, ucl_r)
    h_dom = _get_team(home, dom_r)
    a_dom = _get_team(away, dom_r)

    # Datos UCL
    att_h_u = h_ucl.get("att_h")
    def_h_u = h_ucl.get("def_h")
    att_a_u = a_ucl.get("att_a")
    def_a_u = a_ucl.get("def_a")
    nh_ucl  = h_ucl.get("nh", 0)
    na_ucl  = a_ucl.get("na", 0)
    n_ucl   = min(nh_ucl, na_ucl)

    # Datos domésticos
    att_h_d = h_dom.get("att_h")
    def_h_d = h_dom.get("def_h")
    att_a_d = a_dom.get("att_a")
    def_a_d = a_dom.get("def_a")
    nh_dom  = h_dom.get("nh", 0)
    na_dom  = a_dom.get("na", 0)

    if is_ucl:
        # Peso UCL: crece con datos disponibles, máximo 80%
        ucl_w = min(n_ucl / 6.0, 1.0) * 0.80
        dom_w = 1.0 - ucl_w

        lh_ucl = _clip(att_h_u) * _clip(def_a_u) * mu_h_ucl
        la_ucl = _clip(att_a_u) * _clip(def_h_u) * mu_a_ucl

        lh_dom = _clip(att_h_d) * _clip(def_a_d) * mu_h_dom
        la_dom = _clip(att_a_d) * _clip(def_h_d) * mu_a_dom

        lh = ucl_w * lh_ucl + dom_w * lh_dom
        la = ucl_w * la_ucl + dom_w * la_dom
    else:
        ucl_w  = 0.0
        lh_ucl = la_ucl = mu_h_dom  # no aplica
        lh_dom = _clip(att_h_d) * _clip(def_a_d) * mu_h_dom
        la_dom = _clip(att_a_d) * _clip(def_h_d) * mu_a_dom
        lh = lh_dom
        la = la_dom
        n_ucl = nh_ucl = na_ucl = 0

    meta = {
        "att_h_ucl":   round(att_h_u, 3) if att_h_u else None,
        "def_a_ucl":   round(def_a_u, 3) if def_a_u else None,
        "att_a_ucl":   round(att_a_u, 3) if att_a_u else None,
        "def_h_ucl":   round(def_h_u, 3) if def_h_u else None,
        "n_ucl_h":     nh_ucl,
        "n_ucl_a":     na_ucl,
        "w_ucl_pct":   round(ucl_w * 100),
        "att_h_dom":   round(att_h_d, 3) if att_h_d else None,
        "def_a_dom":   round(def_a_d, 3) if def_a_d else None,
        "att_a_dom":   round(att_a_d, 3) if att_a_d else None,
        "def_h_dom":   round(def_h_d, 3) if def_h_d else None,
        "n_dom_h":     nh_dom,
        "n_dom_a":     na_dom,
        "mu_h":        round(mu_h_ucl if is_ucl else mu_h_dom, 3),
        "mu_a":        round(mu_a_ucl if is_ucl else mu_a_dom, 3),
        "home_adv":    round(ucl_r.get("home_adv", 1.15) if is_ucl else dom_r.get("home_adv", 1.15), 3),
        "forma_h":     h_ucl.get("forma", h_dom.get("forma", "")) or "",
        "forma_a":     a_ucl.get("forma", a_dom.get("forma", "")) or "",
    }
    return lh, la, meta


# ─────────────────────────────────────────────
# UTILIDADES DE PREDICCIÓN
# ─────────────────────────────────────────────

_ALIASES = {
    "fc barcelona": "barcelona",           "real madrid cf": "real madrid",
    "club atletico de madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "fc bayern munchen": "bayern munich",  "fc bayern münchen": "bayern munich",
    "bayer 04 leverkusen": "bayer leverkusen",
    "paris saint-germain fc": "psg",       "paris saint-germain": "psg",
    "arsenal fc": "arsenal",               "chelsea fc": "chelsea",
    "liverpool fc": "liverpool",           "manchester city fc": "manchester city",
    "newcastle united fc": "newcastle united",
    "tottenham hotspur fc": "tottenham hotspur",
    "atalanta bc": "atalanta",             "galatasaray sk": "galatasaray",
    "fk bodo/glimt": "bodo/glimt",
    "sporting clube de portugal": "sporting cp",
    "internazionale": "inter milan",       "inter": "inter milan",
    "borussia dortmund": "dortmund",
}


def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return _ALIASES.get(name, name)


def _best_match(target: str, candidates: list, threshold=0.68) -> str | None:
    t = _norm(target)
    best, score = None, threshold
    for c in candidates:
        r = SequenceMatcher(None, t, _norm(str(c))).ratio()
        if r > score:
            score, best = r, c
    return best


def _ppmf(k: int, lam: float) -> float:
    if lam <= 0 or math.isnan(lam):
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))


def poisson_probs(lh: float, la: float, max_g: int = 9) -> tuple:
    pl = pd_ = pv = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _ppmf(h, lh) * _ppmf(a, la)
            if h > a:    pl  += p
            elif h == a: pd_ += p
            else:        pv  += p
    total = pl + pd_ + pv
    return (pl / total, pd_ / total, pv / total) if total else (1/3, 1/3, 1/3)


def smooth_draw(p_loc: float, p_drw: float, p_vis: float, floor=DRAW_FLOOR) -> tuple:
    if p_drw >= floor:
        return p_loc, p_drw, p_vis
    deficit  = floor - p_drw
    total_lv = p_loc + p_vis
    if total_lv <= 0:
        return p_loc, floor, p_vis
    p_loc -= deficit * (p_loc / total_lv)
    p_vis -= deficit * (p_vis / total_lv)
    return p_loc, floor, p_vis


def form_points(forma: str) -> int:
    """Suma de puntos de forma: W=3, D=1, L=0."""
    return sum({"W": 3, "D": 1, "L": 0}.get(c, 0) for c in forma)


def confidence_pct(p_max: float, p_second: float) -> float:
    """Nivel de confianza: margen entre la prob. más alta y la segunda."""
    return round((p_max - p_second) * 100, 1)


def _safe(v, dec=3):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, dec)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# GENERACIÓN DE PREDICCIONES
# ─────────────────────────────────────────────

# Nombres de columnas en lenguaje natural (header → clave interna)
COLUMN_NAMES = {
    "Fecha":                                           "fecha",
    "Competicion":                                     "competicion",
    "Fase / Ronda":                                    "fase",
    "Equipo Local":                                    "local",
    "Equipo Visitante":                                "visitante",
    "Prediccion":                                      "prediccion",
    "Probabilidad Local % (p_local)":                  "p_local_pct",
    "Probabilidad Empate % (p_empate)":                "p_empate_pct",
    "Probabilidad Visitante % (p_visit)":              "p_visit_pct",
    "Goles Esperados Local (lambda_h)":                "lambda_h",
    "Goles Esperados Visitante (lambda_a)":            "lambda_a",
    "Margen Goles Esperado (ventaja_h_minus_a)":       "ventaja",
    "Ratio Ataque Local UCL (att_h_ucl)":              "att_h_ucl",
    "Ratio Defensa Rival UCL (def_a_ucl)":             "def_a_ucl",
    "Ratio Ataque Visitante UCL (att_a_ucl)":          "att_a_ucl",
    "Ratio Defensa Local UCL (def_h_ucl)":             "def_h_ucl",
    "Partidos UCL Local como Local (n_ucl_h)":         "n_ucl_h",
    "Partidos UCL Visitante como Visitante (n_ucl_a)": "n_ucl_a",
    "Peso UCL en Prediccion % (w_ucl)":                "w_ucl_pct",
    "Ratio Ataque Local Domestico (att_h_dom)":        "att_h_dom",
    "Ratio Defensa Rival Domestico (def_a_dom)":       "def_a_dom",
    "Ratio Ataque Visitante Domestico (att_a_dom)":    "att_a_dom",
    "Ratio Defensa Local Domestico (def_h_dom)":       "def_h_dom",
    "Partidos Domesticos Local (n_dom_h)":             "n_dom_h",
    "Partidos Domesticos Visitante (n_dom_a)":         "n_dom_a",
    "Forma Reciente Local W/D/L (forma_h)":            "forma_h",
    "Forma Reciente Visitante W/D/L (forma_a)":        "forma_a",
    "Puntos de Forma Local (pts_h, W=3 D=1 L=0)":     "pts_forma_h",
    "Puntos de Forma Visitante (pts_a, W=3 D=1 L=0)": "pts_forma_a",
    "Media Goles Local Liga (mu_h)":                   "mu_h",
    "Media Goles Visitante Liga (mu_a)":               "mu_a",
    "Factor Ventaja Local (home_adv)":                 "home_adv",
    "Prob Local % sin suavizar (p_local_raw)":         "p_local_raw",
    "Prob Empate % sin suavizar (p_empate_raw)":       "p_empate_raw",
    "Prob Visitante % sin suavizar (p_visit_raw)":     "p_visit_raw",
    "Nivel de Confianza % (confianza)":                "confianza_pct",
}


def generate_predictions(
    upcoming: list,
    ucl_hist_df: pd.DataFrame,
    dom_hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """Genera predicciones para todos los partidos en `upcoming`."""

    ucl_ratings = compute_ratings(ucl_hist_df)
    dom_ratings = compute_ratings(dom_hist_df)

    print(f"  Ratings UCL: {len(ucl_ratings['teams'])} equipos  "
          f"(mu_h={ucl_ratings['mu_h']:.2f}, mu_a={ucl_ratings['mu_a']:.2f}, "
          f"home_adv={ucl_ratings['home_adv']:.2f})")
    print(f"  Ratings DOM: {len(dom_ratings['teams'])} equipos  "
          f"(mu_h={dom_ratings['mu_h']:.2f}, mu_a={dom_ratings['mu_a']:.2f}, "
          f"home_adv={dom_ratings['home_adv']:.2f})")

    rows = []
    for rec in sorted(upcoming, key=lambda x: x["date"]):
        home     = rec["home_team"]
        away     = rec["away_team"]
        m_date   = rec["date"]
        tour_id  = rec.get("tournament_id")
        tour_name = rec.get("tournament_name", "")
        fase      = rec.get("round_name", "")

        is_ucl = (tour_id == UCL_ID or "champion" in tour_name.lower())

        lh, la, meta = get_lambdas(home, away, ucl_ratings, dom_ratings, is_ucl)

        # Poisson + suavizado
        p_loc_r, p_emp_r, p_vis_r = poisson_probs(lh, la)
        p_loc,   p_emp,   p_vis   = smooth_draw(p_loc_r, p_emp_r, p_vis_r)

        probs   = [p_loc, p_emp, p_vis]
        labels  = ["Local", "Empate", "Visitante"]
        pred    = labels[probs.index(max(probs))]
        p_max   = max(probs)
        p_2nd   = sorted(probs, reverse=True)[1]
        conf    = confidence_pct(p_max, p_2nd)

        forma_h = meta["forma_h"]
        forma_a = meta["forma_a"]

        row = {
            "fecha":        m_date,
            "competicion":  tour_name,
            "fase":         fase,
            "local":        home,
            "visitante":    away,
            "prediccion":   pred,
            "p_local_pct":  round(p_loc * 100, 1),
            "p_empate_pct": round(p_emp * 100, 1),
            "p_visit_pct":  round(p_vis * 100, 1),
            "lambda_h":     _safe(lh),
            "lambda_a":     _safe(la),
            "ventaja":      round(lh - la, 3),
            "att_h_ucl":    meta["att_h_ucl"],
            "def_a_ucl":    meta["def_a_ucl"],
            "att_a_ucl":    meta["att_a_ucl"],
            "def_h_ucl":    meta["def_h_ucl"],
            "n_ucl_h":      meta["n_ucl_h"],
            "n_ucl_a":      meta["n_ucl_a"],
            "w_ucl_pct":    meta["w_ucl_pct"],
            "att_h_dom":    meta["att_h_dom"],
            "def_a_dom":    meta["def_a_dom"],
            "att_a_dom":    meta["att_a_dom"],
            "def_h_dom":    meta["def_h_dom"],
            "n_dom_h":      meta["n_dom_h"],
            "n_dom_a":      meta["n_dom_a"],
            "forma_h":      forma_h or "N/D",
            "forma_a":      forma_a or "N/D",
            "pts_forma_h":  form_points(forma_h),
            "pts_forma_a":  form_points(forma_a),
            "mu_h":         meta["mu_h"],
            "mu_a":         meta["mu_a"],
            "home_adv":     meta["home_adv"],
            "p_local_raw":  round(p_loc_r * 100, 1),
            "p_empate_raw": round(p_emp_r * 100, 1),
            "p_visit_raw":  round(p_vis_r * 100, 1),
            "confianza_pct": conf,
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    # Construir DataFrame con columnas en lenguaje natural
    internal_df = pd.DataFrame(rows)
    col_map = {v: k for k, v in COLUMN_NAMES.items()}
    renamed = {c: col_map.get(c, c) for c in internal_df.columns}
    return internal_df.rename(columns=renamed)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(days_ahead: int = 15, history_days: int = 150):
    print("=" * 70)
    print("  SOCCER PROJECT — Predicciones via Sofascore6 (RapidAPI)")
    print(f"  Modelo: Ratings Ataque/Defensa multiplicativos + Poisson")
    print(f"  Historial: {history_days} dias | Horizonte: {days_ahead} dias")
    print("=" * 70)

    # ── 1. Recolectar TODOS los partidos en un barrido ──
    print(f"\n[1/4] Recolectando datos Sofascore ({history_days + days_ahead + 1} fechas)...")
    historical, upcoming = collect_all_matches(
        days_back=history_days,
        days_ahead=days_ahead,
    )

    if not historical:
        print("  ERROR: No se obtuvieron partidos históricos. Abortando.")
        return

    # ── 2. Construir DataFrames por torneo ──
    print(f"\n[2/4] Preparando datos ({len(historical)} hist, {len(upcoming)} próximos)...")
    hist_df = pd.DataFrame(historical)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df = hist_df.dropna(subset=["home_goals", "away_goals"])

    ucl_df = hist_df[hist_df["tournament_id"] == UCL_ID].copy()
    dom_df = hist_df[hist_df["tournament_id"] != UCL_ID].copy()

    print(f"  UCL: {len(ucl_df)} partidos | Doméstico: {len(dom_df)} partidos")

    # Guardar historial
    ev_path = Path("data/sofascore_events.parquet")
    hist_df.to_parquet(ev_path, index=False)

    # Guardar próximos (para predict_ligamx.py y otros scripts)
    if upcoming:
        up_path = Path("data/sofascore_upcoming.parquet")
        pd.DataFrame(upcoming).to_parquet(up_path, index=False)
        print(f"  Próximos guardados: {up_path} ({len(upcoming)} partidos)")

    # ── 3. Verificar que hay partidos próximos ──
    print(f"\n[3/4] Partidos próximos encontrados: {len(upcoming)}")
    if not upcoming:
        print("  AVISO: No se encontraron partidos programados para los próximos "
              f"{days_ahead} días.")
        print("  Posibles causas: La API no ha publicado los fixtures aún.")
        print("  Intenta ejecutar con --days 20 para ampliar el horizonte.")
        return

    # Mostrar resumen de próximos partidos por competición
    up_df = pd.DataFrame(upcoming)
    for tid, name in TOURNAMENTS_OF_INTEREST.items():
        n = (up_df["tournament_id"] == tid).sum()
        if n:
            print(f"    {name}: {n} partidos")

    # ── 4. Generar predicciones ──
    print(f"\n[4/4] Generando predicciones...")
    preds = generate_predictions(upcoming, ucl_df, dom_df)

    if preds.empty:
        print("  No se generaron predicciones.")
        return

    # ── Imprimir resumen en consola ──
    SEP = "-" * 72
    print()
    print(SEP)
    # Usar nombres originales internos para el print (más cortos)
    # Buscamos las columnas por clave interna
    col_inv = {v: k for k, v in COLUMN_NAMES.items()}  # interno → header largo

    for _, r in preds.iterrows():
        # Acceder por nombre largo (el que tiene el DataFrame)
        fecha    = r[col_inv["fecha"]]
        comp     = str(r[col_inv["competicion"]])
        comp_s   = re.sub(r"UEFA Champions League.*", "UCL", comp)[:15]
        local    = str(r[col_inv["local"]])[:22]
        visit    = str(r[col_inv["visitante"]])[:22]
        pred     = r[col_inv["prediccion"]]
        p_h      = r[col_inv["p_local_pct"]]
        p_e      = r[col_inv["p_empate_pct"]]
        p_v      = r[col_inv["p_visit_pct"]]
        lh_v     = r[col_inv["lambda_h"]]
        la_v     = r[col_inv["lambda_a"]]
        forma_h  = str(r[col_inv["forma_h"]])
        forma_a  = str(r[col_inv["forma_a"]])
        conf     = r[col_inv["confianza_pct"]]
        w_ucl    = r[col_inv["w_ucl_pct"]]
        n_ucl_h  = r[col_inv["n_ucl_h"]]
        n_ucl_a  = r[col_inv["n_ucl_a"]]
        ventaja  = r[col_inv["ventaja"]]

        print(f"  {fecha}  [{comp_s:<15}]  {local:<22} vs  {visit}")
        print(f"  Goles esp.: {lh_v:.2f} - {la_v:.2f}  =>  "
              f"Local {p_h:.1f}%  Empate {p_e:.1f}%  Visitante {p_v:.1f}%  "
              f"[{pred}]  Confianza: {conf:.1f}%")
        print(f"  Forma: {forma_h} vs {forma_a}  | "
              f"Ventaja: {ventaja:+.3f}  | UCL n={n_ucl_h}/{n_ucl_a} peso={w_ucl}%")
        print(SEP)

    # ── Guardar CSV ──
    out_path = Path("data/predicciones_sofascore.csv")
    preds.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  Predicciones guardadas: {out_path}")
    print(f"  {len(preds)} partidos x {len(preds.columns)} columnas")

    dist = preds[col_inv["prediccion"]].value_counts()
    print(f"  Distribucion: {dict(dist)}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones Sofascore6 API")
    parser.add_argument("--days",         type=int, default=15,
                        help="Dias adelante para predicciones (default: 15)")
    parser.add_argument("--history-days", type=int, default=150,
                        help="Dias de historial para ratings (default: 150)")
    args = parser.parse_args()
    main(days_ahead=args.days, history_days=args.history_days)
