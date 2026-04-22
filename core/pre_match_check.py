"""
pre_match_check.py — Análisis 30 minutos antes del partido
===========================================================
Combina predicciones del modelo + plantilla confirmada + clima
para el análisis final antes del primer silbatazo.

Uso:
  python pre_match_check.py                        # partidos de hoy
  python pre_match_check.py --days 1               # hoy + mañana
  python pre_match_check.py --match "Chivas"       # filtrar equipo
  python pre_match_check.py --lineup lineups.json  # con XI confirmados

Formato del archivo de alineaciones (lineups.json):
{
  "jornada": "J16",
  "matches": [
    {
      "home": "Chivas",
      "away": "Toluca",
      "home_xi": ["Volpe", "Sepulveda", "Brizuela", "Hormiga", ...],
      "away_xi": ["Lajud", "Alexis", "Paulinho", ...],
      "home_injuries": ["Roberto Alvarado"],
      "away_injuries": []
    }
  ]
}

Variables que se consideran 30 min antes:
  - Plantilla confirmada (XI titular y lesionados de último minuto)
  - Clima en el estadio (Open-Meteo, gratis, sin API key)
  - Amenazas goleadoras activas (cruce XI vs. top scorers Clausura 2026)
  - Jugadores en alerta de tarjeta (actualizable por jornada)
  - Ajuste de lambda por ausencias de jugadores clave
  - Momios Playdoit en tiempo real + value bet final
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
for _p in (str(_ROOT / "utils"), str(_ROOT)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

CDT = timezone(timedelta(hours=-6))
W   = 76

# ── Coordenadas (lat, lon) y nombre de estadios por equipo ──
STADIUMS: dict[str, tuple[float, float, str]] = {
    "chivas":           (20.6868, -103.4666, "Estadio Akron, Guadalajara"),
    "atlas":            (20.6423, -103.4233, "Estadio El Jalisco, Guadalajara"),
    "america":          (19.3029,  -99.1505, "Estadio Azteca, CDMX"),
    "cruz azul":        (19.3929,  -99.1666, "Estadio Ciudad Deportes, CDMX"),
    "pumas":            (19.3213,  -99.1762, "Estadio CU, CDMX"),
    "tigres":           (25.7262, -100.3099, "Estadio Universitario, MTY"),
    "monterrey":        (25.6694, -100.2394, "Estadio BBVA, MTY"),
    "toluca":           (19.2872,  -99.6613, "Estadio Nemesio Diez, Toluca"),
    "leon":             (21.1250, -101.6840, "Estadio León"),
    "pachuca":          (20.1130,  -98.7291, "Estadio Hidalgo, Pachuca"),
    "atletico san luis":(22.0864, -100.8982, "Est. Alfonso Lastras, SLP"),
    "necaxa":           (21.8966, -102.2912, "Estadio Victoria, Aguascalientes"),
    "puebla":           (19.0469,  -98.2016, "Estadio Cuauhtémoc, Puebla"),
    "santos":           (25.5428, -103.4171, "Estadio El Volcán, Torreón"),
    "tijuana":          (32.5089, -117.0148, "Estadio Caliente, Tijuana"),
    "queretaro":        (20.5888, -100.3902, "Estadio Corregidora, Querétaro"),
    "mazatlan":         (23.2274, -106.4285, "Estadio Kraken, Mazatlán"),
    "juarez":           (31.7300, -106.4700, "Estadio Benito Juárez, Cd. Juárez"),
}

# ── Goleadores Clausura 2026 — fallback hardcoded (ESPN J17, ~Mayo 2026) ──
# La fuente PRIMARIA es ESPN API (fetch_espn_ligamx_scorers en context_enricher.py)
# Este dict se usa solo cuando la API falla.
CLAUSURA_SCORERS: list[dict] = [
    # ── 12 goles ──
    {"player": "Joao Pedro",            "team": "atletico san luis", "goals": 12},
    {"player": "Armando Gonzalez",       "team": "chivas",            "goals": 12},
    {"player": "Paulinho",               "team": "toluca",            "goals": 12},
    # ── 9 goles ──
    {"player": "German Berterame",       "team": "monterrey",         "goals":  9},
    {"player": "Sergio Canales",         "team": "monterrey",         "goals":  9},
    # ── 8 goles ──
    {"player": "Angel Correa",           "team": "tigres",            "goals":  8},
    {"player": "Juan Brunetta",          "team": "tigres",            "goals":  8},
    {"player": "Oscar Estupinan",        "team": "juarez",            "goals":  8},
    # ── 7 goles ──
    {"player": "Brian Rodriguez",        "team": "america",           "goals":  7},
    {"player": "Uros Djurdjevic",        "team": "monterrey",         "goals":  7},
    {"player": "Angel Sepulveda",        "team": "chivas",            "goals":  7},
    {"player": "Gabriel Fernandez",      "team": "cruz azul",         "goals":  7},
    # ── 6 goles ──
    {"player": "Emiliano Gomez",         "team": "puebla",            "goals":  6},
    {"player": "Frank Thierry Boya",     "team": "tijuana",           "goals":  6},
    {"player": "Diber Cambindo",         "team": "leon",              "goals":  6},
    {"player": "Ali Avila",              "team": "queretaro",         "goals":  6},
    # ── 5 goles ──
    {"player": "Diego Gonzalez",         "team": "atlas",             "goals":  5},
    {"player": "Kevin Castaneda",        "team": "tijuana",           "goals":  5},
    {"player": "Jose Paradela",          "team": "cruz azul",         "goals":  5},
    {"player": "Ezequiel Bullaude",      "team": "santos",            "goals":  5},
    {"player": "Juninho Vieira",         "team": "pumas",             "goals":  5},
]

# ── Jugadores en alerta de suspensión (≥4 tarjetas acumuladas) ──
# ACTUALIZAR antes de cada jornada con datos oficiales de Liga MX
# Suspensión automática: 5 amarillas en la misma fase
YELLOW_ALERTS: dict[str, dict] = {
    # "nombre_normalizado": {"team": "equipo", "yellows": N, "note": "contexto"}
    # Ejemplo (actualizar con datos reales de cada jornada):
    # "nombre jugador": {"team": "chivas", "yellows": 4, "note": "1 mas = suspension"},
}

# ── Factor de irreemplazabilidad de un goleador ──
# 0.40 = el 40% de los goles de un jugador no pueden ser compensados
# por el resto del equipo en el corto plazo
IRREPLACE_FACTOR = 0.40

# ── Playdoit (mismo endpoint que _reporte_playdoit.py) ──
PD_BASE   = "https://sb2frontend-altenar2.biahosted.com/api/widget"
PD_PARAMS = {
    "culture": "es-ES", "timezoneOffset": "360", "integration": "playdoit2",
    "lang": "es-ES", "timezone": "America/Mexico_City",
    "champIds": "10009", "sportId": "66",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. MÓDULO DE CLIMA (Open-Meteo — gratuito, sin API key)
# ─────────────────────────────────────────────────────────────────────────────

# Códigos WMO → descripción y factor de ajuste de lambda
_WMO_TABLE: list[tuple[range, str, float]] = [
    (range(0, 4),   "Despejado / Nublado",         1.00),
    (range(45, 49), "Niebla",                       0.97),
    (range(51, 56), "Llovizna ligera",               0.96),
    (range(56, 68), "Lluvia moderada",               0.93),
    (range(68, 78), "Nieve",                         0.90),
    (range(80, 83), "Chubascos",                     0.91),
    (range(85, 87), "Nieve intensa",                 0.88),
    (range(95, 100),"Tormenta eléctrica",            0.87),
]

def _wmo_info(code: int) -> tuple[str, float]:
    for r, desc, factor in _WMO_TABLE:
        if code in r:
            return desc, factor
    if code < 45:
        return "Parcialmente nublado", 1.00
    return f"Código {code}", 0.95


def fetch_weather(home_team: str, match_hour_cdt: int = -1) -> dict | None:
    """
    Consulta Open-Meteo para el estadio del equipo local.
    Retorna dict con temp, precip, viento, descripción y factor lambda, o None si falla.
    """
    key = _strip(home_team)
    stadium = STADIUMS.get(key)
    if not stadium:
        # Fuzzy fallback
        best, bscore = None, 0.60
        for sk in STADIUMS:
            r = SequenceMatcher(None, key, sk).ratio()
            if r > bscore:
                bscore, best = r, sk
        if best:
            stadium = STADIUMS[best]
    if not stadium:
        return None

    lat, lon, stadium_name = stadium
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "current":   "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone":  "America/Mexico_City",
        "forecast_days": 1,
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return None
        c = r.json()["current"]
        code   = int(c.get("weather_code", 0))
        temp   = float(c.get("temperature_2m", 20))
        precip = float(c.get("precipitation", 0))
        wind   = float(c.get("wind_speed_10m", 0))   # km/h
        humid  = float(c.get("relative_humidity_2m", 60))
        desc, lam_factor = _wmo_info(code)

        # Viento fuerte adicional
        wind_factor = 1.00
        if wind > 35:
            wind_factor = 0.97
            desc += f" + viento fuerte"
        elif wind > 25:
            wind_factor = 0.99

        total_factor = round(lam_factor * wind_factor, 3)

        # Icono
        if code == 0:
            icon = "☀"
        elif code <= 3:
            icon = "⛅"
        elif code <= 48:
            icon = "🌫"
        elif code <= 67:
            icon = "🌧"
        elif code <= 77:
            icon = "❄"
        elif code <= 82:
            icon = "🌦"
        elif code >= 95:
            icon = "⛈"
        else:
            icon = "🌥"

        return {
            "stadium":      stadium_name,
            "temp":         temp,
            "precip":       precip,
            "wind":         wind,
            "humidity":     humid,
            "code":         code,
            "description":  desc,
            "icon":         icon,
            "lambda_factor":total_factor,
        }
    except Exception:
        return None


def _weather_impact_msg(factor: float) -> str:
    if factor >= 0.99:
        return "sin impacto en el juego"
    elif factor >= 0.96:
        return f"impacto leve (-{round((1-factor)*100):.0f}% goles esperados)"
    elif factor >= 0.92:
        return f"impacto moderado (-{round((1-factor)*100):.0f}% goles esperados)"
    else:
        return f"impacto significativo (-{round((1-factor)*100):.0f}% goles esperados)"


# ─────────────────────────────────────────────────────────────────────────────
# 3. MÓDULO DE PLANTILLA Y GOLEADORES
# ─────────────────────────────────────────────────────────────────────────────

def _strip(s: str) -> str:
    """Normaliza: minúsculas, sin tildes, sin prefijos comunes."""
    import unicodedata
    s = str(s).lower().strip()
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    s = re.sub(r"\b(fc|cf|club|cd|unam|uanl|deportivo|atletico)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _get_live_scorers() -> list[dict]:
    """
    Intenta obtener goleadores en vivo desde ESPN API via context_enricher.
    Si falla, devuelve la lista hardcoded local (fallback).
    """
    try:
        from context_enricher import fetch_espn_ligamx_scorers
        live = fetch_espn_ligamx_scorers(ttl_hours=6)
        if live:
            # Normalizar team names a lower para este script
            return [
                {**s, "team": _strip(s.get("team", ""))}
                for s in live
            ]
    except Exception:
        pass
    return CLAUSURA_SCORERS


def build_scorer_model(parquet_path: str = "data/sofascore_events.parquet") -> dict:
    """
    Construye {team_lower: {total_goals, players: [{player, goals, share, gpg}]}}.
    Fuente de scorers: ESPN API en vivo (con fallback a hardcoded).
    team total_goals viene del parquet; player goal share de la lista activa.
    """
    # Obtener goleadores: ESPN API primero, fallback hardcoded
    active_scorers = _get_live_scorers()

    # Total goles por equipo en Clausura 2026 (tournament_id=11620, status=finished)
    team_goals: dict[str, int] = {}
    try:
        df = pd.read_parquet(parquet_path)
        mx = df[(df["tournament_id"] == 11620) &
                (df["status"] == "finished")].copy()
        mx["games"] = 1
        # Goles marcados como local
        for _, row in mx.groupby("home_team").agg({"home_score": "sum"}).iterrows():
            pass
        home_g = mx.groupby("home_team")["home_score"].sum()
        away_g = mx.groupby("away_team")["away_score"].sum()
        all_teams = set(home_g.index) | set(away_g.index)
        for t in all_teams:
            key = _strip(t)
            team_goals[key] = int(home_g.get(t, 0)) + int(away_g.get(t, 0))
    except Exception:
        # Fallback: estimar 20 goles por equipo si no hay parquet
        for s in active_scorers:
            team_goals.setdefault(_strip(s["team"]), 20)

    # Construir modelo por equipo
    model: dict = {}
    for s in active_scorers:
        key = _strip(s["team"])
        if key not in model:
            model[key] = {"total_goals": team_goals.get(key, 20), "players": []}
        model[key]["players"].append({
            "player": s["player"],
            "goals":  s["goals"],
        })

    # Calcular share y goals-per-game (15 jornadas)
    for key, data in model.items():
        total = max(data["total_goals"], 1)
        games = 15  # Clausura 2026: J15 completada
        for p in data["players"]:
            p["share"] = round(p["goals"] / total, 3)
            p["gpg"]   = round(p["goals"] / games, 3)

    return model


def find_player_in_xi(player_name: str, xi: list[str]) -> bool:
    """Busca si un jugador aparece en el XI (fuzzy matching)."""
    pn = _strip(player_name)
    for xi_player in xi:
        xp = _strip(xi_player)
        # Exact
        if pn == xp:
            return True
        # Nombre o apellido incluido
        if pn in xp or xp in pn:
            return True
        # Apellido como sufijo
        parts = pn.split()
        for part in parts:
            if len(part) >= 4 and xp.endswith(part):
                return True
        # Fuzzy
        if SequenceMatcher(None, pn, xp).ratio() >= 0.78:
            return True
    return False


def calc_lineup_impact(team: str, xi: list[str], injuries: list[str],
                       scorer_model: dict) -> dict:
    """
    Calcula factor de ajuste de lambda basado en ausencias de goleadores clave.

    Retorna:
      lambda_factor   : float ≤ 1.0 (1.0 = plantilla completa)
      threats_in_xi   : goleadores que SÍ están en el XI
      absent_scorers  : goleadores que NO están en el XI
      injury_flags    : jugadores listados como lesionados
    """
    key = _strip(team)
    if key not in scorer_model or not xi:
        return {
            "lambda_factor":   1.0,
            "threats_in_xi":   [],
            "absent_scorers":  [],
            "injury_flags":    injuries or [],
            "note":            "sin datos de plantilla o XI no proporcionado",
        }

    data     = scorer_model[key]
    total_g  = data["total_goals"]
    players  = data["players"]

    absent_scorers = []
    threats_in_xi  = []

    for p in players:
        in_xi = find_player_in_xi(p["player"], xi)
        if in_xi:
            threats_in_xi.append(p)
        else:
            absent_scorers.append(p)

    # Calcular reducción de lambda por ausencias
    # Solo aplica para jugadores confirmados ausentes (share × IRREPLACE_FACTOR)
    total_share_absent = sum(p["share"] for p in absent_scorers)
    lambda_factor = max(0.65, 1.0 - total_share_absent * IRREPLACE_FACTOR)

    return {
        "lambda_factor":   round(lambda_factor, 3),
        "threats_in_xi":   threats_in_xi,
        "absent_scorers":  absent_scorers,
        "injury_flags":    injuries or [],
        "note":            f"{len(threats_in_xi)} goleadores en XI / {len(absent_scorers)} ausentes",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. CÁLCULO DE PROBABILIDADES (Poisson + Dixon-Coles simplificado)
# ─────────────────────────────────────────────────────────────────────────────

def _poisson_1x2(lh: float, la: float, max_g: int = 8) -> tuple[float, float, float]:
    ph = pd_ = pv = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = (math.exp(-lh) * lh**h / math.factorial(h) *
                 math.exp(-la) * la**a / math.factorial(a))
            if h > a:    ph  += p
            elif h == a: pd_ += p
            else:        pv  += p
    total = ph + pd_ + pv
    return (ph/total*100, pd_/total*100, pv/total*100) if total else (33.3, 33.3, 33.3)


def _smooth_draw(ph: float, pd_: float, pv: float, floor: float = 15.0):
    if pd_ >= floor:
        return ph, pd_, pv
    deficit = floor - pd_
    total_other = ph + pv
    if total_other < 1e-9:
        return ph, floor, pv
    ph  -= deficit * (ph / total_other)
    pv  -= deficit * (pv / total_other)
    return max(ph, 0), floor, max(pv, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLAYDOIT (reutiliza mismo endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_playdoit() -> list[dict]:
    r    = requests.get(f"{PD_BASE}/GetUpcoming", params=PD_PARAMS, timeout=15)
    data = r.json()
    markets     = {m["id"]: m for m in data["markets"]}
    odds_map    = {o["id"]: o for o in data["odds"]}
    competitors = {c["id"]: c for c in data["competitors"]}
    result = []
    for ev in data["events"]:
        if ev.get("champId") != 10009:
            continue
        start = (datetime.fromisoformat(ev["startDate"].replace("Z", ""))
                 .replace(tzinfo=timezone.utc).astimezone(CDT))
        comps = [competitors.get(cid, {}).get("name", "?") for cid in ev.get("competitorIds", [])]
        if len(comps) < 2:
            continue
        h12 = next((markets[mid] for mid in ev["marketIds"]
                    if markets.get(mid, {}).get("name") == "1x2"), None)
        if not h12:
            continue
        os_ = [odds_map.get(oid) for oid in h12["oddIds"]]
        result.append({
            "home": comps[0], "away": comps[1],
            "dia":  start.strftime("%a %d/%m"),
            "hora": start.strftime("%H:%M"),
            "o_h":  next((o["price"] for o in os_ if o and o.get("typeId") == 1), None),
            "o_d":  next((o["price"] for o in os_ if o and o.get("typeId") == 2), None),
            "o_a":  next((o["price"] for o in os_ if o and o.get("typeId") == 3), None),
        })
    return result


def _find_playdoit(home: str, away: str, pd_list: list[dict]) -> dict | None:
    hn, an = _strip(home), _strip(away)
    best, bscore = None, 0.50
    for item in pd_list:
        sh = SequenceMatcher(None, hn, _strip(item["home"])).ratio()
        sa = SequenceMatcher(None, an, _strip(item["away"])).ratio()
        s  = (sh + sa) / 2
        if s > bscore:
            bscore, best = s, item
    return best


def _no_vig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    ph, pd_, pa = 1/oh, 1/od, 1/oa
    t = ph + pd_ + pa
    return ph/t*100, pd_/t*100, pa/t*100


def _ev(p_pct: float, o: float) -> float:
    return round((p_pct/100 * (o-1) - (1 - p_pct/100)) * 100, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CARGA DE ALINEACIONES
# ─────────────────────────────────────────────────────────────────────────────

def load_lineups(path: str) -> dict:
    """Carga lineups.json. Devuelve {(home_key, away_key): match_dict}."""
    p = Path(path)
    if not p.exists():
        print(f"  AVISO: {path} no encontrado — continuando sin alineaciones")
        return {}
    data    = json.loads(p.read_bytes().decode("utf-8"))
    matches = data if isinstance(data, list) else data.get("matches", [])
    lineup_map = {}
    for m in matches:
        h_key = _strip(m.get("home", ""))
        a_key = _strip(m.get("away", ""))
        lineup_map[(h_key, a_key)] = m
    print(f"  Alineaciones cargadas: {len(lineup_map)} partidos")
    return lineup_map


def _find_lineup(home: str, away: str, lineup_map: dict) -> dict | None:
    kh, ka = _strip(home), _strip(away)
    if (kh, ka) in lineup_map:
        return lineup_map[(kh, ka)]
    best, bscore = None, 0.70
    for (h, a), m in lineup_map.items():
        s = (SequenceMatcher(None, kh, h).ratio() + SequenceMatcher(None, ka, a).ratio()) / 2
        if s > bscore:
            bscore, best = s, m
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 7. REPORTE POR PARTIDO
# ─────────────────────────────────────────────────────────────────────────────

def _bar(p: float, w: int = 14) -> str:
    f = round(p / 100 * w)
    return "█" * f + "░" * (w - f)


def print_match_report(row: pd.Series, weather: dict | None,
                       lineup_h: dict | None, lineup_a: dict | None,
                       pd_item: dict | None) -> list[str]:
    """Imprime el reporte pre-partido. Devuelve lista de value bets detectados."""
    home  = str(row["Equipo Local"])
    away  = str(row["Equipo Visitante"])
    fecha = str(row["Fecha"])
    ph0   = float(row["Probabilidad Local % (p_local)"])
    pde0  = float(row["Probabilidad Empate % (p_empate)"])
    pv0   = float(row["Probabilidad Visitante % (p_visit)"])
    lh0   = float(row["Goles Esperados Local (lambda_h)"])
    la0   = float(row["Goles Esperados Visitante (lambda_a)"])
    pred0 = str(row["Prediccion"])
    conf  = str(row["Nivel de Confianza (ALTA / MEDIA / BAJA)"])
    pos_h = int(row["Posicion Tabla Local (pos_h)"])
    pos_a = int(row["Posicion Tabla Visitante (pos_v)"])
    pts_h = int(row["Puntos en Tabla Local (pts_tabla_h)"])
    pts_a = int(row["Puntos en Tabla Visitante (pts_tabla_v)"])
    forma_h = str(row.get("Forma Reciente Local W/D/L (forma_h)", ""))
    forma_a = str(row.get("Forma Reciente Visitante W/D/L (forma_a)", ""))
    narr    = str(row.get("Narrativa del Partido (narrativa)", ""))
    ah      = row.get("Amarillas Promedio Local (amarillas_local)")
    aa      = row.get("Amarillas Promedio Visitante (amarillas_visita)")
    at_csv  = row.get("Amarillas Total Estimadas (amarillas_total)")
    ch      = row.get("Corners Predichos Local (corners_h)")
    ca      = row.get("Corners Predichos Visitante (corners_a)")
    ct      = row.get("Corners Total Predichos (corners_total)")
    gol_h   = str(row.get("Goleadores Local (goleadores_local)", "N/D"))
    gol_a   = str(row.get("Goleadores Visitante (goleadores_visita)", "N/D"))
    alerta_e = str(row.get("Alerta Empate Probable (alerta_empate)", ""))

    hora = pd_item["hora"] if pd_item else "?"

    print(f"\n{'─'*W}")
    print(f"  {fecha}  {hora} CDT  │  {home}  vs  {away}")
    print(f"  Tabla: #{pos_h} {home[:18]} ({pts_h}pts)   vs   #{pos_a} {away[:18]} ({pts_a}pts)")
    if forma_h:
        print(f"  Forma: {home[:18]} [{forma_h}]  vs  {away[:18]} [{forma_a}]")
    if narr and narr != "nan":
        print(f"  {narr[:72]}")

    # ── CLIMA ──────────────────────────────────────────────────────────
    print()
    if weather:
        imp_msg = _weather_impact_msg(weather["lambda_factor"])
        print(f"  CLIMA  {weather['stadium']}")
        print(f"  {weather['icon']} {weather['description']}  "
              f"{weather['temp']:.0f}°C  Viento {weather['wind']:.0f} km/h  "
              f"Precip {weather['precip']:.1f} mm  →  {imp_msg}")
    else:
        print(f"  CLIMA  No disponible (verifica conexión a Open-Meteo)")

    # ── PREDICCIÓN BASE ─────────────────────────────────────────────────
    print()
    print(f"  PREDICCION BASE  λ {lh0:.2f} vs {la0:.2f}  →  [{pred0}]  Conf={conf}")
    print(f"  Local {ph0:.1f}% {_bar(ph0)}  "
          f"Empate {pde0:.1f}% {_bar(pde0, 10)}  "
          f"Visitante {pv0:.1f}%")
    if alerta_e and alerta_e != "nan" and "Empate" in alerta_e:
        print(f"  ⚠ {alerta_e}")

    # ── AJUSTE POR CLIMA + PLANTILLA ───────────────────────────────────
    weather_factor = weather["lambda_factor"] if weather else 1.0
    lf_h = lineup_h["lambda_factor"] if lineup_h else 1.0
    lf_a = lineup_a["lambda_factor"] if lineup_a else 1.0

    lh_adj = lh0 * weather_factor * lf_h
    la_adj = la0 * weather_factor * lf_a

    has_adjustment = (abs(lh_adj - lh0) > 0.02 or abs(la_adj - la0) > 0.02)

    if has_adjustment:
        ph_adj, pde_adj, pv_adj = _smooth_draw(*_poisson_1x2(lh_adj, la_adj))
        # Predicción ajustada
        labels = ["Local", "Empate", "Visitante"]
        probs  = [ph_adj, pde_adj, pv_adj]
        pred_adj = labels[probs.index(max(probs))]
        # Regla de empate
        if probs[1] >= 25.0 and (max(probs) - probs[1]) < 10.0 and pred_adj != "Empate":
            pred_adj = "Empate"
        print()
        print(f"  PREDICCION AJUSTADA  (clima×plantilla)")
        factors = []
        if weather_factor < 0.99:
            factors.append(f"clima {weather_factor:.2f}x")
        if abs(lf_h - 1.0) > 0.01:
            factors.append(f"{home[:12]} plantilla {lf_h:.2f}x")
        if abs(lf_a - 1.0) > 0.01:
            factors.append(f"{away[:12]} plantilla {lf_a:.2f}x")
        if factors:
            print(f"  Factores: {' | '.join(factors)}")
        print(f"  λ ajustado: {lh_adj:.2f} vs {la_adj:.2f}  "
              f"(base: {lh0:.2f} vs {la0:.2f})")
        print(f"  Local {ph_adj:.1f}% {_bar(ph_adj)}  "
              f"Empate {pde_adj:.1f}% {_bar(pde_adj, 10)}  "
              f"Visitante {pv_adj:.1f}%")
        print(f"  PICK AJUSTADO: [{pred_adj}]")
        ph_use, pde_use, pv_use = ph_adj, pde_adj, pv_adj
    else:
        ph_use, pde_use, pv_use = ph0, pde0, pv0

    # ── PLANTILLA Y GOLEADORES ────────────────────────────────────────
    print()
    print(f"  GOLEADORES Y PLANTILLA")

    def _print_team_section(team: str, lf_info: dict | None, gol_csv: str) -> None:
        print(f"  {team[:26]}:")
        if lf_info and lf_info.get("threats_in_xi"):
            for p in lf_info["threats_in_xi"][:3]:
                share_pct = round(p["share"] * 100)
                gol_s = f"{p['goals']}g ({share_pct}% goles eq)"
                if p == lf_info["threats_in_xi"][0]:
                    print(f"    ⚽ {p['player']} — {gol_s}  ← AMENAZA CLAVE")
                else:
                    print(f"       {p['player']} — {gol_s}")
        elif not lf_csv_is_null(gol_csv):
            parts = [p.strip() for p in gol_csv.split(",") if p.strip()]
            for i, p in enumerate(parts[:3]):
                prefix = "    ⚽" if i == 0 else "      "
                suffix = "  ← AMENAZA CLAVE" if i == 0 else ""
                print(f"  {prefix} {p}{suffix}")
        else:
            print("    (sin goleadores destacados este torneo)")

        if lf_info and lf_info.get("absent_scorers"):
            for p in lf_info["absent_scorers"][:2]:
                share_pct = round(p["share"] * 100)
                print(f"    ✗ AUSENTE: {p['player']} {p['goals']}g ({share_pct}% goles) "
                      f"→ λ -{round(p['share'] * IRREPLACE_FACTOR * 100):.0f}%")

        if lf_info and lf_info.get("injury_flags"):
            for inj in lf_info["injury_flags"][:3]:
                print(f"    🚑 LESION/BAJA: {inj}")

    def lf_csv_is_null(s: str) -> bool:
        return not s or str(s).startswith("N/D") or str(s) == "nan"

    _print_team_section(home, lineup_h, gol_h)
    _print_team_section(away, lineup_a, gol_a)

    # XI completo si fue proporcionado
    if lineup_h and lineup_h.get("xi"):
        print(f"  XI {home[:18]}: {' | '.join(lineup_h['xi'][:11])}")
    if lineup_a and lineup_a.get("xi"):
        print(f"  XI {away[:18]}: {' | '.join(lineup_a['xi'][:11])}")

    # ── TARJETAS AMARILLAS ────────────────────────────────────────────
    print()
    print(f"  TARJETAS AMARILLAS")
    if pd.notna(ah) and pd.notna(aa):
        total_cards = float(ah) + float(aa)
        trend = ""
        if total_cards > 4.5:
            trend = "  ⚠ PARTIDO CALIENTE — alto volumen de tarjetas esperado"
        elif total_cards > 3.5:
            trend = "  (moderado — algunos roces esperados)"
        at_s = f"~{float(at_csv):.1f}" if pd.notna(at_csv) else "?"
        print(f"  {home[:18]} {float(ah):.1f} tarj/p   vs   {away[:18]} {float(aa):.1f} tarj/p   "
              f"Total {at_s}{trend}")
    else:
        print("  (datos de tarjetas no disponibles)")

    # Alertas de suspensión individuales
    if YELLOW_ALERTS:
        alerts_h = [v for k, v in YELLOW_ALERTS.items() if v["team"] == _strip(home)]
        alerts_a = [v for k, v in YELLOW_ALERTS.items() if v["team"] == _strip(away)]
        for a in alerts_h[:3]:
            print(f"  ⚠ SUSPENSION ALERT ({home[:15]}): {a.get('player','?')} "
                  f"{a['yellows']} amarillas — {a.get('note','1 mas = suspension')}")
        for a in alerts_a[:3]:
            print(f"  ⚠ SUSPENSION ALERT ({away[:15]}): {a.get('player','?')} "
                  f"{a['yellows']} amarillas — {a.get('note','1 mas = suspension')}")
    else:
        print("  (actualiza YELLOW_ALERTS en este script antes de cada jornada)")

    # ── CORNERS ───────────────────────────────────────────────────────
    if pd.notna(ch) and pd.notna(ca):
        print()
        print(f"  CORNERS    {home[:18]} {float(ch):.1f}  vs  {away[:18]} {float(ca):.1f}   "
              f"Total ~{float(ct):.1f}")
        if weather and weather["wind"] > 25:
            wind_adj = float(ct) * 0.92
            print(f"  (viento {weather['wind']:.0f} km/h → corners reales ~{wind_adj:.1f})")

    # ── MOMIOS PLAYDOIT ───────────────────────────────────────────────
    print()
    value_bets = []
    if pd_item and all([pd_item.get("o_h"), pd_item.get("o_d"), pd_item.get("o_a")]):
        oh, od, oa = pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]
        vig  = round((1/oh + 1/od + 1/oa - 1) * 100, 2)
        mh, md, ma = _no_vig(oh, od, oa)
        e_h = round(ph_use - mh, 1)
        e_d = round(pde_use - md, 1)
        e_v = round(pv_use - ma, 1)
        ev_h = _ev(ph_use, oh)
        ev_d = _ev(pde_use, od)
        ev_v = _ev(pv_use, oa)

        print(f"  PLAYDOIT  vig={vig:.1f}%  {'[probabilidades ajustadas]' if has_adjustment else '[prediccion base]'}")
        print(f"  {'':11} {'Momio':>8}  {'Casas%':>7}  {'Modelo%':>8}  {'Edge':>7}  {'EV':>7}  Señal")
        print(f"  {'─'*70}")

        for lbl, o_v, casas, modelo, edge, ev in [
            ("Local",     oh, mh, ph_use,  e_h, ev_h),
            ("Empate",    od, md, pde_use, e_d, ev_d),
            ("Visitante", oa, ma, pv_use,  e_v, ev_v),
        ]:
            senal = "◄◄ VALOR ALTO" if edge >= 8 else (
                    "◄ VALOR"       if edge >= 5 else (
                    "leve+"         if edge >= 3 else (
                    "neutro"        if edge >= -3 else "sin edge")))
            print(f"  {lbl:<11} {o_v:>8.4f}  {casas:>6.1f}%  {modelo:>7.1f}%  "
                  f"{edge:>+6.1f}%  {ev:>+6.1f}%  {senal}")
            if edge >= 5:
                value_bets.append(
                    f"{home} vs {away}  {lbl} @ {o_v:.4f}  edge={edge:+.1f}%  EV={ev:+.1f}%"
                )

        print(f"  {'─'*70}")

        # VEREDICTO
        best_ev  = max(ev_h, ev_d, ev_v)
        best_lbl = ["Local", "Empate", "Visitante"][[ev_h, ev_d, ev_v].index(best_ev)]
        best_edge = [e_h, e_d, e_v][[ev_h, ev_d, ev_v].index(best_ev)]
        if best_edge >= 8:
            verdict = f"APOSTAR: {best_lbl} @ {[oh,od,oa][['Local','Empate','Visitante'].index(best_lbl)]:.4f} (edge fuerte {best_edge:+.1f}%)"
        elif best_edge >= 5:
            verdict = f"CONSIDERAR: {best_lbl} (edge {best_edge:+.1f}% — valor presente)"
        elif best_ev > 0:
            verdict = f"Sin edge claro. EV levemente positivo en {best_lbl} ({best_ev:+.1f}%)"
        else:
            verdict = "Sin valor vs Playdoit — línea en precio justo o cara"
        print()
        print(f"  ► VEREDICTO: {verdict}")
    else:
        print("  [sin momios Playdoit disponibles para este partido]")

    return value_bets


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-match check — análisis 30 min antes del partido"
    )
    ap.add_argument("--days",    type=int, default=0,
                    help="Días hacia adelante (0=solo hoy, 1=hoy+mañana, etc.)")
    ap.add_argument("--match",   default="",
                    help="Filtrar por nombre de equipo (ej. 'Chivas')")
    ap.add_argument("--lineup",  default="",
                    help="Ruta al JSON de alineaciones confirmadas")
    ap.add_argument("--no-weather", action="store_true",
                    help="Omitir consulta de clima (más rápido)")
    args = ap.parse_args()

    now     = datetime.now(CDT)
    today   = now.strftime("%Y-%m-%d")
    horizon = (now + timedelta(days=args.days)).strftime("%Y-%m-%d")

    # ── Encabezado ──
    print()
    print("=" * W)
    print(f"  PRE-MATCH CHECK  —  Liga MX Clausura 2026")
    print(f"  {now.strftime('%A %d/%m/%Y  %H:%M')} CDT")
    print(f"  Variables: plantilla + clima + goleadores + tarjetas + momios Playdoit")
    print("=" * W)

    # ── Cargar predicciones ──
    df     = pd.read_csv("data/ligamx_predicciones.csv")
    ligamx = df[df["Fecha"].between(today, horizon)].copy()
    if args.match:
        m = args.match.lower()
        ligamx = ligamx[
            ligamx["Equipo Local"].str.lower().str.contains(m) |
            ligamx["Equipo Visitante"].str.lower().str.contains(m)
        ]
    if ligamx.empty:
        print(f"\n  Sin partidos en el rango {today} → {horizon}")
        return

    print(f"\n  {len(ligamx)} partido(s) encontrado(s) en el CSV")

    # ── Construir modelo de goleadores ──
    print("  Construyendo modelo de goleadores...")
    scorer_model = build_scorer_model()

    # ── Cargar alineaciones ──
    lineup_map: dict = {}
    if args.lineup:
        lineup_map = load_lineups(args.lineup)

    # ── Fetch Playdoit ──
    print("  Obteniendo momios Playdoit...", end=" ")
    try:
        pd_list = fetch_playdoit()
        print(f"{len(pd_list)} partidos")
    except Exception as e:
        print(f"ERROR ({e})")
        pd_list = []

    # ── Procesar cada partido ──
    all_value_bets: list[str] = []

    for _, row in ligamx.sort_values("Fecha").iterrows():
        home = str(row["Equipo Local"])
        away = str(row["Equipo Visitante"])

        # Clima
        weather = None
        if not args.no_weather:
            print(f"\n  Consultando clima para {home}...", end=" ")
            weather = fetch_weather(home)
            if weather:
                print(f"OK ({weather['description']})")
            else:
                print("sin datos")
            time.sleep(0.3)   # evitar rate limit

        # Playdoit
        pd_item = _find_playdoit(home, away, pd_list)

        # Alineaciones
        lineup_data = _find_lineup(home, away, lineup_map) if lineup_map else None
        home_xi      = lineup_data.get("home_xi", []) if lineup_data else []
        away_xi      = lineup_data.get("away_xi", []) if lineup_data else []
        home_injuries = lineup_data.get("home_injuries", []) if lineup_data else []
        away_injuries = lineup_data.get("away_injuries", []) if lineup_data else []

        lf_home = calc_lineup_impact(home, home_xi, home_injuries, scorer_model) if home_xi else None
        lf_away = calc_lineup_impact(away, away_xi, away_injuries, scorer_model) if away_xi else None

        # Agregar xi al dict para display
        if lf_home and home_xi:
            lf_home["xi"] = home_xi
        if lf_away and away_xi:
            lf_away["xi"] = away_xi

        # Reporte
        vbets = print_match_report(row, weather, lf_home, lf_away, pd_item)
        all_value_bets.extend(vbets)

    # ── Resumen de value bets ──
    print()
    print("=" * W)
    if all_value_bets:
        print(f"  VALUE BETS DETECTADAS ({len(all_value_bets)})")
        print("=" * W)
        for vb in all_value_bets:
            print(f"  ★ {vb}")
    else:
        print("  Sin value bets con edge ≥5% en los partidos analizados")
    print("=" * W)
    print()
    print("  NOTAS IMPORTANTES")
    print("  - Las probabilidades ajustadas solo aplican cuando se usa --lineup")
    print("  - Actualiza YELLOW_ALERTS en este script antes de cada jornada")
    print("  - Actualiza CLAUSURA_SCORERS tras cada jornada (línea ~37)")
    print("  - El modelo no garantiza resultados; usa bankroll responsable")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
