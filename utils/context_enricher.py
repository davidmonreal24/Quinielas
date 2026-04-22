"""
context_enricher.py
Enriquece las predicciones con:
  1. Corrección Dixon-Coles (ajuste de empate)
  2. Índice de motivación por posición de tabla
  3. Goleadores top (API-Football, última temporada disponible)
  4. Estimación de tiros de esquina (a partir de λ)
  5. Análisis situacional del partido (significado del encuentro)
"""
from __future__ import annotations
import sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(_ROOT / "utils"))
    sys.path.insert(0, str(_ROOT))

import json
import math
import requests
import functools
import unicodedata
from typing import Optional
from utils.config import APIFOOTBALL_KEY, APIFOOTBALL_BASE, APIFOOTBALL_IDS  # noqa: E402

# ──────────────────────────────────────────────────────────────
# 1. CORRECCIÓN DIXON-COLES
# ──────────────────────────────────────────────────────────────
# ρ negativo: aumenta P(0-0) y P(1-1), reduce P(0-1) y P(1-0)
# Valor calibrado para fútbol: ρ ≈ -0.13  (Dixon & Coles 1997)
DC_RHO = -0.13

def _tau(x: int, y: int, lh: float, la: float, rho: float) -> float:
    """Factor de corrección Dixon-Coles para resultados {0-0,0-1,1-0,1-1}."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0

def dixon_coles_probs(lh: float, la: float,
                      max_goals: int = 8,
                      rho: float = DC_RHO) -> tuple[float, float, float]:
    """
    Calcula P(Local), P(Empate), P(Visitante) con corrección Dixon-Coles.
    Más preciso que Poisson puro para resultados de baja puntuación.
    """
    p_home = p_draw = p_away = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_ij = (math.exp(-lh) * lh**i / math.factorial(i) *
                    math.exp(-la) * la**j / math.factorial(j))
            if i <= 1 and j <= 1:
                p_ij *= _tau(i, j, lh, la, rho)
            if i > j:
                p_home += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away += p_ij
    total = p_home + p_draw + p_away
    if total < 1e-9:
        return 1/3, 1/3, 1/3
    return p_home / total, p_draw / total, p_away / total


# ──────────────────────────────────────────────────────────────
# 2. SUAVIZADO DE EMPATE (floor 15 %)
# ──────────────────────────────────────────────────────────────
DRAW_FLOOR = 0.15

def smooth_draw(ph: float, pd: float, pv: float,
                floor: float = DRAW_FLOOR) -> tuple[float, float, float]:
    if pd >= floor:
        return ph, pd, pv
    deficit = floor - pd
    total_other = ph + pv
    if total_other < 1e-9:
        return ph, floor, pv
    ph -= deficit * ph / total_other
    pv -= deficit * pv / total_other
    return max(ph, 0), floor, max(pv, 0)


# ──────────────────────────────────────────────────────────────
# 3. ÍNDICE DE MOTIVACIÓN
# ──────────────────────────────────────────────────────────────
# Liga MX Clausura: 18 equipos, top-8 → Liguilla, bottom-3 (cociente)
LIGAMX_TOTAL = 18
LIGAMX_PLAYOFF = 8      # posiciones 1-8 clasifican
LIGAMX_DIRECT_QF = 4    # posiciones 1-4 directo a cuartos
LIGAMX_RELZONE = 3      # posiciones 16-18 en zona de riesgo (cociente)

# UCL: eliminación directa → motivación máxima siempre
UCL_MOTIVATION = 1.25

def ligamx_situation(pos: int, pts: int, pts_leader: int,
                     pts_8th: int, pts_16th: int,
                     jornada: int = 0, total_jornadas: int = 17) -> dict:
    """
    Retorna situación y factor de motivación para un equipo de Liga MX.
    pos               : posición actual (1=líder)
    pts               : puntos actuales
    pts_leader        : puntos del 1er lugar
    pts_8th           : puntos del 8vo lugar (último en zona de Liguilla)
    pts_16th          : puntos del 16vo lugar (límite zona de riesgo cociente)
    jornada           : jornada actual (para detectar si ya está eliminado matemáticamente)
    total_jornadas    : total de jornadas en la fase regular
    """
    pts_remaining = (total_jornadas - jornada) * 3
    pts_to_playoffs  = max(0, pts_8th  - pts)
    pts_to_leader    = max(0, pts_leader - pts)
    pts_above_danger = pts - pts_16th    # negativo = en zona de riesgo

    # Eliminado matemáticamente: no puede alcanzar el playoff
    already_eliminated = (pts_remaining > 0) and (pts + pts_remaining < pts_8th)

    if already_eliminated:
        # Sin incentivo de tabla → motivación mínima (juegan por orgullo/cociente)
        label  = "Eliminado / Sin incentivo Liguilla"
        icon   = "💤"
        factor = 0.88
    elif pos <= LIGAMX_DIRECT_QF:
        label = "Líder / Cuartos directos"
        icon  = "🏆"
        factor = 1.08
    elif pos <= LIGAMX_PLAYOFF:
        label = "Zona Liguilla"
        icon  = "⚽"
        factor = 1.12
    elif pos <= LIGAMX_PLAYOFF + 3:
        label = "Persiguiendo Liguilla"
        icon  = "🔥"
        factor = 1.18
    elif pos <= LIGAMX_TOTAL - LIGAMX_RELZONE:
        label = "Zona Media / Segura"
        icon  = "😐"
        factor = 1.00
    elif pos <= LIGAMX_TOTAL - 1:
        label = "Zona de Riesgo (cociente)"
        icon  = "⚠️"
        factor = 1.20
    else:
        label = "Último lugar"
        icon  = "🆘"
        factor = 1.25

    # Extra boost si la diferencia con el 8vo es ≤ 3 puntos y aún no eliminado
    if not already_eliminated and LIGAMX_PLAYOFF < pos <= LIGAMX_PLAYOFF + 4 and pts_to_playoffs <= 3:
        factor = min(factor + 0.05, 1.30)

    return {
        "label": label,
        "icon":  icon,
        "factor": factor,
        "pts_to_playoffs":  pts_to_playoffs,
        "pts_to_leader":    pts_to_leader,
        "pts_above_danger": pts_above_danger,
        "eliminated":       already_eliminated,
    }


def build_ligamx_table_context(table_rows: list[dict]) -> dict[str, dict]:
    """
    Recibe lista de {team, pos, pts} ordenada y devuelve contexto por equipo.
    """
    if not table_rows:
        return {}
    pts_leader = table_rows[0]["pts"]
    pts_8th    = table_rows[min(7, len(table_rows)-1)]["pts"]
    pts_16th   = table_rows[min(15, len(table_rows)-1)]["pts"] if len(table_rows) > 15 else 0

    ctx = {}
    for row in table_rows:
        team = row["team"]
        ctx[team] = ligamx_situation(
            pos=row["pos"], pts=row["pts"],
            pts_leader=pts_leader,
            pts_8th=pts_8th,
            pts_16th=pts_16th,
        )
    return ctx


def apply_motivation(lh: float, la: float,
                     factor_h: float, factor_a: float) -> tuple[float, float]:
    """
    Ajusta λ según motivación relativa de cada equipo.
    Un equipo con factor 1.20 vs uno con 1.00 incrementa su λ un 10%.
    """
    avg = (factor_h + factor_a) / 2
    if avg < 1e-9:
        return lh, la
    lh_adj = lh * (factor_h / avg)
    la_adj = la * (factor_a / avg)
    return lh_adj, la_adj


# ──────────────────────────────────────────────────────────────
# 4. GOLEADORES TOP (API-Football)
# ──────────────────────────────────────────────────────────────
APIFOOTBALL_HDR  = {"x-apisports-key": APIFOOTBALL_KEY}
LEAGUE_IDS       = APIFOOTBALL_IDS
LAST_AVAILABLE   = {"ligamx": 2024, "ucl": 2024}

# ─── ESPN Core API (gratuita, sin key) ────────────────────────────────────────
# ESPN Core API: sports.core.api.espn.com — requiere resolver $ref por atleta/equipo
# Liga MX Clausura 2026 = season 2025, type 1 (Apertura 2025 = type 2)
ESPN_CORE_LEADERS_URL = (
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/mex.1"
    "/seasons/2025/types/1/leaders"
)

# Slugs ESPN por liga
ESPN_SLUGS = {
    "ligamx": "mex.1",
    "ucl":    "uefa.champions",
}

# ─── Goleadores Clausura 2026 — fallback hardcoded (J17, ESPN Core API) ──────
# IMPORTANTE: Este diccionario es el FALLBACK cuando la API de ESPN falla.
# Fuente: ESPN Core API sports.core.api.espn.com — Clausura 2026 (season=2025, type=1)
# Errores corregidos vs. versiones anteriores:
#   Djurdjevic: Monterrey → Atlas | Sepúlveda: Chivas → Cruz Azul
#   Diber Cambindo: León → Necaxa | +5 jugadores (Ruvalcaba, Angulo, Zendejas, Ocampos, Díaz)
CLAUSURA_2026_SCORERS: list[dict] = [
    # ── 12 goles ──
    {"player": "Joao Pedro",             "team": "Atletico San Luis", "goals": 12, "assists": 0},
    {"player": "Armando Gonzalez",        "team": "Chivas",            "goals": 12, "assists": 0},
    {"player": "Paulinho",                "team": "Toluca",            "goals": 12, "assists": 0},
    # ── 9 goles ──
    {"player": "German Berterame",        "team": "Monterrey",         "goals":  9, "assists": 0},
    {"player": "Sergio Canales",          "team": "Monterrey",         "goals":  9, "assists": 0},
    # ── 8 goles ──
    {"player": "Angel Correa",            "team": "Tigres",            "goals":  8, "assists": 4},
    {"player": "Juan Brunetta",           "team": "Tigres",            "goals":  8, "assists": 0},
    {"player": "Oscar Estupinan",         "team": "Juarez",            "goals":  8, "assists": 0},
    # ── 7 goles ──
    {"player": "Brian Rodriguez",         "team": "America",           "goals":  7, "assists": 4},
    {"player": "Uros Djurdjevic",         "team": "Atlas",             "goals":  7, "assists": 0},
    {"player": "Angel Sepulveda",         "team": "Cruz Azul",         "goals":  7, "assists": 0},
    {"player": "Gabriel Fernandez",       "team": "Cruz Azul",         "goals":  7, "assists": 0},
    # ── 6 goles ──
    {"player": "Emiliano Gomez",          "team": "Puebla",            "goals":  6, "assists": 0},
    {"player": "Frank Thierry Boya",      "team": "Tijuana",           "goals":  6, "assists": 0},
    {"player": "Diber Cambindo",          "team": "Necaxa",            "goals":  6, "assists": 0},
    {"player": "Ali Avila",               "team": "Queretaro",         "goals":  6, "assists": 0},
    # ── 5 goles ──
    {"player": "Diego Gonzalez",          "team": "Atlas",             "goals":  5, "assists": 5},
    {"player": "Kevin Castaneda",         "team": "Tijuana",           "goals":  5, "assists": 4},
    {"player": "Jorge Ruvalcaba",         "team": "Pumas",             "goals":  5, "assists": 0},
    {"player": "Jesus Angulo",            "team": "Toluca",            "goals":  5, "assists": 0},
    {"player": "Alejandro Zendejas",      "team": "America",           "goals":  5, "assists": 0},
    {"player": "Lucas Ocampos",           "team": "Monterrey",         "goals":  5, "assists": 6},
    {"player": "Ismael Diaz",             "team": "Leon",              "goals":  5, "assists": 0},
]

# ─── Asistencias Clausura 2026 — ESPN J17 ────────────────────────────────────
CLAUSURA_2026_ASSISTS: list[dict] = [
    {"player": "Alexis Vega",             "team": "Toluca",            "assists": 9},
    {"player": "Nicolas Castro",          "team": "Toluca",            "assists": 7},
    {"player": "Lucas Ocampos",           "team": "Monterrey",         "assists": 6},
    {"player": "Diego Gonzalez",          "team": "Atlas",             "assists": 5},
    {"player": "Jose Abella",             "team": "Santos Laguna",     "assists": 5},
    {"player": "Jose Paradela",           "team": "Cruz Azul",         "assists": 5},
    {"player": "Ramiro Arciga",           "team": "Tijuana",           "assists": 5},
    {"player": "Richard Ledezma",         "team": "Chivas",            "assists": 5},
    {"player": "Angel Correa",            "team": "Tigres",            "assists": 4},
    {"player": "Kevin Castaneda",         "team": "Tijuana",           "assists": 4},
    {"player": "Adalberto Carrasquilla",  "team": "Pumas",             "assists": 4},
    {"player": "Carlos Rodriguez",        "team": "Cruz Azul",         "assists": 4},
    {"player": "Efrain Alvarez",          "team": "Chivas",            "assists": 4},
    {"player": "Brian Rodriguez",         "team": "America",           "assists": 4},
    {"player": "Victor Guzman",           "team": "Pachuca",           "assists": 4},
    {"player": "Juan Manuel Sanabria",    "team": "Atletico San Luis", "assists": 4},
    {"player": "Jesus Vega",              "team": "Tijuana",           "assists": 4},
]


# ─── ESPN Core API auto-fetch ─────────────────────────────────────────────────

def _resolve_espn_ref(url: str) -> dict:
    """Resolve an ESPN Core API $ref URL. Returns {} on any error."""
    import urllib.request as _req
    try:
        url = url.replace("http://", "https://")
        with _req.urlopen(url, timeout=5) as r:
            return json.load(r)
    except Exception:
        return {}


def _fetch_espn_core_leaders(category: str = "goals") -> list[dict]:
    """
    Fetches Liga MX leaders from ESPN Core API.
    Resolves athlete/team $ref references to get full names.
    Returns [{player, team, <category>, source}] or [] on failure.
    """
    try:
        r = requests.get(ESPN_CORE_LEADERS_URL, timeout=12,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    categories = data.get("categories", [])
    cat = next((c for c in categories if c.get("name") == category), None)
    if not cat:
        return []

    result: list[dict] = []
    team_cache: dict[str, str] = {}

    for entry in cat.get("leaders", [])[:25]:
        try:
            ath_ref  = entry.get("athlete", {}).get("$ref", "")
            team_ref = entry.get("team", {}).get("$ref", "")
            value    = int(float(entry.get("value", 0)))

            ath_data = _resolve_espn_ref(ath_ref)
            name     = ath_data.get("fullName", "?")

            if team_ref not in team_cache:
                team_data = _resolve_espn_ref(team_ref)
                raw_name  = team_data.get("displayName", "?")
                # Normalize to canonical name used in TEAM_ALIASES values
                canon = TEAM_ALIASES.get(
                    _strip_accents(raw_name.lower()), raw_name
                )
                team_cache[team_ref] = canon
            team_name = team_cache[team_ref]

            result.append({
                "player": name,
                "team":   team_name,
                category: value,
                "source": "espn_core_api",
            })
        except Exception:
            continue

    return result


def fetch_espn_ligamx_scorers(ttl_hours: int = 6) -> list[dict]:
    """
    Descarga tabla de goleadores Liga MX desde ESPN Core API (gratuita, sin key).
    Caché local en data/_espn_cache/ligamx_scorers.json (TTL configurable).
    Retorna [] si falla (llamador usa CLAUSURA_2026_SCORERS como fallback).
    """
    import time as _time
    from pathlib import Path as _Path

    cache_dir  = _Path("data/_espn_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ligamx_scorers.json"

    # ── Leer caché si es vigente ──
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_bytes().decode("utf-8"))
            age_h  = (_time.time() - cached.get("fetched_at", 0)) / 3600
            if age_h < ttl_hours and cached.get("scorers"):
                return cached["scorers"]
        except Exception:
            pass

    # ── Llamar ESPN Core API ──
    scorers = _fetch_espn_core_leaders("goals")
    if not scorers:
        return []

    # Añadir campo "goals" como alias para compatibilidad
    cache_file.write_bytes(
        json.dumps({"fetched_at": _time.time(), "scorers": scorers},
                   ensure_ascii=False).encode("utf-8")
    )
    print(f"  [ESPN Core] Goleadores actualizados: {len(scorers)} jugadores")
    return scorers


def fetch_espn_ligamx_assists(ttl_hours: int = 6) -> list[dict]:
    """
    Descarga tabla de asistencias Liga MX desde ESPN Core API.
    Retorna [] si falla (llamador usa CLAUSURA_2026_ASSISTS como fallback).
    """
    import time as _time
    from pathlib import Path as _Path

    cache_dir  = _Path("data/_espn_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ligamx_assists.json"

    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_bytes().decode("utf-8"))
            age_h  = (_time.time() - cached.get("fetched_at", 0)) / 3600
            if age_h < ttl_hours and cached.get("assists"):
                return cached["assists"]
        except Exception:
            pass

    assists = _fetch_espn_core_leaders("assists")
    if not assists:
        return []

    cache_file.write_bytes(
        json.dumps({"fetched_at": _time.time(), "assists": assists},
                   ensure_ascii=False).encode("utf-8")
    )
    return assists


@functools.lru_cache(maxsize=4)
def fetch_top_scorers(league: str) -> list[dict]:
    """
    Goleadores Liga MX:
      1. Intenta ESPN API (gratuita, datos en tiempo real, sin key)
      2. Fallback: CLAUSURA_2026_SCORERS hardcoded (actualizado J17)

    UCL: intenta API-Football season 2024.
    """
    if league == "ligamx":
        live = fetch_espn_ligamx_scorers(ttl_hours=6)
        if live:
            return live
        # Fallback hardcoded
        print("  [scorers] ESPN API no disponible — usando datos hardcoded (J17)")
        return CLAUSURA_2026_SCORERS

    # UCL / otras ligas → API-Football
    lid = LEAGUE_IDS.get(league)
    ssn = LAST_AVAILABLE.get(league)
    if lid is None or ssn is None:
        return []
    try:
        r = requests.get(
            f"{APIFOOTBALL_BASE}/players/topscorers",
            headers=APIFOOTBALL_HDR,
            params={"league": lid, "season": ssn},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        data = r.json().get("response", [])
        result = []
        for item in data[:20]:
            p    = item["player"]
            stat = item["statistics"][0]
            result.append({
                "player":  p["name"],
                "team":    stat["team"]["name"],
                "goals":   stat["goals"].get("total") or 0,
                "assists": stat["goals"].get("assists") or 0,
            })
        return result
    except Exception:
        return []


def get_assists_for_team(team_name: str, top_n: int = 3) -> list[dict]:
    """
    Retorna los top asistidores de un equipo en el torneo actual.
    Intenta ESPN API primero, fallback a CLAUSURA_2026_ASSISTS.
    """
    live = fetch_espn_ligamx_assists(ttl_hours=6)
    source = live if live else CLAUSURA_2026_ASSISTS
    key = _strip_accents(team_name.lower().strip())
    resolved = TEAM_ALIASES.get(key)
    if resolved is None:
        return []
    return [a for a in source
            if a.get("team", "").lower() == resolved.lower()][:top_n]


# Aliases: nombre Sofascore → equipo en CLAUSURA_2026_SCORERS (campo "team")
TEAM_ALIASES: dict[str, str] = {
    # Chivas / Guadalajara
    "cd guadalajara":    "Chivas",
    "guadalajara":       "Chivas",
    "chivas":            "Chivas",
    # América
    "club america":      "America",
    "america":           "America",
    # Monterrey
    "cf monterrey":      "Monterrey",
    "monterrey":         "Monterrey",
    # Tigres
    "tigres uanl":       "Tigres",
    "tigres":            "Tigres",
    # Pumas
    "pumas unam":        "Pumas",
    "pumas":             "Pumas",
    # Pachuca
    "cf pachuca":        "Pachuca",
    "pachuca":           "Pachuca",
    # Cruz Azul
    "cruz azul":         "Cruz Azul",
    # Toluca
    "cd toluca":         "Toluca",
    "toluca":            "Toluca",
    # León
    "club leon":         "Leon",
    "leon":              "Leon",
    # Atlético San Luis
    "atletico san luis": "Atletico San Luis",
    "atletico de san luis": "Atletico San Luis",
    # Atlas
    "atlas fc":          "Atlas",
    "atlas":             "Atlas",
    # Mazatlán
    "mazatlan fc":       "Mazatlan",
    "mazatlan":          "Mazatlan",
    # Necaxa
    "club necaxa":       "Necaxa",
    "necaxa":            "Necaxa",
    # Tijuana
    "club tijuana":      "Tijuana",
    "tijuana":           "Tijuana",
    # Puebla
    "club puebla":       "Puebla",
    "puebla":            "Puebla",
    # FC Juárez
    "fc juarez":         "Juarez",
    "juarez":            "Juarez",
    # Santos
    "santos laguna":     "Santos Laguna",
    "santos":            "Santos Laguna",
    # Querétaro
    "queretaro fc":      "Queretaro",
    "queretaro":         "Queretaro",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    )


def scorers_for_team(team_name: str, league: str, top_n: int = 3) -> list[dict]:
    """Retorna los top_n goleadores de un equipo (alias → match exacto → fuzzy)."""
    from difflib import SequenceMatcher
    all_scorers = fetch_top_scorers(league)
    if not all_scorers:
        return []

    # Normalizar: minúsculas + sin acentos para buscar en alias
    key = _strip_accents(team_name.lower().strip())
    resolved = TEAM_ALIASES.get(key, None)

    # Si no hay alias definido, el equipo no es de los top-scorers conocidos
    if resolved is None:
        return []

    # Match exacto (case-insensitive) contra el campo "team" de los scorers
    exact = [s for s in all_scorers if s["team"].lower() == resolved.lower()]
    if exact:
        return exact[:top_n]

    # Fuzzy como último recurso solo con umbral muy alto (≥ 0.80) para evitar falsos positivos
    def sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    teams_in_data = list({s["team"] for s in all_scorers})
    best_team = max(teams_in_data, key=lambda t: sim(resolved, t), default="")
    if sim(resolved, best_team) >= 0.80:
        return [s for s in all_scorers if s["team"] == best_team][:top_n]

    return []


# ──────────────────────────────────────────────────────────────
# 5. ESTIMACIÓN DE TIROS DE ESQUINA
# ──────────────────────────────────────────────────────────────
# Calibración basada en estudios de fútbol:
# Liga MX promedio: ~10 corners/partido (5 cada equipo)
# Relación con λ: más ataque → más corners
CORNERS_PER_LAMBDA = 3.8   # corners por unidad de λ (ajustado para Liga MX)
CORNERS_MIN = 2.0

def estimate_corners(lh: float, la: float) -> dict:
    """
    Estima tiros de esquina para cada equipo basándose en sus λ de goles.
    Retorna rangos inferior, central y superior.
    """
    c_h = max(CORNERS_MIN, lh * CORNERS_PER_LAMBDA)
    c_a = max(CORNERS_MIN, la * CORNERS_PER_LAMBDA)

    def rng(c: float) -> str:
        lo = max(1, round(c - 1.5))
        hi = round(c + 1.5)
        return f"{lo}–{hi}"

    return {
        "corners_h_est": round(c_h, 1),
        "corners_a_est": round(c_a, 1),
        "corners_h_range": rng(c_h),
        "corners_a_range": rng(c_a),
        "total_est":  round(c_h + c_a, 1),
        "total_range": f"{max(2,round(c_h+c_a-3))}–{round(c_h+c_a+3)}",
        "note": "Estimado a partir de λ (sin datos históricos de corners)",
    }


# ──────────────────────────────────────────────────────────────
# 6. ANÁLISIS SITUACIONAL (texto para dashboard)
# ──────────────────────────────────────────────────────────────
def match_narrative(home: str, away: str,
                    ctx_h: Optional[dict], ctx_a: Optional[dict],
                    ph: float, pd: float, pv: float,
                    lh: float, la: float,
                    h2h_h: int = 0, h2h_d: int = 0, h2h_a: int = 0) -> dict:
    """
    Genera un resumen narrativo del partido con factores clave.
    """
    lines = []

    # Narrativa de motivación
    if ctx_h and ctx_a:
        if ctx_h["factor"] > ctx_a["factor"] + 0.05:
            lines.append(f"🔥 {home} llega con mayor urgencia ({ctx_h['label']})")
        elif ctx_a["factor"] > ctx_h["factor"] + 0.05:
            lines.append(f"🔥 {away} llega con mayor urgencia ({ctx_a['label']})")
        else:
            lines.append("⚖️ Motivaciones similares en ambos equipos")

    # Equilibrio de fuerzas
    diff = abs(lh - la)
    if diff < 0.15:
        lines.append("⚖️ Partido muy equilibrado — empate como resultado razonable")
    elif lh > la:
        lines.append(f"⚔️ {home} proyecta más poder ofensivo (λ {lh:.2f} vs {la:.2f})")
    else:
        lines.append(f"⚔️ {away} proyecta más poder ofensivo (λ {la:.2f} vs {lh:.2f})")

    # H2H
    total_h2h = h2h_h + h2h_d + h2h_a
    if total_h2h >= 3:
        dom = ""
        if h2h_h > h2h_a + 1:
            dom = f"{home} domina el historial ({h2h_h}V-{h2h_d}E-{h2h_a}D)"
        elif h2h_a > h2h_h + 1:
            dom = f"{away} domina el historial ({h2h_a}V-{h2h_d}E-{h2h_h}D)"
        else:
            dom = f"Historial igualado ({h2h_h}-{h2h_d}-{h2h_a})"
        lines.append(f"📊 {dom}")

    # Tendencia al empate
    if pd > 0.28:
        lines.append(f"🤝 Alta probabilidad de empate ({pd*100:.1f}%) — considerar X")

    return {
        "narrative": lines,
        "draw_alert": pd > 0.28,
        "diff_lambdas": round(lh - la, 2),
    }
