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
import math
import requests
import functools
from typing import Optional

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
                     pts_8th: int, pts_16th: int) -> dict:
    """
    Retorna situación y factor de motivación para un equipo de Liga MX.
    pos         : posición actual (1=líder)
    pts         : puntos actuales
    pts_leader  : puntos del 1er lugar
    pts_8th     : puntos del 8vo lugar (último en zona de Liguilla)
    pts_16th    : puntos del 16vo lugar (límite zona de riesgo cociente)
    """
    pts_to_playoffs  = max(0, pts_8th  - pts)
    pts_to_leader    = max(0, pts_leader - pts)
    pts_above_danger = pts - pts_16th    # negativo = en zona de riesgo

    if pos <= LIGAMX_DIRECT_QF:
        label = "Líder / Cuartos directos"
        icon  = "🏆"
        # Motivación alta pero más relajada que quien pelea por entrar
        factor = 1.08
    elif pos <= LIGAMX_PLAYOFF:
        label = "Zona Liguilla"
        icon  = "⚽"
        factor = 1.12
    elif pos <= LIGAMX_PLAYOFF + 3:
        label = "Persiguiendo Liguilla"
        icon  = "🔥"
        factor = 1.18   # Alta motivación — muy cerca del playoff
    elif pos <= LIGAMX_TOTAL - LIGAMX_RELZONE:
        label = "Zona Media / Segura"
        icon  = "😐"
        factor = 1.00   # Neutral
    elif pos <= LIGAMX_TOTAL - 1:
        label = "Zona de Riesgo (cociente)"
        icon  = "⚠️"
        factor = 1.20
    else:
        label = "Último lugar"
        icon  = "🆘"
        factor = 1.25

    # Extra boost si la diferencia con el 8vo es ≤ 3 puntos
    if LIGAMX_PLAYOFF < pos <= LIGAMX_PLAYOFF + 4 and pts_to_playoffs <= 3:
        factor = min(factor + 0.05, 1.30)

    return {
        "label": label,
        "icon":  icon,
        "factor": factor,
        "pts_to_playoffs": pts_to_playoffs,
        "pts_to_leader":   pts_to_leader,
        "pts_above_danger": pts_above_danger,
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
APIFOOTBALL_KEY  = "5cf3eb50762eeb4e9cf15173bae1cb65"
APIFOOTBALL_BASE = "https://v3.football.api-sports.io"
APIFOOTBALL_HDR  = {"x-apisports-key": APIFOOTBALL_KEY}

# Liga MX = 262 | UCL = 2
LEAGUE_IDS = {"ligamx": 262, "ucl": 2}
# Última temporada con datos completos disponibles en plan free
LAST_AVAILABLE = {"ligamx": 2024, "ucl": 2024}


@functools.lru_cache(maxsize=4)
def fetch_top_scorers(league: str) -> list[dict]:
    """
    Descarga los top-20 goleadores de la última temporada disponible.
    Retorna lista de {player, team, goals}.
    Cachea en memoria para no repetir llamadas.
    """
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
                "player": p["name"],
                "team":   stat["team"]["name"],
                "goals":  stat["goals"].get("total") or 0,
                "season": ssn,
            })
        return result
    except Exception:
        return []


# Aliases: nombre Sofascore → nombre API-Football (cuando difieren mucho)
TEAM_ALIASES: dict[str, str] = {
    "pumas unam":        "u.n.a.m. - pumas",
    "cd guadalajara":    "guadalajara chivas",
    "guadalajara":       "guadalajara chivas",
    "chivas":            "guadalajara chivas",
    "cf monterrey":      "monterrey",
    "cf pachuca":        "pachuca",
    "cd toluca":         "toluca",
    "atletico san luis": "atletico san luis",
    "atlas fc":          "atlas",
    "mazatlan fc":       "mazatlan",
    "club america":      "club america",
    "club necaxa":       "necaxa",
    "club tijuana":      "club tijuana",
    "club puebla":       "puebla",
    "club leon":         "leon",
    "fc juarez":         "juarez",
    "santos laguna":     "santos laguna",
    "queretaro fc":      "queretaro",
    "cruz azul":         "cruz azul",
    "tigres uanl":       "tigres uanl",
}


def scorers_for_team(team_name: str, league: str, top_n: int = 3) -> list[dict]:
    """Retorna los top_n goleadores de un equipo (alias + fuzzy match)."""
    from difflib import SequenceMatcher
    all_scorers = fetch_top_scorers(league)
    if not all_scorers:
        return []

    # Resolver alias primero
    key = team_name.lower().strip()
    resolved = TEAM_ALIASES.get(key, key)

    def sim(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    teams_in_data = list({s["team"] for s in all_scorers})
    best_team = max(teams_in_data, key=lambda t: sim(resolved, t), default="")
    if sim(resolved, best_team) < 0.40:
        return []

    return [s for s in all_scorers if s["team"] == best_team][:top_n]


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
