"""
=====================================================================
  odds_client.py — The Odds API (v4)
  Key: cargada desde utils/config.py (.env)

  Uso como módulo:
    from odds_client import fetch_odds, find_match, format_edge

  Uso standalone:
    python odds_client.py [--sport UCL] [--days 7]

  Provee:
    - Odds en tiempo real de Pinnacle / Betfair / mercado medio
    - Probabilidades sin vig (no-vig)
    - Cálculo de edge vs modelo
=====================================================================
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import requests

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

from utils.config import ODDS_API_KEY, ODDS_API_BASE as ODDS_BASE  # noqa: E402
CACHE_DIR    = Path("data/_odds_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo liga → sport_key de The Odds API
SPORT_KEYS = {
    "UCL":        "soccer_uefa_champs_league",
    "Europa":     "soccer_uefa_europa_league",
    "EPL":        "soccer_epl",
    "LaLiga":     "soccer_spain_la_liga",
    "Bundesliga": "soccer_germany_bundesliga",
    "SerieA":     "soccer_italy_serie_a",
    "Ligue1":     "soccer_france_ligue_one",
    "LigaMX":     "soccer_mexico_ligamx",
}

# Bookmakers preferidos (cascade: primero disponible gana)
PREFERRED_BOOKMAKERS = [
    "pinnacle",          # el más eficiente — mejor referencia
    "betfair_ex_eu",     # exchange (sin margen propio)
    "matchbook",         # exchange
    "marathonbet",       # alta eficiencia
    "nordicbet",
    "betsson",
]

# TTL del cache en minutos (30 min = balance entre frescura y requests)
CACHE_TTL = 30

# ─────────────────────────────────────────────
# CLIENTE HTTP CON CACHÉ
# ─────────────────────────────────────────────

def _get(url: str, params: dict, cache_key: str | None = None,
         ttl_minutes: int = CACHE_TTL) -> dict | list:
    """GET con caché de archivo. ttl_minutes=0 fuerza descarga."""
    if cache_key:
        p = CACHE_DIR / cache_key
        if p.exists() and ttl_minutes > 0:
            age_min = (datetime.now().timestamp() - p.stat().st_mtime) / 60
            if age_min < ttl_minutes:
                return json.loads(p.read_bytes().decode("utf-8"))

    time.sleep(0.3)
    r = requests.get(url, params=params, timeout=15)
    remaining = r.headers.get("x-requests-remaining", "?")
    used      = r.headers.get("x-requests-used", "?")

    if r.status_code == 401:
        raise RuntimeError("Odds API: clave inválida o sin cuota.")
    if r.status_code == 429:
        raise RuntimeError("Odds API: rate limit excedido.")
    r.raise_for_status()

    data = r.json()
    if cache_key and data:
        (CACHE_DIR / cache_key).write_bytes(
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        )

    print(f"  [Odds API] requests usados={used} | restantes={remaining}")
    return data


# ─────────────────────────────────────────────
# NORMALIZACIÓN DE NOMBRES DE EQUIPOS
# ─────────────────────────────────────────────

_TEAM_ALIASES = {
    "atletico madrid":              "atletico madrid",
    "atletico de madrid":           "atletico madrid",
    "club atletico de madrid":      "atletico madrid",
    "atl. madrid":                  "atletico madrid",
    "atlético madrid":              "atletico madrid",
    "inter milan":                  "inter",
    "internazionale":               "inter",
    "paris saint-germain":          "psg",
    "paris saint germain":          "psg",
    "paris saint-germain fc":       "psg",
    "bayer 04 leverkusen":          "leverkusen",
    "bayer leverkusen":             "leverkusen",
    "tottenham hotspur":            "tottenham",
    "tottenham hotspur fc":         "tottenham",
    "manchester city fc":           "manchester city",
    "newcastle united":             "newcastle",
    "newcastle united fc":          "newcastle",
    "sporting clube de portugal":   "sporting cp",
    "sporting cp":                  "sporting cp",
    "fk bodo/glimt":               "bodo/glimt",
    "bodo/glimt":                   "bodo/glimt",
    "galatasaray sk":               "galatasaray",
    "atalanta bc":                  "atalanta",
    "fc barcelona":                 "barcelona",
    "real madrid cf":               "real madrid",
    "fc bayern munchen":            "bayern munich",
    "fc bayern münchen":            "bayern munich",
    "bayern munich":                "bayern munich",
    "liverpool fc":                 "liverpool",
    "arsenal fc":                   "arsenal",
    "chelsea fc":                   "chelsea",
}


def _norm(name: str) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd)\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return _TEAM_ALIASES.get(name, name)


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# ─────────────────────────────────────────────
# CÁLCULO DE PROBABILIDADES SIN VIG
# ─────────────────────────────────────────────

def no_vig(o_h: float, o_d: float, o_a: float) -> tuple:
    """
    Convierte odds decimales 3-way a probabilidades sin vig.
    Retorna (p_home, p_draw, p_away) o (None, None, None) si datos inválidos.
    """
    try:
        if not all(o > 1.0 for o in [o_h, o_d, o_a]):
            return None, None, None
        p_h, p_d, p_a = 1 / o_h, 1 / o_d, 1 / o_a
        total = p_h + p_d + p_a
        return p_h / total, p_d / total, p_a / total
    except (TypeError, ZeroDivisionError):
        return None, None, None


def overround_pct(o_h: float, o_d: float, o_a: float) -> float:
    """Margen de la casa en %."""
    try:
        return round((1 / o_h + 1 / o_d + 1 / o_a - 1) * 100, 2)
    except (TypeError, ZeroDivisionError):
        return 0.0


# ─────────────────────────────────────────────
# FETCH PRINCIPAL
# ─────────────────────────────────────────────

def fetch_odds(sport: str = "UCL", ttl_minutes: int = CACHE_TTL) -> list:
    """
    Descarga odds para todos los partidos próximos del deporte indicado.

    Retorna lista de dicts:
      {
        "home": str, "away": str, "date": str,
        "preferred":   {"bookmaker", "o_h", "o_d", "o_a", "p_h", "p_d", "p_a", "overround_pct"},
        "avg_market":  {"p_h", "p_d", "p_a"},
        "n_bookmakers": int,
        "bookmakers":   { bk_key: {...} }
      }
    """
    sport_key = SPORT_KEYS.get(sport, sport)
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "eu,uk",
        "markets":    "h2h",
        "oddsFormat": "decimal",
    }
    cache_key = f"odds_{sport_key}.json"
    data = _get(f"{ODDS_BASE}/sports/{sport_key}/odds/", params, cache_key, ttl_minutes)
    if not isinstance(data, list):
        return []

    results = []
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        date = game.get("commence_time", "")[:10]

        bks = {}
        for bk in game.get("bookmakers", []):
            mkt = next((m for m in bk.get("markets", []) if m["key"] == "h2h"), None)
            if not mkt:
                continue
            outcomes = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
            o_h = outcomes.get(home)
            o_d = outcomes.get("Draw")
            o_a = outcomes.get(away)
            if not all([o_h, o_d, o_a]):
                continue
            p_h, p_d, p_a = no_vig(o_h, o_d, o_a)
            if p_h is None:
                continue
            bks[bk["key"]] = {
                "title":         bk["title"],
                "o_h":           round(o_h, 3),
                "o_d":           round(o_d, 3),
                "o_a":           round(o_a, 3),
                "p_h":           round(p_h * 100, 1),
                "p_d":           round(p_d * 100, 1),
                "p_a":           round(p_a * 100, 1),
                "overround_pct": overround_pct(o_h, o_d, o_a),
            }

        # Cascade de bookmakers preferidos
        # Se omiten bookmakers con vig > 20% (indicativo de odds en-vivo/inválidas)
        MAX_VIG = 20.0
        preferred = None
        for bk_key in PREFERRED_BOOKMAKERS:
            if bk_key in bks and bks[bk_key]["overround_pct"] <= MAX_VIG:
                preferred = {**bks[bk_key], "bookmaker_key": bk_key}
                break

        # Si ningún preferido tiene vig razonable, tomar el de menor vig entre todos
        if preferred is None and bks:
            valid = {k: v for k, v in bks.items() if v["overround_pct"] <= MAX_VIG}
            if valid:
                best_bk = min(valid, key=lambda k: valid[k]["overround_pct"])
                preferred = {**valid[best_bk], "bookmaker_key": best_bk}

        # Promedio del mercado — solo bookmakers con vig razonable
        avg_market = None
        valid_bks = {k: v for k, v in bks.items() if v["overround_pct"] <= MAX_VIG}
        if valid_bks:
            bks_for_avg = valid_bks
        else:
            bks_for_avg = bks
        if bks_for_avg:
            avg_ph = sum(b["p_h"] for b in bks_for_avg.values()) / len(bks_for_avg)
            avg_pd = sum(b["p_d"] for b in bks_for_avg.values()) / len(bks_for_avg)
            avg_pa = sum(b["p_a"] for b in bks_for_avg.values()) / len(bks_for_avg)
            avg_market = {
                "p_h": round(avg_ph, 1),
                "p_d": round(avg_pd, 1),
                "p_a": round(avg_pa, 1),
            }

        results.append({
            "home":         home,
            "away":         away,
            "date":         date,
            "preferred":    preferred,
            "avg_market":   avg_market,
            "n_bookmakers": len(valid_bks) if valid_bks else len(bks),
            "bookmakers":   bks,
        })

    return results


# ─────────────────────────────────────────────
# MATCHING DE PARTIDOS
# ─────────────────────────────────────────────

def find_match(home: str, away: str, odds_list: list,
               threshold: float = 0.60) -> dict | None:
    """
    Busca en odds_list el partido que mejor coincide con (home, away).
    Usa fuzzy matching. Retorna None si no hay coincidencia suficiente.
    """
    best, best_score = None, threshold
    for item in odds_list:
        sh = _similarity(home, item["home"])
        sa = _similarity(away, item["away"])
        score = (sh + sa) / 2
        if score > best_score:
            best_score, best = score, item
    return best


# ─────────────────────────────────────────────
# CÁLCULO DE EDGE
# ─────────────────────────────────────────────

def calc_edge(our_prob_pct: float, market_prob_pct: float) -> float:
    """Edge en puntos porcentuales. Positivo = value, negativo = sin value."""
    return round(our_prob_pct - market_prob_pct, 1)


def ev_pct(our_prob: float, decimal_odds: float) -> float:
    """
    Expected Value en %. EV > 0 = apuesta con valor.
    EV = (our_prob × (odds - 1) - (1 - our_prob)) × 100
    """
    return round((our_prob * (decimal_odds - 1) - (1 - our_prob)) * 100, 1)


def format_edge_label(edge: float) -> str:
    """Etiqueta de texto para el edge."""
    if edge >= 8:
        return f"[+{edge:.1f}% EDGE ALTO]"
    elif edge >= 3:
        return f"[+{edge:.1f}% edge]"
    elif edge >= -3:
        return f"[~{edge:+.1f}% neutro]"
    else:
        return f"[{edge:.1f}% sin edge]"


# ─────────────────────────────────────────────
# MODO STANDALONE
# ─────────────────────────────────────────────

def _print_odds_table(sport: str = "UCL", ttl: int = 30):
    print(f"\n{'='*70}")
    print(f"  THE ODDS API — {sport} | Bookmaker preferido: Pinnacle/Betfair")
    print(f"{'='*70}")

    try:
        odds = fetch_odds(sport, ttl)
    except Exception as e:
        print(f"  ERROR: {e}")
        return

    if not odds:
        print("  Sin partidos disponibles.")
        return

    for item in odds:
        home = item["home"]
        away = item["away"]
        date = item["date"]
        pref = item.get("preferred")
        avg  = item.get("avg_market")
        n_bk = item.get("n_bookmakers", 0)

        print(f"\n  {date}  {home} vs {away}  [{n_bk} bookmakers]")
        if pref:
            bk   = pref.get("bookmaker_key", "?")
            or_  = pref.get("overround_pct", 0)
            print(f"  {bk.upper():<12} "
                  f"Local: {pref['o_h']:.2f} ({pref['p_h']:.1f}%)  "
                  f"Empate: {pref['o_d']:.2f} ({pref['p_d']:.1f}%)  "
                  f"Visitante: {pref['o_a']:.2f} ({pref['p_a']:.1f}%)  "
                  f"[vig={or_:.1f}%]")
        if avg:
            print(f"  MERCADO MEDIO  "
                  f"Local: {avg['p_h']:.1f}%  "
                  f"Empate: {avg['p_d']:.1f}%  "
                  f"Visitante: {avg['p_a']:.1f}%")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="The Odds API — consulta de momios")
    ap.add_argument("--sport", default="UCL",
                    choices=list(SPORT_KEYS.keys()),
                    help="Liga/competición (default: UCL)")
    ap.add_argument("--ttl", type=int, default=30,
                    help="TTL cache en minutos (0 = forzar descarga, default: 30)")
    args = ap.parse_args()
    _print_odds_table(args.sport, args.ttl)
