"""
=====================================================================
  Soccer Data Collector — Plantilla Principal
  Fuente: soccerdata (FBref, Understat, Football-Data.co.uk, ClubElo)
  Uso: personal / análisis con IA y ML
=====================================================================
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path

import requests
import soccerdata as sd
import pandas as pd
import cloudscraper

# ─────────────────────────────────────────────
# SESIÓN CLOUDSCRAPER + PROXY ROTATORIO
# Reemplaza la sesión interna de soccerdata para peticiones a FBref.
# cloudscraper bypasea protecciones Cloudflare/anti-bot que causan 403.
# ─────────────────────────────────────────────

_FBREF_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "sec-ch-ua": '"Not)A;Brand";v="8", "Chromium";v="138", "Google Chrome";v="138"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

# APIs públicas para obtener lista de proxies HTTP gratuitos
_PROXY_APIS = [
    # Texto plano, un proxy ip:port por línea
    "https://api.proxyscrape.com/v2/?request=displayproxies"
    "&protocol=http&timeout=5000&country=all&ssl=all&anonymity=all",
    # JSON con campo "data": [{ip, port}, ...]
    "https://proxylist.geonode.com/api/proxy-list"
    "?limit=100&page=1&sort_by=lastChecked&sort_type=desc"
    "&protocols=http,https&speed=fast",
]


class _RotatingCloudscraperSession:
    """
    Sesión compatible con la interfaz interna de soccerdata:
        _session.get(url)    → requests.Response
        _session.headers     → dict-like

    Usa cloudscraper para emular Chrome y superar protecciones anti-bot,
    y rota proxies HTTP gratuitos en cada petición para reducir bloqueos.
    Si todos los proxies fallan cae back a petición directa sin proxy.
    """

    def __init__(self):
        self._cs = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False},
            delay=5,
        )
        self._cs.headers.update(_FBREF_BROWSER_HEADERS)
        self.headers = self._cs.headers   # expuesto para que soccerdata pueda leerlo
        self._proxies = self._fetch_proxies()
        self._idx = 0
        _log = logging.getLogger(__name__)
        _log.info("Proxy pool: %d proxies cargados", len(self._proxies))

    # ── Carga proxies desde APIs públicas ─────────────────────────────────
    def _fetch_proxies(self):
        proxies = []
        for api in _PROXY_APIS:
            try:
                resp = self._cs.get(api, timeout=10)
                if resp.status_code != 200:
                    continue
                ct = resp.headers.get("content-type", "")
                if "json" in ct:
                    for p in resp.json().get("data", []):
                        ip, port = p.get("ip"), p.get("port")
                        if ip and port:
                            proxies.append(f"http://{ip}:{port}")
                else:
                    # texto plano: ip:port por línea
                    for line in resp.text.splitlines():
                        line = line.strip()
                        if line and ":" in line and not line.startswith("#"):
                            proxies.append(f"http://{line}")
            except Exception:
                pass
        return proxies

    # ── Siguiente proxy en rotación circular ──────────────────────────────
    def _next_proxy(self):
        if not self._proxies:
            return None
        proxy = self._proxies[self._idx % len(self._proxies)]
        self._idx += 1
        return proxy

    # ── Método .get() que soccerdata llama para cada URL ──────────────────
    def get(self, url, **kwargs):
        proxy = self._next_proxy()
        if proxy:
            try:
                resp = self._cs.get(
                    url,
                    proxies={"http": proxy, "https": proxy},
                    timeout=30,
                    **kwargs,
                )
                # Si el proxy devuelve 403/407/5xx, cae back a sin proxy
                if resp.status_code in (403, 407, 502, 503, 504):
                    raise ConnectionError(f"Proxy devolvió {resp.status_code}")
                return resp
            except Exception:
                pass  # intenta sin proxy
        # Fallback sin proxy
        return self._cs.get(url, timeout=30, **kwargs)


def _apply_fbref_antiblock(fbref_instance):
    """
    Reemplaza la sesión HTTP de una instancia sd.FBref por
    _RotatingCloudscraperSession (cloudscraper + proxy rotatorio) y
    configura un delay aleatorio de 5-15 s entre peticiones.

    También parchea _init_session para que los reintentos internos de
    soccerdata sigan usando la misma sesión cloudscraper.
    """
    session = _RotatingCloudscraperSession()
    fbref_instance._session      = session
    fbref_instance._init_session = lambda: _RotatingCloudscraperSession()
    fbref_instance.rate_limit    = 5   # mínimo 5 s entre peticiones
    fbref_instance.max_delay     = 10  # adicional aleatorio → total 5-15 s


# ─────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────

# Directorio de caché local (se crea automáticamente)
DATA_DIR = Path(__file__).parent / "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configurar logging para ver qué está descargando
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LIGAS DISPONIBLES EN FBREF (más populares)
# ─────────────────────────────────────────────
# "ENG-Premier League"        → Inglaterra
# "ESP-La Liga"               → España
# "GER-Bundesliga"            → Alemania
# "ITA-Serie A"               → Italia
# "FRA-Ligue 1"               → Francia
# "Big 5 European Leagues Combined" → Las 5 juntas

LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Temporadas disponibles (~2017 en adelante con estadísticas avanzadas)
# Formato: "YYYY-YY"  → "2023-24", "2022-23", "2021-22" ...
SEASONS = ["2526", "2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718"]

# ─────────────────────────────────────────────
# FOOTBALL-DATA.ORG — Configuración API
# ─────────────────────────────────────────────
FDORG_TOKEN = "1f7a12b6f18f418caa88a0f59884e80a"
FDORG_BASE  = "https://api.football-data.org/v4"
FDORG_RATE  = 6.5   # segundos entre peticiones (free tier: 10 req/min)

# Mapeo liga interna → código de competición en football-data.org
FDORG_COMPS = {
    "ENG-Premier League": "PL",
    "ESP-La Liga":        "PD",
    "GER-Bundesliga":     "BL1",
    "ITA-Serie A":        "SA",
    "FRA-Ligue 1":        "FL1",
}

# ─────────────────────────────────────────────
# SPORTSAPI (RapidAPI) — Configuración
# ─────────────────────────────────────────────
SPORTSAPI_KEY  = "c2e4c0828cmsh0e044c60fbedf45p1e0984jsn12eecacdf11d"
SPORTSAPI_HOST = "sportapi7.p.rapidapi.com"
SPORTSAPI_BASE = f"https://{SPORTSAPI_HOST}/api/v1"
SPORTSAPI_RATE = 1.5   # segundos entre peticiones (RapidAPI free tier)

# IDs de torneo único (SofaScore) — si la API devuelve 404 revisa con
# GET /api/v1/sport/1/unique-tournaments o búscalos en sofascore.com
SPORTSAPI_TOURNAMENTS = {
    "ENG-Premier League": 17,
    "ESP-La Liga":         8,
    "GER-Bundesliga":     35,
    "ITA-Serie A":        23,
    "FRA-Ligue 1":        34,
}


# ─────────────────────────────────────────────
# MÓDULO 1: FBREF — Estadísticas avanzadas
# ─────────────────────────────────────────────

def collect_fbref(leagues=LEAGUES, seasons=SEASONS):
    """
    Descarga datos de FBref para las ligas y temporadas indicadas.
    Retorna un diccionario con DataFrames listos para análisis/ML.
    """
    log.info("=== Iniciando descarga FBref ===")
    results = {}

    fbref = sd.FBref(leagues=leagues, seasons=seasons, data_dir=DATA_DIR)
    _apply_fbref_antiblock(fbref)

    # 1. Calendario / resultados de partidos
    log.info("Descargando: Schedule (calendario/resultados)...")
    results["schedule"] = fbref.read_schedule()

    # 2. Estadísticas de equipos por temporada
    stat_types = ["standard", "shooting", "passing", "defense", "possession"]
    for stat in stat_types:
        log.info(f"Descargando team_season_stats: {stat}...")
        results[f"team_season_{stat}"] = fbref.read_team_season_stats(stat_type=stat)

    # 3. Estadísticas de equipos por partido
    log.info("Descargando: Team match stats...")
    results["team_match_stats"] = fbref.read_team_match_stats()

    # 4. Estadísticas de jugadores por temporada
    player_stats = ["standard", "shooting", "passing", "defense", "possession", "misc"]
    for stat in player_stats:
        log.info(f"Descargando player_season_stats: {stat}...")
        results[f"player_season_{stat}"] = fbref.read_player_season_stats(stat_type=stat)

    log.info("=== FBref completado ✓ ===")
    return results


# ─────────────────────────────────────────────
# MÓDULO 2: UNDERSTAT — xG granular por partido
# ─────────────────────────────────────────────

# Ligas soportadas por Understat (nombres distintos a FBref)
UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

def collect_understat(leagues=UNDERSTAT_LEAGUES, seasons=SEASONS):
    log.info("=== Iniciando descarga Understat ===")
    results = {}

    understat = sd.Understat(leagues=leagues, seasons=seasons, data_dir=DATA_DIR)

    log.info("Descargando: Schedule (con xG por partido)...")
    results["schedule_xg"] = understat.read_schedule()

    log.info("Descargando: Player season stats...")
    results["player_season_stats"] = understat.read_player_season_stats()

    log.info("=== Understat completado ✓ ===")
    return results


# ─────────────────────────────────────────────
# MÓDULO 3: FOOTBALL-DATA.CO.UK — Histórico + odds
# ─────────────────────────────────────────────

# Misma convención de temporadas que el resto ("YYZZ")
# Excluye 2526 porque FDC puede no tenerla aún (temporada en curso)
FDC_SEASONS = ["2425", "2324", "2223", "2122", "2021", "1920", "1819", "1718"]

def collect_footballdata(leagues=LEAGUES, seasons=FDC_SEASONS):
    """
    Descarga datos históricos de Football-Data.co.uk via sd.MatchHistory.
    Incluye resultados, goles, tarjetas amarillas (HY/AY) y esquinas (HC/AC).
    Muy rico para backtest de modelos predictivos.
    """
    log.info("=== Iniciando descarga Football-Data.co.uk (MatchHistory) ===")
    results = {}

    fd = sd.MatchHistory(leagues=leagues, seasons=seasons, data_dir=DATA_DIR)

    log.info("Descargando: Partidos históricos (tarjetas, esquinas, odds)...")
    results["schedule_odds"] = fd.read_games()

    log.info("=== Football-Data.co.uk completado ✓ ===")
    return results


# ─────────────────────────────────────────────
# MÓDULO 4: FOOTBALL-DATA.ORG — Tarjetas amarillas
# ─────────────────────────────────────────────
# API oficial con clave personal. Proporciona bookings (tarjetas) por partido.
# Tiros de esquina NO están disponibles en esta API → se guardan como NaN.
# Free tier: 10 req/min → sleep de 6.5 s entre llamadas.
# Los detalles individuales de cada partido se cachean en disco para
# no repetir descargas en ejecuciones sucesivas.

def _fdorg_session():
    """Sesión requests preconfigurada con la API key de football-data.org."""
    s = requests.Session()
    s.headers.update({
        "X-Auth-Token": FDORG_TOKEN,
        "Accept":       "application/json",
    })
    return s


def _fdorg_get(session, url, cache_path=None):
    """
    GET con caché en disco y rate limiting automático.
    Si cache_path existe devuelve el JSON en disco sin hacer petición.
    """
    if cache_path is not None and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    time.sleep(FDORG_RATE)
    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def collect_footballdata_org(days_back=90):
    """
    Descarga tarjetas amarillas de football-data.org para los últimos
    `days_back` días de las 5 ligas principales.

    Flujo por liga:
      1. GET /competitions/{code}/matches?status=FINISHED&dateFrom=…&dateTo=…
         → lista de partidos con IDs y IDs de equipos  (1 llamada)
      2. GET /matches/{id}  para cada partido
         → bookings con card=YELLOW / YELLOW_RED       (1 llamada/partido)

    Los archivos JSON se cachean en data/_fdorg_cache/ para evitar
    re-descargas. Borra la carpeta manualmente para forzar actualización.

    Salida: dict con clave "fdorg_cards" → DataFrame con columnas:
        league, date, home_team, away_team,
        home_yellow_cards, away_yellow_cards,
        home_corners (NaN), away_corners (NaN)
    """
    log.info("=== Iniciando descarga football-data.org API ===")

    session   = _fdorg_session()
    cache_dir = DATA_DIR / "_fdorg_cache"
    date_to   = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    records   = []

    for league, comp in FDORG_COMPS.items():
        log.info("  %s (%s) — últimos %d días...", league, comp, days_back)

        list_url   = (
            f"{FDORG_BASE}/competitions/{comp}/matches"
            f"?status=FINISHED&dateFrom={date_from}&dateTo={date_to}"
        )
        list_cache = cache_dir / f"list_{comp}_{date_from}_{date_to}.json"

        try:
            data    = _fdorg_get(session, list_url, list_cache)
            matches = data.get("matches", [])
            log.info("    %d partidos encontrados", len(matches))
        except Exception as exc:
            log.warning("    Error obteniendo lista de %s: %s", comp, exc)
            continue

        for match in matches:
            match_id  = match.get("id")
            home_info = match.get("homeTeam", {})
            away_info = match.get("awayTeam", {})
            home_name = home_info.get("name", "")
            away_name = away_info.get("name", "")
            home_id   = home_info.get("id")
            utc_date  = match.get("utcDate", "")[:10]

            if not match_id or not home_name:
                continue

            # ── Detalle del partido (incluye bookings) ──────────────────
            detail_cache = cache_dir / f"match_{match_id}.json"
            try:
                detail = _fdorg_get(
                    session,
                    f"{FDORG_BASE}/matches/{match_id}",
                    detail_cache,
                )
            except Exception as exc:
                log.debug("    Partido %s: error detalle — %s", match_id, exc)
                continue

            bookings = detail.get("bookings", [])

            home_y = sum(
                1 for b in bookings
                if b.get("card") in ("YELLOW", "YELLOW_RED")
                and b.get("team", {}).get("id") == home_id
            )
            away_y = sum(
                1 for b in bookings
                if b.get("card") in ("YELLOW", "YELLOW_RED")
                and b.get("team", {}).get("id") != home_id
                and b.get("team") is not None
            )

            records.append({
                "league":            league,
                "date":              utc_date,
                "home_team":         home_name,
                "away_team":         away_name,
                "home_yellow_cards": home_y,
                "away_yellow_cards": away_y,
                "home_corners":      float("nan"),   # no disponible en esta API
                "away_corners":      float("nan"),
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    log.info("=== football-data.org completado ✓ — %d partidos ===", len(df))
    return {"fdorg_cards": df}


# ─────────────────────────────────────────────
# MÓDULO 5: SPORTSAPI (RapidAPI) — Estadísticas detalladas
# ─────────────────────────────────────────────
# Datos por partido: esquinas, amarillas, posesión, tiros al arco, tiros totales.
# Caché en data/_sportsapi_cache/ — borra la carpeta para forzar actualización.

def _sportsapi_session():
    s = requests.Session()
    s.headers.update({
        "x-rapidapi-key":  SPORTSAPI_KEY,
        "x-rapidapi-host": SPORTSAPI_HOST,
    })
    return s


def _sportsapi_get(session, url, cache_path=None):
    """GET con caché JSON en disco y rate limiting automático."""
    if cache_path is not None and cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)
    time.sleep(SPORTSAPI_RATE)
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def _sportsapi_current_season(session, t_id, cache_dir):
    """Devuelve el ID de la temporada más reciente para un torneo."""
    url = f"{SPORTSAPI_BASE}/unique-tournament/{t_id}/seasons"
    try:
        data    = _sportsapi_get(session, url, cache_dir / f"seasons_{t_id}.json")
        seasons = data.get("seasons", [])
        if seasons:
            return seasons[0]["id"]
    except Exception as exc:
        log.debug("Seasons torneo %d: %s", t_id, exc)
    return None


def _extract_stat(groups, key):
    """
    Extrae (home_value, away_value) de una stat key dentro de los groups
    de un período de estadísticas. Maneja valores numéricos y strings con '%'.
    """
    for group in groups:
        for item in group.get("statisticsItems", []):
            if item.get("key") == key:
                def _to_float(v):
                    if v is None:
                        return None
                    if isinstance(v, str):
                        v = v.replace("%", "").strip()
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None
                return _to_float(item.get("homeValue")), _to_float(item.get("awayValue"))
    return None, None


def collect_sportsapi_match_stats(days_back=90):
    """
    Descarga estadísticas de partido via SportsAPI (RapidAPI / SofaScore).

    Por cada liga:
      1. Obtiene el ID de la temporada actual (dinámicamente).
      2. Itera páginas de eventos recientes (~10 por página, orden inverso).
      3. Para cada partido completado dentro de `days_back` días: descarga
         el detalle de estadísticas.
      4. Extrae:
           - cornerKicks          → esquinas
           - yellowCards          → tarjetas amarillas
           - ballPossession       → posesión (%)
           - shotsOnTarget        → tiros al arco
           - totalShots           → tiros totales
           - bigChances           → ocasiones claras
           - tackles              → entradas

    Todas las respuestas se cachean en data/_sportsapi_cache/.

    Salida → "sportsapi_match_stats": DataFrame con columnas:
        league, date, event_id, home_team, away_team,
        home_corners, away_corners,
        home_yellow_cards, away_yellow_cards,
        home_possession_pct, away_possession_pct,
        home_shots_on_target, away_shots_on_target,
        home_shots_total, away_shots_total,
        home_big_chances, away_big_chances
    """
    log.info("=== Iniciando descarga SportsAPI (RapidAPI) ===")

    session   = _sportsapi_session()
    cache_dir = DATA_DIR / "_sportsapi_cache"
    cutoff    = datetime.now() - timedelta(days=days_back)
    records   = []

    for league, t_id in SPORTSAPI_TOURNAMENTS.items():
        log.info("  %s (tournament_id=%d)...", league, t_id)

        season_id = _sportsapi_current_season(session, t_id, cache_dir)
        if season_id is None:
            log.warning("    Sin season ID — saltando %s", league)
            continue
        log.info("    Season ID: %s", season_id)

        # ── Obtener lista de eventos recientes (páginas en orden inverso) ─
        collected_events = []
        for page in range(20):   # máx ~200 partidos por liga
            url   = (f"{SPORTSAPI_BASE}/unique-tournament/{t_id}"
                     f"/season/{season_id}/events/last/{page}")
            cache = cache_dir / f"events_{t_id}_{season_id}_p{page}.json"
            try:
                data   = _sportsapi_get(session, url, cache)
                events = data.get("events", [])
                if not events:
                    break
                stop = False
                for ev in events:
                    ts      = ev.get("startTimestamp", 0)
                    ev_date = datetime.fromtimestamp(ts)
                    status  = ev.get("status", {}).get("type", "")
                    if status == "finished" and ev_date >= cutoff:
                        collected_events.append((ev, ev_date))
                    if ev_date < cutoff:
                        stop = True
                if stop:
                    break
            except Exception as exc:
                log.debug("    Página %d: %s", page, exc)
                break

        log.info("    %d partidos encontrados en rango", len(collected_events))

        # ── Estadísticas por partido ──────────────────────────────────────
        for ev, ev_date in collected_events:
            ev_id    = ev.get("id")
            home_obj = ev.get("homeTeam", {})
            away_obj = ev.get("awayTeam", {})

            stats_url   = f"{SPORTSAPI_BASE}/event/{ev_id}/statistics"
            stats_cache = cache_dir / f"stats_{ev_id}.json"
            try:
                raw = _sportsapi_get(session, stats_url, stats_cache)
                # Buscar período "ALL" (estadísticas del partido completo)
                all_groups = []
                for period in raw.get("statistics", []):
                    if period.get("period", "").upper() == "ALL":
                        all_groups = period.get("groups", [])
                        break
                if not all_groups and raw.get("statistics"):
                    all_groups = raw["statistics"][0].get("groups", [])
            except Exception as exc:
                log.debug("    Stats evento %s: %s", ev_id, exc)
                all_groups = []

            def gs(key):
                return _extract_stat(all_groups, key)

            h_cor, a_cor = gs("cornerKicks")
            h_yel, a_yel = gs("yellowCards")
            h_pos, a_pos = gs("ballPossession")
            h_sot, a_sot = gs("shotsOnGoal")          # API key real (no "shotsOnTarget")
            h_tot, a_tot = gs("totalShotsOnGoal")      # API key real (no "totalShots")
            h_bc,  a_bc  = gs("bigChanceCreated")      # API key real (no "bigChances")

            records.append({
                "league":               league,
                "date":                 ev_date.strftime("%Y-%m-%d"),
                "event_id":             ev_id,
                "home_team":            home_obj.get("name", ""),
                "away_team":            away_obj.get("name", ""),
                "home_corners":         h_cor,
                "away_corners":         a_cor,
                "home_yellow_cards":    h_yel,
                "away_yellow_cards":    a_yel,
                "home_possession_pct":  h_pos,
                "away_possession_pct":  a_pos,
                "home_shots_on_target": h_sot,
                "away_shots_on_target": a_sot,
                "home_shots_total":     h_tot,
                "away_shots_total":     a_tot,
                "home_big_chances":     h_bc,
                "away_big_chances":     a_bc,
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

    log.info("=== SportsAPI completado ✓ — %d partidos ===", len(df))
    return {"sportsapi_match_stats": df}


# ─────────────────────────────────────────────
# MÓDULO 6: CLUB ELO — Rankings históricos
# ─────────────────────────────────────────────

def collect_clubelo():
    """
    Descarga rankings Elo históricos de todos los clubes europeos.
    Útil como feature de fuerza relativa entre equipos.
    """
    log.info("=== Iniciando descarga Club Elo ===")
    elo = sd.ClubElo(data_dir=DATA_DIR)

    log.info("Descargando: Club Elo histórico...")
    elo_history = elo.read_by_date()

    log.info("=== Club Elo completado ✓ ===")
    return {"elo_history": elo_history}


# ─────────────────────────────────────────────
# GUARDAR DATOS A CSV / PARQUET
# ─────────────────────────────────────────────

def save_datasets(all_data: dict, fmt: str = "parquet"):
    """
    Guarda todos los DataFrames en disco.
    fmt: "parquet" (recomendado para ML) o "csv" (más legible)
    """
    output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    for name, df in all_data.items():
        if df is None or df.empty:
            log.warning(f"DataFrame vacío, omitiendo: {name}")
            continue

        # Aplanar MultiIndex si existe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(filter(None, map(str, c))).strip() for c in df.columns]
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()

        filepath = os.path.join(output_dir, f"{name}.{fmt}")
        if fmt == "parquet":
            df.to_parquet(filepath, index=True)
        else:
            df.to_csv(filepath, index=True)

        log.info(f"Guardado: {filepath} ({len(df)} filas)")

    log.info(f"\n✅ Todos los datasets guardados en: {output_dir}")


# ─────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    all_data = {}

    # ── Descomenta los módulos que quieras ejecutar ──

    # 1. Understat (fuente principal — xG, resultados, stats)
    all_data.update(collect_understat())

    # 2. football-data.org (tarjetas amarillas — últimos 90 días, con caché)
    all_data.update(collect_footballdata_org(days_back=90))

    # 3. SportsAPI — esquinas, amarillas, posesión, tiros al arco (últimos 90 días)
    all_data.update(collect_sportsapi_match_stats(days_back=90))

    # 3. Club Elo (rankings históricos)
    # all_data.update(collect_clubelo())

    # 4. Football-Data (histórico + odds — ya no necesario para amarillas/esquinas)
    # all_data.update(collect_footballdata())

    # 5. FBref completo (stats avanzadas — puede tener bloqueos 403)
    # all_data.update(collect_fbref())

    # Guardar todos los datasets
    save_datasets(all_data, fmt="parquet")  # cambia a "csv" si prefieres

    # Vista previa rápida
    print("\n📊 RESUMEN DE DATOS DESCARGADOS:")
    print("-" * 50)
    for name, df in all_data.items():
        print(f"  {name:<35} → {len(df):>6} filas × {df.shape[1]:>3} columnas")
