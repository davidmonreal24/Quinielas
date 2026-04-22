"""
utils/config.py — Configuración centralizada del proyecto
==========================================================
Carga variables del archivo .env en la raíz del proyecto.
Todos los scripts importan desde aquí en lugar de hardcodear valores.

Uso:
    from utils.config import SOFASCORE_KEY, ODDS_API_KEY, PATHS
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Ruta raíz del proyecto (dos niveles arriba de utils/)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ─── Claves API ───────────────────────────────────────────────────────────────
SOFASCORE_KEY    = os.getenv("SOFASCORE_RAPIDAPI_KEY", "")
SOFASCORE_HOST   = os.getenv("SOFASCORE_RAPIDAPI_HOST", "sofascore6.p.rapidapi.com")
APIFOOTBALL_KEY  = os.getenv("APIFOOTBALL_KEY", "")
ODDS_API_KEY     = os.getenv("ODDS_API_KEY", "")

# ─── URLs base ────────────────────────────────────────────────────────────────
SOFASCORE_BASE      = f"https://{SOFASCORE_HOST}/api/sofascore/v1"
APIFOOTBALL_BASE    = "https://v3.football.api-sports.io"
ODDS_API_BASE       = "https://api.the-odds-api.com/v4"
ESPN_CORE_BASE      = "https://sports.core.api.espn.com/v2/sports/soccer/leagues"

# ─── IDs de liga por sistema ──────────────────────────────────────────────────
SOFASCORE_IDS = {
    "ligamx":     11620,
    "ucl":        7,
    "premier":    17,
    "laliga":     8,
    "bundesliga": 35,
    "seriea":     23,
    "ligue1":     34,
}

APIFOOTBALL_IDS = {
    "ligamx": 262,
    "ucl":    2,
}

# ESPN: season=año de inicio, type=1 Clausura / type=2 Apertura
ESPN_LIGAMX_LEADERS = f"{ESPN_CORE_BASE}/mex.1/seasons/2025/types/1/leaders"

# ─── Rutas de datos ───────────────────────────────────────────────────────────
DATA_DIR      = ROOT / "data"
ARCHIVE_DIR   = DATA_DIR / "archive"

PATHS = {
    "events":          DATA_DIR / "sofascore_events.parquet",
    "upcoming":        DATA_DIR / "sofascore_upcoming.parquet",
    "ligamx_csv":      DATA_DIR / "ligamx_predicciones.csv",
    "ligamx_docx":     DATA_DIR / "ligamx_metodologia.docx",
    "odds_cache":      DATA_DIR / "ligamx_odds_cache.json",
    "espn_cache":      DATA_DIR / "_espn_cache",
    "fbref_cache":     DATA_DIR / "_fbref_cache",
    "sofascore_cache": DATA_DIR / "_sofascore_cache",
    "archive":         ARCHIVE_DIR,
}

# ─── Parámetros del modelo Liga MX ───────────────────────────────────────────
MODEL = {
    "min_games":   3,
    "window":      8,
    "alpha_ewm":   0.7,
    "shrink_k":    4,
    "lambda_max":  3.2,
    "draw_floor":  0.15,
    "dc_rho":     -0.13,
    "alta_gap":    0.18,
    "media_gap":   0.10,
    "ev_alta":     0.10,
    "ev_media":    0.05,
}

# ─── Validación al importar ───────────────────────────────────────────────────
def check_keys() -> list[str]:
    """Retorna lista de claves faltantes en .env."""
    missing = []
    if not SOFASCORE_KEY:
        missing.append("SOFASCORE_RAPIDAPI_KEY")
    if not APIFOOTBALL_KEY:
        missing.append("APIFOOTBALL_KEY")
    if not ODDS_API_KEY:
        missing.append("ODDS_API_KEY")
    return missing
