"""
utils/logger.py — Logging centralizado del proyecto
=====================================================
Uso en cualquier script:
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Recolectando datos...")
    log.warning("API sin respuesta, usando caché")
    log.error("Fallo al escribir CSV")

Salidas:
  - Consola: nivel INFO+ (formato compacto)
  - Archivo:  data/logs/YYYY-MM-DD.log (nivel DEBUG+, todo queda registrado)
"""
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

CDT = timezone(timedelta(hours=-6))

_LOG_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"
_FMT_CONSOLE = "%(levelname)-8s [%(name)s] %(message)s"
_FMT_FILE    = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE_FMT    = "%H:%M:%S"


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Retorna un logger con handlers de consola y archivo configurados.
    Llamadas repetidas con el mismo `name` devuelven el mismo logger
    (los handlers no se duplican).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ── Consola ──────────────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(_FMT_CONSOLE, datefmt=_DATE_FMT))
    logger.addHandler(console)

    # ── Archivo diario ────────────────────────────────────────────────────────
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str  = datetime.now(CDT).strftime("%Y-%m-%d")
    log_file  = _LOG_DIR / f"{date_str}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FMT_FILE, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
