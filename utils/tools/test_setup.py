"""
=====================================================================
  TEST RÁPIDO — Verificar instalación y conexión
  Ejecuta esto PRIMERO antes de collect_data.py
  Comando: python test_setup.py
=====================================================================
"""

import sys

print("=" * 55)
print("  SOCCER DATA — Test de instalación")
print("=" * 55)

# ── Test 1: Versión de Python ──
print(f"\n[1/5] Python version: {sys.version.split()[0]}", end="")
assert sys.version_info >= (3, 9), "Se requiere Python 3.9+"
print(" ✓")

# ── Test 2: Librerías requeridas ──
print("[2/5] Importando librerías...", end="")
try:
    import pandas as pd
    import numpy as np
    import soccerdata as sd
    import pyarrow          # para guardar en parquet
    print(" ✓")
except ImportError as e:
    print(f"\n  ❌ Falta instalar: {e}")
    print("  Ejecuta: pip install -r requirements.txt")
    sys.exit(1)

# ── Test 3: Versiones ──
print(f"[3/5] Versiones: pandas={pd.__version__}, "
      f"numpy={np.__version__}, soccerdata={sd.__version__} ✓")

# ── Test 4: Descarga mínima de Understat ──
print("[4/5] Probando conexión a Understat (descarga pequeña)...", end="")
try:
    understat = sd.Understat(leagues="ENG-Premier League", seasons="2023-24")
    schedule = understat.read_schedule()
    assert len(schedule) > 0
    print(f" ✓  ({len(schedule)} partidos descargados)")
except Exception as e:
    print(f"\n  ❌ Error de conexión: {e}")
    print("  Verifica tu conexión a internet.")
    sys.exit(1)

# ── Test 5: Vista previa de datos ──
print("[5/5] Vista previa de datos:")
print()
print(schedule[["home_team", "away_team", "date"]].head(5).to_string())

print()
print("=" * 55)
print("  ✅ Todo listo. Puedes ejecutar collect_data.py")
print("=" * 55)
