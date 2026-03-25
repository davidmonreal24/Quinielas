"""
=====================================================================
  Predicciones MVP — Soccer Project
  Versión simple y transparente: solo predicción de resultado.

  Fuente de datos: Understat via schedule_xg.parquet (2023 en adelante)
  Modelo:          Poisson independiente — sin ML, sin caja negra.

  Cómo se calculan las probabilidades:
    1. Se calcula el xG promedio ofensivo y defensivo de cada equipo
       en los últimos `window` partidos antes del encuentro (rolling form).
    2. λ_local    = (ataque_local + defensa_visitante) / 2
       λ_visitante = (ataque_visitante + defensa_local) / 2
    3. Se simula la distribución Poisson para todos los marcadores (h, a)
       hasta max_goles.  P(local) = Σ P(h>a), P(empate) = Σ P(h=a), etc.
    4. Las columnas intermedias exponen cada componente del cálculo.

  Uso:
    python predict_simple.py [--days 7] [--league "Premier"]
  Output:
    data/predicciones_simple.csv
=====================================================================
"""

import argparse
import math

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")

# Años desde los que se toman datos históricos para calcular la forma
MIN_DATE = "2023-01-01"

# Ventana de partidos recientes para calcular forma
WINDOW = 5

# xG por defecto si no hay datos históricos del equipo
LEAGUE_AVG_XG = 1.35


# ─────────────────────────────────────────────
# MODELO POISSON
# ─────────────────────────────────────────────

def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0 or np.isnan(lam):
        return 1.0 if k == 0 else 0.0
    return float(math.exp(-lam) * (lam ** k) / math.factorial(k))


def poisson_result_probs(lh: float, la: float, max_g: int = 7):
    """
    Calcula P(local gana), P(empate), P(visitante gana) usando
    distribución de Poisson independiente para cada equipo.
    """
    p_local = p_empate = p_visit = 0.0
    for h in range(max_g + 1):
        for a in range(max_g + 1):
            p = _poisson_pmf(h, lh) * _poisson_pmf(a, la)
            if h > a:
                p_local  += p
            elif h == a:
                p_empate += p
            else:
                p_visit  += p
    total = p_local + p_empate + p_visit
    if total > 0:
        p_local /= total
        p_empate /= total
        p_visit  /= total
    return p_local, p_empate, p_visit


# ─────────────────────────────────────────────
# FORMA RECIENTE DEL EQUIPO
# ─────────────────────────────────────────────

def get_team_form(played: pd.DataFrame, team: str, before_date, window: int = WINDOW):
    """
    xG promedio ofensivo y defensivo del equipo en sus últimos `window`
    partidos ANTES de `before_date`. Devuelve (xg_for, xg_against, n_games).
    """
    as_home = (
        played[played["home_team"] == team][["date", "home_xg", "away_xg"]]
        .rename(columns={"home_xg": "xg_for", "away_xg": "xg_against"})
    )
    as_away = (
        played[played["away_team"] == team][["date", "away_xg", "home_xg"]]
        .rename(columns={"away_xg": "xg_for", "home_xg": "xg_against"})
    )
    recent = (
        pd.concat([as_home, as_away])
        .sort_values("date")
        .loc[lambda d: d["date"] < before_date]
        .tail(window)
    )
    if recent.empty:
        return LEAGUE_AVG_XG * 1.05, LEAGUE_AVG_XG * 0.95, 0
    return (
        float(recent["xg_for"].mean()),
        float(recent["xg_against"].mean()),
        len(recent),
    )


# ─────────────────────────────────────────────
# POSICIÓN EN TABLA
# ─────────────────────────────────────────────

def get_table_position(played: pd.DataFrame, team: str, season: str, league: str, before_date):
    """Posición y puntos del equipo en su liga-temporada antes de la fecha."""
    s = played[
        (played["season"] == season) & (played["league"] == league)
        & (played["date"] < before_date)
    ].copy()
    if s.empty:
        return np.nan, np.nan

    pts_map = {}
    for _, r in s.iterrows():
        pts_map.setdefault(r["home_team"], 0)
        pts_map.setdefault(r["away_team"], 0)
        if r["home_goals"] > r["away_goals"]:
            pts_map[r["home_team"]] += 3
        elif r["home_goals"] == r["away_goals"]:
            pts_map[r["home_team"]] += 1
            pts_map[r["away_team"]] += 1
        else:
            pts_map[r["away_team"]] += 3

    if team not in pts_map:
        return np.nan, np.nan

    team_pts = pts_map[team]
    pos = sum(1 for v in pts_map.values() if v > team_pts) + 1
    return pos, team_pts


# ─────────────────────────────────────────────
# HEAD-TO-HEAD
# ─────────────────────────────────────────────

def get_h2h(played: pd.DataFrame, home: str, away: str, before_date, window: int = 5):
    """Historial directo entre los dos equipos (últimos `window` partidos)."""
    mask = (
        (played["date"] < before_date)
        & (
            ((played["home_team"] == home) & (played["away_team"] == away))
            | ((played["home_team"] == away) & (played["away_team"] == home))
        )
    )
    h2h = played[mask].tail(window)
    n = len(h2h)
    if n == 0:
        return 0, np.nan, np.nan, np.nan

    home_wins = int(
        ((h2h["home_team"] == home)  & (h2h["home_goals"] > h2h["away_goals"])).sum()
        + ((h2h["away_team"] == home) & (h2h["away_goals"] > h2h["home_goals"])).sum()
    )
    draws     = int((h2h["home_goals"] == h2h["away_goals"]).sum())
    away_wins = n - home_wins - draws
    return n, home_wins / n, draws / n, away_wins / n


# ─────────────────────────────────────────────
# FORMATO DE SALIDA
# ─────────────────────────────────────────────

def _fmt(v, dec=2):
    try:
        f = float(v)
        return round(f, dec) if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


def _pct(v):
    try:
        f = float(v)
        return round(f * 100, 1) if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(days_ahead: int = 7, league_filter: str = None):
    print("=" * 70)
    print("  SOCCER PROJECT — Predicciones MVP (modelo Poisson transparente)")
    print("=" * 70)

    # ── Cargar datos ──
    schedule = pd.read_parquet(DATA_DIR / "schedule_xg.parquet").reset_index()
    schedule["date"] = pd.to_datetime(schedule["date"])

    # Filtrar desde 2023 para los datos históricos de forma
    history = schedule[
        (schedule["is_result"] == True)
        & (schedule["date"] >= pd.Timestamp(MIN_DATE))
    ].copy()

    # Partidos próximos
    today    = pd.Timestamp.now().normalize()
    end_date = today + pd.Timedelta(days=days_ahead)
    upcoming = schedule[
        (schedule["is_result"] == False)
        & (schedule["date"] >= today)
        & (schedule["date"] <= end_date)
    ].copy()

    if league_filter:
        upcoming = upcoming[upcoming["league"].str.contains(league_filter, case=False)]

    if upcoming.empty:
        print(f"\n  No hay partidos programados en los próximos {days_ahead} días.")
        return

    print(f"\n  Fecha:           {today.strftime('%d/%m/%Y')}")
    print(f"  Ventana búsqueda: próximos {days_ahead} días")
    print(f"  Partidos:         {len(upcoming)}")
    print(f"  Datos históricos: desde {MIN_DATE} ({len(history)} partidos)\n")

    SEP = "-" * 70
    output_rows = []

    for _, match in upcoming.sort_values("date").iterrows():
        home   = match["home_team"]
        away   = match["away_team"]
        date   = match["date"]
        season = match["season"]
        league = match["league"]

        # ── Forma reciente ──
        h_att, h_def, h_n = get_team_form(history, home, date)
        a_att, a_def, a_n = get_team_form(history, away, date)

        # ── λ calculado como media entre ataque propio y defensa del rival ──
        lh = (h_att + a_def) / 2   # goles esperados del local
        la = (a_att + h_def) / 2   # goles esperados del visitante

        # ── Poisson → probabilidades de resultado ──
        p_loc, p_emp, p_vis = poisson_result_probs(lh, la)
        if p_loc >= p_emp and p_loc >= p_vis:
            prediccion = "Local"
        elif p_emp >= p_loc and p_emp >= p_vis:
            prediccion = "Empate"
        else:
            prediccion = "Visitante"

        # ── Contexto adicional (tabla, H2H) ──
        h_pos, h_pts = get_table_position(history, home, season, league, date)
        a_pos, a_pts = get_table_position(history, away, season, league, date)
        h2h_n, h2h_pw_loc, h2h_pw_emp, h2h_pw_vis = get_h2h(history, home, away, date)

        # ── Registro de salida ──
        row = {
            # Identificación
            "fecha":                        date.strftime("%Y-%m-%d"),
            "liga":                         league,
            "local":                        home,
            "visitante":                    away,

            # Resultado predicho
            "prediccion":                   prediccion,
            "p_local_%":                    round(p_loc * 100, 1),
            "p_empate_%":                   round(p_emp * 100, 1),
            "p_visitante_%":                round(p_vis * 100, 1),

            # ── Cómo se calculó: paso 1 — forma reciente (xG rolling) ──
            "xg_ataque_local":              _fmt(h_att),
            "xg_defensa_local":             _fmt(h_def),
            "xg_ataque_visitante":          _fmt(a_att),
            "xg_defensa_visitante":         _fmt(a_def),
            "partidos_muestra_local":       h_n,
            "partidos_muestra_visitante":   a_n,

            # ── Cómo se calculó: paso 2 — λ (goles esperados) ──
            "lambda_local":                 _fmt(lh),
            "lambda_visitante":             _fmt(la),
            # interpretación: local favorable si lambda_local > lambda_visitante
            "ventaja_goles":                _fmt(lh - la),

            # ── Contexto — tabla de posiciones ──
            "pos_tabla_local":              _fmt(h_pos, 0),
            "pts_local":                    _fmt(h_pts, 0),
            "pos_tabla_visitante":          _fmt(a_pos, 0),
            "pts_visitante":                _fmt(a_pts, 0),
            "dif_tabla":                    _fmt(
                (a_pos - h_pos) if not (pd.isna(a_pos) or pd.isna(h_pos)) else np.nan, 0
            ),  # positivo = local está mejor posicionado

            # ── Contexto — H2H ──
            "h2h_partidos":                 h2h_n,
            "h2h_p_gana_local_%":           _pct(h2h_pw_loc),
            "h2h_p_empate_%":               _pct(h2h_pw_emp),
            "h2h_p_gana_visitante_%":       _pct(h2h_pw_vis),
        }
        output_rows.append(row)

        # ── Consola ──
        liga_fmt = league.replace("ENG-Premier League", "PL").replace("ESP-La Liga", "LaLiga") \
                         .replace("GER-Bundesliga", "Bundes").replace("ITA-Serie A", "SerieA") \
                         .replace("FRA-Ligue 1", "Ligue1")
        print(SEP)
        print(f"  {date.strftime('%d/%m/%y')}  {liga_fmt:<8}  "
              f"{home[:24]:<24} vs  {away[:24]}")
        print(f"  Goles esp: {lh:.2f} - {la:.2f}  =>  "
              f"Local {p_loc*100:.1f}%  /  Empate {p_emp*100:.1f}%  /  Visitante {p_vis*100:.1f}%"
              f"  [{prediccion}]")
        print(f"  Forma local:     xG ofensivo {h_att:.2f} / xG concedido {h_def:.2f}"
              f"  (últimos {h_n} partidos)")
        print(f"  Forma visitante: xG ofensivo {a_att:.2f} / xG concedido {a_def:.2f}"
              f"  (últimos {a_n} partidos)")
        if not pd.isna(h_pos):
            print(f"  Tabla: {home[:20]} #{int(h_pos)} ({int(h_pts) if not pd.isna(h_pts) else '?'} pts)"
                  f"  vs  {away[:20]} #{int(a_pos)} ({int(a_pts) if not pd.isna(a_pts) else '?'} pts)")
        if h2h_n > 0:
            print(f"  H2H ({h2h_n} partidos): "
                  f"local gana {h2h_pw_loc*100:.0f}%  /  "
                  f"empate {h2h_pw_emp*100:.0f}%  /  "
                  f"visitante gana {h2h_pw_vis*100:.0f}%")

    print(SEP)
    print(f"\n  Modelo: Poisson independiente")
    print(f"  Fuente: Understat xG (schedule_xg.parquet, desde {MIN_DATE})")

    # ── Guardar CSV ──
    out_path = Path("data/predicciones_simple.csv")
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  CSV guardado: {out_path}")
    print(f"  {len(out_df)} partidos  ×  {len(out_df.columns)} columnas\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicciones MVP — resultados con explicación")
    parser.add_argument("--days",   type=int,   default=7,    help="Días hacia adelante (default: 7)")
    parser.add_argument("--league", type=str,   default=None, help="Filtrar liga: 'Premier', 'LaLiga', etc.")
    args = parser.parse_args()
    main(days_ahead=args.days, league_filter=args.league)
