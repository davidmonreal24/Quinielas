"""
=====================================================================
  Predicción de Próximos Partidos — Soccer Project
  Uso:    python predict_upcoming.py [--days 7] [--league "Premier"]
  Output: tabla en consola + data/predicciones_proximas.csv

  Predicciones incluidas:
    · Resultado:   Local / Empate / Visitante  (XGBoost pre-partido)
    · Total goles: esperados, Over 2.5, BTTS   (modelo Poisson)
    · Marcador exacto más probable              (modelo Poisson)
    · Tarjetas amarillas (local / visitante / total)   [FBref misc]
    · Tiros de esquina (local / visitante / total)     [FBref passing_types]
    · Goleadores más probables por equipo              [player stats]
=====================================================================
"""

import argparse
import math
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

DATA_DIR   = Path("data/processed")
MODELS_DIR = Path("data/models")

WINDOW_FORM   = 5    # últimos N partidos para forma reciente
WINDOW_H2H    = 5    # últimos N enfrentamientos directos
LEAGUE_AVG_XG = 1.4  # fallback cuando no hay rolling xG


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def _safe_sub(a, b):
    """Resta a - b devolviendo NaN si alguno es NaN o None."""
    try:
        a, b = float(a), float(b)
        return np.nan if (np.isnan(a) or np.isnan(b)) else a - b
    except (TypeError, ValueError):
        return np.nan


def _safe_add(a, b):
    """Suma a + b devolviendo NaN si alguno es NaN o None."""
    try:
        a, b = float(a), float(b)
        return np.nan if (np.isnan(a) or np.isnan(b)) else round(a + b, 1)
    except (TypeError, ValueError):
        return np.nan


def _fmt(v, dec=1, suffix=""):
    """Formatea un número o devuelve 'N/D' si es NaN/None."""
    try:
        f = float(v)
        if np.isnan(f):
            return "N/D"
        return f"{f:.{dec}f}{suffix}"
    except (TypeError, ValueError):
        return "N/D"


# ─────────────────────────────────────────────
# MODELO DE POISSON (sin dependencias externas)
# ─────────────────────────────────────────────

def _poisson_pmf(k, lam):
    """P(X=k) para X ~ Poisson(lam). Implementado con math para evitar scipy."""
    if np.isnan(lam) or lam <= 0:
        return 1.0 if k == 0 else 0.0
    return float(np.exp(-lam) * (lam ** k) / math.factorial(k))


def compute_lambdas(hf, af):
    """
    Goles esperados usando Dixon-Robinson simplificado.
    λ_local    = media(ataque_local,     defensa_visitante)
    λ_visitante = media(ataque_visitante, defensa_local)
    Si faltan datos usa la media de liga con pequeño factor local/visitante.
    """
    h_att = hf.get("roll_xg_for",     np.nan)
    h_def = hf.get("roll_xg_against", np.nan)
    a_att = af.get("roll_xg_for",     np.nan)
    a_def = af.get("roll_xg_against", np.nan)

    parts_h = [v for v in [h_att, a_def] if not np.isnan(v)]
    parts_a = [v for v in [a_att, h_def] if not np.isnan(v)]

    lh = float(np.mean(parts_h)) if parts_h else LEAGUE_AVG_XG * 1.1
    la = float(np.mean(parts_a)) if parts_a else LEAGUE_AVG_XG * 0.9
    return lh, la


def goals_predictions(lh, la, max_g=7):
    """
    Predicciones Poisson: goles esperados, over 2.5, BTTS, marcador exacto.
    Itera sobre scores (h, a) hasta max_g para cada equipo.
    """
    score_probs = {
        (h, a): _poisson_pmf(h, lh) * _poisson_pmf(a, la)
        for h in range(max_g + 1)
        for a in range(max_g + 1)
    }
    best_score = max(score_probs, key=score_probs.get)
    best_prob  = score_probs[best_score]
    over_25    = sum(p for (h, a), p in score_probs.items() if h + a > 2)
    btts       = (1 - _poisson_pmf(0, lh)) * (1 - _poisson_pmf(0, la))

    return {
        "lambda_home":       round(lh, 2),
        "lambda_away":       round(la, 2),
        "goles_esperados":   round(lh + la, 2),
        "over_2_5_pct":      round(over_25 * 100, 1),
        "btts_pct":          round(btts * 100, 1),
        "marcador_exacto":   f"{best_score[0]}-{best_score[1]}",
        "marcador_prob_pct": round(best_prob * 100, 1),
    }


# ─────────────────────────────────────────────
# FORMA RECIENTE
# ─────────────────────────────────────────────

def get_team_form(played, team, before_date, window=WINDOW_FORM):
    """xG for/against promedio en los últimos N partidos antes de la fecha."""
    home = (
        played[played["home_team"] == team][["date", "home_xg", "away_xg"]]
        .rename(columns={"home_xg": "xg_for", "away_xg": "xg_against"})
    )
    away = (
        played[played["away_team"] == team][["date", "away_xg", "home_xg"]]
        .rename(columns={"away_xg": "xg_for", "home_xg": "xg_against"})
    )
    recent = (
        pd.concat([home, away])
        .sort_values("date")
        .loc[lambda d: d["date"] < before_date]
        .tail(window)
    )
    if recent.empty:
        return {"roll_xg_for": np.nan, "roll_xg_against": np.nan}
    return {
        "roll_xg_for":     float(recent["xg_for"].mean()),
        "roll_xg_against": float(recent["xg_against"].mean()),
    }


# ─────────────────────────────────────────────
# TABLA DE POSICIONES
# ─────────────────────────────────────────────

def get_table_position(played, season, before_date):
    """Tabla acumulada hasta before_date para la temporada indicada."""
    s = played[(played["season"] == season) & (played["date"] < before_date)].copy()
    if s.empty:
        return {}, {}

    home = s[["home_team", "home_goals", "away_goals"]].copy()
    home["pts"] = np.where(
        home["home_goals"] > home["away_goals"], 3,
        np.where(home["home_goals"] == home["away_goals"], 1, 0),
    )
    home = home.rename(columns={"home_team": "team"})

    away = s[["away_team", "away_goals", "home_goals"]].copy()
    away["pts"] = np.where(
        away["away_goals"] > away["home_goals"], 3,
        np.where(away["away_goals"] == away["home_goals"], 1, 0),
    )
    away = away.rename(columns={"away_team": "team"})

    table = (
        pd.concat([home[["team", "pts"]], away[["team", "pts"]]])
        .groupby("team")["pts"].sum()
        .reset_index()
    )
    table["pos"] = table["pts"].rank(ascending=False, method="min").astype(int)
    return dict(zip(table["team"], table["pos"])), dict(zip(table["team"], table["pts"]))


# ─────────────────────────────────────────────
# HEAD-TO-HEAD
# ─────────────────────────────────────────────

def get_h2h(played, home_team, away_team, before_date, window=WINDOW_H2H):
    """H2H histórico entre dos equipos antes de before_date."""
    mask = (
        (played["date"] < before_date)
        & (
            ((played["home_team"] == home_team) & (played["away_team"] == away_team))
            | ((played["home_team"] == away_team) & (played["away_team"] == home_team))
        )
    )
    h2h = played[mask].tail(window)
    n   = len(h2h)

    if n == 0:
        return {
            "h2h_matches": 0, "h2h_home_win_rate": np.nan,
            "h2h_draw_rate": np.nan, "h2h_away_win_rate": np.nan,
            "h2h_avg_goals": np.nan,
        }

    home_wins = int(
        ((h2h["home_team"] == home_team) & (h2h["resultado"] == 1)).sum()
        + ((h2h["away_team"] == home_team) & (h2h["resultado"] == -1)).sum()
    )
    draws     = int((h2h["resultado"] == 0).sum())
    away_wins = n - home_wins - draws

    return {
        "h2h_matches":       n,
        "h2h_home_win_rate": home_wins / n,
        "h2h_draw_rate":     draws / n,
        "h2h_away_win_rate": away_wins / n,
        "h2h_avg_goals":     float((h2h["home_goals"] + h2h["away_goals"]).mean()),
    }


# ─────────────────────────────────────────────
# CALIDAD DE PLANTILLA
# ─────────────────────────────────────────────

def get_squad_quality(players, team, season, league):
    """np_xg, xa, xg_chain por partido para team-season."""
    s = players[
        (players["team"] == team) & (players["season"] == season)
        & (players["league"] == league)
    ]
    if s.empty:
        s = players[(players["team"] == team) & (players["league"] == league)]
    if s.empty:
        return {"squad_np_xg": np.nan, "squad_xa": np.nan, "squad_xg_chain": np.nan}
    return {
        "squad_np_xg":    float(s["np_xg"].sum()) / 19,
        "squad_xa":       float(s["xa"].sum()) / 19,
        "squad_xg_chain": float(s["xg_chain"].sum()) / 19,
    }


# ─────────────────────────────────────────────
# TARJETAS AMARILLAS (football-data.org)
# ─────────────────────────────────────────────
# Esquinas no disponibles en esta API → siempre N/D.

def _norm_team(name):
    """
    Normaliza un nombre de equipo para fuzzy matching entre fuentes.
    Elimina sufijos legales comunes (FC, CF, AC…) y convierte a minúsculas.
    """
    name = str(name).lower().strip()
    name = re.sub(
        r"\b(fc|cf|sc|ac|as|ss|ssc|afc|bv|sv|fk|sk|ud|rcd|cd|sad|sae)\b",
        "", name,
    )
    # Abreviaturas conocidas
    name = name.replace("paris saint-germain", "psg")
    name = name.replace("atletico de madrid", "atletico madrid")
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _best_team_match(target, candidates, threshold=0.72):
    """
    Devuelve el nombre de `candidates` más parecido a `target`.
    Usa SequenceMatcher sobre nombres normalizados.
    Retorna None si ningún candidato supera el umbral.
    """
    t = _norm_team(target)
    best_name, best_ratio = None, threshold
    for c in candidates:
        ratio = SequenceMatcher(None, t, _norm_team(c)).ratio()
        if ratio > best_ratio:
            best_ratio, best_name = ratio, c
    return best_name


def load_fdorg_cards():
    """Carga fdorg_cards.parquet (football-data.org). Fallback para amarillas."""
    path = DATA_DIR / "fdorg_cards.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_sportsapi_stats():
    """
    Carga sportsapi_match_stats.parquet (SportsAPI / RapidAPI).
    Contiene: esquinas, amarillas, posesión, tiros al arco, ocasiones claras.
    """
    path = DATA_DIR / "sportsapi_match_stats.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).reset_index()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _rolling(df, team, before_date, home_col, away_col, window=5):
    """
    Media móvil de una estadística para `team` en los últimos `window` partidos.
    Combina filas como local (home_col) y visitante (away_col).
    Usa fuzzy matching de nombre para compatibilidad entre fuentes.
    """
    if df is None or df.empty:
        return np.nan
    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    matched   = _best_team_match(team, all_teams)
    if matched is None:
        return np.nan
    past = df[df["date"] < before_date]
    vals = pd.to_numeric(
        pd.concat([
            past[past["home_team"] == matched].tail(window)[home_col],
            past[past["away_team"] == matched].tail(window)[away_col],
        ]),
        errors="coerce",
    ).dropna().tolist()
    return round(float(np.mean(vals)), 1) if vals else np.nan


def get_team_match_stats(sportsapi, fdorg, team, before_date, window=5):
    """
    Estadísticas rolling para un equipo combinando SportsAPI (fuente principal)
    y football-data.org (fallback para amarillas si SportsAPI no tiene datos).

    Retorna dict con:
        avg_yellows          — tarjetas amarillas promedio por partido
        avg_corners          — esquinas promedio por partido
        avg_possession       — posesión (%) promedio
        avg_shots_on_target  — tiros al arco promedio
        avg_big_chances      — ocasiones claras promedio
    """
    def _r(hc, ac, src=sportsapi):
        return _rolling(src, team, before_date, hc, ac, window)

    result = {
        "avg_yellows":         _r("home_yellow_cards",    "away_yellow_cards"),
        "avg_corners":         _r("home_corners",          "away_corners"),
        "avg_possession":      _r("home_possession_pct",   "away_possession_pct"),
        "avg_shots_on_target": _r("home_shots_on_target",  "away_shots_on_target"),
        "avg_big_chances":     _r("home_big_chances",      "away_big_chances"),
    }

    # Fallback para amarillas si SportsAPI no tuvo datos
    if np.isnan(result["avg_yellows"]):
        result["avg_yellows"] = _r("home_yellow_cards", "away_yellow_cards", fdorg)

    return result


# ─────────────────────────────────────────────
# GOLEADORES MÁS PROBABLES
# ─────────────────────────────────────────────

def get_top_scorers(players, team, season, league, top_n=2):
    """
    Jugadores con mayor np_xg acumulado para team-season.
    Devuelve lista de nombres, fallback a temporada más reciente del equipo.
    """
    if players.empty or "np_xg" not in players.columns:
        return ["N/D"]

    col_player = next((c for c in ["player", "name"] if c in players.columns), None)
    if col_player is None:
        return ["N/D"]

    s = players[
        (players["team"] == team) & (players["season"] == season)
        & (players["league"] == league)
    ]
    if s.empty:
        s = players[(players["team"] == team) & (players["league"] == league)]
    if s.empty:
        return ["N/D"]

    top = s.nlargest(top_n, "np_xg")
    return [str(r[col_player]) for _, r in top.iterrows()]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(days_ahead=7, league_filter=None):
    print("=" * 70)
    print("  SOCCER PROJECT — Predicciones de Próximos Partidos")
    print("=" * 70)

    # ── Cargar datos base ──
    schedule = pd.read_parquet(DATA_DIR / "schedule_xg.parquet").reset_index()
    schedule["date"] = pd.to_datetime(schedule["date"])

    try:
        players = pd.read_parquet(DATA_DIR / "player_season_stats.parquet").reset_index()
    except FileNotFoundError:
        players = pd.DataFrame()
        print("  Aviso: player_season_stats.parquet no encontrado — goleadores = N/D")

    sportsapi = load_sportsapi_stats()
    fdorg     = load_fdorg_cards()
    if sportsapi is None and fdorg is None:
        print("  Aviso: sin datos de tarjetas/esquinas — ejecuta collect_data.py primero.")
    elif sportsapi is None:
        print("  Aviso: sportsapi_match_stats.parquet no encontrado — esquinas = N/D")

    # ── Separar jugados / próximos ──
    played   = schedule[schedule["is_result"] == True].copy()
    upcoming = schedule[schedule["is_result"] == False].copy()

    played["resultado"] = np.where(
        played["home_goals"] > played["away_goals"],  1,
        np.where(played["home_goals"] < played["away_goals"], -1, 0),
    )

    today    = pd.Timestamp.now().normalize()
    end_date = today + pd.Timedelta(days=days_ahead)
    upcoming = upcoming[
        (upcoming["date"] >= today) & (upcoming["date"] <= end_date)
    ].copy()

    if league_filter:
        upcoming = upcoming[upcoming["league"].str.contains(league_filter, case=False)]

    if upcoming.empty:
        print(f"\n  No hay partidos programados en los próximos {days_ahead} días.")
        print(f"  (Verifica que collect_data.py se ejecutó recientemente.)")
        return

    # ── Cargar modelo pre-partido ──
    try:
        pm_model    = joblib.load(MODELS_DIR / "pre_match_model.pkl")
        pm_features = joblib.load(MODELS_DIR / "pre_match_feature_names.pkl")
        le          = joblib.load(MODELS_DIR / "label_encoder.pkl")
    except FileNotFoundError:
        print("\n  Error: Ejecuta primero train_model.py para generar pre_match_model.pkl")
        return

    classes    = list(le.classes_)
    idx_local  = classes.index(1)
    idx_empate = classes.index(0)
    idx_visit  = classes.index(-1)
    result_map = {1: "Local", 0: "Empate", -1: "Visitante"}

    print(f"\n  Fecha:    {today.strftime('%d/%m/%Y')}")
    print(f"  Ventana:  próximos {days_ahead} días ({end_date.strftime('%d/%m/%Y')})")
    print(f"  Partidos: {len(upcoming)}")
    src_ok = "SportsAPI+FDOrg" if (sportsapi is not None and fdorg is not None) \
             else ("SportsAPI" if sportsapi is not None else ("FDOrg" if fdorg is not None else "—"))
    print(f"  Fuentes stat: {src_ok}\n")

    # ── Procesar cada partido ──
    feat_rows  = []
    extra_rows = []
    info_rows  = []

    for _, match in upcoming.sort_values("date").iterrows():
        home   = match["home_team"]
        away   = match["away_team"]
        date   = match["date"]
        season = match["season"]
        league = match["league"]

        # --- Cómputos base ---
        hf = get_team_form(played, home, date)
        af = get_team_form(played, away, date)
        h2h = get_h2h(played, home, away, date)
        tbl_pos, tbl_pts = get_table_position(played, season, date)

        hq = get_squad_quality(players, home, season, league) if not players.empty \
             else {"squad_np_xg": np.nan, "squad_xa": np.nan, "squad_xg_chain": np.nan}
        aq = get_squad_quality(players, away, season, league) if not players.empty \
             else {"squad_np_xg": np.nan, "squad_xa": np.nan, "squad_xg_chain": np.nan}

        h_tbl = tbl_pos.get(home, np.nan)
        a_tbl = tbl_pos.get(away, np.nan)
        h_pts = tbl_pts.get(home, np.nan)
        a_pts = tbl_pts.get(away, np.nan)

        # --- Feature vector para el modelo ML ---
        feat_rows.append({
            "home_roll_xg_for":      hf["roll_xg_for"],
            "home_roll_xg_against":  hf["roll_xg_against"],
            "away_roll_xg_for":      af["roll_xg_for"],
            "away_roll_xg_against":  af["roll_xg_against"],
            "form_xg_diff":          _safe_sub(hf["roll_xg_for"],     af["roll_xg_for"]),
            "form_def_diff":         _safe_sub(af["roll_xg_against"],  hf["roll_xg_against"]),
            "home_squad_np_xg":      hq["squad_np_xg"],
            "home_squad_xa":         hq["squad_xa"],
            "home_squad_xg_chain":   hq["squad_xg_chain"],
            "away_squad_np_xg":      aq["squad_np_xg"],
            "away_squad_xa":         aq["squad_xa"],
            "away_squad_xg_chain":   aq["squad_xg_chain"],
            "squad_xg_diff":         _safe_sub(hq["squad_np_xg"], aq["squad_np_xg"]),
            "squad_xa_diff":         _safe_sub(hq["squad_xa"],     aq["squad_xa"]),
            "home_table_pos":        h_tbl,
            "away_table_pos":        a_tbl,
            "table_pos_diff":        _safe_sub(a_tbl, h_tbl),
            "pts_diff":              _safe_sub(h_pts, a_pts),
            "es_local":              1,
            "h2h_matches":           h2h["h2h_matches"],
            "h2h_home_win_rate":     h2h["h2h_home_win_rate"],
            "h2h_draw_rate":         h2h["h2h_draw_rate"],
            "h2h_away_win_rate":     h2h["h2h_away_win_rate"],
            "h2h_avg_goals":         h2h["h2h_avg_goals"],
        })

        # --- Predicciones complementarias ---
        lh, la       = compute_lambdas(hf, af)
        poisson_pred = goals_predictions(lh, la)

        hcc = get_team_match_stats(sportsapi, fdorg, home, date)
        acc = get_team_match_stats(sportsapi, fdorg, away, date)

        top_h = get_top_scorers(players, home, season, league) if not players.empty else ["N/D"]
        top_a = get_top_scorers(players, away, season, league) if not players.empty else ["N/D"]

        extra_rows.append({
            **poisson_pred,
            "amarillas_local":        hcc["avg_yellows"],
            "amarillas_visitante":    acc["avg_yellows"],
            "esquinas_local":         hcc["avg_corners"],
            "esquinas_visitante":     acc["avg_corners"],
            "posesion_local_pct":     hcc["avg_possession"],
            "posesion_visitante_pct": acc["avg_possession"],
            "tiros_puerta_local":     hcc["avg_shots_on_target"],
            "tiros_puerta_visitante": acc["avg_shots_on_target"],
            "ocasiones_local":        hcc["avg_big_chances"],
            "ocasiones_visitante":    acc["avg_big_chances"],
            "goleador_local":         ", ".join(top_h),
            "goleador_visitante":     ", ".join(top_a),
        })
        info_rows.append({
            "date": date, "league": league,
            "home_team": home, "away_team": away,
        })

    feat_df  = pd.DataFrame(feat_rows)
    extra_df = pd.DataFrame(extra_rows)
    info_df  = pd.DataFrame(info_rows)

    # ── Predicción del modelo ML ──
    X          = feat_df.reindex(columns=pm_features).values
    y_pred_enc = pm_model.predict(X)
    y_pred     = le.inverse_transform(y_pred_enc)
    y_proba    = pm_model.predict_proba(X)

    # ── Mostrar resultados ──
    def _fmt_liga(raw):
        return (raw.replace("ENG-Premier League", "Premier")
                   .replace("ESP-La Liga",        "LaLiga ")
                   .replace("GER-Bundesliga",      "Bundes.")
                   .replace("ITA-Serie A",         "SerieA ")
                   .replace("FRA-Ligue 1",         "Ligue1 "))

    SEP = "─" * 70
    output_rows = []

    for i, (_, info) in enumerate(info_df.iterrows()):
        pred  = result_map[y_pred[i]]
        p_loc = y_proba[i, idx_local]  * 100
        p_emp = y_proba[i, idx_empate] * 100
        p_vis = y_proba[i, idx_visit]  * 100
        ex    = extra_df.iloc[i]
        liga  = _fmt_liga(info["league"])

        am_total = _safe_add(ex["amarillas_local"],     ex["amarillas_visitante"])
        es_total = _safe_add(ex["esquinas_local"],      ex["esquinas_visitante"])
        tp_total = _safe_add(ex["tiros_puerta_local"],  ex["tiros_puerta_visitante"])

        print(SEP)
        print(
            f"  {info['date'].strftime('%d/%m/%y')}  {liga:<8}  "
            f"{info['home_team'][:24]:<24} vs  {info['away_team'][:24]}"
        )
        print(
            f"  Resultado:   {pred:<10}  "
            f"Local {p_loc:.1f}%  /  Empate {p_emp:.1f}%  /  Visitante {p_vis:.1f}%"
        )
        print(
            f"  Goles esp.:  {_fmt(ex['goles_esperados'])}   "
            f"Over 2.5: {_fmt(ex['over_2_5_pct'], 0, '%')}   "
            f"BTTS: {_fmt(ex['btts_pct'], 0, '%')}   "
            f"Marcador: {ex['marcador_exacto']} ({_fmt(ex['marcador_prob_pct'], 0, '%')})"
        )
        print(
            f"  Amarillas:   L {_fmt(ex['amarillas_local'])} / "
            f"V {_fmt(ex['amarillas_visitante'])} / "
            f"Total {_fmt(am_total)}    "
            f"Esquinas: L {_fmt(ex['esquinas_local'])} / "
            f"V {_fmt(ex['esquinas_visitante'])} / "
            f"Total {_fmt(es_total)}"
        )
        print(
            f"  Posesión:    L {_fmt(ex['posesion_local_pct'], 0, '%')} / "
            f"V {_fmt(ex['posesion_visitante_pct'], 0, '%')}    "
            f"Tiros puerta: L {_fmt(ex['tiros_puerta_local'])} / "
            f"V {_fmt(ex['tiros_puerta_visitante'])} / "
            f"Total {_fmt(tp_total)}    "
            f"Ocasiones: L {_fmt(ex['ocasiones_local'], 0)} / "
            f"V {_fmt(ex['ocasiones_visitante'], 0)}"
        )
        print(f"  Goleador L:  {ex['goleador_local'][:40]}")
        print(f"  Goleador V:  {ex['goleador_visitante'][:40]}")

        output_rows.append({
            "fecha":                  info["date"].strftime("%Y-%m-%d"),
            "liga":                   info["league"],
            "local":                  info["home_team"],
            "visitante":              info["away_team"],
            "prediccion":             pred,
            "p_local_%":              round(p_loc, 1),
            "p_empate_%":             round(p_emp, 1),
            "p_visitante_%":          round(p_vis, 1),
            "goles_esperados":        ex["goles_esperados"],
            "lambda_local":           ex["lambda_home"],
            "lambda_visitante":       ex["lambda_away"],
            "over_2_5_%":             ex["over_2_5_pct"],
            "btts_%":                 ex["btts_pct"],
            "marcador_exacto":        ex["marcador_exacto"],
            "marcador_exacto_%":      ex["marcador_prob_pct"],
            "amarillas_local":        ex["amarillas_local"],
            "amarillas_visitante":    ex["amarillas_visitante"],
            "amarillas_total":        am_total,
            "esquinas_local":         ex["esquinas_local"],
            "esquinas_visitante":     ex["esquinas_visitante"],
            "esquinas_total":         es_total,
            "posesion_local_%":       ex["posesion_local_pct"],
            "posesion_visitante_%":   ex["posesion_visitante_pct"],
            "tiros_puerta_local":     ex["tiros_puerta_local"],
            "tiros_puerta_visitante": ex["tiros_puerta_visitante"],
            "tiros_puerta_total":     tp_total,
            "ocasiones_local":        ex["ocasiones_local"],
            "ocasiones_visitante":    ex["ocasiones_visitante"],
            "goleador_local":      ex["goleador_local"],
            "goleador_visitante":  ex["goleador_visitante"],
        })

    print(SEP)
    print(f"\n  Modelos: XGBoost pre-partido (resultado) + Poisson independiente (goles)")
    print(f"  Fuentes: Understat (xG/forma), football-data.org (amarillas), player_stats (goleadores)")

    # ── Guardar CSV ──
    out_path = Path("data/predicciones_proximas.csv")
    pd.DataFrame(output_rows).to_csv(out_path, index=False)
    print(f"\n  CSV guardado en: {out_path}")
    print(f"  Columnas: {len(pd.DataFrame(output_rows).columns)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir próximos partidos de fútbol")
    parser.add_argument(
        "--days",   type=int, default=7,
        help="Días hacia adelante a considerar (default: 7)"
    )
    parser.add_argument(
        "--league", type=str, default=None,
        help="Filtrar por liga, ej: 'Premier', 'LaLiga', 'Serie A'"
    )
    args = parser.parse_args()
    main(days_ahead=args.days, league_filter=args.league)
