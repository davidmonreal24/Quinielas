"""
=====================================================================
  Feature Engineering — Soccer Project
  Input:  data/processed/schedule_xg.parquet
          data/processed/player_season_stats.parquet
  Output: data/processed/features.parquet
  Uso:    python feature_engineering.py
=====================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


# ─────────────────────────────────────────────
# 1. TARGET: RESULTADO DEL PARTIDO
# ─────────────────────────────────────────────

def add_result_target(df):
    """Añade columna resultado: 1=Local, 0=Empate, -1=Visitante."""
    df = df.copy()
    df["resultado"] = np.where(
        df["home_goals"] > df["away_goals"],  1,
        np.where(df["home_goals"] < df["away_goals"], -1, 0),
    )
    df["resultado_label"] = df["resultado"].map({1: "Local", 0: "Empate", -1: "Visitante"})
    return df


# ─────────────────────────────────────────────
# 2. DIFERENCIA DE XG (features con xG in-match)
# ─────────────────────────────────────────────

def add_xg_diff(df):
    """xg_diff, xg_total, xg_ratio_local — solo para el modelo retrospectivo."""
    df = df.copy()
    df["xg_diff"]        = df["home_xg"] - df["away_xg"]
    df["xg_total"]       = df["home_xg"] + df["away_xg"]
    df["xg_ratio_local"] = df["home_xg"] / (df["xg_total"] + 1e-6)
    return df


# ─────────────────────────────────────────────
# 3. FORMA RECIENTE (rolling xG — sin data leakage)
# ─────────────────────────────────────────────

def add_rolling_form(df, window=5):
    """
    Rolling xG for/against (últimos `window` partidos antes del partido actual).
    Vectorizado con merge_asof para eficiencia — no itera fila por fila.
    """
    df = df.sort_values("date").copy()
    played = df[df["is_result"] == True][
        ["date", "home_team", "away_team", "home_xg", "away_xg"]
    ].copy()

    home_g = played.rename(columns={
        "home_team": "team", "home_xg": "xg_for", "away_xg": "xg_against",
    })[["date", "team", "xg_for", "xg_against"]]

    away_g = played.rename(columns={
        "away_team": "team", "away_xg": "xg_for", "home_xg": "xg_against",
    })[["date", "team", "xg_for", "xg_against"]]

    tl = pd.concat([home_g, away_g]).sort_values(["team", "date"]).reset_index(drop=True)

    # shift(1): el partido actual no se incluye en su propio rolling
    tl["roll_for"] = tl.groupby("team")["xg_for"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )
    tl["roll_against"] = tl.groupby("team")["xg_against"].transform(
        lambda x: x.shift(1).rolling(window, min_periods=1).mean()
    )

    snap = (
        tl.sort_values("date")
        .drop_duplicates(["team", "date"], keep="last")
        [["date", "team", "roll_for", "roll_against"]]
    )

    df = pd.merge_asof(
        df.sort_values("date"),
        snap.rename(columns={
            "team": "home_team",
            "roll_for": "home_roll_xg_for",
            "roll_against": "home_roll_xg_against",
        }).sort_values("date"),
        on="date", by="home_team", direction="backward",
    )
    df = pd.merge_asof(
        df,
        snap.rename(columns={
            "team": "away_team",
            "roll_for": "away_roll_xg_for",
            "roll_against": "away_roll_xg_against",
        }).sort_values("date"),
        on="date", by="away_team", direction="backward",
    )

    df["form_xg_diff"] = df["home_roll_xg_for"]     - df["away_roll_xg_for"]
    df["form_def_diff"] = df["away_roll_xg_against"] - df["home_roll_xg_against"]
    return df


# ─────────────────────────────────────────────
# 4. CALIDAD DE PLANTILLA (desde player stats)
# ─────────────────────────────────────────────

def add_squad_quality(df, players):
    """
    np_xg, xa, xg_chain por partido para cada equipo.
    Busca la temporada exacta; si no existe, usa la más reciente del equipo en esa liga.
    """
    df = df.copy()

    cache = {}
    def squad_stats(team, season, league):
        key = (team, season, league)
        if key in cache:
            return cache[key]
        s = players[
            (players["team"] == team) & (players["season"] == season)
            & (players["league"] == league)
        ]
        if s.empty:
            s = players[(players["team"] == team) & (players["league"] == league)]
        if s.empty:
            result = (np.nan, np.nan, np.nan)
        else:
            result = (
                float(s["np_xg"].sum()) / 19,
                float(s["xa"].sum())    / 19,
                float(s["xg_chain"].sum()) / 19,
            )
        cache[key] = result
        return result

    stats = df.apply(
        lambda r: pd.Series({
            "h_npxg": squad_stats(r["home_team"], r["season"], r["league"])[0],
            "h_xa":   squad_stats(r["home_team"], r["season"], r["league"])[1],
            "h_ch":   squad_stats(r["home_team"], r["season"], r["league"])[2],
            "a_npxg": squad_stats(r["away_team"], r["season"], r["league"])[0],
            "a_xa":   squad_stats(r["away_team"], r["season"], r["league"])[1],
            "a_ch":   squad_stats(r["away_team"], r["season"], r["league"])[2],
        }), axis=1,
    )

    df["home_squad_np_xg"]    = stats["h_npxg"]
    df["home_squad_xa"]       = stats["h_xa"]
    df["home_squad_xg_chain"] = stats["h_ch"]
    df["away_squad_np_xg"]    = stats["a_npxg"]
    df["away_squad_xa"]       = stats["a_xa"]
    df["away_squad_xg_chain"] = stats["a_ch"]
    df["squad_xg_diff"]       = df["home_squad_np_xg"] - df["away_squad_np_xg"]
    df["squad_xa_diff"]       = df["home_squad_xa"]    - df["away_squad_xa"]
    return df


# ─────────────────────────────────────────────
# 5. VENTAJA DE LOCAL (feature binaria)
# ─────────────────────────────────────────────

def add_home_advantage(df):
    df = df.copy()
    df["es_local"] = 1
    return df


# ─────────────────────────────────────────────
# 6. POSICIÓN EN TABLA (sin data leakage)
# ─────────────────────────────────────────────

def add_table_position(df):
    """
    Posición acumulada ANTES del partido en cada liga-temporada.
    Solo usa partidos anteriores a la fecha actual.
    """
    df = df.sort_values("date").copy()
    played = df[df["is_result"] == True].copy()

    played["home_pts"] = np.where(
        played["home_goals"] > played["away_goals"], 3,
        np.where(played["home_goals"] == played["away_goals"], 1, 0),
    )
    played["away_pts"] = np.where(
        played["away_goals"] > played["home_goals"], 3,
        np.where(played["away_goals"] == played["home_goals"], 1, 0),
    )

    hp = played[["date", "league", "season", "home_team", "home_pts"]].rename(
        columns={"home_team": "team", "home_pts": "pts"})
    ap = played[["date", "league", "season", "away_team", "away_pts"]].rename(
        columns={"away_team": "team", "away_pts": "pts"})

    tp = (pd.concat([hp, ap])
          .sort_values(["league", "season", "team", "date"])
          .reset_index(drop=True))

    # Puntos acumulados SIN incluir el partido actual (cumsum - pts_actual)
    tp["cum_pts"] = tp.groupby(["league", "season", "team"])["pts"].transform(
        lambda x: x.cumsum() - x
    )

    snap = (tp.drop_duplicates(["league", "season", "team", "date"], keep="last")
              [["date", "league", "season", "team", "cum_pts"]]
              .sort_values("date"))

    df = pd.merge_asof(
        df,
        snap.rename(columns={"team": "home_team", "cum_pts": "home_cum_pts"}),
        on="date", by=["league", "season", "home_team"], direction="backward",
    )
    df = pd.merge_asof(
        df,
        snap.rename(columns={"team": "away_team", "cum_pts": "away_cum_pts"}),
        on="date", by=["league", "season", "away_team"], direction="backward",
    )

    # Rank within each (date, league, season) group
    # Efficient: compute rank from cum_pts across all teams that played in that season
    # Build a full snapshot of all teams' cum_pts per date
    all_snap = (snap.rename(columns={"team": "ref_team", "cum_pts": "ref_pts"})
                .drop_duplicates(["league", "season", "ref_team", "date"], keep="last"))

    home_pos = []
    away_pos = []
    home_pts_list = []
    away_pts_list = []

    for _, row in df.iterrows():
        d      = row["date"]
        season = row["season"]
        league = row["league"]
        home   = row["home_team"]
        away   = row["away_team"]
        h_pts  = row.get("home_cum_pts", np.nan)
        a_pts  = row.get("away_cum_pts", np.nan)
        home_pts_list.append(h_pts)
        away_pts_list.append(a_pts)

        # Rank within league-season table at this date
        tbl = all_snap[
            (all_snap["league"] == league) & (all_snap["season"] == season)
            & (all_snap["date"] <= d)
        ].sort_values("date").drop_duplicates("ref_team", keep="last")

        if tbl.empty:
            home_pos.append(np.nan)
            away_pos.append(np.nan)
            continue

        tbl_sorted = tbl.sort_values("ref_pts", ascending=False).reset_index(drop=True)
        pos_map = {r["ref_team"]: i + 1 for i, r in tbl_sorted.iterrows()}
        home_pos.append(pos_map.get(home, np.nan))
        away_pos.append(pos_map.get(away, np.nan))

    df["home_table_pos"] = home_pos
    df["away_table_pos"] = away_pos
    df["home_pts_acum"]  = home_pts_list
    df["away_pts_acum"]  = away_pts_list
    df["table_pos_diff"] = [
        a - h if not (pd.isna(a) or pd.isna(h)) else np.nan
        for a, h in zip(away_pos, home_pos)
    ]
    df["pts_diff"] = [
        h - a if not (pd.isna(h) or pd.isna(a)) else np.nan
        for h, a in zip(home_pts_list, away_pts_list)
    ]
    df = df.drop(columns=["home_cum_pts", "away_cum_pts"], errors="ignore")
    return df


# ─────────────────────────────────────────────
# 7. HEAD-TO-HEAD (sin data leakage)
# ─────────────────────────────────────────────

def add_head_to_head(df, window=5):
    """
    H2H histórico entre pares de equipos usando solo partidos anteriores a la fecha.
    Añade: h2h_matches, h2h_home_win_rate, h2h_draw_rate, h2h_away_win_rate, h2h_avg_goals.
    """
    df = df.sort_values("date").copy()
    played = df[df["is_result"] == True].copy()

    if "resultado" not in played.columns:
        played["resultado"] = np.where(
            played["home_goals"] > played["away_goals"],  1,
            np.where(played["home_goals"] < played["away_goals"], -1, 0),
        )

    h2h_matches_list       = []
    h2h_home_win_rate_list = []
    h2h_draw_rate_list     = []
    h2h_away_win_rate_list = []
    h2h_avg_goals_list     = []

    for _, row in df.iterrows():
        d    = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        mask = (
            (played["date"] < d)
            & (
                ((played["home_team"] == home) & (played["away_team"] == away))
                | ((played["home_team"] == away) & (played["away_team"] == home))
            )
        )
        h2h = played[mask].tail(window)
        n   = len(h2h)

        if n == 0:
            h2h_matches_list.append(0)
            h2h_home_win_rate_list.append(np.nan)
            h2h_draw_rate_list.append(np.nan)
            h2h_away_win_rate_list.append(np.nan)
            h2h_avg_goals_list.append(np.nan)
            continue

        home_wins = int(
            ((h2h["home_team"] == home) & (h2h["resultado"] == 1)).sum()
            + ((h2h["away_team"] == home) & (h2h["resultado"] == -1)).sum()
        )
        draws     = int((h2h["resultado"] == 0).sum())
        away_wins = n - home_wins - draws

        h2h_matches_list.append(n)
        h2h_home_win_rate_list.append(home_wins / n)
        h2h_draw_rate_list.append(draws / n)
        h2h_away_win_rate_list.append(away_wins / n)
        h2h_avg_goals_list.append(
            float((h2h["home_goals"] + h2h["away_goals"]).mean())
        )

    df["h2h_matches"]       = h2h_matches_list
    df["h2h_home_win_rate"] = h2h_home_win_rate_list
    df["h2h_draw_rate"]     = h2h_draw_rate_list
    df["h2h_away_win_rate"] = h2h_away_win_rate_list
    df["h2h_avg_goals"]     = h2h_avg_goals_list
    return df


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  SOCCER PROJECT — Feature Engineering")
    print("=" * 60)

    # ── Cargar datos base ──
    schedule = pd.read_parquet(DATA_DIR / "schedule_xg.parquet").reset_index()
    schedule["date"] = pd.to_datetime(schedule["date"])

    try:
        players = pd.read_parquet(DATA_DIR / "player_season_stats.parquet").reset_index()
        print(f"Players cargados: {len(players)} filas")
    except FileNotFoundError:
        players = pd.DataFrame()
        print("Aviso: player_season_stats.parquet no encontrado — squad quality = NaN")

    # Solo partidos jugados para el dataset de training
    df = schedule[schedule["is_result"] == True].copy()
    print(f"\nPartidos jugados cargados: {len(df)}")

    print("\n[1/7] Añadiendo resultado target...")
    df = add_result_target(df)

    print("[2/7] Añadiendo xG diff (features in-match)...")
    df = add_xg_diff(df)

    print("[3/7] Añadiendo rolling form (vectorizado)...")
    df = add_rolling_form(df)

    if not players.empty:
        print("[4/7] Añadiendo squad quality...")
        df = add_squad_quality(df, players)
    else:
        for col in ["home_squad_np_xg", "home_squad_xa", "home_squad_xg_chain",
                    "away_squad_np_xg", "away_squad_xa", "away_squad_xg_chain",
                    "squad_xg_diff", "squad_xa_diff"]:
            df[col] = np.nan
        print("[4/7] Squad quality = NaN (sin player_season_stats.parquet)")

    print("[5/7] Añadiendo home advantage...")
    df = add_home_advantage(df)

    print("[6/7] Añadiendo table position (puede tardar ~1-2 min)...")
    df = add_table_position(df)

    print("[7/7] Añadiendo head-to-head (puede tardar ~1-2 min)...")
    df = add_head_to_head(df)

    # ── Guardar ──
    output = DATA_DIR / "features.parquet"
    df.to_parquet(output, index=False)

    all_features = [
        "home_xg", "away_xg", "xg_diff", "xg_total", "xg_ratio_local",
        "home_roll_xg_for", "home_roll_xg_against",
        "away_roll_xg_for",  "away_roll_xg_against",
        "form_xg_diff", "form_def_diff",
        "home_squad_np_xg", "home_squad_xa", "home_squad_xg_chain",
        "away_squad_np_xg", "away_squad_xa", "away_squad_xg_chain",
        "squad_xg_diff", "squad_xa_diff",
        "home_table_pos", "away_table_pos", "table_pos_diff", "pts_diff",
        "es_local",
        "h2h_matches", "h2h_home_win_rate", "h2h_draw_rate",
        "h2h_away_win_rate", "h2h_avg_goals",
    ]
    available = [c for c in all_features if c in df.columns]
    fill_rates = {c: f"{df[c].notna().mean()*100:.1f}%" for c in available}

    print(f"\nGuardado: {output}")
    print(f"Shape: {df.shape}")
    print(f"\nCobertura de features:")
    for feat, rate in fill_rates.items():
        print(f"  {feat:<30} {rate}")
