"""
backtesting.py — Análisis dinámico por jornada
================================================
Uso:
  python analysis/backtesting.py --jornada 15
  python analysis/backtesting.py --jornada 16 --csv data/ligamx_predicciones_j16.csv
  python analysis/backtesting.py --list       # ver jornadas disponibles
"""
import sys
import math
import argparse
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS REALES — agregar bloque por cada jornada jugada
# Formato: (local, visitante, goles_local, goles_visitante, fecha_YYYY-MM-DD)
# ─────────────────────────────────────────────────────────────────────────────
RESULTADOS: dict[int, list[tuple]] = {
    15: [
        ("Atlético San Luis", "Pumas UNAM",    0, 2, "2026-04-17"),
        ("Mazatlán FC",       "Querétaro FC",  1, 1, "2026-04-17"),
        ("Cruz Azul",         "Club Tijuana",  1, 1, "2026-04-18"),
        ("Club Necaxa",       "Tigres UANL",   1, 1, "2026-04-18"),
        ("CF Monterrey",      "CF Pachuca",    1, 3, "2026-04-18"),
        ("CD Guadalajara",    "Club Puebla",   5, 0, "2026-04-18"),
        ("Club América",      "CD Toluca",     2, 1, "2026-04-18"),
        ("Club León",         "FC Juárez",     3, 1, "2026-04-19"),
        ("Santos Laguna",     "Atlas FC",      0, 1, "2026-04-19"),
    ],
    # ── Jornada 16 — añadir aquí después de jugarse ──
    # 16: [
    #     ("Equipo A", "Equipo B", gL, gV, "2026-04-25"),
    # ],
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    import re
    s = str(s).lower().strip()
    s = re.sub(r"\b(fc|cf|cd|club|atletico|de)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _find_pred(home: str, away: str, df: pd.DataFrame, cols: dict) -> pd.Series | None:
    from difflib import SequenceMatcher
    hn, an = _norm(home), _norm(away)
    best_score, best_row = 0.0, None
    for _, row in df.iterrows():
        sh = SequenceMatcher(None, hn, _norm(row[cols["home"]])).ratio()
        sa = SequenceMatcher(None, an, _norm(row[cols["away"]])).ratio()
        s = (sh + sa) / 2
        if s > best_score:
            best_score, best_row = s, row
    return best_row if best_score > 0.55 else None


def _outcome(hg: int, ag: int) -> str:
    return "Local" if hg > ag else ("Empate" if hg == ag else "Visitante")


# ─────────────────────────────────────────────────────────────────────────────
# COLUMNAS DEL CSV (mapeo nombre largo → clave corta)
# ─────────────────────────────────────────────────────────────────────────────
CSV_COLS = {
    "home": "Equipo Local",
    "away": "Equipo Visitante",
    "pred": "Prediccion",
    "ph":   "Probabilidad Local % (p_local)",
    "pd_":  "Probabilidad Empate % (p_empate)",
    "pv":   "Probabilidad Visitante % (p_visit)",
    "lh":   "Goles Esperados Local (lambda_h)",
    "la":   "Goles Esperados Visitante (lambda_a)",
    "conf": "Nivel de Confianza (ALTA / MEDIA / BAJA)",
    "fecha":"Fecha",
}


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
def run_backtesting(jornada: int, csv_path: Path) -> pd.DataFrame:
    resultados = RESULTADOS.get(jornada)
    if not resultados:
        print(f"ERROR: No hay resultados registrados para Jornada {jornada}.")
        print(f"  Jornadas disponibles: {sorted(RESULTADOS.keys())}")
        sys.exit(1)

    df_pred = pd.read_csv(csv_path)
    fechas = [r[4] for r in resultados]
    mask = df_pred[CSV_COLS["fecha"]].between(min(fechas), max(fechas))
    preds = df_pred[mask].copy()

    records = []
    for home, away, hg, ag, fecha in resultados:
        real = _outcome(hg, ag)
        row = _find_pred(home, away, preds, CSV_COLS)
        if row is None:
            print(f"  AVISO: sin predicción para {home} vs {away}")
            continue

        ph  = float(row[CSV_COLS["ph"]])  / 100
        pd_ = float(row[CSV_COLS["pd_"]]) / 100
        pv  = float(row[CSV_COLS["pv"]])  / 100
        lh  = float(row[CSV_COLS["lh"]])
        la  = float(row[CSV_COLS["la"]])
        pred = row[CSV_COLS["pred"]]
        conf = row[CSV_COLS["conf"]]

        p_correct = {"Local": ph, "Empate": pd_, "Visitante": pv}[real]
        oh = float(real == "Local")
        od = float(real == "Empate")
        ov = float(real == "Visitante")
        brier    = (ph - oh)**2 + (pd_ - od)**2 + (pv - ov)**2
        log_loss = -math.log(max(p_correct, 1e-7))

        records.append({
            "local": home, "visitante": away, "fecha": fecha,
            "goles_l": hg, "goles_v": ag,
            "resultado_real": real, "prediccion": pred,
            "correcto": (pred == real), "confianza": conf,
            "ph": ph, "pd": pd_, "pv": pv, "lh": lh, "la": la,
            "p_correcto": round(p_correct, 4),
            "brier": round(brier, 4),
            "log_loss": round(log_loss, 4),
        })

    return pd.DataFrame(records)


def print_report(df: pd.DataFrame, jornada: int) -> None:
    n   = len(df)
    acc = df["correcto"].mean()
    bs  = df["brier"].mean()
    ll  = df["log_loss"].mean()
    W   = 72

    print()
    print("=" * W)
    print(f"  BACKTESTING J{jornada} — Liga MX Clausura 2026")
    print("=" * W)
    print(f"  Partidos analizados : {n}")
    print(f"  Accuracy            : {acc*100:.1f}%  ({df['correcto'].sum()}/{n} correctos)")
    print(f"  Brier Score         : {bs:.4f}  [0=perfecto | 0.667=aleatorio | ~0.25=bueno]")
    print(f"  Log Loss            : {ll:.4f}  [0=perfecto | 1.099=aleatorio]")

    # Detalle por partido
    print()
    print(f"  {'Partido':<36} {'Real':<10} {'Pred':<10} {'P(real)':<8} {'Brier':<7} {'OK?'}")
    print(f"  {'-'*68}")
    for _, r in df.iterrows():
        partido = f"{r['local'][:17]} vs {r['visitante'][:14]}"
        ok = "[OK]" if r["correcto"] else "[--]"
        print(f"  {partido:<36} {r['resultado_real']:<10} {r['prediccion']:<10} "
              f"{r['p_correcto']*100:>5.1f}%   {r['brier']:.3f}   {ok}")

    # Sesgo
    print()
    print(f"  {'─'*68}")
    print("  SESGO")
    pred_c = df["prediccion"].value_counts()
    real_c = df["resultado_real"].value_counts()
    print(f"  Predicciones → Local={pred_c.get('Local',0)}  Empate={pred_c.get('Empate',0)}  Visitante={pred_c.get('Visitante',0)}")
    print(f"  Resultados   → Local={real_c.get('Local',0)}  Empate={real_c.get('Empate',0)}  Visitante={real_c.get('Visitante',0)}")
    for outcome, col in [("Local","ph"), ("Empate","pd"), ("Visitante","pv")]:
        avg_p     = df[col].mean() * 100
        real_freq = (df["resultado_real"] == outcome).mean() * 100
        bias      = avg_p - real_freq
        print(f"  {outcome:<12}: modelo {avg_p:.1f}%  |  ocurrió {real_freq:.1f}%  |  sesgo {bias:+.1f}pp")

    # Error lambdas
    print()
    print(f"  {'─'*68}")
    print("  ERROR EN LAMBDAS")
    lh_err  = df["lh"] - df["goles_l"]
    la_err  = df["la"] - df["goles_v"]
    all_err = pd.concat([lh_err, la_err])
    mae  = all_err.abs().mean()
    rmse = math.sqrt((all_err**2).mean())
    print(f"  MAE  : {mae:.3f} goles  |  RMSE: {rmse:.3f} goles")
    print(f"  Sesgo λ_local   : {lh_err.mean():+.3f}  |  Sesgo λ_visita: {la_err.mean():+.3f}")
    print()

    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting dinámico por jornada")
    parser.add_argument("--jornada", type=int, help="Número de jornada a analizar")
    parser.add_argument("--csv",     type=str, default="data/ligamx_predicciones.csv",
                        help="Ruta al CSV de predicciones (default: data/ligamx_predicciones.csv)")
    parser.add_argument("--save",    action="store_true",
                        help="Guardar resultados en data/backtesting_j{N}.csv")
    parser.add_argument("--list",    action="store_true",
                        help="Listar jornadas disponibles")
    args = parser.parse_args()

    if args.list:
        print(f"Jornadas disponibles: {sorted(RESULTADOS.keys())}")
        sys.exit(0)

    if not args.jornada:
        parser.print_help()
        sys.exit(1)

    csv_path = _ROOT / args.csv
    if not csv_path.exists():
        print(f"ERROR: No se encontró el CSV en {csv_path}")
        sys.exit(1)

    df = run_backtesting(args.jornada, csv_path)
    if df.empty:
        print("No se pudieron emparejar predicciones con resultados.")
        sys.exit(1)

    print_report(df, args.jornada)

    if args.save:
        out = _ROOT / f"data/backtesting_j{args.jornada}.csv"
        df.to_csv(out, index=False)
        print(f"  Guardado: {out}")
