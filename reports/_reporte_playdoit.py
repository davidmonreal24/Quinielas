"""
Reporte Liga MX — Momios Playdoit en tiempo real
=================================================
Muestra predicciones del modelo vs. momios de Playdoit con tabla clara.

Columnas de la tabla PLAYDOIT:
  Momio    : momio decimal (cuánto paga Playdoit por cada $1 + la ganancia)
  Casas%   : probabilidad implícita SIN vigorish (lo que el mercado cree)
  Modelo%  : probabilidad de nuestro modelo Dixon-Coles
  Edge     : Modelo% − Casas%  (positivo = ventaja sobre el mercado)
  EV       : Valor Esperado = (P_modelo × momio) − 1  (>0 = rentable a largo plazo)

Uso:
  python _reporte_playdoit.py
  python _reporte_playdoit.py --days 7
  python _reporte_playdoit.py --match "Chivas"
"""
import argparse
import requests
import re
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

import pandas as pd

# ─────────────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────────────
BASE   = "https://sb2frontend-altenar2.biahosted.com/api/widget"
PARAMS = {
    "culture": "es-ES", "timezoneOffset": "360", "integration": "playdoit2",
    "lang": "es-ES", "timezone": "America/Mexico_City",
    "champIds": "10009", "sportId": "66",
}
CDT = timezone(timedelta(hours=-6))
W   = 76

MONTH_ES = {
    1:"ENERO", 2:"FEBRERO", 3:"MARZO", 4:"ABRIL", 5:"MAYO", 6:"JUNIO",
    7:"JULIO", 8:"AGOSTO", 9:"SEPTIEMBRE", 10:"OCTUBRE", 11:"NOVIEMBRE", 12:"DICIEMBRE",
}
DAY_ES = {
    "Mon":"LUNES", "Tue":"MARTES", "Wed":"MIERCOLES", "Thu":"JUEVES",
    "Fri":"VIERNES", "Sat":"SABADO", "Sun":"DOMINGO",
}


# ─────────────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────────────

def _dia_nombre(dia_key: str) -> str:
    """'Fri 25/04' → 'VIERNES 25 DE ABRIL'"""
    try:
        day_abbr, date_part = dia_key.split(" ", 1)
        day_num, month_num = date_part.split("/")
        return (f"{DAY_ES.get(day_abbr, dia_key.upper())} "
                f"{int(day_num)} DE {MONTH_ES[int(month_num)]}")
    except Exception:
        return dia_key.upper()


def _norm(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(fc|cf|club|cd|unam|uanl|deportivo|atletico)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _no_vig(oh: float, od: float, oa: float) -> tuple[float, float, float]:
    ph, pd_, pa = 1/oh, 1/od, 1/oa
    t = ph + pd_ + pa
    return ph/t * 100, pd_/t * 100, pa/t * 100


def _vig(oh: float, od: float, oa: float) -> float:
    return round((1/oh + 1/od + 1/oa - 1) * 100, 2)


def _ev(p_pct: float, o: float) -> float:
    return round((p_pct/100 * (o - 1) - (1 - p_pct/100)) * 100, 1)


def _prob_bar(p: float, width: int = 14) -> str:
    filled = round(p / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ─────────────────────────────────────────────────────
# 3. FETCH PLAYDOIT
# ─────────────────────────────────────────────────────

def fetch_playdoit() -> list[dict]:
    r = requests.get(f"{BASE}/GetUpcoming", params=PARAMS, timeout=15)
    data        = r.json()
    markets     = {m["id"]: m for m in data["markets"]}
    odds_map    = {o["id"]: o for o in data["odds"]}
    competitors = {c["id"]: c for c in data["competitors"]}
    result = []
    for ev in data["events"]:
        if ev.get("champId") != 10009:
            continue
        start = (datetime.fromisoformat(ev["startDate"].replace("Z", ""))
                 .replace(tzinfo=timezone.utc).astimezone(CDT))
        comps = [competitors.get(cid, {}).get("name", "?") for cid in ev.get("competitorIds", [])]
        if len(comps) < 2:
            continue
        h12 = next((markets[mid] for mid in ev["marketIds"]
                    if markets.get(mid, {}).get("name") == "1x2"), None)
        if not h12:
            continue
        os_ = [odds_map.get(oid) for oid in h12["oddIds"]]
        result.append({
            "home": comps[0], "away": comps[1],
            "dia":  start.strftime("%a %d/%m"),
            "hora": start.strftime("%H:%M"),
            "o_h":  next((o["price"] for o in os_ if o and o.get("typeId") == 1), None),
            "o_d":  next((o["price"] for o in os_ if o and o.get("typeId") == 2), None),
            "o_a":  next((o["price"] for o in os_ if o and o.get("typeId") == 3), None),
        })
    return result


def _find_playdoit(home: str, away: str, pd_list: list[dict]) -> dict | None:
    hn, an = _norm(home), _norm(away)
    best, bscore = None, 0.50
    for item in pd_list:
        sh = SequenceMatcher(None, hn, _norm(item["home"])).ratio()
        sa = SequenceMatcher(None, an, _norm(item["away"])).ratio()
        s  = (sh + sa) / 2
        if s > bscore:
            bscore, best = s, item
    return best


# ─────────────────────────────────────────────────────
# 4. FUNCIONES DE DISPLAY
# ─────────────────────────────────────────────────────

def _print_odds_table(ph: float, pd_: float, pv: float,
                      oh: float, od: float, oa: float) -> list[tuple]:
    """
    Tabla de momios con columnas etiquetadas.
    Devuelve lista de value bets: [(outcome, momio, edge, ev), ...]
    """
    vig = _vig(oh, od, oa)
    mh, md, ma = _no_vig(oh, od, oa)
    e = [round(ph - mh, 1), round(pd_ - md, 1), round(pv - ma, 1)]
    ev = [_ev(ph, oh), _ev(pd_, od), _ev(pv, oa)]

    print(f"  PLAYDOIT  vig={vig:.1f}%")
    # Cabecera de columnas
    print(f"  {'':11} {'Momio':>8}  {'Casas%':>7}  {'Modelo%':>8}  {'Edge':>7}  {'EV':>7}  Señal")
    print(f"  {'─'*70}")

    senal = []
    for lbl, o_val, casas, modelo, ei, evi in zip(
        ["Local", "Empate", "Visitante"],
        [oh, od, oa], [mh, md, ma], [ph, pd_, pv], e, ev
    ):
        if ei >= 8:
            s = "◄◄ VALOR ALTO"
        elif ei >= 5:
            s = "◄ VALOR"
        elif ei >= 3:
            s = "leve+"
        elif ei >= -3:
            s = "neutro"
        else:
            s = "sin edge"
        senal.append(s)
        print(f"  {lbl:<11} {o_val:>8.4f}  {casas:>6.1f}%  {modelo:>7.1f}%  "
              f"{ei:>+6.1f}%  {evi:>+6.1f}%  {s}")

    print(f"  {'─'*70}")

    vbets = [(lbl, o_val, ei, evi)
             for lbl, o_val, ei, evi in zip(
                 ["Local", "Empate", "Visitante"], [oh, od, oa], e, ev)
             if ei >= 5]
    return vbets


def _print_scorers(home: str, away: str, row: pd.Series) -> None:
    """Muestra goleadores top de cada equipo con el líder marcado."""
    def _fmt(raw: str, team: str) -> None:
        if not raw or str(raw).startswith("N/D"):
            print(f"    (sin datos de goleadores para este torneo)")
            return
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        for i, p in enumerate(parts[:3]):
            if i == 0:
                print(f"    ⚽ {p}  ← AMENAZA CLAVE")
            else:
                print(f"       {p}")

    gol_h = row.get("Goleadores Local (goleadores_local)", "N/D")
    gol_a = row.get("Goleadores Visitante (goleadores_visita)", "N/D")
    print(f"  GOLEADORES")
    print(f"  {home[:24]}:")
    _fmt(gol_h, home)
    print(f"  {away[:24]}:")
    _fmt(gol_a, away)


def _print_cards_corners(home: str, away: str, row: pd.Series) -> None:
    """Corners y tarjetas en una línea cada uno."""
    try:
        ch  = row.get("Corners Predichos Local (corners_h)")
        ca  = row.get("Corners Predichos Visitante (corners_a)")
        ct  = row.get("Corners Total Predichos (corners_total)")
        ah  = row.get("Amarillas Promedio Local (amarillas_local)")
        aa  = row.get("Amarillas Promedio Visitante (amarillas_visita)")
        at  = row.get("Amarillas Total Estimadas (amarillas_total)")
        src = row.get("Fuente Corners (FBref o lambda)", "")

        if pd.notna(ch) and pd.notna(ca):
            src_tag = f"  [{src}]" if src else ""
            print(f"  CORNERS    {home[:17]} {float(ch):.1f}  vs  "
                  f"{away[:17]} {float(ca):.1f}   Total ~{float(ct):.1f}{src_tag}")

        if pd.notna(ah) and pd.notna(aa):
            alert = ""
            if float(ah) + float(aa) > 4.5:
                alert = "  ⚠ partido caliente (>4.5 tarjetas prom)"
            at_s = f"~{float(at):.1f}" if pd.notna(at) else "?"
            print(f"  AMARILLAS  {home[:17]} {float(ah):.1f}/p  vs  "
                  f"{away[:17]} {float(aa):.1f}/p   Total {at_s}{alert}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte Playdoit — Liga MX")
    ap.add_argument("--days",  type=int, default=10,
                    help="Días hacia adelante (default 10)")
    ap.add_argument("--match", default="",
                    help="Filtrar por nombre de equipo (ej. 'Chivas')")
    args = ap.parse_args()

    now     = datetime.now(CDT)
    today   = now.strftime("%Y-%m-%d")
    horizon = (now + timedelta(days=args.days)).strftime("%Y-%m-%d")

    # ── Cargar predicciones ──
    df     = pd.read_csv("data/ligamx_predicciones.csv")
    ligamx = df[df["Fecha"].between(today, horizon)].copy()
    if args.match:
        m = args.match.lower()
        ligamx = ligamx[
            ligamx["Equipo Local"].str.lower().str.contains(m) |
            ligamx["Equipo Visitante"].str.lower().str.contains(m)
        ]

    # ── Fetch momios Playdoit ──
    print(f"\n  Obteniendo momios Playdoit...", end=" ")
    try:
        pd_list = fetch_playdoit()
        print(f"{len(pd_list)} partidos encontrados")
    except Exception as e:
        print(f"ERROR ({e})")
        pd_list = []

    # ── Encabezado ──
    print()
    print("=" * W)
    print(f"  LIGA MX CLAUSURA 2026  |  {today} → {horizon}")
    print(f"  Modelo: Dixon-Coles + Poisson  |  Odds: Playdoit en tiempo real")
    print(f"  Columnas: Momio | Casas%(sin vig) | Modelo% | Edge | EV")
    print("=" * W)

    value_bets = []
    dia_actual = ""

    for _, row in ligamx.sort_values("Fecha").iterrows():
        home  = str(row["Equipo Local"])
        away  = str(row["Equipo Visitante"])
        ph    = float(row["Probabilidad Local % (p_local)"])
        pde   = float(row["Probabilidad Empate % (p_empate)"])
        pv    = float(row["Probabilidad Visitante % (p_visit)"])
        pred  = str(row["Prediccion"])
        conf  = str(row["Nivel de Confianza (ALTA / MEDIA / BAJA)"])
        lh    = float(row["Goles Esperados Local (lambda_h)"])
        la    = float(row["Goles Esperados Visitante (lambda_a)"])
        pos_h = int(row["Posicion Tabla Local (pos_h)"])
        pos_a = int(row["Posicion Tabla Visitante (pos_v)"])
        pts_h = int(row["Puntos en Tabla Local (pts_tabla_h)"])
        pts_a = int(row["Puntos en Tabla Visitante (pts_tabla_v)"])
        forma_h = str(row.get("Forma Reciente Local W/D/L (forma_h)", ""))
        forma_a = str(row.get("Forma Reciente Visitante W/D/L (forma_a)", ""))
        mot_h = str(row.get("Motivacion Local (motivacion_local)", ""))
        mot_a = str(row.get("Motivacion Visitante (motivacion_visita)", ""))

        pd_item = _find_playdoit(home, away, pd_list)
        dia_key = pd_item["dia"] if pd_item else None
        hora    = pd_item["hora"] if pd_item else "?"
        dia_nom = _dia_nombre(dia_key) if dia_key else ""

        if dia_nom and dia_nom != dia_actual:
            dia_actual = dia_nom
            print()
            print(f"  ── {dia_nom} {'─' * max(0, W - 6 - len(dia_nom))}")

        # ── Cabecera del partido ──
        print()
        print(f"  {hora} CDT  │  {home}  vs  {away}")
        print(f"  Tabla:  #{pos_h} {home[:18]} ({pts_h}pts)   vs   #{pos_a} {away[:18]} ({pts_a}pts)")
        if forma_h or forma_a:
            print(f"  Forma:  {home[:18]} [{forma_h}]  vs  {away[:18]} [{forma_a}]")
        if mot_h and not mot_h.startswith("Zona Media"):
            print(f"  Motiv:  {home[:18]}: {mot_h[:35]}   |  {away[:18]}: {mot_a[:35]}")

        # ── Predicción del modelo ──
        print(f"  MODELO → {pred}  Conf={conf}  λ {lh:.2f} vs {la:.2f}")
        print(f"  Local {ph:.1f}% {_prob_bar(ph)}  "
              f"Empate {pde:.1f}% {_prob_bar(pde, 10)}  "
              f"Visitante {pv:.1f}%")

        # ── Corners y tarjetas ──
        _print_cards_corners(home, away, row)

        # ── Goleadores ──
        _print_scorers(home, away, row)

        # ── Tabla de momios Playdoit ──
        print()
        if pd_item and all([pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]]):
            vbets = _print_odds_table(ph, pde, pv,
                                      pd_item["o_h"], pd_item["o_d"], pd_item["o_a"])
            if vbets:
                print("  ★ VALUE BETS DETECTADAS:")
                for outcome, momio, edge, ev in vbets:
                    line = f"  → {outcome} @ {momio:.4f}   Edge={edge:+.1f}%   EV={ev:+.1f}%"
                    print(line)
                    value_bets.append(
                        f"{dia_nom} {hora}  {home} vs {away}  {line.strip()}"
                    )
            else:
                print("  (sin edge significativo ≥5% — línea en precio justo)")
        else:
            print("  [sin momios en Playdoit para este partido]")

    # ── Resumen ──
    print()
    print("=" * W)
    print(f"  RESUMEN DE PICKS  {today}")
    print(f"  {'HORA':>6}  {'LOCAL':<22}  {'VISITANTE':<22}  {'PICK':<12}  {'PROB':>5}  {'MOMIO':<10}  CONF")
    print(f"  {'─'*70}")
    dia_actual = ""
    for _, row in ligamx.sort_values("Fecha").iterrows():
        home  = str(row["Equipo Local"])
        away  = str(row["Equipo Visitante"])
        ph    = float(row["Probabilidad Local % (p_local)"])
        pde   = float(row["Probabilidad Empate % (p_empate)"])
        pv    = float(row["Probabilidad Visitante % (p_visit)"])
        pred  = str(row["Prediccion"])
        conf  = str(row["Nivel de Confianza (ALTA / MEDIA / BAJA)"])
        pmax  = max(ph, pde, pv)
        pd_item = _find_playdoit(home, away, pd_list)
        hora    = pd_item["hora"] if pd_item else "?"
        dia_key = pd_item["dia"]  if pd_item else None
        dia_nom = _dia_nombre(dia_key) if dia_key else ""
        if dia_nom and dia_nom != dia_actual:
            dia_actual = dia_nom
            print(f"\n  -- {dia_nom} --")
        momio_s = ""
        if pd_item and all([pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]]):
            m = (pd_item["o_h"] if pred == "Local"
                 else (pd_item["o_a"] if pred == "Visitante" else pd_item["o_d"]))
            momio_s = f"@ {m:.4f}"
        print(f"  {hora:>6}  {home:<22}  {away:<22}  {pred:<12}  {pmax:.0f}%  {momio_s:<10}  [{conf}]")

    if value_bets:
        print()
        print("=" * W)
        print("  VALUE BETS  (Edge ≥ 5%)")
        print("=" * W)
        for vb in value_bets:
            print(f"  {vb}")

    # ── Glosario ──
    print()
    print("=" * W)
    print("  GLOSARIO DE COLUMNAS")
    print("  Momio    : cuánto paga Playdoit (ej. 2.50 = ganas $1.50 por $1 apostado)")
    print("  Casas%   : prob. implícita que asigna el mercado, quitando el margen (vig)")
    print("  Modelo%  : prob. que calcula nuestro modelo Dixon-Coles + Poisson")
    print("  Edge     : Modelo% − Casas%  (si >0, el mercado subestima esa opción)")
    print("  EV       : (P_modelo × momio) − 1  (>0% = ganancia esperada a largo plazo)")
    print("  ─────────────────────────────────────────────────────────────────────────")
    print("  Edge ≥ 5%: valor detectado  |  EV > 0%: rentable en muestra grande")
    print("  Nunca apostar más de lo que puedes perder. El modelo no garantiza aciertos.")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
