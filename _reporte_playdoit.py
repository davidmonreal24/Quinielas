"""Reporte Liga MX Jornada 11 — Momios Playdoit en tiempo real"""
import requests, json, re
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
import pandas as pd

# ── 1. Fetch Playdoit odds via Altenar API ──
BASE = "https://sb2frontend-altenar2.biahosted.com/api/widget"
PARAMS = {
    "culture": "es-ES",
    "timezoneOffset": "360",
    "integration": "playdoit2",
    "lang": "es-ES",
    "timezone": "America/Mexico_City",
    "champIds": "10009",
    "sportId": "66",
}
r = requests.get(f"{BASE}/GetUpcoming", params=PARAMS, timeout=15)
data = r.json()

events      = data["events"]
markets     = {m["id"]: m for m in data["markets"]}
odds_map    = {o["id"]: o for o in data["odds"]}
competitors = {c["id"]: c for c in data["competitors"]}

CDT = timezone(timedelta(hours=-6))

playdoit_list = []
for ev in events:
    if ev.get("champId") != 10009:
        continue
    start = (
        datetime.fromisoformat(ev["startDate"].replace("Z", ""))
        .replace(tzinfo=timezone.utc)
        .astimezone(CDT)
    )
    comps = [competitors.get(cid, {}).get("name", "?") for cid in ev.get("competitorIds", [])]
    home, away = comps[0], comps[1]
    h12 = next(
        (markets[mid] for mid in ev["marketIds"] if markets.get(mid, {}).get("name") == "1x2"),
        None,
    )
    if not h12:
        continue
    os_ = [odds_map.get(oid) for oid in h12["oddIds"]]
    o_h = next((o["price"] for o in os_ if o and o.get("typeId") == 1), None)
    o_d = next((o["price"] for o in os_ if o and o.get("typeId") == 2), None)
    o_a = next((o["price"] for o in os_ if o and o.get("typeId") == 3), None)
    playdoit_list.append({
        "home": home, "away": away,
        "dia": start.strftime("%a %d/%m"), "hora": start.strftime("%H:%M"),
        "o_h": o_h, "o_d": o_d, "o_a": o_a,
    })

# ── 2. Load model predictions ──
df = pd.read_csv("data/ligamx_predicciones.csv")
ligamx = df[df["Fecha"].between("2026-03-21", "2026-03-23")].copy()


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(fc|cf|club|cd|unam|uanl|deportivo|atletico)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def find_playdoit(home, away):
    best, bscore = None, 0.5
    hn, an = norm(home), norm(away)
    for item in playdoit_list:
        sh = SequenceMatcher(None, hn, norm(item["home"])).ratio()
        sa = SequenceMatcher(None, an, norm(item["away"])).ratio()
        s = (sh + sa) / 2
        if s > bscore:
            bscore, best = s, item
    return best


def no_vig(oh, od, oa):
    ph, pd_, pa = 1 / oh, 1 / od, 1 / oa
    t = ph + pd_ + pa
    return ph / t * 100, pd_ / t * 100, pa / t * 100


def vig_pct(oh, od, oa):
    return round((1 / oh + 1 / od + 1 / oa - 1) * 100, 2)


def ev_pct(p, o):
    return round((p / 100 * (o - 1) - (1 - p / 100)) * 100, 1)


def edge_lbl(e):
    if e >= 8:
        return f"[+{e:.1f}% EDGE ALTO]"
    elif e >= 3:
        return f"[+{e:.1f}% edge]"
    elif e >= -3:
        return f"[~{e:+.1f}% neutro]"
    else:
        return f"[{e:.1f}% sin edge]"


def bar(p, w=18):
    f = round(p / 100 * w)
    return "[" + "#" * f + "." * (w - f) + "]"


W = 72
DIAS = {
    "Fri 20/03": "VIERNES 20 DE MARZO",
    "Sat 21/03": "SABADO 21 DE MARZO",
    "Sun 22/03": "DOMINGO 22 DE MARZO",
}

print()
print("=" * W)
print("  LIGA MX CLAUSURA 2026 — JORNADA 11")
print("  Momios: PLAYDOIT (tiempo real)  |  Modelo: Ratings Mult. + Poisson")
print("=" * W)

value_bets = []
dia_actual = ""

for _, row in ligamx.sort_values("Fecha").iterrows():
    home  = row["Equipo Local"]
    away  = row["Equipo Visitante"]
    ph    = float(row["Probabilidad Local % (p_local)"])
    pd_   = float(row["Probabilidad Empate % (p_empate)"])
    pv    = float(row["Probabilidad Visitante % (p_visit)"])
    pred  = row["Prediccion"]
    conf  = row["Nivel de Confianza (ALTA / MEDIA / BAJA)"]
    lh    = float(row["Goles Esperados Local (lambda_h)"])
    la    = float(row["Goles Esperados Visitante (lambda_a)"])
    pos_h = int(row["Posicion Tabla Local (pos_h)"])
    pos_a = int(row["Posicion Tabla Visitante (pos_v)"])
    pts_h = int(row["Puntos en Tabla Local (pts_tabla_h)"])
    pts_a = int(row["Puntos en Tabla Visitante (pts_tabla_v)"])

    pd_item = find_playdoit(home, away)
    dia_key = pd_item["dia"] if pd_item else None
    hora    = pd_item["hora"] if pd_item else "?"
    dia_nom = DIAS.get(dia_key, "") if dia_key else ""

    if dia_nom and dia_nom != dia_actual:
        dia_actual = dia_nom
        print()
        print(f"  {'─' * 68}")
        print(f"  {dia_nom}")
        print(f"  {'─' * 68}")

    print()
    print(f"  {hora} CDT  |  {home} vs {away}")
    print(f"  Tabla: #{pos_h} {home[:18]} ({pts_h}pts)   vs   #{pos_a} {away[:18]} ({pts_a}pts)")
    print(f"  MODELO [{pred}] [Confianza {conf}]  lambda {lh:.2f} vs {la:.2f}")
    print(f"  Local {ph:.1f}% {bar(ph)}  Empate {pd_:.1f}%  Visitante {pv:.1f}%")

    if pd_item and all([pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]]):
        oh, od, oa = pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]
        vig = vig_pct(oh, od, oa)
        mh, md, ma = no_vig(oh, od, oa)
        edge_h = round(ph - mh, 1)
        edge_d = round(pd_ - md, 1)
        edge_v = round(pv - ma, 1)
        ev_h   = ev_pct(ph, oh)
        ev_d   = ev_pct(pd_, od)
        ev_v   = ev_pct(pv, oa)

        print()
        print(f"  PLAYDOIT  vig={vig:.1f}%")
        print(f"  Local     {oh:.4f}  ({mh:.1f}%)  modelo={ph:.1f}%  {edge_lbl(edge_h)}  EV={ev_h:+.1f}%")
        print(f"  Empate    {od:.4f}  ({md:.1f}%)  modelo={pd_:.1f}%  {edge_lbl(edge_d)}  EV={ev_d:+.1f}%")
        print(f"  Visitante {oa:.4f}  ({ma:.1f}%)  modelo={pv:.1f}%  {edge_lbl(edge_v)}  EV={ev_v:+.1f}%")

        vbets = []
        if edge_h >= 5:
            vbets.append(f"Local @ {oh:.4f}  edge={edge_h:+.1f}%  EV={ev_h:+.1f}%")
        if edge_d >= 5:
            vbets.append(f"Empate @ {od:.4f}  edge={edge_d:+.1f}%  EV={ev_d:+.1f}%")
        if edge_v >= 5:
            vbets.append(f"Visitante @ {oa:.4f}  edge={edge_v:+.1f}%  EV={ev_v:+.1f}%")
        if vbets:
            print("  *** VALUE PLAYDOIT ***")
            for vb in vbets:
                print(f"    -> {vb}")
                value_bets.append(f"{dia_nom}  {hora}  {home} vs {away}  ->  {vb}")
        else:
            print("  (sin edge significativo en 1X2)")
    else:
        print("  [sin odds en Playdoit]")

# ── Resumen ──
print()
print("=" * W)
print("  RESUMEN JORNADA 11  —  Pata simple + momio Playdoit")
print("=" * W)
dia_actual = ""
for _, row in ligamx.sort_values("Fecha").iterrows():
    home = row["Equipo Local"]
    away = row["Equipo Visitante"]
    ph   = float(row["Probabilidad Local % (p_local)"])
    pd_  = float(row["Probabilidad Empate % (p_empate)"])
    pv   = float(row["Probabilidad Visitante % (p_visit)"])
    pred = row["Prediccion"]
    conf = row["Nivel de Confianza (ALTA / MEDIA / BAJA)"]
    pmax = max(ph, pd_, pv)
    pd_item = find_playdoit(home, away)
    hora    = pd_item["hora"] if pd_item else "?"
    dia_key = pd_item["dia"]  if pd_item else None
    dia_nom = DIAS.get(dia_key, "") if dia_key else ""
    if dia_nom != dia_actual:
        dia_actual = dia_nom
        print(f"  -- {dia_nom} --")
    if pd_item and all([pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]]):
        oh, od, oa = pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]
        momio = oh if pred == "Local" else (oa if pred == "Visitante" else od)
        momio_s = f"@ {momio:.4f}"
    else:
        momio_s = ""
    print(f"  {hora}  {home:<22} vs {away:<22}  {pred:<12} {pmax:.0f}%  {momio_s:<12} [{conf}]")

if value_bets:
    print()
    print("=" * W)
    print("  VALUE BETS vs PLAYDOIT (edge >= 5%)")
    print("=" * W)
    for vb in value_bets:
        print(f"  {vb}")

print()
print("=" * W)
print("  NOTAS:")
print("  - Momios Playdoit en tiempo real via Altenar sportsbook API")
print("  - Edge = prob. modelo - prob. Playdoit sin vig")
print("  - EV > 0 = valor esperado positivo a largo plazo")
print("  - Confianza ALTA: diff modelo > 20pp entre mejor y 2a opcion")
print("=" * W)
