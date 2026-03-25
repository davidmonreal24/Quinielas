"""Reporte Liga MX Jornada 11 - 20-22 Marzo 2026"""
import json, re, math
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
import pandas as pd

CDT = timezone(timedelta(hours=-6))

odds_raw = json.loads(open("data/_odds_cache/odds_soccer_mexico_ligamx.json", encoding="utf-8").read())

DIAS = {
    "Fri 20/03": "Viernes 20 de Marzo",
    "Sat 21/03": "Sabado 21 de Marzo",
    "Sun 22/03": "Domingo 22 de Marzo",
}

def no_vig(oh, od, oa):
    ph, pd_, pa = 1/oh, 1/od, 1/oa
    t = ph + pd_ + pa
    return ph/t, pd_/t, pa/t

def vig_pct(oh, od, oa):
    return round((1/oh + 1/od + 1/oa - 1)*100, 2)

MAX_VIG = 20.0
PREFERRED = ["pinnacle","betfair_ex_eu","matchbook","marathonbet","nordicbet","betsson","onexbet","coolbet"]

def process_game(g):
    home = g["home_team"]; away = g["away_team"]
    bks = {}
    for bk in g.get("bookmakers", []):
        mkt = next((m for m in bk.get("markets", []) if m["key"] == "h2h"), None)
        if not mkt: continue
        out = {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
        oh, od, oa = out.get(home), out.get("Draw"), out.get(away)
        if not all([oh, od, oa]): continue
        v = vig_pct(oh, od, oa)
        if v > MAX_VIG: continue
        ph, pd_, pa = no_vig(oh, od, oa)
        bks[bk["key"]] = {
            "o_h": oh, "o_d": od, "o_a": oa,
            "p_h": round(ph*100, 1), "p_d": round(pd_*100, 1), "p_a": round(pa*100, 1),
            "vig": v, "title": bk["title"],
        }
    preferred = None
    for k in PREFERRED:
        if k in bks:
            preferred = {**bks[k], "bk_key": k}; break
    if preferred is None and bks:
        best = min(bks, key=lambda k: bks[k]["vig"])
        preferred = {**bks[best], "bk_key": best}
    avg = None
    if bks:
        avg = {
            "p_h": round(sum(b["p_h"] for b in bks.values())/len(bks), 1),
            "p_d": round(sum(b["p_d"] for b in bks.values())/len(bks), 1),
            "p_a": round(sum(b["p_a"] for b in bks.values())/len(bks), 1),
        }
    utc = datetime.fromisoformat(g["commence_time"].replace("Z", ""))
    local = utc.replace(tzinfo=timezone.utc).astimezone(CDT)
    dia_key = local.strftime("%a %d/%m")
    return {"home": home, "away": away, "dia": dia_key, "hora": local.strftime("%H:%M"),
            "preferred": preferred, "avg": avg, "n_bk": len(bks)}

odds_list = [process_game(g) for g in odds_raw]

# Predicciones — incluye fecha 2026-03-23 UTC (= domingo 22 CDT)
df = pd.read_csv("data/ligamx_predicciones.csv")
ligamx = df[df["Fecha"].between("2026-03-21", "2026-03-23")].copy()

def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\b(fc|cf|club|cd|unam|uanl)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

def find_odds(home, away):
    best, bscore = None, 0.55
    hn, an = norm(home), norm(away)
    for item in odds_list:
        sh = SequenceMatcher(None, hn, norm(item["home"])).ratio()
        sa = SequenceMatcher(None, an, norm(item["away"])).ratio()
        s = (sh + sa) / 2
        if s > bscore: bscore, best = s, item
    return best

def ev_pct(p, o): return round((p*(o-1)-(1-p))*100, 1)

def edge_lbl(e):
    if e >= 8:   return f"[+{e:.1f}% EDGE ALTO]"
    elif e >= 3: return f"[+{e:.1f}% edge]"
    elif e >= -3: return f"[~{e:+.1f}% neutro]"
    else:        return f"[{e:.1f}% sin edge]"

def bar(p, w=20):
    f = round(p/100*w)
    return "[" + "#"*f + "."*(w-f) + "]"

W = 72
print()
print("=" * W)
print("  LIGA MX CLAUSURA 2026 — JORNADA 11")
print("  Viernes 20 / Sabado 21 / Domingo 22 de Marzo 2026")
print("  Modelo: Ratings Multiplicativos + Poisson (Sofascore)")
print("  Mercado: The Odds API  |  Horarios CDT (hora Mexico)")
print("=" * W)

dia_actual = ""
value_summary = []

for _, row in ligamx.sort_values("Fecha").iterrows():
    home  = row["Equipo Local"];  away  = row["Equipo Visitante"]
    ph    = float(row["Probabilidad Local % (p_local)"])
    pd_   = float(row["Probabilidad Empate % (p_empate)"])
    pv    = float(row["Probabilidad Visitante % (p_visit)"])
    lh    = float(row["Goles Esperados Local (lambda_h)"])
    la    = float(row["Goles Esperados Visitante (lambda_a)"])
    fh    = row["Forma Reciente Local W/D/L (forma_h)"]
    fa    = row["Forma Reciente Visitante W/D/L (forma_a)"]
    pth   = int(row["Puntos Forma Local (pts_h, W=3 D=1 L=0)"])
    pta   = int(row["Puntos Forma Visitante (pts_a, W=3 D=1 L=0)"])
    pos_h = int(row["Posicion Tabla Local (pos_h)"])
    pos_a = int(row["Posicion Tabla Visitante (pos_v)"])
    pts_h = int(row["Puntos en Tabla Local (pts_tabla_h)"])
    pts_a = int(row["Puntos en Tabla Visitante (pts_tabla_v)"])
    pred  = row["Prediccion"]
    conf  = row["Nivel de Confianza (ALTA / MEDIA / BAJA)"]
    h2h_n  = int(row["H2H Partidos Analizados (h2h_n)"])
    h2h_wh = int(row["H2H Victorias Local (h2h_w_h)"])
    h2h_d  = int(row["H2H Empates (h2h_d)"])
    h2h_wa = int(row["H2H Victorias Visitante (h2h_w_a)"])
    att_h  = float(row["Ratio Ataque Local (att_h, >1 = mejor que promedio)"])
    def_a  = float(row["Ratio Defensa Visitante (def_a)"])
    att_a  = float(row["Ratio Ataque Visitante (att_a)"])
    def_h  = float(row["Ratio Defensa Local (def_h, <1 = mejor que promedio)"])

    odds_item = find_odds(home, away)
    dia_key   = odds_item["dia"]  if odds_item else None
    hora      = odds_item["hora"] if odds_item else "?"
    dia_nombre = DIAS.get(dia_key, "") if dia_key else ""

    if dia_nombre and dia_nombre != dia_actual:
        dia_actual = dia_nombre
        print()
        print(f"  {'─'*68}")
        print(f"  {dia_nombre.upper()}")
        print(f"  {'─'*68}")

    print()
    print(f"  {hora} CDT  |  {home} vs {away}")
    print(f"  Tabla: #{pos_h} {home[:18]} ({pts_h}pts)   vs   #{pos_a} {away[:18]} ({pts_a}pts)")
    print()
    print(f"  MODELO  [{pred}]  [Confianza {conf}]")
    print(f"  lambda {lh:.2f} vs {la:.2f}  (att_h={att_h:.2f} def_a={def_a:.2f} | att_a={att_a:.2f} def_h={def_h:.2f})")
    print(f"  Local {ph:.1f}%  {bar(ph)}  Empate {pd_:.1f}%  Visitante {pv:.1f}%")
    print(f"  Forma: {home[:14]} {fh} ({pth}pts)  |  {away[:14]} {fa} ({pta}pts)")
    if h2h_n > 0:
        print(f"  H2H ({h2h_n} partidos): {home[:14]} {h2h_wh}G-{h2h_d}E-{h2h_wa}G {away[:14]}")

    if odds_item and odds_item["preferred"]:
        pref = odds_item["preferred"]
        avg  = odds_item["avg"]
        nbk  = odds_item["n_bk"]
        bkn  = pref["bk_key"].upper()

        edge_h = round(ph  - pref["p_h"], 1)
        edge_d = round(pd_ - pref["p_d"], 1)
        edge_v = round(pv  - pref["p_a"], 1)
        ev_h   = ev_pct(ph  / 100, pref["o_h"])
        ev_d   = ev_pct(pd_ / 100, pref["o_d"])
        ev_v   = ev_pct(pv  / 100, pref["o_a"])

        print()
        print(f"  ODDS — {bkn}  ({nbk} casas)  vig={pref['vig']:.1f}%")
        print(f"  Local     {pref['o_h']:.2f}  ({pref['p_h']:.1f}%)  modelo={ph:.1f}%  {edge_lbl(edge_h)}  EV={ev_h:+.1f}%")
        print(f"  Empate    {pref['o_d']:.2f}  ({pref['p_d']:.1f}%)  modelo={pd_:.1f}%  {edge_lbl(edge_d)}  EV={ev_d:+.1f}%")
        print(f"  Visitante {pref['o_a']:.2f}  ({pref['p_a']:.1f}%)  modelo={pv:.1f}%  {edge_lbl(edge_v)}  EV={ev_v:+.1f}%")
        if avg:
            print(f"  Mercado ({nbk} casas): L {avg['p_h']:.1f}%  E {avg['p_d']:.1f}%  V {avg['p_a']:.1f}%")

        vbets = []
        if edge_h >= 5: vbets.append(f"Local @ {pref['o_h']:.2f}  edge={edge_h:+.1f}%  EV={ev_h:+.1f}%")
        if edge_d >= 5: vbets.append(f"Empate @ {pref['o_d']:.2f}  edge={edge_d:+.1f}%  EV={ev_d:+.1f}%")
        if edge_v >= 5: vbets.append(f"Visitante @ {pref['o_a']:.2f}  edge={edge_v:+.1f}%  EV={ev_v:+.1f}%")
        if vbets:
            print(f"  *** VALUE ({bkn}) ***")
            for vb in vbets:
                print(f"    -> {vb}")
                value_summary.append(f"{dia_nombre}  {hora}  {home} vs {away}  ->  {vb}")
        else:
            print("  (sin edge significativo en 1X2)")
    else:
        print("  [sin odds en Odds API]")

# ── Resumen ──
print()
print("=" * W)
print("  RESUMEN JORNADA 11  —  Pata simple (mayor prob) + momio")
print("=" * W)
dia_actual = ""
for _, row in ligamx.sort_values("Fecha").iterrows():
    home  = row["Equipo Local"];  away  = row["Equipo Visitante"]
    ph    = float(row["Probabilidad Local % (p_local)"])
    pd_   = float(row["Probabilidad Empate % (p_empate)"])
    pv    = float(row["Probabilidad Visitante % (p_visit)"])
    pred  = row["Prediccion"]
    conf  = row["Nivel de Confianza (ALTA / MEDIA / BAJA)"]
    pmax  = max(ph, pd_, pv)
    odds_item = find_odds(home, away)
    hora      = odds_item["hora"] if odds_item else "?"
    dia_key   = odds_item["dia"]  if odds_item else None
    dia_nombre = DIAS.get(dia_key, "") if dia_key else ""
    if dia_nombre != dia_actual:
        dia_actual = dia_nombre
        print(f"  -- {dia_nombre} --")
    pref = odds_item["preferred"] if odds_item else None
    if pref:
        momio = pref["o_h"] if pred == "Local" else (pref["o_a"] if pred == "Visitante" else pref["o_d"])
        momio_s = f"@ {momio:.2f}"
    else:
        momio_s = ""
    print(f"  {hora}  {home:<22} vs {away:<22}  {pred:<12} {pmax:.0f}%  {momio_s:<10} [{conf}]")

if value_summary:
    print()
    print("=" * W)
    print("  VALUE BETS detectadas (edge >= 5% vs mercado sin vig)")
    print("=" * W)
    for vb in value_summary:
        print(f"  {vb}")

print()
print("=" * W)
print("  NOTAS:")
print("  - Edge = prob. modelo - prob. mercado sin vig (Pinnacle o equivalente)")
print("  - EV > 0 = valor esperado positivo a largo plazo")
print("  - Confianza ALTA: diferencia modelo > 20pp entre mejor y 2a opcion")
print("  - Sin alineaciones confirmadas aun. Actualiza con lineup_watcher.py")
print("    cuando se anuncien los XI (~2h antes de cada partido).")
print("=" * W)
