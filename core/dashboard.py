"""Genera dashboard.html con predicciones Liga MX + UCL actualizadas."""
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "utils") not in sys.path:
    sys.path.insert(0, str(_ROOT / "utils"))
    sys.path.insert(0, str(_ROOT))

from utils.config import ODDS_API_KEY  # noqa: E402
import pandas as pd
import requests
import re
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

CDT = timezone(timedelta(hours=-6))
today_str = datetime.now(CDT).strftime("%Y-%m-%d")
now_label = datetime.now(CDT).strftime("%d/%m/%Y %H:%M")

# ══════════════════════════════════════════════════════════════
# 1. PLAYDOIT ODDS (Altenar API)
# ══════════════════════════════════════════════════════════════
BASE = "https://sb2frontend-altenar2.biahosted.com/api/widget"
r = requests.get(f"{BASE}/GetUpcoming", params={
    "culture": "es-ES", "timezoneOffset": "360", "integration": "playdoit2",
    "lang": "es-ES", "timezone": "America/Mexico_City", "champIds": "10009", "sportId": "66",
}, timeout=15)
adata = r.json()
markets     = {m["id"]: m for m in adata["markets"]}
odds_map    = {o["id"]: o for o in adata["odds"]}
competitors = {c["id"]: c for c in adata["competitors"]}

playdoit_list = []
for ev in adata["events"]:
    if ev.get("champId") != 10009:
        continue
    start = datetime.fromisoformat(ev["startDate"].replace("Z", "")).replace(
        tzinfo=timezone.utc).astimezone(CDT)
    comps = [competitors.get(cid, {}).get("name", "?") for cid in ev.get("competitorIds", [])]
    h12 = next((markets[mid] for mid in ev["marketIds"]
                if markets.get(mid, {}).get("name") == "1x2"), None)
    if not h12:
        continue
    os_ = [odds_map.get(oid) for oid in h12["oddIds"]]
    playdoit_list.append({
        "home": comps[0], "away": comps[1],
        "date": start.strftime("%Y-%m-%d"), "hora": start.strftime("%H:%M"),
        "o_h": next((o["price"] for o in os_ if o and o.get("typeId") == 1), None),
        "o_d": next((o["price"] for o in os_ if o and o.get("typeId") == 2), None),
        "o_a": next((o["price"] for o in os_ if o and o.get("typeId") == 3), None),
    })

# ══════════════════════════════════════════════════════════════
# 2. THE ODDS API — UCL
# ══════════════════════════════════════════════════════════════
API_KEY = ODDS_API_KEY
r2 = requests.get(
    "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/odds/",
    params={"apiKey": API_KEY, "regions": "eu", "markets": "h2h", "oddsFormat": "decimal"},
    timeout=15,
)
ucl_odds_raw = r2.json() if r2.status_code == 200 else []

def get_ucl_odds(home, away):
    for g in ucl_odds_raw:
        ht, at = g["home_team"].lower(), g["away_team"].lower()
        h_words = [w for w in home.lower().split() if len(w) > 3]
        a_words = [w for w in away.lower().split() if len(w) > 3]
        if any(w in ht for w in h_words) or any(w in at for w in a_words):
            books = g.get("bookmakers", [])
            bk = next((b for b in books if b["key"] == "pinnacle"), None) or (books[0] if books else None)
            if bk:
                mkt = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
                if mkt:
                    os_ = {o["name"]: o["price"] for o in mkt["outcomes"]}
                    oh = os_.get(g["home_team"])
                    oa = os_.get(g["away_team"])
                    od = next((v for k, v in os_.items()
                               if k not in [g["home_team"], g["away_team"]]), None)
                    if oh and od and oa:
                        return oh, od, oa, bk["key"]
    return None, None, None, ""

# ══════════════════════════════════════════════════════════════
# 3. HELPERS
# ══════════════════════════════════════════════════════════════
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
        if (sh + sa) / 2 > bscore:
            bscore, best = (sh + sa) / 2, item
    return best

def no_vig(oh, od, oa):
    s = 1/oh + 1/od + 1/oa
    return 1/oh/s*100, 1/od/s*100, 1/oa/s*100

def vig_pct(oh, od, oa):
    return round((1/oh + 1/od + 1/oa - 1) * 100, 1)

def ev_pct(p, o):
    return round((p / 100 * (o - 1) - (1 - p / 100)) * 100, 1)

def edge_class(e):
    if e >= 8:   return "edge-high"
    if e >= 3:   return "edge-med"
    if e >= -3:  return "edge-neutral"
    return "edge-low"

def pred_color(pred):
    return {"Local": "#2563eb", "Visitante": "#dc2626", "Empate": "#d97706"}.get(pred, "#6b7280")

DIAS_ES = {
    "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo",
    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles", "Thursday": "Jueves",
}
MESES_ES = {
    "January":"Enero","February":"Febrero","March":"Marzo","April":"Abril",
    "May":"Mayo","June":"Junio","July":"Julio","August":"Agosto",
    "September":"Septiembre","October":"Octubre","November":"Noviembre","December":"Diciembre",
}

def fmt_fecha(fecha_str):
    dt = datetime.strptime(fecha_str, "%Y-%m-%d")
    dia_en = dt.strftime("%A")
    mes_en = dt.strftime("%B")
    return f"{DIAS_ES.get(dia_en, dia_en)} {dt.day} de {MESES_ES.get(mes_en, mes_en)}"

# ══════════════════════════════════════════════════════════════
# 4. LOAD PREDICTIONS
# ══════════════════════════════════════════════════════════════
lmx_df = pd.read_csv("data/ligamx_predicciones.csv")
lmx_df = lmx_df[lmx_df["Fecha"] >= today_str].copy()

sf_df = pd.read_csv("data/predicciones_sofascore.csv")
ucl_df = sf_df[sf_df["Competicion"].str.contains("UCL|Champions|UEFA", na=False, case=False)].copy()
ucl_df = ucl_df[ucl_df["Fecha"] >= today_str].copy()

# ══════════════════════════════════════════════════════════════
# 5. BUILD HTML CARDS
# ══════════════════════════════════════════════════════════════

def prob_bar_html(ph, pd_, pv):
    return f"""
    <div class="prob-bar">
      <div class="prob-seg local"  style="width:{ph:.1f}%"  title="Local {ph:.1f}%"></div>
      <div class="prob-seg empate" style="width:{pd_:.1f}%" title="Empate {pd_:.1f}%"></div>
      <div class="prob-seg visit"  style="width:{pv:.1f}%"  title="Visitante {pv:.1f}%"></div>
    </div>
    <div class="prob-labels">
      <span>{ph:.1f}%</span><span>{pd_:.1f}%</span><span>{pv:.1f}%</span>
    </div>"""

def odds_row_html(label, odds, p_model, bk_p, edge, ev, is_pred=False):
    ec = edge_class(edge)
    ev_cls = "ev-pos" if ev > 0 else "ev-neg"
    bold = ' style="font-weight:700"' if is_pred else ""
    return f"""
      <tr class="odds-row {ec}"{bold}>
        <td class="odds-label">{label}</td>
        <td class="odds-val">{odds:.4f}</td>
        <td class="odds-bkp">({bk_p:.1f}%)</td>
        <td class="odds-model">{p_model:.1f}%</td>
        <td class="odds-edge">edge {edge:+.1f}%</td>
        <td class="{ev_cls}">EV {ev:+.1f}%</td>
      </tr>"""

def value_badge(label, odds, edge, ev):
    return f'<span class="vb-badge">VALUE: {label} @ {odds:.2f} | edge {edge:+.1f}% | EV {ev:+.1f}%</span>'

def lmx_card(row):
    home  = row["Equipo Local"];   away  = row["Equipo Visitante"]
    ph    = float(row["Probabilidad Local % (p_local)"])
    pd_   = float(row["Probabilidad Empate % (p_empate)"])
    pv    = float(row["Probabilidad Visitante % (p_visit)"])
    pred  = row["Prediccion"]
    conf  = row.get("Nivel de Confianza (ALTA / MEDIA / BAJA)", row.get("Nivel de Confianza % (confianza)", ""))
    lh    = float(row["Goles Esperados Local (lambda_h)"])
    la    = float(row["Goles Esperados Visitante (lambda_a)"])
    pos_h = int(row.get("Posicion Tabla Local (pos_h)", 0))
    pos_a = int(row.get("Posicion Tabla Visitante (pos_v)", 0))
    pts_h = int(row.get("Puntos en Tabla Local (pts_tabla_h)", 0))
    pts_a = int(row.get("Puntos en Tabla Visitante (pts_tabla_v)", 0))
    forma_h = str(row.get("Forma Reciente Local W/D/L (forma_h)", ""))
    forma_a = str(row.get("Forma Reciente Visitante W/D/L (forma_a)", ""))
    # Campos nuevos
    motiv_h    = str(row.get("Motivacion Local (motivacion_local)", ""))
    motiv_icon_h = str(row.get("Icono Motivacion Local (motivacion_icon_h)", ""))
    motiv_a    = str(row.get("Motivacion Visitante (motivacion_visita)", ""))
    motiv_icon_a = str(row.get("Icono Motivacion Visitante (motivacion_icon_a)", ""))
    narrativa  = str(row.get("Narrativa del Partido (narrativa)", ""))
    alerta_emp = str(row.get("Alerta Empate Probable (alerta_empate)", "")).lower() == "true"
    corners_rng = str(row.get("Rango Corners Totales (corners_total_rango)", ""))
    corners_h   = row.get("Corners Estimados Local (corners_h_est)", "")
    corners_a   = row.get("Corners Estimados Visitante (corners_a_est)", "")
    gol_h       = str(row.get("Goleadores Local (goleadores_local)", ""))
    gol_a       = str(row.get("Goleadores Visitante (goleadores_visita)", ""))

    pd_item = find_playdoit(home, away)
    hora = pd_item["hora"] if pd_item else "?:??"

    pc = pred_color(pred)
    conf_cls = {"ALTA":"conf-high","MEDIA":"conf-med","BAJA":"conf-low"}.get(str(conf),"conf-low")

    odds_section = ""
    value_section = ""
    if pd_item and all([pd_item.get("o_h"), pd_item.get("o_d"), pd_item.get("o_a")]):
        oh, od, oa = pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]
        vig = vig_pct(oh, od, oa)
        mh, md, ma = no_vig(oh, od, oa)
        edge_h, edge_d, edge_v = round(ph-mh,1), round(pd_-md,1), round(pv-ma,1)
        ev_h, ev_d, ev_v = ev_pct(ph,oh), ev_pct(pd_,od), ev_pct(pv,oa)
        odds_section = f"""
        <div class="odds-block">
          <div class="odds-header">PLAYDOIT <span class="vig-badge">vig {vig:.1f}%</span></div>
          <table class="odds-table">
            {odds_row_html("Local",    oh, ph, mh, edge_h, ev_h, pred=="Local")}
            {odds_row_html("Empate",   od, pd_, md, edge_d, ev_d, pred=="Empate")}
            {odds_row_html("Visitante",oa, pv, ma, edge_v, ev_v, pred=="Visitante")}
          </table>
        </div>"""
        vbets = []
        if edge_h >= 5: vbets.append(value_badge("Local",    oh, edge_h, ev_h))
        if edge_d >= 5: vbets.append(value_badge("Empate",   od, edge_d, ev_d))
        if edge_v >= 5: vbets.append(value_badge("Visitante",oa, edge_v, ev_v))
        if vbets:
            value_section = f'<div class="value-section">{"".join(vbets)}</div>'

    def forma_dots(s):
        out = ""
        for c in str(s)[-5:]:
            col = {"W":"#16a34a","D":"#d97706","L":"#dc2626"}.get(c,"#9ca3af")
            out += f'<span class="dot" style="background:{col}" title="{c}"></span>'
        return out

    # Narrativa lines
    narr_html = ""
    if narrativa and narrativa != "nan":
        lines = [l.strip() for l in narrativa.split("|") if l.strip()]
        narr_html = "".join(f'<div class="narr-line">{l}</div>' for l in lines)

    # Alerta de empate
    draw_alert_html = ""
    if alerta_emp:
        draw_alert_html = f'<div class="draw-alert">🤝 Alta prob. empate ({pd_:.1f}%) — considera apostar X</div>'

    # Contexto adicional
    ctx_html = f"""
    <div class="ctx-block">
      <div class="ctx-row">
        <span class="ctx-label">Motivación</span>
        <span class="ctx-val">{motiv_icon_h} {home[:16]}: {motiv_h}</span>
        <span class="ctx-sep">|</span>
        <span class="ctx-val">{motiv_icon_a} {away[:16]}: {motiv_a}</span>
      </div>
      <div class="ctx-row">
        <span class="ctx-label">Corners est.</span>
        <span class="ctx-val">{home[:10]}: ~{corners_h} | {away[:10]}: ~{corners_a} | Total: {corners_rng}</span>
      </div>
      <div class="ctx-row">
        <span class="ctx-label">Goleadores*</span>
        <span class="ctx-val scorer-text">{home[:12]}: {gol_h[:60]}</span>
      </div>
      <div class="ctx-row">
        <span class="ctx-label"></span>
        <span class="ctx-val scorer-text">{away[:12]}: {gol_a[:60]}</span>
      </div>
    </div>"""

    return f"""
  <div class="match-card">
    <div class="card-top">
      <div class="card-hora">{hora} CDT</div>
      <div class="card-pos">
        <span class="pos-badge">#{pos_h}</span> {home[:20]}
        <span class="pts-label">{pts_h}pts</span>
      </div>
      <div class="card-vs">VS</div>
      <div class="card-pos">
        <span class="pos-badge">#{pos_a}</span> {away[:20]}
        <span class="pts-label">{pts_a}pts</span>
      </div>
    </div>
    <div class="card-mid">
      <div class="pred-block" style="border-color:{pc}">
        <span class="pred-label" style="color:{pc}">{pred}</span>
        <span class="{conf_cls}">{conf}</span>
      </div>
      <div class="lambda-block">
        <span>λ {lh:.2f}</span>
        <span class="lambda-sep">–</span>
        <span>λ {la:.2f}</span>
      </div>
      <div class="forma-block">
        <div class="forma-row">{forma_dots(forma_h)}</div>
        <div class="forma-row">{forma_dots(forma_a)}</div>
      </div>
    </div>
    {prob_bar_html(ph, pd_, pv)}
    {draw_alert_html}
    {odds_section}
    {value_section}
    {ctx_html}
    {f'<div class="narr-block">{narr_html}</div>' if narr_html else ""}
  </div>"""

def ucl_card(row):
    home  = row["Equipo Local"];   away  = row["Equipo Visitante"]
    ph    = float(row["Probabilidad Local % (p_local)"])
    pd_   = float(row["Probabilidad Empate % (p_empate)"])
    pv    = float(row["Probabilidad Visitante % (p_visit)"])
    pred  = row["Prediccion"]
    conf  = float(row.get("Nivel de Confianza % (confianza)", 0))
    lh    = float(row["Goles Esperados Local (lambda_h)"])
    la    = float(row["Goles Esperados Visitante (lambda_a)"])
    forma_h = str(row.get("Forma Reciente Local W/D/L (forma_h)", ""))
    forma_a = str(row.get("Forma Reciente Visitante W/D/L (forma_a)", ""))
    w_ucl   = float(row.get("Peso UCL en Prediccion % (w_ucl)", 0))

    pc = pred_color(pred)
    oh, od, oa, bk = get_ucl_odds(home, away)

    odds_section = ""
    value_section = ""
    if oh and od and oa:
        vig = vig_pct(oh, od, oa)
        mh, md, ma = no_vig(oh, od, oa)
        edge_h, edge_d, edge_v = round(ph-mh,1), round(pd_-md,1), round(pv-ma,1)
        ev_h, ev_d, ev_v = ev_pct(ph,oh), ev_pct(pd_,od), ev_pct(pv,oa)
        odds_section = f"""
        <div class="odds-block">
          <div class="odds-header">{bk.upper()} <span class="vig-badge">vig {vig:.1f}%</span></div>
          <table class="odds-table">
            {odds_row_html("Local",    oh, ph, mh, edge_h, ev_h, pred=="Local")}
            {odds_row_html("Empate",   od, pd_, md, edge_d, ev_d, pred=="Empate")}
            {odds_row_html("Visitante",oa, pv, ma, edge_v, ev_v, pred=="Visitante")}
          </table>
        </div>"""
        vbets = []
        if edge_h >= 5: vbets.append(value_badge("Local",    oh, edge_h, ev_h))
        if edge_d >= 5: vbets.append(value_badge("Empate",   od, edge_d, ev_d))
        if edge_v >= 5: vbets.append(value_badge("Visitante",oa, edge_v, ev_v))
        if vbets:
            value_section = f'<div class="value-section">{"".join(vbets)}</div>'

    def forma_dots(s):
        out = ""
        for c in str(s)[-5:]:
            col = {"W":"#16a34a","D":"#d97706","L":"#dc2626"}.get(c,"#9ca3af")
            out += f'<span class="dot" style="background:{col}" title="{c}"></span>'
        return out

    return f"""
  <div class="match-card ucl-card">
    <div class="card-top">
      <div class="card-hora">UEFA Champions League</div>
      <div class="card-pos ucl-team">{home}</div>
      <div class="card-vs">VS</div>
      <div class="card-pos ucl-team">{away}</div>
    </div>
    <div class="card-mid">
      <div class="pred-block" style="border-color:{pc}">
        <span class="pred-label" style="color:{pc}">{pred}</span>
        <span class="ucl-conf">conf {conf:.0f}%</span>
      </div>
      <div class="lambda-block">
        <span>λ {lh:.2f}</span>
        <span class="lambda-sep">–</span>
        <span>λ {la:.2f}</span>
      </div>
      <div class="lambda-block" style="font-size:0.72rem;color:#6b7280">
        <span>Peso UCL {w_ucl:.0f}%</span>
        <span class="forma-row">{forma_dots(forma_h)} | {forma_dots(forma_a)}</span>
      </div>
    </div>
    {prob_bar_html(ph, pd_, pv)}
    {odds_section}
    {value_section}
  </div>"""

# ══════════════════════════════════════════════════════════════
# 6. GROUP BY DATE AND RENDER
# ══════════════════════════════════════════════════════════════
lmx_by_date = lmx_df.groupby("Fecha")
ucl_by_date  = ucl_df.groupby("Fecha")

lmx_sections = ""
for fecha, group in sorted(lmx_by_date):
    cards = "".join(lmx_card(row) for _, row in group.iterrows())
    lmx_sections += f"""
  <div class="day-section">
    <h3 class="day-header">{fmt_fecha(fecha)}</h3>
    <div class="cards-grid">{cards}</div>
  </div>"""

ucl_sections = ""
for fecha, group in sorted(ucl_by_date):
    cards = "".join(ucl_card(row) for _, row in group.iterrows())
    ucl_sections += f"""
  <div class="day-section">
    <h3 class="day-header ucl-day">{fmt_fecha(fecha)}</h3>
    <div class="cards-grid">{cards}</div>
  </div>"""

# ── Value bets summary ──
vb_rows = ""
for _, row in lmx_df.iterrows():
    home = row["Equipo Local"]; away = row["Equipo Visitante"]
    ph   = float(row["Probabilidad Local % (p_local)"])
    pd_  = float(row["Probabilidad Empate % (p_empate)"])
    pv   = float(row["Probabilidad Visitante % (p_visit)"])
    pred = row["Prediccion"]
    pd_item = find_playdoit(home, away)
    fecha = row["Fecha"]
    hora = pd_item["hora"] if pd_item else "?"
    if pd_item and all([pd_item.get("o_h"), pd_item.get("o_d"), pd_item.get("o_a")]):
        oh, od, oa = pd_item["o_h"], pd_item["o_d"], pd_item["o_a"]
        mh, md, ma = no_vig(oh, od, oa)
        for lbl, p, o, mp in [("Local",ph,oh,mh),("Empate",pd_,od,md),("Visitante",pv,oa,ma)]:
            edge = round(p - mp, 1)
            ev   = ev_pct(p, o)
            if edge >= 5:
                is_pred = "✓" if pred == lbl else ""
                ev_cls  = "ev-pos" if ev > 0 else "ev-neg"
                vb_rows += f"""<tr>
                  <td>{fmt_fecha(fecha)} {hora}</td>
                  <td>{home} vs {away}</td>
                  <td><b>{lbl}</b> {is_pred}</td>
                  <td>{o:.4f}</td>
                  <td>+{edge:.1f}%</td>
                  <td class="{ev_cls}">+{ev:.1f}%</td>
                </tr>"""

for _, row in ucl_df.iterrows():
    home = row["Equipo Local"]; away = row["Equipo Visitante"]
    ph   = float(row["Probabilidad Local % (p_local)"])
    pd_  = float(row["Probabilidad Empate % (p_empate)"])
    pv   = float(row["Probabilidad Visitante % (p_visit)"])
    pred = row["Prediccion"]
    fecha = row["Fecha"]
    oh, od, oa, bk = get_ucl_odds(home, away)
    if oh and od and oa:
        mh, md, ma = no_vig(oh, od, oa)
        for lbl, p, o, mp in [("Local",ph,oh,mh),("Empate",pd_,od,md),("Visitante",pv,oa,ma)]:
            edge = round(p - mp, 1)
            ev   = ev_pct(p, o)
            if edge >= 5:
                is_pred = "✓" if pred == lbl else ""
                ev_cls  = "ev-pos" if ev > 0 else "ev-neg"
                vb_rows += f"""<tr>
                  <td>{fmt_fecha(fecha)} (UCL)</td>
                  <td>{home} vs {away}</td>
                  <td><b>{lbl}</b> {is_pred}</td>
                  <td>{o:.4f}</td>
                  <td>+{edge:.1f}%</td>
                  <td class="{ev_cls}">+{ev:.1f}%</td>
                </tr>"""

# ══════════════════════════════════════════════════════════════
# 7. HTML TEMPLATE
# ══════════════════════════════════════════════════════════════
HTML = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dashboard Predicciones — {now_label}</title>
<style>
  :root {{
    --bg: #0f172a; --bg2: #1e293b; --bg3: #334155;
    --text: #f1f5f9; --text2: #94a3b8; --text3: #64748b;
    --blue: #3b82f6; --green: #22c55e; --red: #ef4444;
    --yellow: #f59e0b; --purple: #a855f7;
    --ucl: #1e40af;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }}

  /* ─── Header ─── */
  .header {{ background: linear-gradient(135deg,#1e3a5f,#0f172a); padding: 24px 32px; border-bottom: 1px solid #1e3a5f; }}
  .header h1 {{ font-size: 1.6rem; font-weight: 700; letter-spacing: -0.5px; }}
  .header h1 span {{ color: var(--blue); }}
  .header .subtitle {{ color: var(--text2); margin-top: 4px; font-size: 0.85rem; }}
  .updated {{ color: var(--text3); font-size: 0.78rem; margin-top: 2px; }}

  /* ─── Tabs ─── */
  .tabs {{ display: flex; gap: 4px; padding: 16px 32px 0; border-bottom: 1px solid var(--bg3); }}
  .tab {{ padding: 8px 20px; border-radius: 6px 6px 0 0; cursor: pointer; font-weight: 600;
          background: var(--bg2); color: var(--text2); border: 1px solid transparent;
          border-bottom: none; transition: all .15s; font-size: 0.88rem; }}
  .tab.active {{ background: var(--bg); color: var(--text); border-color: var(--bg3); border-bottom-color: var(--bg); }}
  .tab:hover:not(.active) {{ color: var(--text); }}

  /* ─── Content ─── */
  .tab-content {{ display: none; padding: 24px 32px; }}
  .tab-content.active {{ display: block; }}

  /* ─── Day section ─── */
  .day-section {{ margin-bottom: 32px; }}
  .day-header {{ font-size: 1rem; font-weight: 700; color: var(--blue); margin-bottom: 12px;
                 padding-bottom: 6px; border-bottom: 2px solid var(--blue); text-transform: uppercase;
                 letter-spacing: 0.5px; }}
  .day-header.ucl-day {{ color: #60a5fa; border-color: #60a5fa; }}

  /* ─── Cards grid ─── */
  .cards-grid {{ display: grid; grid-template-columns: repeat(auto-fill,minmax(340px,1fr)); gap: 16px; }}

  /* ─── Match card ─── */
  .match-card {{ background: var(--bg2); border-radius: 12px; padding: 16px;
                 border: 1px solid var(--bg3); transition: transform .15s, box-shadow .15s; }}
  .match-card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.4); }}
  .ucl-card {{ border-top: 3px solid #3b82f6; }}

  .card-top {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
  .card-hora {{ font-size: 0.75rem; color: var(--text3); background: var(--bg3); padding: 2px 8px;
                border-radius: 4px; white-space: nowrap; }}
  .card-pos {{ flex: 1; font-weight: 600; font-size: 0.88rem; min-width: 0; }}
  .ucl-team {{ font-size: 0.92rem; }}
  .card-vs {{ color: var(--text3); font-size: 0.75rem; font-weight: 700; }}
  .pos-badge {{ display: inline-block; background: var(--bg3); color: var(--text2);
                font-size: 0.7rem; padding: 1px 5px; border-radius: 3px; margin-right: 3px; }}
  .pts-label {{ font-size: 0.72rem; color: var(--text3); margin-left: 3px; }}

  .card-mid {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; flex-wrap: wrap; }}
  .pred-block {{ display: flex; flex-direction: column; align-items: center; border-left: 3px solid;
                 padding: 4px 10px; border-radius: 0 6px 6px 0; background: rgba(255,255,255,.04); }}
  .pred-label {{ font-weight: 800; font-size: 0.9rem; }}
  .lambda-block {{ display: flex; gap: 4px; align-items: center; font-size: 0.82rem; color: var(--text2); }}
  .lambda-sep {{ color: var(--text3); }}

  .conf-high {{ font-size: 0.7rem; color: #4ade80; font-weight: 600; }}
  .conf-med  {{ font-size: 0.7rem; color: #fbbf24; font-weight: 600; }}
  .conf-low  {{ font-size: 0.7rem; color: #f87171; font-weight: 600; }}
  .ucl-conf  {{ font-size: 0.7rem; color: var(--text2); }}

  /* ─── Forma dots ─── */
  .forma-block {{ display: flex; flex-direction: column; gap: 4px; }}
  .forma-row {{ display: flex; gap: 3px; align-items: center; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

  /* ─── Prob bar ─── */
  .prob-bar {{ display: flex; height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 3px; }}
  .prob-seg {{ height: 100%; }}
  .prob-seg.local  {{ background: var(--blue); }}
  .prob-seg.empate {{ background: var(--yellow); }}
  .prob-seg.visit  {{ background: var(--red); }}
  .prob-labels {{ display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text3); margin-bottom: 10px; }}

  /* ─── Odds block ─── */
  .odds-block {{ background: var(--bg3); border-radius: 8px; padding: 10px 12px; margin-top: 8px; }}
  .odds-header {{ font-size: 0.72rem; font-weight: 700; color: var(--text2); margin-bottom: 6px;
                  text-transform: uppercase; letter-spacing: 0.5px; }}
  .vig-badge {{ background: rgba(255,255,255,.1); padding: 1px 6px; border-radius: 3px;
                font-weight: 400; text-transform: none; }}
  .odds-table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  .odds-row td {{ padding: 3px 4px; }}
  .odds-label {{ font-weight: 600; width: 72px; }}
  .odds-val   {{ font-weight: 700; color: var(--text); width: 56px; }}
  .odds-bkp  {{ color: var(--text3); width: 52px; }}
  .odds-model {{ color: var(--text2); width: 48px; }}
  .odds-edge  {{ font-size: 0.75rem; }}
  .ev-pos {{ color: #4ade80; font-weight: 700; }}
  .ev-neg {{ color: #f87171; }}

  /* edge row colors */
  .edge-high td   {{ background: rgba(34,197,94,.08); }}
  .edge-med  td   {{ background: rgba(59,130,246,.06); }}
  .edge-neutral td {{ background: transparent; }}
  .edge-low  td   {{ background: rgba(239,68,68,.05); }}

  /* ─── Value bets section ─── */
  .value-section {{ margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px; }}
  .vb-badge {{ background: linear-gradient(90deg,#065f46,#064e3b); color: #6ee7b7;
               font-size: 0.72rem; padding: 3px 8px; border-radius: 4px; font-weight: 600; }}

  /* ─── Value bets table ─── */
  .vb-table {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
  .vb-table th {{ text-align: left; padding: 8px 10px; color: var(--text2); font-weight: 600;
                  border-bottom: 1px solid var(--bg3); font-size: 0.78rem; text-transform: uppercase; }}
  .vb-table td {{ padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.04); }}
  .vb-table tr:hover td {{ background: rgba(255,255,255,.03); }}

  /* ─── Context block ─── */
  .ctx-block {{ margin-top: 10px; background: rgba(255,255,255,.03); border-radius: 6px;
                padding: 8px 10px; font-size: 0.76rem; }}
  .ctx-row {{ display: flex; gap: 6px; align-items: baseline; margin-bottom: 3px; flex-wrap: wrap; }}
  .ctx-label {{ color: var(--text3); min-width: 76px; font-weight: 600; text-transform: uppercase;
                font-size: 0.68rem; letter-spacing: 0.3px; }}
  .ctx-val {{ color: var(--text2); }}
  .ctx-sep {{ color: var(--text3); }}
  .scorer-text {{ color: #a5b4fc; font-size: 0.73rem; }}

  /* ─── Narrative block ─── */
  .narr-block {{ margin-top: 8px; }}
  .narr-line {{ font-size: 0.76rem; color: var(--text2); padding: 3px 0;
                border-left: 2px solid var(--bg3); padding-left: 8px; margin-bottom: 2px; }}

  /* ─── Draw alert ─── */
  .draw-alert {{ background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.3);
                 color: #fbbf24; font-size: 0.76rem; padding: 5px 10px; border-radius: 5px;
                 margin: 6px 0; font-weight: 600; }}

  /* ─── Responsive ─── */
  @media(max-width:600px) {{
    .tabs {{ padding: 12px 16px 0; }}
    .tab-content {{ padding: 16px; }}
    .cards-grid {{ grid-template-columns: 1fr; }}
    .header {{ padding: 16px; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Soccer <span>Predictions</span> Dashboard</h1>
  <div class="subtitle">Liga MX Clausura 2026 · UEFA Champions League Semifinales</div>
  <div class="updated">Actualizado: {now_label} CDT &nbsp;|&nbsp; Modelo: Dixon-Coles + Ratings Multiplicativos + Motivación &nbsp;|&nbsp; Odds: Playdoit (real) + Pinnacle</div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('ligamx',this)">⚽ Liga MX</div>
  <div class="tab" onclick="showTab('ucl',this)">🏆 UCL Semifinales</div>
  <div class="tab" onclick="showTab('valuebets',this)">💎 Value Bets</div>
</div>

<!-- LIGA MX -->
<div id="tab-ligamx" class="tab-content active">
  <div style="color:var(--text2);margin-bottom:16px;font-size:0.85rem">
    {len(lmx_df)} partidos · Jornadas 12–13 · Fechas: {lmx_df['Fecha'].min()} → {lmx_df['Fecha'].max()} · Momios: Playdoit en tiempo real
  </div>
  {lmx_sections}
</div>

<!-- UCL -->
<div id="tab-ucl" class="tab-content">
  <div style="color:var(--text2);margin-bottom:16px;font-size:0.85rem">
    {len(ucl_df)} partidos · Semifinales Champions League · Momios: Pinnacle
  </div>
  {ucl_sections if ucl_sections else '<p style="color:var(--text3);padding:32px 0">No hay partidos UCL en el horizonte de 10 días.</p>'}
</div>

<!-- VALUE BETS -->
<div id="tab-valuebets" class="tab-content">
  <div style="color:var(--text2);margin-bottom:16px;font-size:0.85rem">
    Apuestas con edge ≥ 5% sobre probabilidad sin vig · ✓ = coincide con predicción del modelo
  </div>
  <table class="vb-table">
    <thead><tr>
      <th>Fecha</th><th>Partido</th><th>Apuesta</th><th>Momio</th><th>Edge</th><th>EV</th>
    </tr></thead>
    <tbody>{vb_rows if vb_rows else '<tr><td colspan="6" style="color:var(--text3);padding:24px">Sin value bets detectados.</td></tr>'}</tbody>
  </table>
</div>

<script>
function showTab(name, el) {{
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
}}
</script>
</body>
</html>"""

Path("dashboard.html").write_text(HTML, encoding="utf-8")
print(f"dashboard.html generado — {len(lmx_df)} partidos Liga MX, {len(ucl_df)} partidos UCL")
print(f"Value bets incluidos: {vb_rows.count('<tr>') if vb_rows else 0}")
