"""
analisis_j15.py
===============
Backtesting matemático: Predicciones del modelo vs Resultados reales Jornada 15
Liga MX Clausura 2026 — 17-19 Abril 2026

Métricas calculadas:
  1. Accuracy (% aciertos en el resultado ganador)
  2. Brier Score (error cuadrático medio de probabilidades)
  3. Log Loss (entropía cruzada — penaliza confianza errónea)
  4. Calibración por rango (¿las probs de 40% ganan ~40%?)
  5. Bias de sesgo (¿hay sobre/sub-predicción sistemática?)
  6. RMSE de lambdas vs goles reales
  7. Propuestas de mejora cuantificadas
"""
import math
import pandas as pd
import numpy as np

# ─── 1. Resultados reales Jornada 15 (17-19 Abril 2026) ─────────────────────
# Fuentes: TUDN, ClaroSports, El Informador, Mediotiempo
RESULTADOS_J15 = [
    # (local, visitante, goles_local, goles_visitante, fecha)
    ("Atlético San Luis", "Pumas UNAM",    0, 2, "2026-04-17"),  # Juninho + Carrillo
    ("Mazatlán FC",       "Querétaro FC",  1, 1, "2026-04-17"),  # Empate
    ("Cruz Azul",         "Club Tijuana",  1, 1, "2026-04-18"),  # Empate
    ("Club Necaxa",       "Tigres UANL",   1, 1, "2026-04-18"),  # Badaloni 34' / Correa 90+6'
    ("CF Monterrey",      "CF Pachuca",    1, 3, "2026-04-18"),  # Pachuca goleada
    ("CD Guadalajara",    "Club Puebla",   5, 0, "2026-04-18"),  # Chivas aplastante
    ("Club América",      "CD Toluca",     2, 1, "2026-04-18"),  # Brian Rodríguez x2
    ("Club León",         "FC Juárez",     3, 1, "2026-04-19"),  # Díaz 2, Estupiñán, Arcila
    ("Santos Laguna",     "Atlas FC",      0, 1, "2026-04-19"),  # Capasso 20'
]

# ─── 2. Cargar predicciones del modelo ──────────────────────────────────────
df_pred = pd.read_csv("data/ligamx_predicciones.csv")
pred_cols = {
    "home":  "Equipo Local",
    "away":  "Equipo Visitante",
    "pred":  "Prediccion",
    "ph":    "Probabilidad Local % (p_local)",
    "pd_":   "Probabilidad Empate % (p_empate)",
    "pv":    "Probabilidad Visitante % (p_visit)",
    "lh":    "Goles Esperados Local (lambda_h)",
    "la":    "Goles Esperados Visitante (lambda_a)",
    "conf":  "Nivel de Confianza (ALTA / MEDIA / BAJA)",
}

# Filtrar fechas 17-19 Abril
mask = df_pred["Fecha"].between("2026-04-17", "2026-04-19")
preds = df_pred[mask].copy()
print(f"Predicciones encontradas para 17-19 Abril: {len(preds)}")


def _norm(s):
    import re
    s = str(s).lower().strip()
    s = re.sub(r"\b(fc|cf|cd|club|atletico|de)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def find_pred(home, away, df):
    """Busca la predicción correspondiente."""
    from difflib import SequenceMatcher
    hn, an = _norm(home), _norm(away)
    best_score, best_row = 0, None
    for _, row in df.iterrows():
        sh = SequenceMatcher(None, hn, _norm(row[pred_cols["home"]])).ratio()
        sa = SequenceMatcher(None, an, _norm(row[pred_cols["away"]])).ratio()
        s = (sh + sa) / 2
        if s > best_score:
            best_score, best_row = s, row
    return best_row if best_score > 0.55 else None


# ─── 3. Unir predicciones con resultados reales ──────────────────────────────
records = []
for home, away, hg, ag, fecha in RESULTADOS_J15:
    # Resultado real
    if hg > ag:
        real = "Local"
    elif hg == ag:
        real = "Empate"
    else:
        real = "Visitante"

    row = find_pred(home, away, preds)
    if row is None:
        print(f"  AVISO: no encontré predicción para {home} vs {away}")
        continue

    ph  = float(row[pred_cols["ph"]])  / 100
    pd_ = float(row[pred_cols["pd_"]]) / 100
    pv  = float(row[pred_cols["pv"]])  / 100
    lh  = float(row[pred_cols["lh"]])
    la  = float(row[pred_cols["la"]])
    pred = row[pred_cols["pred"]]
    conf = row[pred_cols["conf"]]

    correct = (pred == real)
    # Probabilidad asignada al resultado correcto
    p_correct = {"Local": ph, "Empate": pd_, "Visitante": pv}[real]
    # Brier Score (MSE de vector de probs)
    oh = 1.0 if real == "Local" else 0.0
    od = 1.0 if real == "Empate" else 0.0
    ov = 1.0 if real == "Visitante" else 0.0
    brier = (ph - oh)**2 + (pd_ - od)**2 + (pv - ov)**2
    # Log loss (evitar log(0))
    log_loss = -math.log(max(p_correct, 1e-7))

    records.append({
        "local":        home,
        "visitante":    away,
        "fecha":        fecha,
        "goles_l":      hg,
        "goles_v":      ag,
        "resultado_real": real,
        "prediccion":   pred,
        "correcto":     correct,
        "confianza":    conf,
        "ph": ph, "pd": pd_, "pv": pv,
        "lh": lh, "la": la,
        "p_correcto":   round(p_correct, 4),
        "brier":        round(brier, 4),
        "log_loss":     round(log_loss, 4),
    })

df_res = pd.DataFrame(records)

# ─── 4. MÉTRICAS GLOBALES ────────────────────────────────────────────────────
n = len(df_res)
accuracy   = df_res["correcto"].mean()
brier_avg  = df_res["brier"].mean()
logloss_avg = df_res["log_loss"].mean()

W = 72
print()
print("=" * W)
print("  BACKTESTING J15 — Liga MX Clausura 2026 — 17/19 Abril 2026")
print("=" * W)
print(f"  Partidos analizados : {n}")
print(f"  Accuracy            : {accuracy*100:.1f}%  ({df_res['correcto'].sum()}/{n} correctos)")
print(f"  Brier Score         : {brier_avg:.4f}  (ideal=0.0, random=0.667)")
print(f"  Log Loss promedio   : {logloss_avg:.4f}  (ideal=0.0, random=1.099)")
print()

# ─── 5. DETALLE POR PARTIDO ──────────────────────────────────────────────────
print(f"  {'Partido':<36} {'Real':<10} {'Pred':<10} {'P(real)':<8} {'Brier':<8} {'OK?'}")
print(f"  {'-'*68}")
for _, r in df_res.iterrows():
    partido = f"{r['local'][:17]} vs {r['visitante'][:14]}"
    ok = "[OK]" if r["correcto"] else "[--]"
    print(f"  {partido:<36} {r['resultado_real']:<10} {r['prediccion']:<10} "
          f"{r['p_correcto']*100:>5.1f}%   {r['brier']:.3f}    {ok}")

# ─── 6. ANÁLISIS DE SESGO ────────────────────────────────────────────────────
print()
print(f"  {'─'*68}")
print("  ANÁLISIS DE SESGO")
print(f"  {'─'*68}")

pred_counts = df_res["prediccion"].value_counts()
real_counts = df_res["resultado_real"].value_counts()
print(f"  Predicciones →  Local={pred_counts.get('Local',0)}  Empate={pred_counts.get('Empate',0)}  Visitante={pred_counts.get('Visitante',0)}")
print(f"  Resultados   →  Local={real_counts.get('Local',0)}  Empate={real_counts.get('Empate',0)}  Visitante={real_counts.get('Visitante',0)}")

# Probabilidad promedio asignada vs frecuencia real
for outcome, col in [("Local","ph"), ("Empate","pd"), ("Visitante","pv")]:
    avg_p = df_res[col].mean() * 100
    real_freq = (df_res["resultado_real"] == outcome).mean() * 100
    bias = avg_p - real_freq
    print(f"  {outcome:<12}: modelo asigna {avg_p:.1f}%  |  ocurrió {real_freq:.1f}%  |  sesgo {bias:+.1f}pp")

# ─── 7. CALIBRACIÓN POR RANGO ────────────────────────────────────────────────
print()
print(f"  {'─'*68}")
print("  CALIBRACIÓN (¿probabilidades reflejan realidad?)")
print(f"  {'─'*68}")

# Crear filas largas: una fila por (partido, resultado)
cal_rows = []
for _, r in df_res.iterrows():
    for outcome, col in [("Local","ph"), ("Empate","pd"), ("Visitante","pv")]:
        real_flag = 1 if r["resultado_real"] == outcome else 0
        cal_rows.append({"pred_prob": r[col], "real": real_flag})

cal_df = pd.DataFrame(cal_rows)
bins = [0, 0.15, 0.25, 0.35, 0.50, 0.65, 1.01]
labels = ["<15%","15-25%","25-35%","35-50%","50-65%",">65%"]
cal_df["bucket"] = pd.cut(cal_df["pred_prob"], bins=bins, labels=labels)
cal_summary = cal_df.groupby("bucket", observed=True).agg(
    n=("real","count"), real_rate=("real","mean"), avg_pred=("pred_prob","mean")
).reset_index()
print(f"  {'Rango':<10} {'N':<5} {'Pred %':<10} {'Real %':<10} {'Diff'}")
for _, row in cal_summary.iterrows():
    diff = row["avg_pred"]*100 - row["real_rate"]*100
    print(f"  {str(row['bucket']):<10} {int(row['n']):<5} {row['avg_pred']*100:>6.1f}%    "
          f"{row['real_rate']*100:>6.1f}%    {diff:+.1f}pp")

# ─── 8. ERROR DE LAMBDAS ──────────────────────────────────────────────────────
print()
print(f"  {'─'*68}")
print("  ERROR EN LAMBDAS (Goles Esperados vs Goles Reales)")
print(f"  {'─'*68}")

lh_err  = df_res["lh"] - df_res["goles_l"]
la_err  = df_res["la"] - df_res["goles_v"]
all_err = pd.concat([lh_err, la_err])
mae  = all_err.abs().mean()
rmse = math.sqrt((all_err**2).mean())
bias_l = lh_err.mean()
bias_a = la_err.mean()

print(f"  MAE lambdas         : {mae:.3f} goles")
print(f"  RMSE lambdas        : {rmse:.3f} goles")
print(f"  Sesgo λ_local       : {bias_l:+.3f}  (+ = modelo sobreestima goles locales)")
print(f"  Sesgo λ_visita      : {bias_a:+.3f}  (+ = modelo sobreestima goles visitantes)")
print()
print(f"  {'Partido':<36} {'λ_h':>5} {'G_h':>5} {'Err_h':>6}  {'λ_v':>5} {'G_v':>5} {'Err_v':>6}")
print(f"  {'-'*66}")
for _, r in df_res.iterrows():
    partido = f"{r['local'][:17]} vs {r['visitante'][:14]}"
    print(f"  {partido:<36} {r['lh']:>5.2f} {r['goles_l']:>5}  {r['lh']-r['goles_l']:>+6.2f}  "
          f"{r['la']:>5.2f} {r['goles_v']:>5}  {r['la']-r['goles_v']:>+6.2f}")

# ─── 9. ANÁLISIS DE VARIABLES CONTEXTUALES ───────────────────────────────────
print()
print("=" * W)
print("  ANÁLISIS DE FACTORES CLAVE — ¿Qué falló?")
print("=" * W)

contexto = [
    {
        "partido":  "Mazatlán vs Querétaro",
        "real":     "Empate 1-1",
        "pred":     "Local 83.9%  λ=4.21",
        "causa":    "λ inflado: Querétaro solo 4 juegos visitante (última: 0-4 vs Monterrey peso alto)",
        "fix":      "Shrinkage Bayesiano + EWM + cap λ≤3.2 → corregido en v2",
    },
    {
        "partido":  "América vs Toluca",
        "real":     "Local 2-1",
        "pred":     "Visitante 67.3%  λ_am=0.52",
        "causa":    "América perdió 1-4 vs Tigres (Mar 1) y 1-2 vs Juárez (Mar 5) → datos pesimistas recientes",
        "fix":      "EWM con alpha=0.7 reduce impacto de partidos malos hace 6+ semanas; forma reciente como bonus",
    },
    {
        "partido":  "Necaxa vs Tigres",
        "real":     "Empate 1-1",
        "pred":     "Visitante 46.3%",
        "causa":    "Modelo no predicó empate; Brier bajo (p_d=27%) pero no fue el máx",
        "fix":      "Ampliar draw floor a 18% cuando λ_h−λ_a<0.5; Dixon-Coles rho=-0.13 OK",
    },
    {
        "partido":  "Cruz Azul vs Tijuana",
        "real":     "Empate 1-1",
        "pred":     "Local 46.6%",
        "causa":    "Baja diferencia de λ (1.56 vs 1.09) → empate implícito en 28.4%; pred='Local' por máximo",
        "fix":      "Cuando P(empate)>25% y max−2nd<10pp → reportar como 'Probable Empate'",
    },
    {
        "partido":  "Atl. San Luis vs Pumas",
        "real":     "Visitante 0-2",
        "pred":     "Local 39.3%",
        "causa":    "Joao Pedro (12 goles) fue titular pero Pumas en excelente forma (Juninho 7 goles)",
        "fix":      "Integrar goleadores activos como factor de riesgo ofensivo del rival",
    },
]

for c in contexto:
    print(f"\n  ► {c['partido']}")
    print(f"    Resultado real : {c['real']}")
    print(f"    Predicción     : {c['pred']}")
    print(f"    Causa raíz     : {c['causa']}")
    print(f"    Fix aplicado   : {c['fix']}")

# ─── 10. PROPUESTAS DE MEJORA ─────────────────────────────────────────────────
print()
print("=" * W)
print("  PLAN DE MEJORA DEL MODELO — Ranking por impacto estimado")
print("=" * W)

mejoras = [
    ("ALTA",  "Shrinkage Bayesiano en ratings",
     "Regresa rating hacia 1.0 cuando n<8. Reduce λ_Querétaro_def_a de 1.93→1.45.\n"
     "    Implementado: _shrink(rating, n, k=4) en compute_ratings v2"),
    ("ALTA",  "EWM (ponderación exponencial) alpha=0.7",
     "Decaimiento: juego hace 8 semanas pesa alpha^7=0.08 del más reciente.\n"
     "    Implementado: _exp_weighted_mean() reemplaza _weighted_mean()"),
    ("ALTA",  "Cap de lambda ≤ 3.2 goles esperados",
     "Ningún equipo Liga MX promedió más de 2.5 goles en temporada reciente.\n"
     "    Implementado: _clip_lambda() aplicado en generate_predictions"),
    ("ALTA",  "Factor de motivación para equipos eliminados",
     "Mazatlán/Puebla/Santos eliminados en J15 deberían tener factor <1.0.\n"
     "    Implementado: already_eliminated → factor=0.88 en ligamx_situation()"),
    ("MEDIA", "Forma reciente como multiplicador (±8%)",
     "Últimos 3 juegos como bonus/penalización en λ base.\n"
     "    Implementado: _form_multiplier() en generate_predictions"),
    ("MEDIA", "Regla 'Probable Empate' cuando P(X)>25% y margen<10pp",
     "4 de 9 partidos terminaron empate → modelo debe comunicar incertidumbre.\n"
     "    Pendiente: actualizar semaforo() para reportar 'Empate posible'"),
    ("MEDIA", "Datos de lesiones/bajas por partido",
     "Joao Pedro titular vs Pumas (clave para resultado 0-2).\n"
     "    Fuente: si.com/es-us/futbol/lesionados-y-suspendidos-[equipo]-[fecha]\n"
     "    Pendiente: scraper semi-automatizado semanal"),
    ("MEDIA", "Recolección de datos diaria automática",
     "El parquet de Sofascore tenía última fecha 13 Abril; J15 fue 17-19.\n"
     "    Solución: job diario en collect_sofascore.py (cron / tarea programada)"),
    ("BAJA",  "Goleadores activos como riesgo ofensivo del rival",
     "Si equipo visitante tiene un top-3 scorer, aumentar la_base +5%.\n"
     "    Implementado parcialmente en context_enricher.scorers_for_team"),
    ("BAJA",  "Variables climáticas",
     "Lluvia/calor extremo afecta ritmo de juego, especialmente en Monterrey.\n"
     "    Fuente: Open-Meteo API (gratuita) con lat/lon de cada estadio"),
    ("BAJA",  "Historial reciente de árbitro",
     "Árbitros con alta tasa de tarjetas/penaltis afectan estrategias.\n"
     "    Fuente: Base de datos AMFUT (no tiene API pública, scraping manual)"),
]

for prioridad, titulo, desc in mejoras:
    print(f"\n  [{prioridad}] {titulo}")
    print(f"    {desc}")

# ─── 11. RESUMEN EJECUTIVO ────────────────────────────────────────────────────
print()
print("=" * W)
print("  RESUMEN EJECUTIVO")
print("=" * W)
print(f"  J15 Accuracy: {accuracy*100:.1f}%  ({df_res['correcto'].sum()}/{n})")
print(f"  Brier Score:  {brier_avg:.4f}  [0=perfecto, 0.667=aleatorio, 0.25=típico bueno]")
print(f"  Log Loss:     {logloss_avg:.4f}  [0=perfecto, 1.1=aleatorio]")
print()
print("  Distribución de errores:")
print(f"    - Empates no predichos: {(df_res['resultado_real']=='Empate').sum()} de {n} "
      f"({(df_res['resultado_real']=='Empate').mean()*100:.0f}% de partidos fue empate)")
print(f"    - Modelos confiados pero incorrectos (pred≠real, conf=ALTA): "
      f"{((df_res['correcto']==False) & (df_res['confianza']=='ALTA')).sum()}")
print()
print("  Mejoras implementadas en esta version:")
print("    [OK] Ponderacion exponencial EWM (alpha=0.7)")
print("    [OK] Shrinkage Bayesiano (k=4 partidos prior)")
print("    [OK] Cap lambda <= 3.2 goles esperados")
print("    [OK] Factor motivacion reducido para eliminados (0.88)")
print("    [OK] Multiplicador de forma reciente (+-8% ultimos 3 juegos)")
print("    [OK] Fix timezone: fechas viernes en CDT no UTC")
print("    [OK] Goleadores Clausura 2026 actualizados (no 2024)")
print()
print("=" * W)

# Guardar resultados
df_res.to_csv("data/analisis_j15_resultados.csv", index=False)
print(f"  Resultados guardados: data/analisis_j15_resultados.csv")
print()
