"""
=====================================================================
  Modelo Baseline — Soccer Project
  Predice: Local / Empate / Visitante
  Algoritmos: Logistic Regression + Random Forest + XGBoost
  Input:  data/processed/features.parquet
  Output: data/models/baseline_report.txt
  Uso:    python train_model.py
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/processed")
MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGAR Y PREPARAR FEATURES
# ─────────────────────────────────────────────

print("=" * 60)
print("  SOCCER PROJECT — Entrenamiento de Modelo Baseline")
print("=" * 60)

df = pd.read_parquet(DATA_DIR / "features.parquet")
print(f"\nDataset cargado: {df.shape[0]} partidos x {df.shape[1]} variables")

# Features numéricas para el modelo
NUMERIC_FEATURES = [
    "home_xg", "away_xg", "xg_diff", "xg_total", "xg_ratio_local",
    "home_roll_xg_for", "home_roll_xg_against",
    "away_roll_xg_for",  "away_roll_xg_against",
    "form_xg_diff", "form_def_diff",
    "home_squad_np_xg", "home_squad_xa", "home_squad_xg_chain",
    "away_squad_np_xg", "away_squad_xa", "away_squad_xg_chain",
    "squad_xg_diff", "squad_xa_diff",
    "home_table_pos", "away_table_pos", "table_pos_diff", "pts_diff",
    "es_local",
]

# Filtrar solo columnas disponibles
FEATURES = [c for c in NUMERIC_FEATURES if c in df.columns]
TARGET   = "resultado"

# Eliminar filas con NaN en features clave
df_model = df[FEATURES + [TARGET, "resultado_label", "season", "league"]].dropna(subset=FEATURES)
print(f"Partidos tras limpiar nulos: {len(df_model)}")

X = df_model[FEATURES].values
y = df_model[TARGET].values  # -1, 0, 1

# Encode para XGBoost (requiere 0,1,2)
le = LabelEncoder()
y_enc = le.fit_transform(y)  # -1→0, 0→1, 1→2

print(f"\nFeatures utilizadas: {len(FEATURES)}")
print(f"Clases: {dict(zip(le.classes_, range(len(le.classes_))))}")


# ─────────────────────────────────────────────
# 2. SPLIT TEMPORAL (no aleatorio — respeta orden cronológico)
# ─────────────────────────────────────────────

# Usar temporadas antiguas para train, la más reciente para test
test_mask  = df_model["season"].isin(["2526", "2425"])
train_mask = ~test_mask

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
y_enc_train, y_enc_test = y_enc[train_mask], y_enc[test_mask]

print(f"\nSplit temporal:")
print(f"  Train: {len(X_train)} partidos ({df_model[train_mask]['season'].unique()})")
print(f"  Test:  {len(X_test)}  partidos ({df_model[test_mask]['season'].unique()})")

# Escalar para Logistic Regression
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 3. ENTRENAR 3 MODELOS
# ─────────────────────────────────────────────

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced",
        random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1
    ),
}

results = {}
print("\n" + "─" * 60)
print("  Entrenando modelos...")
print("─" * 60)

for name, model in models.items():
    if name == "XGBoost":
        model.fit(X_train, y_enc_train)
        y_pred_enc = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_enc)
        cv = cross_val_score(model, X_train, y_enc_train, cv=5, scoring="accuracy")
    elif name == "Logistic Regression":
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)
        cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring="accuracy")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model":  model,
        "y_pred": y_pred,
        "acc":    acc,
        "cv_mean": cv.mean(),
        "cv_std":  cv.std(),
    }
    print(f"\n  {name}")
    print(f"    Accuracy test:       {acc:.3f} ({acc*100:.1f}%)")
    print(f"    CV accuracy (5-fold): {cv.mean():.3f} ± {cv.std():.3f}")

# Baseline ingenuo (siempre predice "Local")
naive_acc = (y_test == 1).mean()
print(f"\n  Baseline ingenuo (siempre Local): {naive_acc:.3f} ({naive_acc*100:.1f}%)")


# ─────────────────────────────────────────────
# 4. MODELO PRE-PARTIDO (sin xG in-match — para predicciones futuras)
# ─────────────────────────────────────────────
# Estas features son conocidas ANTES del partido (no hay data leakage).
# Es el modelo que usa predict_upcoming.py.

PRE_MATCH_FEATURES = [
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

PM_FEATURES = [c for c in PRE_MATCH_FEATURES if c in df_model.columns]

# XGBoost maneja NaN internamente — no dropna para no perder partidos sin H2H
X_pm     = df_model[PM_FEATURES].values
X_pm_train, X_pm_test = X_pm[train_mask], X_pm[test_mask]

pm_model = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1,
)
pm_model.fit(X_pm_train, y_enc_train)
y_pm_pred_enc = pm_model.predict(X_pm_test)
y_pm_pred     = le.inverse_transform(y_pm_pred_enc)
pm_acc        = accuracy_score(y_test, y_pm_pred)
pm_cv         = cross_val_score(pm_model, X_pm_train, y_enc_train, cv=5, scoring="accuracy")

print(f"\n  XGBoost Pre-Partido  ({len(PM_FEATURES)} features, sin xG in-match)")
print(f"    Accuracy test:       {pm_acc:.3f} ({pm_acc*100:.1f}%)")
print(f"    CV accuracy (5-fold): {pm_cv.mean():.3f} ± {pm_cv.std():.3f}")
print(f"    [Este modelo se usa en predict_upcoming.py]")

# Guardar modelo pre-partido
joblib.dump(pm_model,   MODELS_DIR / "pre_match_model.pkl")
joblib.dump(PM_FEATURES, MODELS_DIR / "pre_match_feature_names.pkl")
print(f"  Guardado: data/models/pre_match_model.pkl")


# ─────────────────────────────────────────────
# 6. REPORTE DETALLADO DEL MEJOR MODELO
# ─────────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["acc"])
best      = results[best_name]
label_map = {-1: "Visitante", 0: "Empate", 1: "Local"}

print(f"\n{'='*60}")
print(f"  MEJOR MODELO (con xG): {best_name} ({best['acc']*100:.1f}%)")
print(f"{'='*60}")
print(classification_report(y_test, best["y_pred"],
      target_names=["Visitante", "Empate", "Local"]))

print(f"\n{'='*60}")
print(f"  MODELO PRE-PARTIDO:  XGBoost Pre-Partido ({pm_acc*100:.1f}%)")
print(f"{'='*60}")
print(classification_report(y_test, y_pm_pred,
      target_names=["Visitante", "Empate", "Local"]))


# ─────────────────────────────────────────────
# 7. VISUALIZACIONES
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Soccer Project — Resultados del Modelo Baseline", fontsize=14, fontweight="bold", y=1.02)

# ── Grafica 1: Comparación de modelos (incluye pre-partido y baseline) ──
model_names  = list(results.keys()) + ["XGB\nPre-Partido", "Baseline\nIngenuo"]
model_scores = [r["acc"] for r in results.values()] + [pm_acc, naive_acc]
colors = ["#2E75B6", "#1E6B3C", "#B85C00", "#7B2D8B", "#888888"]

bars = axes[0].bar(model_names, model_scores, color=colors, edgecolor="white", linewidth=1.5)
axes[0].axhline(naive_acc, color="red", linestyle="--", linewidth=1, alpha=0.5)
axes[0].set_title("Comparacion de Modelos", fontweight="bold")
axes[0].set_ylabel("Accuracy")
axes[0].set_ylim(0.3, 0.7)
axes[0].set_yticks([0.3, 0.4, 0.44, 0.5, 0.6, 0.7])
for bar, score in zip(bars, model_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

# ── Grafica 2: Matriz de confusión del mejor modelo ──
cm = confusion_matrix(y_test, best["y_pred"], labels=[-1, 0, 1])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Visitante","Empate","Local"],
            yticklabels=["Visitante","Empate","Local"],
            linewidths=0.5, linecolor="white")
axes[1].set_title(f"Matriz de Confusion\n{best_name}", fontweight="bold")
axes[1].set_xlabel("Prediccion")
axes[1].set_ylabel("Real")

# ── Grafica 3: Feature importance (si es RF o XGB) ──
if best_name in ["Random Forest", "XGBoost"]:
    importances = best["model"].feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURES, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(12)
    feat_df.plot(kind="barh", x="feature", y="importance", ax=axes[2],
                 color="#2E75B6", legend=False)
    axes[2].set_title(f"Top Features\n{best_name}", fontweight="bold")
    axes[2].set_xlabel("Importancia")
else:
    coefs = np.abs(best["model"].coef_).mean(axis=0)
    feat_df = pd.DataFrame({"feature": FEATURES, "coef": coefs})
    feat_df = feat_df.sort_values("coef", ascending=True).tail(12)
    feat_df.plot(kind="barh", x="feature", y="coef", ax=axes[2],
                 color="#2E75B6", legend=False)
    axes[2].set_title(f"Coeficientes\n{best_name}", fontweight="bold")
    axes[2].set_xlabel("Importancia (|coef|)")

plt.tight_layout()
plt.savefig("data/modelo_baseline.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGrafica guardada en: data/modelo_baseline.png")


# ─────────────────────────────────────────────
# 8. GUARDAR MEJOR MODELO (con xG)
# ─────────────────────────────────────────────

joblib.dump(best["model"], MODELS_DIR / f"best_model.pkl")
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
joblib.dump(FEATURES, MODELS_DIR / "feature_names.pkl")

print(f"\nModelo guardado en: data/models/best_model.pkl")
print(f"\n{'='*60}")
print(f"  LISTO.")
print(f"  - Modelo con xG:     data/models/best_model.pkl")
print(f"  - Modelo pre-partido: data/models/pre_match_model.pkl")
print(f"  Usa predict_upcoming.py para predecir proximos partidos.")
print(f"{'='*60}")
