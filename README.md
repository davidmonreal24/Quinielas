# ⚽ Soccer Data Project — Guía de Setup y Pruebas Locales

## Estructura del proyecto

```
soccer_project/
├── collect_data.py     ← Script principal de recolección de datos
├── test_setup.py       ← Verificar instalación antes de empezar
├── requirements.txt    ← Todas las dependencias
├── .env                ← Variables de entorno (crear manualmente)
└── data/
    └── processed/      ← Datasets generados (parquet/csv)
```

---

## ⚙️ Setup local paso a paso

### Paso 1 — Clonar / crear el proyecto

```bash
mkdir soccer_project
cd soccer_project
```

### Paso 2 — Crear entorno virtual (muy recomendado)

```bash
# Crear entorno virtual
python -m venv venv

# Activar en Mac/Linux:
source venv/bin/activate

# Activar en Windows:
venv\Scripts\activate
```

> 💡 Siempre verifica que el entorno está activo: verás `(venv)` al inicio del prompt.

### Paso 3 — Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⏱️ La primera instalación puede tomar 2-3 minutos.

### Paso 4 — Configurar variables de entorno (opcional)

Crea un archivo `.env` en la raíz del proyecto:

```bash
# .env
SOCCERDATA_DIR=./data          # Directorio de caché
SOCCERDATA_LOGLEVEL=INFO       # DEBUG para más detalle
SOCCERDATA_NOCACHE=False       # True para forzar re-descarga
SOCCERDATA_NOSTORE=False       # True para no guardar caché
```

---

## 🧪 Pruebas locales

### Prueba 1 — Verificar instalación (ejecuta esto primero)

```bash
python test_setup.py
```

Deberías ver algo así:
```
=======================================================
  SOCCER DATA — Test de instalación
=======================================================
[1/5] Python version: 3.11.x ✓
[2/5] Importando librerías... ✓
[3/5] Versiones: pandas=2.x.x, numpy=1.x.x, soccerdata=1.8.x ✓
[4/5] Probando conexión a FBref... ✓  (380 partidos descargados)
[5/5] Vista previa de datos:
...
✅ Todo listo. Puedes ejecutar collect_data.py
```

### Prueba 2 — Descarga completa

```bash
python collect_data.py
```

La primera ejecución descargará los datos de internet (~5-15 min según temporadas).
Las siguientes ejecuciones usarán la caché local y serán mucho más rápidas.

### Prueba 3 — Modo interactivo con Jupyter

```bash
jupyter lab
```

Luego en una celda:
```python
import soccerdata as sd
import pandas as pd

# Probar una descarga rápida
fbref = sd.FBref("ENG-Premier League", "2023-24")
df = fbref.read_schedule()
df.head()
```

---

## 📊 Qué datos obtienes

| Dataset               | Fuente         | Filas aprox. | Uso principal         |
|-----------------------|----------------|-------------|----------------------|
| schedule              | FBref          | ~380/liga   | Resultados partidos  |
| team_season_standard  | FBref          | ~60/liga    | Stats generales equipo|
| team_season_shooting  | FBref          | ~60/liga    | xG, goles, tiros     |
| team_season_passing   | FBref          | ~60/liga    | Pases, progresión    |
| team_season_defense   | FBref          | ~60/liga    | Presión, tackles     |
| player_season_*       | FBref          | ~600/liga   | Stats individuales   |
| schedule_xg           | Understat      | ~380/liga   | xG por partido       |
| schedule_odds         | Football-Data  | ~380/liga   | Histórico + cuotas   |
| elo_history           | Club Elo       | miles       | Rankings históricos  |

---

## ⚠️ Notas importantes

1. **Rate limiting**: soccerdata tiene pausas automáticas entre requests. No hagas scraping masivo en una sola sesión.
2. **Caché local**: los datos se guardan en `~/soccerdata` (o el directorio que configures). No los borres entre sesiones.
3. **Datos en tiempo real**: soccerdata NO provee datos en vivo. Para eso usa API-Football (plan gratuito).
4. **Respeto a las fuentes**: úsalo para uso personal/investigación. No redistribuyas los datos.

---

## 🔜 Próximos pasos sugeridos

- [ ] Explorar los datasets en Jupyter Lab
- [ ] Identificar features clave para el modelo predictivo
- [ ] Construir pipeline de limpieza y feature engineering
- [ ] Entrenar primer modelo baseline (ej. predecir resultado: local/empate/visitante)
