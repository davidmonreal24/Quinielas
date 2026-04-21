# Guía de Instalación en Servidor Dedicado
## Proyecto: Data Analysis Picks — Liga MX + UCL

> **Objetivo:** Desplegar el pipeline de predicciones y el dashboard web en un servidor Linux dedicado, accesible desde cualquier dispositivo y red.  
> **Última actualización:** Abril 2026  
> **Tiempo estimado de instalación:** 45–90 minutos

---

## Índice

1. [Requisitos mínimos de hardware](#1-requisitos-mínimos-de-hardware)
2. [Requisitos de software del servidor](#2-requisitos-de-software-del-servidor)
3. [Arquitectura del sistema en producción](#3-arquitectura-del-sistema-en-producción)
4. [Instalación del sistema operativo](#4-instalación-del-sistema-operativo)
5. [Instalación de dependencias del sistema](#5-instalación-de-dependencias-del-sistema)
6. [Despliegue del proyecto](#6-despliegue-del-proyecto)
7. [Configuración de variables de entorno y claves API](#7-configuración-de-variables-de-entorno-y-claves-api)
8. [Configurar el servidor web (Nginx)](#8-configurar-el-servidor-web-nginx)
9. [Configurar el servicio API (FastAPI + Uvicorn)](#9-configurar-el-servicio-api-fastapi--uvicorn)
10. [Automatización periódica (cron)](#10-automatización-periódica-cron)
11. [Abrir el servidor a internet](#11-abrir-el-servidor-a-internet)
12. [SSL gratuito con Let's Encrypt](#12-ssl-gratuito-con-lets-encrypt)
13. [Comandos de operación diaria](#13-comandos-de-operación-diaria)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Requisitos mínimos de hardware

| Componente | Mínimo recomendado | Notas |
|---|---|---|
| **CPU** | 2 vCPU | El modelo Poisson es ligero; 1 vCPU funciona pero más lento |
| **RAM** | 2 GB | Los parquets y el modelo XGBoost caben holgados |
| **Almacenamiento** | 20 GB SSD | El proyecto ocupa ~500 MB con todos los cachés |
| **Red** | 100 Mbps + IP pública estática | Necesaria para acceso externo |
| **Sistema operativo** | Ubuntu 22.04 LTS o Debian 12 | Cualquier distro Debian-based funciona |

**Proveedores recomendados:** Hetzner Cloud (CX22, ~€4/mes), DigitalOcean Droplet (Basic $6/mes), Contabo VPS (€5/mes). Cualquier VPS con IP pública estática sirve.

---

## 2. Requisitos de software del servidor

| Software | Versión | Rol |
|---|---|---|
| **Python** | 3.10+ (idealmente 3.11) | Runtime de todos los scripts |
| **pip** + **venv** | Incluido en Python 3 | Entorno virtual aislado |
| **Nginx** | 1.18+ | Servidor web / reverse proxy para el dashboard |
| **Git** | 2.x | Para clonar y actualizar el proyecto |
| **Uvicorn** | 0.27+ | Servidor ASGI para `api.py` (FastAPI) |
| **Systemd** | Incluido en Ubuntu/Debian | Gestión de servicios (api.py como daemon) |
| **cron** | Incluido | Automatización de scripts periódicos |
| **curl** / **wget** | Incluido | Verificación de endpoints |

---

## 3. Arquitectura del sistema en producción

```
INTERNET
    │
    │  Puerto 80/443 (HTTP/HTTPS)
    ▼
┌──────────────────────────────────────────────────────────┐
│  NGINX (reverse proxy)                                   │
│                                                          │
│  / → sirve dashboard.html (archivo estático)             │
│  /api/* → proxy_pass http://127.0.0.1:8000               │
└──────────────────────────────────────────────────────────┘
                            │
                            │  Puerto 8000 (interno, solo localhost)
                            ▼
┌──────────────────────────────────────────────────────────┐
│  UVICORN + api.py (FastAPI)                              │
│                                                          │
│  GET /api/picks      → lee ligamx_predicciones.csv       │
│  GET /api/summary    → resumen de predicciones           │
│  GET /api/value-bets → apuestas con edge positivo        │
│  GET /health         → verificación de salud             │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  CRON JOBS (actualizaciones periódicas)                  │
│                                                          │
│  [configurable] collect_sofascore.py  → parquets         │
│  [configurable] predict_ligamx.py    → CSV predicciones  │
│  [configurable] dashboard.py         → dashboard.html    │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│  ARCHIVOS GENERADOS (data/)                              │
│                                                          │
│  sofascore_events.parquet     (histórico de partidos)    │
│  ligamx_predicciones.csv      (74 columnas con modelo)   │
│  dashboard.html               (dashboard web estático)   │
└──────────────────────────────────────────────────────────┘
```

**Flujo de datos en producción:**  
`collect_sofascore.py` → `predict_ligamx.py` → `dashboard.py` → `dashboard.html` servido por Nginx

---

## 4. Instalación del sistema operativo

Se asume acceso SSH al servidor con usuario `root` o usuario con `sudo`.

```bash
# Conectarse al servidor
ssh root@TU_IP_DEL_SERVIDOR

# Actualizar el sistema
apt update && apt upgrade -y

# Crear usuario no-root para correr el proyecto (buena práctica de seguridad)
adduser picks
usermod -aG sudo picks

# Cambiar al usuario nuevo
su - picks
```

---

## 5. Instalación de dependencias del sistema

```bash
# Instalar Python 3.11, git, nginx, y herramientas básicas
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    git nginx curl wget build-essential

# Verificar versiones
python3.11 --version   # debe mostrar Python 3.11.x
nginx -v               # debe mostrar nginx/1.18.x o superior
git --version          # debe mostrar git version 2.x
```

---

## 6. Despliegue del proyecto

### 6.1 Copiar el proyecto al servidor

**Opción A — Desde Git (recomendado si tienes repositorio):**
```bash
cd /home/picks
git clone https://github.com/TU_USUARIO/TU_REPO.git proyecto-quinielas
cd proyecto-quinielas
```

**Opción B — Copiar los archivos manualmente via SCP (desde Windows):**
```powershell
# Ejecutar desde PowerShell en Windows
scp -r "C:\Users\USER\Documents\Proyecto Quinielas\*" picks@TU_IP:/home/picks/proyecto-quinielas/
```

**Opción C — Copiar con rsync (más eficiente, excluye cachés):**
```bash
rsync -avz --exclude='venv/' --exclude='data/_*_cache/' \
    "Proyecto Quinielas/" picks@TU_IP:/home/picks/proyecto-quinielas/
```

### 6.2 Crear el entorno virtual e instalar dependencias

```bash
cd /home/picks/proyecto-quinielas

# Crear entorno virtual
python3.11 -m venv venv

# Activar el entorno virtual
source venv/bin/activate

# Instalar dependencias del proyecto
pip install --upgrade pip
pip install -r requirements.txt

# Instalar dependencias adicionales para el servidor
# (no están en requirements.txt porque en Windows no son necesarias)
pip install fastapi uvicorn[standard] python-multipart aiofiles

# Verificar instalación
python -c "import pandas, numpy, requests; print('OK')"
```

### 6.3 Crear los directorios de datos necesarios

```bash
mkdir -p data/_sofascore_cache data/_espn_cache data/_fbref_cache \
         data/_odds_cache data/_ucl_cache data/_trend_cache
```

### 6.4 Primera ejecución para generar los archivos base

```bash
# Activar entorno si no está activado
source venv/bin/activate

# 1. Recolectar datos históricos (tarda 2-5 minutos la primera vez)
python collect_sofascore.py --history-days 150

# 2. Generar predicciones
python predict_ligamx.py

# 3. Generar dashboard HTML
python dashboard.py

# Verificar que se generaron los archivos
ls -lh data/ligamx_predicciones.csv dashboard.html
```

---

## 7. Configuración de variables de entorno y claves API

> **Importante:** Las claves API actuales están hardcodeadas en los scripts. Para producción, se recomienda moverlas a un archivo `.env` para facilitar rotación de claves sin tocar código.

### 7.1 Crear archivo `.env` con las claves

```bash
cat > /home/picks/proyecto-quinielas/.env << 'EOF'
# ─── APIs de datos ───────────────────────────────────────────
# Sofascore (RapidAPI) — partidos y resultados de todas las ligas
SOFASCORE_RAPIDAPI_KEY=TU_CLAVE_SOFASCORE_RAPIDAPI

# API-Football (RapidAPI) — estadísticas de equipos, tarjetas
APIFOOTBALL_KEY=5cf3eb50762eeb4e9cf15173bae1cb65

# The Odds API — momios UCL (Pinnacle, 500 req/mes gratis)
# Registro gratuito en: https://the-odds-api.com
ODDS_API_KEY=306f7fec9f210e1c341292af655dd0d0

# ─── ESPN Core API ──────────────────────────────────────────
# NO requiere clave — endpoint público de ESPN
# URL: https://sports.core.api.espn.com/v2/sports/soccer/leagues/mex.1/seasons/2025/types/1/leaders

# ─── Altenar/Playdoit API ───────────────────────────────────
# NO requiere clave — endpoint público de Playdoit
# champIds=10009, sportId=66 (Liga MX)
EOF

# Proteger el archivo (solo el usuario puede leerlo)
chmod 600 .env
```

### 7.2 Claves API — resumen y dónde renovarlas

| API | Variable | Dónde obtener nueva clave |
|---|---|---|
| **Sofascore** | `SOFASCORE_RAPIDAPI_KEY` | rapidapi.com → buscar "sofascore6" → Subscribe |
| **API-Football** | `APIFOOTBALL_KEY` | rapidapi.com → API-Football → My Apps |
| **The Odds API** | `ODDS_API_KEY` | the-odds-api.com → My Account |
| **ESPN Core API** | *(sin clave)* | Público, no requiere registro |
| **Altenar/Playdoit** | *(sin clave)* | Público, no requiere registro |

### 7.3 Verificar cuota restante de The Odds API

```bash
# Muestra cuántas peticiones quedan del mes (500 gratis/mes)
curl "https://api.the-odds-api.com/v4/sports?apiKey=TU_CLAVE" \
    -s | python3 -c "import sys; print('OK — respuesta recibida')"
# Los headers x-requests-remaining y x-requests-used muestran el uso
```

---

## 8. Configurar el servidor web (Nginx)

Nginx se encarga de:
- Servir `dashboard.html` como página principal (cualquier dispositivo, cualquier red)
- Redirigir las peticiones `/api/*` al servicio FastAPI interno (puerto 8000)

### 8.1 Crear la configuración de Nginx

```bash
sudo nano /etc/nginx/sites-available/quinielas
```

Pegar el siguiente contenido (reemplazar `TU_IP_O_DOMINIO`):

```nginx
server {
    listen 80;
    server_name TU_IP_O_DOMINIO;   # ← reemplazar con IP pública o dominio

    # Raíz del proyecto — sirve dashboard.html como index
    root /home/picks/proyecto-quinielas;
    index dashboard.html;

    # Servir el dashboard directamente
    location / {
        try_files $uri $uri/ /dashboard.html;
        # Cabeceras para evitar caché del navegador en el dashboard
        add_header Cache-Control "no-cache, must-revalidate";
        add_header Pragma "no-cache";
    }

    # Proxy inverso hacia la API FastAPI
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 60s;
    }

    # Logs
    access_log /var/log/nginx/quinielas_access.log;
    error_log  /var/log/nginx/quinielas_error.log;
}
```

### 8.2 Activar la configuración

```bash
# Habilitar el sitio
sudo ln -s /etc/nginx/sites-available/quinielas /etc/nginx/sites-enabled/

# Desactivar el sitio default de nginx (opcional)
sudo rm -f /etc/nginx/sites-enabled/default

# Verificar sintaxis de la configuración
sudo nginx -t

# Si dice "syntax is ok", reiniciar nginx
sudo systemctl restart nginx
sudo systemctl enable nginx   # inicia automáticamente con el servidor

# Verificar que nginx está corriendo
sudo systemctl status nginx
```

---

## 9. Configurar el servicio API (FastAPI + Uvicorn)

La API (`api.py`) debe correr continuamente en segundo plano y reiniciarse automáticamente si se cae o si el servidor reinicia. Para esto se usa `systemd`.

### 9.1 Crear el archivo de servicio systemd

```bash
sudo nano /etc/systemd/system/quinielas-api.service
```

Pegar el siguiente contenido:

```ini
[Unit]
Description=Data Analysis Picks — API FastAPI
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=picks
WorkingDirectory=/home/picks/proyecto-quinielas
ExecStart=/home/picks/proyecto-quinielas/venv/bin/uvicorn api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --workers 2 \
    --log-level info
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quinielas-api

# Variables de entorno
EnvironmentFile=/home/picks/proyecto-quinielas/.env

[Install]
WantedBy=multi-user.target
```

### 9.2 Activar y arrancar el servicio

```bash
# Recargar la configuración de systemd
sudo systemctl daemon-reload

# Habilitar el servicio (se inicia automáticamente con el servidor)
sudo systemctl enable quinielas-api

# Arrancar el servicio ahora
sudo systemctl start quinielas-api

# Verificar que está corriendo
sudo systemctl status quinielas-api

# Ver los logs en tiempo real
sudo journalctl -u quinielas-api -f
```

### 9.3 Verificar que la API responde

```bash
# Debe devolver JSON con las predicciones
curl http://127.0.0.1:8000/api/picks | python3 -m json.tool | head -30

# Si nginx está configurado, probar desde fuera:
curl http://TU_IP/api/picks
```

---

## 10. Automatización periódica (cron)

Los scripts se ejecutan con `cron`, el programador de tareas nativo de Linux. La frecuencia la define el administrador del servidor según el calendario de cada competencia.

### 10.1 Abrir el crontab del usuario `picks`

```bash
# Activar el crontab del usuario picks
crontab -e
# (la primera vez pregunta el editor; elige 1 para nano)
```

### 10.2 Configuración de cron recomendada

Copiar y pegar el siguiente bloque, **ajustando los horarios** según convenga:

```cron
# ─── Variables de entorno para cron ──────────────────────────────────────────
SHELL=/bin/bash
PATH=/home/picks/proyecto-quinielas/venv/bin:/usr/local/bin:/usr/bin:/bin
PYTHONPATH=/home/picks/proyecto-quinielas
PYTHONIOENCODING=utf-8

# ─── Formato: minuto hora día-del-mes mes día-de-semana comando ──────────────
# Ejemplo: "0 8 * * *" = todos los días a las 08:00 hora del servidor
# Zona horaria: la del servidor (ver con: date && timedatectl)

# ── 1. Recolección diaria de datos (todos los días, 08:00 y 20:00) ────────────
# Descarga partidos + resultados de todas las ligas desde Sofascore
0 8  * * * cd /home/picks/proyecto-quinielas && venv/bin/python collect_sofascore.py --history-days 150 >> logs/collect.log 2>&1
0 20 * * * cd /home/picks/proyecto-quinielas && venv/bin/python collect_sofascore.py --history-days 150 >> logs/collect.log 2>&1

# ── 2. Predicciones Liga MX (viernes 10:00 — día antes de cada jornada) ──────
# Genera ligamx_predicciones.csv con modelo Dixon-Coles completo
0 10 * * 5 cd /home/picks/proyecto-quinielas && venv/bin/python predict_ligamx.py >> logs/predict.log 2>&1

# ── 3. Dashboard HTML (inmediatamente después de las predicciones) ────────────
# Regenera dashboard.html con odds reales de Playdoit + Pinnacle
5 10 * * 5 cd /home/picks/proyecto-quinielas && venv/bin/python dashboard.py >> logs/dashboard.log 2>&1

# ── 4. Actualización de dashboard antes de cada partido (sábado y domingo) ───
# Útil si los momios cambian; regenerar cada 2 horas los fines de semana
0 */2 * * 6,0 cd /home/picks/proyecto-quinielas && venv/bin/python dashboard.py >> logs/dashboard.log 2>&1

# ── 5. (Opcional) Reporte Playdoit en terminal ────────────────────────────────
# Guardar el output del reporte semanal en un archivo para revisión
0 10 * * 5 cd /home/picks/proyecto-quinielas && venv/bin/python _reporte_playdoit.py --days 7 >> logs/reporte.log 2>&1
```

> **Importante:** La zona horaria de `cron` es la del sistema. Para saber qué zona tiene el servidor:
> ```bash
> timedatectl
> # Si no es CDT/CST, ajustar horas arriba o cambiar zona:
> sudo timedatectl set-timezone America/Mexico_City
> ```

### 10.3 Crear la carpeta de logs

```bash
mkdir -p /home/picks/proyecto-quinielas/logs
```

### 10.4 Verificar que cron está activo y los jobs corren

```bash
# Listar los cron jobs configurados
crontab -l

# Ver el log del sistema de cron para confirmar ejecuciones
grep CRON /var/log/syslog | tail -20

# Ver el log de collect (después de la primera ejecución automática)
tail -50 /home/picks/proyecto-quinielas/logs/collect.log
```

### 10.5 Ejecutar un script manualmente fuera de cron (para pruebas)

```bash
cd /home/picks/proyecto-quinielas
source venv/bin/activate

# Ejecutar cualquier script manualmente
python collect_sofascore.py --history-days 150
python predict_ligamx.py
python dashboard.py

# Reiniciar la API después de cambios en el código
sudo systemctl restart quinielas-api
```

---

## 11. Abrir el servidor a internet

### 11.1 Configurar el firewall del servidor (UFW)

```bash
# Habilitar UFW si no está activo
sudo ufw enable

# Permitir SSH (IMPORTANTE: hacerlo antes de activar UFW para no perder acceso)
sudo ufw allow ssh          # puerto 22

# Permitir HTTP y HTTPS (para el dashboard)
sudo ufw allow http         # puerto 80
sudo ufw allow https        # puerto 443

# El puerto 8000 (API interna) NO debe abrirse al exterior
# Nginx actúa como intermediario

# Verificar reglas
sudo ufw status verbose
```

### 11.2 Firewall en el panel del proveedor cloud

La mayoría de proveedores (Hetzner, DigitalOcean, Contabo) tienen un firewall adicional en su panel de control. Verificar que los puertos **22, 80 y 443** están abiertos hacia `0.0.0.0/0` (cualquier IP).

### 11.3 Verificar acceso desde fuera

```bash
# Desde cualquier máquina en internet:
curl http://TU_IP_PUBLICA/          # debe devolver HTML del dashboard
curl http://TU_IP_PUBLICA/api/picks  # debe devolver JSON
```

### 11.4 Configurar un dominio (opcional pero recomendado)

Si se tiene un dominio (ej. `picks.ejemplo.com`), apuntar su registro DNS tipo A a la IP del servidor. Luego actualizar la directiva `server_name` en el archivo de Nginx:

```nginx
server_name picks.ejemplo.com;
```

Y recargar nginx: `sudo systemctl reload nginx`

---

## 12. SSL gratuito con Let's Encrypt

SSL cifra la comunicación y es necesario si se accede desde redes corporativas o dispositivos móviles con restricciones. **Requiere tener un dominio apuntando al servidor** (no funciona con IP pública directa).

```bash
# Instalar Certbot
sudo apt install -y certbot python3-certbot-nginx

# Obtener e instalar certificado SSL automáticamente
sudo certbot --nginx -d picks.ejemplo.com

# Certbot preguntará un email y si redirigir HTTP→HTTPS (recomendado: sí)
# El certificado se renueva automáticamente cada 90 días
```

Verificar la renovación automática:

```bash
sudo certbot renew --dry-run   # debe decir "congratulations, all simulated..."
```

---

## 13. Comandos de operación diaria

### Ver el estado general del sistema

```bash
# Estado de todos los servicios relevantes
sudo systemctl status quinielas-api nginx

# Ver uso de CPU/RAM en tiempo real
htop   # (instalar con: sudo apt install htop)

# Ver espacio en disco
df -h /home/picks/proyecto-quinielas/
```

### Actualizar el código del proyecto

```bash
cd /home/picks/proyecto-quinielas

# Si usas Git:
git pull origin main

# Reiniciar la API para aplicar cambios
sudo systemctl restart quinielas-api

# Regenerar el dashboard con el código actualizado
source venv/bin/activate
python dashboard.py
```

### Forzar actualización del dashboard manualmente

```bash
cd /home/picks/proyecto-quinielas && source venv/bin/activate
python collect_sofascore.py && python predict_ligamx.py && python dashboard.py
echo "Dashboard actualizado: $(date)"
```

### Consultar los logs

```bash
# Logs de la API FastAPI (en tiempo real)
sudo journalctl -u quinielas-api -f

# Logs de nginx (accesos y errores)
sudo tail -f /var/log/nginx/quinielas_access.log
sudo tail -f /var/log/nginx/quinielas_error.log

# Logs de los scripts (generados por cron)
tail -100 /home/picks/proyecto-quinielas/logs/collect.log
tail -100 /home/picks/proyecto-quinielas/logs/predict.log
tail -100 /home/picks/proyecto-quinielas/logs/dashboard.log
```

### Reiniciar servicios después de un reboot del servidor

```bash
# Si nginx y quinielas-api tienen "enable", arrancan solos.
# Para verificar:
sudo systemctl is-enabled nginx quinielas-api
# debe decir "enabled" para ambos

# Si no arrancaron solos:
sudo systemctl start nginx quinielas-api
```

---

## 14. Troubleshooting

### El dashboard no carga (pantalla en blanco o error 502)

```bash
# 1. Verificar que nginx está corriendo
sudo systemctl status nginx

# 2. Verificar que la API está corriendo
sudo systemctl status quinielas-api

# 3. Ver el error específico de nginx
sudo tail -20 /var/log/nginx/quinielas_error.log

# 4. Verificar que el archivo dashboard.html existe
ls -lh /home/picks/proyecto-quinielas/dashboard.html

# Solución común: la API no arrancó → reiniciarla
sudo systemctl restart quinielas-api
```

### Error 502 Bad Gateway en /api/picks

```bash
# La API no está respondiendo en el puerto 8000
curl http://127.0.0.1:8000/api/picks

# Ver logs de la API para identificar el error
sudo journalctl -u quinielas-api --since "10 minutes ago"

# Causa común: error en api.py al leer el CSV
# Verificar que el CSV existe y tiene datos:
python3 -c "import pandas as pd; print(pd.read_csv('data/ligamx_predicciones.csv').shape)"
```

### Los cron jobs no ejecutan

```bash
# Verificar que cron está activo
sudo systemctl status cron

# Ver si cron está intentando ejecutar los jobs
grep CRON /var/log/syslog | tail -20

# Problema común: PATH incompleto en cron
# Solución: usar rutas absolutas en el crontab, como se muestra en la sección 10.2
```

### El script collect_sofascore.py falla con "quota exceeded"

La API de Sofascore en RapidAPI tiene límites mensuales. Si se excede:
```bash
# Ver el error exacto
python collect_sofascore.py 2>&1 | grep -i "quota\|limit\|429"

# Opciones:
# 1. Reducir frecuencia de recolección en cron (de 2 veces/día a 1 vez/día)
# 2. Ampliar el plan en RapidAPI (rapidapi.com → My Apps → Pricing)
# 3. El proyecto funciona con datos del día anterior mientras tanto
```

### La API de Sofascore devuelve datos vacíos para Liga MX

```bash
# Verificar manualmente la respuesta de la API
source venv/bin/activate
python3 -c "
import requests
r = requests.get(
    'https://sofascore6.p.rapidapi.com/api/sofascore/v1/match/list',
    params={'sport_slug': 'football', 'date': '2026-05-01'},
    headers={'x-rapidapi-key': 'TU_CLAVE', 'x-rapidapi-host': 'sofascore6.p.rapidapi.com'}
)
print(r.status_code, r.text[:300])
"
```

### Permisos denegados en archivos de datos

```bash
# Asegurarse que el usuario picks es dueño de todos los archivos
sudo chown -R picks:picks /home/picks/proyecto-quinielas/
sudo chmod -R 755 /home/picks/proyecto-quinielas/
```

### El servidor corre bien pero no es accesible desde internet

```bash
# 1. Verificar que UFW permite el tráfico
sudo ufw status

# 2. Verificar que nginx escucha en la IP pública (no solo localhost)
sudo ss -tlnp | grep nginx

# 3. Probar desde el mismo servidor (simula petición externa)
curl -v http://$(curl -s ifconfig.me)/

# 4. Si el proveedor cloud tiene un firewall separado, verificar en su panel de control
```

---

## Resumen rápido (checklist de instalación)

```
□ 1. Servidor Ubuntu 22.04 accesible por SSH
□ 2. apt install python3.11 python3.11-venv git nginx
□ 3. Copiar proyecto al servidor (/home/picks/proyecto-quinielas)
□ 4. python3.11 -m venv venv && pip install -r requirements.txt
□ 5. pip install fastapi "uvicorn[standard]"
□ 6. Crear .env con las claves API
□ 7. Primera ejecución: collect → predict → dashboard
□ 8. Configurar nginx (/etc/nginx/sites-available/quinielas)
□ 9. Crear servicio systemd (quinielas-api.service) y habilitarlo
□ 10. Configurar crontab con la frecuencia deseada
□ 11. Abrir puertos 22, 80, 443 en UFW y en panel del proveedor
□ 12. (Opcional) Certificado SSL con certbot si se tiene dominio
□ 13. Verificar: curl http://TU_IP/ devuelve el dashboard
□ 14. Verificar: curl http://TU_IP/api/picks devuelve JSON
```

---

*Documento generado para el equipo de Data Analysis Picks — Proyecto Quinielas Liga MX + UCL*
