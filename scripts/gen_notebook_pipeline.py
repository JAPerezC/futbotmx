"""Genera el notebook educativo notebooks/01_pipeline_paso_a_paso.ipynb.

Pipeline futbolístico paso a paso sobre IMG_9852 (22s, 1 gol real en t=18.01s).
Usa los outputs ya procesados en data/processed/runs/IMG_9852_v3/ como
"laboratorio" — el notebook NO invoca SAM 3.1 (que requiere GPU + 10-15s
de carga), pero EXPLICA su uso y carga las predicciones precomputadas.

Se reescribe el notebook completo cada vez. Para regenerar:
    python scripts/gen_notebook_pipeline.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "01_pipeline_paso_a_paso.ipynb"


def md(*lines: str) -> dict:
    """Construye una celda markdown."""
    src = "\n".join(lines)
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.split("\n", -1) if "\n" in src else [src],
    }


def md_block(text: str) -> dict:
    """Celda markdown desde un bloque de texto multilínea."""
    lines = text.strip("\n").splitlines(keepends=True)
    # Asegurar último no tiene \n para evitar línea vacía extra en algunos renderers.
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(text: str) -> dict:
    """Celda de código desde un bloque de texto."""
    lines = text.strip("\n").splitlines(keepends=True)
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def build_cells() -> list[dict]:
    cells: list[dict] = []

    # ------------ Portada ------------
    cells.append(
        md_block(
            """
# Pipeline futbotmx paso a paso — IMG_9852

**Categoría Profesional · Copa FutBotMX 2026** · AJOLOTES FC

Este cuaderno explica el pipeline completo de análisis de un partido de fútbol
robótico, recorriendo cada etapa sobre un clip real del dataset oficial
(`IMG_9852.MOV`, 22 s, con 1 gol en `t=18.01 s`).

## Por qué este notebook

La convocatoria pide repositorio público, código reproducible y resultados
con capturas/GIFs. Este cuaderno cumple esos tres objetivos en un único lugar:

1. **Reproducible**: corre celda por celda en CPU (~30 s en total). No
   carga SAM 3.1 — usa las máscaras precomputadas en
   `data/processed/runs/IMG_9852_v3/` como "laboratorio".
2. **Didáctico**: cada paso del pipeline tiene una sección con motivación,
   código mínimo y visualización.
3. **Honesto**: incluye limitaciones detectadas (artefactos del fallback HSV
   del balón) y enlaces a la solución actual (detector AND `ball_inside_goal_field`
   + `ball_inside_goal`, commit `5ca5c60`).

## Mapa de la sesión

| § | Tema | Salida |
|---|------|--------|
| 0 | Setup | imports + paths |
| 1 | Inspección del video | metadata + frame inicial |
| 2 | Detector del campo (cascade) | 4 esquinas overlay |
| 3 | Porterías por HSV | bboxes amarilla + azul |
| 4 | Homografía 4-puntos | top-down sintetizado |
| 5 | Segmentación SAM 3.1 (conceptual) | máscaras precomputadas |
| 6 | Tracking OC-SORT + Kalman | trayectorias |
| 7 | Re-ID adaptativo por color | clusters HSV |
| 8 | Detección de eventos + gol AND | timeline + highlight |
| 9 | Estadísticas + visualizaciones | dashboard, heatmaps |
| 10 | Conclusión + próximos pasos | cómo correr el pipeline real |
"""
        )
    )

    # ------------ § 0 Setup ------------
    cells.append(
        md_block(
            """
## § 0 — Setup

Importamos las dependencias del proyecto y fijamos los paths. El cuaderno
asume que ya existe el venv (`python -m venv .venv`) y que se instalaron
las dependencias con `pip install -r requirements.txt`.

Si abres este cuaderno desde la raíz del repo, no hace falta cambiar nada.
"""
        )
    )

    cells.append(
        code(
            """
from pathlib import Path
import sys

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["figure.dpi"] = 100

VIDEO_PATH = ROOT / "data" / "raw" / "drive_oficial" / "17Abril" / "Cámaras" / "IMG_9852.MOV"
RUN_DIR = ROOT / "data" / "processed" / "runs" / "IMG_9852_v3"

assert VIDEO_PATH.exists(), f"Falta el video: {VIDEO_PATH}"
assert RUN_DIR.exists(), f"Falta el run procesado: {RUN_DIR}"

print(f"Root del repo: {ROOT}")
print(f"Video:         {VIDEO_PATH.name}")
print(f"Run:           {RUN_DIR.name}")
"""
        )
    )

    # ------------ § 1 Inspección ------------
    cells.append(
        md_block(
            """
## § 1 — Inspección del video

Antes de procesar, **siempre** revisamos metadata y un frame de muestra. Esto
nos dice si la cámara es vertical (portrait egocéntrica de Meta Ray-Ban) u
horizontal (espectador con iPhone), qué fps tiene y a qué resolución vamos
a trabajar.

`src.utils.io.probe` envuelve `ffprobe` para devolver una estructura tipada.
"""
        )
    )

    cells.append(
        code(
            """
from src.utils.io import probe, read_frames

meta = probe(VIDEO_PATH)
print(f"Resolución:  {meta.width}x{meta.height}")
print(f"FPS:         {meta.fps:.2f}")
print(f"Duración:    {meta.duration_s:.2f} s")
print(f"Frames tot.: {int(meta.duration_s * meta.fps)}")

# Sacamos el primer frame leíble (stride=1 para no saltar al principio)
first_frame = None
for idx, frame in read_frames(VIDEO_PATH, stride=1):
    first_frame = frame
    break

# OpenCV trabaja en BGR; matplotlib en RGB.
plt.imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
plt.title(f"Frame 0 de {VIDEO_PATH.name}")
plt.axis("off")
plt.show()
"""
        )
    )

    # ------------ § 2 Campo ------------
    cells.append(
        md_block(
            """
## § 2 — Detector del campo (cascade)

El reglamento define el área de juego como un rectángulo con **líneas
blancas** y escuadras blancas en las esquinas. Nuestro detector v2 usa una
cascada de 3 métodos, de más preciso a más robusto:

1. **Hough sobre líneas blancas**: encuentra las sidelines como segmentos largos
   y clusteriza por ángulo. Esquinas = intersección de las 2 parejas de líneas.
2. **Convex hull de la máscara blanca** (cuando 1 falla): dilata la máscara,
   toma el casco convexo y elige el vértice más lejano del centroide por cuadrante.
3. **Convex hull del fieltro verde** (último recurso): aproximación gruesa cuando
   las líneas blancas están muy fragmentadas.

`src.utils.field_detect_v2.detect_field_geometry` devuelve un `FieldGeometryV2`
con las 4 esquinas, las porterías detectadas y un dict de debug.
"""
        )
    )

    cells.append(
        code(
            """
from src.utils.field_detect_v2 import detect_field_geometry

# IMG_9852 calibra bien con el primer frame; en otros videos el bucle de
# calibrate_homography prueba varios frames hasta dar con uno limpio.
geom = detect_field_geometry(first_frame)
print(f"Detector usado: {geom.debug.get('method')}")
print(f"Esquinas detectadas: {geom.corners_img.tolist()}")
print(f"Porterías detectadas: {[g.color for g in geom.goals]}")

# Overlay del cuadrilátero del campo + porterías
img = cv2.cvtColor(first_frame.copy(), cv2.COLOR_BGR2RGB)
pts = geom.corners_img.astype(int)
for i in range(4):
    cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 4)
    cv2.circle(img, tuple(pts[i]), 12, (255, 215, 0), -1)
for g in geom.goals:
    x1, y1, x2, y2 = [int(v) for v in g.bbox_xyxy]
    color = (255, 200, 0) if g.color == "yellow" else (60, 80, 200)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

plt.imshow(img)
plt.title("Esquinas del campo (verde) + porterías (amarilla/azul)")
plt.axis("off")
plt.show()
"""
        )
    )

    # ------------ § 3 Porterías ------------
    cells.append(
        md_block(
            """
## § 3 — Porterías por color (HSV)

Las cajas de portería en el reglamento robótico son **amarilla** y **azul**,
saturadas y brillantes. Las detectamos con `cv2.inRange` sobre HSV,
restringiendo el ROI al fieltro verde dilatado (para no captar cosas amarillas
fuera del campo, como banderines de equipos).

Trucos importantes:
- Aceptamos azul muy oscuro (casi negro) porque la perspectiva oblicua suele
  oscurecer la portería del fondo.
- Dilatación generosa `75×75` + cierre morfológico para reconectar la
  portería cuando una mano del público la oculta parcialmente.
- Área mínima distinta para cada color (más permisiva en azul).
"""
        )
    )

    cells.append(
        code(
            """
from src.utils.field_detect_v2 import detect_goals_by_color

goals = detect_goals_by_color(first_frame)
for g in goals:
    cx, cy = g.centroid_img
    print(f"  {g.color:6s} bbox={g.bbox_xyxy.tolist()} centroid=({cx:.0f},{cy:.0f}) area={g.area_px} px²")

# Visualizamos las máscaras HSV crudas para entender cómo se ven
hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
yellow_lo = np.array([18, 110, 110], dtype=np.uint8)
yellow_hi = np.array([38, 255, 255], dtype=np.uint8)
blue_lo = np.array([95, 50, 30], dtype=np.uint8)
blue_hi = np.array([135, 255, 255], dtype=np.uint8)

mask_y = cv2.inRange(hsv, yellow_lo, yellow_hi)
mask_b = cv2.inRange(hsv, blue_lo, blue_hi)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
axes[0].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original")
axes[1].imshow(mask_y, cmap="gray")
axes[1].set_title("Máscara amarilla (HSV)")
axes[2].imshow(mask_b, cmap="gray")
axes[2].set_title("Máscara azul (HSV)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        )
    )

    # ------------ § 4 Homografía ------------
    cells.append(
        md_block(
            """
## § 4 — Homografía 4-puntos al top-down

Una vez tenemos las 4 esquinas en imagen, calculamos la **homografía** que
mapea cualquier punto de la imagen al sistema de coordenadas **mundo** del
campo, en milímetros. El reglamento dice que el campo de juego mide
`2190 × 1580 mm`, así que ese rectángulo es nuestro destino.

`compute_homography(corners_tl_tr_br_bl)` usa `cv2.findHomography` con las
4 correspondencias. Después, `project_points(points_img, H)` proyecta cualquier
punto de la imagen al mundo y devuelve milímetros.

Aplicaciones inmediatas:
- Top-down sintetizado (warp del frame) — base del minimap animado.
- Coordenadas mundo del balón → estadísticas (distancia, velocidad).
- Distancias entre robots → detección de colisiones, posesión.
"""
        )
    )

    cells.append(
        code(
            """
from src.utils.calib import compute_homography, project_points

H = compute_homography(geom.corners_img)
print("Matriz de homografía (3x3):")
print(H)

# Warp el frame al top-down (1095 x 790 px, factor 0.5 mm/px)
field_length_mm = 2190
field_width_mm = 1580
scale = 0.5  # 1 px = 2 mm de campo
W_out = int(field_length_mm * scale)
H_out = int(field_width_mm * scale)

# La homografía mapea imagen -> mm. Para warp visual queremos imagen -> px de top-down,
# así que escalamos con una matriz extra.
S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
H_to_px = S @ H

top_down = cv2.warpPerspective(first_frame, H_to_px, (W_out, H_out))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
axes[0].set_title("Imagen original (perspectiva oblicua del espectador)")
axes[1].imshow(cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB))
axes[1].set_title("Top-down sintetizado vía homografía")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        )
    )

    # ------------ § 5 SAM 3.1 ------------
    cells.append(
        md_block(
            """
## § 5 — Segmentación con SAM 3.1 (conceptual)

SAM 3.1 (Segment Anything Model 3.1, Meta AI) es el modelo base obligatorio de
la categoría Profesional. Acepta **prompts de texto**, **box** o **puntos** y
devuelve máscaras de segmentación con score.

En el pipeline real (`scripts/run_pipeline.py`) usamos:
```python
seg = segment_with_text(frame, ["soccer ball", "small mobile robot with a colored flag"], processor, model, threshold=0.2)
```

Hallazgos clave (documentados en `src/segmentation/prompts.py`):
- Prompts SIMPLES (`"soccer robot"`) dan scores ~0.94. Prompts elaborados
  como `"small mobile soccer robot with a colored flag"` bajan a 0.34.
- Threshold 0.2 maximiza recall sin disparar FP.
- Filtros post-procesamiento: score ≥ 0.5, área entre 0.1% y 5% del frame,
  aspect ratio 0.4–4.0 (descartar SAM detectando el balón o cajas amarillas).

**SAM 3.1 requiere GPU y ~10–15 s de carga.** Para mantener este notebook
ejecutable en CPU, **no** invocamos `load_model()`. En su lugar cargamos las
máscaras y bboxes ya guardadas en `tracks.json` del run precomputado.
"""
        )
    )

    cells.append(
        code(
            """
tracks = json.loads((RUN_DIR / "tracks.json").read_text(encoding="utf-8"))
print(f"Frames procesados en el run: {len(tracks['frames'])}")
print(f"Stride: {tracks['stride']} (1 de cada {tracks['stride']} frames)")
print(f"FPS efectivo de salida: {tracks['fps_out']:.2f}")

# Frame del gol (t=18.01s aprox)
goal_frame = min(tracks["frames"], key=lambda f: abs(f["t_s"] - 18.01))
print(f"\\nFrame más cercano al gol (t={goal_frame['t_s']:.2f}s):")
print(f"  balón: cx={goal_frame['ball']['cx']:.0f} cy={goal_frame['ball']['cy']:.0f} src={goal_frame['ball']['source']}")
print(f"  robots: {len(goal_frame['robots'])} detectados")
for r in goal_frame["robots"][:5]:
    print(f"    id={r['track_id']} team={r['team']} bbox={[round(v) for v in r['bbox_xyxy']]}")
"""
        )
    )

    # ------------ § 6 Tracking ------------
    cells.append(
        md_block(
            """
## § 6 — Tracking: OC-SORT (robots) + Kalman (balón)

SAM 3.1 nos da detecciones **frame-a-frame**, pero necesitamos **identidades
persistentes** para hablar de "robot 5 hizo X" o "el balón viajó N mm".

- **Robots** → `OC-SORT` de [BoxMOT](https://github.com/mikel-brostrom/boxmot).
  Asocia detecciones entre frames usando IoU + velocidad + dirección del
  movimiento, recupera identidades tras oclusiones cortas.
- **Balón** → `BallTracker` propio (Kalman 2D lineal). El balón es pequeño
  (~30 px), salta a velocidades altas y se ocluye con frecuencia. Kalman
  mantiene la última posición conocida + velocidad para predecir el siguiente
  frame si SAM 3.1 lo pierde, y un fallback HSV captura el balón cuando SAM
  falla.

Vemos las trayectorias del run completo en top-down:
"""
        )
    )

    cells.append(
        code(
            """
# Sacar todas las posiciones del balón en mm
ball_xy_mm = []
for f in tracks["frames"]:
    wm = f["ball"].get("world_mm")
    if wm is not None:
        ball_xy_mm.append((f["t_s"], wm[0], wm[1]))

ball_xy_mm = np.array(ball_xy_mm)

# Posiciones de robots por track_id
robots_by_id: dict[int, list[tuple[float, float, float]]] = {}
for f in tracks["frames"]:
    for r in f["robots"]:
        tid = r["track_id"]
        cm = r["centroid_mm"]
        robots_by_id.setdefault(tid, []).append((f["t_s"], cm[0], cm[1]))

fig, ax = plt.subplots(figsize=(14, 7))
# Campo top-down como rectángulo de referencia
ax.add_patch(mpatches.Rectangle((0, 0), 2190, 1580, fill=False, ec="black", lw=2))
ax.set_xlim(-300, 2500); ax.set_ylim(1800, -200)

for tid, traj in robots_by_id.items():
    arr = np.array(traj)
    ax.plot(arr[:, 1], arr[:, 2], "-o", markersize=3, alpha=0.6, label=f"robot {tid}")

if len(ball_xy_mm):
    ax.plot(ball_xy_mm[:, 1], ball_xy_mm[:, 2], "-", color="orange", lw=2, label="balón")

ax.set_aspect("equal")
ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title("Trayectorias top-down (IMG_9852, run v3)")
ax.legend(loc="upper right", fontsize=8, ncols=2)
plt.show()
"""
        )
    )

    # ------------ § 7 Re-ID ------------
    cells.append(
        md_block(
            """
## § 7 — Re-ID adaptativo por color (clasificador en línea)

El reglamento permite que cada equipo elija el color de su bandera, así que
el clasificador de equipo NO puede tener colores hardcodeados. Implementamos
`AdaptiveTeamClassifier` (en `src/tracking/reid.py`):

1. **Warmup** (`warmup_frames=30`): recopila features de matiz/saturación
   (HSV) de la zona superior del bbox (donde está la bandera del robot).
2. **k-means 1D** sobre matiz (con peso de saturación) → 2 clusters: team A y B.
3. **Recompute online** cada 15 frames: refresca centros del cluster para
   resistir cambios de iluminación durante el partido.
4. **Voting window** de 20 frames: la etiqueta final por track_id es la
   mayoritaria reciente, evita flicker.

Visualizamos el histograma de matices observados:
"""
        )
    )

    cells.append(
        code(
            """
# Reunir todos los robots con sus matices vistos en el run
hues = []
teams = []
for f in tracks["frames"]:
    for r in f["robots"]:
        h = r.get("team_hue")
        t = r.get("team")
        if h is not None:
            hues.append(h)
            teams.append(t)

hues = np.array(hues)
teams = np.array(teams)

fig, ax = plt.subplots(figsize=(12, 5))
for label, color in [("A", "#c81e82"), ("B", "#f0f0f0"), (None, "#aaaaaa")]:
    mask = teams == label
    if mask.any():
        ax.hist(hues[mask], bins=36, range=(0, 180), alpha=0.6, label=f"team {label}",
                color=color, edgecolor="black")
ax.set_xlabel("Hue (OpenCV, 0–180)")
ax.set_ylabel("Conteos")
ax.set_title("Distribución de matices observados — clusters del AdaptiveTeamClassifier")
ax.legend()
plt.show()

print(f"\\nObservaciones totales: {len(hues)}")
print(f"Robots clasificados como team A: {(teams == 'A').sum()}")
print(f"Robots clasificados como team B: {(teams == 'B').sum()}")
print(f"Sin equipo (warmup): {(teams == None).sum()}")
"""
        )
    )

    # ------------ § 8 Eventos + gol AND ------------
    cells.append(
        md_block(
            """
## § 8 — Detección de eventos rule-based

Los detectores de evento son **rule-based** (no aprendidos): los umbrales
viven en `src/events/rules.py`, son auditables y reproducibles. Esto cumple
el § 3.5 de la convocatoria sin depender de datos anotados que no tenemos.

Tenemos 8 detectores: `kick`, `goal`, `retention`, `no_progress`, `damaged`,
`pass`, `interception`, `collision`. Cargamos los eventos del run y miramos
el timeline:
"""
        )
    )

    cells.append(
        code(
            """
events = json.loads((RUN_DIR / "events.json").read_text(encoding="utf-8"))
print(f"Eventos totales: {len(events)}")

from collections import Counter
counts = Counter(e["type"] for e in events)
for typ, n in counts.most_common():
    print(f"  {typ:14s} {n}")

# Timeline
fig, ax = plt.subplots(figsize=(14, 4))
type_y = {"kick": 0, "pass": 1, "interception": 2, "retention": 3,
          "no_progress": 4, "goal": 5, "collision": 6}
type_color = {"kick": "#ffd60a", "pass": "#c77dff", "interception": "#ff66a5",
              "retention": "#5050ff", "no_progress": "#b0b000", "goal": "#00ff00",
              "collision": "#ff0000"}
for e in events:
    y = type_y.get(e["type"], 7)
    ax.scatter(e["t"], y, color=type_color.get(e["type"], "gray"), s=60)
ax.set_yticks(list(type_y.values()))
ax.set_yticklabels(list(type_y.keys()))
ax.set_xlabel("t (s)")
ax.set_title("Timeline de eventos detectados")
ax.grid(True, alpha=0.3)
plt.show()
"""
        )
    )

    cells.append(
        md_block(
            """
### § 8.1 — Detector de gol con AND (commit 5ca5c60)

El detector más delicado es el de **gol**. Nuestra v3 combina con AND dos
verificaciones independientes:

1. **`ball_inside_goal_field`**: el balón cruzó la **arista del cuadrilátero
   del campo** más cercana a la portería. Es geométricamente correcto y NO
   driftea con la recalibración HSV cada 60 frames. Además incluye un
   **filtro lateral** (proyección del bbox de portería sobre la línea) que
   evita FP cuando el balón está detrás de la línea pero arriba/abajo del arco.
2. **`ball_inside_goal`**: el balón cruzó la **arista interior del bbox HSV**
   con el mismo filtro lateral. Verifica que el balón esté lateralmente cerca
   del arco real, no solo de la línea geométrica.

Solo dispara "gol" cuando AMBOS son `True` en la transición afuera→adentro,
con histeresis (deadband 25 px), cooldown 5 s y guard `t > 1 s`.

Resultados de B6 (3 commits):

| Video | Original | Commit 93547d0 (Fix2) | Commit 5ca5c60 (AND) | Total |
|-------|---------:|----------------------:|---------------------:|-----:|
| IMG_9821 (0 reales) | 11 FP | 6 FP | **4 FP** | **-64 %** |
| IMG_9852 (1 real)   | 1 ✓   | 1 ✓   | **1 ✓**   | recall 100 % |
| video-1144 (1 real) | 8 (1+7) | 1 ✓ | **1 ✓** | -87.5 % FP |

Vemos el frame del gol con sus capas superpuestas:
"""
        )
    )

    cells.append(
        code(
            """
from src.events.rules import goal_line_from_field_edge, ball_inside_goal_field, ball_inside_goal

# Buscar el evento de gol
goal_events = [e for e in events if e["type"] == "goal"]
print(f"Goles detectados: {len(goal_events)}")
g = goal_events[0]
print(f"  t={g['t']:.2f}s color={g['meta']['goal_color']}")
print(f"  método: {g['meta']['method']}")
print(f"  ball_px: ({g['meta']['ball_px'][0]:.0f}, {g['meta']['ball_px'][1]:.0f})")
print(f"  ball_mm: ({g['meta']['ball_mm'][0]:.0f}, {g['meta']['ball_mm'][1]:.0f})")

# Sacar el frame del gol del video
goal_t = g["t"]
goal_frame_img = None
for idx, frame in read_frames(VIDEO_PATH, stride=1):
    t_now = idx / meta.fps
    if t_now >= goal_t:
        goal_frame_img = frame
        break

# Overlay: campo + porterías + línea de gol del CAMPO + balón
viz = cv2.cvtColor(goal_frame_img.copy(), cv2.COLOR_BGR2RGB)
corners = np.array(tracks["corners_img"])
pts = corners.astype(int)
for i in range(4):
    cv2.line(viz, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 200, 0), 3)

bbox = g["meta"]["goal_bbox_xyxy"]
x1, y1, x2, y2 = [int(v) for v in bbox]
cv2.rectangle(viz, (x1, y1), (x2, y2), (255, 200, 0), 3)

# Arista del campo elegida como línea de gol
gl = g["meta"]["goal_line_px"]
cv2.line(viz, (int(gl[0][0]), int(gl[0][1])), (int(gl[1][0]), int(gl[1][1])),
         (255, 0, 0), 5)

# Posición del balón en el momento del gol
bx, by = int(g["meta"]["ball_px"][0]), int(g["meta"]["ball_px"][1])
cv2.circle(viz, (bx, by), 15, (0, 255, 255), 3)

plt.imshow(viz)
plt.title(f"Frame del gol (t={goal_t:.2f}s) — verde:campo, amarillo:bbox, rojo:línea de gol, cian:balón")
plt.axis("off")
plt.show()
"""
        )
    )

    # ------------ § 9 Stats + viz ------------
    cells.append(
        md_block(
            """
## § 9 — Estadísticas y visualizaciones

`MatchStats` (`src/metrics/stats.py`) acumula posesión, distancias y
velocidades. Después de B4 (commit `a597f5e`) filtra saltos de tracking
mayores a 1500 mm (balón) y 500 mm (robots) para evitar artefactos.

Las visualizaciones (`src/viz/`) generan:
- Heatmaps por equipo (densidad de actividad)
- Trails (trayectorias completas con fade)
- Voronoi (control de espacio en el último frame)
- Minimap top-down animado (sincronizado al video)

Mostramos el summary del run y los heatmaps:
"""
        )
    )

    cells.append(
        code(
            """
summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))

print(f"Video: {summary['video']}")
print(f"Duración: {summary['duration_s']:.2f} s")
print(f"Frames procesados: {summary['frames_processed']} ({summary['effective_fps']:.2f} fps efectivos)")
print(f"\\nMARCADOR: A {summary['score']['A']} - {summary['score']['B']} B")
print(f"Posesión: A {summary['possession_pct']['A']:.1f}% / B {summary['possession_pct']['B']:.1f}%")
print(f"\\nBalón:")
print(f"  distancia total: {summary['ball']['distance_mm']:.0f} mm")
print(f"  velocidad máx:   {summary['ball']['max_speed_mm_s']:.0f} mm/s")
print(f"  velocidad prom:  {summary['ball']['avg_speed_mm_s']:.0f} mm/s")
print(f"\\nTracking artifacts (filtrados):")
print(f"  saltos del balón descartados: {summary['tracking_artifacts']['n_jumps_discarded_ball']}")
print(f"  saltos de robots descartados: {summary['tracking_artifacts']['n_jumps_discarded_robots']}")
"""
        )
    )

    cells.append(
        code(
            """
# Mostrar los 4 heatmaps que ya están en disco
heatmap_files = [
    ("heatmap_ball.png", "Densidad del balón"),
    ("heatmap_robots.png", "Densidad de robots (todos)"),
    ("heatmap_team_A.png", "Densidad team A"),
    ("trails.png", "Trayectorias (trails)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (fname, title) in zip(axes.flat, heatmap_files):
    path = RUN_DIR / fname
    if path.exists():
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, f"falta {fname}", ha="center", va="center")
    ax.axis("off")
plt.tight_layout()
plt.show()
"""
        )
    )

    # ------------ § 10 Conclusión ------------
    cells.append(
        md_block(
            """
## § 10 — Conclusión y próximos pasos

### Lo que hemos cubierto

1. Inspección del dataset (cámara, fps, resolución).
2. Cascade de detección del campo (3 métodos con fallback).
3. Detección HSV de porterías con ROI restringido al fieltro.
4. Homografía 4-puntos y proyección al top-down mundo.
5. SAM 3.1 conceptual + uso de máscaras precomputadas.
6. Tracking OC-SORT (robots) + Kalman (balón) con fallback HSV.
7. Re-ID adaptativo por matiz HSV (k-means online).
8. Detección de eventos rule-based + AND-detector de gol (B6).
9. Estadísticas filtradas (B4) y visualizaciones.

### Cómo correr el pipeline completo

```bash
source .venv/Scripts/activate
python scripts/run_pipeline.py \\
    --video data/raw/drive_oficial/17Abril/Cámaras/IMG_9852.MOV \\
    --out data/processed/runs/IMG_9852_run \\
    --stride 5
```

Tiempo estimado en RTX 5080: ~90 s para 22 s de video con stride 5.
Genera `annotated.mp4`, `dashboard.html`, `summary.json`, `events.json`,
`tracks.json`, heatmaps, voronoi, minimap y commentary.md/.txt.

### Limitaciones honestas

- **Falsos positivos en perspectivas oblicuas extremas** (IMG_9821): 4 FP
  residuales son artefactos del fallback HSV del balón. Solución pendiente:
  LoRA dedicado al balón (Opción E del TODO) o filtro de consistencia
  temporal multi-frame.
- **Sin ground truth de tracking**: no podemos calcular HOTA/IDF1/MOTA;
  reportamos mIoU del LoRA (0.912) + validación manual de goles por
  inspección humana.
- **Cámara de espectador (iPhone)** en parte del dataset, no Meta Ray-Ban
  uniforme. Pipeline robusto a perspectiva oblicua pero algunos casos
  límite ven afectados.

### Recursos del repo

- Convocatoria: `../futbol-cv/convocatoria/Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf`
- README orientado a jurado: [`README.md`](../README.md)
- Documentación técnica: [`docs/documentacion-completa.md`](../docs/documentacion-completa.md)
- Bitácora de sesiones: `docs/futbotmx-session-summary.org`
- TODO priorizado: `TODO.org`
- Tests: `pytest tests/` (119/119 verdes al cierre de sesión 2026-05-26)

---

*AJOLOTES FC · Copa FutBotMX 2026 · Profesional · Repo: [github.com/JAPerezC/futbotmx-ajolotesfc](https://github.com/JAPerezC/futbotmx-ajolotesfc)*
"""
        )
    )

    return cells


def main() -> int:
    cells = build_cells()
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (futbotmx .venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"OK escrito: {OUT}")
    print(f"  celdas totales: {len(cells)}")
    print(f"  markdown:       {sum(1 for c in cells if c['cell_type'] == 'markdown')}")
    print(f"  code:           {sum(1 for c in cells if c['cell_type'] == 'code')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
