# Arquitectura propuesta — futbotmx

> Fecha: 2026-05-16
> Estado: propuesta inicial, sujeta a ajuste tras validar SAM 3 localmente
> Fuentes: convocatoria CV (16 pp), reglas del torneo (18 pp), inspección de
> `IMG_9915.MOV`

## 1. Resumen ejecutivo

Pipeline de visión por computadora que:

1. **Segmenta** con **SAM 3.1** (Meta, mar 2026, drop-in con object
   multiplexing 2-7× más rápido) cuatro clases en cada frame: campo,
   balón, robots del equipo A, robots del equipo B.
2. **Trackea en dos capas**: **OC-SORT** (vía BoxMOT) para robots con
   re-ID HSV por bandera; **TOTNet** o TrackNetV3 para balón con
   ventana de 5 frames y robustez a oclusiones largas.
3. **Rectifica** las coordenadas de imagen a coordenadas top-down del campo
   (homografía de 4 puntos sobre las líneas blancas).
4. **Detecta eventos**: gol, kick, pase, intercepción, retención del balón,
   robot dañado, falta de progreso.
5. **Visualiza** sobre la vista cenital reconstruida: heatmap de actividad,
   trails, posesión por equipo y Voronoi.
6. **Innovación SAM 3** (requisito Profesional): fine-tuning LoRA del
   encoder de prompts sobre ~200-500 frames anotados del dataset, con
   prompts especializados al dominio robótico.

Salidas: video segmentado lado-a-lado (≤ 2 min), JSON de eventos con
timestamps, dashboard HTML con visualizaciones, reel Instagram ≥ 30 s.

## 2. Requisitos firmes (convocatoria § 3)

| § | Requisito                                                       | Estado actual |
|---|------------------------------------------------------------------|---------------|
| 3.1.1 | Usar SAM 3 (Meta) como modelo base                          | pendiente     |
| 3.2.2 | Categoría Profesional → innovación SAM 3 obligatoria        | diseño        |
| 3.2.2 | Resultados completamente reproducibles                      | pendiente     |
| 3.5.1 | Segmentar campo, robots aliados, rivales, balón             | pendiente     |
| 3.5.1 | Trackear trayectorias de robots y balón                     | pendiente     |
| 3.5.1 | Detectar eventos (pases, tiros, intercepciones, colisiones) | pendiente     |
| 3.5.2 | ≥ 1 visualización (heatmap/posesión/trails/Voronoi/dash)    | pendiente     |
| 3.5.3 | Video demo ≤ 2 min (original + segmentado lado a lado)      | pendiente     |
| 3.5.3 | **Reel Instagram ≥ 30 s** con link en README (DESCALIFICA)  | pendiente     |
| 3.5.4 | README completo: arquitectura, instalación, hardware…       | parcial       |
| 3.6   | Atribuir todas las dependencias en README y código          | parcial       |
| 3.11  | Licencia abierta (MIT/Apache)                               | hecho (MIT)   |
| 3.7.4 | Repo público al **19 jun 23:59** (DESCALIFICA si privado)   | privado aún   |
| 3.7.4 | Entregable completo (pipeline + video + reel)               | pendiente     |

## 3. Hallazgos del dataset (resumen)

Ver `docs/dataset-inspection.md` para análisis completo. Puntos clave:

- Resolución 1920×1080 @ 29.97 fps, H.264 High, container MOV.
- **Cámara NO es Ray-Ban**: iPhone 16 Pro Max desde la orilla del campo
  (vista oblicua de espectador, paneos suaves).
- Formato del partido en la muestra: **1v1** (reglamento permite hasta 2v2).
- **Identidad de equipo por bandera vertical** (morada vs blanca/verde),
  no por color del chasis.
- Balón naranja brillante (alto contraste) — Categoría Abierta del torneo.

## 4. Reglas del torneo (lo que importa para CV)

### 4.1 Geometría del campo (mm exactos)

| Elemento                  | Medida                                   |
|---------------------------|------------------------------------------|
| Largo total (con paredes) | 243 cm                                   |
| Ancho total               | 182 cm                                   |
| Zona de juego (interior)  | 219 × 158 cm                             |
| Altura de paredes         | ≥ 22 cm (negras mate)                    |
| Líneas blancas            | 2 cm de grosor (±0.5)                    |
| Círculo central           | Ø 60 cm                                  |
| Línea central             | a 121.5 cm de cada línea de gol          |
| Área de penalti           | 25 cm profundidad × 80 cm ancho          |
| Línea de gol              | 60 cm de ancho                           |
| Portería                  | 60 cm ancho × 10 cm alto × 10 cm fondo   |
| Color porterías           | **una amarilla, una azul**               |
| Puntos neutrales          | 4 puntos a 45 cm de cada esquina         |
| Tolerancia dimensional    | ±5 %                                     |

Estas medidas son **constantes en todo torneo oficial** y nos dan un
modelo 3D del campo perfectamente conocido → la homografía es resoluble
con 4 puntos correspondientes (las esquinas interiores del campo).

### 4.2 Robots (sec. 3 del reglamento)

- **Máximo 2 robots por equipo** (formato hasta 2v2).
- Diámetro y altura ≤ 18 cm (Categoría Abierta) o ≤ 22 cm (Ágil).
- **Colores prohibidos**: naranja, amarillo, azul. SAM 3 no debería
  confundir robots con balón o porterías por color.
- Marcador superior blanco ≥ 4 cm para numeración (las banderas
  verticales que se ven en el dataset son personalizaciones).
- Autonomía total, sin manipulación remota durante el juego.

### 4.3 Balón

- Categoría Abierta: **pelota de golf naranja brillante, Ø 42 mm**.
- Categoría Ágil: balón con emisor IR (mismo diámetro).
- El dataset inspeccionado es Categoría Abierta.

### 4.4 Eventos derivables

| Evento                  | Definición reglamentaria             | Cómo detectarlo                 |
|-------------------------|--------------------------------------|----------------------------------|
| Gol                     | balón toca pared trasera de portería | Intersección máscara-bola con ROI de portería rectificado |
| Tiro/kick               | cambio brusco de velocidad del balón | Derivada de posición sobre umbral |
| Pase                    | balón pasa entre 2 robots del mismo equipo sin oponente intermedio | Geometría + asignación temporal |
| Intercepción            | balón cambia de equipo en posesión   | Cambio de robot más cercano de equipo distinto |
| Retención (falta)       | balón pegado a robot > N segundos    | Distancia balón-robot bajo umbral durante ventana |
| Robot dañado            | robot sin movimiento > 60 s          | Velocidad media de máscara ≈ 0 sostenida |
| Falta de progreso       | balón sin movimiento neto + robots cerca | Posición balón σ < umbral en ventana |
| Saque inicial           | balón en círculo central, robots quietos | Heurística pre-juego |
| Empujón en área penalti | contacto robots opuestos en zona     | Solape de máscaras + ROI |

## 5. Arquitectura

### 5.1 Flujo de datos

```
[video.MOV]
   │
   ▼
[ingest]  ── frame_id, timestamp, ndarray RGB
   │
   ▼
[calibrate]  ── homografía H (imagen → top-down 219×158 cm)
   │            usa líneas blancas del primer frame nítido
   ▼
[segment]  ── máscaras por clase (campo, balón, robot_A, robot_B)
   │           SAM 3 con prompts text + box, stride configurable
   ▼
[track]  ── trayectorias con IDs persistentes (ByteTrack sobre máscaras)
   │         re-id auxiliar por color de bandera (HSV clasificador)
   ▼
[rectify]  ── posiciones top-down (cm) por frame, por objeto
   │
   ▼
[events]  ── lista de eventos con timestamp y metadatos
   │
   ├─► [viz/heatmap]     → PNG por equipo
   ├─► [viz/voronoi]     → MP4 por frame
   ├─► [viz/possession]  → series temporales + barras
   ├─► [viz/trails]      → MP4 anotado
   └─► [viz/dashboard]   → HTML estático

[exporters]
   ├─► video_demo.mp4 (≤ 120 s, original + segmentado lado a lado)
   ├─► events.json     (estructura: [{t, type, actors, position_cm, conf}])
   └─► report.html     (índice navegable de outputs)
```

### 5.2 Módulos (mapeo a `src/`)

| Módulo                 | Responsabilidad                              | Tecnología                     |
|------------------------|----------------------------------------------|--------------------------------|
| `src/utils/io.py`      | Lectura de video, escritura de outputs       | OpenCV, ffmpeg                 |
| `src/utils/calib.py`   | Homografía 4-puntos + KLT drift              | OpenCV (findHomography, KLT)   |
| `src/segmentation/sam3.py` | Wrapper SAM 3.1 con prompts text/box     | transformers, torch            |
| `src/segmentation/prompts.py` | Catálogo de prompts y heurísticas     | —                              |
| `src/tracking/robots.py` | OC-SORT sobre robots                       | boxmot (mikel-brostrom)        |
| `src/tracking/ball.py` | TOTNet/TrackNetV3 + Kalman fallback         | torch, scipy                   |
| `src/tracking/reid.py` | Re-ID por color de bandera (HSV)            | OpenCV                         |
| `src/events/rules.py`  | Detectores rule-based (AutoRefs SSL style)  | numpy, scipy                   |
| `src/events/possession.py` | Asignación de posesión + PathCRF fallback | numpy, networkx                |
| `src/viz/heatmap.py`   | Mapas de calor por equipo                   | matplotlib                     |
| `src/viz/voronoi.py`   | Diagramas de Voronoi por frame              | scipy.spatial                  |
| `src/viz/possession.py`| Métricas temporales de posesión             | pandas, plotly                 |
| `src/viz/trails.py`    | Trayectorias anotadas sobre video           | OpenCV                         |
| `src/viz/dashboard.py` | Compositor HTML final                       | Jinja2 + plotly                |
| `scripts/run_pipeline.py` | CLI end-to-end                           | argparse, loguru               |

### 5.3 Stack

- **Python 3.12** (CLAUDE.md, compatibilidad SAM 3.1 / torch 2.10+cu128).
- **torch 2.10+cu128, transformers ≥ 4.45** para SAM 3.1 desde HF.
- **boxmot** (mikel-brostrom) para OC-SORT/ByteTrack/StrongSORT con API
  unificada y soporte Python 3.12.
- **supervision** (Roboflow) para parsear outputs SAM 3.1
  (`sv.Detections.from_inference`) y anotar máscaras/polígonos.
- **opencv-python** (homografía, KLT, drawing).
- **scipy, numpy, pandas, plotly, matplotlib**.
- **networkx** (PathCRF posesión, opcional).
- **ffmpeg** (sistema) para encode del demo.
- **loguru** para logging estructurado.
- **pytest** para tests unitarios mínimos.

## 6. Innovación sobre SAM 3 (requisito § 3.7.3)

Elegimos **al menos dos** de las líneas oficiales para sumar puntos:

### 6.1 Prompt engineering avanzado
- Prompts compuestos: texto descriptivo + caja inicial + ejemplo visual
  desde el primer frame anotado a mano.
- **Cascada de prompts**: el primer prompt rompe el frame en regiones
  (campo / no-campo), el segundo segmenta dentro de la región robot.
- **Re-prompting adaptativo**: cuando confianza < umbral, re-prompt con
  caja sugerida por el tracker (ByteTrack alimenta a SAM 3).

### 6.2 Fine-tuning LoRA (innovación principal)
- Anotar 200-500 frames del dataset (VIA, CVAT o Roboflow free tier).
- LoRA **rank 8** sobre image encoder + mask decoder
  (referencia: `Sompote/SAM3_LoRA`). VRAM esperada: 12 GB.
- ~0.5 % de parámetros entrenables → bajo riesgo de overfitting con
  dataset pequeño.
- Métrica nativa SAM 3: **cgF1@50**. Meta: ≥ 0.65 sobre hold-out, dado
  que las 4 clases son visualmente distinguibles (campo verde uniforme,
  balón naranja brillante, banderas morada vs blanca).
- Pitfall principal: **prompt drift**. Fijar el texto de los prompts en
  una constante compartida entre train y eval.
- Reproducibilidad: notebook + script + seed fijo + checkpoint en
  `data/models/` (gitignored, descargable por separado).

### 6.3 Post-procesamiento geométrico
- **Estimación de velocidad**: derivada de posición top-down. Útil para
  detectar tiros vs. movimientos normales.
- **Predicción de movimiento**: filtro de Kalman 2D para suplir frames
  con balón ocluido.
- **Asignación de posesión**: robot del equipo X más cercano al balón
  durante una ventana temporal.

## 7. Métricas de evaluación

| Métrica                       | Objetivo                                     |
|-------------------------------|----------------------------------------------|
| mIoU por clase                | ≥ 0.75 campo, ≥ 0.65 robots, ≥ 0.50 balón    |
| ID switches por minuto        | ≤ 2 por equipo                                |
| Frames con balón perdido      | < 10 % de la duración                        |
| Latencia inferencia (1080p)   | < 200 ms por frame en RTX 4090               |
| Precisión gol (TP/(TP+FP))    | ≥ 0.85                                       |
| Falsos positivos gol/minuto   | < 1                                          |

## 8. Riesgos críticos

| Riesgo                                       | Probabilidad | Mitigación                                                                |
|----------------------------------------------|--------------|---------------------------------------------------------------------------|
| SAM 3 no corre en hardware local            | media        | Validar día 1; fallback HF Inference / Colab Pro                          |
| Dataset oficial muy distinto al sample       | media        | Bajar 5-10 videos más antes del 25 may                                    |
| Anotación para fine-tuning toma > 5 días     | alta         | Anotar 200 frames mínimos, no perfeccionismo                              |
| ID switches frecuentes por banderas pequeñas | alta         | Re-ID HSV + Kalman + tracking nativo SAM 3.1 si disponible                |
| Olvidar reel IG antes del 19 jun             | media        | Hito explícito en plan-sprint, recordatorio en CLAUDE.md                  |
| Repo público olvidado al 19 jun (DESCALIFICA) | media       | Cron mental para 18 jun: cambiar visibilidad                              |
| Equipo incompleto para registro 22 may      | alta         | Decidir esta semana; plan B individual                                    |
| Métricas reproducibles fallan (§ 3.2.2)     | media        | Seed fijo, lockfile, requirements pinned, instrucciones de hardware       |
| Licencia SAM 3 mal atribuida                | baja         | `THIRD_PARTY_LICENSES.md` + README sección créditos                       |

## 9. Cronograma alineado al calendario oficial

| Fecha             | Hito                                                            |
|-------------------|-----------------------------------------------------------------|
| **2026-05-22**    | Cierre de registro (formulario + URL del repo público)          |
| 2026-05-23 a 06-05 | Fase 1: pipeline mínimo viable end-to-end                     |
| 2026-06-06 a 06-14 | Fase 2: innovación SAM 3 (LoRA + prompts) + visualizaciones   |
| 2026-06-15 a 06-18 | Fase 3: video demo, reel IG, README final, verificación       |
| **2026-06-19 23:59** | Fecha límite de entregable. Repo público bloqueado tras esta hora |
| 2026-06-20 a 06-24 | Periodo de evaluación del Comité                              |
| 2026-06-25        | Publicación de resultados                                       |
| 2026-06-24 a 06-26 | Copa FutBotMX presencial en CDMX                              |
| 2026-06-26        | Premiación                                                      |

**Buffer**: terminar Fase 3 el **2026-06-17** para tener 48 h de holgura
antes del 19 jun.

## 10. Estructura de carpetas final

```
futbotmx/
├── src/
│   ├── segmentation/   ya creada
│   ├── tracking/       ya creada
│   ├── events/         ya creada
│   ├── viz/            ya creada
│   └── utils/          ya creada
├── data/               gitignored
│   ├── raw/            videos originales
│   ├── processed/      frames, máscaras, outputs intermedios
│   ├── annotations/    ← NUEVO: labels para fine-tuning
│   ├── models/         pesos SAM 3 + LoRA (descargables)
│   └── runs/           ← NUEVO: salidas por corrida fechada
├── notebooks/          ← exploración, fine-tuning, evaluación
├── scripts/            ← CLI: run_pipeline.py, annotate.py, finetune.py
├── tests/              pytest
├── docs/               este documento + dataset-inspection + sprint + checklist
├── reports/            ← NUEVO: video demo, GIFs, dashboard estático
└── reel/               ← NUEVO: assets del reel IG (placeholder)
```

## 11. Próximas acciones (orden estricto)

1. Crear `.venv` con Python 3.12 (no 3.9 que es el del sistema).
2. Instalar `torch + transformers` y validar inferencia SAM 3 sobre 1 frame.
3. Bajar 5-10 videos adicionales del Drive, repetir inspección para
   confirmar heterogeneidad de la cámara.
4. Decidir composición del equipo, recolectar documentación 3.3.1.
5. Llenar formulario de registro antes del **22 may 23:59**.
6. Hacer repo público el día del registro.
7. Empezar Fase 1 (pipeline MVP) el 25 may.
