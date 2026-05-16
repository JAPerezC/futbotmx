# Revisión de literatura — decisiones técnicas para futbotmx

> Fecha: 2026-05-16
> Alcance: justificar cada elección del pipeline con evidencia 2024-2026.
> Citas en formato `[@clave]` resueltas contra `refs.bib`.

## Resumen ejecutivo

Tras revisar la literatura más reciente en cuatro frentes (fine-tuning de
SAM 3, tracking de objetos pequeños, homografía de campo deportivo,
detección de eventos en fútbol robótico), las decisiones técnicas son:

1. **SAM 3.1** [@sam31_meta_2026] sobre SAM 3 base — drop-in replacement con
   2-7× más velocidad gracias a object multiplexing.
2. **LoRA rank 8** [@sam3_lora_sompote_2025] para fine-tuning sobre
   ~300 frames anotados — 12 GB VRAM, rendimiento ~90 % del full fine-tune.
3. **Pipeline de tracking en dos capas**: BoxMOT/OC-SORT [@boxmot_2024]
   para robots + TOTNet [@totnet_2025] o TrackNetV3 para balón.
4. **Homografía clásica** OpenCV con 4 puntos manuales al inicio +
   Lucas-Kanade para drift, recalibrando cada 30 frames. PnLCalib
   [@pnlcalib_2024] descartado por estar orientado a campo FIFA.
5. **Eventos rule-based** siguiendo el modelo de los AutoRefs de SSL
   [@ssl_autoref][@erforce_autoref], con PathCRF [@pathcrf_2025] como
   fallback de posesión cuando el balón se oclude.

## 1. Por qué SAM 3.1 y no SAM 3 base

SAM 3 fue lanzado en noviembre 2025 [@sam3_paper_2025]. SAM 3.1 salió el
27 de marzo de 2026 [@sam31_meta_2026] con **object multiplexing**:
detecta y segmenta hasta 16 objetos en un solo forward pass (antes 1 por
paso). Para nuestro caso de 4-5 objetos por frame (campo, 2 robots,
balón, opcionalmente porterías), la ganancia es directa:

- Velocidad 2× en escenas de densidad media (4-10 objetos).
- Velocidad 7× en escenas densas (128+ objetos).
- Mismos checkpoints en HuggingFace, **drop-in replacement** sin
  cambios de código.

El stack del proyecto (`torch 2.10+cu128`, Python 3.12) cumple los
requisitos mínimos de SAM 3.1 (PyTorch 2.7+, CUDA 12.6+).

## 2. Fine-tuning como innovación Profesional

La convocatoria § 3.7.3 exige innovación sobre SAM 3 para Profesional.
Fine-tuning es la línea de mayor peso evaluativo, pero con riesgo de
overfitting si se mal-ejecuta sobre dataset pequeño.

### Estrategia recomendada: LoRA rank 8

| Estrategia                | VRAM   | Riesgo overfit | Recomendado para            |
|---------------------------|--------|----------------|------------------------------|
| Full fine-tune            | 40+ GB | alto           | >10 000 frames               |
| LoRA rank 32              | 20 GB  | medio          | 1 000-5 000 frames           |
| **LoRA rank 8**           | **12 GB** | **bajo**    | **200-500 frames (nuestro caso)** |
| LoRA rank 4               | 8 GB   | muy bajo       | <200 frames                  |
| Adapter prompt encoder    | <8 GB  | mínimo         | dominio ya cubierto          |

El repositorio [Sompote/SAM3_LoRA](https://github.com/Sompote/SAM3_LoRA)
[@sam3_lora_sompote_2025] entrenó SAM 3 con 778 imágenes usando LoRA
rank 8 sobre image encoder + mask decoder con ~0.5 % de parámetros
entrenables. Métrica reportada: mAP@50 entre 0.40 y 0.65; cgF1@50
(métrica nativa SAM 3) entre 0.50 y 0.70.

Para nuestras 4 clases **visualmente distinguibles** (campo verde
uniforme, balón naranja brillante, banderas morada vs blanca) el techo
superior del rango es alcanzable. Meta proyectada: cgF1@50 ≥ 0.65 sobre
hold-out.

### Pitfalls a evitar

- **Overfitting**: con <500 imágenes, rank alto (32) y >100 épocas es
  la combinación más peligrosa. Usar rank 8, dropout 0.1, early stopping
  sobre validation loss.
- **Prompt drift**: SAM 3 usa text prompts. Si el texto en train
  ("robot") difiere del texto en inferencia ("robot con bandera morada"),
  la segmentación falla silenciosamente. **Fijar prompts en una constante
  compartida entre train y eval.**
- **Formato dataset**: el repo oficial espera COCO con máscaras, no
  bounding boxes. Convertir con `pycocotools`.
- **Gated model**: aceptar términos en HuggingFace antes de descargar
  pesos. Ya hecho [✓].

### Alternativa de menor riesgo

Prompt engineering avanzado [@roboflow_sam3_finetune_2025] solo (sin
fine-tuning) puede ser suficiente para cumplir el requisito de
innovación si el MVP toma más tiempo del esperado. No es mutuamente
excluyente con LoRA.

## 3. Tracking: dos capas distintas

### 3.1 Robots (objetos grandes, identidad por color)

SportsMOT [@sportsmot_2023] benchmark (ICCV 2023) reporta que en
deportes con ≤10 jugadores los trackers clásicos rinden:

| Tracker        | HOTA | IDF1 | IDs |
|----------------|------|------|-----|
| ByteTrack      | 63.1 | 77.3 | 2196 |
| OC-SORT        | 63.2 | 77.5 | 1950 |
| DiffMOT        | 76.2 | 76.1 | —    |
| Deep HM-SORT   | 80.1 | —    | —    |

Para nuestros 2-4 robots con identidad estable (bandera de color),
**OC-SORT** dentro de BoxMOT [@boxmot_2024] es suficiente y se integra
en una sola línea. Re-ID auxiliar por **clasificador HSV** sobre el área
de la bandera resuelve los ID switches cuando OC-SORT pierde por
oclusión. No se necesita embedding profundo tipo OSNet.

### 3.2 Balón (objeto de 10-20 px, oclusiones largas)

Los trackers Kalman-IoU **fallan** con balones de <20 px porque
acumulan error en oclusiones largas. La literatura más reciente apunta a
trackers especializados:

- **TrackNetV3**: background subtraction + interpolación de trayectoria.
  Reduce MDE 54-61 % en deportes de raqueta.
- **TOTNet** [@totnet_2025]: 3D convolutions con ventana de 5 frames,
  visibility-weighted loss, occlusion augmentation. Reduce RMSE de
  37.30 → **7.19** en tenis/ping-pong. Diseñado explícitamente para
  oclusiones prolongadas detrás de jugadores.

**Decisión**: usar TOTNet para el balón en frames donde la oclusión
exceda 5 frames; OC-SORT clásico para el resto. SAM 3.1 mask tracking
queda como respaldo si TOTNet no se puede integrar a tiempo (el
inference fallback es solo robot, no balón).

## 4. Homografía: clásica antes que deep learning

[@pnlcalib_2024] es el sucesor de TVCalib [@tvcalib_2023] y la
referencia 2024 para campos de fútbol estándar (FIFA). Sin embargo:

- Asume modelo de campo predefinido (cancha humana, 105×68 m).
- Adaptarlo a nuestra cancha de 219×158 cm requiere redefinir el
  template world y reentrenar la cabeza de keypoints.
- El esfuerzo no se justifica cuando tenemos **dimensiones exactas
  conocidas** y **buen contraste** de las líneas blancas.

### Estrategia recomendada (3 pasos)

1. **Calibración inicial** sobre el primer frame nítido:
   - Detectar las 4 esquinas interiores del campo (intersección
     línea-pared) con Harris/Shi-Tomasi.
   - Mapear a coordenadas mundo: (0,0), (2190,0), (2190,1580), (0,1580) mm.
   - `cv2.findHomography(..., cv2.RANSAC, 3.0)` → matriz `H` base.

2. **Drift inter-frame** con Lucas-Kanade
   (`cv2.calcOpticalFlowPyrLK`) sobre puntos del campo. Cada frame
   actualiza `H` componiendo con la transformación relativa.

3. **Re-anclaje cada 30 frames**: reproyectar el template del campo
   sobre el frame actual y medir error. Si > 5 px,
   recalcular `H` con `findHomography` sobre los keypoints encontrados.

### Bibliotecas

- `cv2.findHomography` + `cv2.warpPerspective` (núcleo).
- `cv2.calcOpticalFlowPyrLK` (KLT para drift).
- `kornia.geometry.homography.find_homography_dlt` (diferenciable si se
  quisiera integrar al pipeline torch en el futuro).

## 5. Detección de eventos: rule-based sobre los AutoRefs SSL

Los **AutoRefs de RoboCup SSL** [@ssl_autoref][@erforce_autoref] resuelven
exactamente nuestro problema: detectar eventos de fútbol robótico a partir
de trayectorias (x, y, t) producidas por un sistema de visión externo.
Son rule-based, código abierto, mantenidos al 2026.

### Umbrales adaptados a nuestro caso

| Evento           | Definición                                | Parámetro inicial          |
|------------------|-------------------------------------------|-----------------------------|
| Gol              | Posición balón en ROI portería            | N=3 frames consecutivos     |
| Kick/tiro        | Δv del balón en 1 frame                   | > 0.5 m/s (golf ball)       |
| Pase             | Cambio posesión mismo equipo + traslado   | distancia > 30 cm           |
| Intercepción     | Cambio posesión equipos opuestos          | —                           |
| Retención (falta)| Distancia balón-robot < R durante T       | R = 90 mm, T = 1.5 s        |
| Falta de progreso| σ posición balón en ventana               | σ < 5 cm en 5 s             |
| Robot dañado     | Velocidad media ≈ 0 sostenida             | < 2 cm/s durante 60 s       |
| Saque inicial    | Balón en círculo central + robots quietos | distancia centro < 5 cm     |

Los valores R=90 mm y la lógica de cambio de posesión vienen
directamente de los AutoRefs SSL.

### Posesión sin posición del balón

Cuando el balón se pierde (oclusión > 5 frames y TOTNet falla),
**PathCRF** [@pathcrf_2025] (arXiv 2602.12080, 2025) infiere posesión
desde grafos de jugadores + CRF. Con 1v1 o 2v2 el grafo es trivial.
Funciona sin etiquetas de evento — útil dado que no tenemos ground truth.

### Por qué rule-based y no ML

[@football_events_springer_2022] reporta que un árbol de decisión
determinista con umbrales empíricos logra >90 % de precisión en
detección de eventos en fútbol humano. Para dataset pequeño (sin
benchmark público de fútbol robótico junior), ML añade riesgo de
overfitting sin ganancia clara.

### Ground truth: anotación manual mínima

No existe benchmark público de eventos en fútbol robótico junior. Se
anotará a mano:

- 2-3 partidos completos (~5-10 minutos de video) con timestamps
  precisos de cada evento.
- Formato JSON: `[{"t": float, "type": str, "actors": [robot_ids],
  "position_cm": [x, y]}, ...]`
- Tool sugerido: VIA (VGG Image Annotator) o CVAT para video.

## 6. Brecha entre lo deseable y lo entregable

Lo deseable (publicación científica) y lo entregable (19 de junio)
divergen en estos puntos:

| Deseable                                            | Entregable a 19 jun                           |
|-----------------------------------------------------|-----------------------------------------------|
| Fine-tuning con 1000+ frames anotados               | LoRA con 200-300 frames                       |
| TrackNetV3 reentrenado sobre balón de golf naranja  | TOTNet preentrenado + Kalman fallback         |
| PnLCalib adaptado a cancha robótica                 | OpenCV findHomography clásica                 |
| Validación cuantitativa contra GT del torneo        | Validación cualitativa + métricas internas    |
| Reel IG con voz, branding, transiciones             | Reel ≥ 30 s con texto + corte simple          |
| Dashboard interactivo Plotly + Streamlit            | Dashboard estático HTML con plotly export     |

Priorizar lo entregable. Lo deseable queda como roadmap post-entrega
documentado en README sección "Trabajo futuro".

## 7. Decisiones revertibles y puntos de decisión

| Decisión              | Cuándo evaluar             | Si falla, fallback          |
|-----------------------|----------------------------|------------------------------|
| LoRA rank 8           | Tras 30 min de entrenamiento; loss curve | rank 4 o solo prompt eng. |
| TOTNet para balón     | Tras integrar en pipeline; FPS | OC-SORT puro + Kalman    |
| Calibración manual    | Frame 0, error de reproyección  | calibración cada 30 frames |
| Rule-based eventos    | Tras procesar 1 partido completo | clasificador RF sobre features de ventana |
| SAM 3.1               | Día 1 de validación        | SAM 3 base                  |

## 8. Siguientes pasos accionables

1. Crear `.venv` con Python 3.12, instalar `torch 2.10+cu128`,
   `transformers ≥ 4.45`, `supervision`, `boxmot`, `opencv-python`.
2. Probar inferencia SAM 3.1 sobre `frame_24s.jpg` con prompt
   `"green felt soccer field with white lines"` y validar que la
   máscara cubre el campo.
3. Decidir herramienta de anotación (VIA vs CVAT vs Roboflow free
   tier) y anotar 50 frames como pilot.
4. Fork de [BoxMOT](https://github.com/mikel-brostrom/boxmot) o
   instalar via pip y correr OC-SORT sobre detecciones de SAM 3.1.
5. Documentar cada paso en `notebooks/01_sam3_baseline.ipynb`,
   `notebooks/02_homography.ipynb`, etc.
