# Inspección de dataset — muestra inicial

> Fecha: 2026-05-16
> Muestra: `data/raw/IMG_9915.MOV` (un video del Drive oficial)
> Origen: https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD

## Resumen ejecutivo

El video oficial **no fue grabado con Meta Ray-Ban Glasses** sino con un
**iPhone 16 Pro Max** sostenido por un espectador a la orilla del campo.
Esto contradice la hipótesis inicial del proyecto (cámara egocéntrica) y
**cambia el diseño del pipeline**: la cámara tiene paneos suaves en lugar
de giros bruscos de cabeza, lo cual es más favorable para SAM 3 + ByteTrack.

## Metadatos técnicos

| Campo            | Valor                                              |
|------------------|----------------------------------------------------|
| Container        | QuickTime / MOV                                    |
| Codec video      | H.264 High Profile, level 4.0, yuv420p             |
| Resolución       | 1920 × 1080 (Full HD)                              |
| Frame rate       | 29.97 fps (30000/1001)                             |
| Duración         | 95.97 s (~1:36)                                    |
| Frames totales   | 2878                                               |
| Bitrate video    | 15.36 Mbps                                         |
| Color space      | BT.709 (HD estándar)                               |
| Tamaño en disco  | 186 MB (194 298 351 bytes)                         |
| Audio estéreo    | AAC LC 48 kHz, 217 kbps                            |
| Audio spatial    | APAC 4 canales, 397 kbps (spatial audio Apple)     |
| Streams metadata | 5 streams `mebx` (sensor data: orientación, GPS…) |

### Metadatos EXIF del dispositivo

| Campo                | Valor                                             |
|----------------------|---------------------------------------------------|
| Make                 | Apple                                             |
| Model                | iPhone 16 Pro Max                                 |
| Software             | iOS 26.3.1                                        |
| Creación             | 2026-04-18 08:35:42 -0600 (hora central México)   |
| GPS (ISO 6709)       | +19.2258, -97.7801, altitud 2365.5 m              |
| Ubicación inferida   | Puebla, MX (UT de Puebla, evidencia en frames)    |

## Análisis visual de 5 frames

Frames extraídos a 0, 24, 48, 72 y 95 segundos en
`data/processed/sample_frames/`.

### Escena

- Cancha rectangular de fieltro **verde** con líneas blancas que marcan
  áreas de portería en forma de "D" (estilo RoboCup MSL/SSL).
- **Vallas perimetrales negras** (rebound walls) y **2 porterías
  amarillas** en lados opuestos.
- Cajas/topes negros laterales adicionales como límites.
- Audiencia visible al fondo (logos de "ROBÓTICA", "UT Puebla").
- Iluminación interior, artificial, uniforme y estable a lo largo del
  video. No hay sombras dinámicas significativas.

### Robots

- **2 robots activos**, uno por equipo, formato 1v1.
- Cuerpo cilíndrico bajo (~10-12 cm visible), llanta circular, electrónica
  expuesta arriba.
- **Identificación por bandera vertical** (flag), no por color de chasis:
  - Equipo A: bandera **morada/violeta**.
  - Equipo B: bandera **blanca con detalles verde-lima**.
- Las banderas son **la firma de identidad de equipo** — si se cae o se
  ocluye, el tracking de identidad se rompe.

### Balón

- Pelota **naranja brillante**, ~3 cm aparente (tipo pelota de tenis o
  similar). Alto contraste contra el fieltro verde.
- Es el objeto **más pequeño y más fácil de perder** del cuadro.
  Se ocluye detrás de los robots y rebota contra paredes con frecuencia.

### Cámara

- Posición: a la orilla del campo, oblicuo, altura de pecho/cintura.
- Movimiento: **paneos suaves**, alguna sacudida (motion blur ocasional
  visible en frame 72s).
- **NO egocéntrico**, **NO cenital**, **NO broadcast**. Es vista de
  espectador con micromovimientos. Más favorable que Meta Ray-Ban para CV.

## Implicaciones para el pipeline SAM 3

### Categorías a segmentar (4 requeridas por convocatoria)

| Categoría        | Prompt SAM 3 sugerido (vocabulario abierto)       |
|------------------|---------------------------------------------------|
| Campo            | `"green felt soccer field with white lines"`      |
| Robots equipo A  | `"small mobile robot with purple flag on top"`    |
| Robots equipo B  | `"small mobile robot with white flag on top"`     |
| Balón            | `"small bright orange ball"`                      |

### Estrategia recomendada

1. **SAM 3.1 (preferido)** con prompts de vocabulario abierto. Si SAM 3.1
   no está disponible, SAM 3 con `transformers>=4.45` (HF).
2. **Frame stride**: procesar 1 de cada 3-5 frames (~6-10 fps efectivos)
   es suficiente para tracking, robots no se mueven a velocidad extrema.
3. **ByteTrack sobre máscaras** + lógica adicional de re-identificación
   basada en **color de bandera** (clasificador HSV simple).
4. **Homografía campo → top-down**: usar las 4 esquinas del campo (o las
   líneas blancas) para proyectar posiciones a vista cenital. Necesario
   para heatmaps y Voronoi.
5. **Detección de balón redundante**: SAM 3 + detector clásico por color
   (HSV naranja) como fallback. El balón es crítico para detectar pases,
   tiros y goles.

### Estimaciones de cómputo

- 2878 frames @ 1080p = ~3000 segmentaciones por video.
- SAM 3.1 en RTX 4090: ~5-10 fps reales con prompts complejos.
- Tiempo total por video de 96 s: ~5-15 min de inferencia.
- VRAM esperada: 8-12 GB (inferencia), 16-24 GB (fine-tuning LoRA).
- Si procesamos cada 3 frames → ~1.5-5 min por video. **Viable.**

### Detalles que ayudan al pipeline

- **Iluminación uniforme** → menos variabilidad, modelo no necesita
  data augmentation agresivo de brillo.
- **Líneas blancas nítidas** → calibración de homografía sin esfuerzo.
- **Balón de alto contraste** → detector por color es backup confiable.
- **Robots de morfología similar** → un solo prompt "robot" puede
  detectar ambos equipos, separación por bandera viene después.

### Detalles que complican el pipeline

- **Banderas son frágiles** → ID switches frecuentes si bandera cambia
  de orientación, se inclina o se ocluye momentáneamente.
- **Manos humanas entran al cuadro** al inicio (posicionamiento) y en
  reemplazos. Hay que filtrar "no robot" del segmentador.
- **Cámara móvil** → tracking absoluto en coordenadas de imagen no
  refleja posición real del robot en el campo. Homografía es obligatoria
  para análisis (no opcional).
- **Balón se ocluye** → tracking necesita interpolación / predicción
  Kalman cuando el balón desaparece momentáneamente.
- **Solo 1v1** en este video → si el torneo es 2v2 o 3v3, la cantidad
  de objetos y oclusiones aumenta significativamente. Confirmar con
  Secihti.

## Pendientes derivados de este análisis

1. **Confirmar con Secihti**: ¿la carpeta `Meta_Glasses` del Drive es
   etiquetado erróneo o realmente hay videos de Ray-Ban además de iPhone?
   Bajar 2-3 videos más para verificar heterogeneidad.
2. **Confirmar formato del torneo**: ¿1v1, 2v2, 3v3? Este video es 1v1.
3. **Actualizar `CLAUDE.md`** para reflejar que el dataset NO es
   exclusivamente Ray-Ban (este video es iPhone).
4. **Considerar streams `mebx`** del iPhone: contienen orientación,
   giroscopio, GPS. Si el equipo aprovecha esos sensores para
   estabilización o fusión, sería un diferenciador frente a otros
   participantes. Extraerlos requiere herramienta específica
   (`pymp4` o similar). Tarea opcional para Fase 2.

## Convención de evaluación local

Frames y outputs de prueba:

- `data/raw/` → originales (solo lectura).
- `data/processed/sample_frames/` → frames JPEG de muestra.
- `data/processed/runs/<fecha>/` → salidas de pipeline por corrida.
