# futbotmx

Sistema de visión por computadora para análisis de partidos de fútbol robótico
(Copa FutBotMX 2026, capítulo Visión por Computadora — Secihti · Meta · CENTRO).

> Categoría: **Profesional** · Modelo base obligatorio: **SAM 3 (Meta)**

## Estado

Repositorio en construcción. Deadline de entrega: **19 de junio de 2026, 23:59**.

## Stack previsto

- **Segmentación**: SAM 3 (Meta) — text/box/point prompts, vocabulario abierto
- **Tracking**: ByteTrack sobre máscaras SAM 3
- **Visualización**: Roboflow Supervision, OpenCV
- **Análisis**: posesión, pases, tiros, intercepciones, colisiones
- **Innovación SAM 3** (requisito Profesional): fine-tuning sobre frames del dataset
  oficial de la Copa FutBotMX

## Estructura

```
futbotmx/
├── src/
│   ├── segmentation/   # Wrapper SAM 3 + utilidades de prompts
│   ├── tracking/       # ByteTrack + lógica de identidades de robots
│   ├── events/         # Detección de pases, tiros, colisiones
│   ├── viz/            # Heatmaps, trails, Voronoi, dashboards
│   └── utils/          # IO, métricas, exporters
├── data/
│   ├── raw/            # Videos oficiales (SOLO LECTURA)
│   ├── processed/      # Outputs del pipeline
│   └── models/         # Pesos SAM 3 y fine-tuned
├── notebooks/          # Exploración, fine-tuning, análisis
├── scripts/            # CLI entrypoints
├── tests/              # Pruebas unitarias
└── docs/               # Documentación técnica
```

## Instalación

Pendiente. Requerirá GPU NVIDIA con CUDA, Python 3.11+, torch 2.x.

## Entregables

- [ ] Pipeline SAM 3 + tracking + eventos
- [ ] Al menos 1 visualización (heatmaps / Voronoi / posesión / dashboard)
- [ ] Video demo ≤ 2 min (original + segmentado lado a lado)
- [ ] Reel Instagram ≥ 30 s (link aquí cuando esté publicado)
- [ ] README final con resultados, capturas, GIFs
- [ ] Licencia: MIT (ver `LICENSE`)

## Dataset

Videos oficiales de la Copa FutBotMX 2026 alojados en Google Drive:

- Drive: https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD
- Estructura: `Meta_Glasses/{17Abril,18abril}/`

La primera muestra inspeccionada fue grabada con **iPhone 16 Pro Max**
desde la orilla del campo (no Meta Ray-Ban como sugiere el nombre de la
carpeta). El tipo de cámara puede ser mixto a lo largo del dataset —
ver `docs/dataset-inspection.md` para el análisis técnico completo.

## Recursos oficiales

- Convocatoria Secihti: https://secihti.mx/futbotmx/
- Formulario registro: https://forms.cloud.microsoft/r/m8cwt7D7i0
- Reglas torneo: https://secihti.mx/wp-content/uploads/2026/01/Reglas_Copa_FutBotMX_v3_2026-01-21.pdf
- SAM 3 repo: https://github.com/facebookresearch/sam3
- Paper SAM 3: https://arxiv.org/abs/2511.16719
- Contacto: futbotmx@secihti.mx
- Sede final: UPIITA-IPN, CDMX, 24-26 jun 2026

## Licencia

Código bajo MIT (ver `LICENSE`). Dependencias de terceros respetan sus
licencias originales — ver `THIRD_PARTY_LICENSES.md`. En particular, **SAM 3
se distribuye bajo licencia propia de Meta (no MIT)** y se trata como
dependencia externa.
