# CLAUDE.md — futbotmx

## Descripción

Sistema de visión por computadora para análisis de partidos de fútbol robótico
de la Copa FutBotMX 2026 (Secihti + Meta + CENTRO). Categoría **Profesional**.

**Modelo base obligatorio**: SAM 3 (Segment Anything Model 3, Meta AI).

## Plazos críticos

- Registro: hasta el 22 de mayo de 2026, 23:59
- Entrega GitHub: 19 de junio de 2026, 23:59 (después: descalificación)
- Evento presencial: 24-26 de junio de 2026 (CDMX)

## Stack previsto

- **Segmentación**: SAM 3 (text/box/point prompts, vocabulario abierto)
- **Tracking**: ByteTrack sobre máscaras + lógica de identidades
- **Eventos**: pases, tiros, intercepciones, colisiones
- **Visualización**: heatmaps, trails, Voronoi, dashboards
- **Innovación obligatoria** (Profesional): fine-tuning SAM 3 sobre dataset
  oficial robótico

## Reglas duras de la convocatoria

- Repositorio **público** en GitHub con licencia MIT/Apache-2.0
- Video demo ≤ 2 min (original + segmentado lado a lado)
- **Reel Instagram ≥ 30 s con link en README — su ausencia DESCALIFICA**
- Cumplir licencia de SAM 3 (Meta)
- No plagio, no forks sin contribución sustantiva
- README orientado a jurado: arquitectura, instalación, hardware,
  resultados con capturas/GIFs, licencia

## Estructura

```
futbotmx/
├── src/{segmentation,tracking,events,viz,utils}/
├── data/{raw,processed,models}/      # gitignored
├── notebooks/, scripts/, tests/, docs/
```

## Convenciones

- Variables y logs en español, funciones/clases en inglés
- Ortografía completa: acentos, ñ, ¿?, ¡!
- `pathlib.Path` para rutas, `encoding="utf-8"` explícito
- `logging` en vez de `print()`
- Trabajar siempre en `.venv` propio

## Commits

- Conventional Commits: `feat|fix|docs|refactor|test|chore(scope): ...`
- Consultar antes de `git commit` y `git push`
- **NO incluir referencias a Claude, AI o asistentes en commits/PRs**

## Datos del dataset oficial

- **URL videos (Google Drive)**: https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD
- **Tipo de cámara**: Meta Ray-Ban Glasses (egocéntrica / primera persona)
- **Estructura**: `Meta_Glasses/{17Abril,18abril}/` (jornadas del torneo regional)
- **Implicación**: vista cambia con el portador, oclusiones frecuentes, sin
  cenital. El pipeline debe asumir cámara en movimiento.

## Datos pendientes (mandar correo a futbotmx@secihti.mx)

- Resolución, fps, codec, tamaño total
- Número de partidos y duración por partido
- Calibración de cámaras Meta Glasses (FOV, distorsión)
- Anotaciones / labels si existen
- Montos exactos de premios
- Criterios cuantitativos de evaluación

## SAM 3 — notas clave

- Compatible 100% con stack del entorno (torch 2.10+cu128, Python 3.12).
- VRAM inferencia: 8-12 GB. Fine-tuning LoRA: 12-24 GB (RTX 4090 viable).
- **SAM 3.1 (27/03/2026)** trae object multiplexing (16-32 objetos/forward) —
  preferir SAM 3.1 sobre SAM 3 si está disponible.
- HF gated: aceptar términos en huggingface.co/facebook/sam3 antes de descargar.
- **Licencia SAM ≠ MIT** — tratar como dependencia externa, NO vendorizar.
  Ver `THIRD_PARTY_LICENSES.md`.

## Precedentes útiles

- Paper "Team-Aware Football Player Tracking with SAM" — arXiv:2512.08467
- Vorp Labs SAM 3 + ByteTrack para NFL — https://www.vorplabs.com/blog/sam3-sports-tracking
- Roboflow fine-tune SAM 3 — https://blog.roboflow.com/fine-tune-sam3

## Recursos oficiales

- Convocatoria PDF: `../futbol-cv/convocatoria/Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf`
- Sitio Secihti: https://secihti.mx/futbotmx/
- Formulario registro: https://forms.cloud.microsoft/r/m8cwt7D7i0
- Reglas torneo: https://secihti.mx/wp-content/uploads/2026/01/Reglas_Copa_FutBotMX_v3_2026-01-21.pdf
- SAM 3 repo: https://github.com/facebookresearch/sam3
- SAM 3 paper: https://arxiv.org/abs/2511.16719
- Contacto oficial: futbotmx@secihti.mx
- Sede final: UPIITA-IPN, CDMX, 24-26 jun 2026
