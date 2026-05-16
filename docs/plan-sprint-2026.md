# Plan de sprint Copa FutBotMX — 2026-05-15 a 2026-06-19

## Hallazgos críticos (2026-05-15)

- **Dataset es egocéntrico** (Meta Ray-Ban Glasses), no broadcast ni cenital.
  Implica: pipeline robusto a cámara en movimiento + oclusiones constantes.
- **SAM 3.1 > SAM 3** para multi-objeto (object multiplexing, 32 fps en H100).
- **Licencia SAM 3 ≠ MIT** — tratar como dependencia externa.
- HF gated: aceptar términos manualmente antes de descargar pesos.

## Fases

### Fase 0 — Bloqueantes (2026-05-15 a 2026-05-22, ~7 días)
- [x] Validar SAM 3 técnicamente (research; falta inferencia local)
- [x] Obtener URL del repositorio oficial de videos
- [ ] Definir composición del equipo (3-4 personas)
- [ ] Reunir documentación: INE/pasaporte de cada integrante,
      perfiles públicos GitHub/LinkedIn
- [ ] Aceptar gating de SAM 3 en HF: huggingface.co/facebook/sam3
- [ ] Descargar muestra del dataset y inspeccionar (resolución, fps, duración)
- [ ] Mandar correo a futbotmx@secihti.mx pidiendo ficha técnica del dataset
- [ ] Llenar formulario: https://forms.cloud.microsoft/r/m8cwt7D7i0
- [ ] Hacer público el repo `futbotmx` en GitHub e incluir URL en el formulario

### Fase 1 — Pipeline mínimo viable (2026-05-23 a 2026-06-05, ~14 días)
- [ ] Wrapper SAM 3 con prompts (campo, robots aliados, robots rivales, balón)
- [ ] Tracking ByteTrack sobre máscaras
- [ ] Lógica básica de eventos: pase, tiro, colisión
- [ ] Output: video segmentado + JSON de eventos
- [ ] Pipeline corre end-to-end sobre 1 partido completo

### Fase 2 — Innovación + visualización (2026-06-06 a 2026-06-14, ~9 días)
- [ ] Innovación SAM 3 (criterio Profesional obligatorio):
      fine-tuning ligero, prompt engineering avanzado o integración con
      otro modelo
- [ ] Al menos 2 visualizaciones de la lista oficial
      (heatmaps, posesión, trails, Voronoi, grafos, dashboard)
- [ ] Métricas cuantitativas: latencia, mIoU, conteo eventos
- [ ] Tests unitarios mínimos

### Fase 3 — Entrega (2026-06-15 a 2026-06-19, ~5 días)
- [ ] Video demo ≤ 2 min (original + segmentado, voz/texto)
- [ ] Reel Instagram ≥ 30 s publicado, link en README
- [ ] README final orientado a jurado
- [ ] Tag de release antes del 19 jun 23:59
- [ ] Verificación final: repo público, licencia, créditos, atribuciones

## Riesgos

| Riesgo                          | Mitigación                                                |
|---------------------------------|-----------------------------------------------------------|
| SAM 3 no corre en GPU disponible| Validar día 1 (Fase 0). Fallback: usar HF Spaces / colab. |
| Dataset robótico muy distinto   | Análisis temprano del dataset. Adaptar prompts y eventos. |
| Tiempo de fine-tuning           | Empezar con prompt engineering (no requiere training).    |
| Reel IG olvidado                | Tarea explícita en Fase 3. Recordatorio en checklist.     |
| Equipo incompleto en registro   | Decidir equipo en semana 1. Plan B: registrar solo.       |

## Métricas de éxito mínimas

- Pipeline corre sin crashes sobre ≥ 1 partido completo
- Detecciones consistentes (>80% de robots trackeados sin ID switch en 30 s)
- Al menos 2 visualizaciones publicables
- Video demo y reel listos antes del 17 de junio (buffer 2 días)
