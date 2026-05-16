# Plan de sprint Copa FutBotMX — 2026-05-15 a 2026-06-19

## Hallazgos críticos

### 2026-05-15
- **SAM 3.1 > SAM 3** para multi-objeto (object multiplexing, 32 fps en H100).
- **Licencia SAM 3 ≠ MIT** — tratar como dependencia externa.
- HF gated: aceptar términos manualmente antes de descargar pesos.

### 2026-05-16 (tras inspeccionar primera muestra)
- **El dataset NO es Ray-Ban egocéntrico** como suponíamos. La primera muestra
  (`data/raw/IMG_9915.MOV`) es de **iPhone 16 Pro Max** desde la orilla del
  campo. Ver `docs/dataset-inspection.md` para análisis completo.
- **Formato del partido (en esta muestra): 1v1**, con cancha tipo RoboCup MSL,
  porterías amarillas, vallas negras perimetrales, balón naranja pequeño.
- **Identidad de equipo se da por bandera vertical** (morada vs. blanca/verde),
  no por color de chasis. ID switches probables si bandera se ocluye.
- **Resolución 1920×1080 @ 29.97 fps** — manejable para SAM 3 sin downsampling
  agresivo.
- **Pendiente**: bajar 2-3 videos más para verificar si todo es iPhone o hay
  videos Ray-Ban en otras subcarpetas del Drive.

### 2026-05-16 (tras leer convocatoria y reglas oficiales completas)
- **Calendario oficial confirmado**: cierre registro 22 may, entregable 19 jun
  23:59, evaluación 20-24 jun, resultados 25 jun, copa presencial 24-26 jun,
  premiación 26 jun.
- **Después del 19 jun NO se permiten cambios al repo** — cualquier modificación
  posterior puede ser motivo de descalificación.
- **Equipos: 2-4 personas** (no individual, según reglamento del torneo);
  la convocatoria CV permite individual o hasta 4. Aplicar el más restrictivo.
- **Categoría del torneo**: Abierta (balón pasivo naranja 42mm). Confirmado por
  el dataset.
- **Máximo 2 robots por equipo** (formato hasta 2v2).
- **Geometría del campo conocida con precisión** (sec. 7 reglamento):
  219×158 cm zona de juego, líneas blancas de 2 cm, círculo central 60 cm.
  Esto permite homografía exacta de 4 puntos.
- **Eventos detectables del reglamento**: gol (balón toca pared trasera de
  portería), retención (prohibida — balón pegado a robot), kick, empujón
  en área de penalti, falta de progreso, robot dañado.
- **Premios económicos NO se entregan a funcionarios o servidores públicos**.
  Aplica a integrantes del equipo — verificar al definir miembros.
- **Menciones especiales sin restricción**: Mejor Visualización y Mejor
  Documentación en ambas categorías. Buen objetivo secundario.
- **Propuesta de arquitectura técnica completa**: ver `docs/architecture.md`.
- **Checklist de cumplimiento**: ver `docs/compliance-checklist.md`.

## Fases

### Fase 0 — Bloqueantes (2026-05-15 a 2026-05-22, ~7 días)
- [x] Validar SAM 3 técnicamente (research; falta inferencia local)
- [x] Obtener URL del repositorio oficial de videos
- [ ] Definir composición del equipo (3-4 personas)
- [ ] Reunir documentación: INE/pasaporte de cada integrante,
      perfiles públicos GitHub/LinkedIn
- [x] Aceptar gating de SAM 3 en HF: huggingface.co/facebook/sam3
- [x] Descargar muestra del dataset y inspeccionar (resolución, fps, duración)
      → ver `docs/dataset-inspection.md`
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

## Calendario oficial (convocatoria § 3.8)

| Fecha               | Hito                                                          |
|---------------------|---------------------------------------------------------------|
| 2026-04-27          | Apertura de registro + publicación del repositorio de videos  |
| **2026-05-22 23:59**| **Cierre de registro** (formulario + URL del repo público)    |
| 2026-05-25 → 06-19  | Periodo de desarrollo y acompañamiento online                 |
| **2026-06-19 23:59**| **Fecha límite de entregable en GitHub** (repo bloqueado)     |
| 2026-06-20 → 06-24  | Periodo de evaluación del Comité                              |
| 2026-06-25          | Publicación de resultados                                     |
| 2026-06-24 → 06-26  | Copa FutBotMX presencial (UPIITA-IPN, CDMX)                   |
| 2026-06-26          | Premiación                                                    |

## Riesgos

| Riesgo                                      | Mitigación                                                       |
|---------------------------------------------|------------------------------------------------------------------|
| SAM 3 no corre en GPU disponible            | Validar día 1 (Fase 0). Fallback: usar HF Spaces / Colab Pro.    |
| Dataset robótico muy distinto al sample     | Bajar 5-10 videos antes del 25 may, re-inspeccionar.             |
| Tiempo de fine-tuning insuficiente          | Empezar con prompt engineering; LoRA solo si MVP funcional.      |
| Reel IG olvidado (DESCALIFICA)              | Tarea explícita en Fase 3. Hito 17 jun en calendario.            |
| Equipo incompleto en registro               | Decidir equipo esta semana. Plan B: equipo de 2 personas.        |
| Repo privado al 19 jun (DESCALIFICA)        | Hito 18 jun: cambiar visibilidad. Verificación manual.           |
| Reproducibilidad falla (req. Profesional)   | Seed fijo, requirements pinneados, instrucciones de hardware.    |
| Cámara móvil rompe homografía estática      | Re-calibrar cada N frames; verificar drift.                      |
| Premio rechazado por estatus funcionario    | Verificar al definir equipo: integrantes NO deben ser servidores |
|                                             | públicos federales/estatales/locales (§ 3.9).                    |

## Métricas de éxito mínimas

- Pipeline corre sin crashes sobre ≥ 1 partido completo (96 s mínimo).
- Detecciones consistentes (>80% de robots trackeados sin ID switch en 30 s).
- Al menos 2 visualizaciones publicables (heatmap + posesión propuestos).
- Video demo y reel IG listos antes del **17 de junio** (buffer 2 días).
- README final aprobado por checklist `docs/compliance-checklist.md`.
