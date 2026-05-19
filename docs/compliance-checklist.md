# Checklist de cumplimiento — Convocatoria FutBotMX 2026

> Última revisión: 2026-05-19
> Fuente: convocatoria oficial (16 pp) + reglas del torneo (18 pp)
> Verificación final obligatoria antes del **2026-06-19 23:59**

## Estado de hoy (2026-05-19)

- 80→83 tests verdes (3 nuevos: recompute online, (h,s) tuples, votación temporal)
- AdaptiveTeamClassifier v2: warmup 30f, recompute cada 15f, vote_window 20,
  uso de saturación además de hue. Resuelve el colapso 100%/0% A
  observado en 3 de 4 videos del último batch.
- Detector v2 (líneas blancas + porterías por color) integrado al pipeline
- Bibliografía actualizada con 7 entradas del estado del arte 2026
  (PnLCalib, SAM3_LoRA, BoT-SORT, OSNet, STRIKER, etc.)
- Investigador externo confirma: votación temporal era exactamente el fix
  que necesitábamos (paper STRIKER IJRIAS 2026)

Leyenda: ✅ hecho · ⏳ en curso · ❌ pendiente · ⚠ riesgo descalificación

## 1. Registro (§ 3.10) — deadline 2026-05-22 23:59

- ❌ Formulario en línea llenado (https://forms.cloud.microsoft/r/m8cwt7D7i0)
- ❌ Nombre del equipo definido
- ❌ Datos de cada integrante: nombre completo, edad, institución,
  estado y ciudad de procedencia
- ❌ Correo electrónico de contacto del equipo
- ❌ Teléfono de contacto del equipo
- ❌ Categoría seleccionada: **Profesional** (forzosa por cláusula 3.7.1)
- ❌ Liga a carpeta digital con documentación 3.3.1
- ❌ Enlace al repositorio público en GitHub
- ❌ Aceptación expresa de las bases de la convocatoria

## 2. Documentación de identidad (§ 3.3.1)

Por cada integrante mexicano: cualquiera de
- ❌ INE
- ❌ Pasaporte mexicano
- ❌ Cédula profesional
- ❌ Cartilla militar

Por cada integrante extranjero con residencia en México:
- ❌ Pasaporte vigente + tarjeta de residente o FMM

Todos los documentos deben:
- ❌ Estar en formato PDF, legibles
- ❌ Coincidir con los datos del formulario
- ❌ Estar vigentes

## 3. Composición del equipo (§ 3.3.2 y reglamento § 2.1)

- ❌ Mínimo 2, máximo 4 integrantes
  (convocatoria permite individual, reglamento exige ≥2; aplica el más
  restrictivo si participamos en el torneo presencial; para el reto CV solo,
  la convocatoria permite individual)
- ❌ Todos mayores de 18 años (CV) / o cumplir rango etario del reglamento
- ❌ Todos mexicanos o extranjeros con residencia
- ❌ Cada integrante con perfil público (GitHub o LinkedIn)
- ⚠ **Ningún integrante puede ser funcionario o servidor público
  federal/estatal/local** si se quiere recibir premio monetario o en
  especie (§ 3.9). Verificar antes de inscribir.

## 4. Categoría Profesional (§ 3.2.2)

Indicios que activan reclasificación automática (§ 3.7.1):
- ⏳ Implementación de fine-tuning de SAM 3
- ⏳ Pipeline de procesamiento con arquitectura de producción
- ⏳ Métricas cuantitativas avanzadas
- ⏳ Integración de múltiples modelos complementarios

Inscribiremos directamente en Profesional para evitar reclasificación.

## 5. Entregables en repositorio público (§ 3.5) — deadline 2026-06-19 23:59

### 5.1 Pipeline (§ 3.5.1)
- ❌ Código funcional que aplique **SAM 3 de Meta**
- ❌ Segmentación de campo, robots aliados, robots rivales, balón
- ❌ Tracking de trayectorias de robots y balón
- ❌ Detección de eventos clave: pases, tiros a gol, intercepciones,
  colisiones (al menos uno de los listados)

### 5.2 Visualización (§ 3.5.2)
Al menos UNA de las siguientes:
- ⏳ Heatmap dinámico por equipo o por robot
- ⏳ Análisis de posesión con métricas temporales
- ⏳ Visualizaciones de flujo (trails, Voronoi, grafos)
- ⏳ Dashboards o anotaciones narrativas
- Plan actual: heatmap + posesión + Voronoi (3 visualizaciones).

### 5.3 Videos (§ 3.5.3)
- ❌ Video de **máximo 2 minutos** mostrando análisis sobre 1 partido
  - Vista original junto a vista segmentada (lado a lado o superpuesto)
  - Indicadores visuales de segmentación, tracking, visualizaciones
  - Breve explicación en texto o voz del enfoque
- ⚠ **Reel publicado en Instagram, mínimo 30 segundos**, link en README
  - **SU AUSENCIA DESCALIFICA** (§ 3.5.3)

### 5.4 README.md (§ 3.5.4)
- ⏳ Descripción del enfoque y arquitectura de la solución
- ⏳ Instrucciones de instalación y reproducción paso a paso
- ⏳ Requisitos de hardware y software (GPU, dependencias)
- ⏳ Resultados con capturas de pantalla o GIFs
- ⏳ Enlace al reel de Instagram (acceso público)
- ⏳ Licencia del proyecto y créditos (§ 3.11)
- ⏳ Atribución de todas las dependencias de terceros (§ 3.6)

### 5.5 Reproducibilidad (§ 3.2.2 Profesional)
- ⏳ Seed fijo en todos los notebooks y scripts
- ⏳ `requirements.txt` con versiones pinneadas
- ⏳ Instrucciones de obtención de pesos SAM 3 (no vendorizar)
- ⏳ Datos de entrada listados (videos del Drive oficial)
- ⏳ Outputs esperados documentados

## 6. Licencia y código de terceros (§ 3.6, § 3.11)

- ✅ Licencia abierta del repo: **MIT** (ver `LICENSE`)
- ✅ `THIRD_PARTY_LICENSES.md` con SAM 3 como dependencia externa
- ⏳ Cada dependencia atribuida en README (Roboflow Supervision, ByteTrack,
  OpenCV, transformers, etc.)
- ⏳ Cada dependencia con licencia respetada
- ⏳ Equipo capaz de explicar el rol de cada dependencia en el pipeline
  (defensa del entregable)
- ✅ Se cumple licencia de SAM 3 (Meta) — no vendorizar, descargar al setup

## 7. Causales de descalificación (§ 3.7.4) — NO incurrir

- ✅ Sin plagio (código y documentación originales)
- ✅ Sin suplantación de identidad
- ⚠ Repo **DEBE estar público** al 2026-06-19 23:59 (actualmente privado)
- ⚠ Entregable **DEBE incluir** pipeline + video demo + reel IG
- ✅ Sin falsificación de nivel (inscribir directamente en Profesional)
- ✅ Respeto a licencias (SAM 3, OpenCV, ByteTrack, etc.)

## 8. Restricciones críticas de fechas

| Fecha               | Restricción                                                        |
|---------------------|--------------------------------------------------------------------|
| 2026-05-22 23:59    | **No se aceptan registros después de esta hora**                   |
| 2026-06-19 23:59    | **No se permiten cambios al repo después de esta hora**            |
| 2026-06-20 → 06-24  | Periodo de evaluación: repo congelado, sin modificaciones          |

## 9. Propiedad intelectual (§ 3.11)

- ✅ Conservamos titularidad plena del código y documentación
- ✅ Otorgamos licencia no exclusiva a Secihti para exhibir/publicar/difundir
- ✅ Cumplimos requisito de licencia abierta (MIT)

## 10. Aviso de privacidad (§ 3.12)

- ✅ Conformes con compartir datos al Comité Evaluador (al registrar)
- ❌ Conscientes de que GitHub y LinkedIn son perfiles públicos compartidos

## 11. Innovación sobre SAM 3 (§ 3.7.3) — requisito Profesional

Mínimo UNA línea: **cubrimos las CUATRO**.

- ✅ **Prompts y contexto**: prompts simples vs elaborados validados
  empíricamente; `"soccer robot"` (0.94) supera a
  `"small mobile soccer robot with a colored flag"` (0.34). Diseño en
  `src/segmentation/prompts.py`.
- ✅ **Fine-tuning LoRA**: infraestructura completa con peft 0.19,
  rank=8 sobre q/k/v/o_proj de vision_encoder + mask_decoder
  (3.88M params, 0.46% del modelo). Dataset pseudo-supervisado de 524
  máscaras de 15 videos. Dry-run validado end-to-end. Documentado en
  `docs/lora-finetuning.md`.
- ✅ **Post-procesamiento**: Kalman 2D del balón con fallback HSV;
  AdaptiveTeamClassifier v2 (recompute online + votación temporal +
  (hue, sat)); homografía con esquinas reales por líneas blancas;
  detección de gol con bbox REAL de portería (amarilla/azul) en lugar
  de ROI virtual.
- ✅ **Integración con trackers**: cascada SAM 3.1 → OC-SORT (robots) +
  Kalman 2D (balón); BoxMOT 18 con OC-SORT, ya soporta StrongSORT/
  BoT-SORT si se requiere mejora futura.

## 12. Verificación final pre-entrega (correr el 2026-06-18)

- ❌ Repo es público y accesible sin login
- ❌ README abre correctamente desde browser
- ❌ Reel de Instagram accesible públicamente
- ❌ Video demo ≤ 2 min, juega sin errores
- ❌ `LICENSE` presente, atribuciones en README
- ❌ Pipeline reproducible siguiendo solo el README
- ❌ Tag de release creado (e.g., `v1.0-entrega`)
- ❌ Captura de pantalla del repo público enviada por correo de respaldo
