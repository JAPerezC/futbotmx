# Prompt para iniciar nueva sesión en `futbotmx`

> Pega el bloque siguiente como primer mensaje en una nueva sesión de
> Claude Code corriendo desde `C:\Users\jorge\work\projects\futbotmx`.
> Es autocontenido: la nueva sesión arranca con todo el contexto necesario.

---

## Prompt (copia desde aquí)

Hola. Esta es la **primera sesión** del proyecto `futbotmx`. Necesito que
arranques con el contexto completo y sin mezclar nada del proyecto vecino
`futbol-cv` (ese proyecto está pausado y NO debe entrar aquí).

### Qué es este proyecto

Participación en el reto **Copa FutBotMX 2026 — Capítulo Visión por
Computadora**, organizado por Secihti (gobierno de México) + Meta + CENTRO.
Hay que construir un sistema de CV que use el modelo **SAM 3 de Meta** para
segmentar, trackear y analizar videos de partidos de **fútbol robótico**.

Categoría: **Profesional** (forzosa por nivel técnico — sec. 3.7.1 cláusula
de reclasificación). Esto obliga a "innovación sobre SAM 3" (fine-tuning,
prompt engineering avanzado o integración con otro modelo).

### Plazos críticos

- **Hoy: 2026-05-15** (verifica con `date`).
- Cierre de registro: **22 mayo 2026, 23:59** (7 días hábiles).
- Deadline entrega GitHub: **19 junio 2026, 23:59** (35 días).
- Evento presencial: 24-26 jun 2026 en UPIITA-IPN, CDMX.

### Hallazgos clave (no perder de vista)

1. **El dataset es EGOCÉNTRICO**, no broadcast ni cenital. Fue capturado con
   **gafas Meta Ray-Ban** durante los partidos. Pipeline debe ser robusto a
   cámara en movimiento y oclusiones constantes.
   - URL Drive: https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD
   - Estructura: `Meta_Glasses/{17Abril,18abril}/`

2. **SAM 3.1 (27/03/2026) > SAM 3**: trae object multiplexing
   (hasta 32 fps con múltiples objetos). Preferir 3.1 si está disponible.

3. **Licencia de SAM 3 NO es MIT/Apache** — es custom Meta SAM License.
   **NO se puede vendorizar dentro del repo**. Tratar como dependencia
   externa (instalar al setup, no commit). Ya documentado en
   `THIRD_PARTY_LICENSES.md`.

4. **HF gated**: huggingface.co/facebook/sam3 requiere aceptar términos
   manualmente antes de descargar pesos. Aún NO se ha hecho.

5. **Stack actual del sistema** (verificado): torch 2.10+cu128, Python 3.12,
   NVIDIA CUDA. Compatible 100% con SAM 3.

6. **Precedentes útiles**:
   - Paper arXiv:2512.08467 "Team-Aware Football Player Tracking with SAM"
   - Vorp Labs SAM 3 + ByteTrack para NFL
   - Roboflow blog fine-tune SAM 3

### Estado del repo (lo que ya está hecho)

- `LICENSE` (MIT) y `THIRD_PARTY_LICENSES.md`
- `README.md` y `CLAUDE.md` con dataset URLs y reglas
- `.gitignore`, `.envrc`, `requirements.txt` base
- Estructura `src/{segmentation,tracking,events,viz,utils}/__init__.py`
- `docs/plan-sprint-2026.md` con fases hasta 19 jun
- `git init` hecho, 13 archivos staged, **commit inicial PENDIENTE**
  (falló por passphrase SSH del `id_ed25519_github`)

### Entregables obligatorios (sec. 3.5 de la convocatoria)

1. Repo público en GitHub con licencia MIT/Apache
2. Pipeline SAM 3 que segmente: campo, robots aliados, robots rivales, balón
3. Tracking de trayectorias robots + balón
4. Detección de eventos (pases, tiros, intercepciones, colisiones)
5. Al menos 1 visualización (heatmap / Voronoi / posesión / trails / dashboard)
6. **Video demo ≤ 2 min** (original + segmentado lado a lado)
7. **Reel Instagram ≥ 30 s con link en README** — **su ausencia DESCALIFICA**
8. README orientado a jurado (arquitectura, instalación, hardware, resultados)

### Pendientes inmediatos en orden

1. **Resolver commit inicial bloqueado por passphrase SSH**. Opciones:
   - `ssh-add ~/.ssh/id_ed25519_github` (carga clave una vez)
   - O commit con passphrase interactiva
   - NO usar `--no-gpg-sign` sin permiso explícito del usuario
2. Crear repo público en GitHub `futbotmx` y `git push -u`
3. Definir nombre del equipo y reunir datos de los 3-4 integrantes
4. Aceptar gating de SAM 3 en https://huggingface.co/facebook/sam3
5. Descargar muestra del dataset desde el Drive y reportar:
   resolución, fps, codec, duración total, # partidos
6. Redactar correo a `futbotmx@secihti.mx` pidiendo ficha técnica del
   dataset + montos de premios + criterios cuantitativos de evaluación
7. Llenar formulario: https://forms.cloud.microsoft/r/m8cwt7D7i0 antes
   del 22 may 23:59 (URL del repo público va aquí)

### Reglas duras de trabajo

- **Idioma**: español, con ortografía completa (acentos, ñ, ¿?, ¡!).
- **Commits**: Conventional Commits (`feat|fix|docs|refactor|test|chore`).
  **NUNCA incluir trailers Co-Authored-By: Claude ni referencias a IA/AI/
  asistente** en mensajes de commit, PRs ni descripciones. Los commits
  deben verse como autoría 100% humana.
- **Git**: confirmar antes de cada `commit` y `push`. No crear/borrar
  ramas sin permiso.
- **Datos**: `data/raw/` y videos descargados son SOLO LECTURA.
- **Secretos**: nunca tokens/keys en código. `.env` y `.secrets/` ya
  están en `.gitignore`.
- **El proyecto vecino `futbol-cv` está PAUSADO**. No tocarlo, no
  comparar, no copiar archivos sin pedir permiso primero.

### Recursos oficiales

- Convocatoria PDF: `../futbol-cv/convocatoria/Convocatoria_CopaFutBotMX-Meta-VF-20260429T020141.pdf`
- Sitio: https://secihti.mx/futbotmx/
- Formulario: https://forms.cloud.microsoft/r/m8cwt7D7i0
- Reglas torneo: https://secihti.mx/wp-content/uploads/2026/01/Reglas_Copa_FutBotMX_v3_2026-01-21.pdf
- SAM 3: https://github.com/facebookresearch/sam3
- HF: https://huggingface.co/facebook/sam3
- Paper SAM 3: https://arxiv.org/abs/2511.16719
- Fine-tune Roboflow: https://blog.roboflow.com/fine-tune-sam3
- Contacto: futbotmx@secihti.mx

### Tu primera acción

1. Lee `CLAUDE.md`, `README.md` y `docs/plan-sprint-2026.md` del repo.
2. Verifica que `git status` muestra los 13 archivos staged sin commit.
3. Pregúntame: ¿avanzo con el commit inicial (necesito passphrase SSH),
   o prefieres priorizar otra cosa (descargar dataset, aceptar gating
   HF, redactar correo a Secihti, definir equipo)?

No empieces a escribir código todavía. Primero validamos contexto y
desbloqueamos lo administrativo.
