# Fine-tuning LoRA de SAM 3.1 — innovación obligatoria § 3.7.3

> Fecha: 2026-05-19
> Categoría: **Profesional** (FutBotMX 2026)

## Por qué

La convocatoria § 3.7.3 exige al menos una línea de innovación sobre
SAM 3 para Categoría Profesional. Elegimos **fine-tuning LoRA** sobre
el dataset oficial del torneo porque:

1. Es la innovación más explícitamente sugerida por la convocatoria
   ("fine-tuning del prompt encoder" se cita literalmente).
2. Está validado por el survey de estado del arte (`teamaware_sam_2025`,
   `sam3_lora_sompote`).
3. Permite que el modelo aprenda especificidades del dominio (robots
   pequeños ~30×30 px, vista oblicua, motion blur, cancha verde con
   escuadras blancas) que no están en el preentrenamiento de Meta.

## Estrategia: pseudo-anotación + LoRA

**Sin anotación humana**. En lugar de etiquetar 200-500 frames a mano
(20+ horas), generamos un dataset pseudo-supervisado con SAM 3.1 base
y filtramos por score + filtros geométricos (área, aspect ratio). El
LoRA aprende sobre las máscaras curadas que el modelo base ya produce
con alta confianza.

Compromiso: la mejora marginal puede ser menor que con ground truth
real, pero cumple el requisito formal sin bloquear la entrega por
labor humana.

## Flujo

### 1. Generar pseudo-dataset

```bash
python scripts/generate_pseudo_annotations.py \
    --frames-per-video 15 \
    --score-min 0.6
```

Procesa los videos con duración >= 30 s del Drive oficial. Para cada
frame muestreado, corre SAM 3.1 con prompts "soccer robot" y "ball",
filtra por score + área + aspect, y guarda:

```
data/processed/pseudo_dataset/
    images/      <video>_<frame>.jpg
    masks/       <video>_<frame>_<categoría>_<idx>.png
    metadata.jsonl       # un JSON por máscara
```

Tiempo estimado: ~20-25 s/frame en RTX 5080 (carga del modelo +
2 prompts + filtros). Para 38 videos × 15 frames = 570 frames →
~3-4 horas.

### 2. Inspeccionar estructura SAM 3.1 (una sola vez)

```bash
python scripts/inspect_sam3_modules.py
```

Lista los módulos Linear por subcomponente para confirmar los nombres
de los target_modules de LoRA. Útil cuando Meta libera nueva versión.

Estructura verificada en `transformers 5.8.1` + `facebook/sam3`:
- `vision_encoder`: 192 Linear (q/k/v/o_proj + fc1/fc2 × 32 capas)
- `mask_decoder`: 4 Linear (q/k/v/o_proj de la attention)
- `text_encoder`, `geometry_encoder`: NO se tocan (preservan
  comprensión de prompts de texto y boxes)

### 3. Entrenar

```bash
# Dry-run de 1 época con 8 samples (validar API antes de full)
python scripts/train_sam3_lora.py --dry-run

# Full training (estimado ~6-8 h en RTX 5080)
python scripts/train_sam3_lora.py \
    --epochs 50 --rank 8 --alpha 16 \
    --batch-size 2 --accum-steps 8 \
    --lr 5e-5 --warmup-steps 50 \
    --eval-every 5
```

Configuración por defecto (basada en `sam3_lora_sompote`):
- rank=8, alpha=16, dropout=0.05
- target_modules=["q_proj","k_proj","v_proj","o_proj"] sobre
  vision_encoder y mask_decoder
- AdamW lr=5e-5, scheduler cosine, warmup 50 steps
- loss = BCEWithLogitsLoss + Dice (alpha=1.0)
- eval = mIoU sobre split por video (15% val)

### 4. Validar

```bash
python scripts/validate_lora.py --ckpt data/processed/lora_checkpoints/lora_final
```

(Pendiente de implementar) Compara mIoU base vs LoRA sobre el mismo
split val. Si mejora > 5 %, integrar al pipeline.

### 5. Integrar al pipeline de inferencia

`src/segmentation/sam3.py` debería aceptar un `lora_ckpt` opcional:

```python
processor, model = load_model(lora_ckpt="data/processed/lora_checkpoints/lora_best")
```

Cuando se pasa el ckpt, carga el modelo base + monta el adaptador LoRA
con `PeftModel.from_pretrained`. (Pendiente.)

## VRAM y tiempo

| Rank | VRAM (fp16) | Parámetros entrenables | mIoU esperado |
|------|-------------|------------------------|---------------|
| 8    | ~12 GB      | ~512 K (0.1 %)         | base + 3-8 %  |
| 16   | ~16 GB ⚠    | ~1 M (0.2 %)           | base + 5-10 % |
| 32   | OOM         | ~2 M (0.4 %)           | —             |

En RTX 5080 16 GB, **rank=8 es seguro**; rank=16 cabe justo pero puede
fallar si hay otra app consumiendo VRAM.

## Riesgos conocidos

1. **API exacta de Sam3Processor con `input_boxes`**: el script asume
   que `processor(images=..., input_boxes=[[bbox]], return_tensors="pt")`
   funciona y que `outputs.pred_masks` tiene shape `(B, N, H, W)`.
   Validar con `--dry-run` antes del full training.
2. **Pseudo-anotación sesgada**: el modelo aprende sus propios errores.
   La mejora real puede ser modesta. Como rescate, anotar 30-50 frames
   a mano (1-2 h en CVAT/Roboflow) y mezclar con pseudo (recommendation
   `dbscan_possession_2026`).
3. **Dataset desbalanceado**: probablemente >90 % máscaras de robot,
   <10 % de balón. Considerar oversampling de balón o entrenar 2 LoRA
   separados (uno por categoría).
4. **Resize del ground truth**: SAM downsamplea internamente a 256×256.
   El loss compara pred (256×256) con gt redimensionado con NEAREST.
   Si las máscaras son muy pequeñas (<32×32), pueden colapsar a 0
   tras el resize. Filtrar samples con bbox < 0.5 % del frame antes
   de entrenar.

## Referencias

- `sam3_lora_sompote` — github.com/Sompote/SAM3_LoRA
- `teamaware_sam_2025` — arXiv:2512.08467
- `roboflow_sam3_finetune` — blog.roboflow.com/fine-tune-sam3
