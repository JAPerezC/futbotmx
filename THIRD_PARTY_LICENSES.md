# Licencias de terceros

Este proyecto está bajo licencia MIT (ver `LICENSE`). Sin embargo, usa
componentes de terceros con licencias propias que se deben respetar.

## SAM 3 (Segment Anything Model 3, Meta AI)

- **Licencia**: SAM License (Meta) — NO es MIT/Apache.
- **Repo**: https://github.com/facebookresearch/sam3
- **Pesos**: https://huggingface.co/facebook/sam3 (gated, requiere aceptar términos)
- **Restricciones clave**:
  - Prohibido uso militar/ITAR, armas, nuclear, entidades sancionadas.
  - Redistribución obligatoria con la misma licencia SAM.
  - Atribución obligatoria en publicaciones.
- **Tratamiento en este repo**: **dependencia externa**. NO se vendoriza dentro
  del código. Se instala desde el repo/HF oficial en el paso de setup.
- Cita: Carion et al. (2025). "SAM 3: Segment Anything with Concepts." arXiv:2511.16719.

## Roboflow Supervision

- **Licencia**: MIT
- **Repo**: https://github.com/roboflow/supervision

## ByteTrack

- **Licencia**: MIT
- **Repo**: https://github.com/ifzhang/ByteTrack

## OpenCV

- **Licencia**: Apache 2.0
- **Repo**: https://github.com/opencv/opencv

## PyTorch

- **Licencia**: BSD-3-Clause
- **Repo**: https://github.com/pytorch/pytorch

---

Si se agregan más dependencias con licencias no MIT/Apache/BSD, documentarlas
explícitamente aquí antes del 19 de junio de 2026.
