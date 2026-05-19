"""Lista módulos Linear del modelo SAM 3.1 para identificar targets de LoRA.

Corre en CPU para no interferir con jobs de inferencia en GPU.
Categoriza módulos en: image_encoder, mask_decoder, prompt_encoder, otros.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()

import torch  # noqa: E402
from transformers import Sam3Model  # noqa: E402


def main() -> int:
    print("Cargando SAM 3.1 en CPU (solo para inspección)...")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32)
    print(f"Tipo del modelo: {type(model).__name__}\n")

    # Agrupar por (subcomponente, sufijo de capa Linear)
    by_component: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_linear = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            total_linear += 1
            parts = name.split(".")
            component = parts[0] if parts else "root"
            suffix = parts[-1]
            by_component[component][suffix] += 1

    print(f"Total módulos Linear: {total_linear}\n")
    for comp, suffixes in sorted(by_component.items()):
        n = sum(suffixes.values())
        print(f"\n=== {comp}  ({n} Linear) ===")
        for suf, cnt in sorted(suffixes.items(), key=lambda x: -x[1]):
            print(f"  {suf:30s} x{cnt}")

    # Listar también nombres top-level
    print("\n\n=== Top-level submódulos ===")
    for name, _ in model.named_children():
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
