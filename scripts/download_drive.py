"""Descarga el folder oficial del Drive de la convocatoria.

Usa truststore para sobrevivir la inspección SSL de Norton (mismo problema
que tuvimos con huggingface_hub). Idempotente: gdown salta archivos ya
descargados.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.network import enable_system_ssl

enable_system_ssl()

import gdown  # noqa: E402

FOLDER_URL = "https://drive.google.com/drive/folders/1TF7-P4rAwPmHFw_TjmNfFU3ORxqnp8CD"
OUT_DIR = ROOT / "data" / "raw" / "drive_oficial"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {FOLDER_URL}")
    print(f"  destino: {OUT_DIR}")
    files = gdown.download_folder(
        FOLDER_URL,
        output=str(OUT_DIR),
        quiet=False,
        remaining_ok=True,
        use_cookies=False,
    )
    print(f"\nDescargados {len(files) if files else 0} archivos en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
