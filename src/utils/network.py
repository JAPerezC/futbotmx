"""Configuración de red: parche SSL para entornos con interceptación.

Si está corriendo bajo un antivirus o proxy que hace SSL inspection
(Norton, ESET, Kaspersky, Cisco AnyConnect, ZScaler, etc.), el cliente
Python normalmente falla con CERTIFICATE_VERIFY_FAILED porque el CA root
del interceptor no está en `certifi`. La solución estándar es delegar
la verificación al cert store del sistema operativo via `truststore`.

Llamar `enable_system_ssl()` UNA vez al inicio del proceso, antes de
cualquier import o request HTTPS (transformers, huggingface_hub, etc.).
Es idempotente: llamarla varias veces no causa problemas.
"""

from __future__ import annotations

_INJECTED = False


def enable_system_ssl() -> None:
    """Hace que Python verifique TLS contra el cert store del SO.

    Necesario en máquinas con SSL inspection corporativo o por antivirus.
    Idempotente.
    """
    global _INJECTED
    if _INJECTED:
        return
    import truststore  # type: ignore

    truststore.inject_into_ssl()
    _INJECTED = True
