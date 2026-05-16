"""Catálogo de prompts SAM 3.1 para fútbol robótico FutBotMX.

Importante (pitfall de literatura, ver docs/literature-review.md § 2):
el texto del prompt DEBE ser idéntico entre fine-tuning e inferencia
para evitar prompt drift. Importa estas constantes en ambos contextos.
"""

from __future__ import annotations

from typing import Final

# Categorías canónicas (ver convocatoria § 3.5.1)
FIELD: Final[str] = "green felt soccer field with white lines"
BALL: Final[str] = "small bright orange golf ball"
ROBOT_TEAM_A: Final[str] = "small mobile robot with purple flag on top"
ROBOT_TEAM_B: Final[str] = "small mobile robot with white flag on top"

ALL_PROMPTS: Final[dict[str, str]] = {
    "field": FIELD,
    "ball": BALL,
    "team_a": ROBOT_TEAM_A,
    "team_b": ROBOT_TEAM_B,
}

# Prompts compuestos para casos avanzados (cascada de detección)
PROMPT_ALL_ROBOTS: Final[str] = "small mobile soccer robot with a colored flag"
PROMPT_GOAL_YELLOW: Final[str] = "yellow rectangular goal wall"
PROMPT_GOAL_BLUE: Final[str] = "blue rectangular goal wall"
