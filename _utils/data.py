from dataclasses import dataclass
from typing import Any


@dataclass
class AdvAttackData:
    stat: list
    in_indices: list
    losses: list
    n: Any
    sample_weight: Any
