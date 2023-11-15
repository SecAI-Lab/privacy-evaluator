from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class AdvAttackData:
    stat: list
    in_indices: list
    losses: list
    n: Any
    sample_weight: Any


@dataclass
class TData:
    train_data: Optional[Any] = None
    train_labels: Optional[Any] = None
    test_data: Optional[Any] = None
    test_labels: Optional[Any] = None
    x_concat: Optional[np.ndarray] = None
    y_concat: Optional[np.ndarray] = None
