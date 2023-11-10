from dataclasses import dataclass
from typing import Any
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
    train_data: Any
    train_labels: Any
    test_data: Any
    test_labels: Any
    x_concat: np.array
    y_concat: np.array
