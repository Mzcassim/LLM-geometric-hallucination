"""Seed management for reproducibility."""

import random
import numpy as np


def set_seed(seed: int) -> None:
    """
    Set random seed for Python, NumPy for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
