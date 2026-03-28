"""Common functions for randomness."""
import random

import numpy as np
import torch


def set_random_seed(seed) -> None:
    """Set random seed in python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
