"""JSON serialization helpers shared by vision nodes."""

import numpy as np


def json_clean(value):
    if isinstance(value, dict):
        return {
            str(key): json_clean(val)
            for key, val in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, (list, tuple)):
        return [json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_clean(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value

