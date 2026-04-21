import numpy as np


def analyze_y(y_bytes: bytes, width: int, height: int, row_stride: int) -> str:
    arr = np.frombuffer(y_bytes, dtype=np.uint8)
    if row_stride == width:
        y = arr.reshape(height, width)
    else:
        y = arr.reshape(height, row_stride)[:, :width]
    mean = float(y.mean())
    std = float(y.std())
    return f"{width}x{height}  mean={mean:.1f}  std={std:.1f}"
