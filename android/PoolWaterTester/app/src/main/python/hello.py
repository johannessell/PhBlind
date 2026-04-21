"""Smoke test that the Chaquopy + numpy + OpenCV toolchain is healthy."""
import sys


def greet() -> str:
    return f"Hello from Python {sys.version.split()[0]}"


def cv_check() -> str:
    import cv2
    import numpy as np
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (9, 9), (255, 0, 0), 1)
    return f"cv2={cv2.__version__} numpy={np.__version__} mean={float(img.mean()):.3f}"
