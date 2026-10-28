from PIL import Image
import io
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def read_image_to_numpy(file_bytes: bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    arr = np.array(img)
    return img, arr


def brightness_mean_u8(arr: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not installed")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(gray.mean() / 255.0)


def variance_of_laplacian(arr: np.ndarray) -> float:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) not installed")
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())
