from pathlib import Path
from uuid import uuid4
import cv2
import numpy as np

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_rgb(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise OSError(f"Cannot decode {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_png(path, image):
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"Cannot encode {path}")
    path = Path(path)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        encoded.tofile(temp)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def image_files(directory):
    entries = {}
    for path in sorted(Path(directory).iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in entries:
                raise ValueError(f"Duplicate image ID {path.stem} in {directory}")
            entries[path.stem] = path
    if not entries:
        raise ValueError(f"No images in {directory}")
    return entries
