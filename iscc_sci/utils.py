from pathlib import Path

import numpy as np
from loguru import logger as log
import time
from contextlib import contextmanager
from urllib.request import urlretrieve
from blake3 import blake3
from numpy._typing import NDArray

import iscc_sci as sci


__all__ = [
    "metrics",
    "get_model",
    "compress",
]


BASE_VERSION = "1.0.0"
BASE_URL = f"https://github.com/iscc/iscc-binaries/releases/download/v{BASE_VERSION}"
MODEL_FILENAME = "iscc-sci-v0.1.0.onnx"
MODEL_URL = f"{BASE_URL}/{MODEL_FILENAME}"
MODEL_PATH = Path(sci.dirs.user_data_dir) / MODEL_FILENAME
MODEL_CHECKSUM = "af95054d463e4c95de4c099a7947dbc2f3db168507fef25e91e6984a6f32dd3c"


@contextmanager
def metrics(name):
    """Context manager for logging performance metrics."""
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    log.debug(name.format(seconds=duration))


def get_model():  # pragma: no cover
    """Check and return local model file if it exists, otherwise download."""
    if MODEL_PATH.exists():
        try:
            return check_integrity(MODEL_PATH, MODEL_CHECKSUM)
        except RuntimeError:
            log.warning("Model file integrity error - redownloading...")
    urlretrieve(MODEL_URL, filename=MODEL_PATH)
    return check_integrity(MODEL_PATH, MODEL_CHECKSUM)


def check_integrity(file_path, checksum):
    # type: (str|Path, str) -> Path
    """
    Check file integrity against blake3 checksum

    :param file_path: path to file to be checked
    :param checksum: blake3 checksum to verify integrity
    :raises RuntimeError: if verification fails
    """
    file_path = Path(file_path)
    file_hasher = blake3(max_threads=blake3.AUTO)
    with metrics("Integrity check took {seconds:.4f} seconds"):
        file_hasher.update_mmap(file_path)
        file_hash = file_hasher.hexdigest()
    if checksum != file_hash:
        msg = f"Failed integrity check for {file_path.name}"
        log.error(msg)
        raise RuntimeError(msg)
    return file_path


def compress(vec, precision):
    # type: (NDArray, int) -> list[float]
    """
    Round down vector values to specified precision to reduce storage requirements.

    :param vec: Embedding vector.
    :param precision: Max number of fractional decimal places.
    :return: Vector as a native python list of rounded floats.
    """
    rounded_array = np.around(vec, decimals=precision)
    compress_list = [round(x, precision) for x in rounded_array.tolist()]
    return compress_list
