from loguru import logger as log
from base64 import b32encode
from pathlib import Path
from typing import List, Tuple
from PIL import Image
import numpy as np
import onnxruntime as rt
from numpy.typing import NDArray
import iscc_sci as sci


__all__ = [
    "code_image_semantic",
    "gen_image_code_semantic",
    "soft_hash_image_semantic",
    "preprocess_image",
]

BIT_LEN_MAP = {
    32: "0000",
    64: "0001",
    96: "0010",
    128: "0011",
    160: "0100",
    192: "0101",
    224: "0110",
    256: "0111",
}

# Lazy loaded ONNX model
_model = None


def code_image_semantic(fp, bits=64):
    # type: (str|Path, int) -> dict
    """
    Generate ISCC Semantic-Code Image from image file.

    :param str|Path fp: Image filepath used for Semantic-Code creation.
    :param int bits: Bit-length of ISCC Semantic-Code Image (default 64, max 256).
    :return: ISCC metadata - `{"iscc": ..., "features": ...}`
    :rtype: dict
    """
    image = Image.open(fp)
    arr = preprocess_image(image)
    iscc_meta = gen_image_code_semantic(arr, bits=bits)
    return iscc_meta


def gen_image_code_semantic(arr, bits=64):
    # type: (NDArray[np.float32], int) -> dict
    """
    Create an ISCC Semantic-Code Image from normalized image array.

    :param NDArray[np.float32] arr: Normalized image array with shape (1, 3, 224, 224)
    :param int bits: Bit-length of ISCC Semantic-Code Image (default 64, max 256).
    :return: ISCC Schema compatible dict with Semantic-Code Image.
    :rtype: dict
    """
    if arr.shape != (1, 3, 512, 512):
        raise ValueError("Array must have shape (1, 3, 512, 512)")

    if bits < 32 or bits % 32:
        raise ValueError(f"Invalid bitlength {bits}")

    mtype = "0001"  # SEMANTIC
    stype = "0001"  # IMAGE
    version = "0000"  # V0
    length = BIT_LEN_MAP[bits]

    header = int(mtype + stype + version + length, 2).to_bytes(2, byteorder="big")
    digest, features = soft_hash_image_semantic(arr, bits=bits)
    code = b32encode(header + digest).decode("ascii").rstrip("=")

    iscc = "ISCC:" + code
    return {"iscc": iscc, "features": features.tolist()}


def soft_hash_image_semantic(arr, bits=64):
    # type: (NDArray[np.float32], int) -> Tuple[bytes, NDArray[np.float32]]
    """
    Calculate semantic image hash from preprocessed image array.

    :param NDArray[np.float32] arr: Preprocessed image array
    :param int bits: Bit-length of semantic image hash (default 64).
    :return: Tuple of image-hash digest and semantic feature vector from model.
    """
    embeddings = vectorize(arr)
    features = embeddings[0][0]
    digest = binarize(features)
    digest = digest[: bits // 8]
    return digest, features


def model():
    # type: () -> rt.InferenceSession
    """Initialize, cache and return inference model"""
    global _model
    if _model is None:
        model_path = sci.get_model()
        log.info(f"Initializing ONNX model for iscc-sci {sci.__version__}")
        with sci.metrics(name="ONNX load time {seconds:.2f} seconds"):
            _model = rt.InferenceSession(model_path)
    return _model


def preprocess_image(image):
    # type: (Image.Image) -> NDArray[np.float32]
    """Preprocess image for inference."""
    with sci.metrics(name="Image preprocessing time {seconds:.4f} seconds"):
        # Resize the image
        image = image.resize((512, 512), Image.BILINEAR)
        # Convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array and ensure type consistency
        image = np.array(image, dtype=np.float32)

        # Rescale and Normalize
        image /= 255.0  # Rescale
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        image = (image - mean) / std

        # Change the order of dimensions to CHW and add batch dimension
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)

    return image.astype(np.float32)


def vectorize(arr):
    # type: (NDArray) -> List[NDArray[np.float32]]
    """Apply inference on preprocessed image data"""
    engine = model()
    with sci.metrics(name="Image inference time {seconds:.4f} seconds"):
        input_name = engine.get_inputs()[0].name
        embeddings = engine.run(None, {input_name: arr})
    return embeddings


def binarize(vec):
    # type: (NDArray) -> bytes
    """Binarize vector embeddings."""

    bits = [1 if num >= 0 else 0 for num in vec]

    # Prepare a bytearray for the result
    result = bytearray()

    # Process each 8 bits (or the remaining in the last iteration)
    for i in range(0, len(bits), 8):
        # Convert 8 bits into a byte
        byte = 0
        for bit in bits[i : i + 8]:
            byte = (byte << 1) | bit
        result.append(byte)
    return bytes(result)
