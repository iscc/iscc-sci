import numpy as np
import pytest
import iscc_sci as sci
from iscc_samples import images


def test_version():
    assert sci.__version__ == "0.2.0"


def test_code_image_semantic_default():
    result = sci.code_image_semantic(images()[0])
    assert result["iscc"] == "ISCC:CEAQ2WTPK2QPZTK4"


def test_code_image_semantic_256bit():
    result = sci.code_image_semantic(images()[1], bits=256)
    assert result["iscc"] == "ISCC:CEDQ2WT7K2Q7YTO47HLGYUURO2RCI24K5VUZOHFMMY42C6O6VLQ6FEA"


def test_gen_image_code_semantic(img_array):
    result = sci.gen_image_code_semantic(img_array)
    assert result["iscc"] == "ISCC:CEAQ2WTPK2QPZTK4"


def test_gen_image_code_semantic_invalid_shape():
    # Create an array with invalid shape
    invalid_array = np.zeros((1, 3, 256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="Array must have shape"):
        sci.gen_image_code_semantic(invalid_array)


def test_gen_image_code_semantic_invalid_bits():
    # Test with invalid bit length (not multiple of 32)
    with pytest.raises(ValueError, match="Invalid bitlength"):
        sci.gen_image_code_semantic(np.zeros((1, 3, 512, 512), dtype=np.float32), bits=33)

    # Test with invalid bit length (less than 32)
    with pytest.raises(ValueError, match="Invalid bitlength"):
        sci.gen_image_code_semantic(np.zeros((1, 3, 512, 512), dtype=np.float32), bits=16)


def test_models():
    from iscc_sci.code_semantic_image import model

    engine = model()
    assert engine


def test_preprocess_image(img_obj):
    result = sci.preprocess_image(img_obj)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3, 512, 512)
