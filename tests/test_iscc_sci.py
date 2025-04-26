import numpy as np
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


def test_models():
    from iscc_sci.code_semantic_image import model

    engine = model()
    assert engine


def test_preprocess_image(img_obj):
    result = sci.preprocess_image(img_obj)
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3, 512, 512)
