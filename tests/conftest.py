import pytest
from iscc_sci.code_semantic_image import preprocess_image
from PIL import Image
from iscc_samples import images


@pytest.fixture
def img_path():
    return images()[0]


@pytest.fixture
def img_obj(img_path):
    return Image.open(img_path)


@pytest.fixture
def img_array(img_obj):
    arr = preprocess_image(img_obj)
    return arr
