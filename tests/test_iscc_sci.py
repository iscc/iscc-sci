import numpy as np
import pytest
import iscc_sci as sci
from iscc_samples import images
from PIL import Image, ImageChops
from iscc_sci.code_semantic_image import remove_transparency, trim_border


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


def test_remove_transparency_rgba():
    """Test remove_transparency with RGBA mode image."""
    # Create a test image with RGBA mode
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))  # Semi-transparent red

    # Apply the function
    result = remove_transparency(img)

    # Check the result
    assert result.mode == "RGB"
    assert result.getpixel((50, 50))[0] > 127  # Red channel should be visible but mixed with white


def test_remove_transparency_la():
    """Test remove_transparency with LA mode image."""
    # Create a test image with LA mode (grayscale with alpha)
    img = Image.new("LA", (100, 100), (100, 128))  # Semi-transparent gray

    # Apply the function
    result = remove_transparency(img)

    # Check the result
    assert result.mode == "RGB"
    assert result.getpixel((50, 50))[0] > 100  # Gray value should be visible but mixed with white


def test_remove_transparency_p_with_transparency():
    """Test remove_transparency with P mode image that has transparency."""
    # Create a P mode image with transparency
    img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))  # Semi-transparent red
    img = img.convert("P")
    img.info["transparency"] = 0  # Add transparency info

    # Apply the function
    result = remove_transparency(img)

    # Check the result
    assert result.mode == "RGB"


def test_remove_transparency_p_without_transparency():
    """Test remove_transparency with P mode image without transparency."""
    # Create a P mode image without transparency
    img = Image.new("P", (100, 100), 0)  # Black image

    # Apply the function
    result = remove_transparency(img)

    # Check the result
    assert result.mode == "RGB"
    assert result.getpixel((50, 50)) == (0, 0, 0)  # Should remain black


def test_remove_transparency_rgb():
    """Test remove_transparency with RGB mode image."""
    # Create a test image with RGB mode
    img = Image.new("RGB", (100, 100), (255, 0, 0))  # Red

    # Apply the function
    result = remove_transparency(img)

    # Check the result
    assert result.mode == "RGB"
    assert result.getpixel((50, 50)) == (255, 0, 0)  # Should remain red


def test_trim_border_with_uniform_border():
    """Test trim_border with an image that has a uniform border."""
    # Create a test image with a uniform border
    img = Image.new("RGB", (100, 100), (255, 255, 255))  # White background
    # Draw a red rectangle in the center (leaving a white border)
    for x in range(25, 75):
        for y in range(25, 75):
            img.putpixel((x, y), (255, 0, 0))  # Red

    # Apply the function
    result = trim_border(img)

    # Check the result
    assert result.size == (50, 50)  # Should be trimmed to 50x50
    assert result.getpixel((0, 0)) == (255, 0, 0)  # Top-left should be red
    assert result.getpixel((49, 49)) == (255, 0, 0)  # Bottom-right should be red


def test_trim_border_without_uniform_border():
    """Test trim_border with an image that has no uniform border."""
    # Create a test image with no uniform border
    img = Image.new("RGB", (100, 100), (255, 0, 0))  # Red background
    # Make the corners different colors
    img.putpixel((0, 0), (0, 255, 0))  # Green
    img.putpixel((99, 0), (0, 0, 255))  # Blue
    img.putpixel((0, 99), (255, 255, 0))  # Yellow
    img.putpixel((99, 99), (0, 255, 255))  # Cyan

    # Apply the function
    result = trim_border(img)

    # Check the result
    assert result.size == (100, 100)  # Size should remain unchanged
    assert result.getpixel((0, 0)) == (0, 255, 0)  # Top-left should still be green
    assert result.getpixel((99, 99)) == (0, 255, 255)  # Bottom-right should still be cyan


def test_trim_border_solid_color():
    """Test trim_border with an image that is entirely one color.

    This tests the edge case where getbbox() returns None because the image
    is a solid color with no differences from the background.
    """
    # Create a solid color image
    img = Image.new("RGB", (100, 100), (255, 0, 0))  # Solid red

    # Create a background with the same color as the top-left pixel
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    # Verify that the difference is all zeros (which will cause getbbox() to return None)
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff)
    assert diff.getbbox() is None  # Confirm that getbbox() returns None for a solid color

    # Apply the function
    result = trim_border(img)

    # Check the result
    assert result.size == (100, 100)  # Size should remain unchanged
    assert result.mode == img.mode  # Mode should be the same
    assert result.getpixel((50, 50)) == (255, 0, 0)  # Should still be red
    # For a solid color image, the function still returns a new image object
    # but with the same properties as the original
