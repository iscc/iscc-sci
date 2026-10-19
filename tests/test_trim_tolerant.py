"""Tests for the repeated tolerant border trim and the `image_trim_tolerant` option."""

import numpy as np
import pytest
from iscc_samples import images
from PIL import Image, ImageDraw, ImageOps
import iscc_sci as sci
from iscc_sci.code_semantic_image import (
    TRIM_MAX_PASSES,
    remove_transparency,
    trim_border,
    trim_border_tolerant,
)

WHITE = (255, 255, 255)
SUBJECT = (60, 90, 120)
DETAIL = (150, 170, 190)


def oracle_pass(img):
    """Single tolerant trim pass computed with numpy (independent reference implementation)."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    keep = (np.abs(a - a[0, 0]) > 16).any(axis=2)
    if not keep.any():
        return img
    rows, cols = np.flatnonzero(keep.any(axis=1)), np.flatnonzero(keep.any(axis=0))
    return img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))


def oracle(img):
    """Reference tolerant trim: single numpy pass repeated until stable or 16 passes."""
    for _ in range(16):
        out = oracle_pass(img)
        if out.size == img.size:
            return out
        img = out
    return img


def pixels(img):
    """Pixel data of an image as numpy array for exact comparison."""
    return np.asarray(img)


def noisy_product(seed=0, size=(300, 200)):
    """Product photo: noisy white background (sigma 2) around a darker textured object."""
    w, h = size
    rng = np.random.default_rng(seed)
    a = 255.0 + rng.normal(0, 2, (h, w, 3))
    oh, ow = h // 2, w // 2
    texture = 90 + rng.normal(0, 20, (oh, ow, 3))
    a[h // 4 : h // 4 + oh, w // 4 : w // 4 + ow] = texture
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def pad(img, border, color):
    """Add a solid border of `border` pixels around `img`."""
    return ImageOps.expand(img, border=border, fill=color)


def reviewer_image():
    """White 200x150 canvas with an 80x60 subject containing a lighter 55x35 detail."""
    img = Image.new("RGB", (200, 150), WHITE)
    img.paste(SUBJECT, (60, 45, 140, 105))
    img.paste(DETAIL, (70, 55, 125, 90))
    return img


def ring_image(rings=20, width=2):
    """Nested rings of `width` pixels alternating between colors 100 apart around a center."""
    size = 2 * rings * width + 10
    img = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(img)
    for i in range(rings):
        gray = 0 if i % 2 == 0 else 100
        o = i * width
        draw.rectangle((o, o, size - 1 - o, size - 1 - o), fill=(gray, gray, gray))
    draw.rectangle(
        (rings * width, rings * width, size - 1 - rings * width, size - 1 - rings * width),
        fill=(200, 0, 0),
    )
    return img


def sample_rgb(fp):
    """Load a sample image as RGB the same way `preprocess_image` does before trimming."""
    return remove_transparency(ImageOps.exif_transpose(Image.open(fp)))


def pattern_file(tmp_path):
    """Sample image pasted on a near-white patterned canvas the exact trim can not remove."""
    y, x = np.mgrid[0:400, 0:600]
    value = (255 - (x * 7 + y * 13) % 5).astype(np.uint8)
    canvas = Image.fromarray(np.stack([value] * 3, axis=2), "RGB")
    canvas.paste(sample_rgb(images()[2]).resize((300, 200)), (150, 100))
    fp = tmp_path / "pattern.png"
    canvas.save(fp)
    return fp


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_trim_tolerant_noisy_white_matches_oracle(seed):
    img = noisy_product(seed)
    result = trim_border_tolerant(img)
    assert result.size == (150, 100)
    assert np.array_equal(pixels(result), pixels(oracle(img)))


def test_trim_tolerant_noisy_white_exact_trim_fails():
    img = noisy_product()
    assert trim_border(img).size[0] > 290  # noise defeats the exact trim


@pytest.mark.parametrize("color", [(200, 30, 30), (0, 0, 0), (30, 160, 60)])
def test_trim_tolerant_colored_padding(color):
    img = noisy_product()
    padded = pad(img, 30, color)
    result = trim_border_tolerant(padded)
    assert np.array_equal(pixels(result), pixels(oracle(padded)))
    assert np.array_equal(pixels(result), pixels(trim_border_tolerant(img)))


@pytest.mark.parametrize("color", [WHITE, (0, 0, 0), (12, 200, 99)])
def test_trim_tolerant_uniform_image(color):
    img = Image.new("RGB", (64, 48), color)
    result = trim_border_tolerant(img)
    assert result is img
    assert np.array_equal(pixels(result), pixels(oracle(img)))


@pytest.mark.parametrize("fp", images(), ids=lambda fp: fp.name)
def test_trim_tolerant_sample_images_match_oracle(fp):
    img = sample_rgb(fp)
    assert np.array_equal(pixels(trim_border_tolerant(img)), pixels(oracle(img)))


def test_trim_tolerant_reviewer_case_variants():
    plain = reviewer_image()
    patches = reviewer_image()
    patches.paste((254, 254, 254), (5, 5, 50, 40))
    patches.paste((253, 253, 253), (150, 110, 195, 145))
    framed = pad(reviewer_image(), 20, (0, 0, 0))
    expected = Image.new("RGB", (55, 35), DETAIL)
    for img in (plain, patches, framed):
        result = trim_border_tolerant(img)
        assert result.size == (55, 35)
        assert np.array_equal(pixels(result), pixels(expected))
        assert np.array_equal(pixels(result), pixels(oracle(img)))
    arrays = [sci.preprocess_image(img) for img in (plain, patches, framed)]
    assert np.array_equal(arrays[0], arrays[1])
    assert np.array_equal(arrays[0], arrays[2])


def test_trim_tolerant_noisy_dark_frame():
    img = noisy_product()
    rng = np.random.default_rng(7)
    frame = rng.integers(6, 11, (img.height + 40, img.width + 40, 3), dtype=np.uint8)
    framed = Image.fromarray(frame, "RGB")
    framed.paste(img, (20, 20))
    result = trim_border_tolerant(framed)
    assert np.array_equal(pixels(result), pixels(trim_border_tolerant(img)))
    assert np.array_equal(pixels(result), pixels(oracle(framed)))


@pytest.mark.parametrize("img", [noisy_product(), pad(reviewer_image(), 20, (0, 0, 0))])
def test_trim_tolerant_idempotent(img):
    once = trim_border_tolerant(img)
    twice = trim_border_tolerant(once)
    assert twice is once


def test_trim_tolerant_pass_cap():
    img = ring_image(rings=20, width=2)
    result = trim_border_tolerant(img)
    assert TRIM_MAX_PASSES == 16
    assert result.size == (img.width - 2 * 2 * 16, img.height - 2 * 2 * 16)
    assert np.array_equal(pixels(result), pixels(oracle(img)))
    # The cap stopped trimming, another call continues with the remaining rings
    assert trim_border_tolerant(result).size < result.size


@pytest.mark.parametrize("mode", ["L", "RGBA", "P", "CMYK"])
def test_trim_tolerant_other_modes(mode):
    rgb = pad(reviewer_image(), 20, (0, 0, 0))
    img = rgb.convert(mode)
    result = trim_border_tolerant(img)
    assert result.mode == mode
    assert result.size == oracle(img).size


def test_option_default():
    assert sci.SciOptions().image_trim_tolerant is True
    assert sci.sci_opts.image_trim_tolerant is True


def test_option_env_var(monkeypatch):
    monkeypatch.setenv("ISCC_SCI_IMAGE_TRIM_TOLERANT", "false")
    assert sci.SciOptions().image_trim_tolerant is False


def test_preprocess_image_option_off_uses_exact_trim():
    img = pad(noisy_product(), 20, (0, 0, 0))
    exact = sci.preprocess_image(img, image_trim_tolerant=False)
    tolerant = sci.preprocess_image(img)
    expected = sci.preprocess_image(trim_border(img), image_trim_tolerant=False)
    assert np.array_equal(exact, expected)
    assert not np.array_equal(exact, tolerant)
    assert np.array_equal(tolerant, sci.preprocess_image(trim_border_tolerant(img)))


def test_preprocess_image_global_option(monkeypatch):
    img = pad(noisy_product(), 20, (0, 0, 0))
    monkeypatch.setattr(sci.sci_opts, "image_trim_tolerant", False)
    assert np.array_equal(sci.preprocess_image(img), sci.preprocess_image(trim_border(img)))


def test_code_image_semantic_per_call_option(tmp_path, monkeypatch):
    fp = pattern_file(tmp_path)
    tolerant = sci.code_image_semantic(fp)["iscc"]
    exact = sci.code_image_semantic(fp, image_trim_tolerant=False)["iscc"]
    assert sci.sci_opts.image_trim_tolerant is True  # per-call option leaves global options as is
    assert exact != tolerant
    monkeypatch.setattr(sci.sci_opts, "image_trim_tolerant", False)
    assert sci.code_image_semantic(fp)["iscc"] == exact
