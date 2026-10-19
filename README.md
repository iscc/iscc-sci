# ISCC - Semantic Image-Code

[![Tests](https://github.com/iscc/iscc-sci/actions/workflows/tests.yml/badge.svg)](https://github.com/iscc/iscc-sci/actions/workflows/tests.yml)
[![Version](https://img.shields.io/pypi/v/iscc-sci.svg)](https://pypi.python.org/pypi/iscc-sci/)
[![Downloads](https://pepy.tech/badge/iscc-sci)](https://pepy.tech/project/iscc-sci)

`iscc-sci` is a **proof of concept implementation** of a semantic Image-Code for the
[ISCC](https://core.iscc.codes) (*International Standard Content Code*). Semantic Image-Codes are
designed to capture and represent the semantic content of images for improved similarity detection.

> [!CAUTION]
> **This is a proof of concept.** All releases with version numbers below v1.0.0 may break backward
> compatibility and produce incompatible Semantic Image-Codes. The algorithms of this `iscc-sci`
> repository are experimental and not part of the official
> [ISO 24138:2024](https://www.iso.org/standard/77899.html) standard.

## What is ISCC Semantic Image-Code

The ISCC framework already comes with an Image-Code that is based on perceptual hashing and can
match near duplicates. The ISCC Semantic Image-Code is planned as a new additional ISCC-UNIT focused
on capturing a more abstract and broad semantic similarity. As such the Semantic Image-Code is
engineered to be robust against a broader range of variations that cannot be matched with the
perceptual Image-Code.

## Features

- **Semantic Similarity**: Leverages deep learning models to generate codes that reflect the
  semantic content of images.
- **Bit-Length Flexibility**: Supports generating codes of various bit lengths (up to 256 bits),
  allowing for adjustable granularity in similarity detection.
- **ISCC Compatible**: Generates codes that are fully compatible with the ISCC specification,
  facilitating integration with existing ISCC-based systems.

## Installation

Ensure you have Python 3.11 or newer installed on your system. The package requires an ONNX runtime
that is selected via install extras. For CPU inference (works everywhere):

```bash
pip install "iscc-sci[cpu]"
```

For NVIDIA CUDA accelerated inference (requires CUDA 12.x and cuDNN 9.x):

```bash
pip install "iscc-sci[gpu]"
```

> [!NOTE]
> Install exactly **one** of the `cpu`/`gpu` extras. The underlying `onnxruntime` and
> `onnxruntime-gpu` packages unpack into the same directory and overwrite each other, so installing
> both silently disables GPU support. A plain `pip install iscc-sci` installs no ONNX runtime and
> fails on import with instructions.

## Usage

To generate a Semantic Image-Code for an image, use the `code_image_semantic` function. You can
specify the bit length of the code to control the level of granularity in the semantic
representation.

```python
import iscc_sci as sci

# Generate a 64-bit ISCC Semantic Image-Code for an image file
image_file_path = "path/to/your/image.jpg"
semantic_code = sci.code_image_semantic(image_file_path, bits=64)

print(semantic_code)
```

## How It Works

`iscc-sci` uses a pre-trained deep learning model based on the 1st Place Solution of the Image
Similarity Challenge (ISC21) to create semantic embeddings of images. The model generates a feature
vector that captures the essential characteristics of the image. This vector is then binarized to
produce a Semantic Image-Code that is robust to variations in image presentation but sensitive to
content differences.

## Model

The model file is downloaded on first use from the
[iscc-binaries v1.0.0](https://github.com/iscc/iscc-binaries/releases/tag/v1.0.0) release and
verified against its BLAKE3 checksum:

- **File:** `iscc-sci-v0.1.0-w16.onnx`, 104,790,555 bytes
- **BLAKE3:** `9f24dd3d0440eaa07adf121b21a1cbd8518fe36048b9d8a72b606d8a6416e8a3`
- **Input:** 1×3×512×512 RGB, normalized as `(x / 255 - 0.5) / 0.5`
- **Output:** 256 floats. The Semantic-Code is made of their sign bits.

**Provenance:**

1. **Backbone:** EfficientNetV2-M, trained by Google on ImageNet-21k and fine-tuned on ImageNet-1k
   ([timm/tf_efficientnetv2_m.in21k_ft_in1k](https://huggingface.co/timm/tf_efficientnetv2_m.in21k_ft_in1k),
   Apache-2.0).
1. **Training:** fine-tuned on the 1M DISC21 training images by Shuhei Yokoo (lyakaap) for the
   Facebook AI Image Similarity Challenge 2021, descriptor track, 1st place
   ([ISC21-Descriptor-Track-1st](https://github.com/lyakaap/ISC21-Descriptor-Track-1st), report
   [arXiv 2112.04323](https://arxiv.org/abs/2112.04323)). The weights are MIT licensed, see
   [LICENSE-MODEL](LICENSE-MODEL).
1. **Export:** converted to ONNX with the `image-embedding/isc` operator of
   [Towhee](https://towhee.io/image-embedding/isc) (Apache-2.0).
1. **Compression:** weights stored as fp16 and cast back to fp32 when the model loads (all maths
   stays fp32), graph simplified with onnxslim, input fixed to 1×3×512×512. This is the same file
   the ISCC C2PA demo uses.

**Licence:** the `iscc-sci` code is Apache-2.0 and the model weights are MIT. The DISC21 training
data is governed by Meta's Image Similarity Dataset License Agreement in the
[competition rules](https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/rules/).
Its clause 2 covers models trained on the data: "Such algorithms, software, machine learning models,
techniques and technologies may be used for academic and commercial purposes." Commercial use is
allowed.

The "CC BY-NC 4.0" licence that is sometimes cited for the model belongs to Meta's evaluation code
repository [facebookresearch/isc2021](https://github.com/facebookresearch/isc2021), not to the
dataset.

## Development

This is a proof of concept and welcomes contributions to enhance its capabilities, efficiency, and
compatibility with the broader ISCC ecosystem. For development, install the project with
[uv](https://docs.astral.sh/uv). The default `uv sync` installs the test group, which provides a CPU
ONNX runtime:

```shell
git clone https://github.com/iscc/iscc-sci.git
cd iscc-sci
uv sync
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an
issue or pull request. For major changes, please open an issue first to discuss what you would like
to change.
