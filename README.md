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

Before you can install `iscc-sci`, you need to have Python 3.8 or newer installed on your system.
Install the library as any other python package:

```bash
pip install iscc-sci
```

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

## Development

This is a proof of concept and welcomes contributions to enhance its capabilities, efficiency, and
compatibility with the broader ISCC ecosystem. For development, you'll need to install the project
in development mode using [Poetry](https://python-poetry.org).

```shell
git clone https://github.com/iscc/iscc-sci.git
cd iscc-sci
poetry install
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please open an
issue or pull request. For major changes, please open an issue first to discuss what you would like
to change.
