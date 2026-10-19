# Changelog

## [0.4.0] - Unreleased

- Added a tolerant border trim to image preprocessing, on by default (`image_trim_tolerant`, env
  `ISCC_SCI_IMAGE_TRIM_TOLERANT`). It crops near-uniform borders (per-channel difference to the
  top-left pixel of at most 16) instead of only exactly uniform ones and repeats until a pass crops
  nothing (at most 16 passes). It is identical to the trim of the Image-Code in iscc-sdk 0.9.7, so
  both codes of an image see the same content.
- `preprocess_image` accepts per-call options, and `code_image_semantic` passes its options on to it
- **Compatibility:** Semantic-Codes change for images with near-uniform borders (typical for product
  photos on white backgrounds). Use `image_trim_tolerant=False` to get the earlier codes.

## [0.3.0] - 2026-06-14

- Renamed CLI command from `sci` to `iscc-sci`
- Migrated build system from poetry to uv
- Added Python 3.14 support, dropped Python 3.9 and 3.10
- Rewrote CLI tests for 100% code coverage
- Aligned `preprocess_image` preflight order with IEP-0004 (remove transparency before trimming
  borders)
- Enabled GPU inference: select the ONNX runtime via exclusive `cpu`/`gpu` extras and request the
  CUDA execution provider in `model()` (`pip install iscc-sci` no longer bundles a runtime — install
  `iscc-sci[cpu]` or `iscc-sci[gpu]`)

## [0.2.0] - 2025-04-28

- Added support for configurable bit-length (32 to 256 bits)
- Improved image preprocessing with better handling of transparency and uniform borders
- Added embedding vector output option for advanced similarity analysis
- Enhanced CLI with debug mode and embedding flag options
- Added vector compression functionality to reduce storage requirements
- Updated dependencies

## [0.1.0] - 2024-02-07

- Initial Proof Of Concept
