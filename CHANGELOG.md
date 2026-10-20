# Changelog

## [0.4.1] - Unreleased

- The tolerant border trim keeps smooth gradients: a pass that cuts only one side must remove a flat
  band (no pixel differs from the reference by more than half the tolerance), otherwise trimming
  stops. The 0.4.0 trim walked down gradients such as skies about 16 levels per pass and could crop
  a landscape to its lower half. Crops are identical to iscc-sdk 0.9.7.
- **Compatibility:** Semantic-Codes change for images with one-sided gradient borders (iscc-sdk
  Image-Codes change the same way)

## [0.4.0] - 2026-10-19

- Added a tolerant border trim to image preprocessing, on by default (`image_trim_tolerant`, env
  `ISCC_SCI_IMAGE_TRIM_TOLERANT`). It crops near-uniform borders (per-channel difference to the
  top-left pixel of at most 16) instead of only exactly uniform ones and repeats until a pass crops
  nothing (at most 16 passes). It is identical to the trim of the Image-Code in iscc-sdk 0.9.7, so
  both codes of an image see the same content.
- `preprocess_image` accepts per-call options, and `code_image_semantic` passes its options on to it
- **Compatibility:** Semantic-Codes change for images with near-uniform borders (typical for product
  photos on white backgrounds). Use `image_trim_tolerant=False` to get the earlier codes.
- Switched to the weight-compressed model file `iscc-sci-v0.1.0-w16.onnx` (fp16 weights cast to fp32
  at load, simplified graph, fixed 1×3×512×512 input): 105 MB instead of 213 MB, loads about 3x
  faster, inference speed unchanged. It is the same file the ISCC C2PA demo uses.
- **Compatibility:** the model switch changes a few Semantic-Codes by one or two bits. Of 71 test
  images, 8 differ in 1 to 2 of 256 bits, and at 64 bits 6 of 71 differ by 1 bit. The 6
  `iscc-samples` test images keep their codes, and embedding values shift in the fourth decimal.
- Added a model card (README section "Model") and the MIT licence notice of the model weights
  (`LICENSE-MODEL`)
- Removed the contradicting `License :: OSI Approved :: BSD License` classifier (the package licence
  is `Apache-2.0`)
- Wheel and sdist now ship `LICENSE` and `LICENSE-MODEL` (PEP 639 `license-files`)

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
