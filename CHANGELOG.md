# Changelog

## [0.3.0] - 2026-06-14

- Aligned `preprocess_image` preflight order with IEP-0004 (remove transparency before trimming borders)

## [0.2.1] - 2026-03-16

- Renamed CLI command from `sci` to `iscc-sci`
- Migrated build system from poetry to uv
- Added Python 3.14 support, dropped Python 3.9 and 3.10
- Rewrote CLI tests for 100% code coverage

## [0.2.0] - 2025-04-28

- Added support for configurable bit-length (32 to 256 bits)
- Improved image preprocessing with better handling of transparency and uniform borders
- Added embedding vector output option for advanced similarity analysis
- Enhanced CLI with debug mode and embedding flag options
- Added vector compression functionality to reduce storage requirements
- Updated dependencies

## [0.1.0] - 2024-02-07

- Initial Proof Of Concept
