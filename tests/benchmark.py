# -*- coding: utf-8 -*-
from loguru import logger as log
from pathlib import Path
import iscc_sci as sci
import argparse
import time


def benchmark(folder):
    """
    Benchmark Image-Code generation for all images in `folder`.

    Per file stats are logged to the console during processing.
    Comprehensive aggregated statistics are shown after processing all images

    :param folder: Folder containing image files for benchmarking
    """
    folder = Path(folder)
    assert folder.is_dir(), f"{folder} is not a directory."

    total_time = 0
    file_count = 0

    for img_path in folder.glob("*.jpg"):
        start_time = time.time()
        try:
            iscc_meta = sci.code_image_semantic(img_path)
        except Exception as e:
            log.error(f"Processing {img_path.name} failed: {e}")
            continue
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        file_count += 1
        log.info(
            f"Processed {img_path.name} in {elapsed_time:.2f} seconds. ISCC: {iscc_meta['iscc']}"
        )

    if file_count > 0:
        avg_time = total_time / file_count
        log.info(
            f"Processed {file_count} images in {total_time:.2f} seconds. Average time per image: {avg_time:.2f} seconds."
        )
    else:
        log.warning("No images found in the provided folder.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark ISCC Semantic-Code Image generation.")
    parser.add_argument(
        "folder", type=str, help="Directory containing image files for benchmarking."
    )
    args = parser.parse_args()

    benchmark(args.folder)


if __name__ == "__main__":
    main()
