import argparse
import glob
import json
from pathlib import Path
from loguru import logger
import iscc_sci as sci


def main():
    parser = argparse.ArgumentParser(description="Generate Semantic Image-Codes.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to image file(s) (supports glob patterns)",
        nargs="?",
    )
    parser.add_argument(
        "-b", "--bits", type=int, default=256, help="Bit-Length of Code (default 256)"
    )

    parser.add_argument(
        "-e", "--embedding", action="store_true", help="Include image embedding vector in output"
    )

    parser.add_argument("-d", "--debug", action="store_true", help="Show debugging messages.")
    args = parser.parse_args()

    if args.path is None:
        parser.print_help()
        return

    if not args.debug:
        logger.remove()

    for path in glob.glob(args.path):
        path = Path(path)
        if path.is_file():
            logger.debug(f"Processing {path.name}")
            sci_meta = sci.code_image_semantic(path, bits=args.bits, embedding=args.embedding)
            if args.embedding:
                print(json.dumps(sci_meta, indent=2))
            else:
                print(sci_meta["iscc"])


if __name__ == "__main__":  # pragma: no cover
    main()
