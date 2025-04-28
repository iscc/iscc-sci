"""ISCC - Semantic-Code Image."""

import os
from importlib import metadata
from platformdirs import PlatformDirs

__version__ = metadata.version("iscc-sci")
APP_NAME = "iscc-sci"
APP_AUTHOR = "iscc"
dirs = PlatformDirs(appname=APP_NAME, appauthor=APP_AUTHOR)
os.makedirs(dirs.user_data_dir, exist_ok=True)

from iscc_sci.options import *
from iscc_sci.utils import *
from iscc_sci.code_semantic_image import *
