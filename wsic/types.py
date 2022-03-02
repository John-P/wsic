"""Type definitions for wsic."""
import re
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]
Magic = Union[bytes, re.Pattern]
