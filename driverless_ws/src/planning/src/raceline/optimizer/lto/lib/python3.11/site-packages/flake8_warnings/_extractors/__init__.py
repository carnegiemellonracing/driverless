from typing import Tuple, Type

from ._base import CODES, Extractor, WarningInfo
from ._decorators import DecoratorsExtractor
from ._docstrings import DocstringsExtractor
from ._stdlib import StdlibExtractor
from ._warnings import WarningsExtractor


__all__ = ["CODES", "EXTRACTORS", "Extractor", "WarningInfo"]
EXTRACTORS: Tuple[Type[Extractor], ...] = (
    DecoratorsExtractor,
    DocstringsExtractor,
    StdlibExtractor,
    WarningsExtractor,
)
