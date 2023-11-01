from typing import Iterator

import astroid

from ._base import Extractor, WarningInfo


class DocstringsExtractor(Extractor):
    """Extractor for warning messages in docstrings.
    """

    def extract(self, node: astroid.NodeNG) -> Iterator[WarningInfo]:
        yield from ()
