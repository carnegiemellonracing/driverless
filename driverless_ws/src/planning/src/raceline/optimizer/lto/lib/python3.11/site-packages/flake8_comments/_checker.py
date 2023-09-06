from tokenize import TokenInfo
from typing import Iterable, Iterator

from ._core import get_redundant_comments
from . import __version__


class Checker:
    name = 'flake8-comments'
    version = __version__
    _tokens: Iterable[TokenInfo]
    _message = 'CM001: Redundant comment found'

    def __init__(
        self, tree=None, filename=None, lines=None,
        file_tokens: Iterable[TokenInfo] = None,
    ):
        assert file_tokens
        self._tokens = file_tokens

    def run(self) -> Iterator[tuple]:
        for comment in get_redundant_comments(self._tokens):
            yield (
                comment.start[0],
                comment.start[1],
                self._message,
                type(self),
            )
