import ast
from typing import Iterator

from .flake_diagnostic import FlakeDiagnostic
from .rules import rules

from .patch_const import LegacyConstantRewriter


class UselessAssert:
    name = "flake8-useless-assert"
    version = "0.4.4"

    def __init__(self, tree: ast.Module) -> None:
        self._tree = tree

    def __iter__(self) -> Iterator[FlakeDiagnostic]:
        LegacyConstantRewriter().visit(self._tree)

        for rule in rules:
            yield from rule(self._tree)
