from typing import TYPE_CHECKING

import astroid

from ._extractors import CODES
from ._finder import WarningFinder


if TYPE_CHECKING:
    from pylint.lint import PyLinter

try:
    from pylint.checkers import BaseChecker
    from pylint.interfaces import IAstroidChecker
except ImportError:
    BaseChecker = object
    IAstroidChecker = object


CODE = 'W99{:02}'


def register(linter: "PyLinter") -> None:
    linter.register_checker(PyLintChecker(linter))


class PyLintChecker(BaseChecker):
    """
    https://pylint.pycqa.org/en/latest/how_tos/custom_checkers.html
    """

    __implements__ = IAstroidChecker

    name = 'flake8_warnings'
    msgs = {CODE.format(code): ('%s', cat.__name__, '') for cat, code in CODES.items()}

    def visit_module(self: BaseChecker, node: astroid.Module) -> None:
        finder = WarningFinder(node)
        for winfo in finder.find():
            self.add_message(
                CODE.format(winfo.code),
                line=winfo.line,
                col_offset=winfo.col,
                args=(winfo.message,),
                node=winfo.node,
            )
