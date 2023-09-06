"""Linter module, flake8 plugin."""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import pkg_resources

if TYPE_CHECKING:
    import sys

    from flake8.options.manager import OptionManager

    if sys.version_info < (3, 8):
        from typing import Any as Final  # noqa: WPS433
    else:
        from typing import Final  # noqa: WPS433, WPS440

else:
    from typing import Any as Final  # noqa: WPS440


MAX_LINE_LENGTH: Final = 88
MAX_COMPLEXITY: Final = 10
ERROR_CODE: Final = "WAL"


class WhatALinter:
    """WhatALinter flake8 plugin."""

    name = "whatalinter"
    version = pkg_resources.get_distribution("python_dev_tools").version

    def __init__(self: WhatALinter, tree: ast.AST) -> None:
        """Empty method, needed by flake8.

        Args:
            tree (AST): unused
        """

    @classmethod
    def add_options(cls: type[WhatALinter], parser: OptionManager) -> None:
        """Add options to CLI parser.

        Args:
            parser (OptionManager): CLI parser
        """
        parser.extend_default_select([ERROR_CODE])
        parser.extend_default_ignore(
            [
                "E203",  # space around : in slice
                "LIT001",  # double quote strings
                "LIT003",  # double quote strings
                "LIT005",  # double quote strings
                # https://github.com/deppen8/pandas-vet/issues/74
                "PD005",  # add => +, too many false positives
                "PD011",  # values(), too many false positives
                "SIM9",  # experimental rules
                "WPS305",  # avoid f-strings
                "WPS306",  # required explicit subclassing of object
                "WPS421",  # Found wrong function call, useful in scripts
            ],
        )
        parser.parser.set_defaults(
            min_python_version="3.8.0",
            max_line_length=MAX_LINE_LENGTH,
            max_complexity=MAX_COMPLEXITY,
            inline_quotes='"',
            no_accept_encodings=True,
            literal_inline_quotes="double",
            literal_multiline_quotes="double",
        )

    def run(self: WhatALinter) -> list:
        """Empty method, needed by flake8.

        Returns:
            Empty list
        """
        return []
