"""Linter (flake8, pylint, custom CLI) for finding usage of deprecated functions.
"""

from ._flake8_plugin import Flake8Checker
from ._pylint_plugin import register


__version__ = '0.4.0'
__all__ = ['Flake8Checker', 'register']
