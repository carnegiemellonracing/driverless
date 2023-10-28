from flake8_plugin_utils import Plugin

from . import __version__
from .visitors import BreakpointVisitor


class BreakpointPlugin(Plugin):
    name = 'flake8-breakpoint'
    version = __version__
    visitors = [BreakpointVisitor]
