#!/usr/bin/env python3
#
#  __init__.py
"""
A helper library for Flake8 plugins.
"""
#
#  Copyright © 2021 Dominic Davis-Foster <dominic@davis-foster.co.uk>
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
#  OR OTHER DEALINGS IN THE SOFTWARE.
#

# stdlib
import ast
from abc import ABC, abstractmethod
from typing import Generic, Iterator, List, Tuple, Type, TypeVar

__all__ = ["_V", "Visitor", "_P", "Plugin"]

__author__: str = "Dominic Davis-Foster"
__copyright__: str = "2021 Dominic Davis-Foster"
__license__: str = "MIT License"
__version__: str = "0.2.1"
__email__: str = "dominic@davis-foster.co.uk"

_P = TypeVar("_P", bound="Plugin")
_V = TypeVar("_V", bound="Visitor")


class Visitor(ast.NodeVisitor):
	"""
	AST node visitor.
	"""

	def __init__(self) -> None:
		#: The list of Flake8 errors identified by the visitor.
		self.errors: List[Tuple[int, int, str]] = []

	def report_error(self, node: ast.AST, error: str):
		"""
		Report an error for the given node.

		:param node:
		:param error:
		"""

		self.errors.append((
				node.lineno,
				node.col_offset,
				error,
				))


class Plugin(ABC, Generic[_V]):
	"""
	Abstract base class for Flake8 plugins.

	:param tree: The abstract syntax tree (AST) to check.

	**Minimum example:**

	.. code=block:: python

		class EncodingsPlugin(Plugin):
			'''
			A Flake8 plugin to identify incorrect use of encodings.

			:param tree: The abstract syntax tree (AST) to check.
			'''

			name: str = __name__
			version: str = __version__  #: The plugin version
	"""

	def __init__(self, tree: ast.AST):

		#: The abstract syntax tree (AST) being checked.
		self._tree = tree

	@property
	@abstractmethod
	def name(self) -> str:
		"""
		The plugin name.
		"""

		raise NotImplementedError

	@property
	@abstractmethod
	def version(self) -> str:
		"""
		The plugin version.
		"""

		raise NotImplementedError

	@property
	@abstractmethod
	def visitor_class(self) -> Type[_V]:
		"""
		The visitor class to use to traverse the AST.
		"""

		raise NotImplementedError

	def run(self: _P) -> Iterator[Tuple[int, int, str, Type[_P]]]:
		"""
		Traverse the Abstract Syntax Tree and identify errors.

		Yields a tuple of (line number, column offset, error message, type(self))
		"""

		visitor: _V = self.visitor_class()
		visitor.visit(self._tree)

		for line, col, msg in visitor.errors:
			yield line, col, msg, type(self)
