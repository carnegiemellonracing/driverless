#!/usr/bin/env python3
#
#  __init__.py
"""
A Flake8 plugin to identify incorrect use of encodings.

.. seealso:: :pep:`597` -- Add optional EncodingWarning

.. TODO::

	Add support for checking e.g. logging.basicConfig(filename="log.txt").
	It has no encoding parameter before 3.9.
	Instead an open stream must be used, with the encoding set there.
"""
#
#  Copyright © 2020-2021 Dominic Davis-Foster <dominic@davis-foster.co.uk>
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
import configparser
import pathlib
import tempfile
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple, Type

# 3rd party
import flake8_helper
from astatine import get_attribute_name, kwargs_from_node
from domdf_python_tools.paths import PathPlus
from domdf_python_tools.typing import PathLike

if TYPE_CHECKING:
	# 3rd party
	from jedi import Script  # type: ignore
	from jedi.api.classes import Name  # type: ignore

__author__: str = "Dominic Davis-Foster"
__copyright__: str = "2020-2021 Dominic Davis-Foster"
__license__: str = "MIT License"
__version__: str = "0.5.0.post1"
__email__: str = "dominic@davis-foster.co.uk"

__all__ = ["Visitor", "ClassVisitor", "Plugin"]

ENC001 = "ENC001 no encoding specified for 'open'."
ENC002 = "ENC002 'encoding=None' used for 'open'."
ENC003 = "ENC003 no encoding specified for 'open' with unknown mode."
ENC004 = "ENC004 'encoding=None' used for 'open' with unknown mode."

ENC011 = "ENC011 no encoding specified for 'configparser.ConfigParser.read'."
ENC012 = "ENC012 'encoding=None' used for 'configparser.ConfigParser.read'."

ENC021 = "ENC021 no encoding specified for 'pathlib.Path.open'."
ENC022 = "ENC022 'encoding=None' used for 'pathlib.Path.open'."
ENC023 = "ENC023 no encoding specified for 'pathlib.Path.read_text'."
ENC024 = "ENC024 'encoding=None' used for 'pathlib.Path.read_text'."
ENC025 = "ENC025 no encoding specified for 'pathlib.Path.write_text'."
ENC026 = "ENC026 'encoding=None' used for 'pathlib.Path.write_text'."

_configparser_read = configparser.ConfigParser().read
_pathlib_open = pathlib.Path().open
_pathlib_read_text = pathlib.Path().read_text
_pathlib_write_text = pathlib.Path().write_text


def mode_is_binary(mode: ast.AST) -> Optional[bool]:
	"""
	Returns whether the mode of the call to :func:`open` is binary.

	Returns :py:obj:`None` if the mode cannot be determined.

	:param mode:
	"""

	if isinstance(mode, ast.Constant):  # pragma: no cover (<py38)
		return 'b' in mode.value
	elif isinstance(mode, ast.Str):  # pragma: no cover (py38+)
		return 'b' in mode.s
	else:
		return None


class Visitor(flake8_helper.Visitor):
	"""
	AST visitor to identify incorrect use of encodings.

	.. versionchanged:: 0.4.0

		The functionality for checking classes has moved to the :class:`~.ClassVisitor` subclass.
	"""

	def check_open_encoding(self, node: ast.Call):
		"""
		Check the call represented by the given AST node is using encodings correctly.

		This function checks :func:`open`, :func:`builtins.open <open>` and :func:`io.open`.

		.. versionchanged:: 0.2.0  Renamed from ``check_encoding``
		"""

		kwargs = kwargs_from_node(node, open)

		# print(node.lineno, node.col_offset)
		# print(node.args, node.keywords)
		# print(kwargs_from_node(node))

		unknown_mode = False

		if "mode" in kwargs:
			is_binary = mode_is_binary(kwargs["mode"])

			if is_binary:
				return
			elif is_binary is None:
				unknown_mode = True

		if "encoding" not in kwargs:
			self.report_error(node, ENC003 if unknown_mode else ENC001)

		elif isinstance(kwargs["encoding"], (ast.Constant, ast.NameConstant)):
			if kwargs["encoding"].value is None:
				self.report_error(node, ENC004 if unknown_mode else ENC002)

	check_encoding = check_open_encoding  # deprecated

	def visit_Call(self, node: ast.Call):  # noqa: D102

		if isinstance(node.func, ast.Name):

			if node.func.id == "open":
				# print(node.func.id)
				self.check_encoding(node)
				return

		elif isinstance(node.func, ast.Attribute):
			if isinstance(node.func.value, ast.Name):

				if node.func.value.id in {"builtins", "io"} and node.func.attr == "open":
					self.check_open_encoding(node)
					return

			if isinstance(node.func.value, ast.Str):  # pragma: no cover
				# Attribute on a string
				return self.generic_visit(node)

			elif isinstance(node.func.value, ast.BinOp):  # pragma: no cover
				# TODO
				# Expressions such as (tmp_pathplus / "code.py").write_text(example_source)
				return self.generic_visit(node)

			elif isinstance(node.func.value, ast.Subscript):  # pragma: no cover
				# TODO
				# Expressions such as my_list[0].run()
				return self.generic_visit(node)

		self.generic_visit(node)


class ClassVisitor(Visitor):
	"""
	AST visitor to identify incorrect use of encodings,
	with support for :class:`pathlib.Path` and :class:`configparser.ConfigParser`.

	.. versionadded:: 0.4.0
	"""  # noqa: D400

	def __init__(self):
		try:
			# 3rd party
			import jedi
		except ImportError as e:
			exc = e.__class__("This class requires 'jedi' to be installed but it could not be imported.")
			exc.__traceback__ = e.__traceback__
			raise exc from None

		super().__init__()
		self.filename = PathPlus("<unknown>")
		self.jedi_script = jedi.Script('')

	def first_visit(self, node: ast.AST, filename: PathPlus):
		"""
		Like :meth:`ast.NodeVisitor.visit`, but configures type inference.

		.. versionadded:: 0.2.0

		:param node:
		:param filename: The path to Python source file the AST node was generated from.
		"""

		# 3rd party
		import jedi  # nodep

		self.filename = PathPlus(filename)
		self.jedi_script = jedi.Script(self.filename.read_text(), path=self.filename)
		self.visit(node)

	def check_configparser_encoding(self, node: ast.Call):
		"""
		Check the call represented by the given AST node is using encodings correctly.

		This function checks :meth:`configparser.ConfigParser.read`.

		.. versionadded:: 0.2.0
		"""

		kwargs = kwargs_from_node(node, _configparser_read)

		if "encoding" not in kwargs:
			self.report_error(node, ENC011)

		elif isinstance(kwargs["encoding"], (ast.Constant, ast.NameConstant)):
			if kwargs["encoding"].value is None:
				self.report_error(node, ENC012)

	def check_pathlib_encoding(self, node: ast.Call, method_name: str):
		"""
		Check the call represented by the given AST node is using encodings correctly.

		This function checks :meth:`pathlib.Path.open`, :meth:`pathlib.Path.read_text`,
		and :meth:`pathlib.Path.write_text`.

		.. versionadded:: 0.3.0
		"""

		function: Callable

		if method_name == "open":
			no_encoding = ENC021
			encoding_none = ENC022
			function = _pathlib_open
		elif method_name == "read_text":
			no_encoding = ENC023
			encoding_none = ENC024
			function = _pathlib_read_text
		elif method_name == "write_text":
			no_encoding = ENC025
			encoding_none = ENC026
			function = _pathlib_write_text
		else:  # pragma: no cover
			# Not a method we understand
			return

		kwargs = kwargs_from_node(node, function)

		unknown_mode = False

		if "mode" in kwargs:
			is_binary = mode_is_binary(kwargs["mode"])

			if is_binary:
				return
			elif is_binary is None:  # pragma: no cover
				# TODO: unknown mode
				unknown_mode = True
				return

		if "encoding" not in kwargs:
			self.report_error(node, no_encoding)

		elif isinstance(kwargs["encoding"], (ast.Constant, ast.NameConstant)):
			if kwargs["encoding"].value is None:
				self.report_error(node, encoding_none)

	def visit_Call(self, node: ast.Call):  # noqa: D102

		if isinstance(node.func, ast.Name):

			if node.func.id == "open":
				# print(node.func.id)
				self.check_encoding(node)
				return

		elif isinstance(node.func, ast.Attribute):
			if isinstance(node.func.value, ast.Name):

				if node.func.value.id in {"builtins", "io"} and node.func.attr == "open":
					self.check_open_encoding(node)
					return

			if isinstance(node.func.value, ast.Str):  # pragma: no cover
				# Attribute on a string
				return self.generic_visit(node)

			elif isinstance(node.func.value, ast.BinOp):  # pragma: no cover
				# TODO
				# Expressions such as (tmp_pathplus / "code.py").write_text(example_source)
				return self.generic_visit(node)

			elif isinstance(node.func.value, ast.Subscript):  # pragma: no cover
				# TODO
				# Expressions such as my_list[0].run()
				return self.generic_visit(node)

			elif self.filename.as_posix() == "<unknown>":
				# no jedi source (run with .visit() or from memory)
				return self.generic_visit(node)

			else:

				try:
					inferred_types = get_inferred_types(self.jedi_script, node)
					method_name = tuple(get_attribute_name(node.func))[-1]
				except NotImplementedError:  # pragma: no cover
					return self.generic_visit(node)

				for class_name in inferred_types:
					if is_configparser_read(class_name, method_name):
						self.check_configparser_encoding(node)

					elif is_pathlib_method(class_name, method_name):
						self.check_pathlib_encoding(node, method_name)

		self.generic_visit(node)


class Plugin(flake8_helper.Plugin[Visitor]):
	"""
	A Flake8 plugin to identify incorrect use of encodings.

	:param tree: The abstract syntax tree (AST) to check.
	"""

	name: str = __name__
	version: str = __version__  #: The plugin version
	visitor_class = Visitor

	def __init__(self, tree: ast.AST, filename: PathLike):
		super().__init__(tree)
		self.filename = PathPlus(filename)

	def run(self) -> Iterator[Tuple[int, int, str, Type["Plugin"]]]:  # noqa: D102

		try:
			# 3rd party
			import jedi

			# jedi.settings.fast_parser = False

			original_cache_dir = jedi.settings.cache_directory

			with tempfile.TemporaryDirectory() as cache_directory:
				jedi.settings.cache_directory = cache_directory

				class_visitor = ClassVisitor()
				class_visitor.first_visit(self._tree, self.filename)

				for line, col, msg in class_visitor.errors:
					yield line, col, msg, type(self)

			jedi.settings.cache_directory = original_cache_dir

		except ImportError:
			visitor = Visitor()
			visitor.visit(self._tree)

			for line, col, msg in visitor.errors:
				yield line, col, msg, type(self)


def is_configparser_read(class_name: str, method_name: str) -> bool:
	"""
	Returns :py:obj:`True` if method is :meth:`configparser.ConfigParser.read` or
	:meth:`configparser.RawConfigParser.read`.

	.. versionadded:: 0.3.0

	:param class_name: The inferred name of the class the method belongs to.
	:param method_name: The name of the record.
	"""  # noqa: D400

	if class_name not in {"configparser.ConfigParser", "configparser.RawConfigParser"}:
		return False

	if method_name != "read":
		return False

	return True


def is_pathlib_method(class_name: str, method_name: str) -> bool:
	"""
	Returns :py:obj:`True` if method is :meth:`pathlib.Path.open`,
	:meth:`read_text() <pathlib.Path.read_text>` or :meth:`write_text() <pathlib.Path.write_text>`.

	.. versionadded:: 0.3.0

	:param class_name: The inferred name of the class the method belongs to.
	:param method_name: The name of the record.
	"""  # noqa: D400

	if class_name not in {"pathlib.Path", "pathlib.WindowsPath", "pathlib.PosixPath"}:
		return False

	if method_name not in {"open", "read_text", "write_text"}:
		return False

	return True


def get_inferred_types(jedi_script: "Script", node: ast.Call) -> List[str]:
	"""
	Returns a list of types inferred by ``jedi`` for the given call node.

	:param jedi_script:
	:param node:
	"""

	attr_names = tuple(get_attribute_name(node.func))
	inferred_types = set()

	inferred_name: "Name"
	for inferred_name in jedi_script.infer(node.lineno, node.func.col_offset):
		inferred_types.add(inferred_name.full_name)

	for inferred_name in jedi_script.infer(node.lineno, node.func.col_offset + len('.'.join(attr_names[:-1]))):
		inferred_types.add(inferred_name.full_name)

	return sorted(filter(None, inferred_types))
