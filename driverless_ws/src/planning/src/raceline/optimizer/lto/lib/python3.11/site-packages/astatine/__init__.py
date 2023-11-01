#!/usr/bin/env python3
#
#  __init__.py
"""
Some handy helper functions for Python's AST module.
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
#  mark_text_ranges from Thonny
#  https://github.com/thonny/thonny/blob/master/thonny/ast_utils.py
#  Copyright (c) 2020 Aivar Annamaa
#  MIT Licensed
#

# stdlib
import ast
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

# 3rd party
from asttokens.asttokens import ASTTokens  # type: ignore[import]
from domdf_python_tools.stringlist import StringList
from domdf_python_tools.utils import posargs2kwargs

Str: Tuple[Type, ...]
Constant: Tuple[Type, ...]
Expr: Tuple[Type, ...]

try:  # pragma: no cover
	# 3rd party
	import typed_ast.ast3
	Str = (ast.Str, typed_ast.ast3.Str)
	Constant = (
			ast.Constant,
			typed_ast.ast3.Constant,  # type: ignore[attr-defined]
			)
	Expr = (ast.Expr, typed_ast.ast3.Expr)

except ImportError:  # pragma: no cover
	Str = (ast.Str, )
	Constant = (ast.Constant, )
	Expr = (ast.Expr, )

__author__: str = "Dominic Davis-Foster"
__copyright__: str = "2021 Dominic Davis-Foster"
__license__: str = "MIT License"
__version__: str = "0.3.2"
__email__: str = "dominic@davis-foster.co.uk"

__all__ = [
		"get_docstring_lineno",
		"get_toplevel_comments",
		"is_type_checking",
		"mark_text_ranges",
		"kwargs_from_node",
		"get_attribute_name",
		"get_contextmanagers",
		"get_constants",
		]


def get_toplevel_comments(source: str) -> StringList:
	"""
	Returns a list of comment lines from ``source`` which occur before the first line of source code
	(including before module-level docstrings).

	:param source:
	"""  # noqa: D400

	comments = StringList()

	for line in source.splitlines():
		if not line.startswith('#'):
			break

		comments.append(line)

	comments.blankline(ensure_single=True)

	return comments


def is_type_checking(node: ast.AST) -> bool:
	"""
	Returns whether the given ``if`` block is ``if typing.TYPE_CHECKING`` or equivalent.

	:param node:
	"""

	if isinstance(node, ast.If):
		node = node.test

	if isinstance(node, ast.NameConstant) and node.value is False:
		return True
	elif isinstance(node, ast.Name) and node.id == "TYPE_CHECKING":
		return True
	elif isinstance(node, ast.Attribute) and node.attr == "TYPE_CHECKING":
		return True
	elif isinstance(node, ast.BoolOp):
		for value in node.values:
			if is_type_checking(value):
				return True

	return False


def mark_text_ranges(node: ast.AST, source: str) -> None:
	"""
	Recursively add the ``end_lineno`` and ``end_col_offset`` attributes to each child of ``node``
	which already has the attributes ``lineno`` and ``col_offset``.

	:param node: An AST node created with :func:`ast.parse`.
	:param source: The corresponding source code for the node.
	"""  # noqa: D400

	ASTTokens(source, tree=node)

	for child in ast.walk(node):
		if hasattr(child, "last_token"):
			child.end_lineno, child.end_col_offset = child.last_token.end  # type: ignore[attr-defined]

			if hasattr(child, "lineno"):
				# Fixes problems with some nodes like binop
				child.lineno, child.col_offset = child.first_token.start  # type: ignore[attr-defined]


def get_docstring_lineno(node: Union[ast.FunctionDef, ast.ClassDef, ast.Module]) -> Optional[int]:
	"""
	Returns the line number of the start of the docstring for ``node``.

	:param node:

	.. warning::

		On CPython 3.6 and 3.7 the line number may not be correct, due to https://bugs.python.org/issue16806.

		CPython 3.8 and above are unaffected, as are PyPy 3.6 and 3.7

		Accurate line numbers on CPython 3.6 and 3.7 may be obtained by using https://github.com/domdfcoding/typed_ast,
		which contains the backported fix from Python 3.8.

	"""

	if not (node.body and isinstance(node.body[0], Expr)):  # pragma: no cover
		return None

	body = node.body[0].value  # type: ignore[attr-defined]

	if isinstance(body, Constant) and isinstance(body.value, str):  # pragma: no cover (<py38)
		return body.lineno
	elif isinstance(body, Str):  # pragma: no cover (py38+)
		return body.lineno
	else:  # pragma: no cover
		return None


def kwargs_from_node(
		node: ast.Call,
		posarg_names: Union[Iterable[str], Callable],
		) -> Dict[str, ast.AST]:
	"""
	Returns a mapping of argument names to the AST nodes representing their values, for the given function call.

	.. versionadded:: 0.3.1

	:param node:
	:param posarg_names: Either a list of positional argument names for the function, or the function object.

	:rtype:

	.. latex:clearpage::
	"""

	args: List[ast.expr] = node.args
	keywords: List[ast.keyword] = node.keywords

	kwargs = {cast(str, kw.arg): kw.value for kw in keywords}

	return posargs2kwargs(
			args,
			posarg_names,
			kwargs,
			)


def get_attribute_name(node: ast.AST) -> Iterable[str]:
	"""
	Returns the elements of the dotted attribute name for the given AST node.

	.. versionadded:: 0.3.1

	:param node:

	:raises NotImplementedError: if the name contains an unknown node
		(i.e. not :class:`ast.Name`, :class:`ast.Attribute`, or :class:`ast.Call`)
	"""

	if isinstance(node, ast.Name):
		yield node.id
	elif isinstance(node, ast.Attribute):
		yield from get_attribute_name(node.value)
		yield node.attr
	elif isinstance(node, ast.Call):
		yield from get_attribute_name(node.func)
	else:
		raise NotImplementedError(type(node))


def get_contextmanagers(with_node: ast.With) -> Dict[Tuple[str, ...], ast.withitem]:
	"""
	For the given ``with`` block, returns a mapping of the contextmanager names to the individual nodes.

	.. versionadded:: 0.3.1

	:param with_node:
	"""

	contextmanagers = {}

	item: ast.withitem
	for item in with_node.items:

		name = tuple(get_attribute_name(item.context_expr))

		contextmanagers[name] = item

	return contextmanagers


def get_constants(module: ast.Module) -> Dict[str, Any]:
	"""
	Returns a ``name: value`` mapping of constants in the given module.

	.. versionadded:: 0.3.1

	:param module:

	:rtype:

	.. latex:clearpage::
	"""

	constants = {}

	for node in module.body:
		if isinstance(node, ast.Assign):
			targets = ['.'.join(get_attribute_name(t)) for t in node.targets]
			value = ast.literal_eval(node.value)

			for target in targets:
				constants[target] = value

	return constants
