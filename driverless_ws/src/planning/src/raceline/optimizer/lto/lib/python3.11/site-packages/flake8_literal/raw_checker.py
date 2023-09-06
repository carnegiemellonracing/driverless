"""Checker for raw string literals."""

from __future__ import annotations

import ast
import enum
import tokenize
from typing import ClassVar, NamedTuple, TYPE_CHECKING

import flake8_literal

from . import checker

if (TYPE_CHECKING):
	from collections.abc import Iterator, Sequence
	from flake8.options.manager import OptionManager


class Message(enum.Enum):
	"""Messages."""

	UNNECESSARY_RAW = (1, 'Remove raw prefix when not using escapes')
	USE_RAW_FOR_SLASH = (2, 'Use raw prefix to avoid escaped slash')
	USE_RAW_FOR_REGEX = (3, 'Use raw prefix for re pattern')

	@property
	def code(self) -> str:
		return (flake8_literal.raw_checker_prefix + str(self.value[0]).rjust(6 - len(flake8_literal.raw_checker_prefix), '0'))

	def text(self, **kwargs) -> str:
		return self.value[1].format(**kwargs)


class RePatternRaw(enum.Enum):
	"""Raw regex type option enum."""

	AVOID = 'avoid'
	ALLOW = 'allow'
	ALWAYS = 'always'

	@classmethod
	def from_str(cls, value: str) -> (RePatternRaw | None):
		for member in cls.__members__.values():
			if (value.lower() == member.value):
				return member
		return None


class Options(checker.Options):
	"""Protocol for options."""

	literal_re_pattern_raw: str
	literal_avoid_escape: bool


class Config(NamedTuple):
	"""Config options."""

	re_pattern_raw: RePatternRaw
	avoid_escape: bool


RE_METHODS = frozenset((
	'compile',
	'search',
	'match',
	'fullmatch',
	'split',
	'findall',
	'finditer',
	'sub',
	'subn',
	'escape',
))


_ast_string_classes: list[type] = [ast.Constant]
try:
	_ast_string_classes.append(ast.Str)  # ast.Str deprecated in 3.8
	_ast_string_classes.append(ast.Bytes)  # ast.Bytes deprecated in 3.8
except AttributeError:
	pass
AST_STRINGS = tuple(_ast_string_classes)


class RawChecker(checker.Checker):
	"""Check string literals for proper raw usage."""

	config: ClassVar[Config]

	tokens: Sequence[tokenize.TokenInfo]
	re_arguments: set[tuple[int, int]]

	@classmethod
	def add_options(cls, option_manager: OptionManager) -> None:
		option_manager.add_option('--literal-re-pattern-raw', default='allow',
		                          action='store', parse_from_config=True,
		                          choices=('avoid', 'allow', 'always'), dest='literal_re_pattern_raw',
		                          help='Use raw strings for regular expressions (default:allow)')

	@classmethod
	def parse_options(cls, options: Options) -> None:  # type: ignore[override]
		super().parse_options(options)
		cls.config = Config(re_pattern_raw=RePatternRaw.from_str(options.literal_re_pattern_raw) or RePatternRaw.ALLOW,
		                    avoid_escape=options.literal_avoid_escape)

	def __init__(self, tree: ast.AST, file_tokens: Sequence[tokenize.TokenInfo]) -> None:
		super().__init__()
		self.tokens = file_tokens
		self.re_arguments = self._find_re_arguments(tree)

	def _find_re_arguments(self, tree: ast.AST) -> set[tuple[int, int]]:
		re_arguments: set[tuple[int, int]] = set()

		def _add_re_argument(node: ast.AST) -> None:
			if (isinstance(node, AST_STRINGS)):
				re_arguments.add((node.lineno, node.col_offset))
			if (isinstance(node, (ast.BinOp, ast.JoinedStr))):
				for child in ast.iter_child_nodes(node):
					_add_re_argument(child)
			if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
				_add_re_argument(node.func.value)

		def _process_node(node: ast.AST, modules: set[str], functions: set[str]) -> None:
			modules = set(modules)
			functions = set(functions)
			for child in ast.iter_child_nodes(node):
				if (isinstance(child, ast.Import)):
					for name in child.names:
						if ('re' == name.name):
							modules.add(name.asname if (name.asname is not None) else name.name)
				elif (isinstance(child, ast.ImportFrom)):
					if ('re' == child.module):
						for name in child.names:
							if (name.name in RE_METHODS):
								functions.add(name.asname if (name.asname is not None) else name.name)
				elif (isinstance(child, ast.Call)):
					function = child.func
					if (isinstance(function, ast.Name) and (function.id in functions)):  # direct function call
						if (child.args):
							_add_re_argument(child.args[0])
					elif ((isinstance(function, ast.Attribute)) and (function.attr in RE_METHODS)):	 # function is an attribute of something
						if (isinstance(function.value, ast.Name) and (function.value.id in modules)):  # function in re module
							if (child.args):
								_add_re_argument(child.args[0])
				_process_node(child, modules, functions)

		_process_node(tree, set(), set())
		return re_arguments

	def _process_literals(self, tokens: Sequence[tokenize.TokenInfo]) -> Iterator[checker.ASTResult]:
		if (not tokens):
			return
		re_pattern = (tokens[0].start in self.re_arguments)
		for token in tokens:
			quote = token.string[-1]
			prefix = token.string[:token.string.index(quote)].lower()
			string = token.string[len(prefix):]

			if (string[0:3] == (quote * 3)):  # multiline
				contents = string[3:-3]
			else:  # inline
				contents = string[1:-1]

			if ('r' in prefix):
				if ((not ('\\' in contents)) and ((not re_pattern) or (RePatternRaw.AVOID == self.config.re_pattern_raw))):
					yield self._ast_token_message(token, Message.UNNECESSARY_RAW)
			else:
				if ((r'\\' in contents) and ('\\' not in contents.replace(r'\\', '')) and self.config.avoid_escape):
					trail_count = 0
					test_contents = contents
					while test_contents.endswith(r'\\'):
						test_contents = test_contents[:-2]
						trail_count += 1
					if (0 == (trail_count % 2)):  # raw strings can't end in an odd number of backslashes
						yield self._ast_token_message(token, Message.USE_RAW_FOR_SLASH)
						continue
				if (re_pattern and (RePatternRaw.ALWAYS == self.config.re_pattern_raw)):
					yield self._ast_token_message(token, Message.USE_RAW_FOR_REGEX)

	def __iter__(self) -> Iterator[checker.ASTResult]:
		"""Primary call from flake8, yield error messages."""
		continuation: list[tokenize.TokenInfo] = []
		for token in self.tokens:
			if (token.type in checker.IGNORE_TOKENS):
				continue

			if (tokenize.STRING == token.type):
				continuation.append(token)
				continue

			for message in self._process_literals(continuation):
				yield message
			continuation = []

		for message in self._process_literals(continuation):
			yield message
