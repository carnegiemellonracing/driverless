"""Checker for quote handling on string literals."""

from __future__ import annotations

import enum
from typing import ClassVar, NamedTuple, TYPE_CHECKING

import flake8_literal

from . import checker

if (TYPE_CHECKING):
	import tokenize
	from collections.abc import Iterator, Sequence
	from flake8.options.manager import OptionManager


class Message(enum.Enum):
	"""Messages."""

	USE_SINGLE = (1, 'Use single quotes for string')
	USE_DOUBLE = (2, 'Use double quotes for string')
	MULTILINE_USE_SINGLE = (3, 'Use single quotes for multiline string')
	MULTILINE_USE_DOUBLE = (4, 'Use double quotes for multiline string')
	DOCSTRING_USE_SINGLE = (5, 'Use single quotes for docstring')
	DOCSTRING_USE_DOUBLE = (6, 'Use double quotes for docstring')
	DOCSTRING_USE_TRIPLE_SINGLE = (7, 'Use triple single quotes for docstring')
	DOCSTRING_USE_TRIPLE_DOUBLE = (8, 'Use triple double quotes for docstring')
	SWITCH_TO_DOUBLE = (11, 'Use double quotes for string to avoid escaped single quote')
	SWITCH_TO_SINGLE = (12, 'Use single quotes for string to avoid escaped double quote')
	UNNECESSARY_ESCAPE_SINGLE = (13, 'Escaped single quote is not necessary')
	UNNECESSARY_ESCAPE_DOUBLE = (14, 'Escaped double quote is not necessary')
	MATCH_CONTINUATION_DOUBLE = (15, 'Use double quotes for continuation strings to match')
	MATCH_CONTINUATION_SINGLE = (16, 'Use single quotes for continuation strings to match')
	UNNECESSARY_RAW = (21, 'Remove raw prefix when not using escapes')
	USE_RAW_PREFIX = (22, 'Use raw prefix to avoid escaped slash')

	@property
	def code(self) -> str:
		return (flake8_literal.quote_checker_prefix + str(self.value[0]).rjust(6 - len(flake8_literal.quote_checker_prefix), '0'))

	def text(self, **kwargs) -> str:
		return self.value[1].format(**kwargs)


class QuoteType(enum.Enum):
	"""Quote type enum."""

	SINGLE = 'single'
	DOUBLE = 'double'

	@classmethod
	def from_str(cls, value: str) -> (QuoteType | None):
		for member in cls.__members__.values():
			if (value.lower() == member.value):
				return member
		return None


QUOTE = {
	QuoteType.SINGLE: "'",
	QuoteType.DOUBLE: '"',
}

OTHER_QUOTE = {
	QuoteType.SINGLE: '"',
	QuoteType.DOUBLE: "'",
}

USE_QUOTE_MESSAGE = {
	QuoteType.SINGLE: Message.USE_SINGLE,
	QuoteType.DOUBLE: Message.USE_DOUBLE,
}

AVOID_ESCAPE_MESSAGE = {
	QuoteType.SINGLE: Message.SWITCH_TO_DOUBLE,
	QuoteType.DOUBLE: Message.SWITCH_TO_SINGLE,
}

UNNECESSARY_ESCAPE_MESSAGE = {
	QuoteType.SINGLE: Message.UNNECESSARY_ESCAPE_DOUBLE,
	QuoteType.DOUBLE: Message.UNNECESSARY_ESCAPE_SINGLE,
}

UNNECESSARY_OTHER_ESCAPE_MESSAGE = {
	QuoteType.SINGLE: Message.UNNECESSARY_ESCAPE_SINGLE,
	QuoteType.DOUBLE: Message.UNNECESSARY_ESCAPE_DOUBLE,
}

MULTILINE_USE_QUOTE_MESSAGE = {
	QuoteType.SINGLE: Message.MULTILINE_USE_SINGLE,
	QuoteType.DOUBLE: Message.MULTILINE_USE_DOUBLE,
}

DOCSTRING_USE_QUOTE_MESSAGE = {
	QuoteType.SINGLE: Message.DOCSTRING_USE_SINGLE,
	QuoteType.DOUBLE: Message.DOCSTRING_USE_DOUBLE,
}

DOCSTRING_USE_TRIPLE_MESSAGE = {
	QuoteType.SINGLE: Message.DOCSTRING_USE_TRIPLE_SINGLE,
	QuoteType.DOUBLE: Message.DOCSTRING_USE_TRIPLE_DOUBLE,
}

MATCH_CONTINUATION_MESSAGE = {
	QuoteType.SINGLE: Message.MATCH_CONTINUATION_DOUBLE,
	QuoteType.DOUBLE: Message.MATCH_CONTINUATION_SINGLE,
}


class Options(checker.Options):
	"""Protocol for options."""

	literal_inline: str
	literal_multiline: str
	literal_docstring: str
	literal_avoid_escape: bool


class Config(NamedTuple):
	"""Config options."""

	inline: QuoteType
	multiline: QuoteType
	docstring: QuoteType
	avoid_escape: bool


class QuoteChecker(checker.LiteralChecker):
	"""Check string literals for proper quotes."""

	config: ClassVar[Config]

	@classmethod
	def add_options(cls, option_manager: OptionManager) -> None:
		option_manager.add_option('--literal-inline-quotes', default='single',
		                          action='store', parse_from_config=True,
		                          choices=('single', 'double'), dest='literal_inline',
		                          help='Quote to use for inline string literals (default: single)')
		option_manager.add_option('--literal-multiline-quotes', default='single',
		                          action='store', parse_from_config=True,
		                          choices=('single', 'double'), dest='literal_multiline',
		                          help='Quote to use for multiline string literals (default: single)')
		option_manager.add_option('--literal-docstring-quotes', default='double',
		                          action='store', parse_from_config=True,
		                          choices=('single', 'double'), dest='literal_docstring',
		                          help='Quote to use for docstrings (default: double)')
		option_manager.add_option('--literal-avoid-escape', default=True, action='store_true',
		                          parse_from_config=True,
		                          help='Avoid escapes in inline string literals when possible (enabled by default)')
		option_manager.add_option('--literal-no-avoid-escape', default=None, action='store_false',
		                          parse_from_config=False, dest='literal_avoid_escape',
		                          help='Disable escape avoidance in inline string literals')
		option_manager.add_option('--literal-include-name', default=False, action='store_true',
		                          parse_from_config=True, dest='literal_include_name',
		                          help='Include plugin name in messages (enabled by default)')
		option_manager.add_option('--literal-no-include-name', default=None, action='store_false',
		                          parse_from_config=False, dest='literal_include_name',
		                          help='Remove plugin name from messages')

	@classmethod
	def parse_options(cls, options: Options) -> None:  # type: ignore[override]
		super().parse_options(options)
		cls.config = Config(inline=QuoteType.from_str(options.literal_inline) or QuoteType.SINGLE,
		                    multiline=QuoteType.from_str(options.literal_multiline) or QuoteType.SINGLE,
		                    docstring=QuoteType.from_str(options.literal_docstring) or QuoteType.DOUBLE,
		                    avoid_escape=options.literal_avoid_escape)

	def _process_literals(self, tokens: Sequence[tokenize.TokenInfo]) -> Iterator[tuple[tuple[int, int], str]]:
		if (not tokens):
			return

		desired = QUOTE[self.config.inline]
		other = OTHER_QUOTE[self.config.inline]
		needs_other = False
		cant_switch = False
		should_switch = set()
		others = []
		for token in tokens:
			quote = token.string[-1]
			prefix = token.string[:token.string.index(quote)].lower()
			string = token.string[len(prefix):]

			if (token in self.docstring_tokens):  # docstring
				if ((quote == QUOTE[self.config.docstring]) and (string[0:3] == (quote * 3))):
					continue
				if (quote == QUOTE[self.config.docstring]):
					yield self._logical_token_message(token, DOCSTRING_USE_TRIPLE_MESSAGE[self.config.docstring])
				else:
					yield self._logical_token_message(token, DOCSTRING_USE_QUOTE_MESSAGE[self.config.docstring])

			elif (string[0:3] == (quote * 3)):  # multiline
				if (quote == QUOTE[self.config.multiline]):
					continue
				yield self._logical_token_message(token, MULTILINE_USE_QUOTE_MESSAGE[self.config.multiline])

			else:  # inline
				contents = string[1:-1]

				if (quote == desired):  # check for escapes
					if ((not self.config.avoid_escape) or ('r' in prefix)):
						continue
					if (other in contents):
						cant_switch = True
					if ((desired in contents) and (other in contents)):
						continue  # both quotes used, nothing to do
					if (desired in contents):
						needs_other = True
						should_switch.add(token)
						yield self._logical_token_message(token, AVOID_ESCAPE_MESSAGE[self.config.inline])
					if ((other in contents) and (('\\' + other) in contents.replace(r'\\', ''))):
						yield self._logical_token_message(token, UNNECESSARY_ESCAPE_MESSAGE[self.config.inline])
				else:
					if ((desired in contents) and (other not in contents) and self.config.avoid_escape):
						if (('\\' + desired) in contents.replace(r'\\', '')):
							yield self._logical_token_message(token, UNNECESSARY_OTHER_ESCAPE_MESSAGE[self.config.inline])
						needs_other = True
						continue
					others.append(token)
		if ((1 < len(tokens)) and needs_other and (not cant_switch)):
			for token in tokens:
				quote = token.string[-1]
				if ((quote != desired) or (token.string[-3:] == (quote * 3)) or (token in should_switch)):
					continue
				yield self._logical_token_message(token, MATCH_CONTINUATION_MESSAGE[self.config.inline])
		else:
			for token in others:
				yield self._logical_token_message(token, USE_QUOTE_MESSAGE[self.config.inline])
