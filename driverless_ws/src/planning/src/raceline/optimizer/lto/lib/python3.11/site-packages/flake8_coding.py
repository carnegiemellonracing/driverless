# -*- coding: utf-8 -*-

import re

__version__ = '1.3.2'


class CodingChecker(object):
    name = 'flake8_coding'
    version = __version__

    def __init__(self, tree, filename):
        self.filename = filename

    @classmethod
    def add_options(cls, parser):
        parser.add_option(
            '--accept-encodings', default='latin-1, utf-8', action='store',
            help="Acceptable source code encodings for `coding:` magic comment"
        )
        parser.add_option(
            '--no-accept-encodings', action='store_true', parse_from_config=True,
            help="Warn for files containing a `coding:` magic comment"
        )

        if hasattr(parser, 'config_options'):  # for flake8 < 3.0
            parser.config_options.append('accept-encodings')
            parser.config_options.append('no-accept-encodings')

    @classmethod
    def parse_options(cls, options):
        if options.no_accept_encodings:
            cls.encodings = None
        else:
            cls.encodings = [e.strip().lower() for e in options.accept_encodings.split(',')]

    def read_headers(self):
        if self.filename in ('stdin', '-', None):
            try:
                # flake8 >= v3.0
                from flake8.engine import pep8 as stdin_utils
            except ImportError:
                from flake8 import utils as stdin_utils
            return stdin_utils.stdin_get_value().splitlines(True)[:2]
        else:
            try:
                import pycodestyle
            except ImportError:
                import pep8 as pycodestyle
            return pycodestyle.readlines(self.filename)[:2]

    def run(self):
        try:
            # PEP-263 says: a magic comment must be placed into the source
            #               files either as first or second line in the file
            lines = self.read_headers()
            if len(lines) == 0:
                return

            for lineno, line in enumerate(lines, start=1):
                matched = re.search(r'coding[:=]\s*([-\w.]+)', line, re.IGNORECASE)
                if matched:
                    if self.encodings:
                        if matched.group(1).lower() not in self.encodings:
                            yield lineno, 0, "C102 Unknown encoding found in coding magic comment", type(self)
                    else:
                        yield lineno, 0, "C103 Coding magic comment present", type(self)
                    break
            else:
                if self.encodings:
                    yield 1, 0, "C101 Coding magic comment not found", type(self)
        except IOError:
            pass
