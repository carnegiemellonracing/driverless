# Copyright 2021 Damien Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for the flake8_secure_coding_standard plugin."""

import ast
import importlib.metadata
import operator
import platform
import stat
from typing import Any, AnyStr, Dict, Generator, List, Tuple, Type, Union

import flake8.options.manager

ast_Constant = ast.Constant

_use_optparse = tuple(int(s) for s in importlib.metadata.version('flake8').split('.')) < (3, 8, 0)


if _use_optparse:  # pragma: no cover
    import optparse  # noqa: F401 pylint: disable=deprecated-module, unused-import
else:
    import argparse

# ==============================================================================

_DEFAULT_MAX_MODE = 0o755

SCS100 = 'SCS100 use of os.path.abspath() and os.path.relpath() should be avoided in favor of os.path.realpath()'
SCS101 = 'SCS101 `eval()` and `exec()` represent a security risk and should be avoided'
SCS102 = 'SCS102 use of `os.system()` should be avoided'
SCS103 = ' '.join(
    [
        'SCS103 use of `shell=True` in subprocess functions or use of functions that internally set it should be',
        'avoided',
    ]
)
SCS104 = 'SCS104 use of `tempfile.mktemp()` should be avoided, prefer `tempfile.mkstemp()`'
SCS105 = ' '.join(
    [
        'SCS105 use of `yaml.load()` should be avoided, prefer `yaml.safe_load()` or',
        '`yaml.load(xxx, Loader=SafeLoader)`',
    ]
)
SCS106 = 'SCS106 use of `jsonpickle.decode()` should be avoided'
SCS107 = 'SCS107 debugging code should not be present in production code (e.g. `import pdb`)'
SCS108 = 'SCS108 `assert` statements should not be present in production code'
SCS109 = ' '.join(
    [
        'SCS109 Use of builtin `open` for writing is discouraged in favor of `os.open` to allow for setting file',
        'permissions',
    ]
)
SCS110 = 'SCS110 Use of `os.popen()` should be avoided, as it internally uses `subprocess.Popen` with `shell=True`'
SCS111 = 'SCS111 Use of `shlex.quote()` should be avoided on non-POSIX platforms (such as Windows)'
SCS112 = 'SCS112 Avoid using `os.open` with unsafe permissions (should be {})'
SCS113 = 'SCS113 Avoid using `pickle.load()` and `pickle.loads()`'
SCS114 = 'SCS114 Avoid using `marshal.load()` and `marshal.loads()`'
SCS115 = 'SCS115 Avoid using `shelve.open()`'
SCS116 = 'SCS116 Avoid using `os.mkdir` and `os.makedirs` with unsafe file permissions (should be {})'
SCS117 = 'SCS117 Avoid using `os.mkfifo` with unsafe file permissions (should be {})'
SCS118 = 'SCS118 Avoid using `os.mknod` with unsafe file permissions (should be {})'
SCS119 = 'SCS119 Avoid using `os.chmod` with unsafe file permissions (W ^ X for group and others)'


# ==============================================================================
# Helper functions


def _read_octal_mode_option(name, value, default):
    """
    Read an integer or list of integer configuration option.

    Args:
        name (str): Name of option
        value (str): Value of option from the configuration file or on the CLI. Its value can be any of:
            - 'yes', 'y', 'true' (case-insensitive)
                The maximum mode value is then set to self.DEFAULT_MAX_MODE
            - a single octal or decimal integer
                The maximum mode value is then set to that integer value
            - a comma-separated list of integers (octal or decimal)
                The allowed mode values are then those found in the list
            - anything else will count as a falseful value
        default (int,list): Default value for option if set to one of
            ('y', 'yes', 'true') in the configuration file or on the CLI

    Returns:
        A single integer or a (possibly empty) list of integers

    Raises:
        ValueError: if the value of the option is not valid
    """

    def _str_to_int(arg):
        try:
            return int(arg, 8)
        except ValueError:
            return int(arg)

    value = value.lower()
    modes = [mode.strip() for mode in value.split(',')]

    if len(modes) > 1:
        # Lists of allowed modes
        try:
            allowed_modes = [_str_to_int(mode) for mode in modes if mode]
        except ValueError as error:
            raise ValueError(f'Unable to convert {modes} elements to integers!') from error
        else:
            if not allowed_modes:
                raise ValueError(f'Calculated empty value for `{name}`!')
            return allowed_modes
    elif modes and modes[0]:
        # Single values (ie. max allowed value for mode)
        try:
            return _str_to_int(value)
        except ValueError as error:
            if value in ('y', 'yes', 'true'):
                return default
            if value in ('n', 'no', 'false'):
                return None
            raise ValueError(f'Invalid value for `{name}`: {value}!') from error
    else:
        raise ValueError(f'Invalid value for `{name}`: {value}!')


# ==============================================================================


def _is_posix():
    """Return True if the current system is POSIX-compatible."""
    # NB: we could simply use `os.name` instead of `platform.system()`. However, that solution would be difficult to
    #     test using `mock` as a few modules (like `pytest`) actually use it internally...
    return platform.system() in ('Linux', 'Darwin')


_is_unix = _is_posix

# ------------------------------------------------------------------------------


def _is_function_call(node: ast.Call, module: AnyStr, function: Union[List[AnyStr], Tuple[AnyStr], AnyStr]) -> bool:
    if not isinstance(function, (list, tuple)):
        function = (function,)
    return (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == module
        and node.func.attr in function
    )


# ==============================================================================


def _is_os_path_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)  # pylint: disable=R0916
        and (
            (isinstance(node.func.value, ast.Name) and node.func.value.id == 'op')
            or (
                isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == 'path'
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == 'os'
            )
        )
        and node.func.attr in ('abspath', 'relpath')
    )


def _is_builtin_open_for_writing(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == 'open':
        mode = ''
        if len(node.args) > 1:
            if isinstance(node.args[1], ast.Name):
                return True  # variable -> to be on the safe side, flag as inappropriate
            if isinstance(node.args[1], ast_Constant):
                mode = node.args[1].value
            if isinstance(node.args[1], ast.Str):
                mode = node.args[1].s
        else:
            for keyword in node.keywords:
                if keyword.arg == 'mode':
                    if not isinstance(keyword.value, ast_Constant):
                        return True  # variable -> to be on the safe side, flag as inappropriate
                    mode = keyword.value.value
                    break
        if any(m in mode for m in 'awx'):
            # Cover:
            #  * open(..., "w")
            #  * open(..., "wb")
            #  * open(..., "a")
            #  * open(..., "x")
            return True
    return False


def _get_mode_arg(node, args_idx):
    mode = None
    node_intern = None
    if len(node.args) > args_idx:
        node_intern = node.args[args_idx]
    elif node.keywords:
        for keyword in node.keywords:
            if keyword.arg == 'mode':
                node_intern = keyword.value
                break
    if node_intern:
        if isinstance(node_intern, ast_Constant):
            mode = node_intern.value
        elif isinstance(node_intern, ast.Num):  # pragma: no cover
            mode = node_intern.n
    return mode


def _is_allowed_mode(node, allowed_modes, args_idx):
    mode = _get_mode_arg(node, args_idx=args_idx)
    if mode is not None and allowed_modes:
        return mode in allowed_modes

    # NB: default to True in all other cases
    return True


def _is_shell_true_call(node: ast.Call) -> bool:
    if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)):
        return False

    # subprocess module
    if node.func.value.id in ('subprocess', 'sp'):
        if node.func.attr in ('call', 'check_call', 'check_output', 'Popen', 'run'):
            for keyword in node.keywords:
                if keyword.arg == 'shell' and isinstance(keyword.value, ast_Constant) and bool(keyword.value.value):
                    return True
            if len(node.args) > 8 and isinstance(node.args[8], ast_Constant) and bool(node.args[8].value):
                return True
        if node.func.attr in ('getoutput', 'getstatusoutput'):
            return True

    # asyncio module
    if (node.func.value.id == 'asyncio' and node.func.attr == 'create_subprocess_shell') or (
        node.func.value.id == 'loop' and node.func.attr == 'subprocess_shell'
    ):
        return True

    return False


def _is_pdb_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.value.id == 'pdb':
            # Cover:
            #  * pdb.func()
            return True
    if isinstance(node.func, ast.Name):
        if node.func.id == 'Pdb':
            # Cover:
            # * Pdb()
            return True
    return False


def _is_mktemp_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Attribute):
        if node.func.attr == 'mktemp':
            # Cover:
            #  * tempfile.mktemp()
            #  * xxxx.mktemp()
            return True
    if isinstance(node.func, ast.Name):
        if node.func.id == 'mktemp':
            # Cover:
            #  * mktemp()
            return True
    return False


def _is_yaml_unsafe_call(node: ast.Call) -> bool:
    _safe_loaders = ('BaseLoader', 'SafeLoader')
    _unsafe_loaders = ('Loader', 'UnsafeLoader', 'FullLoader')
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'yaml':
        if node.func.attr in ('unsafe_load', 'full_load'):
            # Cover:
            #  * yaml.full_load()
            #  * yaml.unsafe_load()
            return True
        if node.func.attr == 'load':
            for keyword in node.keywords:
                if keyword.arg == 'Loader' and isinstance(keyword.value, ast.Name):
                    if keyword.value.id in _unsafe_loaders:
                        # Cover:
                        #  * yaml.load(x, Loader=Loader)
                        #  * yaml.load(x, Loader=UnsafeLoader)
                        #  * yaml.load(x, Loader=FullLoader)
                        return True
                    if keyword.value.id in _safe_loaders:
                        # Cover:
                        #  * yaml.load(x, Loader=BaseLoader)
                        #  * yaml.load(x, Loader=SafeLoader)
                        return False

            if (
                len(node.args) < 2  # pylint: disable=too-many-boolean-expressions
                or (isinstance(node.args[1], ast.Name) and node.args[1].id in _unsafe_loaders)
                or (
                    isinstance(node.args[1], ast.Attribute)
                    and node.args[1].value.id == "yaml"
                    and node.args[1].attr in _unsafe_loaders
                )
            ):
                # Cover:
                #  * yaml.load(x)
                #  * yaml.load(x, Loader)
                #  * yaml.load(x, UnsafeLoader)
                #  * yaml.load(x, FullLoader)
                #  * yaml.load(x, yaml.Loader)
                #  * yaml.load(x, yaml.UnsafeLoader)
                #  * yaml.load(x, yaml.FullLoader)
                return True

    if isinstance(node.func, ast.Name):
        if node.func.id in ('unsafe_load', 'full_load'):
            # Cover:
            #  * unsafe_load(...)
            #  * full_load(...)
            return True
    return False


# ==============================================================================

_unop = {ast.USub: operator.neg, ast.Not: operator.not_, ast.Invert: operator.inv}
_binop = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitXor: operator.xor,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
}
_chmod_known_mode_values = (
    'S_ISUID',
    'S_ISGID',
    'S_ENFMT',
    'S_ISVTX',
    'S_IREAD',
    'S_IWRITE',
    'S_IEXEC',
    'S_IRWXU',
    'S_IRUSR',
    'S_IWUSR',
    'S_IXUSR',
    'S_IRWXG',
    'S_IRGRP',
    'S_IWGRP',
    'S_IXGRP',
    'S_IRWXO',
    'S_IROTH',
    'S_IWOTH',
    'S_IXOTH',
)


def _chmod_get_mode(node):
    """
    Extract the mode constant of a node.

    Args:
        node: an AST node

    Raises:
        ValueError: if a node is encountered that cannot be processed
    """
    if isinstance(node, ast.Name) and node.id in _chmod_known_mode_values:
        return getattr(stat, node.id)
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.attr in _chmod_known_mode_values
        and node.value.id == 'stat'
    ):
        return getattr(stat, node.attr)
    if isinstance(node, ast.UnaryOp):
        return _unop[type(node.op)](_chmod_get_mode(node.operand))
    if isinstance(node, ast.BinOp):
        return _binop[type(node.op)](_chmod_get_mode(node.left), _chmod_get_mode(node.right))

    raise ValueError(f'Do not know how to process node: {ast.dump(node)}')


def _chmod_has_wx_for_go(node):
    if platform.system() == 'Windows':
        # On Windows, only stat.S_IREAD and stat.S_IWRITE can be used, all other bits are ignored
        return False

    try:
        modes = None
        if len(node.args) > 1:
            modes = _chmod_get_mode(node.args[1])
        elif node.keywords:
            for keyword in node.keywords:
                if keyword.arg == 'mode':
                    modes = _chmod_get_mode(keyword.value)
                    break
    except ValueError:
        return False
    else:
        if modes is None:
            # NB: this would be from invalid code such as `os.chmod("file.txt")`
            raise RuntimeError('Unable to extract `mode` argument from function call!')
        # pylint: disable=no-member
        return bool(modes & (stat.S_IWGRP | stat.S_IXGRP | stat.S_IWOTH | stat.S_IXOTH))


# ==============================================================================


class Visitor(ast.NodeVisitor):
    """AST visitor class for the plugin."""

    os_mkdir_modes_allowed = []
    os_mkdir_modes_msg_arg = ''
    os_mkfifo_modes_allowed = []
    os_mkfifo_modes_msg_arg = ''
    os_mknod_modes_allowed = []
    os_mknod_modes_msg_arg = ''
    os_open_modes_allowed = []
    os_open_modes_msg_arg = ''

    mode_msg_map = {
        SCS112: 'open',
        SCS116: 'mkdir',
        SCS117: 'mkfifo',
        SCS118: 'mknod',
    }

    def _format_mode_msg(self, msg_id):
        """Format a mode message."""
        return self.__class__.format_mode_msg(msg_id)

    @classmethod
    def format_mode_msg(cls, msg_id):
        """Format a mode message."""
        return msg_id.format(getattr(cls, f'os_{cls.mode_msg_map[msg_id]}_modes_msg_arg'))

    def __init__(self) -> None:
        """Initialize a Visitor object."""
        self.errors: List[Tuple[int, int, str]] = []
        self._from_imports: Dict[str, str] = {}

    def visit_Call(self, node: ast.Call) -> None:  # pylint: disable=too-many-branches
        """Visitor method called for ast.Call nodes."""
        if _is_pdb_call(node):
            self.errors.append((node.lineno, node.col_offset, SCS107))
        elif _is_mktemp_call(node):
            self.errors.append((node.lineno, node.col_offset, SCS104))
        elif _is_yaml_unsafe_call(node):
            self.errors.append((node.lineno, node.col_offset, SCS105))
        elif _is_function_call(node, module='jsonpickle', function='decode'):
            self.errors.append((node.lineno, node.col_offset, SCS106))
        elif _is_function_call(node, module='os', function='system'):
            self.errors.append((node.lineno, node.col_offset, SCS102))
        elif _is_os_path_call(node):
            self.errors.append((node.lineno, node.col_offset, SCS100))
        elif _is_function_call(node, module='os', function='popen'):
            self.errors.append((node.lineno, node.col_offset, SCS110))
        elif _is_shell_true_call(node):
            self.errors.append((node.lineno, node.col_offset, SCS103))
        elif _is_builtin_open_for_writing(node):
            self.errors.append((node.lineno, node.col_offset, SCS109))
        elif isinstance(node.func, ast.Name) and (node.func.id in ('eval', 'exec')):
            self.errors.append((node.lineno, node.col_offset, SCS101))
        elif not _is_posix() and _is_function_call(node, module='shlex', function='quote'):
            self.errors.append((node.lineno, node.col_offset, SCS111))
        elif (
            _is_function_call(node, module='os', function='open')
            and self.os_open_modes_allowed
            and not _is_allowed_mode(node, self.os_open_modes_allowed, args_idx=2)
        ):
            self.errors.append((node.lineno, node.col_offset, self._format_mode_msg(SCS112)))
        elif _is_function_call(node, module='pickle', function=('load', 'loads')):
            self.errors.append((node.lineno, node.col_offset, SCS113))
        elif _is_function_call(node, module='marshal', function=('load', 'loads')):
            self.errors.append((node.lineno, node.col_offset, SCS114))
        elif _is_function_call(node, module='shelve', function='open'):
            self.errors.append((node.lineno, node.col_offset, SCS115))
        elif _is_function_call(node, module='os', function='chmod') and _chmod_has_wx_for_go(node):
            self.errors.append((node.lineno, node.col_offset, SCS119))
        elif _is_unix():
            if (
                _is_function_call(node, module='os', function=('mkdir', 'makedirs'))
                and self.os_mkdir_modes_allowed
                and not _is_allowed_mode(node, self.os_mkdir_modes_allowed, args_idx=1)
            ):
                self.errors.append((node.lineno, node.col_offset, self._format_mode_msg(SCS116)))
            elif (
                _is_function_call(node, module='os', function='mkfifo')
                and self.os_mkfifo_modes_allowed
                and not _is_allowed_mode(node, self.os_mkfifo_modes_allowed, args_idx=1)
            ):
                self.errors.append((node.lineno, node.col_offset, self._format_mode_msg(SCS117)))
            elif (
                _is_function_call(node, module='os', function='mknod')
                and self.os_mknod_modes_allowed
                and not _is_allowed_mode(node, self.os_mknod_modes_allowed, args_idx=1)
            ):
                self.errors.append((node.lineno, node.col_offset, self._format_mode_msg(SCS118)))

        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        """Visitor method called for ast.Import nodes."""
        for alias in node.names:
            if alias.name == 'pdb':
                # Cover:
                #  * import pdb
                #  * import pdb as xxx
                self.errors.append((node.lineno, node.col_offset, SCS107))

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visitor method called for ast.ImportFrom nodes."""
        for alias in node.names:
            if (node.module is None and alias.name == 'pdb') or node.module == 'pdb':
                # Cover:
                #  * from pdb import xxx
                self.errors.append((node.lineno, node.col_offset, SCS107))
            elif node.module == 'tempfile' and alias.name == 'mktemp':
                # Cover:
                #  * from tempfile import mktemp
                self.errors.append((node.lineno, node.col_offset, SCS104))
            elif node.module in ('os.path', 'op') and alias.name in ('relpath', 'abspath'):
                # Cover:
                #  * from os.path import relpath, abspath
                #  * import os.path as op; from op import relpath, abspath
                self.errors.append((node.lineno, node.col_offset, SCS100))
            elif (node.module == 'subprocess' and alias.name in ('getoutput', 'getstatusoutput')) or (
                node.module == 'asyncio' and alias.name == 'create_subprocess_shell'
            ):
                # Cover:
                # * from subprocess import getoutput
                # * from subprocess import getstatusoutput
                # * from asyncio import create_subprocess_shell
                self.errors.append((node.lineno, node.col_offset, SCS103))
            elif node.module == 'os' and alias.name == 'system':
                # Cover:
                # * from os import system
                self.errors.append((node.lineno, node.col_offset, SCS102))
            elif node.module == 'os' and alias.name == 'popen':
                # Cover:
                # * from os import popen
                self.errors.append((node.lineno, node.col_offset, SCS110))
            elif not _is_posix() and node.module == 'shlex' and alias.name == 'quote':
                # Cover:
                # * from shlex import quote
                # * from shlex import quote as quoted
                self.errors.append((node.lineno, node.col_offset, SCS111))
            elif node.module == 'pickle' and alias.name in ('load', 'loads'):
                # Cover:
                # * from pickle import load
                # * from pickle import loads as load
                self.errors.append((node.lineno, node.col_offset, SCS113))
            elif node.module == 'marshal' and alias.name in ('load', 'loads'):
                # Cover:
                # * from marshal import load
                # * from marshal import loads as load
                self.errors.append((node.lineno, node.col_offset, SCS114))
            elif node.module == 'shelve' and alias.name == 'open':
                # Cover:
                # * from shelve import open
                self.errors.append((node.lineno, node.col_offset, SCS115))
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        """Visitor method called for ast.With nodes."""
        for item in node.items:
            if isinstance(item.context_expr, ast.Call):
                if _is_builtin_open_for_writing(item.context_expr):
                    self.errors.append((node.lineno, node.col_offset, SCS109))
                elif _is_function_call(item.context_expr, module='os', function='open') and not _is_allowed_mode(
                    item.context_expr, self.os_open_modes_allowed, args_idx=2
                ):
                    self.errors.append((node.lineno, node.col_offset, self._format_mode_msg(SCS112)))
                elif _is_function_call(item.context_expr, module='shelve', function='open'):
                    self.errors.append((node.lineno, node.col_offset, SCS115))

    def visit_Assert(self, node: ast.Assert) -> None:
        """Visitor method called for ast.Assert nodes."""
        self.errors.append((node.lineno, node.col_offset, SCS108))
        self.generic_visit(node)


class Plugin:  # pylint: disable=R0903
    """Plugin class."""

    name = __name__
    version = importlib.metadata.version(__name__)

    def __init__(self, tree: ast.AST):
        """Initialize a Plugin object."""
        self._tree = tree

    @classmethod
    def add_options(cls, option_manager: flake8.options.manager.OptionManager) -> None:
        """Add command line options."""
        options_data = (
            (
                "--os-mkdir-mode",
                {
                    'type': str,
                    'parse_from_config': True,
                    'default': False,
                    'dest': "os_mkdir_mode",
                    'help': "If provided, configure how 'mode' parameter of the os.mkdir() function are handled",
                },
            ),
            (
                "--os-mkfifo-mode",
                {
                    'type': str,
                    'parse_from_config': True,
                    'default': False,
                    'dest': "os_mkfifo_mode",
                    'help': "If provided, configure how 'mode' parameter of the os.mkfifo() function are handled",
                },
            ),
            (
                "--os-mknod-mode",
                {
                    'type': str,
                    'parse_from_config': True,
                    'default': False,
                    'dest': "os_mknod_mode",
                    'help': "If provided, configure how 'mode' parameter of the os.mknod() function are handled",
                },
            ),
            (
                "--os-open-mode",
                {
                    'type': str,
                    'parse_from_config': True,
                    'default': False,
                    'dest': "os_open_mode",
                    'help': "If provided, configure how 'mode' parameter of the os.open() function are handled",
                },
            ),
        )

        if _use_optparse:  # pragma: no cover
            cls.add_options_optparse(option_manager, options_data)
        else:
            cls.add_options_argparse(option_manager, options_data)

    @classmethod
    def add_options_optparse(
        cls, option_manager: flake8.options.manager.OptionManager, options_data
    ) -> None:  # pragma: no cover
        """Add command line options using optparse."""

        def octal_mode_option_callback(option, _, value, parser):
            """Octal mode option callback."""
            setattr(parser.values, f'{option.dest}', _read_octal_mode_option(option.dest, value, _DEFAULT_MAX_MODE))

        callback = {'action': 'callback', 'callback': octal_mode_option_callback}
        for opt_str, kwargs in options_data:
            option_manager.add_option(opt_str, **{**kwargs, **callback})

    @classmethod
    def add_options_argparse(cls, option_manager: flake8.options.manager.OptionManager, options_data) -> None:
        """Add command line options using argparse."""

        class OctalModeAction(argparse.Action):
            """Action class for octal mode options."""

            def __call__(self, parser, namespace, values, option_string=None):
                setattr(namespace, self.dest, _read_octal_mode_option(self.dest, values, _DEFAULT_MAX_MODE))

        action = {'action': OctalModeAction}
        for opt_str, kwargs in options_data:
            option_manager.add_option(opt_str, **{**kwargs, **action})

    @classmethod
    def parse_options(cls, options) -> None:
        """Parse command line options."""

        def _set_mode_option(name, modes):
            if isinstance(modes, int) and modes > 0:
                setattr(Visitor, f'os_{name}_modes_allowed', list(range(0, modes + 1)))
                setattr(Visitor, f'os_{name}_modes_msg_arg', f'0 < mode < {oct(modes)}')
            elif modes:
                setattr(Visitor, f'os_{name}_modes_allowed', modes)
                setattr(Visitor, f'os_{name}_modes_msg_arg', f'mode in {[oct(mode) for mode in modes]}')
            else:
                getattr(Visitor, f'os_{name}_modes_allowed').clear()

        _set_mode_option('mkdir', options.os_mkdir_mode)
        _set_mode_option('mkfifo', options.os_mkfifo_mode)
        _set_mode_option('mknod', options.os_mknod_mode)
        _set_mode_option('open', options.os_open_mode)

    def run(self) -> Generator[Tuple[int, int, str, Type[Any]], None, None]:
        """Entry point for flake8."""
        visitor = Visitor()
        visitor.visit(self._tree)

        for line, col, msg in visitor.errors:
            yield line, col, msg, type(self)
