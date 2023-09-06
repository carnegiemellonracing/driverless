import ast
from typing import Callable, List, Optional, Sequence

from .flake_diagnostic import FlakeDiagnostic


# Abstract stuff

class AssertTestVisitor(ast.NodeVisitor):
    def __init__(
        self,
        diagnostic_name: str,
        callback: Callable[[FlakeDiagnostic], None],
        detect_bad_assert_test: Callable[[ast.expr], Optional[str]],
    ):
        self._diagnostic_name = diagnostic_name
        self._callback = callback
        self._detect_bad_assert_test = detect_bad_assert_test

    def visit_Assert(self, node: ast.Assert) -> None:
        message = self._detect_bad_assert_test(node.test)

        if message is None:
            return

        diagnostic = FlakeDiagnostic(
            line=node.lineno,
            col=node.col_offset,
            message="{0} {1}".format(self._diagnostic_name, message),
        )
        self._callback(diagnostic)


Rule = Callable[[ast.Module], Sequence[FlakeDiagnostic]]


def _find_assert(
    diagnostic_name: str,
    detect_bad_assert_test: Callable[[ast.expr], Optional[str]]
) -> Rule:
    def _finder(module: ast.Module) -> Sequence[FlakeDiagnostic]:
        diagnostics: List[FlakeDiagnostic] = []
        visitor = AssertTestVisitor(
            diagnostic_name,
            diagnostics.append,
            detect_bad_assert_test,
        )
        visitor.visit(module)
        return diagnostics
    return _finder


def _detect_assert_test_with_truthy_literal(test: ast.expr) -> Optional[str]:
    if not isinstance(test, ast.Constant):
        return None

    if not test.value:
        return None

    return "`assert` with a truthy value has no effect"


# Rule implementations

def _detect_assert_test_with_0(test: ast.expr) -> Optional[str]:
    if not isinstance(test, ast.Constant):
        return None

    if test.value != 0 or test.value is False:  # NOTE: False == 0
        return None

    return "use `assert False` instead of `assert 0`"


def _detect_assert_test_with_none(test: ast.expr) -> Optional[str]:
    if not isinstance(test, ast.Constant):
        return None

    if test.value is not None:
        return None

    return "use `assert False` instead of `assert None`"


def _is_call_to_format(call: ast.Call) -> bool:
    """
    Check if a call is a call to `str.format`, like '{0}'.format(1).
    """
    if not isinstance(call.func, ast.Attribute):
        return False

    if not isinstance(call.func.value, ast.Constant):
        return False

    if not isinstance(call.func.value.value, str):
        return False

    return call.func.attr == "format"


def _detect_assert_test_with_fstring(test: ast.expr) -> Optional[str]:
    if not isinstance(test, ast.JoinedStr):
        return None

    return "`assert` with an f-string"


def _detect_assert_test_with_format(test: ast.expr) -> Optional[str]:
    if not isinstance(test, ast.Call):
        return None

    if not _is_call_to_format(test):
        return None

    return "`assert` with 'literal'.format(...)"


#: built-in functions `f` such that if `x` is a constant, `f(x)` is a constant
_pure_function_builtins = frozenset({
    "abs", "aiter", "all", "any", "ascii", "bin", "bool",
    "bytearray", "bytes", "callable", "chr", "classmethod",
    "compile", "complex", "dict", "dir", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr",
    "hasattr", "hash", "help", "hex", "id",
    "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "memoryview", "min",
    "oct", "ord", "pow", "property", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "staticmethod",
    "str", "sum", "tuple", "type", "zip",
})


def _is_call_constant(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Name):
        return False

    if call.func.id not in _pure_function_builtins:
        return False

    arg_values = call.args + [kw.value for kw in call.keywords]
    return all(map(_is_constant, arg_values))


def _is_constant(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant):
        return True

    if isinstance(expr, ast.Starred):
        return _is_constant(expr.value)

    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return all(map(_is_constant, expr.elts))

    if isinstance(expr, ast.Dict):
        return all(map(
            _is_constant,
            [k for k in expr.keys if k is not None] + expr.values
        ))

    if isinstance(expr, ast.Compare):
        return all(map(_is_constant, [expr.left, *expr.comparators]))

    if isinstance(expr, ast.IfExp):
        return all(map(_is_constant, [expr.test, expr.body, expr.orelse]))

    if isinstance(expr, ast.BinOp):
        return _is_constant(expr.left) and _is_constant(expr.right)

    if isinstance(expr, ast.Call):
        return _is_call_constant(expr)

    return False


def _detect_assert_with_const_computation(test: ast.expr) -> Optional[str]:
    if isinstance(test, ast.Constant):
        # Already covered by ULA001, ULA002 and ULA003
        return None

    if not _is_constant(test):
        return None

    return "`assert` with constant computation"


# Combining all the rules:

rules: Sequence[Rule] = [
    _find_assert("ULA001", _detect_assert_test_with_truthy_literal),
    _find_assert("ULA002", _detect_assert_test_with_0),
    _find_assert("ULA003", _detect_assert_test_with_none),
    _find_assert("ULA004", _detect_assert_test_with_format),
    _find_assert("ULA005", _detect_assert_test_with_fstring),
    _find_assert("ULA006", _detect_assert_with_const_computation),
]
