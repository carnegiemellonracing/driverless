from typing import Iterator, List, Optional

import astroid

from ._base import NAMES, Extractor, WarningInfo


BRANCHING = (
    astroid.If,
    astroid.With,
    astroid.TryExcept,
    astroid.TryFinally,
    astroid.Return,
    astroid.Raise,
)


class WarningsExtractor(Extractor):
    """Extractor for `wanings.warn()` invocations.
    """

    def extract(self, node: astroid.NodeNG) -> Iterator[WarningInfo]:
        if isinstance(node, (astroid.FunctionDef, astroid.Module)):
            yield from self._extract_from_body(node.body)

    def _extract_from_body(self, body: List[astroid.NodeNG]) -> Iterator[WarningInfo]:
        for node in body:
            if isinstance(node, astroid.Expr):
                node = node.value
            if isinstance(node, BRANCHING):
                return
            warning = self._get_warning(node)
            if warning is not None:
                yield warning

    def _get_warning(self, node: astroid.NodeNG) -> Optional[WarningInfo]:
        # check if it is a call to `warnings.warn`
        if not isinstance(node, astroid.Call):
            return None
        if node.func.as_string() != 'warnings.warn':
            return None
        return WarningInfo(
            message=self._get_message(node),
            category=NAMES.get(self._get_category(node), Warning),
        )

    @staticmethod
    def _get_message(node: astroid.Call) -> str:
        # extract positional category
        if node.args:
            arg_node = node.args[0]
            if isinstance(arg_node, astroid.Const):
                return str(arg_node.value)

        # extract keyword category
        for kwarg in node.keywords:
            if kwarg.arg != 'message':
                continue
            if isinstance(kwarg.value, astroid.Const):
                return str(kwarg.value.value)
        return ' '.join(node.as_string().split())

    @staticmethod
    def _get_category(node: astroid.Call) -> str:
        # extract positional category
        if len(node.args) > 1:
            arg_node = node.args[1]
            if isinstance(arg_node, astroid.Name):
                return arg_node.name

        # extract keyword category
        for kwarg in node.keywords:
            if kwarg.arg != 'category':
                continue
            arg_node = kwarg.value
            if isinstance(arg_node, astroid.Name):
                return arg_node.name
        return 'UserWarning'
