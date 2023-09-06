from typing import Iterator, Optional

import astroid

from ._base import Extractor, WarningInfo


class DecoratorsExtractor(Extractor):
    """Extractor for deprecation decorators.

    A few examples:
        + https://github.com/tantale/deprecated
            + @deprecated
        + https://github.com/briancurtin/deprecation
            + @deprecation.deprecated
    """

    def extract(self, node: astroid.NodeNG) -> Iterator[WarningInfo]:
        if not isinstance(node, (astroid.FunctionDef, astroid.ClassDef)):
            return
        if not node.decorators:
            return
        assert isinstance(node.decorators, astroid.Decorators)
        dec: astroid.NodeNG
        for dec in node.decorators.nodes:
            warn = self._get_warning(dec)
            if warn is not None:
                yield warn

    def _get_warning(self, node: astroid.NodeNG) -> Optional[WarningInfo]:
        if not isinstance(node, astroid.Call):
            return None
        if 'deprecat' not in node.func.as_string():
            return None
        return WarningInfo(
            message=self._get_message(node),
            category=DeprecationWarning,
        )

    def _get_message(self, node: astroid.Call) -> str:
        for arg in node.args:
            msg = self._get_message_from_node(arg)
            if msg:
                return msg
        for kwarg in node.keywords:
            msg = self._get_message_from_node(kwarg.value)
            if msg:
                return msg
        return ' '.join(node.as_string().split())

    @staticmethod
    def _get_message_from_node(node: astroid.NodeNG) -> Optional[str]:
        if not isinstance(node, astroid.Const):
            return None
        if not isinstance(node.value, str):
            return None
        if ' ' not in node.value:
            return None
        return node.value
