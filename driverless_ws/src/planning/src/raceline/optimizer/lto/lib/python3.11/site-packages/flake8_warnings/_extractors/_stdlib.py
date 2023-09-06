from typing import Iterator

import astroid

from ._base import Extractor, WarningInfo


MODULES = frozenset({
    # PEP 594: Removing dead batteries
    'aifc',
    'asynchat',
    'asyncore',
    'audioop',
    'cgi',
    'cgitb',
    'chunk',
    'crypt',
    'imghdr',
    'mailcap',
    'msilib',
    'nntplib',
    'nis',
    'ossaudiodev',
    'pipes',
    'smtpd',
    'sndhdr',
    'spwd',
    'sunau',
    'telnetlib',
    'uu',
    'xdrlib',

    # PEP 632, removed in Python 3.12
    'distutils',

    # deprecated but not announced to be removed
    'optparse',
    'tkinter.tix',
    'xml.etree.cElementTree',
})


class StdlibExtractor(Extractor):
    """Extractor for warning messages in docstrings.
    """

    def extract(self, node: astroid.NodeNG) -> Iterator[WarningInfo]:
        if not isinstance(node, astroid.Module):
            return
        qname = node.qname()
        if qname not in MODULES:
            qname = qname.split('.')[0]
        if qname in MODULES:
            yield WarningInfo(
                message=f'stdlib module {qname} is deprecated',
                category=DeprecationWarning,
            )
