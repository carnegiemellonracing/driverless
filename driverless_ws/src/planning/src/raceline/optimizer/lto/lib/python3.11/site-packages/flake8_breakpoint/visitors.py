import ast

from flake8_plugin_utils import Visitor

from .errors import BreakpointFound, DebugModuleImportFound

BREAKPOINT_ID = 'breakpoint'
DEBUG_MODULES = {'pdb', 'ipdb', 'pudb'}


class BreakpointVisitor(Visitor):
    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == BREAKPOINT_ID:
            self.error_from_node(BreakpointFound, node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in DEBUG_MODULES:
                self.error_from_node(DebugModuleImportFound, node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in DEBUG_MODULES:
            self.error_from_node(DebugModuleImportFound, node)
