import ast


class LegacyConstantRewriter(ast.NodeTransformer):
    """
    Transformer that replaces deprecated constant nodes (Str, Num,
    NameConstant, Ellipsis) with Constant.
    """
    def visit_Str(self, node: ast.Str) -> ast.AST:
        const_node = ast.Constant(value=node.s)
        ast.copy_location(const_node, node)
        return const_node

    def visit_Num(self, node: ast.Num) -> ast.AST:
        const_node = ast.Constant(value=node.n)
        ast.copy_location(const_node, node)
        return const_node

    def visit_NameConstant(self, node: ast.NameConstant) -> ast.AST:
        const_node = ast.Constant(value=node.value)
        ast.copy_location(const_node, node)
        return const_node

    def visit_Ellipsis(self, node: ast.Ellipsis) -> ast.AST:
        const_node = ast.Constant(value=...)
        ast.copy_location(const_node, node)
        return const_node
