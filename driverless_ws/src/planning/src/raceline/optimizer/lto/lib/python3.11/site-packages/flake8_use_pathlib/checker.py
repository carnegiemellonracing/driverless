import ast
import typing
import dataclasses
import functools

import pkg_resources

pkg_name = "flake8-use-pathlib"
pkg_version = pkg_resources.get_distribution(pkg_name).version


@dataclasses.dataclass
class NameResolver(ast.NodeVisitor):
    import_alias: typing.Dict[str, str]
    _name: typing.List[str] = dataclasses.field(init=False, default_factory=list)

    @property
    def name(self) -> str:
        try:
            a = self.import_alias[self._name[-1]]
            self._name[-1] = a
        except (KeyError, IndexError):
            pass
        return ".".join(reversed(self._name))

    def visit_Name(self, node: ast.Name) -> None:
        self._name.append(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        try:
            self._name.append(node.attr)
            self._name.append(node.value.id)  # type: ignore
        except AttributeError:
            self.generic_visit(node)


@dataclasses.dataclass
class PathlibVisitor(ast.NodeVisitor):
    filename: str
    errors: typing.List["Error"] = dataclasses.field(default_factory=list)

    import_alias: typing.Dict[str, str] = dataclasses.field(
        init=False, default_factory=dict
    )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for imp in node.names:
            if imp.asname:
                self.import_alias[imp.asname] = f"{node.module}.{imp.name}"
            else:
                self.import_alias[imp.name] = f"{node.module}.{imp.name}"

    def visit_Import(self, node: ast.Import) -> None:
        for imp in node.names:
            if imp.asname:
                self.import_alias[imp.asname] = imp.name

    def visit_Call(self, node: ast.Call) -> None:
        name_resolver = NameResolver(self.import_alias)
        name_resolver.visit(node.func)

        self.check_for_call_errors(node, name_resolver.name)

    def check_for_call_errors(self, node: ast.AST, name: str) -> None:
        try:
            partial_error = call_errors[name]
            self.errors.append(partial_error(lineno=node.lineno, col=node.col_offset))  # type: ignore
        except KeyError:
            pass


@dataclasses.dataclass
class PathlibChecker:
    name = pkg_name
    version = pkg_version

    tree: ast.AST
    filename: str = "(none)"
    visitor: typing.Type[PathlibVisitor] = dataclasses.field(
        init=False, default=PathlibVisitor
    )

    def run(self) -> typing.Iterable[typing.Tuple[int, int, str, type]]:
        visitor = self.visitor(filename=self.filename)
        visitor.visit(self.tree)
        for error in visitor.errors:
            yield error.as_flake8_tuple()


@dataclasses.dataclass
class Error:
    lineno: int
    col: int
    id: str
    message: str
    vars: typing.Dict[str, typing.Union[str, int]] = dataclasses.field(
        default_factory=dict
    )
    type = PathlibChecker

    def as_flake8_tuple(self) -> typing.Tuple[int, int, str, typing.Type]:
        return (
            self.lineno,
            self.col,
            (f"{self.id} {self.message}").format(**self.vars),
            self.type,
        )


partial_error = functools.partial(functools.partial, Error)
call_errors = {
    "os.path.abspath": partial_error(
        id="PL100",
        message='os.path.abspath("foo") should be replaced by foo_path.resolve()',
    ),
    "os.chmod": partial_error(
        id="PL101",
        message='os.chmod("foo", 0o444) should be replaced by foo_path.chmod(0o444)',
    ),
    "os.mkdir": partial_error(
        id="PL102", message='os.mkdir("foo") should be replaced by foo_path.mkdir()'
    ),
    "os.makedirs": partial_error(
        id="PL103",
        message='os.makedirs("foo/bar") should be replaced by bar_path.mkdir(parents=True)',
    ),
    "os.rename": partial_error(
        id="PL104",
        message='os.rename("foo", "bar") should be replaced by foo_path.rename(Path("bar"))',
    ),
    "os.replace": partial_error(
        id="PL105",
        message='os.replace("foo", "bar") should be replaced by foo_path.replace(Path("bar"))',
    ),
    "os.rmdir": partial_error(
        id="PL106", message='os.rmdir("foo") should be replaced by foo_path.rmdir()'
    ),
    "os.remove": partial_error(
        id="PL107", message='os.remove("foo") should be replaced by foo_path.unlink()'
    ),
    "os.unlink": partial_error(
        id="PL108", message='os.unlink("foo") should be replaced by foo_path.unlink()'
    ),
    "os.getcwd": partial_error(
        id="PL109", message="os.getcwd() should be replaced by Path.cwd()"
    ),
    "os.path.exists": partial_error(
        id="PL110",
        message='os.path.exists("foo") should be replaced by foo_path.exists()',
    ),
    "os.path.expanduser": partial_error(
        id="PL111",
        message='os.path.expanduser("~/foo") should be replaced by foo_path.expanduser()',
    ),
    "os.path.isdir": partial_error(
        id="PL112",
        message='os.path.isdir("foo") should be replaced by foo_path.is_dir()',
    ),
    "os.path.isfile": partial_error(
        id="PL113",
        message='os.path.isfile("foo") should be replaced by foo_path.is_file()',
    ),
    "os.path.islink": partial_error(
        id="PL114",
        message='os.path.islink("foo") should be replaced by foo_path.is_symlink()',
    ),
    "os.readlink": partial_error(
        id="PL115",
        message='os.readlink("foo") should be replaced by foo_path.readlink()',
    ),
    "os.stat": partial_error(
        id="PL116",
        message='os.stat("foo") should be replaced by foo_path.stat() or '
        "foo_path.owner() or foo_path.group()",
    ),
    "os.path.isabs": partial_error(
        id="PL117", message="os.path.isabs should be replaced by foo_path.is_absolute()"
    ),
    "os.path.join": partial_error(
        id="PL118",
        message='os.path.join("foo", "bar") should be replaced by foo_path / "bar"',
    ),
    "os.path.basename": partial_error(
        id="PL119",
        message='os.path.basename("foo/bar") should be replaced by bar_path.name',
    ),
    "os.path.dirname": partial_error(
        id="PL120",
        message='os.path.dirname("foo/bar") should be replaced by bar_path.parent',
    ),
    "os.path.samefile": partial_error(
        id="PL121",
        message='os.path.samefile("foo", "bar") should be replaced by foo_path.samefile(bar_path)',
    ),
    "os.path.splitext": partial_error(
        id="PL122",
        message='os.path.splitext("foo.bar") should be replaced by foo_path.suffix',
    ),
    "open": partial_error(
        id="PL123",
        message='open("foo") should be replaced by Path("foo").open()',
    ),
    "py.path.local": partial_error(
        id="PL124", message="py.path.local is in maintenance mode, use pathlib instead"
    ),
}
