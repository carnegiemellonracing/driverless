import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterator, List, NoReturn, TextIO

from ._finder import WarningFinder


def get_paths(path: Path) -> Iterator[Path]:
    """Recursively yields python files.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.is_file():
        if path.suffix == '.py':
            yield path
        return
    for subpath in path.iterdir():
        if subpath.name[0] == '.':
            continue
        if subpath.name == '__pycache__':
            continue
        yield from get_paths(subpath)


def main(argv: List[str], stream: TextIO) -> int:
    parser = ArgumentParser()
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args(argv)
    found_warnings = 0
    for root in args.paths:
        for path in get_paths(Path(root)):
            path_shown = False
            finder = WarningFinder.from_path(path)
            for warning in finder.find():
                if not path_shown:
                    print(path, file=stream)
                    path_shown = True
                print(' ', warning, file=stream)
                found_warnings += 1
    return min(found_warnings, 100)


def entrypoint() -> NoReturn:
    sys.exit(main(sys.argv[1:], sys.stdout))
