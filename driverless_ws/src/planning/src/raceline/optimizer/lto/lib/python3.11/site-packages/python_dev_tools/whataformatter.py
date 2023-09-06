"""Formatter module, aggregation of formatters."""
from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess  # noqa: S404
import sys
from contextlib import suppress
from pathlib import Path
from typing import Final, NamedTuple

PYTHON_VERSION_REGEX: Final[str] = "py[0-9]+"


class FormatterConfig(NamedTuple):
    """Configuration of formatting tool."""

    name: str
    path: str
    cli_args: list[str]


_formatters_configs: list[FormatterConfig] = [
    FormatterConfig(
        name="autoflake",
        path="autoflake",
        cli_args=[
            "--in-place",
            "--remove-unused-variables",
            "--remove-all-unused-imports",
        ],
    ),
    FormatterConfig(name="ssort", path="ssort", cli_args=[]),
    FormatterConfig(name="docformatter", path="docformatter", cli_args=["--in-place"]),
    FormatterConfig(name="removestar", path="removestar", cli_args=["-i"]),
    FormatterConfig(name="pybetter", path="pybetter", cli_args=["--exclude", "B004"]),
    FormatterConfig(name="pycln", path="pycln", cli_args=["--all"]),
    FormatterConfig(name="pyupgrade", path="pyupgrade", cli_args=["--py37-plus"]),
    FormatterConfig(
        name="isort",
        path="isort",
        cli_args=[],
    ),  # Should be second to last config
    FormatterConfig(
        name="black",
        path="black",
        cli_args=["--target-version=py37"],
    ),  # Should be the last config
]


def _update_target_version_in_formatters_options(target_version: str) -> None:
    """Update target version in formatters' options.

    Args:
        target_version (str): minimal Python version
    """
    for config in _formatters_configs:
        for index, arg in enumerate(config.cli_args):
            config.cli_args[index] = re.sub(PYTHON_VERSION_REGEX, target_version, arg)


def format_file(filepath: str, target_version: str) -> None:
    """Format the file with known formatters.

    Args:
        filepath (str): path of the file to format
        target_version (str): minimal Python version
    """
    _update_target_version_in_formatters_options(target_version)
    copy_file = f"{filepath}.tmp.co.py"
    for config in _formatters_configs:
        shutil.copyfile(filepath, copy_file)
        with suppress(subprocess.CalledProcessError, subprocess.TimeoutExpired):
            try:
                subprocess.run(  # noqa: S603
                    [config.path, *config.cli_args, copy_file],
                    capture_output=True,
                    timeout=10,
                    encoding="utf-8",
                    check=True,
                )
            except FileNotFoundError as exc:
                if exc.filename == config.path:
                    print(f"Formatter {config.name} not found: {config.path}")
            shutil.copyfile(copy_file, filepath)
        Path(copy_file).unlink()


def diff(orig_file: str, target_version: str) -> None:
    """Print diff between original file and formatted file.

    Args:
        orig_file (str): path of the original file
        target_version (str): minimal Python version
    """
    copy_file = f"{orig_file}.co.py"
    shutil.copyfile(orig_file, copy_file)
    format_file(filepath=copy_file, target_version=target_version)
    print(
        "".join(
            difflib.unified_diff(
                [
                    f"{line}\n"
                    for line in Path(orig_file).read_text(encoding="utf-8").splitlines()
                ],
                [
                    f"{line}\n"
                    for line in Path(copy_file).read_text(encoding="utf-8").splitlines()
                ],
                fromfile=orig_file,
                tofile=orig_file,
            ),
        ),
    )
    Path(copy_file).unlink()


def _udpate_os_path() -> None:
    """Update PATH env variable to find linters."""
    paths = os.environ["PATH"].split(os.pathsep)
    script_path = Path(__file__).resolve()
    paths.insert(0, str(script_path.parent))

    # replace /lib/ with /bin/, and add to PATH
    for parent in reversed(script_path.parents):
        if parent.stem == "lib":
            paths.insert(0, str(parent.parent / "bin"))

    os.environ["PATH"] = os.pathsep.join(paths)


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Python formatter combining existing formatters",
    )
    parser.add_argument(
        "file",
        metavar="FILE",
        type=str,
        help="path of the file to format",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="ignored flag (compatibility with black / VS Code)",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        default=False,
        help="display diff instead of writing file",
    )
    parser.add_argument(
        "--target-version",
        type=str,
        default="py39",
        help="target version for formatting",
    )
    return parser


def main() -> None:
    """Entry point."""
    args = _cli_parser().parse_args()
    if re.search(f"^{PYTHON_VERSION_REGEX}$", args.target_version) is None:
        print(
            f"wrong python version (wanted: ^{PYTHON_VERSION_REGEX}$)",
            args.target_version,
            file=sys.stderr,
        )
        return

    _udpate_os_path()
    if args.diff:
        diff(args.file, target_version=args.target_version)
    else:
        format_file(filepath=args.file, target_version=args.target_version)


if __name__ == "__main__":
    main()
