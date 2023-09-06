from __future__ import annotations

import ast
from pathlib import Path

from flake8_future_annotations.checker import FutureAnnotationsChecker


def run_validator_for_test_file(
    filename: str, force_future_annotations: bool
) -> list[tuple[int, int, str, type]]:
    raw_content = Path(filename).read_text(encoding="utf-8")
    tree = ast.parse(raw_content)

    checker = FutureAnnotationsChecker(tree=tree, filename=filename)
    checker.force_future_annotations = force_future_annotations
    return list(checker.run())
