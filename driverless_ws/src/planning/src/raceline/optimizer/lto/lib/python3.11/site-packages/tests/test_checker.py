import configparser
from pathlib import Path

import pytest

from flake8_future_annotations.checker import FutureAnnotationsChecker
from tests.conftest import run_validator_for_test_file

ALL_TEST_FILES = [
    str(filepath)
    for filepath in Path("tests/test_files").glob("*.py")
    if "__init__.py" not in str(filepath)
]


def test_version() -> None:
    config = configparser.ConfigParser()
    config.read("setup.cfg")
    assert config["metadata"]["version"] == FutureAnnotationsChecker.version


@pytest.mark.parametrize("filepath", ALL_TEST_FILES)
def test_ok_cases_produces_no_errors(filepath: str) -> None:
    if "ok" in filepath:
        errors = run_validator_for_test_file(str(filepath), False)

        assert len(errors) == 0, (str(filepath), errors)


@pytest.mark.parametrize("filepath", ALL_TEST_FILES)
def test_file_missing_future_import(filepath: str) -> None:
    if "ok" not in filepath:
        errors = run_validator_for_test_file(str(filepath), False)

        assert len(errors) == 1, (str(filepath), errors)
        assert errors[0][2][:5] == "FA100", (str(filepath), "error code")


@pytest.mark.parametrize("filepath", ALL_TEST_FILES)
def test_ok_cases_produces_no_errors_with_force_future_annotations(
    filepath: str,
) -> None:
    if "uses_future" in filepath:
        errors = run_validator_for_test_file(str(filepath), True)

        assert len(errors) == 0, (str(filepath), errors)


@pytest.mark.parametrize("filepath", ALL_TEST_FILES)
def test_file_missing_future_import_with_force_future_annotations(
    filepath: str,
) -> None:
    if "uses_future" not in filepath:
        errors = run_validator_for_test_file(str(filepath), True)
        expected_code = "FA101" if "ok" in filepath else "FA100"

        assert len(errors) == 1, (str(filepath), errors)
        assert errors[0][2][:5] == expected_code, (str(filepath), "error code")
