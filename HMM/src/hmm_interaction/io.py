"""I/O helpers for raw driving-simulator recordings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import pandas as pd


FILENAME_PATTERN = re.compile(
    r"^(?P<subject>p\d+)_(?P<condition>av0|av1|av2|hv1)\.1\.asc$",
    re.IGNORECASE,
)


def parse_filename(file_path: str | Path) -> dict[str, Any]:
    """Parse and normalize project metadata encoded in an ASC filename."""
    path = Path(file_path)
    match = FILENAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(
            "Expected filename '<subject_id>_<condition>.1.asc', got "
            f"{path.name!r}."
        )

    subject = match.group("subject")
    condition = match.group("condition")
    normalized_subject = f"P{int(subject[1:]):02d}"
    normalized_condition = condition.upper()
    normalized_name = f"{normalized_subject}_{normalized_condition}.1.asc"
    return {
        "source_file": path.name,
        "subject_id": normalized_subject,
        "condition": normalized_condition,
        "filename_case_normalized": path.name != normalized_name,
    }


def read_asc_file(
    file_path: str | Path,
    usecols: Callable[[str], bool] | None = None,
) -> pd.DataFrame:
    """Read a comma- or tab-delimited ASC recording using its header format."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8", errors="replace") as file:
        header = file.readline()
    separator = "\t" if header.count("\t") > header.count(",") else ","
    return pd.read_csv(path, sep=separator, low_memory=False, usecols=usecols)


def write_csv(frame: pd.DataFrame, file_path: str | Path) -> None:
    """Write a UTF-8 CSV and create its parent directory when needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
