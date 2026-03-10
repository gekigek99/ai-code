"""
lib.patch — parse and apply JSON-based PATCH hunks.

Public API:
    PatchApplicationError — raised when a find string is not found in the target file.
    parse_hunks(patches: list[dict]) -> list[tuple[str, str, str]]
        Validate and extract (comment, find, replace) tuples from JSON patch objects.
    apply_patch(file_path: str, patches: list[dict]) -> bool
        Apply JSON patch hunks to an existing file. Raises PatchApplicationError on not-found.
"""

import os
from typing import Dict, List, Tuple


class PatchApplicationError(Exception):
    """Raised when a PATCH hunk's find text is not found in the target file."""


def parse_hunks(patches: List[Dict]) -> List[Tuple[str, str, str]]:
    """Validate and extract (comment, find, replace) tuples from JSON patch objects.

    Each dict must contain:
      - "comment": str (informational; defaults to "" if missing/invalid)
      - "find":    str, non-empty
      - "replace": str (may be empty — empty means delete the matched text)

    Raises ``ValueError`` on structural problems.
    """
    hunks: List[Tuple[str, str, str]] = []

    for idx, entry in enumerate(patches):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Hunk #{idx + 1}: expected a dict, got {type(entry).__name__}."
            )

        # comment — optional, informational
        comment = entry.get("comment", "")
        if not isinstance(comment, str):
            comment = ""

        # find — required, non-empty string
        find = entry.get("find")
        if find is None or not isinstance(find, str):
            raise ValueError(
                f"Hunk #{idx + 1}: \"find\" is missing or not a string."
            )
        if find == "":
            raise ValueError(f"Hunk #{idx + 1} has empty find text.")

        # replace — required, must be string (empty is allowed)
        replace = entry.get("replace")
        if replace is None or not isinstance(replace, str):
            raise ValueError(
                f"Hunk #{idx + 1}: \"replace\" is missing or not a string."
            )

        hunks.append((comment, find, replace))

    return hunks


def apply_patch(file_path: str, patches: List[Dict]) -> bool:
    """Apply a series of JSON-based SEARCH/REPLACE hunks to an existing file.

    Raises ``PatchApplicationError`` if any hunk's find text is not located
    in the file content — exact match only, no fuzzy fallback.

    Returns True on success, False on file I/O error.
    """

    # --- read -----------------------------------------------------------
    if not os.path.exists(file_path):
        print(f"ERROR: File does not exist (PATCH requires an existing file): {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print(f"ERROR: Failed to read {file_path}: {e}")
        return False

    # --- parse hunks ----------------------------------------------------
    try:
        hunks = parse_hunks(patches)
    except ValueError as e:
        print(f"ERROR: [{file_path}] Invalid patch data: {e}")
        return False

    if not hunks:
        # Empty patches list — nothing to apply, file untouched.
        return True

    # --- apply hunks sequentially ---------------------------------------
    for idx, (comment, find_text, replace_text) in enumerate(hunks, start=1):
        pos = content.find(find_text)

        if pos == -1:
            # Hard error — exact match required, no fallback.
            preview_lines = find_text.split("\n")[:3]
            preview = "\n    ".join(preview_lines)
            label = f"'{comment}'" if comment else f"#{idx}"
            raise PatchApplicationError(
                f"[{file_path}] Hunk {label} — find text not found. "
                f"First 3 lines:\n    {preview}"
            )

        # Warn on multiple matches — still replace first only.
        if content.find(find_text, pos + 1) != -1:
            label = f"'{comment}'" if comment else f"#{idx}"
            print(
                f"INFO: [{file_path}] Hunk {label} matched multiple locations; "
                "replacing first occurrence only."
            )

        content = content.replace(find_text, replace_text, 1)

    # --- write back -----------------------------------------------------
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"ERROR: Failed to write {file_path}: {e}")
        return False

    return True
