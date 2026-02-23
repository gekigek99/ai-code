"""
lib.git — Git utilities.

Public API:
    has_uncommitted_changes(ignore_dir_name=None) -> bool
        Return True if the working tree has uncommitted changes, ignoring
        any lines that mention *ignore_dir_name* (e.g. the script's own
        output directory).
"""

import subprocess
from typing import Optional


def has_uncommitted_changes(ignore_dir_name: Optional[str] = None) -> bool:
    """Return True if ``git status --porcelain`` reports changes.

    Lines containing *ignore_dir_name* (if provided) are excluded from the
    check so that the script's own log/output directory does not block
    execution.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )

    changes_count = 0
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        if ignore_dir_name and ignore_dir_name in line:
            continue
        changes_count += 1

    return changes_count != 0
