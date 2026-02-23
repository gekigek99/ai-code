"""
lib.git — Git utilities.

Public API:
    is_git_available() -> bool
        Check whether git is installed and accessible.

    has_uncommitted_changes(ignore_dir_name=None) -> bool
        Return True if the working tree has uncommitted changes, ignoring
        any lines that mention *ignore_dir_name*.

    commit_changes(message, ignore_dir_name=None) -> bool
        Stage all changes and commit with the given message.

    revert_to_last_commit() -> bool
        Discard all uncommitted changes and untracked files.
"""

import subprocess
from typing import Optional


def is_git_available() -> bool:
    """Check whether git is installed and the current directory is a git repo.

    Returns True if ``git status`` exits with code 0.
    """
    try:
        result = subprocess.run(
            ["git", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


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


def commit_changes(
    message: str,
    ignore_dir_name: Optional[str] = None,
) -> bool:
    """Stage all changes and commit with the given message.

    Parameters
    ----------
    message : str
        Commit message.
    ignore_dir_name : str, optional
        Directory name to check — if the only changes are in this directory,
        the commit is still attempted (it contains log artifacts).

    Returns
    -------
    bool
        True if the commit succeeded, False otherwise.
    """
    try:
        # Stage all changes
        add_result = subprocess.run(
            ["git", "add", "-A"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if add_result.returncode != 0:
            print(f"[git] Failed to stage changes: {add_result.stderr.strip()}")
            return False

        # Check if there's anything to commit after staging
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not status_result.stdout.strip():
            print("[git] Nothing to commit (working tree clean after staging).")
            return True  # Not an error — just nothing changed

        # Commit
        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if commit_result.returncode != 0:
            print(f"[git] Commit failed: {commit_result.stderr.strip()}")
            return False

        print(f"[git] Committed: {message}")
        return True

    except subprocess.TimeoutExpired:
        print("[git] Commit timed out.")
        return False
    except Exception as e:
        print(f"[git] Unexpected error during commit: {e}")
        return False


def revert_to_last_commit() -> bool:
    """Discard all uncommitted changes and remove untracked files.

    Runs:
      1. ``git checkout -- .``   (revert tracked file changes)
      2. ``git clean -fd``       (remove untracked files and directories)

    Returns
    -------
    bool
        True if both commands succeeded, False otherwise.
    """
    try:
        # Revert tracked file changes
        checkout_result = subprocess.run(
            ["git", "checkout", "--", "."],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if checkout_result.returncode != 0:
            print(f"[git] Failed to revert changes: {checkout_result.stderr.strip()}")
            return False

        # Remove untracked files and directories
        clean_result = subprocess.run(
            ["git", "clean", "-fd"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if clean_result.returncode != 0:
            print(f"[git] Failed to clean untracked files: {clean_result.stderr.strip()}")
            return False

        print("[git] Reverted to last commit (discarded all uncommitted changes).")
        return True

    except subprocess.TimeoutExpired:
        print("[git] Revert timed out.")
        return False
    except Exception as e:
        print(f"[git] Unexpected error during revert: {e}")
        return False
