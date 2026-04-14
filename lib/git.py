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

    get_recent_commits(n=30, ignore_dir_name=None) -> str
        Return a condensed, token-efficient summary of the last *n* commits
        including per-file numstat diffs.
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


def get_recent_commits(
    n: int = 30,
    ignore_dir_name: Optional[str] = None,
) -> str:
    """Return a condensed, token-efficient summary of the last *n* commits.

    Uses ``git log --pretty --numstat`` to produce a compact history block
    suitable for inclusion in LLM prompts without burning excessive tokens.

    Parameters
    ----------
    n : int
        Maximum number of commits to retrieve (default 30).
    ignore_dir_name : str, optional
        If provided, file-change lines whose filename contains this string
        are silently dropped (useful for excluding AI output artifacts).

    Returns
    -------
    str
        Multi-line plain-text history block, or a parenthesised error
        message if the history cannot be retrieved.
    """
    # Gate on git availability so callers never need to check separately
    if not is_git_available():
        return "(Git history unavailable — git not found)"

    try:
        result = subprocess.run(
            [
                "git", "log",
                # COMMIT_START prefix makes parsing unambiguous even when
                # commit subjects contain tab characters or digits
                "--pretty=format:COMMIT_START %h %s",
                "--numstat",
                "-n", str(n),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "(Git history unavailable — command timed out)"
    except Exception:
        return "(Git history unavailable — unexpected error)"

    if result.returncode != 0:
        return "(Git history unavailable — git log failed)"

    raw = result.stdout.strip()
    if not raw:
        return "(No git history available)"

    # --- Parse raw output into structured commit blocks ---
    # Each commit block starts with a COMMIT_START line, followed by zero or
    # more numstat lines (added\tremoved\tfilename).  Empty lines act as
    # separators between commits.
    commits: list[dict] = []
    current: Optional[dict] = None

    for line in raw.splitlines():
        if line.startswith("COMMIT_START "):
            # Flush previous commit if any
            if current is not None:
                commits.append(current)
            # Parse "COMMIT_START <hash> <subject…>"
            parts = line.split(" ", 2)  # ["COMMIT_START", hash, subject]
            current = {
                "hash": parts[1] if len(parts) > 1 else "???????",
                "subject": parts[2] if len(parts) > 2 else "(no message)",
                "files": [],
                "total_added": 0,
                "total_removed": 0,
            }
        elif line.strip() == "":
            # Blank separator line between commits — skip
            continue
        elif current is not None:
            # Numstat line: "<added>\t<removed>\t<filename>"
            tab_parts = line.split("\t", 2)
            if len(tab_parts) == 3:
                added, removed, filename = tab_parts

                # Skip files belonging to the ignored directory
                if ignore_dir_name and ignore_dir_name in filename:
                    continue

                # Binary files show "-" for both added and removed
                if added == "-" and removed == "-":
                    current["files"].append(f"  {filename} (binary)")
                else:
                    current["files"].append(f"  {filename} +{added} -{removed}")
                    # Accumulate per-commit totals for the summary line
                    try:
                        current["total_added"] += int(added)
                        current["total_removed"] += int(removed)
                    except ValueError:
                        pass

    # Flush the last commit
    if current is not None:
        commits.append(current)

    if not commits:
        return "(No git history available)"

    # --- Build condensed output ---
    lines: list[str] = [f"## Git History (last {n} commits)"]
    for commit in commits:
        n_files = len(commit["files"])
        stats = (f"({n_files} file{'s' if n_files != 1 else ''}, "
                 f"+{commit['total_added']} -{commit['total_removed']})")
        lines.append(f"[{commit['hash']}] {commit['subject']}  {stats}")
        for file_line in commit["files"]:
            lines.append(file_line)

    return "\n".join(lines)
