"""
lib.patch — parse and apply SEARCH/REPLACE hunks from a [PATCH] block.

Public API:
    apply_patch(file_path: str, patch_content: str) -> bool
        Read the existing file at *file_path*, parse *patch_content* into
        ordered SEARCH/REPLACE hunks, apply each sequentially, and write the
        result back.  Returns True on success, False on file-I/O failure.

    parse_hunks(patch_content: str) -> list[tuple[str, str]]
        Extract (search_text, replace_text) pairs from conflict-marker
        delimited hunks inside *patch_content*.
"""

import os
import re
from typing import List, Tuple

# Matches a single SEARCH/REPLACE hunk delimited by git-conflict-style markers.
# DOTALL so '.' covers newlines inside each section.
_HUNK_RE = re.compile(
    rf"{'<'*7} SEARCH\n"
    rf"(?P<search>.*?)"
    rf"=======\n"
    rf"(?P<replace>.*?)"
    rf"{'>'*7} REPLACE",
    re.DOTALL,
)


def _normalize_ws(text: str) -> str:
    """Collapse intra-line whitespace runs to single spaces and strip trailing
    whitespace per line.  Used for the fuzzy-match fallback."""
    lines = text.split("\n")
    return "\n".join(re.sub(r"[ \t]+", " ", line).rstrip() for line in lines)


def _find_original_span(original: str, norm_search: str) -> Tuple[int, int]:
    """Return (start, end) indices inside *original* whose whitespace-normalised
    form matches *norm_search*, or (-1, -1) if not found.

    We normalise *original* line-by-line and then walk through it looking for
    a contiguous region whose normalised form equals *norm_search*.
    """
    orig_lines = original.split("\n")
    search_lines = norm_search.split("\n")
    search_len = len(search_lines)

    if search_len == 0:
        return (-1, -1)

    norm_orig_lines = [re.sub(r"[ \t]+", " ", l).rstrip() for l in orig_lines]

    for i in range(len(norm_orig_lines) - search_len + 1):
        if norm_orig_lines[i : i + search_len] == search_lines:
            # Map line indices back to character offsets in the original string.
            start = sum(len(orig_lines[j]) + 1 for j in range(i))  # +1 for '\n'
            end = start + sum(len(orig_lines[i + j]) + 1 for j in range(search_len))
            # end includes the trailing '\n' of the last matched line; but the
            # search text itself may or may not end with '\n', so trim if the
            # slice overshoots the string length.
            if end > len(original):
                end = len(original)
            return (start, end)

    return (-1, -1)


# ------------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------------

def parse_hunks(patch_content: str) -> List[Tuple[str, str]]:
    f"""Parse *patch_content* into an ordered list of (search, replace) tuples.

    Each hunk is delimited by::

        {'<'*7} SEARCH
        ...
        =======
        ...
        {'>'*7} REPLACE

    One trailing newline is stripped from both search and replace sections
    (artifact of the marker line itself) while all internal whitespace and
    newlines are preserved exactly.
    """
    hunks: List[Tuple[str, str]] = []

    for m in _HUNK_RE.finditer(patch_content):
        search = m.group("search")
        replace = m.group("replace")

        # Strip exactly one trailing newline (marker-line artifact).
        if search.endswith("\n"):
            search = search[:-1]
        if replace.endswith("\n"):
            replace = replace[:-1]

        hunks.append((search, replace))

    if not hunks:
        print("WARNING: No valid SEARCH/REPLACE hunks found in patch content.")

    return hunks


def apply_patch(file_path: str, patch_content: str) -> bool:
    """Apply a series of SEARCH/REPLACE hunks to an existing file.

    Returns True when file I/O succeeded (even if individual hunks were
    skipped), False on any file-level error.
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
    hunks = parse_hunks(patch_content)
    if not hunks:
        # Nothing to apply — file is untouched; still a "success" for I/O.
        return True

    # --- apply hunks sequentially ---------------------------------------
    for idx, (search, replace) in enumerate(hunks, start=1):
        # Edge case: empty search text is meaningless.
        if search == "":
            print(f"WARNING: [{file_path}] Hunk #{idx} has empty SEARCH section — skipped.")
            continue

        # --- exact match ------------------------------------------------
        pos = content.find(search)
        if pos != -1:
            # Info note when multiple occurrences exist.
            if content.find(search, pos + 1) != -1:
                print(
                    f"INFO: [{file_path}] Hunk #{idx} matched multiple locations; "
                    "replacing first occurrence only."
                )
            content = content.replace(search, replace, 1)
            continue

        # --- whitespace-normalised fallback -----------------------------
        norm_search = _normalize_ws(search)
        start, end = _find_original_span(content, norm_search)

        if start != -1:
            if _normalize_ws(content).count(norm_search) > 1:
                print(
                    f"INFO: [{file_path}] Hunk #{idx} matched multiple locations "
                    "(whitespace-normalised); replacing first occurrence only."
                )
            content = content[:start] + replace + content[end:]
            continue

        # --- not found at all -------------------------------------------
        preview_lines = search.split("\n")[:3]
        preview = "\n    ".join(preview_lines)
        print(
            f"WARNING: [{file_path}] Hunk #{idx} — search text not found. "
            f"Hunk SKIPPED.\n"
            f"  First 3 lines of search text:\n    {preview}"
        )

    # --- write back -----------------------------------------------------
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        print(f"ERROR: Failed to write {file_path}: {e}")
        return False

    return True
