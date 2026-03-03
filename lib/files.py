"""
lib.files — file discovery, reading, and exclusion logic.

Public API:
    FileData            — dataclass representing a single discovered file.
    is_excluded(path, exclude_patterns, base_dir=None) -> bool
    is_file_bin(path) -> bool
    add_source(files_to_ai, source_paths, exclude_patterns,
               ai_shared_file_types=None) -> (List[FileData], Set[str])
"""

import os
import fnmatch
from dataclasses import dataclass
from typing import Any, List, Set, Tuple, Optional

from lib.utils import warn
from lib.pdf import is_file_pdf, extract_text_from_pdf


@dataclass
class FileData:
    """Container for a single file's metadata and (possibly converted) content."""
    file_type: str        # "text" | "bin" | "pdf" | "image"
    path_abs: str         # Absolute path on disk
    path_rel: str         # Relative path for display / AI context
    extension: str        # File extension including the dot
    ai_share: bool        # Whether this file's content should be sent to the LLM

    data: Any             # Raw file data (str for text, bytes for binary)
    data_type: str        # "str" | "byte" | "binary"
    data_size: int        # Size in bytes (or characters for text)
    media_type: str       # MIME-like type hint ("text", "pdf", "image/jpeg", etc.)

    ai_interpretable: bool       # Whether the LLM can usefully read this data
    ai_data_converted: str       # Converted representation for the LLM
    ai_data_converted_type: str  # "str" | "base64"
    ai_data_tokens: int          # Rough token estimate (~chars/4)


# ──────────────────────────────────────────────────────────────────────────────
# Exclusion logic
# ──────────────────────────────────────────────────────────────────────────────

def _path_matches_pattern(path: str, pattern: str) -> bool:
    """Check if *path* matches a specific glob *pattern*, supporting
    multi-level (containing ``/``) patterns."""
    norm_path = path.replace("\\", "/")
    norm_pattern = pattern.replace("\\", "/").rstrip("/")

    if "/" in norm_pattern:
        # Multi-level pattern — check if it appears as a contiguous segment
        path_parts = norm_path.split("/")
        pattern_parts = norm_pattern.split("/")
        for i in range(len(path_parts) - len(pattern_parts) + 1):
            match = True
            for j, pp in enumerate(pattern_parts):
                if not fnmatch.fnmatch(path_parts[i + j], pp):
                    match = False
                    break
            if match:
                return True
        return False
    else:
        # Single-level pattern — match against each path component
        for part in norm_path.split("/"):
            if fnmatch.fnmatch(part, norm_pattern):
                return True
        return False


def is_excluded(
    path: str,
    exclude_patterns: List[str],
    base_dir: Optional[str] = None,
) -> bool:
    """Check if *path* matches any pattern in *exclude_patterns*.

    Parameters
    ----------
    path : str
        File or directory path to test.
    exclude_patterns : list[str]
        Glob patterns (may contain ``/`` for multi-level matching).
    base_dir : str, optional
        If provided, relative-path matching is computed from this directory.
    """
    norm_path = os.path.abspath(path).replace("\\", "/")
    base_name = os.path.basename(path).replace("\\", "/")

    # Compute relative path for matching
    if base_dir:
        try:
            rel_path = os.path.relpath(path, base_dir).replace("\\", "/")
        except ValueError:
            rel_path = norm_path
    else:
        try:
            rel_path = os.path.relpath(path).replace("\\", "/")
        except ValueError:
            rel_path = norm_path

    for pattern in exclude_patterns:
        pattern = pattern.replace("\\", "/")
        is_dir_pattern = pattern.endswith("/")
        norm_pattern = pattern.rstrip("/")

        if "/" in norm_pattern:
            # Multi-level pattern — check against relative and absolute paths
            if _path_matches_pattern(rel_path, pattern):
                return True
            if _path_matches_pattern(norm_path, pattern):
                return True
        else:
            # Single-level pattern

            # Dot-prefixed wildcard patterns (e.g. .env*)
            if norm_pattern.startswith(".") and "*" in norm_pattern:
                if fnmatch.fnmatch(base_name, norm_pattern):
                    return True

            # Check base name directly
            if fnmatch.fnmatch(base_name, norm_pattern):
                return True

            # For directories, check if any parent component matches
            if os.path.isdir(path) or is_dir_pattern:
                for part in rel_path.split("/"):
                    if fnmatch.fnmatch(part, norm_pattern):
                        return True

            # For files inside excluded directories
            if not os.path.isdir(path):
                parts = rel_path.split("/")
                for i in range(len(parts) - 1):  # exclude filename
                    if fnmatch.fnmatch(parts[i], norm_pattern):
                        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Binary detection
# ──────────────────────────────────────────────────────────────────────────────

def is_file_bin(path: str) -> bool:
    """Return True if *path* appears to be a binary (non-text) file.

    Reads the first 1 KiB and checks for null bytes — a simple but effective
    heuristic that covers the vast majority of cases.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\0" in chunk
    except Exception:
        return True  # treat as binary if unreadable


# ──────────────────────────────────────────────────────────────────────────────
# Source discovery and reading
# ──────────────────────────────────────────────────────────────────────────────

def add_source(
    files_to_ai: List[FileData],
    source_paths: List[str],
    exclude_patterns: List[str],
    ai_shared_file_types: Optional[List[str]] = None,
) -> Tuple[List[FileData], Set[str]]:
    """Walk *source_paths*, read every non-excluded file, and append FileData
    entries to *files_to_ai*.

    Silently skips any None or non-string entries in *source_paths* — these
    can arise from YAML list items like ``- # comment`` which parse as None.

    Returns
    -------
    (files_to_ai, abs_file_paths) : tuple
        *files_to_ai* — the same list, mutated with new entries.
        *abs_file_paths* — normalised absolute paths of all discovered files
        (used later to tag "original vs new" during apply).
    """
    if ai_shared_file_types is None:
        ai_shared_file_types = []

    print("\nGathering source files...")
    abs_file_paths: Set[str] = set()

    for s in source_paths:
        # Guard against None or non-string entries in source_paths.
        # YAML ``- # comment`` lines parse as None list items; other callers
        # (e.g. stepize step["source"]) may also produce non-string entries.
        if s is None or not isinstance(s, str) or not s.strip():
            warn(f"WARNING: Skipping invalid source entry: {s!r}")
            continue

        abs_entry = os.path.abspath(s)
        if not os.path.exists(abs_entry):
            warn(f"WARNING: Source entry not found: {s}")
            continue

        # Single file
        if os.path.isfile(abs_entry):
            if not is_excluded(abs_entry, exclude_patterns):
                abs_file_paths.add(abs_entry)
            else:
                print(f"Excluding file: {s}")
            continue

        # Directory walk
        if os.path.isdir(abs_entry):
            print(f"Scanning directory: {abs_entry}")
            for root, dirnames, filenames in os.walk(abs_entry):
                # Filter directories in-place so os.walk skips them
                dirs_to_check = dirnames[:]
                dirnames.clear()
                for dirname in dirs_to_check:
                    dirpath = os.path.join(root, dirname)
                    if not is_excluded(dirpath, exclude_patterns, abs_entry):
                        dirnames.append(dirname)
                # Files
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if not is_excluded(filepath, exclude_patterns, abs_entry):
                        abs_file_paths.add(filepath)
        else:
            print(f"Skipping unknown source: {s}")

    # Normalise and sort for deterministic ordering
    abs_file_paths = set(
        os.path.normcase(os.path.abspath(p)) for p in sorted(abs_file_paths)
    )

    print(f"Discovered {len(abs_file_paths)} source file(s)")
    print("\nReading files...")

    for abs_path in abs_file_paths:
        rel_path = f".\\{os.path.relpath(abs_path, os.getcwd())}"

        if not os.path.isfile(abs_path):
            print(f"Skipped: {rel_path} (not found)")
            continue

        _, extension = os.path.splitext(abs_path)

        # ── Binary files ─────────────────────────────────────────────────────
        if is_file_bin(abs_path):
            with open(abs_path, "rb") as f:
                file_data = f.read()

            if is_file_pdf(abs_path):
                file_data_text = extract_text_from_pdf(abs_path)
                files_to_ai.append(
                    FileData(
                        file_type="pdf",
                        path_abs=abs_path,
                        path_rel=rel_path,
                        extension=extension,
                        ai_share="pdf" in ai_shared_file_types,
                        data=file_data,
                        data_type="byte",
                        data_size=len(file_data),
                        media_type="pdf",
                        ai_interpretable=True,
                        ai_data_converted=file_data_text,
                        ai_data_converted_type="str",
                        ai_data_tokens=len(file_data_text) // 4,
                    )
                )
            else:
                files_to_ai.append(
                    FileData(
                        file_type="bin",
                        path_abs=abs_path,
                        path_rel=rel_path,
                        extension=extension,
                        ai_share=True,
                        data=file_data,
                        data_type="byte",
                        data_size=len(file_data),
                        media_type="pdf",
                        ai_interpretable=False,
                        ai_data_converted="[BINARY CONTENT]",
                        ai_data_converted_type="str",
                        ai_data_tokens=20,
                    )
                )
            continue

        # ── Text files ───────────────────────────────────────────────────────
        encoding_used = "utf-8"
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                file_data_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(abs_path, "r", encoding="cp1252") as f:
                    file_data_text = f.read()
                encoding_used = "cp1252"
            except UnicodeDecodeError:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    file_data_text = f.read()
                encoding_used = "utf-8 (errors=replace)"

        files_to_ai.append(
            FileData(
                file_type="text",
                path_abs=abs_path,
                path_rel=rel_path,
                extension=extension,
                ai_share=True,
                data=file_data_text,
                data_type="str",
                data_size=len(file_data_text),
                media_type="text",
                ai_interpretable=True,
                ai_data_converted=file_data_text,
                ai_data_converted_type="str",
                ai_data_tokens=len(file_data_text) // 4,
            )
        )

        if encoding_used != "utf-8":
            warn(f"Decoded {rel_path} using {encoding_used}")

    return files_to_ai, abs_file_paths
