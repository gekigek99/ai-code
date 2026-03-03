"""
lib.tree — directory-tree building and display.

Public API:
    get_directory_tree(base_dirs, exclude_patterns, files_to_ai=None)
        -> (clean_tree: str, ai_file_listing: str)

    Build a human-readable tree and a flat file listing suitable for LLM
    context.  The ANSI-coloured version is printed to the console; the
    returned ``clean_tree`` has ANSI codes stripped.

    The ``ai_file_listing`` includes file size (KB) for all files and
    token estimates for files shared with the LLM, so Claude can gauge
    the relative size and importance of each file.
"""

import os
from typing import List, Optional, Set, Tuple

from lib.files import FileData, is_excluded
from lib.utils import COLOR_GREEN, COLOR_RESET, warn


def _resolve_tree_dirs(raw_dirs: List[str]) -> List[str]:
    """Resolve a list of source entries (files or directories) into a
    deduplicated list of *directory* paths suitable for tree walking.

    - Existing directory → kept as-is.
    - Existing file → replaced with its parent directory.
    - Non-existent → parent used if it exists; otherwise warned and skipped.
    """
    seen: Set[str] = set()
    result: List[str] = []

    for entry in raw_dirs:
        abs_entry = os.path.abspath(entry)

        if os.path.isdir(abs_entry):
            norm = os.path.normcase(abs_entry)
            if norm not in seen:
                seen.add(norm)
                result.append(abs_entry)

        elif os.path.isfile(abs_entry):
            parent = os.path.dirname(abs_entry)
            norm = os.path.normcase(parent)
            if norm not in seen:
                seen.add(norm)
                result.append(parent)

        else:
            parent = os.path.dirname(abs_entry)
            if os.path.isdir(parent):
                norm = os.path.normcase(parent)
                if norm not in seen:
                    seen.add(norm)
                    result.append(parent)
            else:
                warn(f"WARNING: tree_dirs entry not found and parent doesn't exist: {entry}")

    return result


def get_directory_tree(
    base_dirs: List[str],
    exclude_patterns: List[str],
    files_to_ai: Optional[List[FileData]] = None,
) -> Tuple[str, str]:
    """Build a directory tree for the given *base_dirs*.

    Returns
    -------
    (clean_tree, ai_file_listing) : tuple[str, str]
        *clean_tree* — ANSI-free, human-readable tree string.
        *ai_file_listing* — one ``./relative/path`` per line for LLM consumption,
        with file size (KB) for all files and token estimates appended for
        files included in the AI context.
    """
    base_dirs = _resolve_tree_dirs(base_dirs)

    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())

    display_lines: List[str] = []
    clean_lines: List[str] = []

    # Each entry is (relative_path, token_estimate_or_None, size_kb).
    # token_estimate is populated only for files present in files_to_ai
    # (i.e. files whose content is shared with the LLM).
    # size_kb is always populated for all discovered files.
    ai_file_entries: List[Tuple[str, Optional[int], float]] = []

    files_to_ai = files_to_ai or []
    files_to_ai_norm = {
        os.path.normcase(os.path.abspath(f.path_abs)): f
        for f in files_to_ai
    }

    def _is_nested(child: str, parents: List[str]) -> bool:
        child = os.path.abspath(child)
        for parent in parents:
            parent = os.path.abspath(parent)
            try:
                if os.path.commonpath([child, parent]) == parent and child != parent:
                    return True
            except ValueError:
                continue
        return False

    def _walk_dir(path: str, prefix: str = "", base_dir: Optional[str] = None) -> None:
        if is_excluded(path, exclude_patterns, base_dir or path):
            return
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            err_line = f"{prefix}[Error reading {path}]: {e}"
            display_lines.append(err_line)
            clean_lines.append(err_line)
            return

        filtered_entries = [
            entry for entry in entries
            if not is_excluded(os.path.join(path, entry), exclude_patterns, base_dir or path)
        ]

        for i, entry in enumerate(filtered_entries):
            abs_path = os.path.join(path, entry)
            is_last = i == len(filtered_entries) - 1
            connector = "└── " if is_last else "├── "
            base_name = os.path.basename(abs_path)
            filedata = files_to_ai_norm.get(os.path.normcase(os.path.abspath(abs_path)))

            if os.path.isfile(abs_path):
                # Compute file size (used in both tree display and AI listing)
                try:
                    size_kb = os.path.getsize(abs_path) / 1024
                except Exception:
                    size_kb = 0.0

                # Collect relative path, token estimate, and KB for AI listing.
                # Every discovered file gets its KB size recorded; only files
                # in files_to_ai get a token estimate.
                rel_to_root = "./" + os.path.relpath(abs_path, project_root).replace("\\", "/")
                token_est = None
                if filedata:
                    token_est = getattr(filedata, "ai_data_tokens", None)
                ai_file_entries.append((rel_to_root, token_est, size_kb))

                pad = base_name.ljust(20)[len(base_name):]

                if filedata:
                    token_str = f"~{token_est:5.0f} tokens" if token_est is not None else "~N/A tokens"
                    display_entry = (
                        f"{COLOR_GREEN}{base_name} {pad} "
                        f"[{size_kb:5.1f} KB | {token_str}]{COLOR_RESET}"
                    )
                    clean_entry = (
                        f"{base_name} {pad} "
                        f"[{size_kb:5.1f} KB | {token_str}]"
                    )
                else:
                    display_entry = f"{base_name} {pad} [{size_kb:5.1f} KB ]"
                    clean_entry = display_entry

                display_lines.append(f"{prefix}{connector}{display_entry}")
                clean_lines.append(f"{prefix}{connector}{clean_entry}")
            else:
                display_lines.append(f"{prefix}{connector}{entry}")
                clean_lines.append(f"{prefix}{connector}{entry}")

            if os.path.isdir(abs_path):
                extension = "    " if is_last else "│   "
                _walk_dir(abs_path, prefix + extension, base_dir or path)

    # Normalise and validate base directories
    abs_dirs = [os.path.abspath(d) for d in base_dirs]
    missing_dirs = [d for d in abs_dirs if not os.path.isdir(d)]

    if missing_dirs:
        err_header = "\nERROR: The following directory(ies) do not exist:"
        display_lines.append(err_header)
        clean_lines.append(err_header)
        for d in missing_dirs:
            err_entry = f" - {os.path.relpath(d, start=project_root)}"
            display_lines.append(err_entry)
            clean_lines.append(err_entry)
        display_tree = "\n".join(display_lines)
        clean_tree = "\n".join(clean_lines)
        print(display_tree)
        return clean_tree, ""

    # Remove nested entries to avoid duplicate printing
    roots_to_print: List[str] = []
    for d in abs_dirs:
        if not _is_nested(d, roots_to_print):
            roots_to_print.append(d)

    for root_dir in roots_to_print:
        if is_excluded(root_dir, exclude_patterns):
            continue
        rel_name = os.path.relpath(root_dir, start=project_root)
        header = f"\nDirectory tree structure for {rel_name}/"
        display_lines.append(header)
        clean_lines.append(header)
        _walk_dir(root_dir, base_dir=root_dir)

    # Console output (ANSI-coloured)
    print("\n".join(display_lines))

    # Clean tree for export (no ANSI)
    clean_tree = "\n".join(clean_lines)

    # Flat AI file listing with file size (KB) for all files and token
    # estimates for files shared with the LLM.  This gives Claude both
    # the physical file size and the token cost of shared files, matching
    # the information shown in the console tree display.
    ai_listing_parts = ["Project file structure:"]
    for file_path, token_est, size_kb in ai_file_entries:
        if token_est is not None:
            # File is shared with Claude — show KB and token estimate
            ai_listing_parts.append(f"{file_path}  ({size_kb:.1f} KB | ~{token_est} tokens)")
        else:
            # File not shared — show KB only so Claude knows it exists and its size
            ai_listing_parts.append(f"{file_path}  ({size_kb:.1f} KB)")
    ai_file_listing = "\n".join(ai_listing_parts)

    return clean_tree, ai_file_listing
