"""
lib.tree — directory-tree building and display.

Public API:
    get_directory_tree(base_dirs, exclude_patterns, files_to_ai=None)
        -> (clean_tree: str, ai_file_listing: str)

    Build a human-readable tree and a flat file listing suitable for LLM
    context.  The ANSI-coloured version is printed to the console; the
    returned ``clean_tree`` has ANSI codes stripped.

    Token estimation is tailored per file type for accuracy:
      - **PDF files**: text is extracted via PyMuPDF and tokens are estimated
        as ``len(extracted_text) // 4``.
      - **Image files** (JPEG, PNG, GIF, WebP): pixel dimensions are read
        from file headers and tokens are estimated using Claude's official
        formula ``(width * height) / 750``, with automatic downscaling to
        1568 px max edge to match Claude's internal resizing.
      - **All other files**: ``file_size_bytes / 4`` (equivalently
        ``size_kb * 256``).
"""

import os
from typing import List, Optional, Set, Tuple

from lib.files import FileData, is_excluded, is_file_bin
from lib.images import get_image_dimensions, estimate_image_tokens
from lib.pdf import is_file_pdf, extract_text_from_pdf
from lib.utils import COLOR_GREEN, COLOR_RESET, warn

# Supported image extensions for pixel-based token estimation in the tree.
# Matches the formats supported by Claude's vision API.
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


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

    Token estimation is per-type for accuracy:
      - PDF: ``len(extracted_text) // 4``
      - Image (JPEG/PNG/GIF/WebP): ``(width * height) / 750`` (max 1568 px edge)
      - Other: ``int(size_kb * 256)`` ≈ ``file_size_bytes / 4``

    If a file is already in *files_to_ai* (content was read during source
    discovery), its ``ai_data_tokens`` is reused for consistency.

    Returns
    -------
    (clean_tree, ai_file_listing) : tuple[str, str]
        *clean_tree* — ANSI-free, human-readable tree string.
        *ai_file_listing* — one ``./relative/path`` per line for LLM consumption,
        with file size (KB) and token estimates for every discovered file.
    """
    base_dirs = _resolve_tree_dirs(base_dirs)

    print("Building directory tree...")
    project_root = os.path.abspath(os.getcwd())

    display_lines: List[str] = []
    clean_lines: List[str] = []

    # Each entry is (relative_path, size_kb, est_tokens).
    # Token estimates are per-type: PDF→extracted text, image→pixel dims,
    # other→file_size/4.  If a file was already processed in files_to_ai,
    # its ai_data_tokens is reused directly.
    ai_file_entries: List[Tuple[str, float, int]] = []

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
            is_shared = os.path.normcase(os.path.abspath(abs_path)) in files_to_ai_norm

            if os.path.isfile(abs_path):
                # Compute file size — used for both display and token estimation
                try:
                    size_kb = os.path.getsize(abs_path) / 1024
                except Exception:
                    size_kb = 0.0

                # --- Per-type token estimation ---
                # 1. If file already processed in files_to_ai, reuse its token count
                # 2. PDF: extract text and estimate from extracted text length
                # 3. Image: read pixel dimensions, use (w*h)/750
                # 4. Other: file_size_bytes / 4
                norm_abs = os.path.normcase(os.path.abspath(abs_path))
                if norm_abs in files_to_ai_norm:
                    est_tokens = files_to_ai_norm[norm_abs].ai_data_tokens
                else:
                    ext_lower = os.path.splitext(abs_path)[1].lower()
                    if ext_lower == ".pdf" and is_file_pdf(abs_path):
                        try:
                            pdf_text = extract_text_from_pdf(abs_path)
                            est_tokens = max(1, len(pdf_text) // 4)
                        except Exception:
                            est_tokens = int(size_kb * 256)  # fallback
                    elif ext_lower in _IMAGE_EXTENSIONS:
                        dims = get_image_dimensions(abs_path)
                        if dims is not None:
                            est_tokens = estimate_image_tokens(dims[0], dims[1])
                        else:
                            est_tokens = int(size_kb * 256)  # fallback
                    else:
                        est_tokens = int(size_kb * 256)

                # Collect for AI file listing — every discovered file gets
                # an entry with size and token estimate.
                rel_to_root = "./" + os.path.relpath(abs_path, project_root).replace("\\", "/")
                ai_file_entries.append((rel_to_root, size_kb, est_tokens))

                pad = base_name.ljust(20)[len(base_name):]
                token_str = f"~{est_tokens:5} tokens"

                # Shared files are highlighted green in terminal output;
                # all files show both KB and token estimate.
                if is_shared:
                    display_entry = (
                        f"{COLOR_GREEN}{base_name} {pad} "
                        f"[{size_kb:5.1f} KB | {token_str}]{COLOR_RESET}"
                    )
                    clean_entry = (
                        f"{base_name} {pad} "
                        f"[{size_kb:5.1f} KB | {token_str}]"
                    )
                else:
                    display_entry = f"{base_name} {pad} [{size_kb:5.1f} KB | {token_str}]"
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

    # Flat AI file listing with file size (KB) and per-type token estimates
    # for ALL discovered files.  Token estimates are computed per file type
    # (PDF→extracted text, image→pixel dims, other→file_size/4) for accuracy.
    ai_listing_parts = ["Project file structure:"]
    for file_path, size_kb, est_tokens in ai_file_entries:
        ai_listing_parts.append(f"{file_path}  ({size_kb:.1f} KB | ~{est_tokens} tokens)")
    ai_file_listing = "\n".join(ai_listing_parts)

    return clean_tree, ai_file_listing
