"""
lib.apply — apply file operations (write, move, delete, patch) from parsed response blocks.

Public API:
    claude_data_to_file(text_data, abs_file_paths=None) -> None
        Parse *text_data* for {'+'*5} blocks and apply each operation to disk.
"""

import os
import shutil
from typing import List, Optional, Set, Tuple

from lib.patch import apply_patch
from lib.validation import block_pattern
from lib.utils import COLOR_RED, COLOR_YELLOW, COLOR_BLUE, COLOR_CYAN, COLOR_GREEN, COLOR_RESET


def claude_data_to_file(
    text_data: str,
    abs_file_paths: Optional[Set[str]] = None,
) -> None:
    """Parse Claude's response for file-operation blocks and apply them to disk.

    Parameters
    ----------
    text_data : str
        Raw response text containing ``{'+'*5}`` delimited blocks.
    abs_file_paths : set[str], optional
        Normalised absolute paths of originally discovered source files.
        Used to tag operations as "in original source" or "new/external" in
        the detailed report.
    """
    print("Applying changes to disk...")

    if abs_file_paths is None:
        abs_file_paths = set()

    files_written = 0
    files_deleted = 0
    files_moved = 0
    files_patched = 0

    # (action, source, destination_or_none, existed_before, in_original)
    detailed_entries: List[Tuple[str, str, Optional[str], bool, bool]] = []

    for m in block_pattern.finditer(text_data):
        source_path = m.group("source").strip()
        tag = (m.group("tag") or m.group("move") or "").strip()
        destination = (m.group("dest") or "").strip()
        content = m.group("content").strip()

        abs_source = os.path.normcase(os.path.abspath(source_path))
        in_original = abs_source in abs_file_paths
        existed_before = os.path.exists(source_path)

        # ====================== DELETE ======================
        if tag == "DELETE":
            if existed_before:
                try:
                    os.remove(source_path)
                    files_deleted += 1
                except Exception as e:
                    print(f"Error deleting {source_path}: {e}")
            else:
                print(f"File not found (for deletion): {source_path}")

            detailed_entries.append(("DELETE", source_path, None, existed_before, in_original))
            continue

        # ====================== MOVE ========================
        if tag == "MOVE":
            if not destination:
                print(f"Error: MOVE command for {source_path} missing destination path")
                detailed_entries.append(("MOVE_ERROR", source_path, None, existed_before, in_original))
                continue

            if not existed_before:
                print(f"File not found (for move): {source_path}")
                detailed_entries.append(("MOVE_ERROR", source_path, destination, False, in_original))
                continue

            dest_dir = os.path.dirname(destination) or "."
            try:
                os.makedirs(dest_dir, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {dest_dir} for move: {e}")
                detailed_entries.append(("MOVE_ERROR", source_path, destination, existed_before, in_original))
                continue

            try:
                shutil.move(source_path, destination)
                files_moved += 1
            except Exception as e:
                print(f"Error moving {source_path} to {destination}: {e}")
                detailed_entries.append(("MOVE_ERROR", source_path, destination, existed_before, in_original))
                continue

            detailed_entries.append(("MOVE", source_path, destination, existed_before, in_original))
            continue

        # ====================== PATCH =======================
        if tag == "PATCH":
            success = apply_patch(source_path, content)
            if success:
                files_patched += 1
                detailed_entries.append(("PATCH", source_path, None, existed_before, in_original))
            else:
                detailed_entries.append(("PATCH_ERROR", source_path, None, existed_before, in_original))
            continue

        # ====================== WRITE (default) =============
        target_dir = os.path.dirname(source_path) or "."
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {target_dir}: {e}")
            detailed_entries.append(("WRITE_ERROR", source_path, None, existed_before, in_original))
            continue

        try:
            with open(source_path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            files_written += 1
        except Exception as e:
            print(f"Error writing {source_path}: {e}")
            detailed_entries.append(("WRITE_ERROR", source_path, None, existed_before, in_original))
            continue

        detailed_entries.append(("WRITE", source_path, None, existed_before, in_original))

    # ====================== REPORT ==========================
    print("\nReport:")
    print(f" {files_written} file(s) written, {files_patched} file(s) patched, {files_deleted} file(s) deleted, {files_moved} file(s) moved.")

    if detailed_entries:
        print("\nReport (detailed):")
        for entry in detailed_entries:
            action, source, dest, existed_before, in_original = entry

            if action == "DELETE":
                if in_original:
                    line = f"{COLOR_RED}{source} [DELETED]{COLOR_RESET}"
                else:
                    line = f"{COLOR_YELLOW}{source} [DELETED - WARNING, not in source]{COLOR_RESET}"

            elif action == "MOVE":
                if in_original:
                    line = f"{COLOR_CYAN}{source} -> {dest} [MOVED]{COLOR_RESET}"
                else:
                    line = f"{COLOR_YELLOW}{source} -> {dest} [MOVED - WARNING, source not in original]{COLOR_RESET}"

            elif action == "MOVE_ERROR":
                dest_str = f" -> {dest}" if dest else ""
                line = f"{COLOR_YELLOW}{source}{dest_str} [MOVE FAILED]{COLOR_RESET}"

            elif action == "PATCH":
                if in_original:
                    line = f"{COLOR_GREEN}{source} [PATCHED]{COLOR_RESET}"
                else:
                    line = f"{COLOR_YELLOW}{source} [PATCHED - WARNING, not in source]{COLOR_RESET}"

            elif action == "PATCH_ERROR":
                line = f"{COLOR_YELLOW}{source} [PATCH FAILED]{COLOR_RESET}"

            elif action == "WRITE":
                if not existed_before:
                    line = f"{COLOR_BLUE}{source} [NEW]{COLOR_RESET}"
                elif in_original:
                    line = f"{source} [UPDATED]"
                else:
                    line = f"{COLOR_YELLOW}{source} [UPDATED - WARNING, not in source]{COLOR_RESET}"

            elif action == "WRITE_ERROR":
                line = f"{COLOR_YELLOW}{source} [WRITE FAILED]{COLOR_RESET}"

            else:
                line = f"{source} [UNKNOWN ACTION: {action}]"

            print(f" - {line}")
