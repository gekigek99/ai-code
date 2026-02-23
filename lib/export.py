"""
lib.export — file export and prompt logging utilities.

Public API:
    export_md_file(data, filename, output_dir) -> None
    log_prompt(content, logs_dir) -> None
"""

import os
import datetime


def export_md_file(data: str, filename: str, output_dir: str) -> None:
    """Write *data* to ``<output_dir>/<filename>`` as UTF-8 text.

    Parameters
    ----------
    data : str
        Content to write (typically Markdown).
    filename : str
        Target file name (not a full path).
    output_dir : str
        Directory in which to create the file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{data}")
    print(f"\nSaved to: {os.path.relpath(filepath).replace(os.sep, '/')}")


def log_prompt(content: str, logs_dir: str) -> None:
    """Append *content* to ``<logs_dir>/prompt.log`` with a timestamp header.

    Parameters
    ----------
    content : str
        The prompt text to log.
    logs_dir : str
        Directory containing the log file.
    """
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "prompt.log")
    timestamp = datetime.datetime.now().isoformat()
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"\n=== Prompt logged at {timestamp} ===\n{content}\n")
    except Exception as e:
        from lib.utils import warn
        warn(f"Warning: Could not write to log file: {e}")
