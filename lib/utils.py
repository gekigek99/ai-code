"""
lib.utils — shared low-level utilities used across multiple modules.

Public API:
    COLOR_RED, COLOR_YELLOW, COLOR_BLUE, COLOR_CYAN, COLOR_GREEN, COLOR_RESET
        ANSI SGR escape codes for terminal colouring.

    strip_ansi(text) -> str
        Remove all ANSI escape sequences from *text*.

    warn(msg) -> None
        Print a warning message in yellow to stdout.
"""

import re

# ------------------------------------------------------------------------------
# ANSI colour codes
# ------------------------------------------------------------------------------

COLOR_RED    = "\033[38;5;88m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE   = "\033[34m"
COLOR_CYAN   = "\033[36m"
COLOR_GREEN  = "\033[32m"
COLOR_RESET  = "\033[0m"

# ------------------------------------------------------------------------------
# Compiled regex for stripping ANSI escape sequences.
# Matches all Select Graphic Rendition (SGR) sequences of the form ESC[…m
# as well as common CSI sequences (cursor movement, erase, etc.).
# Used to produce clean text for file export and AI payloads.
# ------------------------------------------------------------------------------
_ANSI_ESCAPE_RE = re.compile(r'\033\[[0-9;]*[A-Za-z]')


def strip_ansi(text: str) -> str:
    """Remove all ANSI escape sequences from *text* and return plain text.

    Essential for strings built with ANSI colour codes for console display
    that must later be written to a log file, exported as Markdown, or sent
    to an LLM where escape codes would be noise.
    """
    return _ANSI_ESCAPE_RE.sub('', text)


def warn(msg: str) -> None:
    """Print a warning in yellow to stdout."""
    print(f"{COLOR_YELLOW}{msg}{COLOR_RESET}")
