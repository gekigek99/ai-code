"""
lib.validation — JSON response parsing and structural validation.

Public API:
    ResponseParseError  — exception for invalid/unparseable JSON responses.
    parse_response_json(raw_text) -> dict
        Parse and validate Claude's JSON response.
    validate_claude_response(text_data) -> bool
        Validate and print summary of Claude's JSON response.
"""

import json
from typing import List

from lib.utils import COLOR_GREEN, COLOR_YELLOW, COLOR_RESET, warn


# ──────────────────────────────────────────────────────────────────────────────
# Exception
# ──────────────────────────────────────────────────────────────────────────────

class ResponseParseError(Exception):
    """Raised when Claude's JSON response cannot be parsed or is structurally invalid."""


# ──────────────────────────────────────────────────────────────────────────────
# Allowed actions and their required keys beyond the universal {action, path}
# ──────────────────────────────────────────────────────────────────────────────

_ACTION_REQUIRED_KEYS = {
    "EDIT":   {"content": str},
    "DELETE": {},
    "MOVE":   {"destination": str},
    "PATCH":  {"patches": list},
}


# ──────────────────────────────────────────────────────────────────────────────
# Core parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_response_json(raw_text: str) -> dict:
    """Parse and structurally validate Claude's JSON response.

    Returns the parsed dict on success.
    Raises ``ResponseParseError`` on any failure — empty input, malformed
    JSON, missing keys, wrong types, unknown actions.
    """

    # ── Empty / whitespace guard ─────────────────────────────────────────
    if not raw_text or not raw_text.strip():
        raise ResponseParseError("Response is empty or whitespace-only.")

    text = raw_text.strip()

    # ── Strip optional ```json … ``` fences ──────────────────────────────
    if text.startswith("```json"):
        text = text[len("```json"):].lstrip("\n")
        if text.endswith("```"):
            text = text[:-3].rstrip("\n")
    elif text.startswith("```"):
        text = text[3:].lstrip("\n")
        if text.endswith("```"):
            text = text[:-3].rstrip("\n")

    # ── JSON decode ──────────────────────────────────────────────────────
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Build a context preview around the error position
        pos = exc.pos or 0
        start = max(0, pos - 50)
        end = min(len(text), pos + 50)
        preview = text[start:end]
        raise ResponseParseError(
            f"JSON decode failed: {exc.msg} (line {exc.lineno}, col {exc.colno}, pos {pos}). "
            f"Context: ...{preview!r}..."
        ) from exc

    # ── Top-level structure ──────────────────────────────────────────────
    if not isinstance(parsed, dict):
        raise ResponseParseError(
            f"Top-level JSON value must be an object (dict), got {type(parsed).__name__}."
        )

    if "files" not in parsed:
        raise ResponseParseError(
            "Top-level object is missing required key \"files\"."
        )

    files = parsed["files"]
    if not isinstance(files, list):
        raise ResponseParseError(
            f"\"files\" must be a list, got {type(files).__name__}."
        )

    # ── Per-entry validation ─────────────────────────────────────────────
    for idx, entry in enumerate(files):
        _prefix = f"files[{idx}]"

        if not isinstance(entry, dict):
            raise ResponseParseError(
                f"{_prefix}: entry must be an object, got {type(entry).__name__}."
            )

        # action — required, must be known
        action = entry.get("action")
        if action is None:
            raise ResponseParseError(f"{_prefix}: missing required key \"action\".")
        if action not in _ACTION_REQUIRED_KEYS:
            raise ResponseParseError(
                f"{_prefix}: unknown action \"{action}\". "
                f"Allowed: {', '.join(sorted(_ACTION_REQUIRED_KEYS))}."
            )

        # path — required, non-empty string
        path = entry.get("path")
        if path is None:
            raise ResponseParseError(f"{_prefix}: missing required key \"path\".")
        if not isinstance(path, str) or not path.strip():
            raise ResponseParseError(f"{_prefix}: \"path\" must be a non-empty string.")

        # action-specific keys
        required = _ACTION_REQUIRED_KEYS[action]
        for key, expected_type in required.items():
            val = entry.get(key)
            if val is None:
                raise ResponseParseError(
                    f"{_prefix} (action={action}): missing required key \"{key}\"."
                )
            if not isinstance(val, expected_type):
                raise ResponseParseError(
                    f"{_prefix} (action={action}): \"{key}\" must be {expected_type.__name__}, "
                    f"got {type(val).__name__}."
                )
            # Extra constraints
            if key == "destination" and not val.strip():
                raise ResponseParseError(
                    f"{_prefix} (action=MOVE): \"destination\" must be a non-empty string."
                )
            if key == "patches" and len(val) == 0:
                raise ResponseParseError(
                    f"{_prefix} (action=PATCH): \"patches\" must be a non-empty list."
                )

    return parsed


# ──────────────────────────────────────────────────────────────────────────────
# Human-friendly validation wrapper
# ──────────────────────────────────────────────────────────────────────────────

def validate_claude_response(text_data: str) -> bool:
    """Parse, validate, and print a summary of Claude's JSON response.

    Returns True on success, False on failure.
    """
    try:
        parsed = parse_response_json(text_data)
    except ResponseParseError as exc:
        print(f"\n{COLOR_YELLOW}{'=' * 60}{COLOR_RESET}")
        print(f"{COLOR_YELLOW}  RESPONSE VALIDATION FAILED{COLOR_RESET}")
        print(f"{COLOR_YELLOW}{'=' * 60}{COLOR_RESET}")
        warn(str(exc))
        print(f"{COLOR_YELLOW}{'=' * 60}{COLOR_RESET}\n")
        return False

    files = parsed["files"]
    total = len(files)

    # Build action breakdown
    counts: dict[str, int] = {}
    for entry in files:
        a = entry["action"]
        counts[a] = counts.get(a, 0) + 1

    breakdown = ", ".join(f"{count} {action}" for action, count in sorted(counts.items()))
    summary = f"VALIDATION PASSED: {total} file entry(ies) — {breakdown}"
    print(f"{COLOR_GREEN}{summary}{COLOR_RESET}")

    return True
