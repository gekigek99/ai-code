"""
lib.validation — response structural validation and block-detection regex.

Public API:
    block_pattern       — compiled regex that matches file-operation blocks.
    validate_claude_response(text_data) -> bool
        Check structural integrity of a Claude response before applying
        file operations.
"""

import re
from typing import List

from lib.utils import COLOR_GREEN, COLOR_YELLOW, COLOR_RESET, warn

# ──────────────────────────────────────────────────────────────────────────────
# Block-detection regex
# ──────────────────────────────────────────────────────────────────────────────
# Header line:  {'+'*5} <source_path> [optional TAG] [optional MOVE dest]<newline>
# Body:         anything (non-greedy)
# Closing line: {'+'*5} (exactly, at start of line, nothing else)
#
# Named groups:
#   source  – source file path
#   tag     – EDIT | DELETE (optional)
#   move    – literal "MOVE" when present
#   dest    – destination path for MOVE (optional)
#   content – everything between the header and the closing marker
# ──────────────────────────────────────────────────────────────────────────────
block_pattern = re.compile(
    r'^'
    r'\+{5}\s+'                                     # opening {'+'*5} + whitespace
    r'(?P<source>.+?)'                              # source path (non-greedy)
    r'(?:\s*\[(?P<tag>EDIT|DELETE)\])?'             # optional [EDIT] or [DELETE]
    r'(?:\s*\[(?P<move>MOVE)\]\s+(?P<dest>.+?))?'  # optional [MOVE] + dest path
    r'\s*\n'                                        # end of header
    r'(?P<content>.*?)'                             # content (non-greedy, DOTALL)
    r'^\+{5}$',                                     # closing {'+'*5} on its own line
    re.MULTILINE | re.DOTALL,
)

# Marker-line regex — counts every line starting with 5+ plus signs
_MARKER_LINE_RE = re.compile(r'^\+{5,}', re.MULTILINE)


def validate_claude_response(text_data: str) -> bool:
    """Validate structural integrity of Claude's response.

    Checks performed:
      1. Lines beginning with ``{'+'*5}`` must come in an even count (2 per block).
      2. Data outside matched blocks is less than 3 % of total response length.

    Returns True if all checks pass, False otherwise.  Warnings are printed
    to stdout.
    """
    if not text_data or not text_data.strip():
        warn("VALIDATION SKIP: Response is empty or whitespace-only.")
        return False

    warnings: List[str] = []
    passed = True

    # ── Check 1: marker lines must be even ───────────────────────────────────
    marker_lines = _MARKER_LINE_RE.findall(text_data)
    marker_count = len(marker_lines)

    if marker_count == 0:
        msg = f"VALIDATION WARN: No {'+'*5} marker lines found in response – no file blocks detected."
        warnings.append(msg)
        passed = False
    elif marker_count % 2 != 0:
        msg = (
            f"VALIDATION FAIL: {'+'*5} marker line count is ODD ({marker_count}). "
            f"Expected an even number (exactly 2 per block). "
            f"This means at least one block is missing its header or closing marker."
        )
        warnings.append(msg)
        passed = False
    else:
        block_count_est = marker_count // 2
        msg = f"VALIDATION OK: {'+'*5} marker lines = {marker_count} (= {block_count_est} block(s)), count is even."
        print(f"{COLOR_GREEN}{msg}{COLOR_RESET}")

    # ── Check 2: outside-block percentage < 3 % ─────────────────────────────
    total_len = len(text_data)
    inside_len = sum(m.end() - m.start() for m in block_pattern.finditer(text_data))
    outside_len = total_len - inside_len
    outside_pct = (outside_len / total_len) * 100.0 if total_len > 0 else 0.0
    matched_blocks = len(list(block_pattern.finditer(text_data)))

    pct_msg = (
        f"Response length: {total_len} chars | "
        f"Inside blocks: {inside_len} chars | "
        f"Outside blocks: {outside_len} chars | "
        f"Outside percentage: {outside_pct:.2f}% | "
        f"Matched blocks: {matched_blocks}"
    )
    print(pct_msg)

    OUTSIDE_THRESHOLD_PCT = 3.0
    if outside_pct > OUTSIDE_THRESHOLD_PCT:
        # Collect text that is outside matched blocks for diagnostics
        outside_snippets: List[str] = []
        prev_end = 0
        for m in block_pattern.finditer(text_data):
            gap = text_data[prev_end:m.start()].strip()
            if gap:
                outside_snippets.append(gap)
            prev_end = m.end()
        trailing = text_data[prev_end:].strip()
        if trailing:
            outside_snippets.append(trailing)

        combined_outside = "\n---\n".join(outside_snippets)
        if len(combined_outside) > 500:
            combined_outside = combined_outside[:500] + "... [truncated]"

        msg = (
            f"VALIDATION FAIL: {outside_pct:.2f}% of response data is outside {'+'*5} blocks "
            f"(threshold: {OUTSIDE_THRESHOLD_PCT:.1f}%). "
            f"This likely means Claude emitted prose, explanations, or malformed blocks.\n"
            f"Outside content preview:\n{combined_outside}"
        )
        warnings.append(msg)
        passed = False
    else:
        ok_msg = f"VALIDATION OK: Outside-block percentage {outside_pct:.2f}% is within {OUTSIDE_THRESHOLD_PCT:.1f}% threshold."
        print(f"{COLOR_GREEN}{ok_msg}{COLOR_RESET}")

    # ── Emit accumulated warnings ────────────────────────────────────────────
    if warnings:
        print(f"\n{COLOR_YELLOW}{'='*60}{COLOR_RESET}")
        print(f"{COLOR_YELLOW}  RESPONSE VALIDATION WARNINGS ({len(warnings)}){COLOR_RESET}")
        print(f"{COLOR_YELLOW}{'='*60}{COLOR_RESET}")
        for w in warnings:
            warn(w)
        print(f"{COLOR_YELLOW}{'='*60}{COLOR_RESET}\n")
    else:
        print(f"{COLOR_GREEN}VALIDATION PASSED: All checks OK.{COLOR_RESET}")

    return passed
