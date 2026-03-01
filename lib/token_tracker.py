"""
lib.token_tracker — token usage breakdown tracking and ASCII visualisation.

Public API:
    TokenBreakdown  — dataclass tracking per-component token estimates.
    display_token_breakdown(breakdown) -> None
        Print a coloured ASCII bar chart of token usage to stdout.

Token estimates use the ~chars/4 heuristic, consistent with the rest of the
codebase.  These are rough approximations — actual tokenisation varies by
model — but sufficient for cost/context-window awareness.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from lib.utils import COLOR_CYAN, COLOR_GREEN, COLOR_YELLOW, COLOR_RESET


@dataclass
class TokenBreakdown:
    """Tracks estimated token counts for each component of an LLM request.

    Each field corresponds to a distinct section of the assembled prompt.
    All values are approximate (~chars/4).
    """
    system: int = 0              # System prompt tokens
    long_term_memory: int = 0    # Long-term project memory tokens
    short_term_memory: int = 0   # Short-term workflow memory tokens
    git_history: int = 0         # Git commit history tokens
    file_data: int = 0           # Source file content tokens
    prompt: int = 0              # User prompt + memory update instructions tokens

    @property
    def total(self) -> int:
        """Sum of all component token estimates."""
        return (
            self.system
            + self.long_term_memory
            + self.short_term_memory
            + self.git_history
            + self.file_data
            + self.prompt
        )

    def _components(self) -> List[Tuple[str, int]]:
        """Return ordered list of (label, token_count) pairs for display."""
        return [
            ("System", self.system),
            ("Long-term Memory", self.long_term_memory),
            ("Short-term Memory", self.short_term_memory),
            ("Git History", self.git_history),
            ("File Data", self.file_data),
            ("Prompt", self.prompt),
        ]


# ── Bar chart configuration ──────────────────────────────────────────────────
_BAR_WIDTH = 25       # Number of characters for the bar
_BAR_FILLED = "■"
_BAR_EMPTY = " "
_LABEL_WIDTH = 18     # Padding for component labels
_TOKEN_WIDTH = 10     # Padding for token count column


def display_token_breakdown(breakdown: TokenBreakdown) -> None:
    """Print a coloured ASCII bar chart of token usage to stdout."""
    total = breakdown.total
    if total == 0:
        print(f"\n{COLOR_CYAN}[Token Breakdown] No token data available.{COLOR_RESET}")
        return

    components = breakdown._components()
    total_token_str = f"~{total:,} tk"

    print(f"\n{COLOR_CYAN}TOKEN USAGE BREAKDOWN{COLOR_RESET}")
    print(f"{COLOR_CYAN}{"Total".ljust(_LABEL_WIDTH+2)} {total_token_str.rjust(_TOKEN_WIDTH)} {COLOR_RESET}")

    for label, tokens in components:
        # Skip components with zero tokens to reduce noise
        if tokens == 0:
            continue

        # Calculate bar fill and percentage
        pct = (tokens / total) * 100.0 if total > 0 else 0.0
        filled_count = round((tokens / total) * _BAR_WIDTH) if total > 0 else 0
        filled_count = max(filled_count, 1) if tokens > 0 else 0  # at least 1 bar if non-zero
        empty_count = _BAR_WIDTH - filled_count

        bar = _BAR_FILLED * filled_count + _BAR_EMPTY * empty_count
        token_str = f"~{tokens:,} tk"

        # Build the line with consistent padding
        padded_label = label.ljust(_LABEL_WIDTH)
        padded_tokens = token_str.rjust(_TOKEN_WIDTH)
        pct_str = f"{pct:5.1f}%"

        print(f"{COLOR_CYAN} └ {padded_label}{padded_tokens}  {bar}  {pct_str}{COLOR_RESET}")
