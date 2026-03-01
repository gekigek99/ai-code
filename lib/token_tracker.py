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

    Components map to what the LLM actually receives:
      - system:              The system prompt (from yaml ``system`` field)
      - long_term_memory:    Long-term project memory (.ai-code/memory/long-term.md)
      - short_term_memory:   Short-term workflow memory (ai-steps context)
      - git_history:         Recent git commit history
      - source_files:        Content of all source files shared with the LLM
      - file_tree:           The project directory tree listing
      - user_prompt:         The user's prompt text (from yaml ``prompt`` field)
      - memory_instructions: Inline memory update instructions appended to prompt
    """
    system: int = 0               # System prompt tokens
    long_term_memory: int = 0     # Long-term project memory tokens
    short_term_memory: int = 0    # Short-term workflow memory tokens
    git_history: int = 0          # Git commit history tokens
    source_files: int = 0         # Source file content tokens
    file_tree: int = 0            # Directory tree listing tokens
    user_prompt: int = 0          # User prompt from yaml tokens
    memory_instructions: int = 0  # Inline memory update instructions tokens

    @property
    def total(self) -> int:
        """Sum of all component token estimates."""
        return (
            self.system
            + self.long_term_memory
            + self.short_term_memory
            + self.git_history
            + self.source_files
            + self.file_tree
            + self.user_prompt
            + self.memory_instructions
        )

    def _components(self) -> List[Tuple[str, int]]:
        """Return ordered list of (label, token_count) pairs for display.

        Ordered from context framing (system, memory) through source data
        to the user's actual request, matching the logical flow of what
        the LLM processes.
        """
        return [
            ("System", self.system),
            ("Long-term Memory", self.long_term_memory),
            ("Short-term Memory", self.short_term_memory),
            ("Git History", self.git_history),
            ("Source Files", self.source_files),
            ("File Tree", self.file_tree),
            ("User Prompt", self.user_prompt),
            ("Memory Instructions", self.memory_instructions),
        ]


# ── Bar chart configuration ──────────────────────────────────────────────────
_BAR_WIDTH = 25       # Number of characters for the bar
_BAR_FILLED = "█"
_BAR_EMPTY = " "
_LABEL_WIDTH = 20     # Padding for component labels (widened for longer names)
_TOKEN_WIDTH = 10     # Padding for token count column


def display_token_breakdown(breakdown: TokenBreakdown) -> None:
    """Print a coloured ASCII bar chart of token usage to stdout.

    Zero-token components are omitted to reduce noise.  The total line
    appears first as a summary, followed by each non-zero component with
    a proportional bar and percentage.
    """
    total = breakdown.total
    if total == 0:
        print(f"\n{COLOR_CYAN}[Token Breakdown] No token data available.{COLOR_RESET}")
        return

    components = breakdown._components()
    total_token_str = f"~{total:,} tk"

    print(f"\n{COLOR_CYAN}TOKEN USAGE BREAKDOWN{COLOR_RESET}")
    print(f"{COLOR_CYAN}{"Total".ljust(_LABEL_WIDTH + 2)} {total_token_str.rjust(_TOKEN_WIDTH)} {COLOR_RESET}")

    for label, tokens in components:
        # Skip components with zero tokens to reduce noise
        if tokens == 0:
            continue

        # Calculate bar fill and percentage
        pct = (tokens / total) * 100.0 if total > 0 else 0.0
        filled_count = round((tokens / total) * _BAR_WIDTH) if total > 0 else 0
        filled_count = max(filled_count, 1) if tokens > 0 else 0  # at least 1 bar if non-zero
        empty_count = _BAR_WIDTH - filled_count

        bar = _BAR_FILLED * filled_count + _BAR_EMPTY * empty_count + "|"
        token_str = f"~{tokens:,} tk"

        # Build the line with consistent padding
        padded_label = label.ljust(_LABEL_WIDTH)
        padded_tokens = token_str.rjust(_TOKEN_WIDTH)
        pct_str = f"{pct:5.1f}%"

        print(f"{COLOR_CYAN} └ {padded_label}{padded_tokens}  {bar}  {pct_str}{COLOR_RESET}")
