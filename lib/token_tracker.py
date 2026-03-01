"""
lib.token_tracker — token usage breakdown tracking and ASCII visualisation.

Public API:
    TokenBreakdown  — dataclass tracking per-component token estimates.
    display_token_breakdown(breakdown) -> None
        Print a coloured ASCII bar chart of token usage to stdout.
    compute_and_display_breakdown(**kwargs) -> TokenBreakdown
        Unified function: compute token estimates from raw inputs, display
        the bar chart, and return the populated TokenBreakdown.  All tools
        and workflows should use this instead of manually setting fields.

Token estimates use the ~chars/4 heuristic, consistent with the rest of the
codebase.  These are rough approximations — actual tokenisation varies by
model — but sufficient for cost/context-window awareness.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

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
      - tool_context:        Tool-specific overhead — meta-instructions, example
                             data, section headers, etc. sent by tools like
                             gen-source, expand, or stepize alongside the user
                             prompt.  Zero for standard -ai workflows.
      - memory_instructions: Inline memory update instructions appended to prompt
    """
    system: int = 0               # System prompt tokens
    long_term_memory: int = 0     # Long-term project memory tokens
    short_term_memory: int = 0    # Short-term workflow memory tokens
    git_history: int = 0          # Git commit history tokens
    source_files: int = 0         # Source file content tokens
    file_tree: int = 0            # Directory tree listing tokens
    user_prompt: int = 0          # User prompt from yaml tokens
    tool_context: int = 0         # Tool-specific meta-instructions / example data tokens
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
            + self.tool_context
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
            ("Tool Context", self.tool_context),
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


def compute_and_display_breakdown(
    *,
    system: str = "",
    memory_result: Any = None,
    files_to_ai: Optional[List[Any]] = None,
    ai_file_listing: str = "",
    user_prompt: str = "",
    tool_context_chars: int = 0,
    memory_instructions: str = "",
) -> TokenBreakdown:
    """Unified token breakdown computation, display, and return.

    Computes token estimates from raw inputs using the ~chars/4 heuristic,
    displays the ASCII bar chart, and returns the populated TokenBreakdown.

    All tools and workflows should use this single entry point instead of
    manually constructing a TokenBreakdown and setting fields — which is
    error-prone (e.g. setting nonexistent field names on the dataclass
    silently creates orphan attributes that ``total`` never includes).

    Parameters
    ----------
    system : str
        The full system prompt text.  Tokens estimated as ``len // 4``.
    memory_result : object, optional
        Result from ``build_memory_block()``.  Must have attributes:
        ``long_term_tokens``, ``short_term_tokens``, ``git_history_tokens``.
        If None, all memory token counts default to 0.
    files_to_ai : list[FileData], optional
        List of discovered file objects.  Source file tokens are computed as
        ``sum(f.ai_data_tokens for f in files_to_ai if f.ai_share)``.
        If None or empty, source_files defaults to 0.
    ai_file_listing : str
        The directory tree listing string.  For gen-source workflows this
        is the ``tree_str``; for other tools it is the ``ai_file_listing``
        returned by ``get_directory_tree()``.  Tokens estimated as
        ``len // 4``.
    user_prompt : str
        The user's raw prompt text (from yaml ``prompt`` field or the
        minimal/expanded prompt in ai-steps).  This is the *semantic*
        user request, excluding any tool wrapper instructions.
    tool_context_chars : int
        Pre-computed character count for tool-specific overhead
        (meta-instructions, example data, section headers) that wraps or
        accompanies the user prompt.  Callers typically compute this as
        ``len(meta_prompt) - len(user_prompt)`` or by summing the
        non-prompt/non-tree parts of the assembled message.
        Tokens estimated as ``chars // 4``.  Defaults to 0.
    memory_instructions : str
        Inline memory update instructions appended to the prompt.
        Only used by ``execute_prompt`` when ``cfg.memory_auto_update``
        is True.  Tokens estimated as ``len // 4``.

    Returns
    -------
    TokenBreakdown
        Fully populated breakdown with all component token estimates.
    """
    breakdown = TokenBreakdown()

    # System prompt
    breakdown.system = len(system) // 4

    # Memory components — extracted from the memory_result object returned by
    # build_memory_block().  Uses getattr for resilience against missing attrs.
    if memory_result is not None:
        breakdown.long_term_memory = getattr(memory_result, "long_term_tokens", 0)
        breakdown.short_term_memory = getattr(memory_result, "short_term_tokens", 0)
        breakdown.git_history = getattr(memory_result, "git_history_tokens", 0)

    # Source file content — sum of per-file token estimates for all shared files
    if files_to_ai:
        breakdown.source_files = sum(
            f.ai_data_tokens for f in files_to_ai if f.ai_share
        )

    # Directory tree listing
    breakdown.file_tree = len(ai_file_listing) // 4

    # User prompt (the semantic request, excluding tool wrappers)
    breakdown.user_prompt = len(user_prompt) // 4

    # Tool-specific overhead (meta-instructions, examples, section headers)
    breakdown.tool_context = max(0, tool_context_chars // 4)

    # Inline memory update instructions
    breakdown.memory_instructions = len(memory_instructions) // 4

    display_token_breakdown(breakdown)
    return breakdown
