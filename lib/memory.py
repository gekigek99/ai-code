"""
lib.memory — core memory system for AI context persistence.

Manages reading, writing, assembling, and inline-updating project memory
that is injected into LLM prompts to provide cross-session continuity.

Memory updates are performed **inline**: the main execution prompt includes
instructions for Claude to output a ``memory/long-term.md`` file block
alongside regular code blocks.  This eliminates separate API calls for
memory updates, saving tokens and latency.

Public API:
    load_long_term_memory(memory_dir)   -> str
    save_long_term_memory(memory_dir, content) -> None
    load_short_term_memory(memory_dir)  -> str
    save_short_term_memory(memory_dir, content) -> None
    clear_short_term_memory(memory_dir) -> None

    build_memory_block(cfg, include_short_term=False) -> str
        Assemble the full memory text block for prompt injection (read-only context).

    build_memory_update_instructions(cfg) -> str
        Build instructions appended to the user prompt that ask Claude to
        output an updated ``memory/long-term.md`` block inline with other
        file blocks.  Returns empty string when memory updates are disabled.

    extract_and_save_memory_from_response(cfg, response_text) -> str
        Parse the response for a ``memory/long-term.md`` block, save it to
        disk, and return the response with the memory block removed so it
        is not applied as a project file.

File layout inside ``memory_dir``:
    long-term.md    — long-term project memory (architecture, routes, schema…)
    short-term.md   — short-term workflow memory (current ai-steps context)
"""

import os
from typing import Optional

from lib.config import Config
from lib.git import get_recent_commits
from lib.utils import warn

# ──────────────────────────────────────────────────────────────────────────────
# Constants — canonical filenames inside the memory directory.
# Every function in this module references these rather than hardcoded strings
# so renaming is a single-line change.
# ──────────────────────────────────────────────────────────────────────────────
_LONG_TERM_FILENAME = "long-term.md"
_SHORT_TERM_FILENAME = "short-term.md"


# ══════════════════════════════════════════════════════════════════════════════
# File I/O — long-term memory
# ══════════════════════════════════════════════════════════════════════════════

def load_long_term_memory(memory_dir: str) -> str:
    """Read ``long-term.md`` from *memory_dir* and return its contents.

    Returns an empty string when the file does not exist (first run).
    """
    path = os.path.join(memory_dir, _LONG_TERM_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        warn(f"[memory] Failed to read long-term memory at {path}: {e}")
        return ""


def save_long_term_memory(memory_dir: str, content: str) -> None:
    """Atomically write *content* to ``long-term.md`` inside *memory_dir*.

    Uses write-to-tmp-then-replace so a crash mid-write never corrupts the
    existing file — either the old or the new version is always intact.
    """
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, _LONG_TERM_FILENAME)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception as e:
        warn(f"[memory] Failed to save long-term memory at {path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# File I/O — short-term memory
# ══════════════════════════════════════════════════════════════════════════════

def load_short_term_memory(memory_dir: str) -> str:
    """Read ``short-term.md`` from *memory_dir* and return its contents.

    Returns an empty string when the file does not exist.
    """
    path = os.path.join(memory_dir, _SHORT_TERM_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        warn(f"[memory] Failed to read short-term memory at {path}: {e}")
        return ""


def save_short_term_memory(memory_dir: str, content: str) -> None:
    """Atomically write *content* to ``short-term.md`` inside *memory_dir*.

    Same atomic-write pattern as :func:`save_long_term_memory`.
    """
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, _SHORT_TERM_FILENAME)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception as e:
        warn(f"[memory] Failed to save short-term memory at {path}: {e}")


def clear_short_term_memory(memory_dir: str) -> None:
    """Delete ``short-term.md`` if it exists.  Silently ignores absence."""
    path = os.path.join(memory_dir, _SHORT_TERM_FILENAME)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # Nothing to remove — expected on first run or after clear
    except Exception as e:
        warn(f"[memory] Failed to clear short-term memory at {path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Memory block assembly (read-only context injection)
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_block(cfg: Config, include_short_term: bool = False) -> str:
    """Assemble the complete memory text block for prompt injection.

    This is **read-only context** — it tells Claude what it already knows
    about the project.  The *update* instructions are handled separately
    by :func:`build_memory_update_instructions`.

    The block is wrapped in ``[MEMORY START]`` / ``[MEMORY END]`` markers
    so the LLM can clearly distinguish memory context from source code and
    the user's prompt.

    Parameters
    ----------
    cfg : Config
        Resolved configuration (provides all memory toggles, paths, and limits).
    include_short_term : bool
        When True *and* short-term memory is enabled in config, the
        ``short-term.md`` contents are included.  Typically True only for
        the ``-ai-steps`` workflow where inter-step context matters.

    Returns
    -------
    str
        Assembled memory block, or empty string if memory is disabled.
    """
    if not cfg.memory_enabled:
        return ""

    parts: list[str] = ["[MEMORY START]\n"]

    # ── Long-term project memory ─────────────────────────────────────────────
    if cfg.memory_long_term_enabled:
        content = load_long_term_memory(cfg.memory_dir)
        if content:
            parts.append(f"## Project Memory (Long-term)\n{content}\n")

            # Rough token estimate: 1 token ≈ 4 characters.  Warn the LLM
            # when memory is approaching the configured soft cap so it can
            # prioritise compaction in the inline update.
            token_estimate = len(content) // 4
            if token_estimate > cfg.memory_long_term_max_tokens:
                parts.append(
                    "(Note: Project memory is approaching token limit "
                    "— compact aggressively in the memory update output)\n"
                )
        else:
            parts.append(
                "## Project Memory (Long-term)\n"
                "(No project memory yet — will be created from this execution)\n"
            )
        parts.append("")  # blank line separator

    # ── Short-term workflow memory ───────────────────────────────────────────
    if include_short_term and cfg.memory_short_term_enabled:
        content = load_short_term_memory(cfg.memory_dir)
        if content:
            # Only include when content exists — no placeholder for short-term
            parts.append(f"## Workflow Memory (Short-term)\n{content}\n\n")

    # ── Git history ──────────────────────────────────────────────────────────
    if cfg.memory_git_history_enabled:
        history = get_recent_commits(
            n=cfg.memory_git_history_commits,
            ignore_dir_name=cfg.script_dir_name,
        )
        parts.append(history)
        parts.append("\n")

    parts.append("[MEMORY END]")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Inline memory update — instructions appended to the user prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_update_instructions(cfg: Config) -> str:
    """Build instructions that ask Claude to output an updated memory file inline.

    These instructions are appended to the user prompt so that Claude
    produces a ``memory/long-term.md`` file block alongside its regular
    code output — eliminating the need for a separate API call to update
    memory.  Claude already has the existing memory from the
    ``[MEMORY START]`` context block and sees all source files in the
    prompt, so it has everything needed to produce an accurate update.

    Parameters
    ----------
    cfg : Config
        Resolved configuration.

    Returns
    -------
    str
        Memory update instruction text to append to the user prompt.
        Empty string when memory updates are disabled.
    """
    # Gate on all three required flags: memory system enabled, long-term
    # enabled, and auto-update enabled.  If any is off, no inline update.
    if not cfg.memory_enabled or not cfg.memory_long_term_enabled or not cfg.memory_auto_update:
        return ""

    existing = load_long_term_memory(cfg.memory_dir)

    # Reference the existing memory already in the [MEMORY START] block
    # rather than repeating it — saves tokens.
    if existing.strip():
        context_ref = (
            "Update the existing Project Memory (Long-term) shown in the "
            "[MEMORY START] block above, incorporating the source files "
            "you have seen and the changes you are making."
        )
    else:
        context_ref = (
            "Create a fresh project memory from the source files you have "
            "seen and the changes you are making (no existing memory was found)."
        )

    marker = "+" * 5

    return f"""

# ----- MEMORY UPDATE ----- #
{context_ref}

In addition to your regular file output, also output an updated project memory file block:
{marker} ./memory/long-term.md [EDIT]
<updated project memory>
{marker}

Memory format rules:
- Ultra-condensed bullet points only, no prose or explanations
- Organise into these sections (omit a section only if truly empty):
  ## Architecture
  ## Key Files & Their Purpose
  ## Database Schema Summary
  ## API Routes Summary
  ## Key Functions/Variables
  ## Conventions & Patterns
- Reflect both the existing codebase AND the changes you just made
- Remove outdated entries that contradict your changes
- NEVER include API keys, secrets, passwords, or PII
- Stay within approximately 2000 tokens
"""


# ══════════════════════════════════════════════════════════════════════════════
# Inline memory extraction — parse memory block from Claude response
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_save_memory_from_response(cfg: Config, response_text: str) -> str:
    """Extract ``memory/long-term.md`` block from the response, save it, and
    return the response with the memory block removed.

    This is called after Claude responds to a prompt that included inline
    memory update instructions (via :func:`build_memory_update_instructions`).
    The memory block is saved to ``cfg.memory_dir`` and stripped from the
    response so that the apply module does not try to create it as a
    project file.

    Parameters
    ----------
    cfg : Config
        Resolved configuration (provides ``memory_dir`` and toggles).
    response_text : str
        The full Claude response text potentially containing a
        ``memory/long-term.md`` file block.

    Returns
    -------
    str
        The response text with the memory block removed.  Unchanged if no
        memory block was found or if memory updates are disabled.
    """
    if not cfg.memory_enabled or not cfg.memory_long_term_enabled or not cfg.memory_auto_update:
        return response_text

    # Lazy import to avoid circular dependency — validation imports are
    # lightweight and only needed here.
    from lib.validation import block_pattern

    for m in block_pattern.finditer(response_text):
        source_path = m.group("source").strip()

        # Match any path ending with the long-term memory filename, e.g.
        # "./memory/long-term.md", "memory/long-term.md", etc.
        if source_path.endswith(_LONG_TERM_FILENAME):
            content = m.group("content").strip()

            if content:
                save_long_term_memory(cfg.memory_dir, content)
                print(f"[memory] ✓ Long-term memory extracted and saved ({len(content)} chars)")
            else:
                warn("[memory] Memory block found but content was empty — skipping save.")

            # Remove the memory block from the response.  Use match span
            # positions for precise excision rather than string replacement,
            # which could match incorrectly if similar text appears elsewhere.
            cleaned = response_text[:m.start()] + response_text[m.end():]
            return cleaned

    # No memory block found — this is not an error; Claude may have chosen
    # not to output one (e.g. for very small changes).  Log for visibility.
    warn("[memory] No memory/long-term.md block found in response — memory not updated.")
    return response_text
