"""
lib.memory — core memory system for AI context persistence.

Manages reading, writing, assembling, and inline-updating project memory
that is injected into LLM prompts to provide cross-session continuity.

Memory updates are performed **inline**: the main execution prompt includes
instructions for Claude to output a memory entry in the JSON ``files``
array alongside regular code entries.  This eliminates separate API calls
for memory updates, saving tokens and latency.

Memory layout:
    Long-term memory:  ``<parent_of_script_dir>/.ai-code/memory/long-term.md``
        Tracked in the master project's git history.
    Short-term memory: ``<script_dir>/memory/short-term.md``
        Ephemeral workflow state; lives inside the ai-code tool directory.

Public API:
    load_long_term_memory(memory_long_term_dir)   -> str
    save_long_term_memory(memory_long_term_dir, content) -> None
    load_short_term_memory(memory_short_term_dir)  -> str
    save_short_term_memory(memory_short_term_dir, content) -> None
    clear_short_term_memory(memory_short_term_dir) -> None

    MemoryBlockResult — dataclass returned by build_memory_block with text
        and per-component token estimates.

    build_memory_block(cfg, include_short_term=False) -> MemoryBlockResult
        Assemble the full memory text block for prompt injection (read-only context).

    build_memory_update_instructions(cfg) -> str
        Build instructions appended to the user prompt that ask Claude to
        include an updated ``.ai-code/memory/long-term.md`` entry in the
        JSON ``files`` array.  Returns empty string when memory updates
        are disabled.

    extract_and_save_memory_from_response(cfg, response_text) -> dict
        Parse the JSON response, find the ``long-term.md`` entry in the
        ``files`` array, save it to cfg.memory_long_term_dir, remove it
        from the array, and return the modified parsed dict so downstream
        code never sees the memory entry.
"""

import os
from dataclasses import dataclass
from typing import Optional

from lib.config import Config
from lib.git import get_recent_commits
from lib.utils import warn

# ------------------------------------------------------------------------------
# Constants — canonical filenames.
# Every function in this module references these rather than hardcoded strings
# so renaming is a single-line change.
# ------------------------------------------------------------------------------
_LONG_TERM_FILENAME = "long-term.md"
_SHORT_TERM_FILENAME = "short-term.md"

# Relative path used in Claude's JSON file entries for the memory file.
# This path is relative to the ai-code script directory (the working dir
# when ai-code runs).  Claude outputs this path in the "path" field of
# its JSON entry, and extract_and_save_memory_from_response matches it.
#
# Layout:  <project_root>/.ai-code/memory/long-term.md
# From ai-code dir: ../.ai-code/memory/long-term.md
_MEMORY_BLOCK_PATH = "../.ai-code/memory/long-term.md"


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass for build_memory_block
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MemoryBlockResult:
    """Result from :func:`build_memory_block` containing the assembled text
    and per-component token estimates for the token usage breakdown graph.

    Token estimates use the ~chars/4 heuristic.
    """
    text: str                   # Full assembled memory block (or empty string)
    long_term_tokens: int = 0   # Tokens from long-term project memory
    short_term_tokens: int = 0  # Tokens from short-term workflow memory
    git_history_tokens: int = 0 # Tokens from git commit history


# ══════════════════════════════════════════════════════════════════════════════
# File I/O — long-term memory
# ══════════════════════════════════════════════════════════════════════════════

def load_long_term_memory(memory_long_term_dir: str) -> str:
    """Read ``long-term.md`` from *memory_long_term_dir* and return its contents.

    *memory_long_term_dir* is expected to be ``<project_root>/.ai-code/memory/``
    (resolved by config.py).  The file read is
    ``<project_root>/.ai-code/memory/long-term.md``.

    Returns an empty string when the file does not exist (first run).
    """
    path = os.path.join(memory_long_term_dir, _LONG_TERM_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        warn(f"[memory] Failed to read long-term memory at {path}: {e}")
        return ""


def save_long_term_memory(memory_long_term_dir: str, content: str) -> None:
    """Atomically write *content* to ``long-term.md`` inside *memory_long_term_dir*.

    Uses write-to-tmp-then-replace so a crash mid-write never corrupts the
    existing file — either the old or the new version is always intact.
    """
    os.makedirs(memory_long_term_dir, exist_ok=True)
    path = os.path.join(memory_long_term_dir, _LONG_TERM_FILENAME)
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

def load_short_term_memory(memory_short_term_dir: str) -> str:
    """Read ``short-term.md`` from *memory_short_term_dir* and return its contents.

    Returns an empty string when the file does not exist.
    """
    path = os.path.join(memory_short_term_dir, _SHORT_TERM_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        warn(f"[memory] Failed to read short-term memory at {path}: {e}")
        return ""


def save_short_term_memory(memory_short_term_dir: str, content: str) -> None:
    """Atomically write *content* to ``short-term.md`` inside *memory_short_term_dir*.

    Same atomic-write pattern as :func:`save_long_term_memory`.
    """
    os.makedirs(memory_short_term_dir, exist_ok=True)
    path = os.path.join(memory_short_term_dir, _SHORT_TERM_FILENAME)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception as e:
        warn(f"[memory] Failed to save short-term memory at {path}: {e}")


def clear_short_term_memory(memory_short_term_dir: str) -> None:
    """Delete ``short-term.md`` if it exists.  Silently ignores absence."""
    path = os.path.join(memory_short_term_dir, _SHORT_TERM_FILENAME)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # Nothing to remove — expected on first run or after clear
    except Exception as e:
        warn(f"[memory] Failed to clear short-term memory at {path}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Memory block assembly (read-only context injection)
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_block(cfg: Config, include_short_term: bool = False) -> MemoryBlockResult:
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
    MemoryBlockResult
        Contains the assembled text block and per-component token estimates.
        ``text`` is empty string if memory is entirely disabled.
    """
    if not cfg.memory_enabled:
        return MemoryBlockResult(text="")

    parts: list[str] = ["[MEMORY START]\n"]
    long_term_tokens = 0
    short_term_tokens = 0
    git_history_tokens = 0

    # -- Long-term project memory ---------------------------------------------
    if cfg.memory_long_term_enabled:
        content = load_long_term_memory(cfg.memory_long_term_dir)
        if content:
            long_term_section = f"## Project Memory (Long-term)\n{content}\n"
            parts.append(long_term_section)
            long_term_tokens = len(long_term_section) // 4

            # Rough token estimate: 1 token ≈ 4 characters.  Warn the LLM
            # when memory is approaching the configured soft cap so it can
            # prioritise compaction in the inline update.
            content_tokens = len(content) // 4
            if content_tokens > cfg.memory_long_term_max_tokens:
                compaction_note = (
                    "(Note: Project memory is approaching token limit "
                    "— compact aggressively in the memory update output)\n"
                )
                parts.append(compaction_note)
                long_term_tokens += len(compaction_note) // 4
        else:
            placeholder = (
                "## Project Memory (Long-term)\n"
                "(No project memory yet — will be created from this execution)\n"
            )
            parts.append(placeholder)
            long_term_tokens = len(placeholder) // 4
        parts.append("")  # blank line separator

    # -- Short-term workflow memory -------------------------------------------
    if include_short_term and cfg.memory_short_term_enabled:
        content = load_short_term_memory(cfg.memory_short_term_dir)
        if content:
            # Only include when content exists — no placeholder for short-term
            short_term_section = f"## Workflow Memory (Short-term)\n{content}\n\n"
            parts.append(short_term_section)
            short_term_tokens = len(short_term_section) // 4

    # -- Git history ----------------------------------------------------------
    if cfg.memory_git_history_enabled:
        history = get_recent_commits(
            n=cfg.memory_git_history_commits,
            ignore_dir_name=cfg.script_dir_name,
        )
        parts.append(history)
        parts.append("\n")
        git_history_tokens = len(history) // 4

    parts.append("[MEMORY END]")
    text = "\n".join(parts)

    return MemoryBlockResult(
        text=text,
        long_term_tokens=long_term_tokens,
        short_term_tokens=short_term_tokens,
        git_history_tokens=git_history_tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Inline memory update — instructions appended to the user prompt
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_update_instructions(cfg: Config) -> str:
    """Build instructions that ask Claude to include an updated memory entry
    in the JSON ``files`` array.

    These instructions are appended to the user prompt so that Claude
    produces an EDIT entry for ``.ai-code/memory/long-term.md`` in its
    JSON ``files`` array alongside regular code entries — eliminating the
    need for a separate API call to update memory.  Claude already has the
    existing memory from the ``[MEMORY START]`` context block and sees all
    source files in the prompt, so it has everything needed to produce an
    accurate update.

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

    existing = load_long_term_memory(cfg.memory_long_term_dir)

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

    return f"""

# ----- MEMORY UPDATE ----- #
{context_ref}

In addition to your regular file output, include an additional entry in your `files` JSON array:
{{
  "action": "EDIT",
  "path": "./{_MEMORY_BLOCK_PATH}",
  "content": "<updated project memory content>"
}}

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
# Inline memory extraction — parse memory entry from Claude's JSON response
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_save_memory_from_response(cfg: Config, response_text: str) -> dict:
    """Extract ``long-term.md`` entry from the parsed JSON response, save it,
    and return the response dict with the memory entry removed from ``files``.

    This is called after Claude responds to a prompt that included inline
    memory update instructions (via :func:`build_memory_update_instructions`).
    The memory entry is saved to ``cfg.memory_long_term_dir`` and removed
    from the ``files`` array so that the apply module does not try to create
    it as a project file (the .ai-code/ directory is in exclude_patterns
    and lives in the project root, not the cwd).

    Matches any entry where ``action == "EDIT"`` and ``path`` ends with
    ``long-term.md`` and contains ``.ai-code/`` to avoid false positives.

    Parameters
    ----------
    cfg : Config
        Resolved configuration (provides ``memory_long_term_dir`` and toggles).
    response_text : str
        The raw Claude JSON response text.

    Returns
    -------
    dict
        The parsed JSON response with the memory entry removed from the
        ``files`` array.  If parsing fails, returns ``{"files": []}``.
    """
    from lib.validation import parse_response_json, ResponseParseError

    # -- Parse JSON response ----------------------------------------------
    try:
        parsed = parse_response_json(response_text)
    except ResponseParseError as e:
        warn(f"[memory] Failed to parse response JSON: {e}")
        return {"files": []}

    # When memory updates are disabled, return parsed dict unchanged —
    # no extraction or removal needed.
    if not cfg.memory_enabled or not cfg.memory_long_term_enabled or not cfg.memory_auto_update:
        return parsed

    # -- Search for the memory entry in the files array -------------------
    memory_index = None
    for idx, entry in enumerate(parsed["files"]):
        if entry.get("action") != "EDIT":
            continue
        raw_path = entry.get("path", "")
        norm_path = raw_path.replace("\\", "/")
        if norm_path.endswith(_LONG_TERM_FILENAME) and ".ai-code/" in norm_path:
            memory_index = idx
            break

    if memory_index is not None:
        memory_entry = parsed["files"][memory_index]
        content = memory_entry.get("content", "").strip()

        if content:
            save_long_term_memory(cfg.memory_long_term_dir, content)
            print(f"[memory] ✓ Long-term memory extracted and saved ({len(content)} chars)")
        else:
            warn("[memory] Memory entry found but content was empty — skipping save.")

        # Remove the memory entry so downstream apply never sees it.
        parsed["files"] = [
            entry for i, entry in enumerate(parsed["files"])
            if i != memory_index
        ]
    else:
        # No memory entry found — not an error; Claude may have chosen
        # not to output one (e.g. for very small changes).
        warn("[memory] No .ai-code/memory/long-term.md entry found in response — memory not updated.")

    return parsed
