"""
lib.memory — core memory system for AI context persistence.

Manages reading, writing, assembling, and updating project memory that is
injected into LLM prompts to provide cross-session continuity.

Public API:
    load_long_term_memory(memory_dir)   -> str
    save_long_term_memory(memory_dir, content) -> None
    load_short_term_memory(memory_dir)  -> str
    save_short_term_memory(memory_dir, content) -> None
    clear_short_term_memory(memory_dir) -> None

    build_memory_block(cfg, include_short_term=False) -> str
        Assemble the full memory text block for prompt injection.

    build_memory_update_prompt(response_text, existing_memory, source_files_summary) -> str
        Build the meta-prompt that asks Claude to produce updated project memory.

    update_long_term_memory(cfg, response_text, source_files_summary) -> bool
        Orchestrate the full memory update flow (load → prompt → parse → save).
        Never raises — returns False on failure.

File layout inside ``memory_dir``:
    project.md      — long-term project memory (architecture, routes, schema…)
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
_LONG_TERM_FILENAME = "project.md"
_SHORT_TERM_FILENAME = "short-term.md"

# Truncation limit for the AI response text included in the memory-update
# prompt.  Keeps the update call small and cheap.
_RESPONSE_TRUNCATION_CHARS = 3000


# ══════════════════════════════════════════════════════════════════════════════
# File I/O — long-term memory
# ══════════════════════════════════════════════════════════════════════════════

def load_long_term_memory(memory_dir: str) -> str:
    """Read ``project.md`` from *memory_dir* and return its contents.

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
    """Atomically write *content* to ``project.md`` inside *memory_dir*.

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
# Memory block assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_block(cfg: Config, include_short_term: bool = False) -> str:
    """Assemble the complete memory text block for prompt injection.

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
            # prioritise compaction on the next auto-update.
            token_estimate = len(content) // 4
            if token_estimate > cfg.memory_long_term_max_tokens:
                parts.append(
                    "(Note: Project memory is approaching token limit "
                    "and may be compacted on next update)\n"
                )
        else:
            parts.append(
                "## Project Memory (Long-term)\n"
                "(No project memory yet — will be created after first execution)\n"
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
# Memory update prompt builder
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_update_prompt(
    response_text: str,
    existing_memory: str,
    source_files_summary: str,
) -> str:
    """Build the meta-prompt that asks Claude to produce updated project memory.

    The prompt is designed to be cheap (small input, small output) and
    produces a single ``+++++ ./memory/project.md [EDIT]`` block that the
    caller can parse and save.

    Parameters
    ----------
    response_text : str
        The AI response from the just-completed execution.  Truncated to
        the first ~3 000 characters to keep update calls inexpensive.
    existing_memory : str
        Current contents of ``project.md`` (may be empty on first run).
    source_files_summary : str
        List of source files that were shared with the AI for context.
    """
    # Truncate response to keep the update prompt small and cheap
    if len(response_text) > _RESPONSE_TRUNCATION_CHARS:
        truncated_response = response_text[:_RESPONSE_TRUNCATION_CHARS] + "\n(truncated)"
    else:
        truncated_response = response_text

    # Tell Claude whether this is a creation or an update
    if existing_memory.strip():
        memory_context = (
            "EXISTING PROJECT MEMORY (update/compact this):\n"
            "--- BEGIN EXISTING MEMORY ---\n"
            f"{existing_memory}\n"
            "--- END EXISTING MEMORY ---"
        )
    else:
        memory_context = (
            "EXISTING PROJECT MEMORY: (none — this is the first memory creation)\n"
            "Create a fresh project memory from the information below."
        )

    marker = "+" * 5

    return f"""TASK: Update (or create) the project memory file based on what was just implemented.

{memory_context}

SOURCE FILES INVOLVED:
{source_files_summary}

AI RESPONSE (what was just implemented):
--- BEGIN RESPONSE ---
{truncated_response}
--- END RESPONSE ---

INSTRUCTIONS:
Produce an updated project.md that is:
- Ultra-condensed: bullet points only, no prose or explanations
- Organised into these sections (omit a section only if truly empty):
  ## Architecture
  ## Key Files & Their Purpose
  ## Database Schema Summary
  ## API Routes Summary
  ## Key Functions/Variables
  ## Conventions & Patterns
- Removes outdated entries that contradict the new changes
- NEVER includes API keys, secrets, passwords, or PII
- Stays within approximately 2000 tokens

OUTPUT FORMAT: Write the updated memory inside a single file block:
{marker} ./memory/project.md [EDIT]
<your updated project memory here>
{marker}

Do not include any other file blocks. Only the memory/project.md block."""


# ══════════════════════════════════════════════════════════════════════════════
# Memory update orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def update_long_term_memory(
    cfg: Config,
    response_text: str,
    source_files_summary: str,
) -> bool:
    """Orchestrate the full long-term memory update flow.

    Sequence: load existing → build update prompt → call Claude → parse
    response → save updated memory.

    This function is intentionally wrapped in a top-level try/except so
    that a memory-update failure **never** crashes the main workflow.

    Parameters
    ----------
    cfg : Config
        Resolved configuration.
    response_text : str
        The AI response from the just-completed execution.
    source_files_summary : str
        Newline-separated list of source file paths that were shared.

    Returns
    -------
    bool
        True if the update succeeded (or was skipped because disabled),
        False on any failure.
    """
    # Early exit when memory update is not configured
    if not cfg.memory_enabled or not cfg.memory_long_term_enabled or not cfg.memory_auto_update:
        return True

    try:
        # Lazy imports — kept inside the function to avoid circular
        # dependencies and to keep the import cost out of the module level
        # for callers that never invoke this function.
        from lib.providers.claude import prompt_claude
        from lib.validation import block_pattern

        print("\n[memory] Updating long-term project memory...")

        # 1. Load existing memory
        existing_memory = load_long_term_memory(cfg.memory_dir)

        # 2. Build the update prompt
        update_prompt = build_memory_update_prompt(
            response_text=response_text,
            existing_memory=existing_memory,
            source_files_summary=source_files_summary,
        )

        # 3. Call Claude with a small, cheap request
        result = prompt_claude(
            api_key=cfg.anthropic_api_key,
            model=cfg.anthropic_model,
            system=(
                "You are a memory management assistant. Produce concise, "
                "structured project memory files. Never include secrets, "
                "API keys, or PII."
            ),
            messages=[{"role": "user", "content": update_prompt}],
            max_tokens=4000,
            temperature=0.5,
            websearch=False,
            thinking_budget=0,
            stream=False,
        )

        if result["status"] != "ok":
            warn(
                f"[memory] Claude returned non-ok status: {result['status']} — "
                f"{result.get('error', '(no error detail)')}"
            )
            return False

        ai_text = result["data_response"]

        # 4. Parse the response for a memory/project.md block
        match = None
        for m in block_pattern.finditer(ai_text):
            source_path = m.group("source").strip()
            # Accept any path that ends with the long-term memory filename
            if source_path.endswith(_LONG_TERM_FILENAME):
                match = m
                break

        if match:
            new_memory = match.group("content").strip()
        else:
            # Fallback: Claude may have produced plain text without the
            # file-block wrapper.  Use the entire response as memory content.
            warn("[memory] No file block found in response — using raw response as memory.")
            new_memory = ai_text.strip()

        if not new_memory:
            warn("[memory] Produced empty memory content — skipping save.")
            return False

        # 5. Save the updated memory
        save_long_term_memory(cfg.memory_dir, new_memory)
        print(f"[memory] ✓ Long-term memory updated ({len(new_memory)} chars)")
        return True

    except Exception as e:
        # Memory update must never crash the main workflow
        warn(f"[memory] Update failed (non-fatal): {type(e).__name__}: {e}")
        return False
