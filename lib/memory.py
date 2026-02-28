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

    update_long_term_memory_from_source(cfg, files_to_ai) -> bool
        Scan source files and update long-term memory with a codebase map.
        Called during ai-steps before each step execution so memory always
        reflects the current state of the codebase.  Never raises.

File layout inside ``memory_dir``:
    long-term.md      — long-term project memory (architecture, routes, schema…)
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

# Truncation limit for the AI response text included in the memory-update
# prompt.  Keeps the update call small and cheap.
_RESPONSE_TRUNCATION_CHARS = 3000

# Source-scan digest limits — controls how much file content is sent to
# Claude during pre-execution memory scans.  These caps keep the API call
# cheap while providing enough context to map the codebase structure.
_SOURCE_DIGEST_MAX_TOTAL_CHARS = 10000
_SOURCE_DIGEST_MAX_PER_FILE_CHARS = 600


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
# Source-scan digest builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_source_digest(files_to_ai: list) -> str:
    """Build a compact digest of source files for memory scanning.

    For each text/pdf file shared with the AI, includes the file path and
    a truncated preview of the content (first ~600 chars).  This captures
    imports, class/function definitions, route declarations, and other
    top-level structure — enough for Claude to map the codebase without
    sending full file contents.

    Parameters
    ----------
    files_to_ai : list[FileData]
        File entries from ``add_source``.  Only ``ai_share=True`` text/pdf
        files are included; images and binaries are skipped.

    Returns
    -------
    str
        Concatenated file previews, capped at ~10 000 chars total.
    """
    parts: list[str] = []
    total_chars = 0
    included_count = 0

    for f in files_to_ai:
        # Only scan text-interpretable files — images and raw binaries
        # don't contribute meaningful structural info for memory mapping.
        if not f.ai_share or f.file_type not in ("text", "pdf"):
            continue

        preview = (f.ai_data_converted or "")[:_SOURCE_DIGEST_MAX_PER_FILE_CHARS]
        entry = f"### {f.path_rel}\n{preview}\n---\n"

        if total_chars + len(entry) > _SOURCE_DIGEST_MAX_TOTAL_CHARS:
            # Count remaining files for the truncation notice
            remaining = sum(
                1 for ff in files_to_ai
                if ff.ai_share and ff.file_type in ("text", "pdf")
            ) - included_count
            parts.append(f"(... {remaining} more file(s) truncated for brevity)")
            break

        parts.append(entry)
        total_chars += len(entry)
        included_count += 1

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Memory update prompt builders
# ══════════════════════════════════════════════════════════════════════════════

def build_memory_update_prompt(
    response_text: str,
    existing_memory: str,
    source_files_summary: str,
) -> str:
    """Build the meta-prompt that asks Claude to produce updated project memory.

    The prompt is designed to be cheap (small input, small output) and
    produces a single ``+++++ ./memory/long-term.md [EDIT]`` block that the
    caller can parse and save.

    Parameters
    ----------
    response_text : str
        The AI response from the just-completed execution.  Truncated to
        the first ~3 000 characters to keep update calls inexpensive.
    existing_memory : str
        Current contents of ``long-term.md`` (may be empty on first run).
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
Produce an updated long-term.md that is:
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
{marker} ./memory/long-term.md [EDIT]
<your updated project memory here>
{marker}

Do not include any other file blocks. Only the memory/long-term.md block."""


def _build_source_scan_prompt(existing_memory: str, source_digest: str) -> str:
    """Build the meta-prompt for scanning source files to update long-term memory.

    Unlike :func:`build_memory_update_prompt` which captures *changes*
    after execution, this prompt captures the *current state* of source
    files before execution.  The goal is to keep the memory map in sync
    with the evolving codebase during multi-step workflows where files
    change between steps.

    Parameters
    ----------
    existing_memory : str
        Current contents of ``long-term.md``.
    source_digest : str
        Compact file previews from :func:`_build_source_digest`.

    Returns
    -------
    str
        The meta-prompt for the source-scan memory update.
    """
    if existing_memory.strip():
        memory_context = (
            "EXISTING PROJECT MEMORY (merge new findings into this):\n"
            "--- BEGIN EXISTING MEMORY ---\n"
            f"{existing_memory}\n"
            "--- END EXISTING MEMORY ---"
        )
    else:
        memory_context = (
            "EXISTING PROJECT MEMORY: (none — create fresh from source scan)\n"
        )

    marker = "+" * 5

    return f"""TASK: Scan source file previews and update the project memory with a codebase map.

{memory_context}

SOURCE FILES (preview of each file's beginning):
--- BEGIN SOURCE DIGEST ---
{source_digest}
--- END SOURCE DIGEST ---

INSTRUCTIONS:
Update long-term.md to serve as a fast-reference map of the codebase. Focus on:
- **Architecture**: Module structure, how components connect, data flow patterns
- **Key Files & Their Purpose**: What each file/module does (one line each)
- **Database Schema Summary**: Tables, columns, relationships (if visible)
- **API Routes Summary**: Endpoint paths and handlers (if visible)
- **Key Functions/Variables**: Important functions, their locations, parameters
- **Conventions & Patterns**: Naming patterns, shared utilities, config structure

Rules:
- Keep existing accurate entries; add/update based on new source scan
- Remove entries that clearly conflict with current source content
- Ultra-condensed: bullet points only, no prose or explanations
- NEVER include API keys, secrets, passwords, or PII
- Stay within approximately 2000 tokens
- If a file's preview is truncated, preserve any existing memory entries for it

OUTPUT FORMAT:
{marker} ./memory/long-term.md [EDIT]
<updated memory>
{marker}

Only output the memory/long-term.md block, nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
# Memory update orchestrators
# ══════════════════════════════════════════════════════════════════════════════

def update_long_term_memory(
    cfg: Config,
    response_text: str,
    source_files_summary: str,
) -> bool:
    """Orchestrate the full long-term memory update flow (post-execution).

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

        print("\n[memory] Updating long-term project memory (post-execution)...")

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

        # 4. Parse the response for a memory/long-term.md block
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


def update_long_term_memory_from_source(cfg: Config, files_to_ai: list) -> bool:
    """Update long-term memory by scanning current source file contents.

    Called during the ai-steps workflow *before* each step execution so
    that the memory map always reflects the current state of the codebase
    (including changes from previously executed steps).  This ensures
    Claude has an accurate picture of where to find functions, variables,
    routes, and schema definitions even as the codebase evolves.

    The scan is lightweight: only the first ~600 chars of each file are
    sent (enough to capture imports, exports, class/function definitions,
    and route declarations), capped at ~10 000 chars total.

    Parameters
    ----------
    cfg : Config
        Resolved configuration.
    files_to_ai : list[FileData]
        Source file entries from ``add_source()``.  Only text/pdf files
        with ``ai_share=True`` are included in the digest.

    Returns
    -------
    bool
        True if the update succeeded (or was skipped because disabled/empty),
        False on any failure.  Never raises.
    """
    # Gate on memory configuration — skip entirely if disabled
    if not cfg.memory_enabled or not cfg.memory_long_term_enabled:
        return True

    try:
        from lib.providers.claude import prompt_claude
        from lib.validation import block_pattern

        # 1. Build a compact digest of source file previews
        source_digest = _build_source_digest(files_to_ai)
        if not source_digest.strip():
            # No scannable files — nothing to update
            return True

        print(f"[memory] Source scan: {len(source_digest)} chars digest from {len(files_to_ai)} file(s)")

        # 2. Load existing memory so Claude can merge/update
        existing_memory = load_long_term_memory(cfg.memory_dir)

        # 3. Build the source-scan prompt
        scan_prompt = _build_source_scan_prompt(existing_memory, source_digest)

        # 4. Call Claude — small/cheap: low max_tokens, low temperature for
        #    deterministic mapping, no web search, no thinking
        result = prompt_claude(
            api_key=cfg.anthropic_api_key,
            model=cfg.anthropic_model,
            system=(
                "You are a memory management assistant. Produce concise, "
                "structured project memory files. Never include secrets, "
                "API keys, or PII."
            ),
            messages=[{"role": "user", "content": scan_prompt}],
            max_tokens=4000,
            temperature=0.3,
            websearch=False,
            thinking_budget=0,
            stream=False,
        )

        if result["status"] != "ok":
            warn(
                f"[memory] Source scan Claude call failed: {result['status']} — "
                f"{result.get('error', '(no error detail)')}"
            )
            return False

        ai_text = result["data_response"]

        # 5. Parse the response for a memory/long-term.md block
        match = None
        for m in block_pattern.finditer(ai_text):
            source_path = m.group("source").strip()
            if source_path.endswith(_LONG_TERM_FILENAME):
                match = m
                break

        new_memory = match.group("content").strip() if match else ai_text.strip()

        if not new_memory:
            warn("[memory] Source scan produced empty memory — skipping save.")
            return False

        # 6. Save the updated memory
        save_long_term_memory(cfg.memory_dir, new_memory)
        print(f"[memory] ✓ Long-term memory updated from source scan ({len(new_memory)} chars)")
        return True

    except Exception as e:
        # Source-scan memory update must never crash the main workflow
        warn(f"[memory] Source scan update failed (non-fatal): {type(e).__name__}: {e}")
        return False
