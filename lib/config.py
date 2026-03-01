"""
lib.config — YAML configuration loading and Config dataclass.

Public API:
    Config          — dataclass holding all resolved configuration values.
    load_config(script_dir) -> Config
        Read ``ai-code-prompt.yaml`` from *script_dir*, resolve all paths and
        defaults, and return a fully populated Config instance.

No module-level side effects.  ``load_config`` must be called explicitly from
``main()`` and the returned Config passed to all downstream functions.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# File-output-pattern suffix appended to the user-provided system prompt.
# This tells the LLM how to format file edits in its response so that the
# apply module can parse them reliably.
#
# Block format:
#   {'+'*5} ./path/to/file.ext [TAG]    ← header: opening marker + path + tag
#   <file contents>                   ← body
#   {'+'*5}                             ← closing marker (standalone line)
# ──────────────────────────────────────────────────────────────────────────────
_FILE_OUTPUT_PATTERN_SUFFIX = """

# ----- file output patterns ----- #

If you want to create or overwrite a file, return the full updated code in this format:
{marker} ./path/to/file.ext [EDIT]
  <complete contents of the file after changes>
{marker}

If you want to move or rename a file, write:
{marker} ./path/to/old/file.ext [MOVE] ./path/to/new/file.ext
  (don't write data, no content needed)
{marker}

If a file should be deleted, write:
{marker} ./path/to/file.ext [DELETE]
  (don't write data, no content needed)
{marker}

Do not use  ``` blocks.
Output only updated, new, or deleted files.
""".format(marker="+" * 5)


@dataclass
class Config:
    """Immutable-ish container for all resolved configuration values.

    Every downstream function receives (parts of) this object rather than
    reading module-level globals.
    """

    # ── Source / prompt ──────────────────────────────────────────────────────
    source: List[str]
    tree_dirs: List[str]
    exclude_patterns: List[str]
    prompt: str
    system: str  # Full system prompt with file-output-pattern suffix appended

    # ── Anthropic / Claude ───────────────────────────────────────────────────
    anthropic_api_key: str
    anthropic_model: str
    anthropic_max_tokens: int
    anthropic_max_tokens_think: int
    anthropic_temperature: float

    # ── Web search ───────────────────────────────────────────────────────────
    websearch: bool
    websearch_max_results: int

    # ── Paths ────────────────────────────────────────────────────────────────
    script_dir: str           # Directory containing ai-code.py
    script_dir_name: str      # Basename of script_dir (used for git filtering)
    logs_dir: str             # <script_dir>/logs/
    claude_output_dir: str    # <script_dir>/logs/claude/

    # ── Memory ───────────────────────────────────────────────────────────────
    memory_enabled: bool              # Master toggle for entire memory system
    memory_long_term_enabled: bool    # Toggle long-term project memory (memory/long-term.md)
    memory_short_term_enabled: bool   # Toggle short-term ai-steps memory (memory/short-term.md)
    memory_git_history_enabled: bool  # Toggle git history context (last N commits)
    memory_git_history_commits: int   # Number of recent commits to include
    memory_long_term_max_tokens: int  # Soft cap: triggers compaction when exceeded
    memory_short_term_max_tokens: int # Soft cap for short-term memory
    memory_auto_update: bool          # Auto-update project memory after each execution
    memory_dir: str                   # <script_dir>/memory/


def load_config(script_dir: str) -> Config:
    """Read ``ai-code-prompt.yaml`` from *script_dir* and return a Config.

    Parameters
    ----------
    script_dir : str
        Absolute path to the directory containing the entry-point script
        (``ai-code.py``).  The YAML config file is expected at
        ``<script_dir>/ai-code-prompt.yaml``.

    Returns
    -------
    Config
        Fully resolved configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    """
    config_path = os.path.join(script_dir, "ai-code-prompt.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # ── Extract top-level keys with safe defaults ────────────────────────────
    anthropic = raw.get("ANTHROPIC", {})
    source = raw.get("source") or ["."]
    tree_dirs = raw.get("tree_dirs") or source
    exclude_patterns = raw.get("exclude_patterns") or []
    prompt = raw.get("prompt") or ""

    # Always exclude the memory directory from source context — memory files
    # have their own dedicated injection path and must never be bundled as
    # regular source code.  Appended unconditionally so that user configs
    # cannot accidentally include memory content in the source payload.
    if "memory/" not in exclude_patterns:
        exclude_patterns.append("memory/")

    # Build the full system prompt by appending the file-output-pattern suffix
    raw_system = raw.get("system") or ""
    system = raw_system + _FILE_OUTPUT_PATTERN_SUFFIX

    # ── Anthropic settings ───────────────────────────────────────────────────
    anthropic_api_key = anthropic.get("API_KEY", "")
    anthropic_model = anthropic.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    anthropic_max_tokens = int(anthropic.get("MAX_TOKENS", 4000))
    anthropic_max_tokens_think = int(anthropic.get("MAX_TOKENS_THINK", 0))

    # Temperature is optional; default to 1.0 when absent or None
    raw_temp = anthropic.get("TEMPERATURE")
    anthropic_temperature = float(raw_temp) if raw_temp is not None else 1.0

    # ── Web search ───────────────────────────────────────────────────────────
    websearch = bool(raw.get("WEBSEARCH", False))
    websearch_max_results = int(raw.get("WEBSEARCH_MAX_RESULTS", 5))

    # ── Memory ───────────────────────────────────────────────────────────────
    # Follows the same nested-dict pattern as ANTHROPIC: grab the sub-dict
    # first, then pull individual keys with defaults.  Every key has a
    # sensible default so an entirely absent MEMORY section still yields a
    # working (enabled) memory subsystem.
    memory = raw.get("MEMORY", {})
    memory_enabled = bool(memory.get("ENABLED", True))
    memory_long_term_enabled = bool(memory.get("LONG_TERM_ENABLED", True))
    memory_short_term_enabled = bool(memory.get("SHORT_TERM_ENABLED", True))
    memory_git_history_enabled = bool(memory.get("GIT_HISTORY_ENABLED", True))
    memory_git_history_commits = int(memory.get("GIT_HISTORY_COMMITS", 30))
    memory_long_term_max_tokens = int(memory.get("LONG_TERM_MAX_TOKENS", 2000))
    memory_short_term_max_tokens = int(memory.get("SHORT_TERM_MAX_TOKENS", 1000))
    memory_auto_update = bool(memory.get("AUTO_UPDATE", True))

    # ── Paths ────────────────────────────────────────────────────────────────
    script_dir_name = os.path.basename(script_dir)
    logs_dir = os.path.join(script_dir, "logs")
    claude_output_dir = os.path.join(logs_dir, "claude")
    memory_dir = os.path.join(script_dir, "memory")

    # Ensure output directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(claude_output_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)

    return Config(
        source=source,
        tree_dirs=tree_dirs,
        exclude_patterns=exclude_patterns,
        prompt=prompt,
        system=system,
        anthropic_api_key=anthropic_api_key,
        anthropic_model=anthropic_model,
        anthropic_max_tokens=anthropic_max_tokens,
        anthropic_max_tokens_think=anthropic_max_tokens_think,
        anthropic_temperature=anthropic_temperature,
        websearch=websearch,
        websearch_max_results=websearch_max_results,
        script_dir=script_dir,
        script_dir_name=script_dir_name,
        logs_dir=logs_dir,
        claude_output_dir=claude_output_dir,
        memory_enabled=memory_enabled,
        memory_long_term_enabled=memory_long_term_enabled,
        memory_short_term_enabled=memory_short_term_enabled,
        memory_git_history_enabled=memory_git_history_enabled,
        memory_git_history_commits=memory_git_history_commits,
        memory_long_term_max_tokens=memory_long_term_max_tokens,
        memory_short_term_max_tokens=memory_short_term_max_tokens,
        memory_auto_update=memory_auto_update,
        memory_dir=memory_dir,
    )
