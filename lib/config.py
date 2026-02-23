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
#   +++++ ./path/to/file.ext [TAG]   ← header: opening marker + path + tag
#   <file contents>                   ← body
#   +++++                             ← closing marker (standalone line)
#
# Because the opening marker and the closing marker are each on their own line
# starting with +++++, every well-formed response has an *even* count of lines
# that begin with +++++ — exactly 2 per block.
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

    # ── Paths ────────────────────────────────────────────────────────────────
    script_dir_name = os.path.basename(script_dir)
    logs_dir = os.path.join(script_dir, "logs")
    claude_output_dir = os.path.join(logs_dir, "claude")

    # Ensure output directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(claude_output_dir, exist_ok=True)

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
    )
