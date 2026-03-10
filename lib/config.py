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


# ------------------------------------------------------------------------------
# JSON output suffix — instructs Claude to respond with structured JSON
# containing file operations instead of free-form text.
# ------------------------------------------------------------------------------

def _build_json_output_suffix(patch_enabled: bool) -> str:
    """Return the system prompt suffix that instructs Claude to output JSON.

    When *patch_enabled* is False the PATCH action and all its rules are
    omitted entirely so Claude never attempts partial-file edits.
    """

    # -- PATCH action definition (conditional) -----------------------------
    patch_action_block = ""
    if patch_enabled:
        patch_action_block = """
- **PATCH**: Make small, targeted changes to an existing file. Requires a `patches` array (no `content` key). Each object in the array has:
  - `"comment"`: human-readable explanation of what this hunk changes.
  - `"find"`: exact text to locate in the file (must match exactly, including indentation and whitespace).
  - `"replace"`: replacement text (may be empty string `""` to delete the matched text)."""

    # -- PATCH rules (conditional) -----------------------------------------
    patch_rules_block = ""
    if patch_enabled:
        patch_rules_block = """
### PATCH rules
- Every `find` string must match EXACTLY in the target file. If it does not match, you have made an error.
- Each patch replaces the first occurrence found.
- Multiple patches in one entry are applied top to bottom.
- Use PATCH when changing less than ~40% of a file. Use EDIT for new files or major rewrites.
- An empty `replace` string deletes the matched text."""

    # -- PATCH schema example (conditional) --------------------------------
    patch_schema_example = ""
    if patch_enabled:
        patch_schema_example = """,
    {
      "action": "PATCH",
      "path": "./relative/path/to/existing.ext",
      "patches": [
        {
          "comment": "describe what this hunk changes",
          "find": "exact text to find",
          "replace": "replacement text"
        }
      ]
    }"""

    return f"""

# ----- JSON output format ----- #

Your COMPLETE response must be a single valid JSON object. Do not include any text, explanation, markdown, or code fences outside the JSON.

## Schema

```
{{
  "files": [
    {{
      "action": "EDIT",
      "path": "./relative/path/to/file.ext",
      "content": "full file content as a JSON string"
    }},
    {{
      "action": "DELETE",
      "path": "./relative/path/to/file.ext"
    }},
    {{
      "action": "MOVE",
      "path": "./relative/path/to/old.ext",
      "destination": "./relative/path/to/new.ext"
    }}{patch_schema_example}
  ]
}}
```

## Action definitions

- **EDIT**: Create a new file or completely rewrite an existing file. `content` is required and contains the full file contents. All special characters (newlines, quotes, backslashes, tabs) must be properly JSON-escaped.
- **DELETE**: Remove a file. Only `action` and `path` required. No `content`, no `patches`.
- **MOVE**: Move or rename a file. Requires `path` (source) and `destination` (target). No `content`, no `patches`.{patch_action_block}
{patch_rules_block}

## JSON string escaping

All string values in the JSON must be valid JSON strings. Escape newlines as \\n, tabs as \\t, double quotes as \\", and backslashes as \\\\. Do not use actual newline characters inside JSON string values — use the \\n escape sequence.

## Output constraint

Output ONLY the JSON object. No markdown, no code fences, no explanatory text before or after the JSON.
"""


def _sanitize_string_list(raw_list) -> List[str]:
    """Filter a YAML-parsed list to contain only non-empty strings.

    YAML list entries like ``- # comment`` parse as ``None`` because the
    ``-`` creates a list item and ``# ...`` is a comment (producing no
    value).  This silently drops such entries plus any other non-string
    or empty-string entries, preventing downstream ``os.path.abspath(None)``
    TypeError crashes.

    Parameters
    ----------
    raw_list : any
        The value parsed from YAML.  Expected to be a list but may be
        None or a non-list type.

    Returns
    -------
    list[str]
        Cleaned list containing only non-empty strings.
    """
    if not isinstance(raw_list, list):
        return []
    return [str(entry) for entry in raw_list if entry is not None and str(entry).strip()]


@dataclass
class Config:
    """Immutable-ish container for all resolved configuration values.

    Every downstream function receives (parts of) this object rather than
    reading module-level globals.
    """

    # -- Source / prompt ------------------------------------------------------
    source: List[str]
    tree_dirs: List[str]
    exclude_patterns: List[str]
    prompt: str
    system: str  # Full system prompt with JSON output suffix appended

    # -- Anthropic / Claude ---------------------------------------------------
    anthropic_api_key: str
    anthropic_model: str
    anthropic_max_tokens: int
    anthropic_max_tokens_think: int
    anthropic_temperature: float

    # -- Features -------------------------------------------------------------
    patch_enabled: bool           # Allow PATCH action in JSON responses

    # -- Web search -----------------------------------------------------------
    websearch: bool
    websearch_max_results: int

    # -- Paths ----------------------------------------------------------------
    script_dir: str           # Directory containing ai-code.py
    script_dir_name: str      # Basename of script_dir (used for git filtering)
    logs_dir: str             # <script_dir>/logs/
    claude_output_dir: str    # <script_dir>/logs/claude/

    # -- Memory ---------------------------------------------------------------
    memory_enabled: bool              # Master toggle for entire memory system
    memory_long_term_enabled: bool    # Toggle long-term project memory
    memory_short_term_enabled: bool   # Toggle short-term ai-steps memory
    memory_git_history_enabled: bool  # Toggle git history context (last N commits)
    memory_git_history_commits: int   # Number of recent commits to include
    memory_long_term_max_tokens: int  # Soft cap: triggers compaction when exceeded
    memory_short_term_max_tokens: int # Soft cap for short-term memory
    memory_auto_update: bool          # Auto-update project memory after each execution
    # Long-term memory lives in the *parent* project at .ai-code/memory/ so it
    # is tracked in the master project's git history, not the ai-code submodule.
    memory_long_term_dir: str         # <parent_of_script_dir>/.ai-code/memory/
    # Short-term memory lives inside the script directory at memory/ since it
    # is ephemeral workflow state that should not pollute the master project.
    memory_short_term_dir: str        # <script_dir>/memory/


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

    # -- Extract top-level keys with safe defaults ----------------------------
    anthropic = raw.get("ANTHROPIC", {})

    # Sanitize all list-type config values to remove None entries that arise
    # from YAML lines like ``- # comment`` (which parse as None list items).
    # This prevents TypeError crashes in downstream code that calls
    # os.path.abspath() or similar on each entry.
    source = _sanitize_string_list(raw.get("source")) or ["."]
    tree_dirs = _sanitize_string_list(raw.get("tree_dirs")) or source
    exclude_patterns = _sanitize_string_list(raw.get("exclude_patterns")) or []
    prompt = raw.get("prompt") or ""

    # Always exclude the .ai-code directory from source context — memory files
    # have their own dedicated injection path and must never be bundled as
    # regular source code.
    if ".ai-code/" not in exclude_patterns:
        exclude_patterns.append(".ai-code/")

    # Build the full system prompt by appending the JSON output suffix.
    raw_system = raw.get("system") or ""
    patch_enabled = raw.get("PATCH_ENABLED") or ""
    suffix = _build_json_output_suffix(bool(patch_enabled))
    system = raw_system + suffix

    # -- Anthropic settings ---------------------------------------------------
    anthropic_api_key = anthropic.get("API_KEY", "")
    anthropic_model = anthropic.get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    anthropic_max_tokens = int(anthropic.get("MAX_TOKENS", 4000))
    anthropic_max_tokens_think = int(anthropic.get("MAX_TOKENS_THINK", 0))

    # Temperature is optional; default to 1.0 when absent or None
    raw_temp = anthropic.get("TEMPERATURE")
    anthropic_temperature = float(raw_temp) if raw_temp is not None else 1.0

    # -- Features -------------------------------------------------------------
    patch_enabled = bool(raw.get("PATCH_ENABLED", True))

    # -- Web search -----------------------------------------------------------
    websearch = bool(raw.get("WEBSEARCH", False))
    websearch_max_results = int(raw.get("WEBSEARCH_MAX_RESULTS", 5))

    # -- Memory ---------------------------------------------------------------
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

    # -- Paths ----------------------------------------------------------------
    script_dir_name = os.path.basename(script_dir)
    logs_dir = os.path.join(script_dir, "logs")
    claude_output_dir = os.path.join(logs_dir, "claude")

    # Long-term memory directory: lives in the *parent* project at
    # .ai-code/memory/ so that memory files are version-controlled alongside
    # the master project's source code.
    memory_long_term_dir = os.path.join(script_dir, "..", ".ai-code", "memory")
    memory_long_term_dir = os.path.normpath(memory_long_term_dir)

    # Short-term memory directory: lives inside the script directory at memory/
    # because it is ephemeral workflow state (ai-steps context).
    memory_short_term_dir = os.path.join(script_dir, "memory")
    memory_short_term_dir = os.path.normpath(memory_short_term_dir)

    # Ensure output directories exist
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(claude_output_dir, exist_ok=True)
    os.makedirs(memory_long_term_dir, exist_ok=True)
    os.makedirs(memory_short_term_dir, exist_ok=True)

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
        patch_enabled=patch_enabled,
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
        memory_long_term_dir=memory_long_term_dir,
        memory_short_term_dir=memory_short_term_dir,
    )
