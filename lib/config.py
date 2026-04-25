"""
lib.config — YAML configuration loading and Config dataclass.

Public API:
    Config          — dataclass holding all resolved configuration values.
    load_config(script_dir) -> Config
        Read ``ai-code-prompt.yaml`` from *script_dir*, resolve all paths and
        defaults, and return a fully populated Config instance.

Provider selection:
    The active LLM provider is chosen via the top-level ``PROVIDER`` YAML key
    (``anthropic`` or ``openrouter``).  Each provider has its own sub-section
    with ``API_KEY`` and ``MODEL``.  Provider-agnostic generation knobs
    (``MAX_TOKENS``, ``MAX_TOKENS_THINK``, ``TEMPERATURE``) live at the top
    level and apply to whichever provider is active.

No module-level side effects.  ``load_config`` must be called explicitly from
``main()`` and the returned Config passed to all downstream functions.
"""

import os
from dataclasses import dataclass
from typing import List

import yaml


# ------------------------------------------------------------------------------
# Allowed provider identifiers.  Centralised here so the loader, the dispatcher,
# and any future tooling all reference the same canonical list.
# ------------------------------------------------------------------------------
_ALLOWED_PROVIDERS = ("anthropic", "openrouter")


# ------------------------------------------------------------------------------
# JSON output suffix — instructs the LLM to respond with structured JSON
# containing file operations instead of free-form text.
# ------------------------------------------------------------------------------

def _build_json_output_suffix(patch_enabled: bool) -> str:
    """Return the system prompt suffix that instructs the LLM to output JSON.

    When *patch_enabled* is False the PATCH action and all its rules are
    omitted entirely so the LLM never attempts partial-file edits.
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

Do NOT include any preamble, reasoning, thinking, commentary, or explanation outside the JSON object. Your entire response must start with `{{` and end with `}}`. Any text outside the JSON — including phrases like "Here is the JSON", "Let me analyze", or "I'll implement" — is a format violation.
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

    Provider-agnostic LLM access:
        ``provider``     selects the active backend ("anthropic" | "openrouter").
        ``api_key``      resolved from the active provider's section.
        ``model``        resolved from the active provider's section.
        ``max_tokens``,
        ``max_tokens_think``,
        ``temperature``  apply to whichever provider is active.

    Callers should NEVER hard-code provider-specific names — use
    ``lib.providers.prompt_llm(cfg, ...)`` to dispatch transparently.
    """

    # -- Source / prompt ------------------------------------------------------
    source: List[str]
    tree_dirs: List[str]
    exclude_patterns: List[str]
    prompt: str
    system: str  # Full system prompt with JSON output suffix appended

    # -- LLM provider (provider-agnostic) -------------------------------------
    provider: str            # "anthropic" | "openrouter"
    api_key: str             # Resolved from <provider>.API_KEY
    model: str               # Resolved from <provider>.MODEL
    max_tokens: int          # Top-level MAX_TOKENS
    max_tokens_think: int    # Top-level MAX_TOKENS_THINK
    temperature: float       # Top-level TEMPERATURE (default 1.0)

    # -- Features -------------------------------------------------------------
    patch_enabled: bool           # Allow PATCH action in JSON responses
    sound_enabled: bool           # Play audible bell when workflows complete

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


def _resolve_provider_section(raw: dict, provider: str) -> dict:
    """Return the provider-specific YAML section as a dict.

    Performs case-insensitive lookup against the YAML top-level keys so that
    ``anthropic:``, ``Anthropic:``, and ``ANTHROPIC:`` all resolve correctly.
    This is purely defensive — the canonical key is lowercase per the example
    config — but it spares users from confusing silent failures when they
    typo the section header.
    """
    target = provider.lower()
    for key, value in raw.items():
        if isinstance(key, str) and key.lower() == target:
            return value if isinstance(value, dict) else {}
    return {}


def load_config(script_dir: str) -> Config:
    """Read ``ai-code-prompt.yaml`` from *script_dir* and return a Config.

    Provider resolution:
      1. ``PROVIDER`` top-level key selects the active backend.
      2. ``<provider>:`` sub-section supplies ``API_KEY`` and ``MODEL``.
      3. ``MAX_TOKENS``, ``MAX_TOKENS_THINK``, ``TEMPERATURE`` are top-level
         and provider-agnostic.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If ``PROVIDER`` is unknown, or the chosen provider section is missing
        ``API_KEY`` or ``MODEL``.
    """
    config_path = os.path.join(script_dir, "ai-code-prompt.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # -- Source / prompt -----------------------------------------------------
    # Sanitize all list-type config values to remove None entries that arise
    # from YAML lines like ``- # comment`` (which parse as None list items).
    source = _sanitize_string_list(raw.get("source")) or ["."]
    tree_dirs = _sanitize_string_list(raw.get("tree_dirs")) or source
    exclude_patterns = _sanitize_string_list(raw.get("exclude_patterns")) or []
    prompt = raw.get("prompt") or ""

    # Always exclude the .ai-code directory from source context — memory files
    # have their own dedicated injection path and must never be bundled as
    # regular source code.
    if ".ai-code/" not in exclude_patterns:
        exclude_patterns.append(".ai-code/")

    # -- System prompt + JSON output suffix ---------------------------------
    raw_system = raw.get("system") or ""
    patch_enabled = bool(raw.get("PATCH_ENABLED", True))
    suffix = _build_json_output_suffix(patch_enabled)
    system = raw_system + suffix

    # -- LLM provider selection ---------------------------------------------
    # Default to "anthropic" so configs predating the provider system still
    # work after upgrade (assuming they retain an ``anthropic:`` section).
    provider_raw = raw.get("PROVIDER")
    provider = (provider_raw or "anthropic").strip().lower()
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(
            f"Unknown PROVIDER {provider!r}. "
            f"Allowed: {', '.join(_ALLOWED_PROVIDERS)}."
        )

    provider_section = _resolve_provider_section(raw, provider)
    api_key = (provider_section.get("API_KEY") or "").strip()
    model = (provider_section.get("MODEL") or "").strip()

    if not api_key:
        raise ValueError(
            f"Missing API_KEY in '{provider}' section of {config_path}. "
            f"Add:\n  {provider}:\n    API_KEY: <your-key>\n    MODEL: <model>"
        )
    if not model:
        raise ValueError(
            f"Missing MODEL in '{provider}' section of {config_path}. "
            f"Add:\n  {provider}:\n    API_KEY: <your-key>\n    MODEL: <model>"
        )

    # -- Provider-agnostic generation knobs ---------------------------------
    # Top-level so they apply uniformly regardless of which provider is on.
    max_tokens = int(raw.get("MAX_TOKENS", 4000))
    max_tokens_think = int(raw.get("MAX_TOKENS_THINK", 0))
    raw_temp = raw.get("TEMPERATURE")
    temperature = float(raw_temp) if raw_temp is not None else 1.0

    # -- Features ------------------------------------------------------------
    sound_enabled = bool(raw.get("SOUND_ENABLED", True))

    # -- Web search ---------------------------------------------------------
    websearch = bool(raw.get("WEBSEARCH", False))
    websearch_max_results = int(raw.get("WEBSEARCH_MAX_RESULTS", 5))

    # -- Memory -------------------------------------------------------------
    memory = raw.get("MEMORY", {}) or {}
    memory_enabled = bool(memory.get("ENABLED", True))
    memory_long_term_enabled = bool(memory.get("LONG_TERM_ENABLED", True))
    memory_short_term_enabled = bool(memory.get("SHORT_TERM_ENABLED", True))
    memory_git_history_enabled = bool(memory.get("GIT_HISTORY_ENABLED", True))
    memory_git_history_commits = int(memory.get("GIT_HISTORY_COMMITS", 30))
    memory_long_term_max_tokens = int(memory.get("LONG_TERM_MAX_TOKENS", 2000))
    memory_short_term_max_tokens = int(memory.get("SHORT_TERM_MAX_TOKENS", 1000))
    memory_auto_update = bool(memory.get("AUTO_UPDATE", True))

    # -- Paths --------------------------------------------------------------
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
        provider=provider,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        max_tokens_think=max_tokens_think,
        temperature=temperature,
        patch_enabled=patch_enabled,
        sound_enabled=sound_enabled,
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

