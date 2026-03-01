"""
lib.tools.tool_source_generate — generate a source file list via Claude.

Public API:
    generate_source(cfg, prompt, tree_str, example_source=None,
                    output_dir=None, include_short_term_memory=False) -> dict
        Ask Claude which files/dirs are relevant for a given prompt,
        based on the project's directory tree.  Returns a parsed list.

        Memory context (long-term project memory, short-term workflow
        memory, git commit history) is automatically loaded from ``cfg``
        and injected into the prompt so Claude can make informed decisions
        about which source files are relevant — e.g. knowing which files
        were recently modified, what the project architecture looks like,
        and what the current workflow step involves.
"""

import os
from typing import Any, Dict, List, Optional

import yaml

from lib.config import Config
from lib.export import export_md_file
from lib.memory import build_memory_block
from lib.token_tracker import TokenBreakdown, display_token_breakdown
from lib.prompt_builder import generate_prompt_for_gen_source
from lib.providers.claude import prompt_claude
from lib.validation import block_pattern, validate_claude_response


def generate_source(
    cfg: Config,
    prompt: str,
    tree_str: str,
    example_source: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    include_short_term_memory: bool = False,
) -> Dict[str, Any]:
    """Ask Claude to produce a YAML source list for a given prompt.

    Memory context (long-term project memory, short-term workflow memory,
    git commit history with per-file numstat diffs) is automatically built
    from ``cfg`` and prepended as the first content block so Claude has
    project awareness when deciding which files are relevant.  This mirrors
    the memory injection pattern used by ``execute_prompt``.

    Parameters
    ----------
    cfg : Config
        Resolved configuration (API key, model, system prompt, memory
        settings, etc.).
    prompt : str
        The user prompt — Claude uses it to decide which files are relevant
        (its instructions are NOT executed).
    tree_str : str
        Clean human-readable directory tree (no ANSI).
    example_source : list[str], optional
        Existing source config to show Claude as a formatting example.
        Defaults to ``cfg.source``.
    output_dir : str, optional
        Directory for saving artifacts.  Defaults to ``cfg.claude_output_dir``.
    include_short_term_memory : bool
        When True *and* short-term memory is enabled in config, the
        ``short-term.md`` contents are included in the memory block.
        Typically True during ai-steps Phase 2+ where inter-step context
        has already been established.

    Returns
    -------
    dict
        ``status``         — ``"ok"`` | ``"error"``
        ``source_list``    — list[str] of file/directory paths (empty on error)
        ``source_yaml``    — raw YAML string extracted from response (empty on error)
        ``raw_response``   — full Claude response text
        ``thinking``       — thinking content (if extended thinking enabled)
        ``error``          — error message (None on success)
    """
    if example_source is None:
        example_source = cfg.source
    if output_dir is None:
        output_dir = cfg.claude_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # ── Token breakdown tracker ──────────────────────────────────────────────
    breakdown = TokenBreakdown()
    breakdown.system = len(cfg.system) // 4

    # ── Build memory context block ───────────────────────────────────────────
    # Assemble long-term memory, short-term memory (if requested), and git
    # history (commit messages, titles, per-file line diffs) into a single
    # block.  This gives Claude project awareness — architecture knowledge,
    # recent changes, and workflow context — when deciding which files are
    # relevant for the given prompt.  Without this, Claude would rely solely
    # on the directory tree and prompt text, missing important context about
    # which files were recently modified or are architecturally significant.
    memory_result = build_memory_block(
        cfg, include_short_term=include_short_term_memory,
    )
    memory_block = memory_result.text

    # Transfer memory token counts to breakdown
    breakdown.long_term_memory = memory_result.long_term_tokens
    breakdown.short_term_memory = memory_result.short_term_tokens
    breakdown.git_history = memory_result.git_history_tokens

    if memory_block:
        print(f"[tool_source_generate] Memory block: {len(memory_block)} chars "
              f"(~{len(memory_block) // 4} tokens | "
              f"LT={memory_result.long_term_tokens} ST={memory_result.short_term_tokens} "
              f"Git={memory_result.git_history_tokens})")

    # ── Build the gen-source message ─────────────────────────────────────────
    message_content = generate_prompt_for_gen_source(prompt, example_source, tree_str)

    # Track prompt tokens — the gen-source meta-prompt + user prompt + tree
    prompt_tokens = sum(len(part.get("text", "")) for part in message_content) // 4
    breakdown.prompt = prompt_tokens

    # Prepend memory block as the first content item so Claude sees project
    # context (architecture, recent commits, workflow state) before the
    # directory tree and prompt — establishing a knowledge baseline that
    # informs source file selection.  Same injection pattern as
    # build_message_content() in prompt_builder.py.
    if memory_block:
        message_content.insert(0, {"type": "text", "text": memory_block})

    # gen-source has no file data (only tree + memory + prompt)
    breakdown.file_data = 0

    # ── Display token breakdown graph ────────────────────────────────────────
    display_token_breakdown(breakdown)

    print("\n[tool_source_generate] Asking Claude for relevant source list...")

    # ── Call Claude ──────────────────────────────────────────────────────────
    result = prompt_claude(
        api_key=cfg.anthropic_api_key,
        model=cfg.anthropic_model,
        system=cfg.system,
        messages=[{"role": "user", "content": message_content}],
        max_tokens=cfg.anthropic_max_tokens,
        temperature=cfg.anthropic_temperature,
        websearch=cfg.websearch,
        websearch_max_results=cfg.websearch_max_results,
        thinking_budget=cfg.anthropic_max_tokens_think,
        stream=True,
        recv_path=os.path.join(output_dir, "gen-source-recv.md"),
    )

    # Export the message content for debugging — includes memory block so
    # developers can inspect exactly what context Claude received.
    export_md_file(
        f"{cfg.system}\n\n{message_content}",
        "gen-source-message_content.md",
        output_dir,
    )

    if result["status"] != "ok":
        error_msg = result.get("error") or "No response received from Claude"
        print(f"\n[tool_source_generate] Error: {error_msg}")
        return {
            "status": "error",
            "source_list": [],
            "source_yaml": "",
            "raw_response": result.get("data_response", ""),
            "thinking": result.get("thinking_content", ""),
            "error": error_msg,
        }

    data_response = result["data_response"]
    thinking_content = result.get("thinking_content", "")

    # ── Validate response structure ──────────────────────────────────────────
    validate_claude_response(data_response)

    # ── Export artifacts ─────────────────────────────────────────────────────
    if data_response:
        export_md_file(data_response, "gen-source-clauderesponse.md", output_dir)
    if thinking_content:
        export_md_file(thinking_content, "gen-source-thinking.md", output_dir)
    if result.get("raw_data"):
        raw_data_str = "\n\n".join(
            f"Event Type: {item['type']}\nData: {item['event']}"
            for item in result["raw_data"]
        )
        export_md_file(raw_data_str, "gen-source-rawdata.md", output_dir)

    # ── Extract source.md block and parse YAML ───────────────────────────────
    source_yaml = ""
    source_list: List[str] = []

    for m in block_pattern.finditer(data_response):
        source_path = m.group("source").strip()
        content = m.group("content").strip()
        if "source.md" in source_path:
            source_yaml = content
            break

    if not source_yaml:
        print("[tool_source_generate] WARNING: No source.md block found in response")
        return {
            "status": "error",
            "source_list": [],
            "source_yaml": "",
            "raw_response": data_response,
            "thinking": thinking_content,
            "error": "No source.md block found in Claude response",
        }

    # Parse the YAML to extract the source list
    try:
        parsed = yaml.safe_load(source_yaml)
        if isinstance(parsed, dict):
            source_list = parsed.get("source", [])
        elif isinstance(parsed, list):
            # Handle case where YAML is just a list without 'source' key
            source_list = parsed
        else:
            source_list = []

        # Ensure all entries are strings
        source_list = [str(s) for s in source_list if s]

    except yaml.YAMLError as e:
        print(f"[tool_source_generate] WARNING: Failed to parse YAML: {e}")
        return {
            "status": "error",
            "source_list": [],
            "source_yaml": source_yaml,
            "raw_response": data_response,
            "thinking": thinking_content,
            "error": f"Failed to parse source YAML: {e}",
        }

    print(f"\n[tool_source_generate] Generated source list ({len(source_list)} entries):")
    for entry in source_list:
        print(f"  - {entry}")

    return {
        "status": "ok",
        "source_list": source_list,
        "source_yaml": source_yaml,
        "raw_response": data_response,
        "thinking": thinking_content,
        "error": None,
    }
