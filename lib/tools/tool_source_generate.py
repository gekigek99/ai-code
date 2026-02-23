"""
lib.tools.tool_source_generate — generate a source file list via Claude.

Public API:
    generate_source(cfg, prompt, tree_str, example_source=None) -> dict
        Ask Claude which files/dirs are relevant for a given prompt,
        based on the project's directory tree.  Returns a parsed list.
"""

import os
from typing import Any, Dict, List, Optional

import yaml

from lib.config import Config
from lib.export import export_md_file
from lib.prompt_builder import generate_prompt_for_gen_source
from lib.providers.claude import prompt_claude
from lib.validation import block_pattern, validate_claude_response


def generate_source(
    cfg: Config,
    prompt: str,
    tree_str: str,
    example_source: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask Claude to produce a YAML source list for a given prompt.

    Parameters
    ----------
    cfg : Config
        Resolved configuration (API key, model, system prompt, etc.).
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

    # ── Build the gen-source message ─────────────────────────────────────────
    message_content = generate_prompt_for_gen_source(prompt, example_source, tree_str)

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

    # Export the message content for debugging
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
