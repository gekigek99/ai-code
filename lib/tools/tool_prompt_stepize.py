"""
lib.tools.tool_prompt_stepize — decompose a prompt into ordered implementation steps.

Public API:
    stepize_prompt(cfg, expanded_prompt, source_paths) -> dict
        Ask Claude to break an expanded prompt into atomic, ordered steps.
        Returns steps with ``category`` fields and a ``feature_title``.

        Memory is updated inline: when ``cfg.memory_auto_update`` is True,
        Claude outputs an updated ``.ai-code/memory/long-term.md`` entry
        in the JSON ``files`` array alongside the steps.yaml entry.  This
        ensures memory is kept current whenever Claude reads project files,
        even during non-code-writing phases like step decomposition.

        Web search is forwarded from ``cfg.websearch`` to ``prompt_claude()``
        so Claude can search the web when enabled in the configuration.

        Exports ``stepize-userfullprompt.md`` showing exactly what data is
        sent to Claude (system prompt + all content blocks in order).
"""

import os
from typing import Any, Dict, List, Optional

import yaml

from lib.config import Config
from lib.files import FileData, add_source
from lib.memory import (
    build_memory_block,
    build_memory_update_instructions,
    extract_and_save_memory_from_response,
)
from lib.token_tracker import compute_and_display_breakdown
from lib.tree import get_directory_tree
from lib.prompt_builder import build_message_content, build_stepize_meta_prompt, build_readable_prompt_export
from lib.providers import prompt_llm
from lib.validation import parse_response_json, validate_claude_response, ResponseParseError
from lib.export import export_md_file
from lib.utils import warn


def stepize_prompt(
    cfg: Config,
    expanded_prompt: str,
    source_paths: List[str],
    exclude_patterns: Optional[List[str]] = None,
    tree_dirs: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Decompose an expanded prompt into ordered implementation steps.

    Claude receives the source file content, long-term project memory, and a
    meta-prompt instructing it to produce a steps.yaml EDIT entry in its
    JSON ``files`` array.

    When ``cfg.memory_auto_update`` is True, inline memory update instructions
    are appended to the meta-prompt so Claude also outputs an updated
    ``.ai-code/memory/long-term.md`` entry.  This ensures the project memory
    is refreshed whenever Claude reads source files — not just during code
    execution phases.  The memory entry is extracted and saved to disk, then
    removed from the parsed response before extracting the steps.yaml entry.

    Web search is forwarded from ``cfg.websearch`` and
    ``cfg.websearch_max_results`` to ``prompt_claude()``, allowing Claude
    to search the web during step decomposition when enabled.

    Exports ``stepize-userfullprompt.md`` via ``build_readable_prompt_export()``
    so the artifact exactly represents what Claude receives.

    Parameters
    ----------
    cfg : Config
        Resolved configuration.
    expanded_prompt : str
        The comprehensive implementation specification to decompose.
    source_paths : list[str]
        File/directory paths to read as context.
    exclude_patterns : list[str], optional
        Glob exclusion patterns.  Defaults to ``cfg.exclude_patterns``.
    tree_dirs : list[str], optional
        Directories for tree display.  Defaults to ``cfg.tree_dirs``.
    output_dir : str, optional
        Directory for saving artifacts.  Defaults to ``cfg.claude_output_dir``.

    Returns
    -------
    dict
        ``status``          — ``"ok"`` | ``"error"``
        ``steps``           — list[dict] of step definitions (empty on error)
        ``feature_title``   — str, short label for the overall feature
        ``raw_response``    — full Claude response text
        ``thinking``        — thinking content (if extended thinking enabled)
        ``error``           — error message (None on success)

    Each step dict has keys:
        ``number``    — int, 1-based step index
        ``title``     — str, human-readable step title
        ``category``  — str, affected system area (e.g. "database", "api", "admin")
        ``prompt``    — str, the full prompt text for implementing the step
        ``source``    — list[str], file/directory paths needed for the step
    """
    if exclude_patterns is None:
        exclude_patterns = cfg.exclude_patterns
    if tree_dirs is None:
        tree_dirs = cfg.tree_dirs
    if output_dir is None:
        output_dir = cfg.claude_output_dir

    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Discover and read source files ------------------------------------
    files_to_ai: List[FileData] = []
    files_to_ai, _ = add_source(
        files_to_ai, source_paths, exclude_patterns,
    )

    # -- 2. Build directory tree ----------------------------------------------
    _, ai_file_listing = get_directory_tree(
        tree_dirs, exclude_patterns, files_to_ai,
    )

    # -- 2b. Build memory context (long-term only, no short-term for stepize) --
    # During stepize (Phase 2) the expanded prompt IS the primary context —
    # short-term memory would be redundant.  Long-term memory provides Claude
    # with project-wide awareness (architecture, conventions, schema) so it can
    # produce more accurate step decompositions and source-file selections.
    memory_result = build_memory_block(cfg, include_short_term=False)
    memory_block = memory_result.text

    # -- 2c. Build inline memory update instructions --------------------------
    # Appended to the meta-prompt so Claude outputs an updated memory file
    # entry alongside the steps.yaml entry.  This keeps the project memory
    # current whenever Claude reads source files — even during non-code-writing
    # phases like step decomposition.
    memory_instructions = build_memory_update_instructions(cfg)
    if memory_instructions:
        print(f"[tool_prompt_stepize] Inline memory update instructions appended ({len(memory_instructions)} chars)")

    # Log websearch status for this specific tool invocation
    if cfg.websearch:
        print(f"[tool_prompt_stepize] Web search: ENABLED (max_results={cfg.websearch_max_results})")

    # -- 3. Build the stepize meta-prompt -------------------------------------
    meta_prompt = build_stepize_meta_prompt(expanded_prompt)

    # Append memory update instructions to the meta-prompt so they are part
    # of the prompt Claude sees.  The stepize meta-prompt explicitly allows
    # the memory file entry alongside the steps.yaml entry.
    full_meta_prompt = meta_prompt + memory_instructions if memory_instructions else meta_prompt

    # -- 4. Build message content ---------------------------------------------
    # memory_block is prepended as the first content item so Claude sees
    # project context before source files and the meta-prompt.
    message_content, _ = build_message_content(
        files_to_ai, full_meta_prompt, ai_file_listing, memory_block=memory_block,
    )

    # -- Display token breakdown ----------------------------------------------
    # user_prompt = expanded_prompt (the semantic user content being decomposed)
    # tool_context = meta_prompt wrapper instructions (total meta minus embedded prompt)
    # memory_instructions tracked separately for accurate breakdown
    tool_context_chars = max(0, len(meta_prompt) - len(expanded_prompt))
    compute_and_display_breakdown(
        system=cfg.system,
        memory_result=memory_result,
        files_to_ai=files_to_ai,
        ai_file_listing=ai_file_listing,
        user_prompt=expanded_prompt,
        tool_context_chars=tool_context_chars,
        memory_instructions=memory_instructions,
    )

    # -- Export exactly what Claude receives -----------------------------------
    # build_readable_prompt_export produces a human-readable string combining
    # system prompt + all message_content blocks in order, so the artifact
    # exactly represents what the LLM sees.
    readable_prompt = build_readable_prompt_export(cfg.system, message_content)
    export_md_file(readable_prompt, "stepize-userfullprompt.md", output_dir)

    # -- 5. Call the LLM ------------------------------------------------------
    # Provider-agnostic dispatch — prompt_llm reads cfg.provider and forwards
    # to the appropriate backend.  Web search forwarding, system prompt, and
    # generation knobs are all encapsulated by the dispatcher.
    print(f"[tool_prompt_stepize] Asking LLM to decompose prompt into steps (provider={cfg.provider}, model={cfg.model})...")
    result = prompt_llm(
        cfg=cfg,
        messages=[{"role": "user", "content": message_content}],
        stream=True,
        recv_path=os.path.join(output_dir, "stepize-recv.md"),
    )

    if result["status"] != "ok":
        error_msg = result.get("error") or "No response received"
        print(f"\n[tool_prompt_stepize] Error: {error_msg}")
        return {
            "status": "error",
            "steps": [],
            "feature_title": "",
            "raw_response": result.get("data_response", ""),
            "thinking": result.get("thinking_content", ""),
            "error": error_msg,
        }

    data_response = result["data_response"]
    thinking_content = result.get("thinking_content", "")

    # -- 6. Validate response -------------------------------------------------
    validate_claude_response(data_response)

    # -- 7. Export artifacts ---------------------------------------------------
    if data_response:
        export_md_file(data_response, "stepize-clauderesponse.md", output_dir)
    if thinking_content:
        export_md_file(thinking_content, "stepize-thinking.md", output_dir)
    if result.get("raw_data"):
        raw_data_str = "\n\n".join(
            f"Event Type: {item['type']}\nData: {item['event']}"
            for item in result["raw_data"]
        )
        export_md_file(raw_data_str, "stepize-rawdata.md", output_dir)

    # -- 7b. Extract and save memory from response ----------------------------
    # Parse JSON, extract the .ai-code/memory/long-term.md entry, save it to
    # cfg.memory_long_term_dir, and return the parsed dict with the memory
    # entry removed.  Downstream code operates on the dict, not raw text.
    parsed_response = extract_and_save_memory_from_response(cfg, data_response)

    # -- 8. Extract steps.yaml from parsed JSON files array -------------------
    steps_yaml = ""
    for entry in parsed_response.get("files", []):
        path = entry.get("path", "")
        if ("steps.yaml" in path or "steps.yml" in path) and entry.get("action") == "EDIT":
            steps_yaml = entry.get("content", "").strip()
            break

    if not steps_yaml:
        print("[tool_prompt_stepize] WARNING: No steps.yaml entry found in response")
        return {
            "status": "error",
            "steps": [],
            "feature_title": "",
            "raw_response": data_response,
            "thinking": thinking_content,
            "error": "No steps.yaml entry found in Claude response",
        }

    # Parse the YAML
    try:
        parsed = yaml.safe_load(steps_yaml)
        if isinstance(parsed, dict):
            raw_steps = parsed.get("steps", [])
            # Extract the top-level feature_title for commit messages
            feature_title = parsed.get("feature_title", "")
        elif isinstance(parsed, list):
            raw_steps = parsed
            feature_title = ""
        else:
            raw_steps = []
            feature_title = ""
    except yaml.YAMLError as e:
        print(f"[tool_prompt_stepize] WARNING: Failed to parse steps YAML: {e}")
        return {
            "status": "error",
            "steps": [],
            "feature_title": "",
            "raw_response": data_response,
            "thinking": thinking_content,
            "error": f"Failed to parse steps YAML: {e}",
        }

    # Ensure feature_title is a non-empty string; fall back to a generic label
    if not feature_title or not isinstance(feature_title, str):
        feature_title = "ai-steps"

    # Normalise step dicts to ensure required keys exist
    steps: List[Dict[str, Any]] = []
    for i, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            warn(f"[tool_prompt_stepize] Skipping non-dict step entry at index {i}")
            continue

        step = {
            "number": raw_step.get("number", i + 1),
            "title": raw_step.get("title", f"Step {i + 1}"),
            "category": raw_step.get("category", "general"),
            "prompt": raw_step.get("prompt", ""),
            "source": raw_step.get("source", []),
        }

        # Ensure category is a non-empty string
        if not step["category"] or not isinstance(step["category"], str):
            step["category"] = "general"

        # Ensure source is a list of strings
        if isinstance(step["source"], str):
            step["source"] = [step["source"]]
        step["source"] = [str(s) for s in step["source"] if s]

        if not step["prompt"]:
            warn(f"[tool_prompt_stepize] Step {step['number']} has no prompt text")

        steps.append(step)

    # Save the parsed steps
    export_md_file(steps_yaml, "steps.yaml", output_dir)

    print(f"\n[tool_prompt_stepize] Feature: {feature_title}")
    print(f"[tool_prompt_stepize] Decomposed into {len(steps)} step(s):")
    for step in steps:
        print(f"  Step {step['number']} [{step['category']}]: {step['title']} ({len(step['source'])} source entries)")

    return {
        "status": "ok",
        "steps": steps,
        "feature_title": feature_title,
        "raw_response": data_response,
        "thinking": thinking_content,
        "error": None,
    }
