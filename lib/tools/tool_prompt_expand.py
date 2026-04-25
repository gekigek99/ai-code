"""
lib.tools.tool_prompt_expand — expand a minimal prompt into a detailed specification.

Public API:
    expand_prompt(cfg, minimal_prompt, source_paths) -> dict
        Send a meta-prompt to Claude asking it to expand a minimal user prompt
        into a comprehensive implementation specification.

        Memory is updated inline: when ``cfg.memory_auto_update`` is True,
        Claude outputs an updated ``.ai-code/memory/long-term.md`` EDIT
        entry in the JSON ``files`` array alongside the expanded-prompt.md
        entry.  This ensures memory is kept current whenever Claude reads
        project files, even during non-code-writing phases like prompt
        expansion.

        Web search is forwarded from ``cfg.websearch`` to ``prompt_claude()``
        so Claude can search the web when enabled in the configuration.

        Exports ``expand-userfullprompt.md`` showing exactly what data is
        sent to Claude (system prompt + all content blocks in order).
"""

import os
from typing import Any, Dict, List, Optional

from lib.config import Config
from lib.files import FileData, add_source
from lib.memory import (
    build_memory_block,
    build_memory_update_instructions,
    extract_and_save_memory_from_response,
)
from lib.token_tracker import compute_and_display_breakdown
from lib.tree import get_directory_tree
from lib.prompt_builder import build_message_content, build_expand_meta_prompt, build_readable_prompt_export
from lib.providers import prompt_llm
from lib.validation import parse_response_json, validate_claude_response, ResponseParseError
from lib.export import export_md_file
from lib.utils import warn


def expand_prompt(
    cfg: Config,
    minimal_prompt: str,
    source_paths: List[str],
    exclude_patterns: Optional[List[str]] = None,
    tree_dirs: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Expand a minimal prompt into a detailed implementation specification.

    Claude receives the source file content, long-term project memory, and a
    meta-prompt instructing it to produce a comprehensive specification WITHOUT
    implementing any code.  The expanded prompt is returned in a JSON ``files``
    array entry with ``action: "EDIT"`` and a path containing
    ``expanded-prompt``.

    When ``cfg.memory_auto_update`` is True, inline memory update instructions
    are appended to the meta-prompt so Claude also outputs an updated
    ``.ai-code/memory/long-term.md`` entry in the JSON ``files`` array.  This
    ensures the project memory is refreshed whenever Claude reads source
    files — not just during code execution phases.  The memory entry is
    extracted and saved to disk, then removed from the parsed dict before
    extracting the expanded prompt.

    Web search is forwarded from ``cfg.websearch`` and
    ``cfg.websearch_max_results`` to ``prompt_claude()``, allowing Claude
    to search the web during prompt expansion when enabled.

    Exports ``expand-userfullprompt.md`` via ``build_readable_prompt_export()``
    so the artifact exactly represents what Claude receives.

    Parameters
    ----------
    cfg : Config
        Resolved configuration.
    minimal_prompt : str
        The user's original minimal prompt to expand.
    source_paths : list[str]
        File/directory paths to read as context for expansion.
    exclude_patterns : list[str], optional
        Glob exclusion patterns.  Defaults to ``cfg.exclude_patterns``.
    tree_dirs : list[str], optional
        Directories for tree display.  Defaults to ``cfg.tree_dirs``.
    output_dir : str, optional
        Directory for saving artifacts.  Defaults to ``cfg.claude_output_dir``.

    Returns
    -------
    dict
        ``status``           — ``"ok"`` | ``"error"``
        ``expanded_prompt``  — the expanded specification text (empty on error)
        ``raw_response``     — full Claude response text
        ``thinking``         — thinking content (if extended thinking enabled)
        ``error``            — error message (None on success)
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

    # -- 2b. Build memory context (long-term only, no short-term for expansion)
    # During expand (Phase 1) no workflow has started yet so short-term memory
    # does not exist.  Long-term memory gives Claude project-wide awareness
    # (architecture, conventions, schema) which improves expansion quality.
    memory_result = build_memory_block(cfg, include_short_term=False)
    memory_block = memory_result.text

    # -- 2c. Build inline memory update instructions --------------------------
    # Appended to the meta-prompt so Claude outputs an updated memory file
    # entry in the JSON files array alongside the expanded-prompt entry.
    # This keeps the project memory current whenever Claude reads source
    # files — even during non-code-writing phases like prompt expansion.
    memory_instructions = build_memory_update_instructions(cfg)
    if memory_instructions:
        print(f"[tool_prompt_expand] Inline memory update instructions appended ({len(memory_instructions)} chars)")

    # Log websearch status for this specific tool invocation
    if cfg.websearch:
        print(f"[tool_prompt_expand] Web search: ENABLED (max_results={cfg.websearch_max_results})")

    # -- 3. Build the expand meta-prompt --------------------------------------
    meta_prompt = build_expand_meta_prompt(minimal_prompt)

    # Append memory update instructions to the meta-prompt so they are part
    # of the prompt Claude sees.  The expand meta-prompt explicitly allows
    # the memory file entry alongside the expanded-prompt.md entry.
    full_meta_prompt = meta_prompt + memory_instructions if memory_instructions else meta_prompt

    # -- 4. Build message content using source files + meta-prompt ------------
    # memory_block is prepended as the first content item so Claude sees
    # project context before source files and the meta-prompt.
    message_content, _ = build_message_content(
        files_to_ai, full_meta_prompt, ai_file_listing, memory_block=memory_block,
    )

    # -- Display token breakdown ----------------------------------------------
    # user_prompt = minimal_prompt (the semantic user request)
    # tool_context = meta_prompt wrapper instructions (total meta minus embedded prompt)
    # memory_instructions tracked separately for accurate breakdown
    tool_context_chars = max(0, len(meta_prompt) - len(minimal_prompt))
    compute_and_display_breakdown(
        system=cfg.system,
        memory_result=memory_result,
        files_to_ai=files_to_ai,
        ai_file_listing=ai_file_listing,
        user_prompt=minimal_prompt,
        tool_context_chars=tool_context_chars,
        memory_instructions=memory_instructions,
    )

    # -- Export exactly what Claude receives -----------------------------------
    # build_readable_prompt_export produces a human-readable string combining
    # system prompt + all message_content blocks in order, so the artifact
    # exactly represents what the LLM sees.
    readable_prompt = build_readable_prompt_export(cfg.system, message_content)
    export_md_file(readable_prompt, "expand-userfullprompt.md", output_dir)

    # -- 5. Call the LLM ------------------------------------------------------
    # Provider-agnostic dispatch — prompt_llm reads cfg.provider and forwards
    # to the appropriate backend (Claude native or OpenRouter).  Web search,
    # system prompt, and generation knobs flow through the dispatcher.
    print(f"[tool_prompt_expand] Asking LLM to expand the prompt (provider={cfg.provider}, model={cfg.model})...")
    result = prompt_llm(
        cfg=cfg,
        messages=[{"role": "user", "content": message_content}],
        stream=True,
        recv_path=os.path.join(output_dir, "expand-recv.md"),
    )

    if result["status"] != "ok":
        error_msg = result.get("error") or "No response received"
        print(f"\n[tool_prompt_expand] Error: {error_msg}")
        return {
            "status": "error",
            "expanded_prompt": "",
            "raw_response": result.get("data_response", ""),
            "thinking": result.get("thinking_content", ""),
            "error": error_msg,
        }

    data_response = result["data_response"]
    thinking_content = result.get("thinking_content", "")

    # -- 6. Validate JSON response structure ----------------------------------
    validate_claude_response(data_response)

    # -- 7. Export artifacts ---------------------------------------------------
    # Export the raw response before any parsing or memory extraction so the
    # artifact captures exactly what Claude returned — critical for debugging
    # when JSON parsing or memory extraction fails.
    if data_response:
        export_md_file(data_response, "expand-clauderesponse.md", output_dir)
    if thinking_content:
        export_md_file(thinking_content, "expand-thinking.md", output_dir)
    if result.get("raw_data"):
        raw_data_str = "\n\n".join(
            f"Event Type: {item['type']}\nData: {item['event']}"
            for item in result["raw_data"]
        )
        export_md_file(raw_data_str, "expand-rawdata.md", output_dir)

    # -- 7b. Extract and save memory from response ----------------------------
    # Parse the JSON response, find the .ai-code/memory/long-term.md entry
    # in the files array, save it to cfg.memory_long_term_dir, and remove it
    # from the parsed dict.  Returns the modified parsed dict so the memory
    # entry doesn't interfere with expanded-prompt extraction below.
    parsed_response = extract_and_save_memory_from_response(cfg, data_response)

    # -- 8. Extract expanded prompt from JSON files array ---------------------
    # Search for an EDIT entry whose path contains "expanded-prompt".
    # The meta-prompt instructs Claude to output the expanded specification
    # as an EDIT entry for ./expanded-prompt.md in the JSON files array.
    expanded_prompt = ""
    for entry in parsed_response.get("files", []):
        path = entry.get("path", "")
        if "expanded-prompt" in path and entry.get("action") == "EDIT":
            expanded_prompt = entry.get("content", "").strip()
            break

    if not expanded_prompt:
        # Fallback: if no matching JSON entry found, use the entire raw
        # response as the expanded prompt.  Handles cases where Claude
        # doesn't follow the JSON format correctly.
        warn("[tool_prompt_expand] No expanded-prompt entry found in JSON — using full response as expanded prompt")
        expanded_prompt = data_response.strip()

    # Save the extracted expanded prompt separately
    export_md_file(expanded_prompt, "expanded-prompt.md", output_dir)

    print(f"\n[tool_prompt_expand] Expanded prompt: {len(expanded_prompt)} chars")

    return {
        "status": "ok",
        "expanded_prompt": expanded_prompt,
        "raw_response": data_response,
        "thinking": thinking_content,
        "error": None,
    }
