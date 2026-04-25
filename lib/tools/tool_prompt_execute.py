"""
lib.tools.tool_prompt_execute — execute a prompt against source files via Claude.

Public API:
    execute_prompt(cfg, prompt, source_paths, ...) -> dict
        Full pipeline: discover files → build tree → build memory context →
        append inline memory update instructions → assemble message →
        call Claude → validate → extract memory from response →
        optionally apply → export artifacts.

Memory updates are performed **inline**: the prompt includes instructions
for Claude to include an EDIT entry for ``.ai-code/long-term.md`` in the
JSON ``files`` array alongside regular code entries.  After receiving the
response, the memory entry is extracted, saved to disk, and removed from
the parsed dict before file application.  This eliminates separate API
calls for memory updates.

Exports ``userfullprompt.md`` using ``build_readable_prompt_export()`` so
the artifact exactly represents what Claude receives (system prompt + all
content blocks in order).
"""

import os
from typing import Any, Dict, List, Optional, Set

from lib.config import Config
from lib.files import FileData, add_source
from lib.images import add_images
from lib.tree import get_directory_tree
from lib.prompt_builder import build_message_content, build_readable_prompt_export
from lib.memory import (
    build_memory_block,
    build_memory_update_instructions,
    extract_and_save_memory_from_response,
)
from lib.token_tracker import compute_and_display_breakdown
from lib.providers import prompt_llm
from lib.validation import parse_response_json, validate_claude_response, ResponseParseError
from lib.apply import claude_data_to_file
from lib.export import export_md_file
from lib.utils import warn


def execute_prompt(
    cfg: Config,
    prompt: str,
    source_paths: List[str],
    exclude_patterns: Optional[List[str]] = None,
    tree_dirs: Optional[List[str]] = None,
    ai_shared_file_types: Optional[List[str]] = None,
    image_paths: Optional[List[str]] = None,
    apply_to_disk: bool = True,
    output_dir: Optional[str] = None,
    label: str = "",
    include_short_term_memory: bool = False,
) -> Dict[str, Any]:
    """Execute a full prompt-to-application pipeline.

    Orchestrates the complete flow from file discovery through Claude
    invocation to disk application.  Memory context (long-term project
    memory, short-term workflow memory, git history) is automatically
    loaded and injected into the prompt based on ``cfg`` memory settings.

    When ``cfg.memory_auto_update`` is True, inline memory update
    instructions are appended to the prompt so Claude includes an EDIT
    entry for ``.ai-code/long-term.md`` in its JSON ``files`` array
    alongside code entries.  After response, the memory entry is
    extracted, saved, and removed before applying code changes to disk.
    No separate API calls are made for memory.

    Web search is forwarded from ``cfg.websearch`` to ``prompt_claude()``,
    allowing Claude to search the web during generation when enabled.

    Exports ``userfullprompt.md`` via ``build_readable_prompt_export()``
    so the artifact is an exact textual representation of what Claude
    receives (system prompt + all content blocks in order).

    Parameters
    ----------
    cfg : Config
        Resolved configuration.
    prompt : str
        The prompt text to send to Claude.
    source_paths : list[str]
        File/directory paths to discover and read as context.
    exclude_patterns : list[str], optional
        Glob patterns to exclude.  Defaults to ``cfg.exclude_patterns``.
    tree_dirs : list[str], optional
        Directories for tree display.  Defaults to ``cfg.tree_dirs``.
    ai_shared_file_types : list[str], optional
        Additional file types to share (e.g. ``["pdf"]``).
    image_paths : list[str], optional
        Image file paths to attach to the prompt.
    apply_to_disk : bool
        If True, apply file edits from the response to disk.
    output_dir : str, optional
        Directory for saving artifacts.  Defaults to ``cfg.claude_output_dir``.
    label : str
        Prefix label for artifact filenames (e.g. ``"step-1-"``).
    include_short_term_memory : bool
        When True, short-term workflow memory (``short-term.md``) is
        included in the memory block alongside long-term memory and git
        history.  Typically True only for the ``-ai-steps`` workflow
        where inter-step context matters.

    Returns
    -------
    dict
        ``status``              — ``"ok"`` | ``"error"`` | ``"no_response"``
        ``response``            — Claude's raw text response (for export/display)
        ``thinking``            — thinking content (if extended thinking enabled)
        ``validation_ok``       — bool, whether validation passed
        ``files_applied``       — bool, whether file edits were applied
        ``original_abs_paths``  — set[str] of discovered source file paths
        ``error``               — error message (None on success)
    """
    if exclude_patterns is None:
        exclude_patterns = cfg.exclude_patterns
    if tree_dirs is None:
        tree_dirs = cfg.tree_dirs
    if ai_shared_file_types is None:
        ai_shared_file_types = []
    if output_dir is None:
        output_dir = cfg.claude_output_dir
    if image_paths is None:
        image_paths = []

    os.makedirs(output_dir, exist_ok=True)

    # -- 1. Prepare files_to_ai with optional images -------------------------
    files_to_ai: List[FileData] = []
    if image_paths:
        files_to_ai = add_images(files_to_ai, image_paths)

    # -- 2. Discover and read source files ------------------------------------
    files_to_ai, original_abs_paths = add_source(
        files_to_ai, source_paths, exclude_patterns, ai_shared_file_types,
    )

    # -- 3. Build directory tree ----------------------------------------------
    clean_tree, ai_file_listing = get_directory_tree(
        tree_dirs, exclude_patterns, files_to_ai,
    )

    # -- 3b. Build memory context block (read-only) --------------------------
    # Assemble long-term memory, short-term memory (if requested), and git
    # history into a single block for prompt injection.  This gives Claude
    # project context before it sees any source files or the user prompt.
    memory_result = build_memory_block(cfg, include_short_term=include_short_term_memory)
    memory_block = memory_result.text

    if memory_block:
        print(f"[tool_prompt_execute] Memory block: {len(memory_block)} chars "
              f"(~{len(memory_block) // 4} tokens | "
              f"LT={memory_result.long_term_tokens} ST={memory_result.short_term_tokens} "
              f"Git={memory_result.git_history_tokens})")

    # -- 3c. Append inline memory update instructions to prompt ---------------
    # When enabled, this tells Claude to include an EDIT entry for
    # .ai-code/long-term.md in its JSON files array alongside its regular
    # file output.  Claude already has the existing memory from the
    # [MEMORY START] context block and all source files in the prompt, so
    # it can produce an accurate update in the same API call — eliminating
    # the need for separate memory update API calls.
    memory_instructions = build_memory_update_instructions(cfg)
    full_prompt = prompt + memory_instructions if memory_instructions else prompt
    if memory_instructions:
        print(f"[tool_prompt_execute] Inline memory update instructions appended ({len(memory_instructions)} chars)")

    # Log websearch status for this execution — important for debugging
    # whether Claude had web access during this specific call.
    if cfg.websearch:
        print(f"[tool_prompt_execute] Web search: ENABLED (max_results={cfg.websearch_max_results})")

    # -- 4. Build message content ---------------------------------------------
    # Pass the memory block so it is prepended as the first content item,
    # establishing project context before source files and the user prompt.
    message_content, data_files = build_message_content(
        files_to_ai, full_prompt, ai_file_listing, memory_block=memory_block,
    )

    # -- Display token breakdown ----------------------------------------------
    compute_and_display_breakdown(
        system=cfg.system,
        memory_result=memory_result,
        files_to_ai=files_to_ai,
        ai_file_listing=ai_file_listing,
        user_prompt=prompt,
        memory_instructions=memory_instructions,
    )

    # -- 5. Export assembled prompt for record-keeping ------------------------
    # build_readable_prompt_export combines system prompt + all message_content
    # blocks into a human-readable string that exactly represents what Claude
    # receives, in the same order it sees the content.  This replaces the
    # previous manual concatenation which could drift from actual message order.
    readable_prompt = build_readable_prompt_export(cfg.system, message_content)
    export_md_file(readable_prompt, f"{label}userfullprompt.md", output_dir)

    # -- 6. Call Claude -------------------------------------------------------
    # websearch and websearch_max_results are forwarded from cfg so that
    # Claude can perform web searches when enabled — applies to all workflows
    # (ai, ai-steps, gen-source) uniformly.
    print(f"\n[tool_prompt_execute] Sending request to LLM (provider={cfg.provider}, model={cfg.model})...")
    # Provider-agnostic dispatch — prompt_llm reads cfg.provider and forwards
    # to the appropriate backend (Claude native or OpenRouter).
    result = prompt_llm(
        cfg=cfg,
        messages=[{"role": "user", "content": message_content}],
        stream=True,
        recv_path=os.path.join(output_dir, f"{label}clauderesponse-recv.md"),
    )

    if result["status"] == "error":
        error_msg = result.get("error", "Unknown API error")
        print(f"\n[tool_prompt_execute] Error: {error_msg}")
        return {
            "status": "error",
            "response": result.get("data_response", ""),
            "thinking": result.get("thinking_content", ""),
            "validation_ok": False,
            "files_applied": False,
            "original_abs_paths": original_abs_paths,
            "error": error_msg,
        }

    if result["status"] == "no_response":
        print("\n[tool_prompt_execute] No response received from Claude.")
        return {
            "status": "no_response",
            "response": "",
            "thinking": result.get("thinking_content", ""),
            "validation_ok": False,
            "files_applied": False,
            "original_abs_paths": original_abs_paths,
            "error": "No response received",
        }

    data_response = result["data_response"]
    thinking_content = result.get("thinking_content", "")

    # -- 7. Export raw artifacts FIRST ----------------------------------------
    # Export the raw response before any parsing or memory extraction so the
    # artifact captures exactly what Claude returned — critical for debugging
    # when JSON parsing or memory extraction fails.
    if data_response:
        export_md_file(data_response, f"{label}clauderesponse.md", output_dir)
        print(f"\n[Saved data response]")
    if thinking_content:
        export_md_file(thinking_content, f"{label}thinking.md", output_dir)
        print(f"\n[Saved thinking content ({len(thinking_content)} chars)]")
    if result.get("raw_data"):
        raw_data_str = "\n\n".join(
            f"Event Type: {item['type']}\nData: {item['event']}"
            for item in result["raw_data"]
        )
        export_md_file(raw_data_str, f"{label}rawdata.md", output_dir)
        print(f"[Saved {len(result['raw_data'])} raw events to file]")

    # -- 8. Validate JSON structure -------------------------------------------
    print("\n\nValidating Claude response structure...")
    validation_ok = validate_claude_response(data_response)

    if not validation_ok:
        warn(
            "Response validation detected issues (see warnings above). "
            "Proceeding with file application if enabled, but review results carefully."
        )

    # -- 9. Extract and save memory from response -----------------------------
    # Parse the JSON response, find the .ai-code/long-term.md entry in the
    # files array, save it to cfg.memory_long_term_dir, and remove it from
    # the parsed dict.  Returns the modified parsed dict so downstream
    # claude_data_to_file never sees the memory entry.
    parsed_response = extract_and_save_memory_from_response(cfg, data_response)

    # -- 10. Apply to disk ----------------------------------------------------
    files_applied = False
    if apply_to_disk:
        print("\nApplying Claude's response to disk...")
        claude_data_to_file(parsed_response, original_abs_paths, patch_enabled=cfg.patch_enabled)
        files_applied = True

    return {
        "status": "ok",
        "response": data_response,
        "thinking": thinking_content,
        "validation_ok": validation_ok,
        "files_applied": files_applied,
        "original_abs_paths": original_abs_paths,
        "error": None,
    }
