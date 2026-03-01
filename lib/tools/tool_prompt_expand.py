"""
lib.tools.tool_prompt_expand — expand a minimal prompt into a detailed specification.

Public API:
    expand_prompt(cfg, minimal_prompt, source_paths) -> dict
        Send a meta-prompt to Claude asking it to expand a minimal user prompt
        into a comprehensive implementation specification.

        Memory is updated inline: when ``cfg.memory_auto_update`` is True,
        Claude outputs an updated ``.ai-code/memory/long-term.md`` block
        alongside the expanded-prompt.md block.  This ensures memory is
        kept current whenever Claude reads project files, even during
        non-code-writing phases like prompt expansion.

        Web search is forwarded from ``cfg.websearch`` to ``prompt_claude()``
        so Claude can search the web when enabled in the configuration.
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
from lib.prompt_builder import build_message_content, build_expand_meta_prompt
from lib.providers.claude import prompt_claude
from lib.validation import block_pattern, validate_claude_response
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
    implementing any code.  The expanded prompt is returned in a
    ``{'+'*5} ./expanded-prompt.md [EDIT]`` block.

    When ``cfg.memory_auto_update`` is True, inline memory update instructions
    are appended to the meta-prompt so Claude also outputs an updated
    ``.ai-code/memory/long-term.md`` block.  This ensures the project memory
    is refreshed whenever Claude reads source files — not just during code
    execution phases.  The memory block is extracted and saved to disk, then
    stripped from the response before extracting the expanded prompt.

    Web search is forwarded from ``cfg.websearch`` and
    ``cfg.websearch_max_results`` to ``prompt_claude()``, allowing Claude
    to search the web during prompt expansion when enabled.

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

    # ── 1. Discover and read source files ────────────────────────────────────
    files_to_ai: List[FileData] = []
    files_to_ai, _ = add_source(
        files_to_ai, source_paths, exclude_patterns,
    )

    # ── 2. Build directory tree ──────────────────────────────────────────────
    _, ai_file_listing = get_directory_tree(
        tree_dirs, exclude_patterns, files_to_ai,
    )

    # ── 2b. Build memory context (long-term only, no short-term for expansion)
    # During expand (Phase 1) no workflow has started yet so short-term memory
    # does not exist.  Long-term memory gives Claude project-wide awareness
    # (architecture, conventions, schema) which improves expansion quality.
    memory_result = build_memory_block(cfg, include_short_term=False)
    memory_block = memory_result.text

    # ── 2c. Build inline memory update instructions ──────────────────────────
    # Appended to the meta-prompt so Claude outputs an updated memory file
    # block alongside the expanded-prompt.md block.  This keeps the project
    # memory current whenever Claude reads source files — even during
    # non-code-writing phases like prompt expansion.
    memory_instructions = build_memory_update_instructions(cfg)
    if memory_instructions:
        print(f"[tool_prompt_expand] Inline memory update instructions appended ({len(memory_instructions)} chars)")

    # Log websearch status for this specific tool invocation
    if cfg.websearch:
        print(f"[tool_prompt_expand] Web search: ENABLED (max_results={cfg.websearch_max_results})")

    # ── 3. Build the expand meta-prompt ──────────────────────────────────────
    meta_prompt = build_expand_meta_prompt(minimal_prompt)

    # Append memory update instructions to the meta-prompt so they are part
    # of the prompt Claude sees.  The expand meta-prompt explicitly allows
    # the memory file block alongside the expanded-prompt.md block.
    full_meta_prompt = meta_prompt + memory_instructions if memory_instructions else meta_prompt

    # ── 4. Build message content using source files + meta-prompt ────────────
    # memory_block is prepended as the first content item so Claude sees
    # project context before source files and the meta-prompt.
    message_content, _ = build_message_content(
        files_to_ai, full_meta_prompt, ai_file_listing, memory_block=memory_block,
    )

    # ── Display token breakdown ──────────────────────────────────────────────
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

    # ── 5. Call Claude ───────────────────────────────────────────────────────
    # websearch and websearch_max_results are forwarded from cfg so that
    # Claude can perform web searches when enabled — applies uniformly to
    # all tool invocations within any workflow.
    print("[tool_prompt_expand] Asking Claude to expand the prompt...")
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

    # ── 6. Validate response ─────────────────────────────────────────────────
    validate_claude_response(data_response)

    # ── 7. Export artifacts ───────────────────────────────────────────────────
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

    # ── 7b. Extract and save memory from response ────────────────────────────
    # Parse the .ai-code/memory/long-term.md block from the response, save it
    # to cfg.memory_long_term_dir, and strip it from the response before
    # extracting the expanded-prompt block.  This ensures the memory file
    # doesn't interfere with tool-specific block extraction.
    data_response = extract_and_save_memory_from_response(cfg, data_response)

    # ── 8. Extract expanded prompt from response block ───────────────────────
    expanded_prompt = ""
    for m in block_pattern.finditer(data_response):
        source_path = m.group("source").strip()
        content = m.group("content").strip()
        if "expanded-prompt" in source_path:
            expanded_prompt = content
            break

    if not expanded_prompt:
        # Fallback: if no block found, use the entire response as the expanded prompt
        # (Claude may have output plain text instead of a block)
        warn("[tool_prompt_expand] No expanded-prompt block found — using full response as expanded prompt")
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
