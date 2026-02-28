"""
lib.tools.tool_prompt_execute — execute a prompt against source files via Claude.

Public API:
    execute_prompt(cfg, prompt, source_paths, ...) -> dict
        Full pipeline: discover files → optionally scan source to memory →
        build tree → build memory context → assemble message → call Claude →
        validate → optionally apply → export artifacts → update long-term memory.
"""

import os
from typing import Any, Dict, List, Optional, Set

from lib.config import Config
from lib.files import FileData, add_source
from lib.images import add_images
from lib.tree import get_directory_tree
from lib.prompt_builder import build_message_content
from lib.memory import (
    build_memory_block,
    update_long_term_memory,
    update_long_term_memory_from_source,
)
from lib.providers.claude import prompt_claude
from lib.validation import validate_claude_response
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
    scan_source_to_memory: bool = False,
) -> Dict[str, Any]:
    """Execute a full prompt-to-application pipeline.

    Orchestrates the complete flow from file discovery through Claude
    invocation to disk application and memory update.  Memory context
    (long-term project memory, short-term workflow memory, git history)
    is automatically loaded and injected into the prompt based on the
    ``cfg`` memory settings.  After successful disk application, long-term
    memory is auto-updated when ``cfg.memory_auto_update`` is True.

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
        where inter-step context matters.  Long-term memory and git
        history inclusion are controlled by their own ``cfg`` toggles
        and are always included when enabled — this flag only governs
        the short-term component.
    scan_source_to_memory : bool
        When True, after discovering and reading source files, a
        pre-execution memory update is triggered: file previews are sent
        to Claude to update ``long-term.md`` with a codebase map.  This
        ensures the memory reflects the *current* state of the codebase
        before execution, which is critical during ai-steps where files
        change between steps.  The updated memory is then included in
        the prompt via ``build_memory_block()``.  Failure is non-fatal.

    Returns
    -------
    dict
        ``status``              — ``"ok"`` | ``"error"`` | ``"no_response"``
        ``response``            — Claude's text response
        ``thinking``            — thinking content (if extended thinking enabled)
        ``validation_ok``       — bool, whether validation passed
        ``files_applied``       — bool, whether file edits were applied
        ``original_abs_paths``  — set[str] of discovered source file paths
        ``error``               — error message (None on success)

    Notes
    -----
    Memory update failure is non-fatal: the function still returns
    ``"ok"`` even if memory update fails, since memory is a side effect
    that must never block the primary execution pipeline.
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

    # ── 1. Prepare files_to_ai with optional images ─────────────────────────
    files_to_ai: List[FileData] = []
    if image_paths:
        files_to_ai = add_images(files_to_ai, image_paths)

    # ── 2. Discover and read source files ────────────────────────────────────
    files_to_ai, original_abs_paths = add_source(
        files_to_ai, source_paths, exclude_patterns, ai_shared_file_types,
    )

    # ── 2b. Source-scan memory update (pre-execution) ────────────────────────
    # During ai-steps, the codebase evolves with each step.  Before executing
    # the current step's prompt, scan the source files we just read and update
    # long-term memory so it reflects the latest codebase state.  This gives
    # Claude an accurate map of functions, variables, routes, and schema when
    # the memory block is assembled in step 3b.
    if scan_source_to_memory and cfg.memory_enabled and cfg.memory_long_term_enabled:
        print("[tool_prompt_execute] Scanning source files to update long-term memory...")
        scan_ok = update_long_term_memory_from_source(cfg, files_to_ai)
        if scan_ok:
            print("[tool_prompt_execute] ✓ Memory updated from source scan")
        else:
            warn("[tool_prompt_execute] ⚠ Source scan memory update failed (non-fatal)")

    # ── 3. Build directory tree ──────────────────────────────────────────────
    clean_tree, ai_file_listing = get_directory_tree(
        tree_dirs, exclude_patterns, files_to_ai,
    )

    # ── 3b. Build memory context block ───────────────────────────────────────
    # Assemble long-term memory, short-term memory (if requested), and git
    # history into a single block for prompt injection.  This gives Claude
    # project context before it sees any source files or the user prompt.
    # NOTE: If scan_source_to_memory was True, the long-term memory was just
    # updated in step 2b, so this block will contain the freshly scanned map.
    memory_block = build_memory_block(cfg, include_short_term=include_short_term_memory)
    if memory_block:
        print(f"[tool_prompt_execute] Memory block: {len(memory_block)} chars (~{len(memory_block) // 4} tokens)")

    # ── 4. Build message content ─────────────────────────────────────────────
    # Pass the memory block so it is prepended as the first content item,
    # establishing project context before source files and the user prompt.
    message_content, data_files = build_message_content(
        files_to_ai, prompt, ai_file_listing, memory_block=memory_block,
    )

    # Approximate token estimate
    estimated_tokens = (len(str(message_content)) + len(str(cfg.system))) // 4
    print(f"\nInput tokens [ESTIMATED]: {estimated_tokens}")

    # ── 5. Export assembled prompt for record-keeping ────────────────────────
    # Include the memory block in the exported prompt so developers can
    # inspect exactly what context Claude received for debugging purposes.
    export_md_file(
        "\n\n".join([cfg.system, memory_block, prompt, ai_file_listing, data_files]),
        f"{label}userfullprompt.md",
        output_dir,
    )
    export_md_file(
        str(message_content),
        f"{label}message_content.md",
        output_dir,
    )

    # ── 6. Call Claude ───────────────────────────────────────────────────────
    print(f"\n[tool_prompt_execute] Sending request to Claude...")
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

    # ── 7. Validate ──────────────────────────────────────────────────────────
    print("\n\nValidating Claude response structure...")
    validation_ok = validate_claude_response(data_response)

    if not validation_ok:
        warn(
            "Response validation detected issues (see warnings above). "
            "Proceeding with file application if enabled, but review results carefully."
        )

    # ── 8. Apply to disk ─────────────────────────────────────────────────────
    files_applied = False
    if apply_to_disk:
        print("\nApplying Claude's response to disk...")
        claude_data_to_file(data_response, original_abs_paths)
        files_applied = True

    # ── 9. Export artifacts ───────────────────────────────────────────────────
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

    # ── 10. Update long-term project memory (post-execution) ─────────────────
    # Triggered only when memory is enabled, auto-update is on, and changes
    # were actually applied to disk.  This captures *what changed* in memory,
    # complementing the pre-execution source scan (step 2b) which captures
    # the state *before* changes.  Failure is non-fatal.
    if cfg.memory_enabled and cfg.memory_auto_update and apply_to_disk:
        print("[tool_prompt_execute] Updating long-term project memory...")
        source_summary = "\n".join(f"- {f.path_rel}" for f in files_to_ai if f.ai_share)
        memory_updated = update_long_term_memory(cfg, data_response, source_summary)
        if memory_updated:
            print("[tool_prompt_execute] ✓ Project memory updated")
        else:
            warn("[tool_prompt_execute] ⚠ Project memory update failed (non-fatal)")

    return {
        "status": "ok",
        "response": data_response,
        "thinking": thinking_content,
        "validation_ok": validation_ok,
        "files_applied": files_applied,
        "original_abs_paths": original_abs_paths,
        "error": None,
    }
