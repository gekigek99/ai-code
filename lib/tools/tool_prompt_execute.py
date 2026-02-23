"""
lib.tools.tool_prompt_execute — execute a prompt against source files via Claude.

Public API:
    execute_prompt(cfg, prompt, source_paths, ...) -> dict
        Full pipeline: discover files → build tree → assemble message →
        call Claude → validate → optionally apply → export artifacts.
"""

import os
from typing import Any, Dict, List, Optional, Set

from lib.config import Config
from lib.files import FileData, add_source
from lib.images import add_images
from lib.tree import get_directory_tree
from lib.prompt_builder import build_message_content
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
) -> Dict[str, Any]:
    """Execute a full prompt-to-application pipeline.

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

    # ── 3. Build directory tree ──────────────────────────────────────────────
    clean_tree, ai_file_listing = get_directory_tree(
        tree_dirs, exclude_patterns, files_to_ai,
    )

    # ── 4. Build message content ─────────────────────────────────────────────
    message_content, data_files = build_message_content(
        files_to_ai, prompt, ai_file_listing,
    )

    # Approximate token estimate
    estimated_tokens = (len(str(message_content)) + len(str(cfg.system))) // 4
    print(f"\nInput tokens [ESTIMATED]: {estimated_tokens}")

    # ── 5. Export assembled prompt for record-keeping ────────────────────────
    export_md_file(
        "\n\n".join([cfg.system, prompt, ai_file_listing, data_files]),
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

    return {
        "status": "ok",
        "response": data_response,
        "thinking": thinking_content,
        "validation_ok": validation_ok,
        "files_applied": files_applied,
        "original_abs_paths": original_abs_paths,
        "error": None,
    }
