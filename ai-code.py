#!/usr/bin/env python3
"""
ai-code — AI-assisted code generation tool powered by Anthropic Claude.

This is the slim entry point.  All business logic lives in the ``lib/``
package; this file contains only CLI parsing, config loading, and the
``main()`` orchestration function that delegates to the appropriate workflow.

Usage:
    python ai-code.py                 # dry run (gather + export prompt)
    python ai-code.py -ai             # send to Claude and apply edits
    python ai-code.py -ai -f          # skip uncommitted-git check
    python ai-code.py -last           # re-apply last saved response
    python ai-code.py -gen-source     # ask Claude for a source list
    python ai-code.py -ai-steps       # multi-step automated workflow
    python ai-code.py -ai-steps -continue  # resume interrupted workflow
"""

# for claude auto-update:
# - never use 5 consecutive "+" signs, use only {'+'*5}

import os
import sys

# ── lib imports (explicit, per module) ───────────────────────────────────────
from lib.cli import build_arg_parser
from lib.config import load_config
from lib.files import add_source
from lib.images import add_images
from lib.tree import get_directory_tree
from lib.export import export_md_file, log_prompt
from lib.validation import validate_claude_response
from lib.apply import claude_data_to_file
from lib.prompt_builder import build_message_content
from lib.token_tracker import compute_and_display_breakdown
from lib.memory import build_memory_block, build_memory_update_instructions
from lib.utils import warn

# Workflow imports — each encapsulates a full CLI workflow
from lib.workflows.workflow_ai import run_ai_workflow
from lib.workflows.workflow_gen_source import run_gen_source_workflow
from lib.workflows.workflow_ai_steps import run_ai_steps_workflow


def main() -> None:
    """Orchestrate the full ai-code workflow based on CLI flags.

    Delegates to the appropriate workflow module:
      -last        → re-apply saved response (handled inline — trivial flow)
      -gen-source  → workflow_gen_source
      -ai-steps    → workflow_ai_steps
      -ai          → workflow_ai
      (none)       → dry run (inline — gather sources, build prompt, export)
    """

    # ── 1. Parse CLI arguments ───────────────────────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── 1.1 Validate flag combinations ───────────────────────────────────────
    # -continue is only meaningful with -ai-steps
    if args.continue_steps and not args.run_ai_steps:
        print("ERROR: -continue can only be used together with -ai-steps.")
        print("Usage: python ai-code.py -ai-steps -continue")
        sys.exit(1)

    # ── 2. Load configuration ────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir)

    # ── 3. Handle -last: re-apply saved response ────────────────────────────
    # This is a simple offline flow — no API call, no workflow needed.
    if args.run_last:
        print(f"Applying files from last Claude response ({cfg.claude_output_dir}/clauderesponse.md)...")
        md_path = os.path.join(cfg.claude_output_dir, "clauderesponse.md")
        if not os.path.isfile(md_path):
            print(f"File not found: {md_path}")
            sys.exit(2)

        # Discover original source paths for the apply report
        _, original_source_abs_file_paths = add_source(
            [], cfg.source, cfg.exclude_patterns, [],
        )

        with open(md_path, encoding="utf-8") as f:
            data_response = f.read()

        if data_response.strip():
            validation_ok = validate_claude_response(data_response)
            if not validation_ok:
                warn("Validation warnings detected on saved response (see above). Proceeding anyway.")
            claude_data_to_file(data_response, original_source_abs_file_paths)
        else:
            print("Empty clauderesponse file.")
        return

    # ── 4. Handle -gen-source: delegate to workflow ──────────────────────────
    if args.run_gen_source:
        run_gen_source_workflow(cfg, args)
        return

    # ── 5. Handle -ai-steps: delegate to workflow ────────────────────────────
    if args.run_ai_steps:
        run_ai_steps_workflow(cfg, args)
        return

    # ── 6. Handle -ai: delegate to workflow ──────────────────────────────────
    if args.run_ai:
        run_ai_workflow(cfg, args)
        return

    # ── 7. Dry run: gather sources, build prompt, export ─────────────────────
    # No API call — just show what would be sent.
    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    files_to_ai: list = []
    if args.image_paths:
        files_to_ai = add_images(files_to_ai, args.image_paths)
        ai_shared_file_types.append("img")

    files_to_ai, _ = add_source(
        files_to_ai, cfg.source, cfg.exclude_patterns, ai_shared_file_types,
    )

    clean_tree, ai_file_listing = get_directory_tree(
        cfg.tree_dirs, cfg.exclude_patterns, files_to_ai,
    )

    # Build memory block for dry run token estimation
    memory_result = build_memory_block(cfg, include_short_term=False)

    # Build memory update instructions to estimate their token cost
    memory_instructions = build_memory_update_instructions(cfg)
    full_prompt = cfg.prompt + memory_instructions if memory_instructions else cfg.prompt

    message_content, data_files = build_message_content(
        files_to_ai, full_prompt, ai_file_listing, memory_block=memory_result.text,
    )

    # ── Token breakdown for dry run (unified function) ───────────────────────
    compute_and_display_breakdown(
        system=cfg.system,
        memory_result=memory_result,
        files_to_ai=files_to_ai,
        ai_file_listing=ai_file_listing,
        user_prompt=cfg.prompt,
        memory_instructions=memory_instructions,
    )

    # Export assembled prompt for record-keeping
    export_md_file(
        "\n\n".join([cfg.system, cfg.prompt, ai_file_listing, data_files]),
        "userfullprompt.md",
        cfg.logs_dir,
    )
    export_md_file(str(message_content), "message_content.md", cfg.claude_output_dir)

    print("Not executing AI request (dry run). Use -ai, -ai-steps, or -gen-source to execute.")


if __name__ == "__main__":
    main()
