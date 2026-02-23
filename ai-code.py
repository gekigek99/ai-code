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
"""

# for claude auto-update:
# - never use 5 consecutive "+"" signs, use only {"+"*5}

import os
import sys

# ── lib imports (explicit, per module) ───────────────────────────────────────
from lib.cli import build_arg_parser
from lib.config import load_config
from lib.files import add_source
from lib.images import add_images
from lib.tree import get_directory_tree
from lib.git import has_uncommitted_changes
from lib.export import export_md_file, log_prompt
from lib.validation import validate_claude_response, block_pattern
from lib.apply import claude_data_to_file
from lib.prompt_builder import build_message_content, generate_prompt_for_gen_source
from lib.providers.claude import prompt_claude
from lib.utils import warn

# Lazy import — only loaded when -gen-source extracts content to clipboard
# import pyperclip


def main() -> None:
    """Orchestrate the full ai-code workflow based on CLI flags."""

    # ── 1. Parse CLI arguments ───────────────────────────────────────────────
    parser = build_arg_parser()
    args = parser.parse_args()

    # ── 2. Load configuration ────────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(script_dir)

    # ── 3. Determine which file types to share with the AI ───────────────────
    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    files_to_ai: list = []
    if args.image_paths:
        files_to_ai = add_images(files_to_ai, args.image_paths)
        ai_shared_file_types.append("img")

    # ── 4. Handle -last: re-apply saved response ────────────────────────────
    if args.run_last:
        print(f"Applying files from last Claude response ({cfg.claude_output_dir}/clauderesponse.md)...")
        md_path = os.path.join(cfg.claude_output_dir, "clauderesponse.md")
        if not os.path.isfile(md_path):
            print(f"File not found: {md_path}")
            sys.exit(2)

        # We still need the original source paths for the apply report
        _, original_source_abs_file_paths = add_source(
            [], cfg.source, cfg.exclude_patterns, ai_shared_file_types,
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

    # ── 5. Discover and read source files ────────────────────────────────────
    files_to_ai, original_source_abs_file_paths = add_source(
        files_to_ai, cfg.source, cfg.exclude_patterns, ai_shared_file_types,
    )

    # ── 6. Build directory tree ──────────────────────────────────────────────
    # Returns:
    #   clean_tree      — human-readable (no ANSI) for logging / export
    #   ai_file_listing — flat "./path" list for LLM context
    clean_tree, ai_file_listing = get_directory_tree(
        cfg.tree_dirs, cfg.exclude_patterns, files_to_ai,
    )

    # ── 7. Handle -gen-source ────────────────────────────────────────────────
    if args.run_gen_source:
        print("Generating adapted-to-prompt source via Claude...")
        gen_source_message_content = generate_prompt_for_gen_source(
            cfg.prompt, cfg.source, clean_tree,
        )

        gen_result = prompt_claude(
            api_key=cfg.anthropic_api_key,
            model=cfg.anthropic_model,
            system=cfg.system,
            messages=[{"role": "user", "content": gen_source_message_content}],
            max_tokens=cfg.anthropic_max_tokens,
            temperature=cfg.anthropic_temperature,
            websearch=cfg.websearch,
            websearch_max_results=cfg.websearch_max_results,
            thinking_budget=cfg.anthropic_max_tokens_think,
            stream=True,
            recv_path=os.path.join(cfg.claude_output_dir, "gen-source-recv.md"),
        )

        export_md_file(
            f"{cfg.system}\n\n{gen_source_message_content}",
            "message_content.md",
            cfg.claude_output_dir,
        )

        if gen_result["status"] == "ok":
            validate_claude_response(gen_result["data_response"])

            # Extract source.md content and copy to clipboard
            for m in block_pattern.finditer(gen_result["data_response"]):
                source_path = m.group("source").strip()
                content = m.group("content").strip()
                if "source.md" in source_path:
                    try:
                        import pyperclip
                        pyperclip.copy(content)
                        print("\n[Copied generated source to clipboard!]")
                    except ImportError:
                        warn("pyperclip not installed — could not copy to clipboard.")

            if gen_result.get("data_response"):
                export_md_file(gen_result["data_response"], "gen-source-clauderesponse.md", cfg.claude_output_dir)
            if gen_result.get("thinking_content"):
                export_md_file(gen_result["thinking_content"], "gen-source-thinking.md", cfg.claude_output_dir)
            if gen_result.get("raw_data"):
                raw_data_str = "\n\n".join(
                    f"Event Type: {item['type']}\nData: {item['event']}"
                    for item in gen_result["raw_data"]
                )
                export_md_file(raw_data_str, "gen-source-rawdata.md", cfg.claude_output_dir)

        elif gen_result["status"] == "no_response":
            print("\nNo response received from Claude for generation.")
        else:
            print(f"\nError calling Claude for generation: {gen_result.get('error')}")

        return  # exit after gen-source (do not proceed to -ai)

    # ── 8. Build message content for normal flow ─────────────────────────────
    message_content, data_files = build_message_content(
        files_to_ai, cfg.prompt, ai_file_listing,
    )

    # Approximate token estimate
    print(f"Input tokens [ESTIMATED]: {(len(str(message_content)) + len(str(cfg.system))) // 4}")

    # ── 9. Export assembled prompt for record-keeping ────────────────────────
    export_md_file(
        "\n\n".join([cfg.system, cfg.prompt, ai_file_listing, data_files]),
        "userfullprompt.md",
        cfg.logs_dir,
    )
    export_md_file(str(message_content), "message_content.md", cfg.claude_output_dir)

    # ── 10. If -ai not set, stop here (dry run) ─────────────────────────────
    if not args.run_ai:
        print("Not executing AI request...")
        return

    # ── 11. Git safety check ─────────────────────────────────────────────────
    if not args.force and has_uncommitted_changes(ignore_dir_name=cfg.script_dir_name):
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        sys.exit(1)

    # ── 12. Call Claude ──────────────────────────────────────────────────────
    print("Sending request to Claude...")
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
        recv_path=os.path.join(cfg.claude_output_dir, "clauderesponse-recv.md"),
    )

    if result["status"] == "ok":
        # ── 13. Validate and apply ───────────────────────────────────────────
        print("\n\nValidating Claude response structure...")
        validation_ok = validate_claude_response(result["data_response"])

        if not validation_ok:
            warn(
                "Response validation detected issues (see warnings above). "
                "Proceeding with file application, but review results carefully."
            )

        print("\nApplying Claude's response to disk...")
        claude_data_to_file(result["data_response"], original_source_abs_file_paths)

        # ── 14. Export response artefacts ────────────────────────────────────
        if result.get("data_response"):
            export_md_file(result["data_response"], "clauderesponse.md", cfg.claude_output_dir)
            print("\n[Saved data response]")
        if result.get("thinking_content"):
            export_md_file(result["thinking_content"], "thinking.md", cfg.claude_output_dir)
            print(f"\n[Saved thinking content ({len(result['thinking_content'])} chars)]")
        if result.get("raw_data"):
            raw_data_str = "\n\n".join(
                f"Event Type: {item['type']}\nData: {item['event']}"
                for item in result["raw_data"]
            )
            export_md_file(raw_data_str, "rawdata.md", cfg.claude_output_dir)
            print(f"[Saved {len(result['raw_data'])} raw events to file]")

    elif result["status"] == "no_response":
        print("\nNo response received from Claude.")
    else:
        print(f"\nError calling Claude: {result.get('error')}")

    # ── 15. Log prompt for history tracking ──────────────────────────────────
    log_prompt(cfg.prompt, cfg.logs_dir)


if __name__ == "__main__":
    main()
