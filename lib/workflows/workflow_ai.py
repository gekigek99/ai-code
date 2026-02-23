"""
lib.workflows.workflow_ai — standard single-shot AI prompt execution.

Public API:
    run_ai_workflow(cfg, args) -> None
        Execute the -ai workflow: check git, execute prompt, log result.
"""

import sys
from argparse import Namespace

from lib.config import Config
from lib.git import has_uncommitted_changes
from lib.export import log_prompt
from lib.tools.tool_prompt_execute import execute_prompt


def run_ai_workflow(cfg: Config, args: Namespace) -> None:
    """Execute the ``-ai`` workflow.

    Steps:
      1. Check for uncommitted git changes (unless ``-f`` / force).
      2. Gather AI-shared file types and image paths from CLI args.
      3. Call ``execute_prompt`` with the user's prompt and configured source.
      4. Log the prompt for history tracking.

    Parameters
    ----------
    cfg : Config
        Fully resolved configuration.
    args : Namespace
        Parsed CLI arguments (needs ``force``, ``include_pdf``, ``image_paths``).
    """
    # ── 1. Git safety check ──────────────────────────────────────────────────
    if not args.force and has_uncommitted_changes(ignore_dir_name=cfg.script_dir_name):
        print("ERROR: You have uncommitted changes in your git repository.")
        print("Please commit or stash your changes before running this script.")
        print("Use -f to force execution.")
        sys.exit(1)

    # ── 2. Determine file types and images ───────────────────────────────────
    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    image_paths: list = args.image_paths if args.image_paths else []

    # ── 3. Execute the prompt ────────────────────────────────────────────────
    result = execute_prompt(
        cfg=cfg,
        prompt=cfg.prompt,
        source_paths=cfg.source,
        ai_shared_file_types=ai_shared_file_types,
        image_paths=image_paths,
        apply_to_disk=True,
        label="",
    )

    if result["status"] == "ok":
        print("\n[workflow_ai] Prompt executed and applied successfully.")
    elif result["status"] == "no_response":
        print("\n[workflow_ai] No response received from Claude.")
    else:
        print(f"\n[workflow_ai] Error: {result.get('error')}")

    # ── 4. Log prompt for history ────────────────────────────────────────────
    log_prompt(cfg.prompt, cfg.logs_dir)
