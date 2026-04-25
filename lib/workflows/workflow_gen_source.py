"""
lib.workflows.workflow_gen_source — source list generation workflow.

Public API:
    run_gen_source_workflow(cfg, args) -> None
        Execute the -gen-source workflow: build tree, ask Claude for a source
        list, copy to clipboard, export artifacts.
"""

from argparse import Namespace

from lib.config import Config
from lib.files import add_source
from lib.tree import get_directory_tree
from lib.tools.tool_source_generate import generate_source
from lib.utils import warn, COLOR_CYAN, COLOR_RESET, play_bell


def run_gen_source_workflow(cfg: Config, args: Namespace) -> None:
    """Execute the ``-gen-source`` workflow.

    Steps:
      1. Discover source files and build the directory tree (for context).
      2. Call ``generate_source`` to ask Claude for a recommended source list.
      3. Copy the result to the clipboard (if pyperclip is available).
      4. Artifacts are exported by the tool.

    Parameters
    ----------
    cfg : Config
        Fully resolved configuration.
    args : Namespace
        Parsed CLI arguments.
    """
    # -- Show workflow configuration ------------------------------------------
    print(f"\n{COLOR_CYAN}{'=' * 60}")
    print(f"  GEN-SOURCE WORKFLOW")
    print(f"  Provider: {cfg.provider}")
    print(f"  Model: {cfg.model}")
    print(f"  Web search: {'ENABLED (max_results=' + str(cfg.websearch_max_results) + ')' if cfg.websearch else 'DISABLED'}")
    print(f"{'=' * 60}{COLOR_RESET}\n")

    print("Generating adapted-to-prompt source via Claude...")

    # -- 1. Build the directory tree for context ------------------------------
    # We need the tree to show Claude the project structure so it can decide
    # which files are relevant.  We discover source files first so the tree
    # can show token annotations.
    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    files_to_ai, _ = add_source(
        [], cfg.source, cfg.exclude_patterns, ai_shared_file_types,
    )
    clean_tree, _ = get_directory_tree(
        cfg.tree_dirs, cfg.exclude_patterns, files_to_ai,
    )

    # -- 2. Generate the source list ------------------------------------------
    result = generate_source(
        cfg=cfg,
        prompt=cfg.prompt,
        tree_str=clean_tree,
        example_source=cfg.source,
    )

    if result["status"] != "ok":
        print(f"\n[workflow_gen_source] Failed: {result.get('error')}")
        return

    # -- 3. Copy to clipboard -------------------------------------------------
    if result["source_yaml"]:
        try:
            import pyperclip
            pyperclip.copy(result["source_yaml"])
            print("\n[Copied generated source YAML to clipboard!]")
        except ImportError:
            warn("pyperclip not installed — could not copy to clipboard.")

    print(f"\n[workflow_gen_source] Source generation complete ({len(result['source_list'])} entries).")

    # Completion bell
    if cfg.sound_enabled:
        play_bell()
