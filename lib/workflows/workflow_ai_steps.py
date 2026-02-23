"""
lib.workflows.workflow_ai_steps — automated multi-step AI workflow.

Public API:
    run_ai_steps_workflow(cfg, args) -> None
        Execute the -ai-steps workflow:
          Phase 1: Expand the user's minimal prompt.
          Phase 2: Decompose the expanded prompt into ordered steps.
          Phase 3: Execute each step with user confirmation and git commits.
"""

import os
import sys
from argparse import Namespace

from lib.config import Config
from lib.files import add_source
from lib.tree import get_directory_tree
from lib.git import has_uncommitted_changes, commit_changes, revert_to_last_commit, is_git_available
from lib.export import export_md_file, log_prompt
from lib.utils import warn, COLOR_GREEN, COLOR_YELLOW, COLOR_CYAN, COLOR_RESET

from lib.tools.tool_source_generate import generate_source
from lib.tools.tool_prompt_expand import expand_prompt
from lib.tools.tool_prompt_stepize import stepize_prompt
from lib.tools.tool_prompt_execute import execute_prompt
from lib.tools.tool_user_confirm import confirm_step

# Maximum retry attempts for a single step before forced skip
_MAX_STEP_RETRIES = 3


def _build_current_tree(cfg: Config, ai_shared_file_types: list) -> str:
    """Helper: rebuild the directory tree from current disk state.

    Returns the clean (no ANSI) tree string suitable for passing to
    source generation tools.
    """
    files_to_ai, _ = add_source(
        [], cfg.source, cfg.exclude_patterns, ai_shared_file_types,
    )
    clean_tree, _ = get_directory_tree(
        cfg.tree_dirs, cfg.exclude_patterns, files_to_ai,
    )
    return clean_tree


def run_ai_steps_workflow(cfg: Config, args: Namespace) -> None:
    """Execute the ``-ai-steps`` automated multi-step workflow.

    This workflow requires git to be available and the working tree to be
    clean.  Each accepted step is committed, providing rollback points.

    Parameters
    ----------
    cfg : Config
        Fully resolved configuration.
    args : Namespace
        Parsed CLI arguments.
    """
    # ── Pre-flight checks ────────────────────────────────────────────────────
    if not is_git_available():
        print("ERROR: git is required for the -ai-steps workflow but is not available.")
        print("Please install git or ensure it is in your PATH.")
        sys.exit(1)

    if has_uncommitted_changes(ignore_dir_name=cfg.script_dir_name):
        print("ERROR: The -ai-steps workflow requires a clean git working tree.")
        print("Please commit or stash all changes before running -ai-steps.")
        sys.exit(1)

    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    # Base output directory for all ai-steps artifacts
    steps_output_dir = os.path.join(cfg.claude_output_dir, "ai-steps")
    os.makedirs(steps_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AI-STEPS WORKFLOW")
    print(f"  Minimal prompt: {cfg.prompt[:80]}{'...' if len(cfg.prompt) > 80 else ''}")
    print(f"{'='*60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1: PROMPT EXPANSION
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{COLOR_CYAN}{'─'*60}")
    print(f"  PHASE 1: PROMPT EXPANSION")
    print(f"{'─'*60}{COLOR_RESET}\n")

    phase1_dir = os.path.join(steps_output_dir, "phase1-expand")
    os.makedirs(phase1_dir, exist_ok=True)

    # 1.1 Generate source for prompt expansion
    print("[Phase 1.1] Generating source list for prompt expansion...")
    tree_str = _build_current_tree(cfg, ai_shared_file_types)

    source_result = generate_source(
        cfg=cfg,
        prompt=cfg.prompt,
        tree_str=tree_str,
        example_source=cfg.source,
        output_dir=phase1_dir,
    )

    if source_result["status"] != "ok":
        print(f"\n[Phase 1.1] FAILED: {source_result.get('error')}")
        print("Cannot proceed without a source list. Aborting.")
        return

    expand_source_paths = source_result["source_list"]
    print(f"[Phase 1.1] Source list: {len(expand_source_paths)} entries")

    # 1.2 Expand the minimal prompt
    print("\n[Phase 1.2] Expanding minimal prompt...")
    expand_result = expand_prompt(
        cfg=cfg,
        minimal_prompt=cfg.prompt,
        source_paths=expand_source_paths,
        output_dir=phase1_dir,
    )

    if expand_result["status"] != "ok":
        print(f"\n[Phase 1.2] FAILED: {expand_result.get('error')}")
        print("Cannot proceed without an expanded prompt. Aborting.")
        return

    expanded_prompt_text = expand_result["expanded_prompt"]
    print(f"[Phase 1.2] Expanded prompt: {len(expanded_prompt_text)} chars")

    # Save the expanded prompt for reference
    export_md_file(expanded_prompt_text, "expanded-prompt.md", steps_output_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2: STEP DECOMPOSITION
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{COLOR_CYAN}{'─'*60}")
    print(f"  PHASE 2: STEP DECOMPOSITION")
    print(f"{'─'*60}{COLOR_RESET}\n")

    phase2_dir = os.path.join(steps_output_dir, "phase2-stepize")
    os.makedirs(phase2_dir, exist_ok=True)

    # 2.1 Generate source for step-ization (may differ from phase 1 source)
    print("[Phase 2.1] Generating source list for step decomposition...")
    tree_str = _build_current_tree(cfg, ai_shared_file_types)

    step_source_result = generate_source(
        cfg=cfg,
        prompt=expanded_prompt_text,
        tree_str=tree_str,
        example_source=cfg.source,
        output_dir=phase2_dir,
    )

    if step_source_result["status"] != "ok":
        print(f"\n[Phase 2.1] FAILED: {step_source_result.get('error')}")
        print("Cannot proceed without a source list. Aborting.")
        return

    stepize_source_paths = step_source_result["source_list"]
    print(f"[Phase 2.1] Source list: {len(stepize_source_paths)} entries")

    # 2.2 Decompose expanded prompt into steps
    print("\n[Phase 2.2] Decomposing expanded prompt into steps...")
    steps_result = stepize_prompt(
        cfg=cfg,
        expanded_prompt=expanded_prompt_text,
        source_paths=stepize_source_paths,
        output_dir=phase2_dir,
    )

    if steps_result["status"] != "ok":
        print(f"\n[Phase 2.2] FAILED: {steps_result.get('error')}")
        print("Cannot proceed without step decomposition. Aborting.")
        return

    steps = steps_result["steps"]
    print(f"\n[Phase 2.2] Decomposed into {len(steps)} step(s)")

    # Save steps for reference
    import yaml as _yaml
    steps_yaml_str = _yaml.safe_dump({"steps": steps}, sort_keys=False, allow_unicode=True)
    export_md_file(steps_yaml_str, "steps.yaml", steps_output_dir)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3: STEP EXECUTION LOOP
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{COLOR_CYAN}{'─'*60}")
    print(f"  PHASE 3: STEP EXECUTION ({len(steps)} steps)")
    print(f"{'─'*60}{COLOR_RESET}\n")

    completed_steps = 0
    skipped_steps = 0

    for step in steps:
        step_number = step["number"]
        step_title = step["title"]
        step_prompt = step["prompt"]
        step_source = step.get("source", cfg.source)

        step_dir = os.path.join(steps_output_dir, f"step-{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        print(f"\n{COLOR_GREEN}{'═'*60}")
        print(f"  STEP {step_number}/{len(steps)}: {step_title}")
        print(f"{'═'*60}{COLOR_RESET}\n")

        current_prompt = step_prompt
        retry_count = 0

        while True:
            # ── 3.1 Generate fresh source for this step ─────────────────────
            # Rebuild tree because previous steps may have changed files
            print(f"[Step {step_number}.1] Generating source list for this step...")
            tree_str = _build_current_tree(cfg, ai_shared_file_types)

            step_source_result = generate_source(
                cfg=cfg,
                prompt=current_prompt,
                tree_str=tree_str,
                example_source=step_source if step_source else cfg.source,
                output_dir=os.path.join(step_dir, f"source-gen-attempt-{retry_count}"),
            )

            if step_source_result["status"] != "ok":
                warn(f"[Step {step_number}.1] Source generation failed: {step_source_result.get('error')}")
                warn(f"Falling back to step-defined source list: {step_source}")
                exec_source_paths = step_source if step_source else cfg.source
            else:
                exec_source_paths = step_source_result["source_list"]
                print(f"[Step {step_number}.1] Source list: {len(exec_source_paths)} entries")

            # ── 3.2 Execute the step ────────────────────────────────────────
            print(f"\n[Step {step_number}.2] Executing step...")
            exec_result = execute_prompt(
                cfg=cfg,
                prompt=current_prompt,
                source_paths=exec_source_paths,
                apply_to_disk=True,
                output_dir=step_dir,
                label=f"attempt-{retry_count}-",
            )

            if exec_result["status"] != "ok":
                warn(f"[Step {step_number}.2] Execution failed: {exec_result.get('error')}")
                # Revert any partial changes
                print("Reverting changes from failed execution...")
                revert_to_last_commit()

                retry_count += 1
                if retry_count >= _MAX_STEP_RETRIES:
                    warn(f"[Step {step_number}] Max retries ({_MAX_STEP_RETRIES}) reached. Skipping step.")
                    skipped_steps += 1
                    break

                # Ask user whether to retry or skip
                user_result = confirm_step(step_number, f"{step_title} [EXECUTION FAILED]")
                if user_result["action"] == "retry" and user_result["modification"]:
                    current_prompt = step_prompt + "\n\nAdditional instructions:\n" + user_result["modification"]
                    continue
                elif user_result["action"] == "retry":
                    continue
                elif user_result["action"] == "skip":
                    skipped_steps += 1
                    break
                elif user_result["action"] == "quit":
                    print("\n[ai-steps] Quitting workflow.")
                    _print_summary(completed_steps, skipped_steps, len(steps))
                    log_prompt(cfg.prompt, cfg.logs_dir)
                    return
                else:
                    # "continue" after failure — treat as skip
                    skipped_steps += 1
                    break

            # ── 3.3 Ask user to confirm ─────────────────────────────────────
            user_result = confirm_step(step_number, step_title)

            if user_result["action"] == "continue":
                # Accept: commit the changes
                commit_msg = f"ai-steps: step {step_number} - {step_title}"
                committed = commit_changes(commit_msg, ignore_dir_name=cfg.script_dir_name)
                if committed:
                    print(f"[Step {step_number}] Changes committed: {commit_msg}")
                else:
                    warn(f"[Step {step_number}] Git commit failed or nothing to commit.")
                completed_steps += 1
                break

            elif user_result["action"] == "retry":
                # Revert changes, modify prompt, re-execute
                print(f"[Step {step_number}] Reverting changes for retry...")
                revert_to_last_commit()

                retry_count += 1
                if retry_count >= _MAX_STEP_RETRIES:
                    warn(f"[Step {step_number}] Max retries ({_MAX_STEP_RETRIES}) reached. Skipping step.")
                    skipped_steps += 1
                    break

                if user_result["modification"]:
                    current_prompt = step_prompt + "\n\nAdditional instructions:\n" + user_result["modification"]
                    print(f"[Step {step_number}] Prompt modified with user input. Retrying...")
                else:
                    print(f"[Step {step_number}] Retrying with same prompt...")
                continue

            elif user_result["action"] == "skip":
                print(f"[Step {step_number}] Reverting changes and skipping...")
                revert_to_last_commit()
                skipped_steps += 1
                break

            elif user_result["action"] == "quit":
                print(f"[Step {step_number}] Reverting changes and quitting...")
                revert_to_last_commit()
                _print_summary(completed_steps, skipped_steps, len(steps))
                log_prompt(cfg.prompt, cfg.logs_dir)
                return

    # ── Final summary ────────────────────────────────────────────────────────
    _print_summary(completed_steps, skipped_steps, len(steps))
    log_prompt(cfg.prompt, cfg.logs_dir)


def _print_summary(completed: int, skipped: int, total: int) -> None:
    """Print a summary of the ai-steps workflow execution."""
    remaining = total - completed - skipped
    print(f"\n{'='*60}")
    print(f"  AI-STEPS WORKFLOW SUMMARY")
    print(f"{'='*60}")
    print(f"  Total steps:     {total}")
    print(f"  Completed:       {completed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Remaining:       {remaining}")
    print(f"{'='*60}\n")
