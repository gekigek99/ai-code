"""
lib.workflows.workflow_ai_steps — automated multi-step AI workflow.

Public API:
    run_ai_steps_workflow(cfg, args) -> None
        Execute the -ai-steps workflow:
          Phase 1: Expand the user's minimal prompt.
          Phase 2: Decompose the expanded prompt into ordered steps.
          Phase 3: Execute each step with user confirmation and git commits.

        Supports ``-continue`` to resume from the last saved checkpoint
        after a crash or connection loss.

        Workflow artifacts (logs, steps.yaml, workflow-state.yaml, short-term
        memory) are intentionally preserved after workflow completion.  This
        allows the user to review results at any time.  Cleanup only occurs
        at the start of the *next* ``-ai-steps`` invocation (without
        ``-continue``), and only after explicit user confirmation.

        Short-term memory is maintained throughout the workflow lifecycle,
        giving Claude context about the overall mission, progress, and
        current step when executing individual steps.  It persists on disk
        after completion and across ``-continue`` resumes and user quits.
        Short-term memory is stored inside the ai-code directory at
        ``<script_dir>/memory/short-term.md``.

        Long-term memory is updated inline during each step execution —
        Claude outputs an updated ``.ai-code/long-term.md`` block alongside
        code blocks, so the memory always reflects the current codebase
        state without separate API calls.  Long-term memory is stored at
        ``<parent_of_script_dir>/.ai-code/long-term.md``.

        Memory context (long-term, short-term, git history) is also
        injected during source generation phases so Claude can make
        informed decisions about which files are relevant.

Commit message format:
    feature_title: ai-step X/Y: category - step_title
    e.g. User Preferences: ai-step 1/5: database - Add preferences table
"""

import hashlib
import os
import shutil
import sys
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set

import yaml as _yaml

from lib.config import Config
from lib.files import add_source
from lib.tree import get_directory_tree
from lib.git import has_uncommitted_changes, commit_changes, revert_to_last_commit, is_git_available
from lib.export import export_md_file, log_prompt
from lib.memory import save_short_term_memory, clear_short_term_memory
from lib.utils import warn, COLOR_GREEN, COLOR_YELLOW, COLOR_CYAN, COLOR_RESET, play_bell

from lib.tools.tool_source_generate import generate_source
from lib.tools.tool_prompt_expand import expand_prompt
from lib.tools.tool_prompt_stepize import stepize_prompt
from lib.tools.tool_prompt_execute import execute_prompt
from lib.tools.tool_user_confirm import confirm_step

# Maximum retry attempts for a single step before forced skip
_MAX_STEP_RETRIES = 3

# Name of the YAML state file used for -continue resume capability
_STATE_FILENAME = "workflow-state.yaml"


# ══════════════════════════════════════════════════════════════════════════════
# State persistence helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_prompt_hash(prompt: str) -> str:
    """Return an MD5 hex digest of *prompt* for identity validation.

    Used by ``-continue`` to verify that the saved state matches the
    current prompt — if the user changed the prompt, the old state is
    invalid and we must start fresh.
    """
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def _save_workflow_state(state: Dict[str, Any], output_dir: str) -> None:
    """Atomically write *state* to the workflow state file.

    Uses a write-to-temp-then-rename pattern so that a crash during write
    cannot corrupt the state file — either the old or the new version is
    always intact on disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, _STATE_FILENAME)
    tmp_path = state_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(state, f, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, state_path)
    except Exception as e:
        warn(f"[state] Failed to save workflow state: {e}")


def _load_workflow_state(output_dir: str) -> Optional[Dict[str, Any]]:
    """Load the workflow state file, returning None if absent or corrupt."""
    state_path = os.path.join(output_dir, _STATE_FILENAME)
    if not os.path.isfile(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f)
    except Exception as e:
        warn(f"[state] Failed to load workflow state: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Previous workflow cleanup
# ══════════════════════════════════════════════════════════════════════════════

def _has_previous_workflow_artifacts(steps_output_dir: str) -> bool:
    """Return True if the ai-steps output directory contains any artifacts
    from a previous workflow run (state file, phase dirs, step dirs, etc.).

    An empty or non-existent directory returns False.
    """
    if not os.path.isdir(steps_output_dir):
        return False
    # Check for any files or subdirectories — any content means a previous run
    try:
        return len(os.listdir(steps_output_dir)) > 0
    except OSError:
        return False


def _describe_previous_workflow(steps_output_dir: str) -> None:
    """Print a summary of the previous workflow artifacts found on disk.

    Reads the saved state file (if present) to show feature title, step
    counts, and completion status.  Also lists the artifact directory
    contents at a high level so the user knows what will be deleted.
    """
    state = _load_workflow_state(steps_output_dir)

    print(f"\n{COLOR_YELLOW}{'=' * 60}")
    print(f"  PREVIOUS WORKFLOW ARTIFACTS DETECTED")
    print(f"{'=' * 60}{COLOR_RESET}")

    if state:
        feature = state.get("feature_title", "unknown")
        completed = state.get("completed_steps", [])
        skipped = state.get("skipped_steps", [])
        all_steps = state.get("steps") or []
        total = len(all_steps)
        phase = state.get("phase_completed", 0)

        print(f"  Feature:   {feature}")
        print(f"  Phase:     {phase}/3 completed")
        if total > 0:
            print(f"  Steps:     {len(completed)} completed, {len(skipped)} skipped, {total} total")
            remaining = total - len(completed) - len(skipped)
            if remaining == 0:
                print(f"  Status:    FINISHED (all steps processed)")
            else:
                print(f"  Status:    INCOMPLETE ({remaining} steps remaining)")
        print()

    # List top-level contents of the artifact directory
    try:
        entries = sorted(os.listdir(steps_output_dir))
        if entries:
            print(f"  Artifact directory: {steps_output_dir}")
            print(f"  Contents:")
            for entry in entries:
                entry_path = os.path.join(steps_output_dir, entry)
                if os.path.isdir(entry_path):
                    # Count files inside subdirectory for context
                    file_count = sum(len(files) for _, _, files in os.walk(entry_path))
                    print(f"    📁 {entry}/ ({file_count} file{'s' if file_count != 1 else ''})")
                else:
                    print(f"    📄 {entry}")
            print()
    except OSError:
        pass

    print(f"  Starting a new workflow will {COLOR_YELLOW}DELETE{COLOR_RESET} all listed artifacts.")
    print(f"  Use {COLOR_CYAN}-ai-steps -continue{COLOR_RESET} to resume the previous workflow instead.")
    print(f"{'=' * 60}")


def _confirm_and_cleanup_previous_workflow(
    steps_output_dir: str,
    memory_short_term_dir: str,
) -> bool:
    """Check for previous workflow artifacts, ask user to confirm deletion.

    Called at the start of a fresh (non-continue) ``-ai-steps`` run.  If
    previous artifacts exist, the user is shown a summary and asked to
    confirm deletion.  On confirmation, the entire ai-steps output
    directory is removed and recreated empty, and short-term memory is
    cleared.

    Parameters
    ----------
    steps_output_dir : str
        Path to the ai-steps output directory (``logs/claude/ai-steps/``).
    memory_short_term_dir : str
        Path to the short-term memory directory (``<script_dir>/memory/``).

    Returns
    -------
    bool
        True if the workflow should proceed (no artifacts found, or user
        confirmed deletion).  False if the user declined (workflow should
        abort).
    """
    if not _has_previous_workflow_artifacts(steps_output_dir):
        return True  # Nothing to clean up — proceed immediately

    _describe_previous_workflow(steps_output_dir)

    while True:
        try:
            choice = input("\nDelete previous workflow artifacts and start fresh? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[EOF/Interrupt — aborting]")
            return False

        if choice == "y":
            # Remove entire ai-steps output directory and recreate empty
            try:
                shutil.rmtree(steps_output_dir)
                print(f"[cleanup] Deleted: {steps_output_dir}")
            except Exception as e:
                warn(f"[cleanup] Failed to delete artifact directory: {e}")
                print("Please remove the directory manually and retry.")
                return False

            os.makedirs(steps_output_dir, exist_ok=True)

            # Clear short-term memory — it belongs to the old workflow
            clear_short_term_memory(memory_short_term_dir)
            print("[cleanup] Short-term memory cleared.")

            print("[cleanup] Previous workflow artifacts deleted. Starting fresh.\n")
            return True

        elif choice == "n":
            print(f"\n[ai-steps] Aborting. Use {COLOR_CYAN}-ai-steps -continue{COLOR_RESET} to resume the previous workflow.")
            return False

        else:
            print("  Please enter 'y' or 'n'.")


# ══════════════════════════════════════════════════════════════════════════════
# Short-term memory helper
# ══════════════════════════════════════════════════════════════════════════════

def _update_short_term(
    memory_short_term_dir: str,
    minimal_prompt: str,
    expanded_summary: str,
    steps: Optional[List[Dict[str, Any]]],
    feature_title: str,
    completed: Set[int],
    skipped: Set[int],
    current_step: Optional[Dict[str, Any]],
    phase_status: Dict[str, str],
) -> None:
    """Build and save short-term memory reflecting current workflow state.

    This provides Claude with context about the overall ai-steps mission
    when executing individual steps, so it understands the bigger picture.

    The content is structured as a lightweight Markdown document with
    sections for goal, phase progress, step overview (with completion
    markers), current step detail, and a condensed expansion summary.

    Short-term memory is saved to ``memory_short_term_dir`` which lives
    inside the ai-code script directory (``<script_dir>/memory/``).

    This function is intentionally fire-and-forget — a failure to update
    short-term memory must never block or crash the workflow.
    """
    lines = ["# Current Workflow\n"]

    # Goal section — what we're trying to accomplish
    lines.append("## Goal")
    lines.append(minimal_prompt.strip()[:300])  # Truncate very long prompts
    lines.append("")

    # Feature context
    lines.append(f"## Feature: {feature_title}")
    lines.append("")

    # Phase progress
    lines.append("## Phase Progress")
    for phase, status in phase_status.items():
        lines.append(f"- {phase}: {status}")
    lines.append("")

    # Steps overview (if available)
    if steps:
        lines.append("## Steps Overview")
        for s in steps:
            num = s["number"]
            title = s["title"]
            cat = s.get("category", "general")
            if num in completed:
                marker = "✓"
            elif num in skipped:
                marker = "⊘ skipped"
            elif current_step and num == current_step["number"]:
                marker = "← current"
            else:
                marker = ""
            lines.append(f"  {num}. [{cat}] {title} {marker}")
        lines.append("")

    # Current step detail
    if current_step:
        lines.append(f"## Current Step: {current_step['number']}. {current_step['title']}")
        lines.append(f"Category: {current_step.get('category', 'general')}")
        lines.append("")

    # Key context from expansion (condensed)
    lines.append("## Expanded Specification Summary")
    lines.append(expanded_summary.strip()[:500])
    lines.append("")

    content = "\n".join(lines)
    save_short_term_memory(memory_short_term_dir, content)


# ══════════════════════════════════════════════════════════════════════════════
# Tree / source helpers
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Commit message formatting
# ══════════════════════════════════════════════════════════════════════════════

def _format_commit_message(
    step_number: int,
    total_steps: int,
    step_title: str,
    step_category: str,
    feature_title: str,
) -> str:
    """Build a structured commit message with category and feature context.

    Format: ``feature_title: ai-step X/Y: category - step_title``
    e.g.  ``User Preferences: ai-step 1/5: database - Add preferences table``

    Parameters
    ----------
    step_number : int
        1-based step index.
    total_steps : int
        Total number of steps in the workflow.
    step_title : str
        Short description of what this step implements.
    step_category : str
        System area affected (e.g. "database", "admin", "api").
    feature_title : str
        Overall feature label (e.g. "User Preferences").
    """
    return f"{feature_title}: ai-step {step_number}/{total_steps}: {step_category} - {step_title}"


# ══════════════════════════════════════════════════════════════════════════════
# Progress display helpers
# ══════════════════════════════════════════════════════════════════════════════

def _step_prefix(step_number: int, total_steps: int) -> str:
    """Return a consistent ``[Step X/Y]`` prefix for log messages."""
    return f"[Step {step_number}/{total_steps}]"


def _print_step_header(
    step_number: int,
    total_steps: int,
    step_title: str,
    step_category: str,
    feature_title: str,
    completed: int,
    skipped: int,
) -> None:
    """Print a prominent header for the step being executed, including
    live progress counts and the category/feature context."""
    remaining = total_steps - completed - skipped - 1  # -1 = current step
    print(f"\n{COLOR_GREEN}{'═' * 60}")
    print(f"  STEP {step_number}/{total_steps}: {step_title}")
    print(f"  Category: {step_category} | Feature: {feature_title}")
    print(f"  Progress: {completed} completed | {skipped} skipped | {remaining} after this")
    print(f"{'═' * 60}{COLOR_RESET}\n")


def _print_summary(completed: int, skipped: int, total: int) -> None:
    """Print a summary of the ai-steps workflow execution."""
    remaining = total - completed - skipped
    print(f"\n{'=' * 60}")
    print(f"  AI-STEPS WORKFLOW SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total steps:     {total}")
    print(f"  Completed:       {completed}")
    print(f"  Skipped:         {skipped}")
    print(f"  Remaining:       {remaining}")
    print(f"{'=' * 60}\n")


def _websearch_status_str(cfg: Config) -> str:
    """Return a human-readable string describing websearch configuration."""
    if cfg.websearch:
        return f"ENABLED (max_results={cfg.websearch_max_results})"
    return "DISABLED"


# ══════════════════════════════════════════════════════════════════════════════
# Main workflow
# ══════════════════════════════════════════════════════════════════════════════

def run_ai_steps_workflow(cfg: Config, args: Namespace) -> None:
    """Execute the ``-ai-steps`` automated multi-step workflow.

    This workflow requires git to be available. Each accepted step is
    committed with a structured message

    **Artifact retention policy:**  Workflow artifacts (logs, phase
    directories, ``steps.yaml``, ``workflow-state.yaml``, and short-term
    memory) are preserved after workflow completion — both on successful
    finish and on user quit.  Cleanup only occurs at the beginning of
    the *next* ``-ai-steps`` invocation (without ``-continue``), after
    the user explicitly confirms deletion of the previous artifacts.
    This lets the user review results, debug issues, or resume at any
    time without losing context.

    Short-term memory is updated at each phase transition and before
    each step execution, providing Claude with awareness of the overall
    mission, progress, and current step when executing individual steps.
    Short-term memory is stored at ``<script_dir>/memory/short-term.md``.

    Long-term memory is updated inline during each step execution —
    Claude outputs a ``.ai-code/long-term.md`` block alongside code blocks.
    This is handled by ``execute_prompt`` via
    ``build_memory_update_instructions`` and
    ``extract_and_save_memory_from_response``, so the memory always
    reflects the current codebase state without separate API calls.
    Long-term memory is stored at ``<parent_of_script_dir>/.ai-code/long-term.md``.

    Memory context (long-term, short-term, git history) is also injected
    into source generation calls so Claude has project awareness when
    deciding which files are relevant for each phase/step.

    When ``args.continue_steps`` is True, the workflow resumes from the
    last saved checkpoint — skipping already-completed phases and steps.
    If uncommitted changes are found (e.g. from a crash), they are
    automatically reverted before resuming.

    Parameters
    ----------
    cfg : Config
        Fully resolved configuration.
    args : Namespace
        Parsed CLI arguments (``continue_steps``, ``include_pdf``).
    """
    continue_mode = getattr(args, "continue_steps", False)

    # -- Pre-flight checks ----------------------------------------------------
    if not is_git_available():
        print("ERROR: git is required for the -ai-steps workflow but is not available.")
        print("Please install git or ensure it is in your PATH.")
        sys.exit(1)

    ai_shared_file_types: list = []
    if args.include_pdf:
        ai_shared_file_types.append("pdf")

    # Base output directory for all ai-steps artifacts
    steps_output_dir = os.path.join(cfg.claude_output_dir, "ai-steps")

    # -- Previous workflow cleanup (fresh runs only) --------------------------
    # When starting a fresh workflow (no -continue), check if artifacts from
    # a previous run exist.  If so, show a summary and ask the user to
    # confirm deletion before proceeding.  This ensures no accidental data
    # loss while keeping the workflow directory clean for the new run.
    if not continue_mode:
        if _has_previous_workflow_artifacts(steps_output_dir):
            should_proceed = _confirm_and_cleanup_previous_workflow(
                steps_output_dir, cfg.memory_short_term_dir,
            )
            if not should_proceed:
                return  # User declined — abort without error

    os.makedirs(steps_output_dir, exist_ok=True)

    # -- State initialisation -------------------------------------------------
    # These variables track workflow progress and are populated either from
    # a fresh run or from a saved state file when resuming.
    expanded_prompt_text: Optional[str] = None
    steps: Optional[List[Dict[str, Any]]] = None
    feature_title: str = "ai-steps"  # default fallback; replaced by stepize result
    completed_steps_set: Set[int] = set()
    skipped_steps_set: Set[int] = set()
    prompt_hash = _compute_prompt_hash(cfg.prompt)

    # -- Handle -continue: load saved state -----------------------------------
    if continue_mode:
        saved_state = _load_workflow_state(steps_output_dir)

        if saved_state is None:
            warn("[continue] No saved workflow state found. Starting fresh.")
            continue_mode = False  # fall through to normal flow
        elif saved_state.get("prompt_hash") != prompt_hash:
            warn(
                "[continue] Prompt has changed since the last run. "
                "Saved state is invalid — starting fresh."
            )
            continue_mode = False
        else:
            # Valid state found — restore progress
            expanded_prompt_text = saved_state.get("expanded_prompt")
            steps = saved_state.get("steps")
            feature_title = saved_state.get("feature_title", "ai-steps")
            completed_steps_set = set(saved_state.get("completed_steps", []))
            skipped_steps_set = set(saved_state.get("skipped_steps", []))

            phase_reached = saved_state.get("phase_completed", 0)

            print(f"\n{COLOR_CYAN}[continue] Resuming from saved state:{COLOR_RESET}")
            print(f"  Feature: {feature_title}")
            if expanded_prompt_text:
                print(f"  Expanded prompt: {len(expanded_prompt_text)} chars (Phase 1 done)")
            if steps:
                print(f"  Steps loaded: {len(steps)} total")
            if completed_steps_set:
                print(f"  Completed steps: {sorted(completed_steps_set)}")
            if skipped_steps_set:
                print(f"  Skipped steps: {sorted(skipped_steps_set)}")

            # If there are uncommitted changes (crash residue), revert them
            if has_uncommitted_changes(ignore_dir_name=cfg.script_dir_name):
                print(f"\n{COLOR_YELLOW}[continue] Uncommitted changes detected (likely from interrupted step).{COLOR_RESET}")
                print("[continue] Reverting to last commit to restore clean state...")
                reverted = revert_to_last_commit()
                if not reverted:
                    print("ERROR: Failed to revert uncommitted changes. Please resolve manually.")
                    sys.exit(1)

    # -- Non-continue mode: require clean working tree ------------------------
    if not continue_mode:
        if has_uncommitted_changes(ignore_dir_name=cfg.script_dir_name):
            print("ERROR: The -ai-steps workflow requires a clean git working tree.")
            print("Please commit or stash all changes before running -ai-steps.")
            sys.exit(1)

    # -- Show workflow configuration ------------------------------------------
    # Display all relevant settings upfront so the user knows what's active
    # for the entire multi-step workflow.  Web search status is especially
    # important since it applies to all API calls (source gen, expand,
    # stepize, and execute).
    print(f"\n{'=' * 60}")
    print(f"  AI-STEPS WORKFLOW")
    print(f"  Provider: {cfg.provider}")
    print(f"  Model: {cfg.model}")
    print(f"  Web search: {_websearch_status_str(cfg)}")
    print(f"  Extended thinking: {'ENABLED (budget=' + str(cfg.max_tokens_think) + ')' if cfg.max_tokens_think > 0 else 'DISABLED'}")
    print(f"  Minimal prompt: {cfg.prompt[:80]}{'...' if len(cfg.prompt) > 80 else ''}")
    if continue_mode:
        print(f"  Mode: RESUMING from saved checkpoint")
        print(f"  Feature: {feature_title}")
    print(f"{'=' * 60}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1: PROMPT EXPANSION
    # ═════════════════════════════════════════════════════════════════════════
    if expanded_prompt_text is None:
        print(f"\n{COLOR_CYAN}{'-' * 60}")
        print(f"  PHASE 1: PROMPT EXPANSION")
        print(f"{'-' * 60}{COLOR_RESET}\n")

        phase1_dir = os.path.join(steps_output_dir, "phase1-expand")
        os.makedirs(phase1_dir, exist_ok=True)

        # 1.1 Generate source for prompt expansion
        # include_short_term_memory=False: short-term memory is not yet
        # populated at this point (it's created at the end of Phase 1).
        # Long-term memory and git history ARE included so Claude knows
        # the project architecture and recent changes when selecting files.
        print("[Phase 1.1] Generating source list for prompt expansion...")
        tree_str = _build_current_tree(cfg, ai_shared_file_types)

        source_result = generate_source(
            cfg=cfg,
            prompt=cfg.prompt,
            tree_str=tree_str,
            example_source=cfg.source,
            output_dir=phase1_dir,
            include_short_term_memory=False,
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

        # Checkpoint: save state after Phase 1 completes
        _save_workflow_state({
            "prompt_hash": prompt_hash,
            "phase_completed": 1,
            "expanded_prompt": expanded_prompt_text,
            "feature_title": feature_title,
            "steps": None,
            "completed_steps": [],
            "skipped_steps": [],
        }, steps_output_dir)
        print("[Phase 1] ✓ State saved (Phase 1 complete)")

        # Initialize short-term memory with workflow goal and expansion summary.
        # This gives Claude early context about the mission even before steps
        # are decomposed, useful if Phase 2 itself needs project awareness.
        # Short-term memory is stored inside the ai-code script directory.
        _update_short_term(
            cfg.memory_short_term_dir,
            minimal_prompt=cfg.prompt,
            expanded_summary=expanded_prompt_text[:500],
            steps=None,
            feature_title=feature_title,
            completed=set(),
            skipped=set(),
            current_step=None,
            phase_status={"expand": "Complete", "stepize": "Pending", "execute": "Pending"},
        )
    else:
        print(f"\n{COLOR_CYAN}{'-' * 60}")
        print(f"  PHASE 1: PROMPT EXPANSION — SKIPPED (loaded from state)")
        print(f"{'-' * 60}{COLOR_RESET}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2: STEP DECOMPOSITION
    # ═════════════════════════════════════════════════════════════════════════
    if steps is None:
        print(f"\n{COLOR_CYAN}{'-' * 60}")
        print(f"  PHASE 2: STEP DECOMPOSITION")
        print(f"{'-' * 60}{COLOR_RESET}\n")

        phase2_dir = os.path.join(steps_output_dir, "phase2-stepize")
        os.makedirs(phase2_dir, exist_ok=True)

        # 2.1 Generate source for step-ization
        # include_short_term_memory=True: short-term memory was populated at
        # the end of Phase 1 with the workflow goal and expansion summary.
        # Including it gives Claude awareness of the overall mission when
        # deciding which files are relevant for decomposition.
        print("[Phase 2.1] Generating source list for step decomposition...")
        tree_str = _build_current_tree(cfg, ai_shared_file_types)

        step_source_result = generate_source(
            cfg=cfg,
            prompt=expanded_prompt_text,
            tree_str=tree_str,
            example_source=cfg.source,
            output_dir=phase2_dir,
            include_short_term_memory=True,
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
        # Extract feature_title from the stepize result for commit messages
        feature_title = steps_result.get("feature_title", "ai-steps")
        print(f"\n[Phase 2.2] Feature: {feature_title}")
        print(f"[Phase 2.2] Decomposed into {len(steps)} step(s)")

        # Save steps for reference
        steps_yaml_str = _yaml.safe_dump(
            {"feature_title": feature_title, "steps": steps},
            sort_keys=False,
            allow_unicode=True,
        )
        export_md_file(steps_yaml_str, "steps.yaml", steps_output_dir)

        # Checkpoint: save state after Phase 2 completes
        _save_workflow_state({
            "prompt_hash": prompt_hash,
            "phase_completed": 2,
            "expanded_prompt": expanded_prompt_text,
            "feature_title": feature_title,
            "steps": steps,
            "completed_steps": sorted(completed_steps_set),
            "skipped_steps": sorted(skipped_steps_set),
        }, steps_output_dir)
        print("[Phase 2] ✓ State saved (Phase 2 complete)")

        # Update short-term memory with the full step list so Claude has a
        # roadmap of the entire implementation plan before execution begins.
        _update_short_term(
            cfg.memory_short_term_dir,
            minimal_prompt=cfg.prompt,
            expanded_summary=expanded_prompt_text[:500],
            steps=steps,
            feature_title=feature_title,
            completed=completed_steps_set,
            skipped=skipped_steps_set,
            current_step=None,
            phase_status={"expand": "Complete", "stepize": "Complete", "execute": "Starting"},
        )
    else:
        print(f"\n{COLOR_CYAN}{'-' * 60}")
        print(f"  PHASE 2: STEP DECOMPOSITION — SKIPPED (loaded from state)")
        print(f"  Feature: {feature_title}")
        print(f"{'-' * 60}{COLOR_RESET}\n")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3: STEP EXECUTION LOOP
    # ═════════════════════════════════════════════════════════════════════════
    total_steps = len(steps)
    completed_count = len(completed_steps_set)
    skipped_count = len(skipped_steps_set)

    print(f"\n{COLOR_CYAN}{'-' * 60}")
    print(f"  PHASE 3: STEP EXECUTION ({total_steps} steps total)")
    print(f"  Feature: {feature_title}")
    print(f"  Web search: {_websearch_status_str(cfg)}")
    if completed_count or skipped_count:
        print(f"  Resuming: {completed_count} completed, {skipped_count} skipped, "
              f"{total_steps - completed_count - skipped_count} remaining")
    print(f"{'-' * 60}{COLOR_RESET}\n")

    for step in steps:
        step_number = step["number"]
        step_title = step["title"]
        step_category = step.get("category", "general")
        step_prompt = step["prompt"]
        step_source = step.get("source", cfg.source)

        # -- Skip already-processed steps (from -continue) -------------------
        if step_number in completed_steps_set:
            print(f"{COLOR_CYAN}[Step {step_number}/{total_steps}] "
                  f"SKIPPED (already completed): {step_title}{COLOR_RESET}")
            continue

        if step_number in skipped_steps_set:
            print(f"{COLOR_YELLOW}[Step {step_number}/{total_steps}] "
                  f"SKIPPED (previously skipped): {step_title}{COLOR_RESET}")
            continue

        # -- Step execution ---------------------------------------------------
        step_dir = os.path.join(steps_output_dir, f"step-{step_number}")
        os.makedirs(step_dir, exist_ok=True)

        prefix = _step_prefix(step_number, total_steps)
        _print_step_header(
            step_number, total_steps, step_title,
            step_category, feature_title,
            completed_count, skipped_count,
        )

        current_prompt = step_prompt
        retry_count = 0

        while True:
            # -- 3.1 Generate fresh source for this step ---------------------
            # Rebuild tree because previous steps may have changed files.
            # include_short_term_memory=True: during step execution, short-term
            # memory contains the full step list, progress markers, and current
            # step context — critical for Claude to understand which files are
            # relevant in the context of the overall implementation plan.
            print(f"{prefix} Generating source list...")
            tree_str = _build_current_tree(cfg, ai_shared_file_types)

            step_source_result = generate_source(
                cfg=cfg,
                prompt=current_prompt,
                tree_str=tree_str,
                example_source=step_source if step_source else cfg.source,
                output_dir=os.path.join(step_dir, f"source-gen-attempt-{retry_count}"),
                include_short_term_memory=True,
            )

            if step_source_result["status"] != "ok":
                warn(f"{prefix} Source generation failed: {step_source_result.get('error')}")
                warn(f"{prefix} Falling back to step-defined source list: {step_source}")
                exec_source_paths = step_source if step_source else cfg.source
            else:
                exec_source_paths = step_source_result["source_list"]
                print(f"{prefix} Source list: {len(exec_source_paths)} entries")

            # -- 3.2 Execute the step ----------------------------------------
            # Update short-term memory with current step context so Claude
            # knows which step it is executing, what has been completed, and
            # the overall mission — providing big-picture awareness for each
            # atomic step execution.
            _update_short_term(
                cfg.memory_short_term_dir,
                minimal_prompt=cfg.prompt,
                expanded_summary=expanded_prompt_text[:500],
                steps=steps,
                feature_title=feature_title,
                completed=completed_steps_set,
                skipped=skipped_steps_set,
                current_step=step,
                phase_status={"expand": "Complete", "stepize": "Complete", "execute": f"Step {step_number}/{total_steps}"},
            )

            print(f"\n{prefix} Executing step...")
            # Memory updates happen inline: execute_prompt appends memory
            # update instructions to the prompt and extracts the memory block
            # from the response.  No separate API calls needed.
            exec_result = execute_prompt(
                cfg=cfg,
                prompt=current_prompt,
                source_paths=exec_source_paths,
                apply_to_disk=True,
                output_dir=step_dir,
                label=f"attempt-{retry_count}-",
                include_short_term_memory=True,
            )

            if exec_result["status"] != "ok":
                warn(f"{prefix} Execution failed: {exec_result.get('error')}")
                # Revert any partial changes
                print(f"{prefix} Reverting changes from failed execution...")
                revert_to_last_commit()
                # Re-save state after revert (in case git clean removed it)
                _save_workflow_state({
                    "prompt_hash": prompt_hash,
                    "phase_completed": 2,
                    "expanded_prompt": expanded_prompt_text,
                    "feature_title": feature_title,
                    "steps": steps,
                    "completed_steps": sorted(completed_steps_set),
                    "skipped_steps": sorted(skipped_steps_set),
                }, steps_output_dir)

                retry_count += 1
                if retry_count >= _MAX_STEP_RETRIES:
                    warn(f"{prefix} Max retries ({_MAX_STEP_RETRIES}) reached. Skipping step.")
                    skipped_steps_set.add(step_number)
                    skipped_count += 1
                    _save_workflow_state({
                        "prompt_hash": prompt_hash,
                        "phase_completed": 2,
                        "expanded_prompt": expanded_prompt_text,
                        "feature_title": feature_title,
                        "steps": steps,
                        "completed_steps": sorted(completed_steps_set),
                        "skipped_steps": sorted(skipped_steps_set),
                    }, steps_output_dir)
                    break

                # Ask user whether to retry or skip
                if cfg.sound_enabled:
                    play_bell()

                user_result = confirm_step(step_number, f"{step_title} [EXECUTION FAILED]")
                if user_result["action"] == "retry" and user_result["modification"]:
                    current_prompt = step_prompt + "\n\nAdditional instructions:\n" + user_result["modification"]
                    continue
                elif user_result["action"] == "retry":
                    continue
                elif user_result["action"] == "skip":
                    skipped_steps_set.add(step_number)
                    skipped_count += 1
                    _save_workflow_state({
                        "prompt_hash": prompt_hash,
                        "phase_completed": 2,
                        "expanded_prompt": expanded_prompt_text,
                        "feature_title": feature_title,
                        "steps": steps,
                        "completed_steps": sorted(completed_steps_set),
                        "skipped_steps": sorted(skipped_steps_set),
                    }, steps_output_dir)
                    break
                elif user_result["action"] == "quit":
                    print(f"\n{prefix} Quitting workflow.")
                    _save_workflow_state({
                        "prompt_hash": prompt_hash,
                        "phase_completed": 2,
                        "expanded_prompt": expanded_prompt_text,
                        "feature_title": feature_title,
                        "steps": steps,
                        "completed_steps": sorted(completed_steps_set),
                        "skipped_steps": sorted(skipped_steps_set),
                    }, steps_output_dir)
                    _print_summary(completed_count, skipped_count, total_steps)
                    log_prompt(cfg.prompt, cfg.logs_dir)
                    return
                else:
                    # "continue" after failure — treat as skip
                    skipped_steps_set.add(step_number)
                    skipped_count += 1
                    _save_workflow_state({
                        "prompt_hash": prompt_hash,
                        "phase_completed": 2,
                        "expanded_prompt": expanded_prompt_text,
                        "feature_title": feature_title,
                        "steps": steps,
                        "completed_steps": sorted(completed_steps_set),
                        "skipped_steps": sorted(skipped_steps_set),
                    }, steps_output_dir)
                    break

            # -- 3.3 Ask user to confirm -------------------------------------
            # Play bell BEFORE prompting user input
            if cfg.sound_enabled:
                play_bell()

            user_result = confirm_step(step_number, step_title)

            if user_result["action"] == "continue":
                # Accept: commit with structured message including category and feature
                commit_msg = _format_commit_message(
                    step_number, total_steps, step_title,
                    step_category, feature_title,
                )
                committed = commit_changes(commit_msg, ignore_dir_name=cfg.script_dir_name)
                if committed:
                    print(f"{prefix} ✓ Changes committed: {commit_msg}")
                else:
                    warn(f"{prefix} Git commit failed or nothing to commit.")
                completed_steps_set.add(step_number)
                completed_count += 1

                # Save state after successful commit
                _save_workflow_state({
                    "prompt_hash": prompt_hash,
                    "phase_completed": 2,
                    "expanded_prompt": expanded_prompt_text,
                    "feature_title": feature_title,
                    "steps": steps,
                    "completed_steps": sorted(completed_steps_set),
                    "skipped_steps": sorted(skipped_steps_set),
                }, steps_output_dir)
                print(f"{prefix} ✓ State saved")
                break

            elif user_result["action"] == "retry":
                # Revert changes, modify prompt, re-execute
                print(f"{prefix} Reverting changes for retry...")
                revert_to_last_commit()
                # Re-save state after revert
                _save_workflow_state({
                    "prompt_hash": prompt_hash,
                    "phase_completed": 2,
                    "expanded_prompt": expanded_prompt_text,
                    "feature_title": feature_title,
                    "steps": steps,
                    "completed_steps": sorted(completed_steps_set),
                    "skipped_steps": sorted(skipped_steps_set),
                }, steps_output_dir)

                retry_count += 1
                if retry_count >= _MAX_STEP_RETRIES:
                    warn(f"{prefix} Max retries ({_MAX_STEP_RETRIES}) reached. Skipping step.")
                    skipped_steps_set.add(step_number)
                    skipped_count += 1
                    _save_workflow_state({
                        "prompt_hash": prompt_hash,
                        "phase_completed": 2,
                        "expanded_prompt": expanded_prompt_text,
                        "feature_title": feature_title,
                        "steps": steps,
                        "completed_steps": sorted(completed_steps_set),
                        "skipped_steps": sorted(skipped_steps_set),
                    }, steps_output_dir)
                    break

                if user_result["modification"]:
                    current_prompt = step_prompt + "\n\nAdditional instructions:\n" + user_result["modification"]
                    print(f"{prefix} Prompt modified with user input. Retrying...")
                else:
                    print(f"{prefix} Retrying with same prompt...")
                continue

            elif user_result["action"] == "skip":
                print(f"{prefix} Reverting changes and skipping...")
                revert_to_last_commit()
                skipped_steps_set.add(step_number)
                skipped_count += 1
                _save_workflow_state({
                    "prompt_hash": prompt_hash,
                    "phase_completed": 2,
                    "expanded_prompt": expanded_prompt_text,
                    "feature_title": feature_title,
                    "steps": steps,
                    "completed_steps": sorted(completed_steps_set),
                    "skipped_steps": sorted(skipped_steps_set),
                }, steps_output_dir)
                break

            elif user_result["action"] == "quit":
                # User quit — preserve all artifacts (state, logs, short-term
                # memory) on disk so -continue can resume with full workflow
                # context intact.
                print(f"{prefix} Reverting changes and quitting...")
                revert_to_last_commit()
                # Save state so -continue can resume later
                _save_workflow_state({
                    "prompt_hash": prompt_hash,
                    "phase_completed": 2,
                    "expanded_prompt": expanded_prompt_text,
                    "feature_title": feature_title,
                    "steps": steps,
                    "completed_steps": sorted(completed_steps_set),
                    "skipped_steps": sorted(skipped_steps_set),
                }, steps_output_dir)
                _print_summary(completed_count, skipped_count, total_steps)
                log_prompt(cfg.prompt, cfg.logs_dir)
                return

    # -- Final summary --------------------------------------------------------
    _print_summary(completed_count, skipped_count, total_steps)
    log_prompt(cfg.prompt, cfg.logs_dir)

    # Mark Phase 3 as complete in the saved state so that
    # _describe_previous_workflow shows "3/3" instead of "2/3".
    _save_workflow_state({
        "prompt_hash": prompt_hash,
        "phase_completed": 3,
        "expanded_prompt": expanded_prompt_text,
        "feature_title": feature_title,
        "steps": steps,
        "completed_steps": sorted(completed_steps_set),
        "skipped_steps": sorted(skipped_steps_set),
    }, steps_output_dir)

    # -- Artifact retention ---------------------------------------------------
    # All workflow artifacts (state file, logs, steps.yaml, short-term memory)
    # are intentionally preserved after completion.  This allows the user to:
    #   - Review the full execution history and AI responses
    #   - Debug issues with specific steps
    #   - Resume with -continue if steps were skipped
    #
    # Cleanup happens at the START of the next -ai-steps invocation (without
    # -continue), after the user explicitly confirms deletion via
    # _confirm_and_cleanup_previous_workflow().
    if completed_count + skipped_count >= total_steps:
        print(f"\n{COLOR_GREEN}[ai-steps] Workflow complete.{COLOR_RESET}")
        print(f"  Artifacts preserved at: {steps_output_dir}")
        print(f"  Run {COLOR_CYAN}-ai-steps{COLOR_RESET} again to start a new workflow (will prompt to delete these).")
        print(f"  Run {COLOR_CYAN}-ai-steps -continue{COLOR_RESET} to re-run skipped steps (if any).")
