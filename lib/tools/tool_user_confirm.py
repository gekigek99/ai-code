"""
lib.tools.tool_user_confirm — interactive user confirmation and feedback.

Public API:
    confirm_step(step_number, step_title) -> dict
        Prompt the user to accept, retry, skip, or quit after a step is applied.
"""

import sys


def confirm_step(step_number: int, step_title: str) -> dict:
    """Prompt the user interactively after a step has been applied.

    Presents options:
      [y] accept and continue
      [r] retry with modifications (prompts for additional text)
      [s] skip this step (revert changes)
      [q] quit the workflow (revert changes)

    Parameters
    ----------
    step_number : int
        The 1-based step index.
    step_title : str
        Human-readable step title for display.

    Returns
    -------
    dict
        ``{"action": "continue"|"retry"|"skip"|"quit", "modification": str|None}``
        *modification* is only non-None when action is ``"retry"``.
    """
    print(f"\n{'='*60}")
    print(f"  Step {step_number}: {step_title}")
    print(f"{'='*60}")
    print("  [y] Accept and continue")
    print("  [r] Retry with modifications")
    print("  [s] Skip this step (revert changes)")
    print("  [q] Quit workflow (revert changes)")
    print(f"{'='*60}")

    while True:
        try:
            choice = input("\nYour choice [y/r/s/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Treat EOF / Ctrl+C as quit to avoid hanging
            print("\n[EOF/Interrupt detected — quitting]")
            return {"action": "quit", "modification": None}

        if choice in ("y", "yes"):
            return {"action": "continue", "modification": None}

        elif choice in ("r", "retry"):
            print("\nEnter modifications to the step prompt (press Enter twice to finish):")
            lines = []
            try:
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        # Double empty line signals end of input
                        lines.pop()  # remove trailing blank
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\n[EOF/Interrupt during input — using what was entered]")

            modification = "\n".join(lines).strip()
            if not modification:
                print("[No modification entered — treating as accept]")
                return {"action": "continue", "modification": None}

            return {"action": "retry", "modification": modification}

        elif choice in ("s", "skip"):
            return {"action": "skip", "modification": None}

        elif choice in ("q", "quit"):
            return {"action": "quit", "modification": None}

        else:
            print(f"  Invalid choice: '{choice}'. Please enter y, r, s, or q.")
