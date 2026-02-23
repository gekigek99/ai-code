"""
lib.cli — command-line interface definition.

Public API:
    build_arg_parser() -> argparse.ArgumentParser
        Construct the argument parser with all supported CLI flags.
        Using argparse gives automatic ``-h`` / ``--help`` generation.
"""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct and return the argument parser with all supported flags.

    Using argparse gives us:
      * Automatic -h / --help generation from the per-flag descriptions.
      * Centralised flag definitions — add a flag once and it appears in help,
        parsing, and the returned namespace.
      * Type checking and nargs handling (e.g. -img takes one-or-more paths).
    """
    parser = argparse.ArgumentParser(
        prog="ai-code",
        description=(
            "AI-assisted code generation tool powered by Anthropic Claude.\n"
            "Reads project source files, builds a prompt, optionally sends it\n"
            "to the Claude API, and applies the returned file edits to disk."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python ai-code.py                 Gather sources & build prompt (dry run)\n"
            "  python ai-code.py -ai             Send prompt to Claude and apply edits\n"
            "  python ai-code.py -ai -f          Same, but skip uncommitted-git check\n"
            "  python ai-code.py -ai -pdf        Include PDF files in AI context\n"
            "  python ai-code.py -ai -img a.png  Attach image(s) to the prompt\n"
            "  python ai-code.py -last            Re-apply last saved Claude response\n"
            "  python ai-code.py -gen-source      Ask Claude to generate a source list\n"
            "  python ai-code.py -ai-steps        Run multi-step automated workflow\n"
            "  python ai-code.py -ai-steps -continue  Resume interrupted workflow\n"
        ),
    )

    # ── Boolean flags ────────────────────────────────────────────────────────
    parser.add_argument(
        "-ai",
        action="store_true",
        dest="run_ai",
        help=(
            "Send the assembled prompt to the Claude API and apply the "
            "returned file edits to disk.  Without this flag the script "
            "performs a dry run: gathers sources, builds the prompt, and "
            "exports it to the output directory without making an API call."
        ),
    )

    parser.add_argument(
        "-last",
        action="store_true",
        dest="run_last",
        help=(
            "Skip the API call and instead re-apply the last saved Claude "
            "response (clauderesponse.md in the output directory).  Useful "
            "for retrying file application after a manual review or edit."
        ),
    )

    parser.add_argument(
        "-gen-source",
        action="store_true",
        dest="run_gen_source",
        help=(
            "Ask Claude to generate a recommended source-file list for the "
            "current prompt.  The result is copied to the clipboard and "
            "saved as gen-source-clauderesponse.md.  The normal -ai flow "
            "is NOT executed when this flag is present."
        ),
    )

    parser.add_argument(
        "-ai-steps",
        action="store_true",
        dest="run_ai_steps",
        help=(
            "Run the automated multi-step AI workflow:\n"
            "  Phase 1: Expand the minimal prompt into a detailed specification.\n"
            "  Phase 2: Decompose the specification into ordered steps.\n"
            "  Phase 3: Execute each step with user confirmation and git commits.\n"
            "Requires git to be available and the working tree to be clean."
        ),
    )

    parser.add_argument(
        "-continue",
        action="store_true",
        dest="continue_steps",
        help=(
            "Resume an interrupted -ai-steps workflow from the last saved "
            "checkpoint.  Skips already-completed and previously-skipped steps. "
            "If uncommitted changes are detected (from a crash mid-step), they "
            "are automatically reverted.  Must be used together with -ai-steps."
        ),
    )

    parser.add_argument(
        "-f",
        action="store_true",
        dest="force",
        help=(
            "Force execution even when the git working tree has uncommitted "
            "changes.  By default the script refuses to run -ai if there are "
            "pending changes, to prevent accidental overwrites."
        ),
    )

    parser.add_argument(
        "-pdf",
        action="store_true",
        dest="include_pdf",
        help=(
            "Include PDF files found in the source directories in the AI "
            "context.  Text is extracted from PDFs via PyMuPDF and sent as "
            "plain text.  Without this flag, PDF files are discovered but "
            "not shared with the model."
        ),
    )

    # ── Value-carrying flags ─────────────────────────────────────────────────
    parser.add_argument(
        "-img",
        action="append",
        default=[],
        metavar="PATH",
        dest="image_paths",
        help=(
            "Attach an image file to the prompt.  May be specified multiple "
            "times for multiple images (e.g. -img a.png -img b.jpg).  "
            "Supported formats: JPEG, PNG, GIF, WebP.  Images are base64-"
            "encoded and sent as vision content blocks."
        ),
    )

    return parser
