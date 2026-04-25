# **AI Code**

A modular Python tool to run Anthropic Claude on your codebase, generate or update files, and manage outputs locally.

---

## Architecture

```
./ai-code.py                     # Slim entry point: CLI parsing + main() orchestration
./ai-code-prompt.yaml            # User config (gitignored)
./ai-code-prompt.yaml.example    # Config template
./lib/
    ├── __init__.py              # Package init
    ├── utils.py                 # ANSI colours, strip_ansi(), warn()
    ├── config.py                # YAML config loading, Config dataclass
    ├── cli.py                   # Argument parser (build_arg_parser)
    ├── files.py                 # File discovery, reading, exclusion, FileData dataclass
    ├── images.py                # Image processing: media type detection, base64 encoding
    ├── pdf.py                   # PDF text extraction (PyMuPDF)
    ├── tree.py                  # Directory tree building
    ├── git.py                   # Git utilities (uncommitted changes, commit, revert)
    ├── export.py                # Markdown export, prompt logging
    ├── validation.py            # Response validation, block_pattern regex
    ├── apply.py                 # File application logic (write/move/delete)
    ├── prompt_builder.py        # Message/content assembly for AI calls
    ├── memory.py                # Memory system: read/write/assemble memory context
    ├── token_tracker.py         # Token usage breakdown tracking and ASCII graph
    ├── providers/
    │   ├── __init__.py          # Provider package init
    │   └── claude.py            # Claude-specific: prompt_claude(), streaming, websearch
    ├── tools/                   # Reusable workflow tool functions
    │   ├── __init__.py          # Package init
    │   ├── tool_source_generate.py   # Generate source file list via Claude
    │   ├── tool_prompt_execute.py    # Execute prompt: discover → build → call → validate → apply
    │   ├── tool_prompt_expand.py     # Expand minimal prompt into detailed specification
    │   ├── tool_prompt_stepize.py    # Decompose prompt into ordered implementation steps
    │   └── tool_user_confirm.py      # Interactive user confirmation/feedback
    └── workflows/               # High-level workflow orchestrators
        ├── __init__.py          # Package init
        ├── workflow_ai.py            # Standard single-shot execution (-ai)
        ├── workflow_gen_source.py     # Source list generation (-gen-source)
        └── workflow_ai_steps.py      # Automated multi-step pipeline (-ai-steps)
./memory/                        # Short-term workflow memory (inside ai-code dir)
    └── short-term.md            # Current ai-steps workflow state
../.ai-code/                     # Long-term project memory (parent project root)
    └── memory/
        └── long-term.md         # Persistent project architecture memory
./logs/
    ├── prompt.log               # Prompt history
    ├── userfullprompt.md        # Full assembled prompt export
    └── claude/                  # Claude interaction outputs
        ├── clauderesponse.md
        ├── clauderesponse-recv.md
        ├── message_content.md
        ├── thinking.md
        ├── rawdata.md
        ├── gen-source-clauderesponse.md
        ├── gen-source-recv.md
        ├── gen-source-thinking.md
        ├── gen-source-rawdata.md
        └── ai-steps/            # Multi-step workflow artifacts
            ├── workflow-state.yaml  # Resume checkpoint (for -continue)
            ├── expanded-prompt.md
            ├── steps.yaml
            ├── phase1-expand/
            ├── phase2-stepize/
            ├── step-1/
            ├── step-2/
            └── step-N/
```

## Features

- Scans source folders with configurable include/exclude patterns
- Skips binary files automatically
- Shows a directory tree of your project with token estimates
- **Token usage breakdown graph**: displays per-component token estimates (system, long-term memory, short-term memory, git history, file data, prompt) before each API call
- Streams prompts to Claude and captures responses in real time
- Validates response structure before applying changes
- PDF text extraction support
- Image attachment support (JPEG, PNG, GIF, WebP)
- Web search integration (optional)
- Extended thinking support
- Source-list generation mode (`-gen-source`)
- **Multi-step automated workflow** (`-ai-steps`): expands prompt → decomposes into steps → executes each with user confirmation and git commits
- **Crash-resilient resume** (`-ai-steps -continue`): resumes an interrupted workflow from the last saved checkpoint
- **Structured commit messages**: `[category: feature_title]: step X/Y - step_title` for clear git history

## Memory System

The memory system provides Claude with persistent context across invocations:

- **Long-term memory** (`../.ai-code/memory/long-term.md`): Stores project architecture, key files, schema summaries, API routes, conventions. Lives in the parent project root at `.ai-code/memory/` so it's tracked in the master project's git history.
- **Short-term memory** (`./memory/short-term.md`): Stores current ai-steps workflow state (goal, progress, step list). Lives inside the ai-code directory since it's ephemeral.
- **Git history**: Last N commits with per-file numstat diffs, providing recent change context.

Memory is automatically injected into every Claude request and updated inline (no separate API calls).

## Workflows

### `-ai` — Standard Execution

Single-shot workflow: sends your prompt and source files to Claude, validates the response, and applies file edits to disk.

### `-gen-source` — Source List Generation

Asks Claude to recommend which files/directories are relevant for your prompt based on the project's directory tree. Useful for narrowing down the source list before a large refactoring.

### `-ai-steps` — Automated Multi-Step Pipeline

A fully automated workflow for complex tasks:

1. **Phase 1 — Prompt Expansion**: Generates a source list for the minimal prompt, then asks Claude to expand it into a comprehensive implementation specification.
2. **Phase 2 — Step Decomposition**: Generates a source list for the expanded prompt, then asks Claude to decompose it into ordered, atomic implementation steps (with per-step prompts, source lists, and category labels). Also generates a `feature_title` for the overall change.
3. **Phase 3 — Step Execution**: Iterates through each step:
   - Generates a fresh source list (files may have changed from previous steps).
   - Executes the step prompt against the source files.
   - Displays live progress: `Step X/Y` with completed/skipped/remaining counts and category/feature context.
   - Asks for user confirmation:
     - **Accept**: commits the changes with a structured message (`[category: feature]: step X/Y - title`) and proceeds.
     - **Retry**: reverts changes, optionally modifies the step prompt, and re-executes.
     - **Skip**: reverts changes and moves to the next step.
     - **Quit**: reverts changes and stops the workflow.

**Commit message format**: Each accepted step is committed with a structured message:
```
[database: User Preferences]: step 1/5 - Add preferences table
[api: User Preferences]: step 2/5 - Create CRUD endpoints
[frontend: User Preferences]: step 3/5 - Build settings page
```

**Requirements**: git must be available and the working tree must be clean.

### `-ai-steps -continue` — Resume Interrupted Workflow

If an `-ai-steps` run is interrupted (crash, connection loss, power failure, or manual quit), use `-continue` to resume:

```sh
python ai-code.py -ai-steps -continue
```

The workflow saves a checkpoint (`workflow-state.yaml`) after each phase and step. On resume:

- **Prompt validation**: verifies that the prompt hasn't changed since the original run (starts fresh if it has).
- **Phase skip**: phases that already completed (expansion, decomposition) are skipped — their results (including `feature_title`) are loaded from the state file.
- **Step skip**: steps that were already committed or skipped are bypassed.
- **Dirty tree recovery**: if uncommitted changes are found (from a crash mid-step), they are automatically reverted before resuming.

The state file is automatically removed once all steps have been processed.

## Token Usage Breakdown

Every API call displays a coloured ASCII bar chart showing the token composition:

```
┌──────────────────────────────────────────────────────────────────┐
│  TOKEN USAGE BREAKDOWN                                          │
│  Total Estimated: ~12,450 tokens                                │
│                                                                  │
│  System              ~2,000 tk  ████████░░░░░░░░░░░░░░░░░  16.1%│
│  Long-term Memory    ~1,200 tk  █████░░░░░░░░░░░░░░░░░░░░   9.6%│
│  Short-term Memory     ~300 tk  █░░░░░░░░░░░░░░░░░░░░░░░░   2.4%│
│  Git History           ~800 tk  ███░░░░░░░░░░░░░░░░░░░░░░   6.4%│
│  File Data           ~7,500 tk  ██████████████████████████  60.2%│
│  Prompt                ~650 tk  ███░░░░░░░░░░░░░░░░░░░░░░   5.2%│
└──────────────────────────────────────────────────────────────────┘
```

This helps identify which components are consuming the most context window space and optimize accordingly.

## Prerequisites

- Python 3.8+
- Anthropic API key for Claude
- Git (required for `-ai-steps`, advised for `-ai`)

## Installation

```sh
git clone https://github.com/gekigek99/ai-code.git
```

Or as a git submodule:

```sh
git submodule add https://github.com/gekigek99/ai-code.git
```

Install dependencies:

```sh
pip install anthropic pyyaml pymupdf pyperclip
```

## Configuration

Copy the example config and edit it:

```sh
cp ai-code-prompt.yaml.example ai-code-prompt.yaml
```

Edit `ai-code-prompt.yaml`:

```yaml
source: ["."]

exclude_patterns: [
  ".git*",
  "logs/",
  "__pycache__/"
]

tree_dirs: ["."]

prompt: |
  Refactor and improve the code.

system: |
  AI assistant for code tasks.

# Provider selection — "anthropic" or "openrouter".
PROVIDER: anthropic

anthropic:
  API_KEY: sk-ant-api03-YOUR-KEY-HERE
  MODEL: claude-sonnet-4-20250514

openrouter:
  API_KEY: sk-or-v1-YOUR-KEY-HERE
  MODEL: anthropic/claude-sonnet-4

# Provider-agnostic generation settings
MAX_TOKENS: 32000
MAX_TOKENS_THINK: 5000

WEBSEARCH: false
WEBSEARCH_MAX_RESULTS: 5

# Memory system configuration
MEMORY:
  ENABLED: true
  LONG_TERM_ENABLED: true       # Stored at ../.ai-code/memory/long-term.md
  SHORT_TERM_ENABLED: true      # Stored at ./memory/short-term.md
  GIT_HISTORY_ENABLED: true
  GIT_HISTORY_COMMITS: 30
  LONG_TERM_MAX_TOKENS: 2000
  SHORT_TERM_MAX_TOKENS: 1000
  AUTO_UPDATE: true
```

## Usage

| Command | Description |
|---|---|
| `python ai-code.py` | Dry run — gather sources, build prompt, export to logs |
| `python ai-code.py -ai` | Send prompt to Claude and apply returned edits |
| `python ai-code.py -ai -f` | Force — skip uncommitted git changes check |
| `python ai-code.py -last` | Re-apply last saved Claude response |
| `python ai-code.py -gen-source` | Ask Claude to generate a recommended source list |
| `python ai-code.py -ai-steps` | Run automated multi-step workflow |
| `python ai-code.py -ai-steps -continue` | Resume an interrupted multi-step workflow |
| `python ai-code.py -pdf` | Include PDF text content in AI context |
| `python ai-code.py -img path/to/image.png` | Attach image(s) to the prompt |
| `python ai-code.py -h` | Show full help with all flags |

Flags can be combined: `python ai-code.py -ai -f -pdf -img screenshot.png`

## Exclude Patterns

Add glob patterns in `exclude_patterns`, e.g.: `*.pyc`, `.env*`, `node_modules/`.

Multi-level patterns with `/` are supported: `src/generated/`.

## Tool Architecture

The codebase follows a three-layer architecture:

1. **Infrastructure** (`lib/`): Low-level modules for file I/O, API calls, validation, token tracking, etc.
2. **Tools** (`lib/tools/`): Reusable, stateless functions that compose infrastructure modules into well-defined operations (source generation, prompt execution, expansion, step decomposition, user confirmation).
3. **Workflows** (`lib/workflows/`): High-level orchestrators that compose tools into complete CLI workflows (`-ai`, `-gen-source`, `-ai-steps`).

Each tool returns a structured dict with at minimum a `status` key (`"ok"` or `"error"`), making error handling consistent across all workflows.

---

Remember to star the project on [GitHub](https://github.com/gekigek99/ai-code)!
