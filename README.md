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
    ├── git.py                   # Git utilities (uncommitted changes check)
    ├── export.py                # Markdown export, prompt logging
    ├── validation.py            # Response validation, block_pattern regex
    ├── apply.py                 # File application logic (write/move/delete)
    ├── prompt_builder.py        # Message/content assembly for AI calls
    └── providers/
        ├── __init__.py          # Provider package init
        └── claude.py            # Claude-specific: prompt_claude(), streaming, websearch
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
        └── gen-source-rawdata.md
```

## Features

- Scans source folders with configurable include/exclude patterns
- Skips binary files automatically
- Shows a directory tree of your project with token estimates
- Streams prompts to Claude and captures responses in real time
- Writes, updates, moves, and deletes files based on `+++++` markers
- Validates response structure before applying changes
- PDF text extraction support
- Image attachment support (JPEG, PNG, GIF, WebP)
- Web search integration (optional)
- Extended thinking support
- Source-list generation mode (`-gen-source`)

## Prerequisites

- Python 3.8+
- Anthropic API key for Claude
- Git (advised but optional — used for safety checks)

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

ANTHROPIC:
  API_KEY: sk-ant-api03-YOUR-KEY-HERE
  MAX_TOKENS: 32000
  MAX_TOKENS_THINK: 5000
  CLAUDE_MODEL: claude-sonnet-4-20250514

WEBSEARCH: false
WEBSEARCH_MAX_RESULTS: 5
```

## Usage

| Command | Description |
|---|---|
| `python ai-code.py` | Dry run — gather sources, build prompt, export to logs |
| `python ai-code.py -ai` | Send prompt to Claude and apply returned edits |
| `python ai-code.py -ai -f` | Force — skip uncommitted git changes check |
| `python ai-code.py -last` | Re-apply last saved Claude response |
| `python ai-code.py -gen-source` | Ask Claude to generate a recommended source list |
| `python ai-code.py -pdf` | Include PDF text content in AI context |
| `python ai-code.py -img path/to/image.png` | Attach image(s) to the prompt |
| `python ai-code.py -h` | Show full help with all flags |

Flags can be combined: `python ai-code.py -ai -f -pdf -img screenshot.png`

## Exclude Patterns

Add glob patterns in `exclude_patterns`, e.g.: `*.pyc`, `.env*`, `node_modules/`.

Multi-level patterns with `/` are supported: `src/generated/`.

---

Remember to star the project on [GitHub](https://github.com/gekigek99/ai-code)!
