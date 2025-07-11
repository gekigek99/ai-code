# **AI Code**

A simple Python tool to run Anthropic Claude on your codebase, generate or update files, and manage outputs locally.

---

## Features

- Scans source folders with include/exclude patterns
- Skips binaries automatically
- Shows a directory tree of your project
- Streams prompts to Claude and captures responses
- Writes/updates/deletes files based on markers

## Prerequisites

- Python 3.8+
- Anthropic API key for Claude
- Git (advised but optional)

## Installation

```sh
git clone https://github.com/gekigek99/ai-code.git
```

or better as a git repository submodule:

```sh
git submodule add https://github.com/gekigek99/ai-code.git
```

## Configuration

Create a .env file:

```sh
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_CLAUDE_MODEL=claude-sonnet-4-20250514
ANTHROPIC_MAX_TOKENS=20000
ANTHROPIC_MAX_TOKENS_THINK=5000
```

## Edit ai-claude-prompt.yaml

```yaml
source_dirs:
  - .

exclude_patterns:
  - ai-claude
  - .git/
  - __pycache__/

system: |
  "AI assistant for code tasks"

prompt: |
  "Refactor and improve the code."
```

## Usage

1. Generate prompt only for prompt analysis
`python ai-claude.py`

2. Send to Claude and apply results
`python ai-claude.py -ai`

3. Force even with git changes
`python ai-claude.py -ai -f`

4. Reapply last response
`python ai-claude.py -readlast`

## Exclude Patterns

Add glob patterns in exclude_patterns, e.g.: *.pyc, .env*, node_modules/.

## Outputs

AI outputs use markers like:

+++ path/to/file.py +++

with optional [UPDATE] or [DELETE] tags.
