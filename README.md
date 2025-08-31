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

Create ai-claude-prompt.yaml from the ai-claude-prompt.yaml.example:

## Edit ai-claude-prompt.yaml

```yaml
source: [
  "."
]

exclude_patterns: [
  "ai-claude/",
  ".git*",
  "__pycache__/"
]

tree_dirs: [
  ".",
]

prompt: |
  "Refactor and improve the code."

system: |
  "AI assistant for code tasks"

ANTHROPIC:
  API_KEY: sk-ant-api03-EXAMPLE-KpU7vgAA
  # key name: test_api

  MAX_TOKENS: 32000
  MAX_TOKENS_THINK: 3000

  CLAUDE_MODEL: claude-sonnet-4-20250514
  # claude-sonnet-4-20250514
  # claude-opus-4-20250514
  # claude-opus-4-1-20250805
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

5. Share pdfs text content
`python ai-claude.py -pdf`

5. Share images
`python ai-claude.py -img path/to/image1.ext -img path/to/image2.ext`

## Exclude Patterns

Add glob patterns in exclude_patterns, e.g.: *.pyc, .env*, node_modules/.

---

Remember to star the project on [GitHub](https://github.com/gekigek99/ai-code)!
