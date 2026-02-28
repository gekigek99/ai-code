## Architecture
- CLI-driven AI code generation tool (`ai-code.py` entrypoint)
- Reads project config from `ai-code-prompt.yaml` (gitignored, contains API keys)
- Three main workflows: `workflow_ai` (single prompt), `workflow_ai_steps` (multi-step), `workflow_gen_source` (source generation)
- Provider abstraction layer (`lib/providers/`) — currently Claude (`claude.py`)
- Tool system (`lib/tools/`) for modular prompt operations
- Memory system (`lib/memory.py`) for cross-session LLM context persistence
- Memory dir: `memory/long-term.md` (long-term, git-tracked), `memory/short-term.md` (ephemeral, gitignored)

## Key Files & Their Purpose
- `ai-code.py` — CLI entrypoint
- `lib/cli.py` — argument parsing, command dispatch
- `lib/config.py` — `Config` class, loads `ai-code-prompt.yaml`
- `lib/memory.py` — memory load/save/build/update (long-term & short-term)
- `lib/prompt_builder.py` — assembles LLM prompts with context, memory, source
- `lib/providers/__init__.py` — provider registry/abstraction
- `lib/providers/claude.py` — Anthropic Claude API integration
- `lib/tools/__init__.py` — tool registry
- `lib/tools/tool_prompt_execute.py` — execute a prompt against LLM
- `lib/tools/tool_prompt_expand.py` — expand/elaborate prompts
- `lib/tools/tool_prompt_stepize.py` — break prompt into sequential steps
- `lib/tools/tool_source_generate.py` — generate source code from LLM output
- `lib/tools/tool_user_confirm.py` — interactive user confirmation
- `lib/workflows/workflow_ai.py` — single-shot AI prompt workflow
- `lib/workflows/workflow_ai_steps.py` — multi-step AI workflow (reads source between steps)
- `lib/workflows/workflow_gen_source.py` — source file generation workflow
- `lib/apply.py` — apply generated code changes to filesystem
- `lib/export.py` — export prompt/response artifacts
- `lib/files.py` — file reading, glob resolution, source collection
- `lib/git.py` — git integration (recent commits, diff)
- `lib/images.py` — image handling for multimodal prompts
- `lib/pdf.py` — PDF text extraction for prompt context
- `lib/tree.py` — directory tree generation for context
- `lib/validation.py` — config and input validation
- `lib/utils.py` — shared helpers (`warn`, etc.)
- `ai-code-prompt.yaml.example` — example config template

## Key Functions/Variables
- `lib/memory.py`:
  - `load_long_term_memory(memory_dir) -> str`
  - `save_long_term_memory(memory_dir, content)`
  - `load_short_term_memory(memory_dir) -> str`
  - `save_short_term_memory(memory_dir, content)`
  - `clear_short_term_memory(memory_dir)`
  - `build_memory_block(cfg, include_short_term=False) -> str`
  - `build_memory_update_prompt(response_text, existing_memory, source_files_summary) -> str`
  - `update_long_term_memory(cfg, response_text, source_files_summary) -> bool`
  - `update_long_term_memory_from_source(cfg, files_to_ai) -> bool` — scans source files pre-step to refresh memory
- `lib/config.py`: `Config` class — central config object passed everywhere
- `_LONG_TERM_FILENAME = "long-term.md"`, `_SHORT_TERM_FILENAME = "short-term.md"`
- `_RESPONSE_TRUNCATION_CHARS = 3000`

## Conventions & Patterns
- All public functions use `cfg: Config` as primary context carrier
- Memory functions never raise — return `False` on failure, use `warn()` for errors
- Short-term memory is gitignored; long-term memory is git-tracked
- Config file (`ai-code-prompt.yaml`) is gitignored (contains API keys)
- `ai-code-prompt.yaml.example` is tracked as a template
- Source files use `+++++` delimited blocks for file edits in LLM output
- Tools are registered in `lib/tools/__init__.py`
- Workflows orchestrate tools in sequence
- `workflow_ai_steps` calls `update_long_term_memory_from_source()` before each step execution to keep memory current with codebase state
- Provider abstraction allows swapping LLM backends
- Truncation limits control prompt size for memory update calls