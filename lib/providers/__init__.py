"""
lib.providers — LLM provider implementations.

Each provider module exposes a ``prompt_<provider>(...)`` function that handles
API communication, streaming, and response capture.  All providers share the
same function signature and return-dict shape so callers can swap providers
with minimal code changes.

Available providers:
    claude     — Anthropic Claude (native API).
                 Function: ``prompt_claude``  (see ``providers/claude.py``)
                 Requires: ``anthropic`` package, ``ANTHROPIC.API_KEY``.

    openrouter — OpenRouter unified gateway (OpenAI-compatible API).
                 Function: ``prompt_openrouter``  (see ``providers/openrouter.py``)
                 Requires: ``openai`` package, ``OPENROUTER.API_KEY``.
                 Provides access to hundreds of models (Claude, GPT, Gemini,
                 Llama, DeepSeek, etc.) through a single endpoint.
                 Web search via the ``plugins=[{\"id\": \"web\"}]`` parameter;
                 reasoning/thinking via the ``reasoning`` parameter.

Unified return shape (all providers):
    {
        "status":           "ok" | "no_response" | "error",
        "data_response":    str,   # accumulated visible text
        "thinking_content": str,   # accumulated reasoning text (if any)
        "raw_data":         list,  # raw event/chunk captures for debugging
        "error":            str | None,
    }

Adding a new provider is a matter of dropping a sibling module that exposes
a ``prompt_<name>`` function with the same signature and return shape.
"""

