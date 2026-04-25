"""
lib.providers — LLM provider implementations + provider-agnostic dispatcher.

Each provider module exposes a ``prompt_<provider>(...)`` function that handles
API communication, streaming, and response capture.  All providers share the
same function signature and return-dict shape so callers can swap providers
with minimal code changes.

Available providers:
    claude     — Anthropic Claude (native API).
                 Function: ``prompt_claude``  (see ``providers/claude.py``)
                 Requires: ``anthropic`` package, ``anthropic.API_KEY`` in YAML.

    openrouter — OpenRouter unified gateway (OpenAI-compatible API).
                 Function: ``prompt_openrouter``  (see ``providers/openrouter.py``)
                 Requires: ``openai`` package, ``openrouter.API_KEY`` in YAML.
                 Provides access to hundreds of models (Claude, GPT, Gemini,
                 Llama, DeepSeek, etc.) through a single endpoint.

Unified return shape (all providers):
    {
        "status":           "ok" | "no_response" | "error",
        "data_response":    str,   # accumulated visible text
        "thinking_content": str,   # accumulated reasoning text (if any)
        "raw_data":         list,  # raw event/chunk captures for debugging
        "error":            str | None,
    }

Public API:
    prompt_llm(cfg, messages, *, stream=True, recv_path=None) -> dict
        Provider-agnostic dispatcher.  Reads ``cfg.provider`` and forwards
        to the appropriate ``prompt_<provider>`` function with all relevant
        Config fields mapped to the provider's keyword arguments.
        This is the function ALL tools and workflows should call — they
        must never import ``prompt_claude``/``prompt_openrouter`` directly.

Adding a new provider is a matter of:
    1. Drop a sibling module that exposes a ``prompt_<name>`` function with
       the same signature and return shape.
    2. Add the name to ``_ALLOWED_PROVIDERS`` in ``lib.config``.
    3. Add a branch in ``prompt_llm()`` below.
"""

from typing import Any, Dict, List, Optional


def prompt_llm(
    cfg: Any,
    messages: List[Dict[str, Any]],
    *,
    stream: bool = True,
    recv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Dispatch an LLM request to the provider configured in *cfg*.

    All Config fields needed by either provider are forwarded transparently,
    so callers only need to pass *cfg* + the conversation *messages* and
    optionally the streaming/recv settings.

    Provider modules are imported lazily so that users on the openrouter
    backend don't need the ``anthropic`` SDK installed (and vice versa).

    Parameters
    ----------
    cfg : Config
        Resolved configuration containing ``provider``, ``api_key``,
        ``model``, ``max_tokens``, ``max_tokens_think``, ``temperature``,
        ``websearch``, ``websearch_max_results``, ``system``.
    messages : list[dict]
        Conversation messages in Anthropic content-block format.  The
        OpenRouter provider transparently converts to OpenAI multipart.
    stream : bool
        Whether to use streaming mode.  Default True.
    recv_path : str, optional
        If set, streaming text chunks are appended to this file in real time.

    Returns
    -------
    dict
        Unified result dict (see module docstring).

    Raises
    ------
    ValueError
        If ``cfg.provider`` is not a known provider identifier.
    """
    provider = (getattr(cfg, "provider", "") or "").strip().lower()

    # All providers share this exact kwarg signature — defined once so a
    # signature change in one provider is caught immediately by the others.
    common_kwargs: Dict[str, Any] = dict(
        api_key=cfg.api_key,
        model=cfg.model,
        system=cfg.system,
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        websearch=cfg.websearch,
        websearch_max_results=cfg.websearch_max_results,
        thinking_budget=cfg.max_tokens_think,
        stream=stream,
        recv_path=recv_path,
    )

    if provider == "anthropic":
        # Lazy import: avoids importing the ``anthropic`` SDK when the user
        # is configured to use OpenRouter (and may not have it installed).
        from lib.providers.claude import prompt_claude
        return prompt_claude(**common_kwargs)

    if provider == "openrouter":
        # Lazy import: avoids importing the ``openai`` SDK when the user
        # is configured to use Anthropic (and may not have it installed).
        from lib.providers.openrouter import prompt_openrouter
        return prompt_openrouter(**common_kwargs)

    raise ValueError(
        f"Unknown provider {provider!r}. "
        f"Allowed: 'anthropic', 'openrouter'. "
        f"Set PROVIDER in ai-code-prompt.yaml to a supported value."
    )

