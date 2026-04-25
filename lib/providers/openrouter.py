"""
lib.providers.openrouter — OpenRouter API interaction.

Public API:
    prompt_openrouter(...) -> dict
        Send a request to OpenRouter with streaming support and return
        a result dict with keys: status, data_response, thinking_content,
        raw_data, error.

OpenRouter implements the OpenAI Chat Completions API specification, so
this module uses the official ``openai`` Python SDK pointed at the
OpenRouter base URL.  The function signature, return shape, and behaviour
intentionally mirror :func:`lib.providers.claude.prompt_claude` so the
two providers are interchangeable from the caller's perspective.

Key mappings between the unified API and OpenRouter:
  * ``system``               → first message with ``role="system"``
  * ``messages``              → converted from Anthropic-style content
                                blocks to OpenAI multipart format
                                (text + image_url with data URI).
  * ``thinking_budget``       → ``reasoning.max_tokens`` in extra_body
  * ``websearch``             → ``plugins=[{"id": "web", ...}]`` in extra_body
  * ``websearch_max_results`` → ``plugins[0].max_results``

All ``openai`` SDK imports are isolated here so additional providers
(Anthropic native, Gemini, etc.) remain trivially swappable.
"""

import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# OpenRouter's OpenAI-compatible endpoint.
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional headers used by OpenRouter for ranking/attribution on openrouter.ai.
# These are harmless if absent — they only affect public leaderboard stats.
_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/gekigek99/ai-code",
    "X-Title": "ai-code",
}


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def prompt_openrouter(
    client: Optional[OpenAI] = None,
    api_key: Optional[str] = None,
    model: str = "anthropic/claude-sonnet-4",
    system: str = "",
    messages: Optional[List[Dict[str, Any]]] = None,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    websearch: bool = False,
    websearch_max_results: int = 5,
    thinking_budget: int = 0,
    stream: bool = True,
    recv_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Call OpenRouter and return a structured result.

    Supports both streaming (``stream=True``) and synchronous
    (``stream=False``) modes.  Streaming prints chunks in real time and
    optionally appends them to *recv_path*.  Synchronous mode returns the
    full response at once — used by lightweight internal calls.

    Parameters
    ----------
    client : OpenAI, optional
        Pre-configured client.  If ``None``, one is created from *api_key*
        with ``base_url`` pointing to OpenRouter.
    api_key : str, optional
        Used only when *client* is not provided.
    model : str
        OpenRouter model slug (e.g. ``"anthropic/claude-sonnet-4"``,
        ``"openai/gpt-5"``, ``"deepseek/deepseek-v4-pro"``).
    system : str
        System prompt.  Prepended as a ``role="system"`` message because
        OpenAI-compatible APIs do not use a separate system parameter.
    messages : list[dict]
        Conversation messages.  Anthropic-style content blocks (e.g.
        ``{"type": "image", "source": {...}}``) are converted to OpenAI
        multipart format automatically.
    max_tokens : int
        Maximum output tokens.
    temperature : float
        Sampling temperature.
    websearch : bool
        Enable the OpenRouter web-search plugin via the ``plugins``
        request parameter.  When True, the plugin uses native search for
        providers that support it (Anthropic, OpenAI, Perplexity, xAI)
        and falls back to Exa for others.
    websearch_max_results : int
        Maximum web-search hits per request.
    thinking_budget : int
        If > 0, enables reasoning via the ``reasoning.max_tokens``
        extra_body parameter.  Forwarded to the underlying provider; only
        models that support reasoning will honour it.
    stream : bool
        Whether to use streaming.
    recv_path : str, optional
        If set, text chunks are appended to this file in real time.

    Returns
    -------
    dict with keys:
        status            – "ok" | "no_response" | "error"
        data_response     – accumulated visible text
        thinking_content  – accumulated reasoning text
        raw_data          – list of raw event captures
        error             – error string if status == "error", else None
    """
    if client is None:
        if not api_key:
            raise ValueError("Either `client` or `api_key` must be provided")
        client = OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=api_key,
            default_headers=_DEFAULT_HEADERS,
        )

    messages = messages or []

    # -- Convert Anthropic-style content blocks to OpenAI multipart format ---
    # The rest of the codebase builds messages with Claude's schema (image
    # blocks use {type: "image", source: {type: "base64", media_type, data}}).
    # OpenRouter expects OpenAI's schema (image_url with data URI), so we
    # transform here rather than forcing every caller to know about both.
    converted_messages = _convert_messages_to_openai(messages)

    # -- Prepend system prompt as first message ------------------------------
    # OpenAI-compatible APIs accept the system prompt as a regular message
    # with role="system" rather than a separate top-level parameter.
    if system:
        converted_messages = [{"role": "system", "content": system}] + converted_messages

    # -- Build extra_body for OpenRouter-specific features -------------------
    # extra_body fields are passed through to OpenRouter unchanged. Anything
    # not part of the OpenAI spec (plugins, reasoning) belongs here.
    extra_body: Dict[str, Any] = {}

    if websearch:
        # The OpenRouter web plugin: native search for Anthropic/OpenAI/
        # Perplexity/xAI, Exa fallback for others.  Standardised citation
        # output via choice.message.annotations[].url_citation.
        extra_body["plugins"] = [{
            "id": "web",
            "max_results": int(websearch_max_results),
        }]

    if thinking_budget and int(thinking_budget) > 0:
        # OpenRouter forwards reasoning config to providers that support it
        # (e.g. Anthropic extended thinking, OpenAI o-series, DeepSeek R1).
        # Models without reasoning support silently ignore this field.
        extra_body["reasoning"] = {
            "max_tokens": int(thinking_budget),
        }

    # -- Clear recv file if provided -----------------------------------------
    if recv_path:
        try:
            with open(recv_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            print(f"[warn] Could not clear recv file {recv_path}: {e}")

    # -- Build common API kwargs ---------------------------------------------
    api_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": converted_messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if extra_body:
        api_kwargs["extra_body"] = extra_body

    # -- Dispatch to streaming or synchronous handler ------------------------
    if stream:
        return _handle_streaming(client, api_kwargs, recv_path)
    else:
        return _handle_synchronous(client, api_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Message format conversion (Anthropic blocks → OpenAI multipart)
# ══════════════════════════════════════════════════════════════════════════════

def _convert_messages_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert Anthropic-style content blocks to OpenAI multipart format.

    Anthropic format:
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "base64",
                                          "media_type": "image/png",
                                          "data": "..."}}
        ]}

    OpenAI/OpenRouter format:
        {"role": "user", "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]}

    String content is passed through unchanged.  Unknown block types are
    silently dropped to prevent the API from rejecting the entire request
    over a single bad block.
    """
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Plain string content — no transformation needed.
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        # List of content blocks — transform each block to OpenAI shape.
        if not isinstance(content, list):
            # Defensive: coerce unknown types to string.
            converted.append({"role": role, "content": str(content)})
            continue

        new_content: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            btype = block.get("type")

            if btype == "text":
                new_content.append({
                    "type": "text",
                    "text": block.get("text", ""),
                })

            elif btype == "image":
                # Anthropic image block → OpenAI image_url with data URI.
                source = block.get("source", {}) or {}
                src_type = source.get("type", "base64")
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")

                if src_type == "base64":
                    url = f"data:{media_type};base64,{data}"
                else:
                    # URL-type sources are passed through as-is.
                    url = data

                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })

            elif btype == "image_url":
                # Already OpenAI format — pass through.
                new_content.append(block)

            # Other block types (tool_use, tool_result, etc.) are dropped
            # silently. ai-code does not currently send these to providers,
            # so a stricter error here would be premature.

        converted.append({"role": role, "content": new_content})

    return converted


# ══════════════════════════════════════════════════════════════════════════════
# Synchronous (non-streaming) handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_synchronous(
    client: OpenAI,
    api_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle a non-streaming chat-completions call.

    OpenRouter's non-streaming response is a complete ``ChatCompletion``
    object whose ``choices[0].message`` contains the final text in
    ``content`` and (when reasoning was enabled) the thinking text in
    ``reasoning``.  Web-search citations live in ``message.annotations``.
    """
    try:
        response = client.chat.completions.create(**api_kwargs, stream=False)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to start request: {type(e).__name__}: {e}",
            "data_response": "",
            "thinking_content": "",
            "raw_data": [],
        }

    data_response = ""
    thinking_content = ""
    raw_data: List[Dict[str, Any]] = []

    try:
        # Capture the raw response for debugging — model_dump_json gives
        # cleaner output than str() on Pydantic models.
        try:
            raw_repr = response.model_dump_json()
        except Exception:
            raw_repr = str(response)
        raw_data.append({"type": "completion", "event": raw_repr})

        if response.choices:
            msg = response.choices[0].message
            data_response = getattr(msg, "content", "") or ""
            # Some OpenRouter providers expose reasoning trace via the
            # non-standard ``reasoning`` field on the message.
            thinking_content = getattr(msg, "reasoning", "") or ""

            # Web-search citations are returned in message.annotations[]
            # — capture them in raw_data for later inspection.
            annotations = getattr(msg, "annotations", None) or []
            if annotations:
                raw_data.append({
                    "type": "annotations",
                    "event": str(annotations),
                })

    except Exception as e:
        return {
            "status": "error",
            "error": f"Response parsing failed: {type(e).__name__}: {e}",
            "data_response": data_response,
            "thinking_content": thinking_content,
            "raw_data": raw_data,
        }

    status = "ok" if data_response.strip() else "no_response"
    return {
        "status": status,
        "data_response": data_response,
        "thinking_content": thinking_content,
        "raw_data": raw_data,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Streaming handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_streaming(
    client: OpenAI,
    api_kwargs: Dict[str, Any],
    recv_path: Optional[str],
) -> Dict[str, Any]:
    """Handle a streaming chat-completions call with real-time chunk output.

    OpenRouter streaming follows the OpenAI SSE format:
      * Each chunk's ``choices[0].delta.content`` carries text deltas.
      * ``choices[0].delta.reasoning`` carries thinking deltas (when the
        model supports it and reasoning was requested via extra_body).
      * ``choices[0].delta.annotations[]`` carries web-search citations
        as they accumulate.
      * ``choices[0].finish_reason`` becomes non-None on the final chunk.

    Citations are de-duplicated by URL because OpenRouter sometimes emits
    the same annotation across multiple delta chunks during streaming.
    """
    try:
        response_iter = client.chat.completions.create(**api_kwargs, stream=True)
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to start request: {type(e).__name__}: {e}",
            "data_response": "",
            "thinking_content": "",
            "raw_data": [],
        }

    data_response = ""
    thinking_content = ""
    raw_data: List[Dict[str, Any]] = []

    # -- Web search citation tracking ---------------------------------------
    # Streaming chunks may repeat the same annotation across deltas; track
    # which URLs we've already printed so the terminal output stays clean.
    seen_citation_urls: set = set()
    citations_printed = 0

    # Whether we've printed the "Claude is thinking..." header so that
    # repeated reasoning deltas don't print the header on every chunk.
    thinking_header_printed = False

    try:
        for chunk in response_iter:
            # -- Capture raw chunk for debugging artifacts -------------------
            try:
                raw_repr = chunk.model_dump_json() if hasattr(chunk, "model_dump_json") else str(chunk)
            except Exception:
                raw_repr = repr(chunk)
            raw_data.append({
                "type": "chunk",
                "event": raw_repr,
            })

            # Defensive: some chunks (e.g. ping events) have no choices.
            if not getattr(chunk, "choices", None):
                continue

            choice = chunk.choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # -- Visible text delta -----------------------------------------
            text = getattr(delta, "content", None)
            if text:
                print(text, end="", flush=True)
                data_response += text
                if recv_path:
                    try:
                        with open(recv_path, "a", encoding="utf-8") as f:
                            f.write(text)
                    except Exception as e:
                        print(f"[warn] Failed to append to recv file: {e}")

            # -- Reasoning (thinking) delta ---------------------------------
            # OpenRouter exposes provider reasoning via delta.reasoning when
            # reasoning was requested.  Print a header once on first chunk so
            # the user knows the model is thinking, then accumulate silently.
            reasoning = getattr(delta, "reasoning", None)
            if reasoning:
                if not thinking_header_printed:
                    print("\n[Model is reasoning...]")
                    thinking_header_printed = True
                thinking_content += reasoning

            # -- Web search annotations -------------------------------------
            # OpenRouter normalises citations across providers into the OpenAI
            # annotation schema:
            #   {"type": "url_citation",
            #    "url_citation": {"url": ..., "title": ..., "content": ...,
            #                     "start_index": ..., "end_index": ...}}
            annotations = getattr(delta, "annotations", None) or []
            for ann in annotations:
                ann_type = _safe_get(ann, "type")
                if ann_type != "url_citation":
                    continue

                citation = _safe_get(ann, "url_citation")
                if citation is None:
                    continue

                url = _safe_get(citation, "url") or ""
                if not url or url in seen_citation_urls:
                    continue

                seen_citation_urls.add(url)
                citations_printed += 1
                _print_websearch_citation(citation, citations_printed)

    except Exception as e:
        return {
            "status": "error",
            "error": f"Streaming/iteration failed: {type(e).__name__}: {e}",
            "data_response": data_response,
            "thinking_content": thinking_content,
            "raw_data": raw_data,
        }

    # -- Web search summary --------------------------------------------------
    if citations_printed > 0:
        print(f"\n\033[36m{'═' * 50}\033[0m")
        print(f"\033[36m[WEBSEARCH SUMMARY] "
              f"Citations used: {citations_printed}\033[0m")
        print(f"\033[36m{'═' * 50}\033[0m\n")

    status = "ok" if data_response.strip() else "no_response"

    return {
        "status": status,
        "data_response": data_response,
        "thinking_content": thinking_content,
        "raw_data": raw_data,
        "error": None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_get(obj: Any, key: str) -> Any:
    """Resilient getter that works for both Pydantic models and plain dicts.

    The OpenAI SDK returns Pydantic objects for most fields, but extension
    fields (like OpenRouter's annotations) are sometimes deserialised as
    plain dicts depending on SDK version.  This helper accepts both.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _print_websearch_citation(citation: Any, index: int) -> None:
    """Pretty-print a web-search citation that appears inline in the response.

    Mirrors :func:`lib.providers.claude._print_websearch_citation` so that
    terminal output is visually consistent across providers.
    """
    title = _safe_get(citation, "title") or "(no title)"
    url = _safe_get(citation, "url") or "(no url)"
    cited_text = _safe_get(citation, "content") or _safe_get(citation, "cited_text") or ""

    # Truncate long cited text for readability in terminal output.
    if isinstance(cited_text, str) and len(cited_text) > 200:
        cited_text = cited_text[:200] + "..."

    print(f"\n\033[36m  ╭- CITATION #{index} -------------------\033[0m")
    print(f"\033[36m  │ Title : {title}\033[0m")
    print(f"\033[36m  │ URL   : {url}\033[0m")
    if cited_text:
        print(f"\033[36m  │ Text  : {cited_text}\033[0m")
    print(f"\033[36m  ╰-------------------------------------\033[0m\n")

