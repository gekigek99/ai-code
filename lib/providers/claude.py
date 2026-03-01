"""
lib.providers.claude — Anthropic Claude API interaction.

Public API:
    prompt_claude(...) -> dict
        Send a request to Claude with streaming support and return
        a result dict with keys: status, data_response, thinking_content,
        raw_data, error.

All Anthropic SDK imports are isolated here, making it straightforward to
add alternative providers (OpenAI, Gemini, etc.) as sibling modules.
"""

import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic


def prompt_claude(
    client: Optional[Anthropic] = None,
    api_key: Optional[str] = None,
    model: str = "claude-opus-4-1-20250805",
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
    """Call Anthropic Claude and return a structured result.

    Supports both streaming (``stream=True``) and synchronous
    (``stream=False``) modes.  Streaming prints chunks in real time and
    optionally appends them to *recv_path*.  Synchronous mode returns the
    full response at once — used by lightweight internal calls such as
    memory updates where real-time output is unnecessary.

    Parameters
    ----------
    client : Anthropic, optional
        Pre-configured client.  If ``None``, one is created from *api_key*.
    api_key : str, optional
        Used only when *client* is not provided.
    model : str
        Claude model identifier.
    system : str
        System prompt.
    messages : list[dict]
        Conversation messages.
    max_tokens : int
        Maximum output tokens.
    temperature : float
        Sampling temperature.
    websearch : bool
        Enable web-search tool.
    websearch_max_results : int
        Maximum web-search hits.
    thinking_budget : int
        If > 0, enable extended thinking with this token budget.
    stream : bool
        Whether to use streaming.
    recv_path : str, optional
        If set, text chunks are appended to this file in real time.

    Returns
    -------
    dict with keys:
        status            – "ok" | "no_response" | "error"
        data_response     – accumulated visible text
        thinking_content  – accumulated thinking text
        raw_data          – list of raw event captures
        error             – error string if status == "error", else None
    """
    if client is None:
        if not api_key:
            raise ValueError("Either `client` or `api_key` must be provided")
        client = Anthropic(api_key=api_key)

    messages = messages or []

    # ── Build tools list ─────────────────────────────────────────────────────
    # Web search tool is only added when explicitly enabled.  The status is
    # logged once per call (not "WEBSEARCH active!" on every call, which was
    # noisy in multi-step workflows).
    tools: list = []
    if websearch:
        tools = [{
            "name": "web_search",
            "type": "web_search_20250305",
            "max_uses": int(websearch_max_results),
        }]

    # Clear recv file if provided
    if recv_path:
        try:
            with open(recv_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            print(f"[warn] Could not clear recv file {recv_path}: {e}")

    # ── Build common API kwargs ──────────────────────────────────────────────
    api_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "system": system,
        "messages": messages,
        "thinking": (
            {"type": "enabled", "budget_tokens": int(thinking_budget)}
            if thinking_budget and int(thinking_budget) > 0
            else {"type": "disabled"}
        ),
    }

    # Only include tools key when tools are present — some API configurations
    # may behave differently with an empty tools list vs no tools key at all.
    if tools:
        api_kwargs["tools"] = tools

    # ── Dispatch to streaming or synchronous handler ─────────────────────────
    if stream:
        return _handle_streaming(client, api_kwargs, recv_path)
    else:
        return _handle_synchronous(client, api_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Synchronous (non-streaming) handler
# ══════════════════════════════════════════════════════════════════════════════

def _handle_synchronous(
    client: Anthropic,
    api_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Handle a non-streaming API call.

    When ``stream=False``, ``client.messages.create()`` returns a complete
    ``Message`` object.  We extract text and thinking content directly from
    the response's ``content`` blocks rather than iterating over events.
    """
    try:
        response = client.messages.create(**api_kwargs, stream=False)
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
        # Capture the raw response for debugging
        raw_data.append({
            "type": "message",
            "event": str(response),
        })

        # Iterate over the content blocks in the completed Message object.
        # Each block has a ``type`` attribute: "text", "thinking", "tool_use", etc.
        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", "") or ""
                data_response += text

            elif block_type == "thinking":
                thinking_content += getattr(block, "thinking", "") or ""

            # Web-search citation blocks and tool_use blocks are captured
            # in raw_data but not parsed further for sync calls, which are
            # only used for lightweight internal tasks (e.g. memory update).
            else:
                raw_data.append({
                    "type": f"content_block_{block_type}",
                    "event": str(block),
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
    client: Anthropic,
    api_kwargs: Dict[str, Any],
    recv_path: Optional[str],
) -> Dict[str, Any]:
    """Handle a streaming API call with real-time chunk output.

    Properly handles all Anthropic streaming event types including:
    - text_delta: regular text output
    - thinking_delta: extended thinking content
    - input_json_delta: tool input streaming (e.g. web search query)
    - citations_delta: web search citation references
    - content_block_start with server_tool_use: tool invocation start
    - content_block_start with web_search_tool_result: search results
    """
    try:
        response_iter = client.messages.create(**api_kwargs, stream=True)
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
        for event in response_iter:
            # Capture raw event (best-effort)
            try:
                raw_data.append({
                    "type": getattr(event, "type", "(unknown)"),
                    "event": str(event),
                })
            except Exception:
                raw_data.append({
                    "type": getattr(event, "type", "(unknown)"),
                    "event": repr(event),
                })

            try:
                if event.type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if delta is None:
                        continue
                    d_type = getattr(delta, "type", None)

                    if d_type == "thinking_delta":
                        thinking_content += getattr(delta, "thinking", "")

                    elif d_type == "text_delta":
                        chunk = getattr(delta, "text", "") or ""
                        print(chunk, end="", flush=True)
                        data_response += chunk
                        if recv_path:
                            try:
                                with open(recv_path, "a", encoding="utf-8") as f:
                                    f.write(chunk)
                            except Exception as e:
                                print(f"[warn] Failed to append to recv file: {e}")

                    elif d_type == "input_json_delta":
                        # Tool input being streamed (e.g. the web search query).
                        # Print it so the user can see what Claude is searching for.
                        partial = getattr(delta, "partial_json", "") or ""
                        if partial:
                            print(partial, end="", flush=True)

                    elif d_type == "citations_delta":
                        # Web search citation — a single citation object per delta
                        # event with url, title, cited_text, etc.  These appear
                        # inline as Claude writes text that references search results.
                        citation = getattr(delta, "citation", None)
                        if citation is not None:
                            _print_websearch_citation({
                                "title": getattr(citation, "title", "") or "",
                                "url": getattr(citation, "url", "") or "",
                                "cited_text": getattr(citation, "cited_text", "") or "",
                                "page_age": getattr(citation, "page_age", None),
                            })

                    else:
                        # Capture unhandled delta subtypes for debugging but don't
                        # spam stdout — they may be SDK additions we don't know yet.
                        pass

                elif event.type == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    if cb is None:
                        continue

                    cb_type = getattr(cb, "type", None)

                    if cb_type == "server_tool_use":
                        # Claude is invoking a server-side tool (web search).
                        # Log the tool name so the user sees the search happening.
                        tool_name = getattr(cb, "name", "unknown_tool")
                        tool_id = getattr(cb, "id", "")
                        print(f"\n{'─' * 40}")
                        print(f"[WEBSEARCH] Claude invoking tool: {tool_name} (id={tool_id})")
                        print(f"{'─' * 40}")

                    elif cb_type == "web_search_tool_result":
                        # Server-side web search results.  The content attribute
                        # is a list of result objects.  Each has type, url, title,
                        # encrypted_content, and optionally page_age.  The actual
                        # readable cited text comes later via citations_delta —
                        # here we log the URLs/titles for visibility.
                        content_items = getattr(cb, "content", None) or []
                        if isinstance(content_items, (list, tuple)):
                            result_count = len(content_items)
                            print(f"\n{'─' * 40}")
                            print(f"[WEBSEARCH] Search returned {result_count} result(s):")
                            for idx, item in enumerate(content_items):
                                item_type = getattr(item, "type", "")

                                if item_type == "web_search_result":
                                    title = getattr(item, "title", "") or "(no title)"
                                    url = getattr(item, "url", "") or "(no url)"
                                    page_age = getattr(item, "page_age", None)
                                    age_str = f" | age: {page_age}" if page_age else ""
                                    print(f"  [{idx + 1}] {title}")
                                    print(f"      {url}{age_str}")

                                elif item_type == "web_search_result_error":
                                    error_msg = getattr(item, "error_message", "Unknown search error")
                                    print(f"  [{idx + 1}] ERROR: {error_msg}")

                                else:
                                    # Unknown result item type — log for debugging
                                    print(f"  [{idx + 1}] Unknown type: {item_type}")

                            print(f"{'─' * 40}\n")
                        else:
                            print(f"\n[WEBSEARCH] Result block received (non-list content: {type(content_items).__name__})")

                    elif cb_type == "text":
                        # Regular text content block starting — no action needed.
                        # Text content arrives via text_delta events.
                        pass

                    elif cb_type == "thinking":
                        # Thinking block starting — the actual content arrives
                        # via thinking_delta events.
                        print("\n[Claude is thinking...]")

                    else:
                        # Unknown content block type — capture for debugging.
                        # Could be a new SDK type we haven't handled yet.
                        pass

                elif event.type == "message_stop":
                    break

                else:
                    # Ignore other event types silently (message_start,
                    # content_block_stop, message_delta, ping, etc.)
                    pass

            except AttributeError as e:
                print(
                    f"[error] AttributeError while processing event "
                    f"'{getattr(event, 'type', str(event))}': {e}"
                )
                continue

    except Exception as e:
        return {
            "status": "error",
            "error": f"Streaming/iteration failed: {type(e).__name__}: {e}",
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


def _print_websearch_citation(entry: Dict[str, Any]) -> None:
    """Pretty-print a web-search citation that appears inline in Claude's text.

    Citations are emitted via citations_delta events as Claude writes text
    that references web search results.  Each citation links a span of
    Claude's output to a specific source URL and quoted text.
    """
    title = entry.get("title") or "(no title)"
    url = entry.get("url") or "(no url)"
    cited_text = entry.get("cited_text") or ""
    page_age = entry.get("page_age")

    # Truncate long cited text for readability in terminal
    if len(cited_text) > 200:
        cited_text = cited_text[:200] + "..."

    print(f"\n  ╭─ CITATION ─────────────────────────")
    print(f"  │ Title : {title}")
    print(f"  │ URL   : {url}")
    if cited_text:
        print(f"  │ Text  : {cited_text}")
    if page_age:
        print(f"  │ Age   : {page_age}")
    print(f"  ╰─────────────────────────────────────\n")
