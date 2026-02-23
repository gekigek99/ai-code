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
    """Call Anthropic Claude with streaming and return a structured result.

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

    tools: list = []
    if websearch:
        print("WEBSEARCH active!")
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

    # ── Start the API request ────────────────────────────────────────────────
    try:
        response_iter = client.messages.create(
            model=model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system,
            messages=messages,
            tools=tools if tools else [],
            thinking=(
                {"type": "enabled", "budget_tokens": int(thinking_budget)}
                if thinking_budget and int(thinking_budget) > 0
                else {"type": "disabled"}
            ),
            stream=bool(stream),
        )
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

    # ── Stream processing ────────────────────────────────────────────────────
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

                    elif d_type == "citations_delta":
                        citation_obj = getattr(delta, "citation", None)
                        citation_list = (
                            citation_obj if isinstance(citation_obj, (list, tuple))
                            else [citation_obj] if citation_obj is not None
                            else []
                        )
                        for cit in citation_list:
                            try:
                                _print_websearch_entry({
                                    "title": getattr(cit, "title", "") or "",
                                    "url": getattr(cit, "url", "") or "",
                                    "citation_text": getattr(cit, "cited_text", "") or "",
                                    "page_age": getattr(cit, "page_age", None),
                                    "source": "citation_delta",
                                })
                            except Exception as e:
                                print(f"[debug] Failed to parse citation delta: {e}")

                    else:
                        print(f"\n[debug] Unhandled content_block_delta subtype: {d_type}")

                elif event.type == "content_block_start":
                    cb = getattr(event, "content_block", None)
                    content_items = (
                        getattr(cb, "content", None)
                        or getattr(cb, "results", None)
                        or getattr(cb, "content_items", None)
                    )
                    if isinstance(content_items, (list, tuple)):
                        for idx, item in enumerate(content_items):
                            try:
                                _print_websearch_entry({
                                    "title": getattr(item, "title", ""),
                                    "url": getattr(item, "url", ""),
                                    "citation_text": (
                                        getattr(item, "snippet", "")
                                        or getattr(item, "excerpt", "")
                                        or getattr(item, "cited_text", "")
                                        or ""
                                    ),
                                    "page_age": getattr(item, "page_age", None),
                                    "source": "web_search_result",
                                    "result_index": idx,
                                })
                            except Exception as e:
                                print(f"[debug] Failed to parse web_search result item: {e}")
                    else:
                        try:
                            title = getattr(cb, "title", "") or ""
                            url = getattr(cb, "url", "") or ""
                            snippet = getattr(cb, "snippet", "") or ""
                            page_age = getattr(cb, "page_age", None)
                            if any([title, url, snippet, page_age]):
                                _print_websearch_entry({
                                    "title": title,
                                    "url": url,
                                    "citation_text": snippet,
                                    "page_age": page_age,
                                    "source": "web_search_result_unstructured",
                                })
                        except Exception:
                            print("\n[debug] Unhandled content_block_start structure")

                elif event.type == "message_stop":
                    break

                elif event.type == "thinking_block_start":
                    print("\n[Claude is thinking...]")

                else:
                    # Ignore unhandled event types silently
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


def _print_websearch_entry(entry: Dict[str, Any]) -> None:
    """Pretty-print a web-search result entry to stdout."""
    print("\n--- WEBSEARCH HIT ---")
    print(f"Title       : {entry.get('title') or '(title not found)'}")
    print(f"URL         : {entry.get('url') or '(no url)'}")
    print(f"Citation    : {entry.get('citation_text') or entry.get('cited_text') or '(citation text not found)'}")
    print(f"Page age    : {entry.get('page_age') or '(page age not found)'}")
    print(f"Source      : {entry.get('source') or '(source unknown)'}")
    idx = entry.get("result_index")
    if idx is not None:
        print(f"[result ID {idx}]")
    print("---------------------\n")
