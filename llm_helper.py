"""Shared LLM helper for calling Groq and parsing JSON responses safely."""

import json
import re
from typing import Any

MODEL_NAME = "openai/gpt-oss-120b"
RETRY_PROMPT = "Return valid JSON only, no explanation"

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_START_RE = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_END_RE = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove optional <think> blocks and outer markdown code fences."""
    cleaned = _THINK_RE.sub("", text).strip()
    if cleaned.startswith("```"):
        cleaned = _FENCE_START_RE.sub("", cleaned, count=1)
        cleaned = _FENCE_END_RE.sub("", cleaned, count=1)
    return cleaned.strip()


def _iter_json_candidates(text: str) -> list[dict[str, Any] | list[Any]]:
    """Extract all top-level valid JSON objects or arrays from text in order."""
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []

    candidates: list[dict[str, Any] | list[Any]] = []
    decoder = json.JSONDecoder()
    idx = 0
    length = len(cleaned)

    while idx < length:
        ch = cleaned[idx]
        if ch in "{[":
            try:
                parsed, end_idx = decoder.raw_decode(cleaned, idx)
                if isinstance(parsed, (dict, list)):
                    candidates.append(parsed)
                    idx = end_idx
                    continue
            except json.JSONDecodeError:
                pass
        idx += 1

    return candidates


def _extract_first_json(
    text: str,
    expected_type: type | None = None,
) -> dict[str, Any] | list[Any]:
    """Strip markdown fences and extract the first matching JSON object or array."""
    fenced_blocks = re.findall(
        r"```(?:json)?\s*\n?(.*?)```",
        _THINK_RE.sub("", text),
        flags=re.DOTALL | re.IGNORECASE,
    )
    search_pool: list[dict[str, Any] | list[Any]] = []
    for block in fenced_blocks:
        search_pool.extend(_iter_json_candidates(block))
    if not search_pool:
        search_pool = _iter_json_candidates(text)

    if not search_pool:
        raise ValueError("No valid JSON object or array found in model response.")

    if expected_type is dict:
        for item in search_pool:
            if isinstance(item, dict):
                return item
        for item in search_pool:
            if isinstance(item, list) and len(item) == 1 and isinstance(item[0], dict):
                return item[0]
        raise ValueError("Expected a JSON object in model response.")

    if expected_type is list:
        for item in search_pool:
            if isinstance(item, list) and item:
                return item
        for item in search_pool:
            if isinstance(item, list):
                return item
        for item in search_pool:
            if isinstance(item, dict):
                for key in ("queries", "search_queries", "items"):
                    if isinstance(item.get(key), list):
                        return item[key]
        raise ValueError("Expected a JSON array in model response.")

    for item in search_pool:
        if item:
            return item
    return search_pool[0]


def _invoke_model(client: Any, messages: list[dict[str, str]], max_tokens: int) -> str:
    """Call Groq chat completions with an explicit max_tokens limit."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        reasoning_effort="low",
    )
    return (response.choices[0].message.content or "").strip()


def call_llm_json(
    client: Any,
    messages: list[dict[str, str]],
    max_tokens: int,
    expected_type: type | None = None,
) -> dict[str, Any] | list[Any]:
    """Call the LLM and parse JSON, retrying once if parsing fails."""
    content = _invoke_model(client, messages, max_tokens=max_tokens)
    try:
        return _extract_first_json(content, expected_type=expected_type)
    except ValueError:
        retry_messages = list(messages)
        if content:
            retry_messages.append({"role": "assistant", "content": content})
        retry_messages.append({"role": "user", "content": RETRY_PROMPT})

        retry_content = _invoke_model(client, retry_messages, max_tokens=max_tokens)
        return _extract_first_json(retry_content, expected_type=expected_type)
