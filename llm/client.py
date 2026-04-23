"""Thin Anthropic client. Reads API key from env only — never from a hardcoded value."""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()


class LLMError(Exception):
    """Base class for LLM-stage failures."""


class APIKeyMissing(LLMError):
    pass


class ResponseParseError(LLMError):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def call_claude(
    system_prompt: str,
    user_message: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    api_key: str | None = None,
) -> dict:
    """Send to Anthropic, parse JSON response, raise typed errors on failure."""
    import anthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise APIKeyMissing(
            "ANTHROPIC_API_KEY not set. Create .env (copy from .env.example) and add your key, "
            "or pass api_key explicitly."
        )

    # Append a hard reminder to the system prompt so Claude emits JSON only.
    # Not as strong as assistant prefill, but prefill isn't supported on every model.
    hardened_system = (
        system_prompt.rstrip()
        + "\n\nCRITICAL OUTPUT CONTRACT:\n"
        + "- Respond with a SINGLE JSON object and nothing else.\n"
        + "- No preamble, no reasoning prose, no markdown fences, no trailing text.\n"
        + "- Your response MUST start with `{` and end with `}`.\n"
    )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=hardened_system,
        messages=[{"role": "user", "content": user_message}],
    )
    text = _strip_fences(response.content[0].text)

    return _parse_json_lenient(text)


def _parse_json_lenient(text: str) -> dict:
    """Parse JSON from a response that may have preamble/trailing prose.

    Tries strict parse first, then falls back to extracting the largest balanced
    `{...}` substring using a simple brace-counting state machine that's aware
    of string literals.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        raise ResponseParseError(f"Claude returned non-JSON:\n{text[:500]}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError as exc:
                        raise ResponseParseError(
                            f"Extracted a balanced JSON block but parsing failed: {exc}\n---\n{candidate[:500]}"
                        )

    raise ResponseParseError(
        f"Response was not valid JSON and no balanced `{{...}}` block found.\n---\n{text[:500]}"
    )
