"""Serialize Hermes (OpenAI-format) messages into snapcompact archive text.

Uses the same scope format the snapcompact reading guide teaches the model:
  - ``¶user:``  user turns
  - ``¶ai:``    assistant text
  - ``¶think:`` assistant reasoning (omitted by default for Anthropic models)
  - ``¶call:``  tool calls with ``<out>…</out>`` results

Tool results are truncated to keep the archive dense; exact output can
always be re-derived from the workspace.
"""

from __future__ import annotations

import json
import re
from typing import Any

# -- Truncation defaults (match omp's snapcompact) ---------------------------

TOOL_RESULT_MAX_CHARS = 2000
TOOL_ARG_MAX_CHARS = 500
TOOL_CALL_MAX_CHARS = 2000
TRUNCATE_HEAD_RATIO = 0.6

# Full-block glyph printed in place of newline runs.
NEWLINE_GLYPH = "\u2588"

# -- Data URL elision ---------------------------------------------------------

_DATA_URL_RE = re.compile(
    r"data:[A-Za-z][\w.+\-]*/[\w.+\-]+(?:;[\w!#$%&'*+.^|~\-]+=[\w!#$%&'*+.^|~\-]+)*"
    r";base64,[A-Za-z0-9+/=\s]*",
    re.IGNORECASE,
)


def _elide_data_urls(text: str) -> str:
    """Replace inline base64 data URLs with a short placeholder."""
    def _repl(m: re.Match[str]) -> str:
        full = m.group(0)
        # Extract mime from between data: and ;base64,
        semi = full.index(";base64,")
        mime = full[5:semi]
        payload_len = len(full) - semi - 8
        return f"[data URL omitted: {mime}, {payload_len} base64 chars]"
    return _DATA_URL_RE.sub(_repl, text)


# -- Truncation ---------------------------------------------------------------

def _truncate(text: str, max_chars: int, head_ratio: float = TRUNCATE_HEAD_RATIO) -> str:
    """Keep head and tail of ``text``, eliding the middle."""
    if len(text) <= max_chars:
        return text
    ratio = max(0.0, min(1.0, head_ratio))
    head = int(max_chars * ratio)
    tail = max_chars - head
    elided = len(text) - max_chars
    tail_part = text[-tail:] if tail > 0 else ""
    return f"{text[:head]} [...{elided}ch elided...] {tail_part}"


# -- Helpers -------------------------------------------------------------------

def _text_from_content(content: Any) -> str:
    """Extract plain text from a message's content field."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _format_args(arguments: str | dict[str, Any], max_arg: int, max_total: int,
                 head_ratio: float) -> str:
    """Format tool call arguments into a compact string."""
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return _truncate(_elide_data_urls(arguments), max_total, head_ratio)
    else:
        args = arguments

    if not isinstance(args, dict):
        return _truncate(_elide_data_urls(str(args)), max_total, head_ratio)

    pairs = []
    for key, value in args.items():
        if key == "i":  # Hermes intent field — handled separately
            continue
        val_str = json.dumps(value) if not isinstance(value, str) else value
        pairs.append(f"{key}={_truncate(_elide_data_urls(val_str), max_arg, head_ratio)}")

    return _truncate(", ".join(pairs), max_total, head_ratio)


# -- Public API ----------------------------------------------------------------

def serialize_messages(
    messages: list[dict[str, Any]],
    *,
    tool_result_max_chars: int = TOOL_RESULT_MAX_CHARS,
    tool_arg_max_chars: int = TOOL_ARG_MAX_CHARS,
    tool_call_max_chars: int = TOOL_CALL_MAX_CHARS,
    head_ratio: float = TRUNCATE_HEAD_RATIO,
    include_thinking: bool = False,
) -> str:
    """Serialize an OpenAI-format message list into snapcompact archive text.

    Args:
        messages: OpenAI-format message list (role/content/tool_calls/…).
        include_thinking: Whether to emit ``¶think:`` sections. Defaults to
            False — Anthropic models trip on reasoning replayed as text.
    """
    parts: list[str] = []
    last_prefix: str | None = None

    # Index tool results by tool_call_id for merging into their call blocks.
    tool_results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id", "")
            text = _text_from_content(msg.get("content", ""))
            if text and call_id:
                tool_results[call_id] = text

    merged_ids: set[str] = set()

    def _push(prefix: str, content: str) -> None:
        nonlocal last_prefix
        if parts and last_prefix == prefix:
            sep = "" if parts[-1].endswith("\n") or content.startswith("\n") else "\n"
            parts[-1] += sep + content
        else:
            parts.append(prefix + content)
            last_prefix = prefix

    def _render_result(raw: str) -> str:
        body = _truncate(_elide_data_urls(raw), tool_result_max_chars, head_ratio)
        return f"<out>\n{body}\n</out>"

    for msg in messages:
        role = msg.get("role", "")

        if role == "user":
            text = _text_from_content(msg.get("content", ""))
            if text.strip():
                _push("¶user:", text)

        elif role == "assistant":
            content = msg.get("content")
            # Handle structured content blocks (Anthropic style)
            if isinstance(content, list):
                for block in content:
                    btype = block.get("type", "")
                    if btype == "text":
                        text = block.get("text", "")
                        if text.strip():
                            _push("¶ai:", text)
                    elif btype == "thinking" and include_thinking:
                        thinking = block.get("thinking", "")
                        if thinking.strip():
                            _push("¶think:", thinking)
            elif isinstance(content, str) and content.strip():
                _push("¶ai:", content)

            # Tool calls (OpenAI format)
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "unknown")
                raw_args = func.get("arguments", "{}")
                call_id = tc.get("id", "")

                # Extract intent from args if present
                try:
                    parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except (json.JSONDecodeError, TypeError):
                    parsed = {}
                intent = ""
                if isinstance(parsed, dict):
                    intent = str(parsed.get("i", "")).strip()

                args_str = _format_args(raw_args, tool_arg_max_chars,
                                        tool_call_max_chars, head_ratio)

                line = f"{name}({args_str})"
                if intent:
                    line += f"//{intent}"

                lines = [line]
                result_text = tool_results.get(call_id)
                if result_text is not None:
                    merged_ids.add(call_id)
                    lines.append(_render_result(result_text))
                _push("¶call:", "\n".join(lines))

        elif role == "tool":
            call_id = msg.get("tool_call_id", "")
            if call_id in merged_ids:
                continue
            text = tool_results.get(call_id, "")
            if text:
                _push("¶call:", f"\n{_render_result(text)}")

        # Skip system messages — they're always kept verbatim by the engine.

    return "\n\n".join(parts)
