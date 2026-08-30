"""Snapcompact context engine for Hermes.

Dual-mode context engine:

- **summarize** (default) — LLM prose summary, same behavior as the
  built-in compressor.  Installing the plugin changes nothing until you
  opt in.
- **snapcompact** — local, deterministic bitmap-frame archival via
  @oh-my-pi/snapcompact.  Vision models read the frames back at ~1/3
  the input token cost.

Switch live with ``/compact-mode snapcompact`` or ``/compact-mode summarize``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

from agent.context_engine import ContextEngine

from .serializer import NEWLINE_GLYPH, serialize_messages

logger = logging.getLogger(__name__)

# -- Bridge location ----------------------------------------------------------

_BRIDGE_DIR = Path(__file__).parent / "bridge"
_BRIDGE_SCRIPT = _BRIDGE_DIR / "render.ts"

# -- Hermes model → snapcompact ShapeTarget mapping --------------------------

# Map Hermes provider names to snapcompact wire API identifiers.
_PROVIDER_TO_API: dict[str, str] = {
    "anthropic": "anthropic-messages",
    "amazon-bedrock": "bedrock-converse-stream",
    "openai": "openai-completions",
    "google": "google-generative-ai",
    "google-vertex": "google-vertex",
    "openrouter": "openai-completions",
}


def _find_bun() -> str:
    """Locate the bun binary, preferring PATH then common install locations."""
    import shutil
    found = shutil.which("bun")
    if found:
        return found
    for candidate in [
        os.path.expanduser("~/.bun/bin/bun"),
        "/usr/local/bin/bun",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Could not find 'bun' binary. Install Bun (https://bun.sh) or add it to PATH."
    )


def _call_bridge(request: dict[str, Any], *, timeout: float = 120) -> dict[str, Any]:
    """Call the Bun bridge script with a JSON request, return parsed response."""
    bun = _find_bun()
    result = subprocess.run(
        [bun, "run", str(_BRIDGE_SCRIPT)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_BRIDGE_DIR),
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "unknown error"
        raise RuntimeError(f"snapcompact bridge failed: {detail}")
    return json.loads(result.stdout)


# -- Summary prompt -----------------------------------------------------------

_SUMMARY_TEMPLATE = textwrap.dedent("""\
    Resume prior conversation. Earlier turns archived under HISTORY below, \
    oldest→newest. Read HISTORY fully; continue the live conversation following it.

    Archived transcript scopes:
    - `¶user:`, `¶think:`, `¶ai:`, `¶call:`: user, assistant reasoning, assistant reply, tool call.
    - Unprefixed following lines: current scope. Consecutive same-kind blocks omit repeated prefix.
    - Tool call: `¶call:name(args)//intent`; trailing `//intent` optional. `<out>…</out>`: tool output.

    Reading HISTORY:
    - Plain text: verbatim transcript; rely on it exactly.
    {image_guide}\
    {truncation_note}\
    - If an exact earlier detail matters and a section is unclear, re-derive \
    from workspace (re-read files, re-run commands), rather than guess.

    HISTORY
    ===================""")

_IMAGE_GUIDE_TEMPLATE = textwrap.dedent("""\
    - Some middle sections: images, not text. Each image: one page of that \
    transcript, in reading order between marked delimiters. Solid black cell: \
    newline; runs of spaces collapse to one.
      - Frame: one grid {cols} characters wide, up to {rows} rows tall; read \
    left→right, top→bottom. No word wrap; words may break across rows.
    """)


# -- Engine -------------------------------------------------------------------


class SnapcompactEngine(ContextEngine):
    """Context engine that archives history as dense bitmap PNG frames."""

    # -- Identity -------------------------------------------------------------

    @property
    def name(self) -> str:
        return "snapcompact"

    # -- Configuration --------------------------------------------------------

    # Compress at 75% context usage (inherited default).
    threshold_percent: float = 0.75

    # Suppress routine "compacting…" status for automatic passes — snapcompact
    # is fast and deterministic, no need to announce every pass.
    emit_automatic_compaction_status: bool = False

    def __init__(self, context_length: int = 200_000) -> None:
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

        # "summarize" by default — identical to the built-in compressor.
        # "/compact-mode snapcompact" opts in to bitmap rendering.
        self.mode: str = "summarize"

        # Plugin LLM access — set via set_llm() from register().
        self._llm: object | None = None

        # Per-session state
        self._model_id: str = ""
        self._provider: str = ""
        self._api_mode: str = ""
        self._archive_text: str = ""
        self._truncated_chars: int = 0
        self._previous_summary: str = ""

        self._bridge_checked = False

    def set_llm(self, llm: object | None) -> None:
        """Inject plugin LLM access handle (called by register)."""
        self._llm = llm

    # -- Core interface -------------------------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens > 0 and tokens >= self.threshold_tokens

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact using the active mode."""
        if self.mode == "snapcompact":
            return self._compress_snapcompact(
                messages, current_tokens, focus_topic, force, memory_context,
            )
        return self._compress_summarize(
            messages, current_tokens, focus_topic, force, memory_context,
        )

    # -- Snapcompact mode -----------------------------------------------------

    def _compress_snapcompact(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact via bitmap-frame rendering."""
        self._ensure_bridge()

        # Identify which messages to archive vs keep.
        # Strategy: protect system prompt + first N + last N messages.
        system_msgs, conversation = self._split_system(messages)
        if len(conversation) <= self.protect_first_n + self.protect_last_n:
            return messages  # Nothing to compress

        keep_head = conversation[: self.protect_first_n]
        keep_tail = conversation[-self.protect_last_n :]
        to_archive = conversation[self.protect_first_n : -self.protect_last_n]

        if not to_archive:
            return messages

        # Check if any of the models we talk to are Anthropic — if so,
        # suppress ¶think: sections to avoid reasoning_extraction errors.
        is_anthropic = "claude" in self._model_id.lower() or self._provider == "anthropic"

        # Serialize archived messages to compact text.
        serialized = serialize_messages(
            to_archive,
            include_thinking=not is_anthropic,
        )
        if not serialized.strip():
            return messages

        # Build the accumulated archive source: previous archive + new text.
        if self._archive_text:
            archive_source = f"{self._archive_text}{NEWLINE_GLYPH}{serialized}"
        elif self._previous_summary:
            # Carry forward a text-based summary from the built-in compressor.
            archive_source = (
                f"[Summary of earlier history] {self._previous_summary}"
                f" [Recent conversation] {serialized}"
            )
        else:
            archive_source = serialized

        # Determine shape target for the renderer.
        shape_target: dict[str, str] = {}
        if self._model_id:
            shape_target["id"] = self._model_id
        api = _PROVIDER_TO_API.get(self._provider, "")
        if api:
            shape_target["api"] = api

        # Call the bridge to render frames.
        try:
            response = _call_bridge({
                "action": "render",
                "text": archive_source,
                "model": shape_target or None,
                "maxFrames": 80,
            })
        except Exception:
            logger.exception("snapcompact bridge render failed; returning messages unchanged")
            return messages

        if "error" in response:
            logger.error("snapcompact bridge error: %s", response["error"])
            return messages

        images = response.get("images", [])
        shape = response.get("shape", {})
        geo = response.get("geometry", {})

        # Persist archive source for re-rendering on next compaction.
        self._archive_text = archive_source
        self.compression_count += 1

        # Build the summary message with reading guide.
        cols = geo.get("cols", "?")
        rows = geo.get("rows", "?")

        image_guide = ""
        if images:
            image_guide = _IMAGE_GUIDE_TEMPLATE.format(cols=cols, rows=rows)

        truncation_note = ""
        if self._truncated_chars > 0:
            truncation_note = (
                f"- About {self._truncated_chars} characters of older middle "
                f"history dropped to fit archive budget.\n"
            )

        summary_text = _SUMMARY_TEMPLATE.format(
            image_guide=image_guide,
            truncation_note=truncation_note,
        )

        # Construct the summary content blocks: text guide + image frames.
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": summary_text},
        ]
        for img in images:
            data = img.get("data", "")
            mime = img.get("mimeType", "image/png")
            block: dict[str, Any] = {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
            detail = img.get("detail") or shape.get("imageDetail")
            if detail:
                block["image_url"]["detail"] = detail
            content_blocks.append(block)

        # Also append the archive source as a trailing text block (head+tail)
        # so models always have verbatim text at the chronological edges.
        if len(archive_source) > 0:
            frame_capacity = geo.get("capacity", 10000)
            text_edge = min(frame_capacity, len(archive_source) // 3)
            if text_edge > 0 and len(archive_source) > text_edge * 2:
                text_head = archive_source[:text_edge]
                text_tail = archive_source[-text_edge:]
                if images:
                    content_blocks.append(
                        {"type": "text", "text": "-------------- imaged middle above\n" + text_tail}
                    )
                    # Insert text head right after the summary, before images
                    content_blocks.insert(1, {
                        "type": "text",
                        "text": text_head + "\n-------------- imaged middle below\n",
                    })
                else:
                    content_blocks.append(
                        {"type": "text", "text": archive_source}
                    )
            else:
                content_blocks.append(
                    {"type": "text", "text": archive_source}
                )

        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": content_blocks,
        }

        # Build the compressed message list.
        result = list(system_msgs) + [summary_msg] + list(keep_head) + list(keep_tail)

        # Reset prompt token tracking — the host will re-measure after the
        # compressed request goes out.
        self.last_prompt_tokens = -1

        frame_chars = sum(img.get("chars", 0) for img in images) if images else 0
        total_chars = frame_chars + len(archive_source)
        logger.info(
            "snapcompact: archived %d chars onto %d frame(s), compression #%d",
            total_chars, len(images), self.compression_count,
        )

        return result

    # -- Summarize mode -------------------------------------------------------

    def _compress_summarize(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact via LLM prose summary."""
        system_msgs, conversation = self._split_system(messages)
        if len(conversation) <= self.protect_first_n + self.protect_last_n:
            return messages

        keep_head = conversation[: self.protect_first_n]
        keep_tail = conversation[-self.protect_last_n :]
        to_archive = conversation[self.protect_first_n : -self.protect_last_n]
        if not to_archive:
            return messages

        is_anthropic = "claude" in self._model_id.lower() or self._provider == "anthropic"
        serialized = serialize_messages(to_archive, include_thinking=not is_anthropic)
        if not serialized.strip():
            return messages

        if self._llm is None:
            ok, detail = self.ensure_ready()
            if not ok:
                raise RuntimeError(
                    "compression unavailable: summarize mode has no LLM access "
                    f"and snapcompact fallback is not ready ({detail})"
                )
            logger.warning("summarize: no LLM access; falling back to snapcompact mode")
            return self._compress_snapcompact(
                messages, current_tokens, focus_topic, force, memory_context,
            )

        focus = f" Focus on preserving details about: {focus_topic}" if focus_topic else ""
        prompt = (
            "Summarize the following conversation history into a concise but "
            "complete handoff document. Preserve key decisions, file paths, "
            "code changes, error details, and current task state. Do NOT "
            f"omit actionable specifics.{focus}\n\n{serialized}"
        )
        try:
            summary = self._llm.complete(prompt)
        except Exception:
            if not self.bridge_ready():
                raise
            logger.exception("LLM summary failed; falling back to snapcompact mode")
            return self._compress_snapcompact(
                messages, current_tokens, focus_topic, force, memory_context,
            )

        self._previous_summary = summary
        self.compression_count += 1
        self.last_prompt_tokens = -1

        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": (
                "Resume prior conversation. Summary of earlier context:\n\n"
                + summary
            ),
        }
        result = list(system_msgs) + [summary_msg] + list(keep_head) + list(keep_tail)

        logger.info(
            "snapcompact(summarize): compressed %d messages into %d-char summary, #%d",
            len(to_archive), len(summary), self.compression_count,
        )
        return result
    # -- Optional overrides ---------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        self._archive_text = ""
        self._truncated_chars = 0
        self._previous_summary = ""

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        pass

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._archive_text = ""
        self._truncated_chars = 0
        self._previous_summary = ""

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
        api_mode: str = "",
    ) -> None:
        super().update_model(model, context_length, base_url, api_key, provider, api_mode)
        self._model_id = model
        self._provider = provider
        self._api_mode = api_mode

    def get_status(self) -> dict[str, Any]:
        status = super().get_status()
        status["engine"] = "snapcompact"
        status["mode"] = self.mode
        status["archive_chars"] = len(self._archive_text)
        return status

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        _, conversation = self._split_system(messages)
        return len(conversation) > self.protect_first_n + self.protect_last_n

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _split_system(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split messages into system-prompt prefix and conversation body."""
        system: list[dict[str, Any]] = []
        conversation: list[dict[str, Any]] = []
        past_system = False
        for msg in messages:
            if not past_system and msg.get("role") == "system":
                system.append(msg)
            else:
                past_system = True
                conversation.append(msg)
        return system, conversation

    def ensure_ready(self) -> tuple[bool, str]:
        """Verify (and if needed set up) the rendering bridge.

        Called eagerly at plugin registration so problems surface at startup
        with an actionable message — never mid-session.  Returns
        ``(ok, detail)``; ``detail`` names the exact fix when not ok.
        """
        if self._bridge_checked:
            return True, "ready"
        if not _BRIDGE_SCRIPT.is_file():
            return False, (
                f"bridge script missing at {_BRIDGE_SCRIPT} — reinstall the plugin"
            )
        try:
            bun = _find_bun()
        except FileNotFoundError:
            return False, (
                "Bun is not installed. Fix: curl -fsSL https://bun.sh/install | bash"
            )
        node_modules = _BRIDGE_DIR / "node_modules"
        if not node_modules.is_dir():
            logger.info("snapcompact: installing bridge dependencies...")
            try:
                subprocess.run(
                    [bun, "install"],
                    cwd=str(_BRIDGE_DIR),
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=120,
                )
            except subprocess.CalledProcessError as exc:
                detail = (exc.stderr or exc.stdout or "").strip()[-500:]
                return False, (
                    f"bun install failed: {detail} — "
                    f"run 'cd {_BRIDGE_DIR} && bun install' manually"
                )
            except Exception as exc:
                return False, (
                    f"bun install failed ({exc}) — "
                    f"run 'cd {_BRIDGE_DIR} && bun install' manually"
                )
        self._bridge_checked = True
        return True, "ready"

    def bridge_ready(self) -> bool:
        """Non-raising readiness probe."""
        ok, _ = self.ensure_ready()
        return ok

    def _ensure_bridge(self) -> None:
        """Raise with an actionable message when the bridge is unusable."""
        ok, detail = self.ensure_ready()
        if not ok:
            raise RuntimeError(f"snapcompact bridge unavailable: {detail}")
