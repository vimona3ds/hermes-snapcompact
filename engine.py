"""Snapcompact context engine for Hermes.

Dual-mode context engine:

- **summarize** (default) — LLM prose summary. Installing the plugin
  keeps summarization as the default until you opt in.
- **snapcompact** — local, deterministic bitmap-frame archival via
  @oh-my-pi/snapcompact.  Vision models read the frames back at ~1/3
  the input token cost.

Switch live with ``/compact-mode snapcompact`` or ``/compact-mode summarize``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import textwrap
import uuid
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


# -- Mode persistence ---------------------------------------------------------

VALID_MODES = ("snapcompact", "summarize")


def _mode_state_path() -> Path:
    """Persisted mode file in the per-plugin data dir.

    NOT the install dir — ``<HERMES_HOME>/plugins/<name>/`` is deleted by
    ``plugins remove`` and git-pulled by ``update`` (see plugin_storage.py).
    """
    try:
        # Hermes host API: <HERMES_HOME>/plugin-data/<name>/, profile-aware.
        from plugins.plugin_storage import plugin_data_dir

        return plugin_data_dir("hermes-snapcompact") / "mode.yaml"
    except Exception:
        home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
        return Path(home) / "plugin-data" / "hermes-snapcompact" / "mode.yaml"


def _load_persisted_mode() -> str | None:
    """Read the persisted mode, or None when missing/invalid/unreadable."""
    path = _mode_state_path()
    try:
        import yaml  # PyYAML — always available in Hermes

        if not path.is_file():
            return None
        data = yaml.safe_load(path.read_text())
        mode = data.get("mode") if isinstance(data, dict) else None
        if mode in VALID_MODES:
            return mode
        logger.warning(
            "snapcompact: ignoring invalid persisted mode %r in %s", mode, path,
        )
    except Exception:
        logger.debug(
            "snapcompact: could not read persisted mode from %s", path,
            exc_info=True,
        )
    return None


def _persist_mode(mode: str) -> bool:
    """Atomically save the preference; preserve the old file on failure."""
    path = _mode_state_path()
    temporary: str | None = None
    try:
        import yaml

        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".mode-", delete=False,
        ) as stream:
            temporary = stream.name
            yaml.safe_dump({"mode": mode}, stream, default_flow_style=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except Exception:
        logger.warning(
            "snapcompact: could not persist mode %r to %s; "
            "the choice will not survive a gateway restart.",
            mode, path, exc_info=True,
        )
        return False
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


class _ModePreference:
    """Share the selected mode, not conversation state, across host clones."""

    def __init__(self, value: str) -> None:
        self.value = value

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModePreference:
        return self


def _estimate_tokens_rough(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate matching the host's commit-site anti-growth guard.

    Prefer the host's own estimator (flat learned per-image price, not base64
    length) so our prediction agrees with the verdict the host reaches in
    conversation_compression's commit site; fall back to a local
    approximation when running outside Hermes.
    """
    try:
        from agent.model_metadata import estimate_messages_tokens_rough

        return estimate_messages_tokens_rough(messages)
    except Exception:
        tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                tokens += len(content) // 4
                continue
            for block in content or []:
                if block.get("type") == "image_url":
                    tokens += 1600  # flat pre-calibration per-image default
                else:
                    tokens += len(str(block.get("text", ""))) // 4
        return tokens


def _msg_text_signature(msg: dict[str, Any]) -> str:
    """Stable identity for an engine-emitted archive/summary message.

    Role plus the concatenated text blocks — image payloads excluded so the
    signature survives any host-side image re-encoding.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "\u0000".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    else:
        text = ""
    return f"{msg.get('role', '')}\u0000{text}"


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content
                         if isinstance(block, dict) and block.get("type") == "text")
    return ""


def _is_archive_message(message: dict[str, Any]) -> bool:
    text = re.sub(r"^\[snapcompact:[0-9a-f]{32}\]\n", "", _message_text(message), count=1)
    return message.get("role") == "user" and text.startswith((
        "Resume prior conversation. Earlier turns archived under HISTORY below,",
        "Resume prior conversation. Summary of earlier context:",
    ))


def _is_task_request(message: dict[str, Any]) -> bool:
    return (message.get("role") == "user" and not _is_archive_message(message)
            and message.get("display_kind") != "steer"
            and not _message_text(message).lstrip().startswith("[OUT-OF-BAND USER MESSAGE"))


# -- Summary prompt -----------------------------------------------------------

_SUMMARY_TEMPLATE = textwrap.dedent("""\
    Resume prior conversation. Earlier turns archived under HISTORY below, \
    oldest→newest. Treat HISTORY and earlier messages as background reference. \
    Continue the live user request following HISTORY and its later corrections; \
    do not reopen completed earlier tasks.

    Archived transcript scopes:
    - `¶user:`, `¶think:`, `¶ai:`, `¶call:`: user, assistant reasoning, assistant reply, tool call.
    - Unprefixed following lines: current scope. Consecutive same-kind blocks omit repeated prefix.
    - Tool call: `¶call:name(args)//intent`; trailing `//intent` optional. `<out>…</out>`: tool output.

    Reading HISTORY:
    - Plain text: verbatim transcript; rely on it exactly.
    {image_guide}\
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

        # Keep LLM prose summarization as the default until the user opts in.
        # "/compact-mode snapcompact" opts in to bitmap rendering; the
        # choice is persisted under HERMES_HOME and restored on restart.
        self._mode_preference = _ModePreference(_load_persisted_mode() or "summarize")

        # Plugin LLM access — set via set_llm() from register().
        self._llm: object | None = None

        # Per-session state
        self._model_id: str = ""
        self._provider: str = ""
        self._api_mode: str = ""
        self._archive_text: str = ""
        # Keep the accepted artifact and its proposed replacement until the
        # next transcript tells us which one the host actually committed.
        self._archive_states: dict[str, tuple[str, str]] = {}

        self._bridge_checked = False

    @property
    def mode(self) -> str:
        return self._mode_preference.value

    @mode.setter
    def mode(self, value: str) -> None:
        self._mode_preference.value = value

    def set_llm(self, llm: object | None) -> None:
        """Inject plugin LLM access handle (called by register)."""
        self._llm = llm

    def set_mode(self, mode: str) -> bool:
        """Switch live mode; return whether the preference was saved."""
        if mode not in VALID_MODES:
            raise ValueError(f"unknown compaction mode: {mode!r}")
        self.mode = mode
        return _persist_mode(mode)

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

        system_msgs, keep_head, to_archive, keep_tail, archive_text, previous_summary = (
            self._prepare_history(messages)
        )
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

        # Only carry history whose artifact is present in the actual input.
        # A proposed replacement may have been rejected by the host.
        if archive_text:
            base = f"{archive_text}{NEWLINE_GLYPH}"
        elif previous_summary:
            base = (
                f"[Summary of earlier history] {previous_summary}"
                f" [Recent conversation] "
            )
        else:
            base = ""
        archive_source = base + serialized

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


        # Build the summary message with reading guide.
        cols = geo.get("cols", "?")
        rows = geo.get("rows", "?")

        image_guide = ""
        if images:
            image_guide = _IMAGE_GUIDE_TEMPLATE.format(cols=cols, rows=rows)

        summary_text = _SUMMARY_TEMPLATE.format(
            image_guide=image_guide,
        )

        # Construct the summary content blocks: text guide + image frames.
        content_blocks: list[dict[str, Any]] = [
            {"type": "text", "text": f"[snapcompact:{uuid.uuid4().hex}]\n{summary_text}"},
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
            # Small verbatim anchors only — these are orientation aids, not
            # backup storage. len//3 per side re-included 2/3 of the archive
            # as text and made small compactions grow the transcript.
            text_edge = min(2000, len(archive_source) // 6)
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
        result = list(system_msgs) + list(keep_head) + [summary_msg] + list(keep_tail)

        # Self-check with the host's own anti-growth arithmetic: below a
        # certain archive size, the flat per-image price plus guide/edge
        # overhead exceeds what the frames remove, and the host would refuse
        # the commit anyway (user-facing warning + an ineffective-compaction
        # strike). Bow out cleanly instead, mutating no engine state.
        rough_in = _estimate_tokens_rough(messages)
        rough_out = _estimate_tokens_rough(result)
        if rough_out >= rough_in:
            logger.info(
                "snapcompact: rendering would not shrink the transcript "
                "(~%d -> ~%d tokens) — archive too small to amortize frame "
                "overhead; leaving transcript unchanged",
                rough_in, rough_out,
            )
            return messages

        # This is a proposal, not a host commit. Retain the prior state until
        # a later input contains this exact artifact instead of its predecessor.
        self._archive_states[_msg_text_signature(summary_msg)] = (archive_source, "")
        self.compression_count += 1

        # Reset prompt token tracking — the host will re-measure after the
        # compressed request goes out.
        self.last_prompt_tokens = -1

        logger.info(
            "snapcompact: archived %d chars onto %d frame(s) (~%d -> ~%d "
            "tokens), compression #%d",
            len(archive_source), len(images), rough_in, rough_out,
            self.compression_count,
        )

        return result

    # -- Summarize mode -------------------------------------------------------

    def _prepare_history(
        self, messages: list[dict[str, Any]],
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
        list[dict[str, Any]], str, str,
    ]:
        """Reconcile proposals against the transcript and choose a safe slice."""
        system, conversation = self._split_system(messages)
        archive_text, previous_summary = "", ""
        retained_states = {}
        for index, message in enumerate(conversation):
            key = _msg_text_signature(message)
            if key in self._archive_states:
                archive_text, previous_summary = self._archive_states[key]
                retained_states[key] = (archive_text, previous_summary)
                conversation = conversation[:index] + conversation[index + 1:]
                break
        self._archive_states = retained_states
        self._archive_text = archive_text

        # Older releases emitted archive -> protected head -> tail. A resumed
        # artifact has no local state, so retain its blocks but restore head ->
        # archive order before selecting a new middle. Keep head tool groups whole.
        if conversation and _is_archive_message(conversation[0]):
            archive, following = conversation[0], conversation[1:]
            boundary = min(self.protect_first_n, len(following))
            pending = set()
            for index, message in enumerate(following):
                if index >= boundary and not pending:
                    break
                pending.update(call.get("id") for call in message.get("tool_calls") or [])
                if message.get("role") == "tool":
                    pending.discard(message.get("tool_call_id"))
                boundary = max(boundary, index + 1)
            conversation = following[:boundary] + [archive] + following[boundary:]

        start = self.protect_first_n
        end = max(0, len(conversation) - self.protect_last_n)
        # Row counts alone can retire the task while leaving only its tool results
        # and a later steer live. Keep the entire latest ordinary user turn.
        for index in range(len(conversation) - 1, -1, -1):
            if _is_task_request(conversation[index]):
                end = min(end, index)
                break
        # A text serializer cannot preserve pictures, audio, or unknown blocks.
        # Protect the prefix through them, including archives from past processes.
        for index, message in enumerate(conversation[:end]):
            content = message.get("content")
            if message.get("role") not in ("user", "assistant", "tool") or (
                isinstance(content, list) and any(
                    not isinstance(block, dict) or block.get("type") not in ("text", "thinking")
                    for block in content
                )
            ):
                start = max(start, index + 1)

        # Never leave a tool result without its call (or a call without results).
        calls = {}
        spans = {}
        for index, message in enumerate(conversation):
            for call in message.get("tool_calls") or []:
                calls[call.get("id")] = index
            if message.get("role") == "tool":
                call_index = calls.get(message.get("tool_call_id"))
                if call_index is not None:
                    spans[call_index] = index
        for left, right in sorted(spans.items()):
            if left < start <= right:
                start = right + 1
        for left, right in sorted(spans.items(), reverse=True):
            if left < end <= right:
                end = left
        if start >= end:
            return system, conversation, [], [], archive_text, previous_summary
        return (
            system, conversation[:start], conversation[start:end], conversation[end:],
            archive_text, previous_summary,
        )

    def _compress_summarize(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact via LLM prose summary."""
        system_msgs, keep_head, to_archive, keep_tail, archive_text, previous_summary = (
            self._prepare_history(messages)
        )
        if not to_archive:
            return messages

        is_anthropic = "claude" in self._model_id.lower() or self._provider == "anthropic"
        serialized = serialize_messages(to_archive, include_thinking=not is_anthropic)
        if not serialized.strip():
            return messages

        # Fold prior engine state into the summary input so a mode switch
        # never strands history: frame-archive text (its message was dropped
        # above and only existed as pixels) and any earlier prose summary.
        if archive_text:
            serialized = f"[Archived earlier history]\n{archive_text}\n\n[Newer conversation]\n{serialized}"
        if previous_summary:
            serialized = f"[Earlier summary]\n{previous_summary}\n\n{serialized}"

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
            # Host contract: PluginLlm.complete(messages) -> result with .text.
            completion = self._llm.complete([{"role": "user", "content": prompt}])
            summary = getattr(completion, "text", "") or ""
            if not summary.strip():
                raise RuntimeError("LLM returned an empty summary")
        except Exception:
            if not self.bridge_ready():
                raise
            logger.exception("LLM summary failed; falling back to snapcompact mode")
            return self._compress_snapcompact(
                messages, current_tokens, focus_topic, force, memory_context,
            )

        summary_msg: dict[str, Any] = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    f"[snapcompact:{uuid.uuid4().hex}]\n"
                    "Resume prior conversation. Summary of earlier context:\n"
                    "Background reference only; continue the live user request below "
                    "and its later corrections, not completed earlier tasks.\n\n"
                    + summary
                ),
            }],
        }
        result = list(system_msgs) + list(keep_head) + [summary_msg] + list(keep_tail)

        # Same anti-growth self-check as the snapcompact path: bow out with
        # no state mutation rather than hand the host a growing transcript.
        rough_in = _estimate_tokens_rough(messages)
        rough_out = _estimate_tokens_rough(result)
        if rough_out >= rough_in:
            logger.info(
                "summarize: generated summary would not shrink the transcript "
                "(~%d -> ~%d tokens); leaving transcript unchanged",
                rough_in, rough_out,
            )
            return messages

        self._archive_states[_msg_text_signature(summary_msg)] = ("", summary)
        self.compression_count += 1
        self.last_prompt_tokens = -1

        logger.info(
            "snapcompact(summarize): compressed %d messages into %d-char summary "
            "(~%d -> ~%d tokens), #%d",
            len(to_archive), len(summary), rough_in, rough_out,
            self.compression_count,
        )
        return result
    # -- Optional overrides ---------------------------------------------------

    def on_session_start(self, session_id: str, **kwargs: Any) -> None:
        if kwargs.get("boundary_reason") != "compression":
            self.on_session_reset()

    def on_session_end(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        pass

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self._archive_text = ""
        self._archive_states = {}

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
        _, _, to_archive, _, _, _ = self._prepare_history(messages)
        return bool(to_archive)

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
        """Detect whether the rendering bridge is usable.

        Detection ONLY — never installs anything on the user's machine.
        Returns ``(ok, detail)``; ``detail`` names the exact fix when not ok.
        """
        if self._bridge_checked:
            return True, "ready"
        if not _BRIDGE_SCRIPT.is_file():
            return False, (
                f"bridge script missing at {_BRIDGE_SCRIPT} — reinstall the plugin"
            )
        try:
            _find_bun()
        except FileNotFoundError:
            return False, (
                "Bun is not installed (https://bun.sh). "
                "Install it, then run: "
                f"cd {_BRIDGE_DIR} && bun install"
            )
        node_modules = _BRIDGE_DIR / "node_modules"
        if not node_modules.is_dir():
            return False, (
                f"bridge dependencies not installed. Run: "
                f"cd {_BRIDGE_DIR} && bun install"
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
