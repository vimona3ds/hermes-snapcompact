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
from dataclasses import dataclass, field
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


# -- Engine artifacts in the transcript ---------------------------------------

# Every summary or frame archive starts with a marker line and a fixed opening.
# 1.0.0 frame archives had no marker line, so the opening alone also counts.
# Finding artifacts in the transcript (not only in memory) lets a re-created
# agent, a restart or another surface fold them instead of stacking them.
_MARKER_RE = re.compile(r"\[snapcompact:[0-9a-f]{32}\]\n")
_ARTIFACT_RE = re.compile(
    r"(?:\[snapcompact:[0-9a-f]{32}\]\n)?Resume prior conversation\. "
    r"(?P<kind>Summary of earlier context:|Earlier turns archived under HISTORY below)"
)
_SUMMARY_HEADER = "Resume prior conversation. Summary of earlier context:\n\n"
_SUMMARY_BACKGROUND = (
    "Background reference only; continue the live user request below "
    "and its later corrections, not completed earlier tasks.\n\n"
)

# A summary line starting with this says part of the history was lost to
# unreadable page images. Later summaries carry these lines forward verbatim.
_GAP_PREFIX = "[snapcompact:gap]"

# Real input cost of one rendered frame (~1568x1562 PNG). Hermes learns one
# per-image price for every image and, for frames, it came out far too low
# (371 learned vs ~2,500 real on gpt-6-astra, fitted from 44 compactions).
DEFAULT_FRAME_IMAGE_TOKENS = 2500
_MIN_IMAGE_TOKENS, _MAX_IMAGE_TOKENS = 64, 32_768


@dataclass
class _Artifact:
    """A summary or frame archive found in the transcript."""

    message: dict[str, Any]
    kind: str  # "summary" or "frames"
    # Summary body; a frame archive's full source when known, else its stored text.
    text: str
    # Frame archives: stored blocks (reading guide, edge text, page images).
    blocks: list[dict[str, Any]] = field(default_factory=list)
    images: int = 0
    # False when part of the history exists only as page images.
    complete: bool = True


@dataclass
class _Plan:
    """How one compaction splits the transcript."""

    system: list[dict[str, Any]]
    head: list[dict[str, Any]]  # kept verbatim, before the new artifact
    archive: list[dict[str, Any]]  # new history to summarize or render
    tail: list[dict[str, Any]]  # recent turns kept verbatim
    folded: list[_Artifact]  # earlier artifacts to merge in, oldest first

    def worth_compressing(self, *, summarize: bool) -> bool:
        """New history to archive, artifacts to merge, or frames to turn into text."""
        if self.archive or len(self.folded) >= 2:
            return True
        return summarize and any(a.kind == "frames" for a in self.folded)


def _parse_artifact(msg: Any, frame_sources: dict[str, str]) -> _Artifact | None:
    """Recognise an engine summary or frame archive, or return None."""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return None
    content = msg.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
    if not isinstance(blocks, list) or not blocks:
        return None
    if any(not isinstance(b, dict) or b.get("type") not in ("text", "image_url") for b in blocks):
        return None  # Something the fold cannot carry: leave it to the pin rule.
    first = blocks[0]
    match = _ARTIFACT_RE.match(str(first.get("text", ""))) if first.get("type") == "text" else None
    if match is None:
        return None
    stripped = [{**first, "text": _MARKER_RE.sub("", str(first["text"]), count=1)}] + blocks[1:]
    images = sum(1 for b in blocks if b.get("type") == "image_url")
    if match["kind"].startswith("Summary") and not images:
        body = "\n\n".join(str(b.get("text", "")) for b in stripped)
        return _Artifact(msg, "summary", body.removeprefix(_SUMMARY_HEADER).removeprefix(_SUMMARY_BACKGROUND))
    source = frame_sources.get(_msg_text_signature(msg))
    if source is not None:
        return _Artifact(msg, "frames", source, stripped, images)
    # Past-process archive: the reading guide is block 0; the rest is edge text.
    text = "\n".join(str(b.get("text", "")) for b in stripped[1:] if b.get("type") == "text")
    return _Artifact(msg, "frames", text, stripped, images, complete=not images)


def _host_image_cost() -> int:
    """Per-image price Hermes' rough estimator charges (see _estimate_tokens_rough)."""
    try:
        from agent.image_token_cost import current_image_token_cost

        return int(current_image_token_cost())
    except Exception:
        return 1600


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

# Built-in summarizer intro, used when no summary_prompt_file is set or it
# cannot be read.
_BUILTIN_INTRO = (
    "Summarize the following conversation history into a concise but "
    "complete handoff document. Preserve key decisions, file paths, "
    "code changes, error details, and current task state. Do NOT "
    "omit actionable specifics."
)

# The kept tail stays verbatim after the summary. The summarizer sees a short
# digest of it, so it does not call a thing unanswered only because the answer
# sits in the tail. Bounded: a large tool result in the tail must not blow up
# the summary request.
_TAIL_MARK = "[Kept verbatim after your summary: do not summarize or repeat]"
_TAIL_TEXT_CHARS = 300
_TAIL_DIGEST_BYTES = 4096


def _plain_text(content: Any) -> str:
    """Text of a message's content; other blocks (images, thinking) skipped."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(b.get("text", "")) for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _tail_digest(tail: list[dict[str, Any]]) -> str:
    """A read-only digest of the kept tail, at most _TAIL_DIGEST_BYTES (UTF-8).

    User and assistant text is cut to _TAIL_TEXT_CHARS each, on one line;
    tool results become ``tool <name> (N chars)``. Rows that would pass the
    byte limit are dropped and counted in a last line. Empty when there is
    nothing to show.
    """
    names = {
        call.get("id"): (call.get("function") or {}).get("name") or "?"
        for message in tail for call in message.get("tool_calls") or []
        if isinstance(call, dict)
    }
    rows: list[str] = []
    for message in tail:
        role = message.get("role")
        text = _plain_text(message.get("content"))
        if role == "tool":
            name = names.get(message.get("tool_call_id")) or message.get("name") or "?"
            rows.append(f"tool {name} ({len(text)} chars)")
        elif role in ("user", "assistant"):
            line = " ".join(text.split())
            if not line:
                continue
            if len(line) > _TAIL_TEXT_CHARS:
                line = line[: _TAIL_TEXT_CHARS - 1] + "…"
            rows.append(f"{role}: {line}")
    if not rows:
        return ""
    # Room for the mark and a worst-case "omitted" line.
    budget = _TAIL_DIGEST_BYTES - len(_TAIL_MARK.encode("utf-8")) - 64
    kept: list[str] = []
    for row in rows:
        cost = len(row.encode("utf-8")) + 1  # the row and its newline
        if cost > budget:
            break
        kept.append(row)
        budget -= cost
    if len(kept) < len(rows):
        kept.append(f"({len(rows) - len(kept)} more kept messages not shown)")
    return "\n".join([_TAIL_MARK, *kept])


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
        # Full text of frame archives this process rendered, by message
        # signature. Only the pixels survive in the transcript; entries whose
        # archive is no longer in the transcript are dropped on the next pass.
        self._frame_sources: dict[str, str] = {}

        # Our own size math prices each frame image at this many tokens.
        self.frame_image_tokens: int = DEFAULT_FRAME_IMAGE_TOKENS

        # Plugin setting summary_prompt_file: a path, read at every summary.
        # A plain string, so host deep copies of the engine stay cheap and safe.
        self.summary_prompt_file: str = ""

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

    def set_frame_image_tokens(self, value: Any) -> None:
        """Set the per-frame-image price (plugin setting ``frame_image_tokens``)."""
        try:
            tokens = int(value)
        except (TypeError, ValueError):
            tokens = 0
        if not _MIN_IMAGE_TOKENS <= tokens <= _MAX_IMAGE_TOKENS:
            logger.warning(
                "snapcompact: ignoring frame_image_tokens=%r (want %d-%d); using %d",
                value, _MIN_IMAGE_TOKENS, _MAX_IMAGE_TOKENS, DEFAULT_FRAME_IMAGE_TOKENS,
            )
            tokens = DEFAULT_FRAME_IMAGE_TOKENS
        self.frame_image_tokens = tokens

    def set_summary_prompt_file(self, value: Any) -> None:
        """Set the summary intro file (plugin setting ``summary_prompt_file``).

        Only the path is kept; the file is read at every compaction, so edits
        to it apply without a restart.
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            self.summary_prompt_file = ""
            return
        if not isinstance(value, str):
            logger.warning(
                "snapcompact: ignoring summary_prompt_file=%r (want a path); "
                "using the built-in summary prompt", value,
            )
            self.summary_prompt_file = ""
            return
        self.summary_prompt_file = value.strip()

    def _summary_intro(self) -> str:
        """The summary prompt file's text, or the built-in intro when it is
        unset, missing, unreadable or empty (with a warning)."""
        if not self.summary_prompt_file:
            return _BUILTIN_INTRO
        path = Path(os.path.expanduser(self.summary_prompt_file))
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning(
                "summarize: cannot read summary_prompt_file %s (%s); using the "
                "built-in summary prompt", path, exc,
            )
            return _BUILTIN_INTRO
        if not text:
            logger.warning(
                "summarize: summary_prompt_file %s is empty; using the built-in "
                "summary prompt", path,
            )
            return _BUILTIN_INTRO
        return text

    # -- Core interface -------------------------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)
        self.last_completion_tokens = usage.get("completion_tokens", 0)
        self.last_total_tokens = usage.get("total_tokens", 0)

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return tokens > 0 and tokens >= self.threshold_tokens

    def should_compress_preflight(self, messages: list[dict[str, Any]]) -> bool:
        """Catch what the host's estimate misses before any real usage exists.

        With real usage (after the first response) the host's figure already
        includes the true image cost. Before it (new or re-created agent), the
        host prices each frame image far too low; count them honestly here.
        """
        if self.last_prompt_tokens != 0 or self.threshold_tokens <= 0:
            return False
        host, honest = self._estimate(messages)
        return honest > host and honest >= self.threshold_tokens and (
            self.has_content_to_compress(messages)
        )

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

        plan = self._prepare_history(messages, fold_image_only=False)
        if not plan.worth_compressing(summarize=False):
            return messages

        # Check if any of the models we talk to are Anthropic — if so,
        # suppress ¶think: sections to avoid reasoning_extraction errors.
        is_anthropic = "claude" in self._model_id.lower() or self._provider == "anthropic"

        # Serialize archived messages to compact text.
        serialized = serialize_messages(
            plan.archive,
            include_thinking=not is_anthropic,
        )

        # Carry every folded artifact, oldest first, then the new history.
        # Archives whose middle survives only as pixels are never folded here
        # (they cannot be re-rendered from text); they stay in place.
        pieces = [
            a.text if a.kind == "frames" else f"[Summary of earlier history] {a.text}"
            for a in plan.folded
        ]
        if serialized.strip():
            pieces.append(f"[Recent conversation] {serialized}" if plan.folded else serialized)
        archive_source = NEWLINE_GLYPH.join(p for p in pieces if p.strip())
        if not archive_source.strip():
            return messages

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
        result = plan.system + plan.head + [summary_msg] + plan.tail

        # Self-check: below a certain archive size, the frames plus guide and
        # edge text cost more than they remove. Check with the host's own
        # anti-growth arithmetic (else it refuses the commit with a warning and
        # an ineffective-compaction strike) and with honest frame prices.
        # Bow out cleanly instead, mutating no engine state.
        sizes = self._shrink_check(messages, result, "snapcompact: rendering")
        if sizes is None:
            return messages

        # This is a proposal, not a host commit: the next transcript shows
        # whether it was kept. Until then its text lives beside the old ones.
        self._frame_sources[_msg_text_signature(summary_msg)] = archive_source
        self.compression_count += 1

        # Reset prompt token tracking — the host will re-measure after the
        # compressed request goes out.
        self.last_prompt_tokens = -1

        logger.info(
            "snapcompact: archived %d chars onto %d frame(s), folded %d earlier "
            "artifact(s) (~%d -> ~%d tokens; ~%d -> ~%d with frames at %d each), "
            "compression #%d",
            len(archive_source), len(images), len(plan.folded), *sizes,
            self.frame_image_tokens, self.compression_count,
        )

        return result

    # -- Shared helpers ---------------------------------------------------------

    def _prepare_history(
        self, messages: list[dict[str, Any]], *, fold_image_only: bool,
    ) -> _Plan:
        """Find earlier summaries and archives in the transcript; choose a safe slice.

        Earlier artifacts are folded into the next one, so the transcript never
        carries more than one. ``fold_image_only`` also folds archives whose
        middle survives only as page images (the summarizer reads the images
        or the summary records the gap); the frame path keeps those in place.
        """
        system, conversation = self._split_system(messages)
        # Legacy releases put a single archive before the protected opening.
        # Move it behind that opening on a successful proposal. Stacked legacy
        # artifacts remain newest-first for the existing folding logic.
        if (conversation and _is_archive_message(conversation[0])
                and (len(conversation) == 1 or not _is_archive_message(conversation[1]))):
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
        tail_count = min(self.protect_last_n, len(conversation))
        cut = len(conversation) - tail_count
        # Keep the latest ordinary request and all later activity live. Steering
        # and plugin artifacts do not begin a new task turn.
        for index in range(len(conversation) - 1, -1, -1):
            if _is_task_request(conversation[index]):
                cut = min(cut, index)
                break
        found = [
            _parse_artifact(message, self._frame_sources) if index < cut else None
            for index, message in enumerate(conversation)
        ]
        # Forget sources whose archive left the transcript (folded, or a
        # proposal the host rejected).
        present = {_msg_text_signature(a.message) for a in found if a is not None}
        self._frame_sources = {k: v for k, v in self._frame_sources.items() if k in present}

        fold = [a is not None and (a.complete or fold_image_only) for a in found]
        # Each artifact was inserted at the front, so they sit newest first.
        folded = [a for a, f in zip(found, fold) if f][::-1]
        body = [message for message, f in zip(conversation, fold) if not f]

        # Like Hermes' own compressor: once the session has been compacted,
        # the first turns lose their protection, or they would fossilize.
        start = 0 if any(a is not None for a in found) else self.protect_first_n
        end = max(0, len(body) - tail_count)
        for index in range(len(body) - 1, -1, -1):
            if _is_task_request(body[index]):
                end = min(end, index)
                break
        # A text serializer cannot preserve pictures, audio, or unknown blocks.
        # Protect the prefix through them, including archives that could not
        # be folded above.
        for index, message in enumerate(body[:end]):
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
        for index, message in enumerate(body):
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
            return _Plan(system, body[:end], [], body[end:], folded)
        return _Plan(system, body[:start], body[start:end], body[end:], folded)

    def _estimate(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """(host estimate, honest estimate): the latter prices frame images
        at ``frame_image_tokens`` instead of the host's learned price."""
        host = _estimate_tokens_rough(messages)
        frames = sum(
            artifact.images for artifact in (_parse_artifact(m, {}) for m in messages)
            if artifact is not None
        )
        return host, host + frames * max(0, self.frame_image_tokens - _host_image_cost())

    def _shrink_check(
        self, messages: list[dict[str, Any]], result: list[dict[str, Any]], label: str,
    ) -> tuple[int, int, int, int] | None:
        """Sizes (host in/out, honest in/out) when ``result`` shrinks by both
        measures; None (and a log line) when it would not."""
        host_in, honest_in = self._estimate(messages)
        host_out, honest_out = self._estimate(result)
        if host_out >= host_in or honest_out >= honest_in:
            logger.info(
                "%s would not shrink the transcript (~%d -> ~%d tokens; ~%d -> ~%d "
                "with frames at %d each); leaving transcript unchanged",
                label, host_in, host_out, honest_in, honest_out, self.frame_image_tokens,
            )
            return None
        return host_in, host_out, honest_in, honest_out

    # -- Summarize mode -------------------------------------------------------

    def _summarizer_reads_images(self) -> bool:
        """True only when Hermes knows the summary model takes image input."""
        if not (self._provider and self._model_id):
            return False
        try:
            from agent.image_routing import _lookup_supports_vision
            from hermes_cli.config import load_config_readonly

            return _lookup_supports_vision(
                self._provider, self._model_id, load_config_readonly(),
            ) is True
        except Exception:
            logger.debug("summarize: vision lookup failed; reading archives as text", exc_info=True)
            return False

    def _summary_request(
        self, plan: _Plan, serialized: str, focus_topic: str | None, with_images: bool,
        intro: str = _BUILTIN_INTRO,
    ) -> str | list[dict[str, Any]]:
        """The summarizer input: every folded artifact oldest first, then the
        newer conversation, then a read-only digest of the kept tail. A plain
        string unless page images are included."""
        if focus_topic:
            intro += f" Focus on preserving details about: {focus_topic}"
        if plan.folded:
            intro += (
                " Parts marked [Earlier summary] or [Archived earlier history] are older "
                "history of the same conversation, oldest first. Merge them and the newer "
                "conversation into ONE summary, and keep every item from them that is still open."
            )
        blocks: list[dict[str, Any]] = [{"type": "text", "text": intro}]
        for artifact in plan.folded:
            if artifact.kind == "summary":
                blocks.append({"type": "text", "text": f"[Earlier summary]\n{artifact.text}"})
            elif artifact.complete:
                blocks.append({"type": "text", "text": f"[Archived earlier history]\n{artifact.text}"})
            elif with_images:
                blocks.append({"type": "text", "text": (
                    "[Archived earlier history, kept as page images. Read the "
                    "guide, the edge text and every page in order.]"
                )})
                blocks.extend(artifact.blocks)
            else:
                blocks.append({"type": "text", "text": (
                    "[Archived earlier history. Only its start and end survive as text; "
                    f"the middle was kept only as {artifact.images} page images, which "
                    "cannot be read here. Say so in the summary.]\n" + artifact.text
                )})
        if serialized.strip():
            blocks.append({"type": "text", "text": (
                f"[Newer conversation]\n{serialized}" if plan.folded else serialized
            )})
        digest = _tail_digest(plan.tail)
        if digest:
            blocks.append({"type": "text", "text": digest})
        if not with_images:
            return "\n\n".join(block["text"] for block in blocks)
        merged: list[dict[str, Any]] = []
        for block in blocks:
            if block.get("type") == "text" and merged and merged[-1].get("type") == "text":
                merged[-1] = {"type": "text", "text": f"{merged[-1]['text']}\n\n{block['text']}"}
            else:
                merged.append(block)
        return merged

    def _request_tokens(self, request: str | list[dict[str, Any]]) -> int:
        """Honest size of a summarizer request."""
        if isinstance(request, str):
            return len(request) // 4
        return sum(
            self.frame_image_tokens if block.get("type") == "image_url"
            else len(str(block.get("text", ""))) // 4
            for block in request
        )

    def _compress_summarize(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
        force: bool = False,
        memory_context: str = "",
    ) -> list[dict[str, Any]]:
        """Compact via one rolling LLM prose summary."""
        plan = self._prepare_history(messages, fold_image_only=True)
        if not plan.worth_compressing(summarize=True):
            return messages

        is_anthropic = "claude" in self._model_id.lower() or self._provider == "anthropic"
        serialized = serialize_messages(plan.archive, include_thinking=not is_anthropic)
        if not serialized.strip() and not plan.folded:
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

        # Read the prompt file once per compaction, so edits apply at the next one.
        intro = self._summary_intro()

        # Archives from past processes survive only as page images. Show the
        # summarizer the images when its model takes them; otherwise (or if
        # that request fails or is too big) use their stored text.
        image_only = [a for a in plan.folded if not a.complete]
        attempts = [True, False] if image_only and self._summarizer_reads_images() else [False]
        summary = ""
        with_images = False
        for with_images in attempts:
            request = self._summary_request(plan, serialized, focus_topic, with_images, intro)
            if with_images and self._request_tokens(request) > self.threshold_tokens:
                logger.warning(
                    "summarize: request with %d archive page images (~%d tokens) is over "
                    "%d; reading their stored text only",
                    sum(a.images for a in image_only), self._request_tokens(request),
                    self.threshold_tokens,
                )
                continue
            try:
                # Host contract: PluginLlm.complete(messages) -> result with .text.
                completion = self._llm.complete([{"role": "user", "content": request}])
                summary = getattr(completion, "text", "") or ""
                if not summary.strip():
                    raise RuntimeError("LLM returned an empty summary")
                break
            except Exception:
                if with_images:
                    logger.warning(
                        "summarize: summary with archive page images failed; "
                        "retrying from their stored text", exc_info=True,
                    )
                    continue
                if not self.bridge_ready():
                    raise
                logger.exception("LLM summary failed; falling back to snapcompact mode")
                return self._compress_snapcompact(
                    messages, current_tokens, focus_topic, force, memory_context,
                )

        # Say plainly when part of the history could not be read, and keep
        # saying it in every later summary.
        gaps = [
            line for a in plan.folded if a.kind == "summary"
            for line in a.text.splitlines() if line.startswith(_GAP_PREFIX)
        ]
        if image_only and not with_images:
            gaps.append(
                f"{_GAP_PREFIX} Part of this conversation was archived only as page images "
                f"({len(image_only)} archive(s), {sum(a.images for a in image_only)} images) "
                "that could not be read back. Only the start and end of that stretch are "
                "in this summary; its middle is missing."
            )
        gaps = [line for line in dict.fromkeys(gaps) if line not in summary]
        text = (f"[snapcompact:{uuid.uuid4().hex}]\n{_SUMMARY_HEADER}"
                + _SUMMARY_BACKGROUND + summary)
        if gaps:
            text += "\n\n" + "\n".join(gaps)
        summary_msg: dict[str, Any] = {"role": "user", "content": [{"type": "text", "text": text}]}
        result = plan.system + plan.head + [summary_msg] + plan.tail

        # Same anti-growth self-check as the snapcompact path: bow out with
        # no state mutation rather than hand the host a growing transcript.
        sizes = self._shrink_check(messages, result, "summarize: generated summary")
        if sizes is None:
            return messages

        self.compression_count += 1
        self.last_prompt_tokens = -1

        logger.info(
            "snapcompact(summarize): compressed %d messages and folded %d earlier "
            "artifact(s) (%d page images, %s) into a %d-char summary (~%d -> ~%d "
            "tokens; ~%d -> ~%d with frames at %d each), #%d",
            len(plan.archive), len(plan.folded), sum(a.images for a in image_only),
            "read" if with_images else "text only", len(summary), *sizes,
            self.frame_image_tokens, self.compression_count,
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
        self._frame_sources = {}

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
        status["archive_chars"] = sum(len(text) for text in self._frame_sources.values())
        status["summary_prompt_file"] = self.summary_prompt_file
        status["frame_image_tokens"] = self.frame_image_tokens
        return status

    def has_content_to_compress(self, messages: list[dict[str, Any]]) -> bool:
        summarize = self.mode == "summarize"
        plan = self._prepare_history(messages, fold_image_only=summarize)
        return plan.worth_compressing(summarize=summarize)

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
