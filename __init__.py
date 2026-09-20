"""hermes-snapcompact: Snapcompact context engine plugin for Hermes.

Compaction uses LLM prose summaries by default. When you're ready, type:

    /compact-mode snapcompact

to switch to bitmap-frame archival.  Switch back any time with:

    /compact-mode summarize
"""

import logging
import os
import sys

from .engine import SnapcompactEngine, VALID_MODES

logger = logging.getLogger(__name__)


def _auto_activate_engine() -> None:
    """Set context.engine: snapcompact in config.yaml if not already set.

    Only touches the one key.  If the file doesn't exist or isn't parseable,
    we skip silently — the user will just need the manual config line.
    """
    config_path = os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "config.yaml",
    )
    try:
        import yaml  # PyYAML — always available in Hermes

        if os.path.isfile(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        context = config.get("context")
        if not isinstance(context, dict):
            context = {}
            config["context"] = context

        current = context.get("engine", "compressor")
        if current != "snapcompact":
            context["engine"] = "snapcompact"
            with open(config_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            logger.info(
                "snapcompact: set context.engine: snapcompact in %s "
                "(was %r). LLM summaries remain the default — "
                "use /compact-mode snapcompact to enable bitmap rendering.",
                config_path,
                current,
            )
    except Exception:
        logger.debug(
            "snapcompact: could not auto-set context.engine in %s; "
            "set it manually if compression doesn't activate.",
            config_path,
            exc_info=True,
        )


def register(ctx):
    """Called by Hermes plugin discovery."""
    engine = SnapcompactEngine()
    engine.set_llm(getattr(ctx, "llm", None))
    ctx.register_context_engine(engine)

    _auto_activate_engine()

    # Probe the rendering bridge NOW, at plugin load — detection only, we
    # never install anything.  logger goes to the log file; print() makes
    # sure the user actually sees it in the terminal at startup.
    ready, detail = engine.ensure_ready()
    if not ready:
        notice = (
            f"[snapcompact] bitmap rendering unavailable: {detail}\n"
            f"[snapcompact] compaction still works (summarize mode); "
            f"/compact-mode snapcompact is disabled until fixed."
        )
        if engine.mode == "snapcompact":
            # Suspend in-memory only — keep the persisted opt-in so it
            # comes back automatically once the bridge is fixed.
            engine.mode = "summarize"
            notice += (
                "\n[snapcompact] persisted snapcompact mode suspended "
                "for this run."
            )
        print(notice, file=sys.stderr)
        logger.warning(notice)

    # -- /compact-mode --------------------------------------------------------

    def handle_compact_mode(args):
        arg = args.strip().lower() if args else ""

        if not arg:
            ok, bridge_detail = engine.ensure_ready()
            bridge_note = "" if ok else (
                f"\n\n⚠️ snapcompact mode unavailable: {bridge_detail}"
            )
            return (
                f"Current compaction mode: **{engine.mode}**\n\n"
                f"Usage: `/compact-mode <mode>`\n\n"
                f"  `snapcompact` — bitmap-frame archive (local, no LLM call, ~1/3 tokens)\n"
                f"  `summarize`   — LLM prose summary (current Hermes default)"
                f"{bridge_note}"
            )

        if arg not in VALID_MODES:
            return (
                f"Unknown mode `{arg}`. "
                f"Choose: {', '.join(f'`{m}`' for m in VALID_MODES)}"
            )

        if arg == "snapcompact":
            ok, bridge_detail = engine.ensure_ready()
            if not ok:
                return (
                    f"Cannot switch to `snapcompact`: {bridge_detail}\n"
                    f"Staying on `{engine.mode}`."
                )

        old = engine.mode
        saved = engine.set_mode(arg)
        notice = "" if saved else (
            "\nWarning: the mode changed for this process, but could not be saved. "
            "It may revert after restart; check the Hermes log for the filesystem error."
        )
        if old == arg:
            return f"Already using `{arg}`.{notice}"
        return f"Compaction mode: `{old}` → **`{arg}`**.{notice}"

    ctx.register_command(
        "compact-mode",
        handle_compact_mode,
        "Switch compaction strategy (snapcompact / summarize)",
    )
