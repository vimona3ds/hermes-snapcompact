"""hermes-snapcompact: Snapcompact context engine plugin for Hermes.

Dual-mode context engine powered by @oh-my-pi/snapcompact:

- **snapcompact** (default) — renders old turns into dense bitmap PNG
  frames that vision models read back at ~1/3 the token cost.  Local,
  deterministic, no LLM call.
- **summarize** — classic LLM prose summary via the host model.

Switch modes any time with ``/compact-mode``.

Install:
    1. Clone into ~/.hermes/plugins/context_engine/snapcompact/
    2. cd bridge && bun install
    3. Set context.engine: snapcompact in ~/.hermes/config.yaml
"""

from .engine import SnapcompactEngine

_VALID_MODES = ("snapcompact", "summarize")


def register(ctx):
    """Called by Hermes plugin discovery to register the context engine."""
    llm = getattr(ctx, "llm", None)
    engine = SnapcompactEngine(llm=llm)
    ctx.register_context_engine(engine)

    # -- Slash command: /compact-mode ----------------------------------------

    def handle_compact_mode(args):
        """Switch or display the active compaction mode."""
        arg = args.strip().lower() if args else ""

        if not arg:
            return (
                f"Current compaction mode: **{engine.mode}**\n\n"
                f"Usage: `/compact-mode <mode>`\n"
                f"Available modes: {', '.join(f'`{m}`' for m in _VALID_MODES)}\n\n"
                f"- `snapcompact` — bitmap-frame archive (local, no LLM call)\n"
                f"- `summarize` — LLM prose summary (uses active model)"
            )

        if arg not in _VALID_MODES:
            return (
                f"Unknown mode `{arg}`. "
                f"Choose from: {', '.join(f'`{m}`' for m in _VALID_MODES)}"
            )

        old = engine.mode
        engine.mode = arg
        if old == arg:
            return f"Compaction mode already set to `{arg}`."
        return f"Compaction mode switched: `{old}` → `{arg}`."

    ctx.register_command(
        "compact-mode",
        handle_compact_mode,
        "Switch compaction strategy (snapcompact / summarize)",
    )
