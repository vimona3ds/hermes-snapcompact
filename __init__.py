"""hermes-snapcompact: Snapcompact context engine plugin for Hermes.

Replaces the built-in context compressor with a bitmap-frame archival engine
powered by @oh-my-pi/snapcompact. Conversation history is rendered into dense
PNG images that vision-capable LLMs read back at roughly a third of the input
token cost — no LLM summarization call, fully local and deterministic.

Install:
    1. Clone into ~/.hermes/plugins/context_engine/snapcompact/
    2. cd bridge && bun install
    3. Set context.engine: snapcompact in ~/.hermes/config.yaml
"""

from .engine import SnapcompactEngine


def register(ctx):
    """Called by Hermes plugin discovery to register the context engine."""
    engine = SnapcompactEngine()
    ctx.register_context_engine(engine)
