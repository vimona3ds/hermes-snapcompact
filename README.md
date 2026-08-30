# hermes-snapcompact

Snapcompact context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

When the conversation context fills up, instead of asking an LLM to summarize (lossy, slow, expensive), this engine renders the old turns into dense bitmap PNG frames that vision-capable models read back at roughly **1/3 the input token cost** — with near-perfect recall.

Powered by [@oh-my-pi/snapcompact](https://github.com/can1357/oh-my-pi/tree/main/packages/snapcompact), the technique behind [Stencil's write-up](https://stencil.so/blog/snapcompact). The rendering is local and deterministic: no extra LLM call, no API key, no latency beyond rasterization.

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) v0.5.0+
- [Bun](https://bun.sh) v1.3.14+ (for the native snapcompact renderer)
- A vision-capable model (Claude, GPT-5.x, Gemini 3.x, etc.)

## Install

```bash
# Clone into the Hermes context engine plugin directory
git clone https://github.com/johnytest/hermes-snapcompact.git \
    ~/.hermes/plugins/context_engine/snapcompact

# Install the rendering bridge dependencies
cd ~/.hermes/plugins/context_engine/snapcompact/bridge
bun install
```

Then set the engine in your Hermes config:

```yaml
# ~/.hermes/config.yaml
context:
  engine: snapcompact
```

Restart Hermes — the engine is active. When context hits ~75% of the model's window, old turns are archived into bitmap frames automatically.

## How it works

1. **Serialize** — old conversation turns are serialized into a compact text format (`¶user:`, `¶ai:`, `¶call:` scopes with truncated tool output).
2. **Render** — the text is rendered into dense PNG frames using @oh-my-pi/snapcompact's eval-tuned pixel fonts and provider-aware frame shapes.
3. **Resume** — the model sees a reading guide, the bitmap frames, and verbatim text at the chronological edges. It reads the archive and continues seamlessly.

Frame shapes are provider-aware and selected from SQuAD recall evals:

| Provider | Shape | Notes |
|----------|-------|-------|
| Anthropic | `11on16-bw` | 8x13 glyphs, 11px advance; high-res frames for Opus 4.7+/Fable/Mythos |
| Google | `8on22-bw` @2048px | Extra line spacing; Gemini bills fixed per-image budget |
| OpenAI | `8on22-bw` | Same line-spacing win; `detail: "original"` for patch billing |
| Unknown | `8on22-bw` | Safe default with Anthropic-style billing estimate |

## Configuration

The engine uses sensible defaults. No additional configuration beyond `context.engine: snapcompact` is needed.

The standard `compression.threshold` and `compression.protect_last_n` settings from Hermes apply — the engine reads `threshold_percent` (default 0.75) and `protect_first_n` / `protect_last_n` (default 3/6) from the context engine contract.

## Caveats

- **Vision required.** Non-vision models can't read the bitmap frames. The engine will still work (text edges are preserved verbatim) but middle history will be opaque.
- **Bun dependency.** The native renderer in @oh-my-pi/snapcompact requires Bun. Node.js is not supported.
- **Image token billing.** While input tokens drop ~3×, models spend extra output/thinking tokens decoding the images. For short contexts this can offset the savings. The technique shines at 100k+ token sessions.

## License

MIT — see [LICENSE](LICENSE).

The @oh-my-pi/snapcompact renderer is also [MIT-licensed](https://github.com/can1357/oh-my-pi/blob/main/packages/snapcompact/LICENSE).
