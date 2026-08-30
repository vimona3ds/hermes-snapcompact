# hermes-snapcompact

Snapcompact context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

When the conversation context fills up, renders old turns into dense bitmap PNG frames that vision-capable models read back at roughly **1/3 the input token cost** — with near-perfect recall. Powered by [@oh-my-pi/snapcompact](https://github.com/can1357/oh-my-pi/tree/main/packages/snapcompact).

## Requirements

- [Hermes Agent](https://github.com/NousResearch/hermes-agent) v0.5.0+
- [Bun](https://bun.sh) v1.3.14+ (for the native snapcompact renderer)
- A vision-capable model (Claude, GPT-5.x, Gemini 3.x, etc.)

## Install

```bash
hermes plugins install vimona3ds/hermes-snapcompact --enable
```

That's it. Restart Hermes. The plugin auto-configures itself — your existing compaction behavior is unchanged until you opt in.

## Usage

```
/compact-mode                  # show current mode
/compact-mode snapcompact      # switch to bitmap-frame archive
/compact-mode summarize        # switch back to LLM prose summary
```

The default mode is `summarize` — identical to the built-in compressor. When you switch to `snapcompact`, old turns are rendered into dense PNG frames instead of being summarized by an LLM. Switch back any time.

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

## Caveats

- **Vision required.** Non-vision models can't read the bitmap frames. The engine will still work (text edges are preserved verbatim) but middle history will be opaque.
- **Bun dependency.** The native renderer in @oh-my-pi/snapcompact requires Bun. The bridge auto-installs npm dependencies on first use if Bun is on PATH.
- **Image token billing.** While input tokens drop ~3x, models spend extra output/thinking tokens decoding the images. The technique shines at 100k+ token sessions.

## License

MIT — see [LICENSE](LICENSE).

The @oh-my-pi/snapcompact renderer is also [MIT-licensed](https://github.com/can1357/oh-my-pi/blob/main/packages/snapcompact/LICENSE).
