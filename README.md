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
cd ~/.hermes/plugins/hermes-snapcompact/bridge && bun install
```

Restart Hermes. The plugin auto-configures itself — your existing compaction behavior is unchanged until you opt in. The plugin never installs anything itself: if Bun or the bridge dependencies are missing, it flags the exact fix at startup and in `/compact-mode`, and compaction keeps working in summarize mode.

## Usage

```
/compact-mode                  # show current mode
/compact-mode snapcompact      # switch to bitmap-frame archive
/compact-mode summarize        # switch back to LLM prose summary
```

The default mode is `summarize` (LLM prose summaries). When you switch to `snapcompact`, old turns are rendered into dense PNG frames instead. Switch back any time; the choice applies to active conversations in that Hermes process. It is saved atomically to `$HERMES_HOME/plugin-data/hermes-snapcompact/mode.yaml` and restored in newly started processes. If saving fails, `/compact-mode` warns you. If the bridge is unavailable at startup, the saved `snapcompact` preference is temporarily suspended in favor of summaries; restart after repairing the bridge, or select `snapcompact` again.

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
- **Bun dependency.** The native renderer in @oh-my-pi/snapcompact requires Bun. The plugin only detects and flags a missing setup — it never installs anything.
- **Image token billing.** While input tokens drop ~3x, models spend extra output/thinking tokens decoding the images. The technique shines at 100k+ token sessions.
- **Not a lossless backup.** Long tool results and arguments are intentionally truncated; rendering normalizes whitespace and may replace unsupported characters. Keep original files and session history for exact recovery.
- **Restarted archives.** During one running session, either mode can replace its previous archive. After a process restart, existing image archives are kept verbatim because their source text is no longer in memory. Switching to `summarize` does not retroactively decode those older images.
- **Safety before savings.** Images and unsupported content blocks are kept with their surrounding prefix, and tool calls stay with their results. If frames would exceed the 80-frame budget, rendering fails without replacing the conversation. If a candidate would not shrink the rough token estimate, the conversation stays unchanged.
- **Summary fallback.** If the summary model fails or is unavailable, the plugin attempts local bitmap compaction when the bridge is ready. This is logged and still requires a vision-capable conversation model.

## Verification

Tested against Hermes v0.21.3, Bun, and the installed snapcompact renderer on macOS. Other provider APIs and operating systems have not been verified end-to-end here.

Run the regression checks using the Python environment from your Hermes installation:

```bash
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests -v
```

Renderer tests require Bun and `bridge/node_modules`; they skip if dependencies are absent. To use an existing renderer installation without installing again, set `SNAPCOMPACT_NODE_MODULES` to its absolute `node_modules` path. Tests use temporary Hermes homes and do not call a live model.

## Changes in 1.0.1

- Persist the selected mode outside the plugin installation directory, and apply mode changes to active host-created engine copies.
- Reconcile compression proposals with the actual transcript so rejected attempts do not duplicate or replace accepted history.
- Retain archive state at Hermes compression boundaries; support switching between frame archives and summaries.
- Keep pre-restart images and complete tool-call groups rather than silently dropping their contents.
- Fix the Hermes summary API call, reject frame-budget overflow, and preserve ordinary text following embedded data URLs.

## License

MIT — see [LICENSE](LICENSE).

The @oh-my-pi/snapcompact renderer is also [MIT-licensed](https://github.com/can1357/oh-my-pi/blob/main/packages/snapcompact/LICENSE).
