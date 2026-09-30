# hermes-snapcompact

Context engine plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent): when the conversation fills up, old turns are rendered into dense bitmap PNG frames that vision models read back at ~1/3 the input token cost. Powered by [@oh-my-pi/snapcompact](https://github.com/can1357/oh-my-pi/tree/main/packages/snapcompact).

## Install

```bash
hermes plugins install vimona3ds/hermes-snapcompact --enable
cd ~/.hermes/plugins/hermes-snapcompact/bridge && bun install
```

Needs Hermes v0.5.0+, [Bun](https://bun.sh) v1.3.14+, and a vision-capable model. Restart Hermes — if anything is missing, startup prints the fix.

## Use

```
/compact-mode snapcompact   # bitmap-frame archive
/compact-mode summarize     # LLM prose summary (default)
/compact-mode               # show current mode
```

Your choice persists across restarts.

## Good to know

- Non-vision models can't read the frames.
- Lossy: long tool output is truncated. Keep originals for exact recovery.
- Pays off most in long sessions (~100k+ tokens).
- Successful compaction preserves chronological order: opening messages, archived
  history, then the live user request and its later corrections. The entire latest
  task turn stays live. Opening turns can join the rolling archive on later
  compactions.
- Fails safe: with nothing eligible to archive or no token savings, the original
  transcript stays unchanged, including any legacy archive-before-opening layout.
- One rolling summary: each compaction folds earlier summaries and frame archives into the new one, also after a restart. Old frame archives are read as images if the model takes images; otherwise the summary says their middle is missing.
- Frame images cost more than Hermes estimates. The plugin prices each at `frame_image_tokens` (default 2500); override in `config.yaml` under `plugins.entries.hermes-snapcompact.settings`.
- Your own summary prompt: set `summary_prompt_file: /path/to/prompt.md` in the same place (per profile). It replaces the built-in instructions and is re-read at every compaction; if it can't be read, the built-in prompt is used and a warning is logged. The summarizer also sees a short (≤ 4 KB) digest of the recent turns kept after the summary, so it doesn't repeat them or call them unanswered.

## Development

```bash
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests -v
```

Renderer tests skip without Bun and `bridge/node_modules`. See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT — see [LICENSE](LICENSE). The snapcompact renderer is also [MIT-licensed](https://github.com/can1357/oh-my-pi/blob/main/packages/snapcompact/LICENSE).
