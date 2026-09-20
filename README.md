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
- Fails safe: if archiving wouldn't shrink the context, nothing changes.

## Development

```bash
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests -v
```

Renderer tests skip without Bun and `bridge/node_modules`. See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT — see [LICENSE](LICENSE). The snapcompact renderer is also [MIT-licensed](https://github.com/can1357/oh-my-pi/blob/main/packages/snapcompact/LICENSE).
