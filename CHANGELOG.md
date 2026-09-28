# Changelog

## 1.2.0

- Per-profile summary prompt: setting `plugins.entries.hermes-snapcompact.settings.summary_prompt_file` names a text file used as the summarizer's instructions instead of the built-in ones. The file is read at every compaction, so edits apply without a restart; if it is missing, unreadable or empty, a warning is logged and the built-in prompt is used. The merge rule for earlier summaries, the `[snapcompact:gap]` lines and the "Resume prior conversation" header are unchanged.
- The summarizer also sees a read-only digest of the recent turns kept verbatim after the summary (`[Kept verbatim after your summary: do not summarize or repeat]`): user and assistant text up to 300 characters each, tool results as `tool <name> (N chars)`, at most 4 KB in all. Before, it could not see an answer that sat in those turns and called the request unanswered.

## 1.1.0

- Fold, never pin: each compaction merges every earlier summary and frame archive into one rolling summary. 1.0.x kept everything before any image-bearing message, so frame archives and every summary stacked in front of them stayed in context for good.
- Find earlier summaries and archives in the transcript itself (the `[snapcompact:…]` marker, or the 1.0.0 frame opening), not only in memory, so a re-created agent, a restart or a surface switch merges instead of stacking.
- Frame archives from an earlier process: their page images go to the summarizer when Hermes says the model takes images; otherwise, or if that request fails or is too large, their stored start/end text is used and the summary says the middle was kept only as images (`[snapcompact:gap]`, carried into later summaries).
- Price frame images honestly in the plugin's own size checks and in a new preflight check before real usage exists. Setting `plugins.entries.hermes-snapcompact.settings.frame_image_tokens` (default 2500, measured on gpt-6-astra; Hermes had learned 371).
- After the first compaction the first turns are no longer protected, as in Hermes' built-in compressor. Tool calls stay with their results; user pictures, audio and unknown blocks are still kept.

## 1.0.1

- Persist the selected mode outside the plugin installation directory, and apply mode changes to active host-created engine copies.
- Reconcile compression proposals with the actual transcript so rejected attempts do not duplicate or replace accepted history.
- Retain archive state at Hermes compression boundaries; support switching between frame archives and summaries.
- Keep pre-restart images and complete tool-call groups rather than silently dropping their contents.
- Fix the Hermes summary API call, reject frame-budget overflow, and preserve ordinary text following embedded data URLs.

## 1.0.0

- Initial release.
