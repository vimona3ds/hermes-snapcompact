# Changelog

## 1.0.1

- Persist the selected mode outside the plugin installation directory, and apply mode changes to active host-created engine copies.
- Reconcile compression proposals with the actual transcript so rejected attempts do not duplicate or replace accepted history.
- Retain archive state at Hermes compression boundaries; support switching between frame archives and summaries.
- Keep pre-restart images and complete tool-call groups rather than silently dropping their contents.
- Fix the Hermes summary API call, reject frame-budget overflow, and preserve ordinary text following embedded data URLs.

## 1.0.0

- Initial release.
