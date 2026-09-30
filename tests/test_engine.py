"""Regression checks using Hermes' real ContextEngine and PluginLlm contracts.

Run with Hermes' Python environment and its checkout on PYTHONPATH.
Only the renderer and external model call are replaced; no network or user state.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent.plugin_llm import PluginLlm


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "snapcompact_test_plugin", ROOT / "__init__.py",
    submodule_search_locations=[str(ROOT)],
)
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
engine_module = sys.modules[f"{spec.name}.engine"]


def history(tag="FIRST"):
    return [{"role": "system", "content": "System rules"}] + [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": f"{tag}_{i}: " + "Detailed conversation material. " * 800}
        for i in range(14)
    ]


def extend(messages, tag="NEXT"):
    return messages + history(tag)[1:]


def images(messages):
    return [b for m in messages if isinstance(m.get("content"), list)
            for b in m["content"] if b.get("type") == "image_url"]


def task_history():
    """Completed task, fresh task, long tool loop, then an in-flight correction."""
    from agent.prompt_builder import steer_user_row
    messages = history()
    messages[1]['content'] = 'COMPLETED: record sleep'
    messages[2]['content'] = 'Sleep recorded; that task is finished.'
    active = [{'role': 'user', 'content': 'CURRENT: audit installed software'}]
    for i in range(12):
        active += [
            {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': f'audit-{i}', 'type': 'function',
                'function': {'name': 'inspect', 'arguments': '{}'},
            }]},
            {'role': 'tool', 'tool_call_id': f'audit-{i}', 'content': f'audit result {i}'},
        ]
    active += [steer_user_row('Include OpenMuse in the audit.')]
    return messages + active, active


def archive_message(messages):
    return next(m for m in messages if '[snapcompact:' in engine_module._msg_text_signature(m))


PAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}}


def legacy_summary(n):
    """A 1.0.1 prose summary as stored in the transcript."""
    return {"role": "user", "content": [{"type": "text", "text": (
        f"[snapcompact:{n:032x}]\nResume prior conversation. Summary of earlier context:\n\n"
        f"SUMMARY_{n:02d} " + "Summary detail. " * 300
    )}]}


def legacy_frames(n, pages):
    """A 1.0.0 frame archive (no marker line) from a process that has since exited."""
    guide = engine_module._SUMMARY_TEMPLATE.format(
        image_guide=engine_module._IMAGE_GUIDE_TEMPLATE.format(cols=175, rows=120))
    return {"role": "user", "content": [
        {"type": "text", "text": guide},
        {"type": "text", "text": f"FRAME_{n}_HEAD " + "edge " * 300 + "\n-------------- imaged middle below\n"},
        *[copy.deepcopy(PAGE) for _ in range(pages)],
        {"type": "text", "text": "-------------- imaged middle above\n" + "edge " * 300 + f" FRAME_{n}_TAIL"},
    ]}


def tool_call(*ids):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": i, "type": "function", "function": {"name": "place_proposal", "arguments": "{}"}} for i in ids]}


def tool_result(call_id):
    return {"role": "tool", "tool_call_id": call_id, "content": f"RESULT_{call_id} " + "payload " * 400}


def leftovers():
    """The #ada shape: 13 stacked summaries and 4 pinned frame archives
    (34 pages), newest first, then a short live tail."""
    return (
        [{"role": "system", "content": "System rules"}]
        + [legacy_summary(n) for n in range(13, 0, -1)]
        + [legacy_frames(n, pages) for n, pages in ((4, 9), (3, 9), (2, 10), (1, 6))]
        + [{"role": "user", "content": "OLDER_TURN " + "words " * 400},
           tool_call("a"), tool_result("a"),
           tool_call("b", "c", "d"), tool_result("b"), tool_result("c"), tool_result("d"),
           {"role": "user", "content": "LATEST_ASK"}]
    )


def artifacts(messages):
    return [m for m in messages if engine_module._parse_artifact(m, {}) is not None]


def text_of(message):
    content = message["content"]
    return content if isinstance(content, str) else "\n".join(
        b.get("text", "") for b in content if b.get("type") == "text")


def assert_tool_pairs_intact(test, messages):
    ids = {tc["id"] for m in messages for tc in m.get("tool_calls") or []}
    results = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    test.assertEqual(ids, results)


class EngineTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.enterContext(patch.dict(os.environ, HERMES_HOME=home.name))
        self.rendered = []
        self.prompts = []
        self.requests = []
        self.enterContext(patch.object(engine_module, "_call_bridge", self.render))
        self.enterContext(patch.object(engine_module.SnapcompactEngine, "ensure_ready", return_value=(True, "ready")))
        self.engine = engine_module.SnapcompactEngine()
        self.engine.mode = "snapcompact"
        # Exercise the real facade, including messages -> response.text conversion.
        self.engine.set_llm(PluginLlm(plugin_id="hermes-snapcompact", sync_caller=lambda **kw: self.complete(**kw)))

    def render(self, request, **kwargs):
        self.rendered.append(request["text"])
        return {"images": [{"data": "cG5n", "mimeType": "image/png"}],
                "shape": {}, "geometry": {"cols": 175, "rows": 120}}

    def complete(self, **kwargs):
        content = kwargs["messages"][0]["content"]
        self.requests.append(content)
        prompt = content if isinstance(content, str) else "\n".join(
            b.get("text", "") for b in content if b.get("type") == "text")
        self.prompts.append(prompt)
        # Distinct facts, not the whole prompt, survive a lossy summary.
        facts = [name for name in ("FIRST_5", "NEXT_5", "THIRD_5") if name in prompt]
        return "test-provider", "test-model", SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recorded facts: " + ", ".join(facts)))],
            model="test-model", usage=None,
        )

    def test_current_request_and_later_steer_stay_live_after_history(self):
        source, active = task_history()
        before = copy.deepcopy(source)
        for mode in ('snapcompact', 'summarize'):
            with self.subTest(mode=mode):
                self.engine.mode = mode
                out = self.engine.compress(source)
                self.assertEqual(out[:4], source[:4])
                self.assertEqual(out[-len(active):], active)
                self.assertEqual(out[4], archive_message(out))
                self.assertLess(len(out), len(source))
        self.assertEqual(source, before)

    def test_active_only_turn_is_not_archived(self):
        _, active = task_history()
        for mode in ('snapcompact', 'summarize'):
            with self.subTest(mode=mode):
                self.engine.mode = mode
                self.assertFalse(self.engine.has_content_to_compress(active))
                self.assertIs(self.engine.compress(active), active)
        self.assertFalse(self.rendered)
        self.assertFalse(self.prompts)

    def test_plugin_handoffs_are_not_new_user_requests(self):
        for prefix in ('', '[snapcompact:0123456789abcdef0123456789abcdef]\n'):
            for guide in ('Resume prior conversation. Earlier turns archived under HISTORY below,',
                          'Resume prior conversation. Summary of earlier context:'):
                for mode in ('snapcompact', 'summarize'):
                    with self.subTest(prefix=prefix, guide=guide, mode=mode):
                        self.engine.mode = mode
                        source = [{'role': 'user', 'content': 'CURRENT task'}]
                        source += [{'role': 'assistant', 'content': 'tool progress ' * 2000}] * 10
                        source += [{'role': 'user', 'content': [{'type': 'text', 'text': prefix + guide}]}]
                        source += [{'role': 'assistant', 'content': 'more progress'}] * 10
                        self.assertFalse(self.engine.has_content_to_compress(source))
                        self.assertIs(self.engine.compress(source), source)

    def test_repeated_handoff_preserves_current_request_and_does_not_nest_archive(self):
        for mode in ('snapcompact', 'summarize'):
            with self.subTest(mode=mode):
                self.engine.mode = mode
                first = self.engine.compress(history())
                source, active = task_history()
                out = self.engine.compress(first + source[1:])
                self.assertEqual(out[-len(active):], active)
                self.assertEqual(out[:1], first[:1])
                self.assertEqual(len(artifacts(out)), 1)
                retired = self.rendered[-1] if mode == 'snapcompact' else self.prompts[-1]
                self.assertNotIn('[snapcompact:', retired)
                self.assertNotIn('CURRENT: audit installed software', retired.split(engine_module._TAIL_MARK)[0])

    def test_fresh_engine_after_native_persistence_keeps_bitmap_and_active_request(self):
        from hermes_state import SessionDB
        source, active = task_history()
        first = self.engine.compress(history())
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        db = SessionDB(Path(tmp) / 'handoff.db')
        self.addCleanup(db.close)
        db.create_session('handoff', 'test')
        db.append_messages_batch('handoff', first + source[1:])
        restored = db.get_messages_as_conversation('handoff')
        self.assertEqual([(m['role'], m['content']) for m in restored[-len(active):]],
                         [(m['role'], m['content']) for m in active])
        for mode in ('snapcompact', 'summarize'):
            with self.subTest(mode=mode):
                fresh = engine_module.SnapcompactEngine()
                fresh.mode = mode
                fresh.set_llm(self.engine._llm)
                out = fresh.compress(restored)
                self.assertEqual(out[-len(active):], restored[-len(active):])
                if mode == 'snapcompact':
                    self.assertIn(archive_message(restored), out)
                    self.assertEqual(out[:4], restored[:4])
                    self.assertEqual(images(out)[:1], images(first))
                else:
                    self.assertEqual(len(artifacts(out)), 1)
                    self.assertFalse(images(out))
                    self.assertIn("[snapcompact:gap]", text_of(archive_message(out)))

    def test_legacy_leading_archive_stays_after_original_head_on_fresh_engine(self):
        first = self.engine.compress(history())
        source, active = task_history()
        for with_marker in (True, False):
            archive = copy.deepcopy(archive_message(first))
            if not with_marker:
                archive['content'][0]['text'] = archive['content'][0]['text'].split('\n', 1)[1]
            legacy = first[:1] + [archive] + first[1:4] + first[5:] + source[1:]
            for mode in ('snapcompact', 'summarize'):
                with self.subTest(with_marker=with_marker, mode=mode):
                    fresh = engine_module.SnapcompactEngine()
                    fresh.mode = mode
                    fresh.set_llm(self.engine._llm)
                    out = fresh.compress(legacy)
                    self.assertEqual(out[-len(active):], active)
                    if mode == 'snapcompact':
                        self.assertEqual(out[:4], first[:4])
                        self.assertEqual(out[4], archive)
                        self.assertEqual(images(out)[:1], images([archive]))
                    else:
                        self.assertEqual(len(artifacts(out)), 1)
                        self.assertFalse(images(out))
                        self.assertIn("[snapcompact:gap]", text_of(archive_message(out)))

    def test_legacy_archive_without_savings_leaves_original_transcript_intact(self):
        first = self.engine.compress(history())
        legacy = first[:1] + [archive_message(first)] + first[1:4] + first[5:]
        before = copy.deepcopy(legacy)
        render_count = len(self.rendered)
        for mode in ("snapcompact", "summarize"):
            with self.subTest(mode=mode):
                fresh = engine_module.SnapcompactEngine()
                fresh.mode = mode
                fresh.set_llm(self.engine._llm)
                with patch.object(fresh, "_shrink_check", return_value=None):
                    self.assertIs(fresh.compress(legacy), legacy)
                self.assertEqual(legacy, before)
        self.assertEqual(len(self.rendered), render_count)

    def test_fold_only_places_archive_before_long_active_turn(self):
        _, active = task_history()
        for mode in ('snapcompact', 'summarize'):
            with self.subTest(mode=mode):
                fresh = self.summarizer() if mode == 'summarize' else self.engine
                fresh.mode = mode
                older = [legacy_summary(1), legacy_summary(2)]
                for message in older:
                    message['content'][0]['text'] += 'Archived material. ' * 3000
                source = older + active
                out = fresh.compress(source)
                self.assertEqual(out[-len(active):], active)
                self.assertEqual(len(artifacts(out)), 1)
                self.assertEqual(out[0], archive_message(out))

    def test_repeated_frames_replace_predecessor(self):
        first = self.engine.compress(history())
        second = self.engine.compress(extend(first))
        self.assertEqual(len(images(second)), 1)
        self.assertEqual(self.rendered[-1].count("FIRST_5:"), 1)
        self.assertIn("NEXT_5:", self.rendered[-1])

    def test_both_mode_switches_preserve_earlier_facts(self):
        first = self.engine.compress(history())
        self.engine.mode = "summarize"
        second = self.engine.compress(extend(first))
        self.assertIn("FIRST_5:", self.prompts[-1])
        self.assertFalse(images(second))
        self.engine.mode = "snapcompact"
        third = self.engine.compress(extend(second, "THIRD"))
        self.assertEqual(len(images(third)), 1)
        self.assertIn("FIRST_5", self.rendered[-1])
        self.assertIn("NEXT_5", self.rendered[-1])

    def test_real_host_compression_boundary_retains_archive(self):
        first = self.engine.compress(history())
        self.engine.on_session_start("same-session", boundary_reason="compression", old_session_id="same-session")
        self.engine.mode = "summarize"
        second = self.engine.compress(extend(first))
        self.assertIn("FIRST_5:", self.prompts[-1])
        self.assertFalse(images(second))

    def test_rejected_candidate_then_mode_switch_keeps_original_facts(self):
        committed = self.engine.compress(history())
        original = extend(committed)
        self.engine.mode = "summarize"
        self.engine.compress(original)  # Host rejects this proposed summary.
        self.engine.mode = "snapcompact"
        self.engine.compress(extend(original, "THIRD"))
        self.assertIn("FIRST_5:", self.rendered[-1])
        self.assertEqual(self.rendered[-1].count("NEXT_5:"), 1)

    def test_identical_later_messages_are_not_mistaken_for_retry(self):
        source = history()
        first = self.engine.compress(source)
        # A genuinely later identical batch, following a committed artifact.
        new_messages = [source[0], archive_message(first)] + copy.deepcopy(source[1:])
        self.engine.compress(new_messages)
        self.assertEqual(self.rendered[-1].count("FIRST_5:"), 2)

    def test_mode_switch_reaches_active_host_clones(self):
        active = copy.deepcopy(self.engine)
        self.engine.set_mode("summarize")
        out = active.compress(history())
        self.assertFalse(images(out))
        self.assertIn("FIRST_5", engine_module.serialize_messages(out))

    def test_cloned_sessions_do_not_share_history(self):
        other = copy.deepcopy(self.engine)
        self.engine.compress(history())
        other.compress(history("OTHER"))
        self.assertNotIn("FIRST_5:", self.rendered[-1])

    def test_rejected_candidate_retries_do_not_duplicate(self):
        source = history()
        self.engine.compress(source)
        self.engine.compress(source)  # Original input means candidate rejected.
        self.assertEqual(self.rendered[-1].count("FIRST_5:"), 1)

    def test_foreign_images_survive_middle_of_transcript(self):
        source = history()
        picture = {"role": "user", "content": [
            {"type": "text", "text": "Only copy of the earlier archive"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,b2xk"}},
        ]}
        source.insert(5, picture)
        for mode in ("snapcompact", "summarize"):
            with self.subTest(mode=mode):
                engine = engine_module.SnapcompactEngine()
                engine.mode = mode
                engine.set_llm(self.engine._llm)
                out = engine.compress(source)
                self.assertIn(picture, out)

    def test_tool_call_and_result_are_kept_together(self):
        source = history()
        call = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ]}
        result = {"role": "tool", "tool_call_id": "call-1", "content": "Important file contents"}
        source[3:3] = [call, result]  # Initial protected head ends inside the pair.
        source[-5:-5] = [copy.deepcopy(call), copy.deepcopy(result)]
        source[-7]["tool_calls"][0]["id"] = "call-2"
        source[-6]["tool_call_id"] = "call-2"
        out = self.engine.compress(source)
        ids = {tc["id"] for m in out for tc in m.get("tool_calls", [])}
        for m in out:
            if m["role"] == "tool":
                self.assertIn(m["tool_call_id"], ids)
        for call_id in ids:
            self.assertTrue(any(m.get("tool_call_id") == call_id for m in out))

    def test_failure_does_not_mutate_transcript(self):
        source = history()
        before = copy.deepcopy(source)
        with patch.object(engine_module, "_call_bridge", side_effect=RuntimeError("renderer failed")):
            self.assertIs(self.engine.compress(source), source)
        self.assertEqual(source, before)
        self.engine.compress(source)
        self.assertEqual(self.rendered[-1].count("FIRST_5:"), 1)

    def test_small_conversation_and_growth_leave_input_intact(self):
        source = history()[:6]
        self.assertIs(self.engine.compress(source), source)
        source = [{"role": "user", "content": str(i)} for i in range(15)]
        self.assertIs(self.engine.compress(source), source)

    def test_reset_does_not_leak_previous_session(self):
        self.engine.compress(history())
        self.engine.on_session_reset()
        self.engine.compress(history("OTHER"))
        self.assertNotIn("FIRST_5:", self.rendered[-1])

    def test_host_message_repair_does_not_strand_summary(self):
        from agent.agent_runtime_helpers import repair_message_sequence
        self.engine.mode = "summarize"
        first = self.engine.compress(history())
        repair_message_sequence(None, first)
        self.engine.mode = "snapcompact"
        out = self.engine.compress(extend(first))
        self.assertIn("Recorded facts: FIRST_5", self.rendered[-1])
        self.assertEqual(len(images(out)), 1)

    def test_summary_failure_falls_back_without_losing_archive(self):
        first = self.engine.compress(history())
        self.engine.mode = "summarize"
        with patch.object(self.engine._llm, "complete", side_effect=RuntimeError("provider unavailable")):
            with self.assertLogs(engine_module.logger, level="ERROR"):
                out = self.engine.compress(extend(first))
        self.assertEqual(len(images(out)), 1)
        self.assertEqual(self.rendered[-1].count("FIRST_5:"), 1)

    def test_empty_summary_does_not_replace_history(self):
        self.engine.mode = "summarize"
        with patch.object(self.engine._llm, "complete", return_value=SimpleNamespace(text=" ")):
            with self.assertLogs(engine_module.logger, level="ERROR"):
                out = self.engine.compress(history())
        self.assertEqual(len(images(out)), 1)
        self.assertIn("FIRST_5:", self.rendered[-1])

    def test_unavailable_summary_and_renderer_fail_without_mutation(self):
        self.engine.mode = "summarize"
        source = history()
        before = copy.deepcopy(source)
        with patch.object(self.engine._llm, "complete", side_effect=RuntimeError("provider unavailable")):
            with patch.object(self.engine, "ensure_ready", return_value=(False, "missing")):
                with self.assertRaises(RuntimeError):
                    self.engine.compress(source)
        self.assertEqual(source, before)

    # -- Leftover archives (1.0.x pinned and stacked them) --------------------

    def summarizer(self):
        """A fresh engine, as after agent re-creation or a surface switch."""
        engine = engine_module.SnapcompactEngine()
        engine.mode = "summarize"
        engine.set_llm(self.engine._llm)
        engine.update_model("gpt-test", 272_000, provider="openai-codex")
        return engine

    def vision(self, supported):
        return patch("agent.image_routing._lookup_supports_vision", return_value=supported)

    def test_leftover_archives_fold_into_one_summary(self):
        source = leftovers()
        with self.vision(False):
            out = self.summarizer().compress(source)
            self.assertFalse(self.summarizer().has_content_to_compress(out))
        self.assertEqual(len(artifacts(out)), 1)
        self.assertFalse(images(out))
        # System, one summary, and the live tail verbatim; the tail cut moved
        # back so the first call keeps its result.
        self.assertEqual(out[0], source[0])
        self.assertEqual(out[2:], source[-7:])
        assert_tool_pairs_intact(self, out)
        # Everything reached the summarizer, oldest first.
        prompt = self.prompts[-1]
        order = ["FRAME_1_HEAD", "FRAME_1_TAIL", "FRAME_4_TAIL", "SUMMARY_01", "SUMMARY_13", "OLDER_TURN"]
        positions = [prompt.index(key) for key in order]
        self.assertEqual(positions, sorted(positions))
        # The unreadable middles are disclosed, not silently dropped.
        summary = text_of(out[1])
        self.assertIn("[snapcompact:gap]", summary)
        self.assertIn("34 images", summary)

    def test_archive_pages_go_to_a_summarizer_that_reads_images(self):
        with self.vision(True):
            out = self.summarizer().compress(leftovers())
        self.assertEqual(sum(b.get("type") == "image_url" for b in self.requests[-1]), 34)
        self.assertEqual(len(artifacts(out)), 1)
        self.assertFalse(images(out))
        self.assertNotIn("[snapcompact:gap]", text_of(out[1]))

    def test_rejected_pages_fall_back_to_stored_text(self):
        def text_only(**kwargs):
            if not isinstance(kwargs["messages"][0]["content"], str):
                raise RuntimeError("model rejected image input")
            return self.complete(**kwargs)
        engine = self.summarizer()
        engine.set_llm(PluginLlm(plugin_id="hermes-snapcompact", sync_caller=text_only))
        with self.vision(True), self.assertLogs(engine_module.logger, level="WARNING"):
            out = engine.compress(leftovers())
        self.assertEqual(len(artifacts(out)), 1)
        self.assertIn("[snapcompact:gap]", text_of(out[1]))

    def test_gap_note_survives_later_summaries(self):
        with self.vision(False):
            first = self.summarizer().compress(leftovers())
            second = self.summarizer().compress(extend(first))
        self.assertEqual(len(artifacts(second)), 1)
        self.assertIn("[snapcompact:gap]", text_of(second[1]))

    def test_recreated_agents_merge_summaries_instead_of_stacking(self):
        out = self.summarizer().compress(history())
        for tag in ("NEXT", "THIRD"):
            out = self.summarizer().compress(extend(out, tag))
            self.assertEqual(len(artifacts(out)), 1)
        self.assertIn("[Earlier summary]\nRecorded facts: FIRST_5, NEXT_5", self.prompts[-1])
        self.assertIn("Recorded facts: FIRST_5, NEXT_5, THIRD_5", text_of(out[1]))

    def test_user_picture_stays_while_old_summaries_fold(self):
        source = leftovers()
        picture = {"role": "user", "content": [{"type": "text", "text": "Photo of the receipt"}, copy.deepcopy(PAGE)]}
        source.insert(18, picture)  # right after the archives
        with self.vision(True):
            out = self.summarizer().compress(source)
        self.assertIn(picture, out)
        self.assertEqual(len(artifacts(out)), 1)
        assert_tool_pairs_intact(self, out)

    # -- Summary prompt file and the kept-tail digest -----------------------------

    def prompt_file(self, text):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        path = Path(folder.name) / "compact-prompt.md"
        path.write_text(text, encoding="utf-8")
        return path

    def assert_one_summary(self, out):
        self.assertEqual(len(artifacts(out)), 1)
        self.assertIn(engine_module._SUMMARY_HEADER, text_of(archive_message(out)))

    def test_prompt_file_is_read_at_every_compaction(self):
        path = self.prompt_file("PROMPT_V1: write Ada's handoff.\n")
        engine = self.summarizer()
        engine.set_summary_prompt_file(str(path))
        out = engine.compress(history())
        self.assertTrue(self.prompts[-1].startswith("PROMPT_V1"))
        self.assertNotIn(engine_module._BUILTIN_INTRO, self.prompts[-1])
        self.assert_one_summary(out)
        # An edit applies at the next compaction, with no new engine.
        path.write_text("PROMPT_V2: write Ada's handoff.\n", encoding="utf-8")
        out = engine.compress(extend(out))
        prompt = self.prompts[-1]
        self.assertTrue(prompt.startswith("PROMPT_V2"))
        self.assertNotIn("PROMPT_V1", prompt)
        # The merge rule still follows the file's text, then the earlier summary.
        self.assertLess(prompt.index("Merge them"), prompt.index("[Earlier summary]\n"))
        self.assert_one_summary(out)
        self.assertIn("Recorded facts: FIRST_5, NEXT_5", text_of(out[1]))

    def test_missing_or_empty_prompt_file_falls_back_to_the_built_in_prompt(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        empty = Path(folder.name) / "empty.md"
        empty.write_text("  \n", encoding="utf-8")
        for path in (Path(folder.name) / "missing.md", empty, Path(folder.name)):
            with self.subTest(path=path.name):
                engine = self.summarizer()
                engine.set_summary_prompt_file(str(path))
                with self.assertLogs(engine_module.logger, level="WARNING"):
                    out = engine.compress(history())
                self.assertTrue(self.prompts[-1].startswith(engine_module._BUILTIN_INTRO))
                self.assert_one_summary(out)

    def test_prompt_file_removed_between_compactions_falls_back(self):
        path = self.prompt_file("PROMPT_V1: write Ada's handoff.\n")
        engine = self.summarizer()
        engine.set_summary_prompt_file(str(path))
        out = engine.compress(history())
        path.unlink()
        with self.assertLogs(engine_module.logger, level="WARNING"):
            out = engine.compress(extend(out))
        self.assertTrue(self.prompts[-1].startswith(engine_module._BUILTIN_INTRO))
        self.assert_one_summary(out)

    def test_summarizer_sees_a_bounded_digest_of_the_kept_tail(self):
        big = {"role": "tool", "tool_call_id": "big", "content": "BIG_RESULT " + "x" * 100_000}
        source = history() + [
            tool_call("big"), big,
            {"role": "assistant", "content": "ANSWER_SENT " + "Here is today's plan. " * 100},
            {"role": "user", "content": "LATEST_ASK"},
        ]
        out = self.summarizer().compress(source)
        self.assert_one_summary(out)
        self.assertEqual(out[-6:], source[-6:])  # the tail itself stays verbatim
        prompt = self.prompts[-1]
        mark = "[Kept verbatim after your summary: do not summarize or repeat]"
        self.assertEqual(prompt.count(mark), 1)
        digest = prompt[prompt.index(mark):]
        self.assertLessEqual(len(digest.encode("utf-8")), 4096)
        self.assertNotIn("BIG_RESULT", prompt)
        rows = digest.splitlines()[1:]
        self.assertIn(f"tool place_proposal ({len(big['content'])} chars)", rows)
        self.assertIn("user: LATEST_ASK", rows)
        self.assertTrue(any(row.startswith("assistant: ANSWER_SENT") for row in rows))
        for row in rows:
            role, _, text = row.partition(": ")
            if role in ("user", "assistant"):
                self.assertLessEqual(len(text), 300)
        # The archived slice is still summarized in full, before the digest.
        self.assertLess(prompt.index("FIRST_5:"), prompt.index(mark))

    def test_tail_digest_stays_under_4_kb_however_long_the_tail(self):
        tail = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"TURN_{i} " + "é" * 5_000}
                for i in range(40)]
        digest = engine_module._tail_digest(tail)
        self.assertLessEqual(len(digest.encode("utf-8")), 4096)
        self.assertIn("TURN_0 ", digest)
        self.assertNotIn("TURN_39 ", digest)
        self.assertRegex(digest.splitlines()[-1], r"^\(\d+ more kept messages not shown\)$")
        self.assertEqual(engine_module._tail_digest([]), "")

    # -- Frame image price ------------------------------------------------------

    def test_preflight_counts_frame_pages_at_their_real_price(self):
        engine = self.summarizer()
        source = leftovers()
        host = engine_module._estimate_tokens_rough(source)
        underpriced = 34 * (engine.frame_image_tokens - engine_module._host_image_cost())
        engine.threshold_tokens = host + underpriced // 2
        self.assertFalse(engine.should_compress(host))
        self.assertTrue(engine.should_compress_preflight(source))
        # Once real usage exists it already includes the true image cost.
        engine.update_from_response({"prompt_tokens": host})
        self.assertFalse(engine.should_compress_preflight(source))

    def test_frames_that_really_cost_more_than_their_text_are_not_committed(self):
        pages = {"images": [{"data": "cG5n", "mimeType": "image/png"}] * 16,
                 "shape": {}, "geometry": {"cols": 175, "rows": 120}}
        source = history()
        with patch.object(engine_module, "_call_bridge", return_value=pages):
            self.assertIs(self.engine.compress(source), source)
            # At the host's per-image price the same frames look like a saving.
            self.engine.set_frame_image_tokens(engine_module._host_image_cost())
            self.assertIsNot(self.engine.compress(source), source)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.home = Path(home.name)
        self.enterContext(patch.dict(os.environ, HERMES_HOME=home.name))

    def test_default_and_both_restart_choices(self):
        self.assertEqual(engine_module.SnapcompactEngine().mode, "summarize")
        engine = engine_module.SnapcompactEngine()
        for mode in ("snapcompact", "summarize"):
            engine.set_mode(mode)
            self.assertEqual(engine_module.SnapcompactEngine().mode, mode)
        self.assertFalse((self.home / "plugins").exists())

    def test_corrupt_state_safely_falls_back(self):
        path = engine_module._mode_state_path()
        for content in ("mode: [", "mode: impossible", "[]", "", "mode: null"):
            with self.subTest(content=content):
                path.write_text(content)
                self.assertEqual(engine_module.SnapcompactEngine().mode, "summarize")

    def test_invalid_choice_preserves_saved_preference(self):
        engine = engine_module.SnapcompactEngine()
        engine.set_mode("snapcompact")
        with self.assertRaises(ValueError):
            engine.set_mode("invalid")
        self.assertEqual(engine_module.SnapcompactEngine().mode, "snapcompact")

    def test_profiles_do_not_share_preference(self):
        engine_module.SnapcompactEngine().set_mode("snapcompact")
        with patch.dict(os.environ, HERMES_HOME=str(self.home / "other")):
            self.assertEqual(engine_module.SnapcompactEngine().mode, "summarize")
        self.assertEqual(engine_module.SnapcompactEngine().mode, "snapcompact")

    def test_failed_atomic_replace_preserves_saved_choice(self):
        engine = engine_module.SnapcompactEngine()
        engine.set_mode("snapcompact")
        with patch.object(engine_module.os, "replace", side_effect=OSError("disk failure")):
            with self.assertLogs(engine_module.logger, level="WARNING"):
                self.assertFalse(engine.set_mode("summarize"))
        self.assertEqual(engine.mode, "summarize")
        self.assertEqual(engine_module.SnapcompactEngine().mode, "snapcompact")

    def register(self, ready=True, settings=None):
        commands = {}
        engines = []
        ctx = SimpleNamespace(
            llm=None, register_context_engine=engines.append,
            register_command=lambda name, handler, description: commands.update({name: handler}),
            get_config=lambda key, default=None: (settings or {}).get(key, default),
        )
        with patch.object(engine_module.SnapcompactEngine, "ensure_ready", return_value=(ready, "test bridge")):
            plugin.register(ctx)
        return engines[0], commands["compact-mode"]

    def test_frame_image_price_comes_from_plugin_settings(self):
        engine, _ = self.register(settings={"frame_image_tokens": 3100})
        self.assertEqual(engine.frame_image_tokens, 3100)
        with self.assertLogs(engine_module.logger, level="WARNING"):
            engine, _ = self.register(settings={"frame_image_tokens": "lots"})
        self.assertEqual(engine.frame_image_tokens, engine_module.DEFAULT_FRAME_IMAGE_TOKENS)

    def test_summary_prompt_file_comes_from_plugin_settings(self):
        engine, _ = self.register()
        self.assertEqual(engine.summary_prompt_file, "")
        engine, _ = self.register(settings={"summary_prompt_file": "/srv/prompts/ada.md"})
        self.assertEqual(engine.summary_prompt_file, "/srv/prompts/ada.md")
        # Host clones (one per agent) keep the path, not a handle to ctx.
        self.assertEqual(copy.deepcopy(engine).summary_prompt_file, "/srv/prompts/ada.md")
        with self.assertLogs(engine_module.logger, level="WARNING"):
            engine, _ = self.register(settings={"summary_prompt_file": ["not", "a", "path"]})
        self.assertEqual(engine.summary_prompt_file, "")

    def test_unavailable_bridge_preserves_opt_in_until_repaired(self):
        engine_module.SnapcompactEngine().set_mode("snapcompact")
        with self.assertLogs(plugin.logger, level="WARNING"):
            engine, command = self.register(ready=False)
        self.assertEqual(engine.mode, "summarize")
        self.assertEqual(engine_module.SnapcompactEngine().mode, "snapcompact")
        with patch.object(engine, "ensure_ready", return_value=(False, "missing")):
            command("snapcompact")
        self.assertEqual(engine.mode, "summarize")
        with patch.object(engine, "ensure_ready", return_value=(True, "ready")):
            command("snapcompact")
        self.assertEqual(engine.mode, "snapcompact")

    def test_explicit_summarize_while_degraded_cancels_saved_opt_in(self):
        engine_module.SnapcompactEngine().set_mode("snapcompact")
        with self.assertLogs(plugin.logger, level="WARNING"):
            engine, command = self.register(ready=False)
        command("summarize")
        self.assertEqual(engine_module.SnapcompactEngine().mode, "summarize")


class HostProviderContractTests(unittest.TestCase):
    """The summary request must survive Hermes' real provider adapter, not only the PluginLlm facade.

    Hermes routes openai-codex auxiliary calls through ``_CodexCompletionsAdapter``, whose request
    builder iterates ``messages`` and calls ``.get`` on each one. 1.0.0 passed the prompt as a bare
    string, so every summary on a Codex-backed install failed with
    ``'str' object has no attribute 'get'`` and fell back to frames (137 times on one VPS,
    2026-09-21 to 2026-09-28).
    """

    def setUp(self):
        try:
            from agent.auxiliary_client import _CodexCompletionsAdapter
        except ImportError:
            self.skipTest("this Hermes has no Codex auxiliary adapter")
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.enterContext(patch.dict(os.environ, HERMES_HOME=home.name))
        # No renderer: a failed summary raises here instead of falling back to frames.
        self.enterContext(patch.object(engine_module.SnapcompactEngine, "ensure_ready", return_value=(False, "no renderer in this test")))
        self.adapter = _CodexCompletionsAdapter(None, "gpt-test")
        self.requests = []
        self.engine = engine_module.SnapcompactEngine()
        self.engine.mode = "summarize"
        self.engine.set_llm(PluginLlm(plugin_id="hermes-snapcompact", sync_caller=self.codex))

    def codex(self, **kw):
        """The host's own chat -> Responses request builder, then a canned reply (no network)."""
        request, model, _timeout = self.adapter._build_responses_kwargs({"model": "gpt-test", "messages": kw["messages"]})
        self.requests.append(request)
        return "openai-codex", model, SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Handoff: FIRST_5 decided."))],
            model=model, usage=None,
        )

    def test_summary_request_passes_the_codex_request_builder(self):
        out = self.engine.compress(history())
        self.assertEqual(len(self.requests), 1)
        self.assertIn("FIRST_5:", json.dumps(self.requests[0]["input"]))
        self.assertIn("Handoff: FIRST_5 decided.", engine_module.serialize_messages(out))
        self.assertFalse(images(out))

    def test_archive_pages_reach_the_codex_request_as_images(self):
        self.engine.update_model("gpt-test", 272_000, provider="openai-codex")
        with patch("agent.image_routing._lookup_supports_vision", return_value=True):
            out = self.engine.compress(leftovers())
        self.assertEqual(json.dumps(self.requests[-1]["input"]).count('"input_image"'), 34)
        self.assertEqual(len(artifacts(out)), 1)
        self.assertFalse(images(out))



if __name__ == "__main__":
    unittest.main()
