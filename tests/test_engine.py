"""Regression checks using Hermes' real ContextEngine and PluginLlm contracts.

Run with Hermes' Python environment and its checkout on PYTHONPATH.
Only the renderer and external model call are replaced; no network or user state.
"""
import copy
import importlib.util
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


class EngineTests(unittest.TestCase):
    def setUp(self):
        home = tempfile.TemporaryDirectory()
        self.addCleanup(home.cleanup)
        self.enterContext(patch.dict(os.environ, HERMES_HOME=home.name))
        self.rendered = []
        self.prompts = []
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
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        # Distinct facts, not the whole prompt, survive a lossy summary.
        facts = [name for name in ("FIRST_5", "NEXT_5", "THIRD_5") if name in prompt]
        return "test-provider", "test-model", SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Recorded facts: " + ", ".join(facts)))],
            model="test-model", usage=None,
        )

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
        new_messages = [source[0], first[1]] + copy.deepcopy(source[1:])
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

    def register(self, ready=True):
        commands = {}
        engines = []
        ctx = SimpleNamespace(
            llm=None, register_context_engine=engines.append,
            register_command=lambda name, handler, description: commands.update({name: handler}),
        )
        with patch.object(engine_module.SnapcompactEngine, "ensure_ready", return_value=(ready, "test bridge")):
            plugin.register(ctx)
        return engines[0], commands["compact-mode"]

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



if __name__ == "__main__":
    unittest.main()
