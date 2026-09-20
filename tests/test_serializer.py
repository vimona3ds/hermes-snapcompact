import unittest

from test_engine import engine_module


class SerializerTests(unittest.TestCase):
    def serialize_tool_result(self, text):
        return engine_module.serialize_messages([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call", "function": {"name": "read", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call", "content": text},
        ])

    def test_data_url_elision_preserves_following_words(self):
        result = self.serialize_tool_result("data:image/png;base64,AA== KEEP_ME SECRET")
        self.assertIn("KEEP_ME SECRET", result)
        self.assertNotIn("AA==", result)

    def test_case_insensitive_base64_marker_is_supported(self):
        result = self.serialize_tool_result("data:image/png;BASE64,AA== IMPORTANT")
        self.assertIn("IMPORTANT", result)
        self.assertNotIn("AA==", result)


if __name__ == "__main__":
    unittest.main()
