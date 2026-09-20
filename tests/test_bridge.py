"""Run the real Bun renderer; never install dependencies automatically."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bun = shutil.which("bun")
        modules = Path(os.environ.get("SNAPCOMPACT_NODE_MODULES", ROOT / "bridge/node_modules"))
        if not cls.bun or not modules.is_dir():
            raise unittest.SkipTest("Bun and installed bridge dependencies required")
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.script = Path(cls.work.name) / "render.ts"
        shutil.copy2(ROOT / "bridge/render.ts", cls.script)
        (cls.script.parent / "node_modules").symlink_to(modules.resolve(), target_is_directory=True)

    def bridge(self, **request):
        result = subprocess.run(
            [self.bun, "run", str(self.script)], input=json.dumps(request),
            text=True, capture_output=True, timeout=30,
        )
        return result.returncode, json.loads(result.stdout)

    def test_overflow_is_rejected_not_silently_truncated(self):
        _, shape = self.bridge(action="geometry", variant="8on22-bw")
        capacity = shape["geometry"]["cols"] * shape["geometry"]["rows"]
        code, result = self.bridge(action="render", text="A" * capacity + "TAIL", variant="8on22-bw", maxFrames=1)
        self.assertNotEqual(code, 0)
        self.assertIn("error", result)
        self.assertNotIn("images", result)

    def test_exact_capacity_succeeds(self):
        _, shape = self.bridge(action="geometry", variant="8on22-bw")
        capacity = shape["geometry"]["cols"] * shape["geometry"]["rows"]
        code, result = self.bridge(action="render", text="A" * capacity, variant="8on22-bw", maxFrames=1)
        self.assertEqual(code, 0)
        self.assertEqual(result["frameCount"], 1)

    def test_frame_count_honors_variant_and_wide_glyphs(self):
        for text, variant in (("A" * 10000, "silver16-bw"), ("漢" * 6000, "auto")):
            with self.subTest(variant=variant):
                _, counted = self.bridge(action="frames", text=text, variant=variant)
                code, rendered = self.bridge(action="render", text=text, variant=variant)
                self.assertEqual(code, 0)
                self.assertEqual(counted["frames"], rendered["frameCount"])

    def test_zero_limit_cannot_discard_nonempty_archive(self):
        code, result = self.bridge(action="render", text="KEEP", maxFrames=0)
        self.assertNotEqual(code, 0)
        self.assertNotIn("images", result)


if __name__ == "__main__":
    unittest.main()
