import json
import tempfile
import unittest
from pathlib import Path

import analyze_chat_task_regressions as analyzer


def row(task_id: str, task_type: str, passed: bool, text: str) -> dict:
    return {
        "id": task_id,
        "type": task_type,
        "answer": "A",
        "generated_text": text,
        "score": {"passed": passed, "expected": "A"},
        "tokens_per_second": 10.0,
        "ttft_seconds": 0.1,
    }


class ChatTaskRegressionAnalysisTests(unittest.TestCase):
    def test_summarizes_regressions_fixes_and_type_counts(self) -> None:
        result = {
            "baseline": {
                "aggregate": {"tasks": 4, "passes": 2, "accuracy": 0.5},
                "rows": [
                    row("mcq_0", "mcq", True, "A"),
                    row("json_0", "json_keys", False, "{}"),
                    row("num_0", "number", True, "42"),
                    row("none_0", "contains_none", False, "bad"),
                ],
            },
            "fused": {
                "aggregate": {"tasks": 4, "passes": 2, "accuracy": 0.5},
                "rows": [
                    row("mcq_0", "mcq", False, "B"),
                    row("json_0", "json_keys", True, '{"a":1}'),
                    row("num_0", "number", True, "42"),
                    row("none_0", "contains_none", False, "bad"),
                ],
            },
        }

        report = analyzer.analyze_result(result, "toy")

        self.assertEqual("toy", report["label"])
        self.assertEqual(0, report["pass_delta"])
        self.assertEqual(["mcq_0"], [item["id"] for item in report["regressions"]])
        self.assertEqual(["json_0"], [item["id"] for item in report["fixes"]])
        self.assertEqual(1, report["type_summary"]["mcq"]["regressions"])
        self.assertEqual(1, report["type_summary"]["json_keys"]["fixes"])
        self.assertEqual(1, report["type_summary"]["number"]["both_pass"])
        self.assertEqual(1, report["type_summary"]["contains_none"]["both_fail"])

    def test_cli_writes_json_and_markdown(self) -> None:
        result = {
            "baseline": {"aggregate": {"tasks": 1, "passes": 1, "accuracy": 1.0}, "rows": [row("a", "mcq", True, "A")]},
            "fused": {"aggregate": {"tasks": 1, "passes": 0, "accuracy": 0.0}, "rows": [row("a", "mcq", False, "B")]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            out_json = root / "out.json"
            out_md = root / "out.md"
            source.write_text(json.dumps(result), encoding="utf-8")

            rc = analyzer.main(["--candidate", f"toy={source}", "--out-json", str(out_json), "--out-md", str(out_md)])

            self.assertEqual(0, rc)
            self.assertEqual("toy", json.loads(out_json.read_text(encoding="utf-8"))["reports"][0]["label"])
            self.assertIn("mcq", out_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
