import json
import tempfile
import unittest
from pathlib import Path

import eval_chat_task_benchmark as bench
import generate_deterministic_task_stress as stress


class DeterministicTaskStressTests(unittest.TestCase):
    def test_generate_tasks_is_deterministic_and_diverse(self) -> None:
        first = stress.generate_tasks()
        second = stress.generate_tasks()

        self.assertEqual(first, second)
        self.assertEqual(len(first), stress.DEFAULT_TASK_COUNT)
        self.assertEqual(len({task["id"] for task in first}), len(first))
        self.assertGreaterEqual(
            {task["type"] for task in first},
            {
                "mcq",
                "number",
                "json_keys",
                "contains_all",
                "contains_none",
                "sentence_count",
                "word_count",
                "all_of",
            },
        )

    def test_generated_tasks_load_with_native_evaluator(self) -> None:
        rows = stress.generate_tasks()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stress.jsonl"
            stress.write_jsonl(rows, path)
            loaded = bench.load_tasks(path)

        self.assertEqual(rows, loaded)

    def test_cli_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stress.jsonl"
            rc = stress.main(["--out", str(path), "--count", "16"])

            self.assertEqual(rc, 0)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 16)
            self.assertEqual(json.loads(lines[0])["id"], "stress_mcq_000")

    def test_generate_extended_stress_slice_has_84_unique_rows(self) -> None:
        rows = stress.generate_tasks(84)

        self.assertEqual(len(rows), 84)
        self.assertEqual(len({task["id"] for task in rows}), 84)
        self.assertEqual(rows[-1]["id"], "stress_all_of_extra_005")


if __name__ == "__main__":
    unittest.main()
