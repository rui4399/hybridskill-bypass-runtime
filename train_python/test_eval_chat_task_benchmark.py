import json
import tempfile
import unittest
from pathlib import Path

import eval_chat_task_benchmark as bench


class EvalChatTaskBenchmarkTests(unittest.TestCase):
    def test_extract_choice_letter_handles_common_formats(self) -> None:
        self.assertEqual(bench.extract_choice_letter("Answer: B"), "B")
        self.assertEqual(bench.extract_choice_letter("The correct option is (c)."), "C")
        self.assertEqual(bench.extract_choice_letter("I choose D because..."), "D")
        self.assertEqual(bench.extract_choice_letter("<think>\n\n</think>\n\nB. peak GPU memory"), "B")
        self.assertIsNone(bench.extract_choice_letter("No option here"))

    def test_score_mcq_number_contains_and_json_keys(self) -> None:
        mcq = {"id": "m1", "type": "mcq", "answer": "B"}
        number = {"id": "g1", "type": "number", "answer": "42"}
        contains = {"id": "i1", "type": "contains_all", "answer": ["latency", "memory"]}
        json_keys = {"id": "j1", "type": "json_keys", "answer": ["risk", "next_step"]}

        self.assertTrue(bench.score_task(mcq, "Answer: B")["passed"])
        self.assertFalse(bench.score_task(mcq, "Answer: A")["passed"])
        self.assertTrue(bench.score_task(number, "The result is 42.")["passed"])
        self.assertFalse(bench.score_task(number, "The result is 43.")["passed"])
        self.assertTrue(bench.score_task(contains, "Latency and memory are the deployment constraints.")["passed"])
        self.assertFalse(bench.score_task(contains, "Latency only.")["passed"])
        self.assertTrue(bench.score_task(json_keys, '{"risk":"drift","next_step":"audit"}')["passed"])
        self.assertFalse(bench.score_task(json_keys, '{"risk":"drift"}')["passed"])

    def test_load_tasks_jsonl_and_aggregate(self) -> None:
        rows = [
            {"id": "m1", "type": "mcq", "prompt": "Pick one", "answer": "A"},
            {"id": "n1", "type": "number", "prompt": "Compute", "answer": "12"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tasks.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            tasks = bench.load_tasks(path)

        self.assertEqual([task["id"] for task in tasks], ["m1", "n1"])
        scored = [
            {"score": {"passed": True}},
            {"score": {"passed": False}},
        ]
        aggregate = bench.aggregate_rows(scored)
        self.assertEqual(aggregate["tasks"], 2)
        self.assertEqual(aggregate["passes"], 1)
        self.assertEqual(aggregate["accuracy"], 0.5)

    def test_apply_task_limit(self) -> None:
        tasks = [{"id": str(i)} for i in range(5)]
        self.assertEqual(len(bench.apply_task_limit(tasks, 0)), 5)
        self.assertEqual([task["id"] for task in bench.apply_task_limit(tasks, 2)], ["0", "1"])
        with self.assertRaises(ValueError):
            bench.apply_task_limit(tasks, -1)

    def test_load_mmlu_style_jsonl(self) -> None:
        row = {
            "question": "Which metric reports memory?",
            "choices": ["BLEU", "peak GPU memory", "ROUGE", "perplexity"],
            "answer": 1,
            "subject": "edge_ai",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mmlu.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            tasks = bench.load_tasks(path, task_format="mmlu")
        self.assertEqual(tasks[0]["type"], "mcq")
        self.assertEqual(tasks[0]["answer"], "B")
        self.assertIn("A. BLEU", tasks[0]["prompt"])
        self.assertIn("D. perplexity", tasks[0]["prompt"])

    def test_load_gsm8k_style_jsonl_extracts_final_answer(self) -> None:
        row = {"question": "How many bytes?", "answer": "Two per byte. #### 4096"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            tasks = bench.load_tasks(path, task_format="gsm8k")
        self.assertEqual(tasks[0]["type"], "number")
        self.assertEqual(tasks[0]["answer"], "4096")
        self.assertIn("Answer with only the final number", tasks[0]["prompt"])

    def test_load_ifeval_keyword_existence_subset(self) -> None:
        row = {
            "prompt": "Mention latency and memory.",
            "instruction_id_list": ["keywords:existence"],
            "kwargs": [{"keywords": ["latency", "memory"]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ifeval.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            tasks = bench.load_tasks(path, task_format="ifeval")
        self.assertEqual(tasks[0]["type"], "contains_all")
        self.assertEqual(tasks[0]["answer"], ["latency", "memory"])

    def test_load_ifeval_structured_and_negative_constraints(self) -> None:
        row = {
            "prompt": "Return JSON with risk, but do not use the word unsafe.",
            "instruction_id_list": [
                "detectable_format:json_format",
                "keywords:forbidden_words",
                "length_constraints:number_sentences",
            ],
            "kwargs": [
                {},
                {"forbidden_words": ["unsafe"]},
                {"num_sentences": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ifeval.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            tasks = bench.load_tasks(path, task_format="ifeval")

        self.assertEqual(tasks[0]["type"], "all_of")
        self.assertEqual([sub["type"] for sub in tasks[0]["answer"]], ["json_valid", "contains_none", "sentence_count"])
        self.assertTrue(bench.score_task(tasks[0], '{"risk":"drift"}')["passed"])
        self.assertFalse(bench.score_task(tasks[0], '{"risk":"unsafe drift"}')["passed"])
        self.assertFalse(bench.score_task(tasks[0], '{"risk":"drift"} Extra sentence.')["passed"])

    def test_score_word_count_and_json_valid(self) -> None:
        self.assertTrue(bench.score_task({"id": "j", "type": "json_valid", "answer": True}, '{"a":1}')["passed"])
        self.assertTrue(
            bench.score_task({"id": "j", "type": "json_valid", "answer": True}, '<think>\n\n</think>\n\n{"a":1}')[
                "passed"
            ]
        )
        self.assertFalse(bench.score_task({"id": "j", "type": "json_valid", "answer": True}, 'prefix {"a":1}')["passed"])
        self.assertTrue(
            bench.score_task({"id": "w", "type": "word_count", "answer": {"count": 3}}, "latency memory throughput")[
                "passed"
            ]
        )
        self.assertFalse(
            bench.score_task({"id": "w", "type": "word_count", "answer": {"count": 3}}, "latency memory")["passed"]
        )

    def test_format_task_prompt_can_request_no_think_once(self) -> None:
        prompt = bench.format_task_prompt("Answer with only A.", no_think=True)
        self.assertTrue(prompt.endswith("/no_think"))
        self.assertEqual(prompt.count("/no_think"), 1)
        already = bench.format_task_prompt("Answer with only A.\n/no_think", no_think=True)
        self.assertEqual(already.count("/no_think"), 1)

    def test_place_model_prefers_wrapped_inner_model(self) -> None:
        class Inner:
            def __init__(self) -> None:
                self.device = None

            def to(self, device: object) -> None:
                self.device = device

        class Wrapper:
            def __init__(self) -> None:
                self.model = Inner()
                self.outer_moved = False

            def to(self, device: object) -> None:
                self.outer_moved = True

        wrapper = Wrapper()
        bench.place_model(wrapper, "cuda:0")
        self.assertEqual(wrapper.model.device, "cuda:0")
        self.assertFalse(wrapper.outer_moved)


if __name__ == "__main__":
    unittest.main()
