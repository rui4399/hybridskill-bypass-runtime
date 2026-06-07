from __future__ import annotations

import unittest

import eval_chat_task_ollama as ollama_eval


class EvalChatTaskOllamaTests(unittest.TestCase):
    def test_evaluate_rows_reuses_task_scoring(self) -> None:
        tasks = [
            {
                "id": "mcq_1",
                "type": "mcq",
                "prompt": "Answer with only the option letter.\nA. x\nB. y",
                "answer": "B",
            }
        ]

        def fake_generate(model: str, prompt: str, max_new_tokens: int, endpoint: str) -> dict:
            self.assertEqual(model, "local")
            self.assertIn("/no_think", prompt)
            return {"generated_text": "B", "ttft_seconds": 0.1, "tokens_per_second": 12.0}

        rows = ollama_eval.evaluate_rows(
            tasks,
            model="local",
            max_new_tokens=4,
            endpoint="http://127.0.0.1:11434",
            no_think=True,
            generate_fn=fake_generate,
        )
        self.assertTrue(rows[0]["score"]["passed"])
        self.assertEqual(rows[0]["tokens_per_second"], 12.0)


if __name__ == "__main__":
    unittest.main()
