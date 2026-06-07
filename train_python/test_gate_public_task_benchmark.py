from __future__ import annotations

import argparse
import unittest

import gate_public_task_benchmark as gate


def args(**overrides):
    values = {
        "min_cases": 2,
        "min_total_tasks": 100,
        "require_formats": "mmlu,gsm8k",
        "max_memory_ratio": 0.90,
        "min_mean_tokens_per_second": 0.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def case(label: str, task_format: str, tasks: int, memory_ratio: float = 0.5) -> dict:
    return {
        "label": label,
        "task_format": task_format,
        "task_count": tasks,
        "passes": 0,
        "accuracy": 0.0,
        "mean_tokens_per_second": 10.0,
        "mean_ttft_seconds": 0.2,
        "guard_returncode": 0,
        "killed_by_guard": False,
        "killed_by_timeout": False,
        "guard_max_memory_used_ratio": memory_ratio,
    }


class GatePublicTaskBenchmarkTests(unittest.TestCase):
    def test_passes_two_format_subset_under_guard(self) -> None:
        result = gate.build_result([case("mmlu", "mmlu", 50), case("gsm8k", "gsm8k", 50)], args())
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["total_tasks"], 100)

    def test_fails_when_task_count_is_too_small(self) -> None:
        result = gate.build_result([case("mmlu", "mmlu", 4), case("gsm8k", "gsm8k", 4)], args())
        self.assertFalse(result["passed"])
        self.assertTrue(any("total tasks" in failure for failure in result["failures"]))

    def test_fails_when_guard_exceeds_limit(self) -> None:
        result = gate.build_result([case("mmlu", "mmlu", 50, 0.95), case("gsm8k", "gsm8k", 50)], args())
        self.assertFalse(result["passed"])
        self.assertTrue(any("max memory ratio" in failure for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
