from __future__ import annotations

import argparse
import unittest

import gate_chat_task_regression_analysis as gate


def args(**overrides):
    values = {
        "candidate": "konly_layers17",
        "min_candidates": 3,
        "min_tasks": 84,
        "min_task_types": 8,
        "min_baseline_passes": 46,
        "min_fused_passes": 45,
        "min_fused_accuracy": 45 / 84,
        "max_pass_loss": 1,
        "max_regressions": 1,
        "max_regressions_per_type": 1,
        "require_type_fused_pass": False,
        "min_speedup": 0.95,
        "max_memory_ratio": 0.90,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def type_summary(regressions: int = 1) -> dict:
    return {
        "json_keys": {
            "tasks": 12,
            "baseline_passes": 7,
            "fused_passes": 6,
            "regressions": regressions,
            "fixes": 0,
            "both_pass": 6,
            "both_fail": 5,
        },
        "mcq": {"tasks": 12, "baseline_passes": 10, "fused_passes": 10, "regressions": 0, "fixes": 0, "both_pass": 10, "both_fail": 2},
        "number": {"tasks": 12, "baseline_passes": 3, "fused_passes": 3, "regressions": 0, "fixes": 0, "both_pass": 3, "both_fail": 9},
        "all_of": {"tasks": 12, "baseline_passes": 9, "fused_passes": 9, "regressions": 0, "fixes": 0, "both_pass": 9, "both_fail": 3},
        "contains_all": {"tasks": 12, "baseline_passes": 0, "fused_passes": 0, "regressions": 0, "fixes": 0, "both_pass": 0, "both_fail": 12},
        "contains_none": {"tasks": 12, "baseline_passes": 12, "fused_passes": 12, "regressions": 0, "fixes": 0, "both_pass": 12, "both_fail": 0},
        "sentence_count": {"tasks": 6, "baseline_passes": 4, "fused_passes": 4, "regressions": 0, "fixes": 0, "both_pass": 4, "both_fail": 2},
        "word_count": {"tasks": 6, "baseline_passes": 1, "fused_passes": 1, "regressions": 0, "fixes": 0, "both_pass": 1, "both_fail": 5},
    }


def report(label: str = "konly_layers17", regressions: int = 1, pass_delta: int = -1, speedup: float = 0.97) -> dict:
    return {
        "label": label,
        "tasks": 84,
        "baseline_passes": 46,
        "fused_passes": 46 + pass_delta,
        "pass_delta": pass_delta,
        "speedup": speedup,
        "type_summary": type_summary(regressions=regressions),
        "regressions": [{"id": "r0", "type": "json_keys", "expected": ["risk"], "baseline_text": "ok", "fused_text": "bad"}]
        * regressions,
        "fixes": [],
        "path": "candidate.json",
    }


def analysis(**overrides) -> dict:
    values = {
        "reports": [
            report("qonly_layers17"),
            report("vonly_layers17", regressions=3, pass_delta=-3),
            report("konly_layers17"),
        ]
    }
    values.update(overrides)
    return values


def guard(memory_ratio: float = 0.5) -> dict:
    return {
        "returncode": 0,
        "killed_by_guard": False,
        "killed_by_timeout": False,
        "max_memory_used_ratio": memory_ratio,
        "max_memory_used_mib": 4000,
        "memory_total_mib": 8000,
    }


class GateChatTaskRegressionAnalysisTests(unittest.TestCase):
    def test_passes_bounded_regression_candidate(self) -> None:
        result = gate.build_result(analysis(), guard(), args())
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["regressions"], 1)

    def test_raises_for_missing_candidate(self) -> None:
        with self.assertRaises(ValueError):
            gate.build_result(analysis(), guard(), args(candidate="missing"))

    def test_fails_when_regressions_exceed_budget(self) -> None:
        data = analysis(reports=[report("konly_layers17", regressions=2)])
        result = gate.build_result(data, guard(), args(min_candidates=1))
        self.assertFalse(result["passed"])
        self.assertTrue(any("regressions" in failure for failure in result["failures"]))

    def test_fails_when_speed_is_too_low(self) -> None:
        data = analysis(reports=[report("konly_layers17", speedup=0.7)])
        result = gate.build_result(data, guard(), args(min_candidates=1))
        self.assertFalse(result["passed"])
        self.assertTrue(any("speedup" in failure for failure in result["failures"]))

    def test_fails_when_guard_memory_exceeds_limit(self) -> None:
        result = gate.build_result(analysis(), guard(memory_ratio=0.95), args())
        self.assertFalse(result["passed"])
        self.assertTrue(any("memory ratio" in failure for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
