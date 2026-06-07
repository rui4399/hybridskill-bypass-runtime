from __future__ import annotations

import argparse
import unittest

import gate_selector_runtime_smoke as gate


def args(**overrides):
    values = {
        "min_kernel_configs": 1,
        "min_replaced_modules": 1,
        "min_selector_calls": 1,
        "min_compression_vs_fp32": 1.0,
        "min_generated_tokens": 1,
        "max_memory_ratio": 0.90,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def generation(selector: bool = True) -> dict:
    summary = []
    if selector:
        summary.append(
            {
                "selection": "selector",
                "requested_batch": 12,
                "rows": 2048,
                "cols": 1024,
                "batch": 16,
                "block_m": 16,
                "block_n": 8,
                "block_k": 128,
                "status": "fp16_win",
                "count": 2,
            }
        )
    return {
        "mode": "triton_grouped",
        "kernel_config_count": 8,
        "replaced_module_count": 1,
        "replaced_modules": [{"module": "model.layers.0.self_attn.q_proj", "runtime_config_summary": summary}],
        "selected_compression_vs_fp32": 7.6,
        "ttft_seconds": 1.2,
        "tokens_per_second": 3.4,
        "generated_tokens_text_retokenized": 4,
    }


def guard(memory_ratio: float = 0.50) -> dict:
    return {
        "returncode": 0,
        "killed_by_guard": False,
        "killed_by_timeout": False,
        "max_memory_used_ratio": memory_ratio,
        "max_memory_used_mib": 4096,
        "memory_total_mib": 8192,
    }


class GateSelectorRuntimeSmokeTests(unittest.TestCase):
    def test_passes_when_selector_was_used_under_guard(self) -> None:
        result = gate.build_result(generation(), guard(), args())
        self.assertTrue(result["passed"])
        self.assertEqual(result["summary"]["selector_call_count"], 2)

    def test_fails_when_selector_was_not_used(self) -> None:
        result = gate.build_result(generation(selector=False), guard(), args())
        self.assertFalse(result["passed"])
        self.assertTrue(any("selector calls" in failure for failure in result["failures"]))

    def test_fails_when_guard_memory_exceeds_limit(self) -> None:
        result = gate.build_result(generation(), guard(memory_ratio=0.95), args())
        self.assertFalse(result["passed"])
        self.assertTrue(any("max memory ratio" in failure for failure in result["failures"]))


if __name__ == "__main__":
    unittest.main()
