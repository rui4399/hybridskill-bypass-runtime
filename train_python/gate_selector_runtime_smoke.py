#!/usr/bin/env python3
from __future__ import annotations

"""Gate selector-driven ESMP runtime smoke outputs."""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def selector_events(generation: dict[str, Any]) -> list[dict[str, Any]]:
    events = []
    for module in generation.get("replaced_modules", []):
        for event in module.get("runtime_config_summary", []):
            if event.get("selection") == "selector" and int(event.get("count", 0)) > 0:
                item = dict(event)
                item["module"] = module.get("module")
                events.append(item)
    return events


def check_generation(generation: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if generation.get("mode") != "triton_grouped":
        failures.append(f"generation mode is {generation.get('mode')!r}, expected 'triton_grouped'")
    if int(generation.get("kernel_config_count", 0)) < args.min_kernel_configs:
        failures.append(f"kernel config count {generation.get('kernel_config_count')} < {args.min_kernel_configs}")
    if int(generation.get("replaced_module_count", 0)) < args.min_replaced_modules:
        failures.append(f"replaced modules {generation.get('replaced_module_count')} < {args.min_replaced_modules}")
    events = selector_events(generation)
    total_selector_calls = sum(int(event.get("count", 0)) for event in events)
    if total_selector_calls < args.min_selector_calls:
        failures.append(f"selector calls {total_selector_calls} < {args.min_selector_calls}")
    compression = float(generation.get("selected_compression_vs_fp32", 0.0))
    if compression < args.min_compression_vs_fp32:
        failures.append(f"selected compression {compression:.4f} < {args.min_compression_vs_fp32:.4f}")
    if generation.get("ttft_seconds") is None:
        failures.append("missing TTFT")
    tps = generation.get("tokens_per_second")
    if tps is None or float(tps) <= 0.0:
        failures.append(f"invalid tokens_per_second: {tps!r}")
    generated = int(generation.get("generated_tokens_text_retokenized", 0))
    if generated < args.min_generated_tokens:
        failures.append(f"generated tokens {generated} < {args.min_generated_tokens}")
    return failures


def check_guard(guard: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    if int(guard.get("returncode", -1)) != 0:
        failures.append(f"guarded command returncode {guard.get('returncode')} != 0")
    if guard.get("killed_by_guard"):
        failures.append("guard killed command for memory")
    if guard.get("killed_by_timeout"):
        failures.append("guard killed command for timeout")
    max_memory_ratio = float(guard.get("max_memory_used_ratio", 1.0))
    if max_memory_ratio > args.max_memory_ratio:
        failures.append(f"max memory ratio {max_memory_ratio:.4f} > {args.max_memory_ratio:.4f}")
    return failures


def build_result(generation: dict[str, Any], guard: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    events = selector_events(generation)
    failures = check_generation(generation, args) + check_guard(guard, args)
    return {
        "passed": not failures,
        "failures": failures,
        "summary": {
            "mode": generation.get("mode"),
            "kernel_config_count": generation.get("kernel_config_count"),
            "replaced_module_count": generation.get("replaced_module_count"),
            "selector_event_count": len(events),
            "selector_call_count": sum(int(event.get("count", 0)) for event in events),
            "selected_compression_vs_fp32": generation.get("selected_compression_vs_fp32"),
            "ttft_seconds": generation.get("ttft_seconds"),
            "tokens_per_second": generation.get("tokens_per_second"),
            "generated_tokens": generation.get("generated_tokens_text_retokenized"),
            "guard_max_memory_used_ratio": guard.get("max_memory_used_ratio"),
            "guard_max_memory_used_mib": guard.get("max_memory_used_mib"),
            "guard_memory_total_mib": guard.get("memory_total_mib"),
        },
        "selector_events": events,
        "thresholds": {
            "min_kernel_configs": args.min_kernel_configs,
            "min_replaced_modules": args.min_replaced_modules,
            "min_selector_calls": args.min_selector_calls,
            "min_compression_vs_fp32": args.min_compression_vs_fp32,
            "min_generated_tokens": args.min_generated_tokens,
            "max_memory_ratio": args.max_memory_ratio,
        },
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# Selector Runtime Smoke Gate",
        "",
        f"Status: **{'PASS' if result['passed'] else 'FAIL'}**",
        "",
        "## Summary",
        "",
        f"- mode: `{summary['mode']}`",
        f"- kernel configs loaded: {summary['kernel_config_count']}",
        f"- replaced modules: {summary['replaced_module_count']}",
        f"- selector events: {summary['selector_event_count']}",
        f"- selector calls: {summary['selector_call_count']}",
        f"- selected compression vs FP32: {fmt(summary['selected_compression_vs_fp32'])}x",
        f"- TTFT: {fmt(summary['ttft_seconds'])} s",
        f"- tokens/s: {fmt(summary['tokens_per_second'])}",
        f"- generated tokens: {summary['generated_tokens']}",
        f"- guard peak memory: {summary['guard_max_memory_used_mib']} / {summary['guard_memory_total_mib']} MiB ({fmt(summary['guard_max_memory_used_ratio'])})",
        "",
        "## Selector Events",
        "",
        "| module | requested batch | selected batch | shape | BM | BN | BK | status | count |",
        "|---|---:|---:|---|---:|---:|---:|---|---:|",
    ]
    for event in result["selector_events"]:
        lines.append(
            f"| `{event.get('module')}` | {event.get('requested_batch')} | {event.get('batch')} | "
            f"`{event.get('rows')}x{event.get('cols')}` | {event.get('block_m')} | {event.get('block_n')} | "
            f"{event.get('block_k')} | `{event.get('status')}` | {event.get('count')} |"
        )
    lines.extend(["", "## Failures", ""])
    if result["failures"]:
        lines.extend(f"- {failure}" for failure in result["failures"])
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate selector-driven ESMP generation smoke outputs.")
    parser.add_argument("--generation-json", type=Path, required=True)
    parser.add_argument("--guard-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/real_system_packer_2026-06-05/selector_runtime_smoke_gate.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/real_system_packer_2026-06-05/SELECTOR_RUNTIME_SMOKE_GATE.md"))
    parser.add_argument("--min-kernel-configs", type=int, default=1)
    parser.add_argument("--min-replaced-modules", type=int, default=1)
    parser.add_argument("--min-selector-calls", type=int, default=1)
    parser.add_argument("--min-compression-vs-fp32", type=float, default=1.0)
    parser.add_argument("--min-generated-tokens", type=int, default=1)
    parser.add_argument("--max-memory-ratio", type=float, default=0.90)
    args = parser.parse_args()

    result = build_result(load_json(args.generation_json), load_json(args.guard_json), args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.out_md, result)
    print(json.dumps({"passed": result["passed"], "failures": result["failures"], "out_json": str(args.out_json)}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
