#!/usr/bin/env python3
"""Generate a deterministic local stress slice for chat task policy checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TASK_COUNT = 42


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mcq_tasks() -> list[dict[str, Any]]:
    rows = [
        (
            "Which metric reports first-token response delay?",
            ["TTFT", "BLEU", "ROUGE", "F1"],
            "A",
        ),
        (
            "Which artifact stores two signed 4-bit weights in one byte?",
            ["Packed INT4 weights", "FP32 activations", "Tokenizer merges", "KV cache"],
            "A",
        ),
        (
            "Which split is commonly used for web-text calibration?",
            ["C4", "GSM8K", "HumanEval", "MBPP"],
            "A",
        ),
        (
            "What does a quality-preserving role policy selector reject first?",
            ["Pass-count regression", "Lower file size", "Shorter README", "More commits"],
            "A",
        ),
        (
            "Which projection role is the current conservative candidate in this repo?",
            ["V only", "Q only", "K only", "Full QKV"],
            "A",
        ),
        (
            "Which benchmark style checks instruction constraints?",
            ["IFEval", "WikiText2", "C4", "PTB"],
            "A",
        ),
    ]
    tasks: list[dict[str, Any]] = []
    for idx, (question, choices, answer) in enumerate(rows):
        options = " ".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        tasks.append(
            {
                "id": f"stress_mcq_{idx:03d}",
                "type": "mcq",
                "prompt": f"Answer with only the option letter. {question} {options}",
                "answer": answer,
                "source_style": "deterministic_stress_mcq",
            }
        )
    return tasks


def _number_tasks() -> list[dict[str, Any]]:
    specs = [
        ("A row has 8192 signed 4-bit weights. Two weights fit in one byte. How many bytes are needed?", 4096),
        ("A selected-row kernel reads 32 rows, each row has 4096 signed 4-bit weights. How many packed bytes are read?", 65536),
        ("A matrix has 80 percent 4-bit rows and 20 percent 8-bit rows. What is the average bit width?", 4.8),
        ("A policy keeps 3 of 24 layers dense. How many layers are packed?", 21),
        ("A benchmark has 48 prompts and runs baseline plus fused once. How many generations are measured?", 96),
        ("A GPU has 8151 MiB total memory. A 90 percent guard allows at most how many MiB if rounded down?", 7335),
    ]
    return [
        {
            "id": f"stress_number_{idx:03d}",
            "type": "number",
            "prompt": f"Answer with only the final number. {prompt}",
            "answer": str(answer),
            "source_style": "deterministic_stress_numeric",
        }
        for idx, (prompt, answer) in enumerate(specs)
    ]


def _json_key_tasks() -> list[dict[str, Any]]:
    specs = [
        ("Return minified JSON only with keys metric and value. Use metric for TTFT.", ["metric", "value"]),
        ("Return minified JSON only with keys risk and evidence. Describe calibration split instability.", ["risk", "evidence"]),
        ("Return minified JSON only with keys policy and decision. The policy should be V-only.", ["policy", "decision"]),
        ("Return minified JSON only with keys memory_mb and tokens_per_second.", ["memory_mb", "tokens_per_second"]),
        ("Return minified JSON only with keys layer and role.", ["layer", "role"]),
        ("Return minified JSON only with keys dataset and pass_count.", ["dataset", "pass_count"]),
    ]
    return [
        {
            "id": f"stress_json_keys_{idx:03d}",
            "type": "json_keys",
            "prompt": prompt,
            "answer": keys,
            "source_style": "deterministic_stress_json_keys",
        }
        for idx, (prompt, keys) in enumerate(specs)
    ]


def _contains_all_tasks() -> list[dict[str, Any]]:
    specs = [
        ("In one sentence, state why PPL alone is incomplete for instruction models.", ["perplexity", "task", "accuracy"]),
        ("Name three physical deployment metrics for edge LLM inference.", ["ttft", "throughput", "memory"]),
        ("State the core reason calibration robustness matters.", ["calibration", "split", "instability"]),
        ("Write a caveat about packed compression and real runtime.", ["compression", "latency", "speedup"]),
        ("Describe what a deterministic stress slice should avoid.", ["download", "leakage", "random"]),
        ("State what must be measured before claiming phone deployment.", ["device", "ttft", "tokens"]),
    ]
    return [
        {
            "id": f"stress_contains_all_{idx:03d}",
            "type": "contains_all",
            "prompt": prompt,
            "answer": required,
            "source_style": "deterministic_stress_contains_all",
        }
        for idx, (prompt, required) in enumerate(specs)
    ]


def _contains_none_tasks() -> list[dict[str, Any]]:
    specs = [
        ("Describe a conservative result without using the word SOTA.", ["sota"]),
        ("Describe a local diagnostic without using the word official.", ["official"]),
        ("Describe a packed kernel experiment without using the word proven.", ["proven"]),
        ("Describe a negative result without using the word failure.", ["failure"]),
        ("Describe C-drive cleanup without using the word delete.", ["delete"]),
        ("Describe a benchmark slice without using the word universal.", ["universal"]),
    ]
    return [
        {
            "id": f"stress_contains_none_{idx:03d}",
            "type": "contains_none",
            "prompt": prompt,
            "answer": forbidden,
            "source_style": "deterministic_stress_contains_none",
        }
        for idx, (prompt, forbidden) in enumerate(specs)
    ]


def _length_tasks() -> list[dict[str, Any]]:
    sentence_specs = [
        ("Explain the role-policy selector in exactly one sentence.", 1),
        ("Explain why layer20 expansion is risky in exactly two sentences.", 2),
        ("Explain why a disk guard is useful in exactly one sentence.", 1),
    ]
    word_specs = [
        ("Answer in exactly three words: name latency, memory, throughput.", 3),
        ("Answer in exactly four words: summarize robust calibration evidence.", 4),
        ("Answer in exactly five words: summarize conservative V-only policy.", 5),
    ]
    tasks: list[dict[str, Any]] = []
    for idx, (prompt, count) in enumerate(sentence_specs):
        tasks.append(
            {
                "id": f"stress_sentence_count_{idx:03d}",
                "type": "sentence_count",
                "prompt": prompt,
                "answer": {"count": count},
                "source_style": "deterministic_stress_sentence_count",
            }
        )
    for idx, (prompt, count) in enumerate(word_specs):
        tasks.append(
            {
                "id": f"stress_word_count_{idx:03d}",
                "type": "word_count",
                "prompt": prompt,
                "answer": {"count": count},
                "source_style": "deterministic_stress_word_count",
            }
        )
    return tasks


def _all_of_tasks() -> list[dict[str, Any]]:
    specs: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "Return one JSON object that includes risk and next_step, and do not use the word unsafe.",
            [
                {"type": "json_keys", "answer": ["risk", "next_step"]},
                {"type": "contains_none", "answer": ["unsafe"]},
            ],
        ),
        (
            "Write one sentence that mentions calibration and memory, and do not use the word universal.",
            [
                {"type": "sentence_count", "answer": {"count": 1}},
                {"type": "contains_all", "answer": ["calibration", "memory"]},
                {"type": "contains_none", "answer": ["universal"]},
            ],
        ),
        (
            "Return one JSON object with keys policy and speedup, and do not use the word SOTA.",
            [
                {"type": "json_keys", "answer": ["policy", "speedup"]},
                {"type": "contains_none", "answer": ["sota"]},
            ],
        ),
        (
            "Write exactly four words that include ttft and memory.",
            [
                {"type": "word_count", "answer": {"count": 4}},
                {"type": "contains_all", "answer": ["ttft", "memory"]},
            ],
        ),
        (
            "Return one JSON object with keys layer and verdict, and do not use the word proven.",
            [
                {"type": "json_keys", "answer": ["layer", "verdict"]},
                {"type": "contains_none", "answer": ["proven"]},
            ],
        ),
        (
            "Write one sentence that mentions held out prompt and avoids the word official.",
            [
                {"type": "sentence_count", "answer": {"count": 1}},
                {"type": "contains_all", "answer": ["held", "out", "prompt"]},
                {"type": "contains_none", "answer": ["official"]},
            ],
        ),
    ]
    return [
        {
            "id": f"stress_all_of_{idx:03d}",
            "type": "all_of",
            "prompt": prompt,
            "answer": checks,
            "source_style": "deterministic_stress_all_of",
        }
        for idx, (prompt, checks) in enumerate(specs)
    ]


def _extra_mcq_tasks() -> list[dict[str, Any]]:
    rows = [
        (
            "Which artifact should stay local and out of Git?",
            ["ESMP binaries", "README text", "unit tests", "small JSONL tasks"],
            "A",
        ),
        (
            "Which guard rejects a run before launch when the GPU is already busy?",
            ["max-start-memory-ratio", "temperature top-p", "beam width", "weight decay"],
            "A",
        ),
        (
            "Which evidence is stronger than PPL for an instruction-following slice?",
            ["Task pass count", "File name length", "Commit count", "Markdown size"],
            "A",
        ),
        (
            "Which role policy survived the latest three-split gate?",
            ["Q only", "K only", "V only", "Full QKV"],
            "A",
        ),
        (
            "Which result should be excluded if the run timed out?",
            ["Full-QKV stress", "Q-only V1", "Q-only stress", "IFEval Q-only"],
            "A",
        ),
        (
            "Which storage item should not be deleted without explicit choice?",
            ["Model cache", "Repo pycache", "UV cache", "Temp scratch"],
            "A",
        ),
    ]
    tasks: list[dict[str, Any]] = []
    for idx, (question, choices, answer) in enumerate(rows):
        options = " ".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        tasks.append(
            {
                "id": f"stress_mcq_extra_{idx:03d}",
                "type": "mcq",
                "prompt": f"Answer with only the option letter. {question} {options}",
                "answer": answer,
                "source_style": "deterministic_stress_mcq_extra",
            }
        )
    return tasks


def _extra_number_tasks() -> list[dict[str, Any]]:
    specs = [
        ("A task slice has 84 rows. Baseline and fused are both measured. How many generations are run?", 168),
        ("A policy passes 22 of 42 tasks. How many tasks did it fail?", 20),
        ("A model uses 4033 MiB of an 8151 MiB GPU. Rounded down, what percent is used?", 49),
        ("A cache cleanup frees 1265 MB and another frees 122 MB. How many MB are freed in total?", 1387),
        ("A three-split selector sees pass deltas 0, 0, and 0. What is the total pass delta?", 0),
        ("A 4-bit row with 1024 weights uses how many packed bytes?", 512),
    ]
    return [
        {
            "id": f"stress_number_extra_{idx:03d}",
            "type": "number",
            "prompt": f"Answer with only the final number. {prompt}",
            "answer": str(answer),
            "source_style": "deterministic_stress_numeric_extra",
        }
        for idx, (prompt, answer) in enumerate(specs)
    ]


def _extra_json_key_tasks() -> list[dict[str, Any]]:
    specs = [
        ("Return minified JSON only with keys selected_policy and evidence.", ["selected_policy", "evidence"]),
        ("Return minified JSON only with keys c_free_gb and guard_limit.", ["c_free_gb", "guard_limit"]),
        ("Return minified JSON only with keys split and pass_delta.", ["split", "pass_delta"]),
        ("Return minified JSON only with keys role and packed.", ["role", "packed"]),
        ("Return minified JSON only with keys benchmark and caveat.", ["benchmark", "caveat"]),
        ("Return minified JSON only with keys artifact and local_only.", ["artifact", "local_only"]),
    ]
    return [
        {
            "id": f"stress_json_keys_extra_{idx:03d}",
            "type": "json_keys",
            "prompt": prompt,
            "answer": keys,
            "source_style": "deterministic_stress_json_keys_extra",
        }
        for idx, (prompt, keys) in enumerate(specs)
    ]


def _extra_contains_all_tasks() -> list[dict[str, Any]]:
    specs = [
        ("State the next validation target for the selected role policy.", ["q-only", "larger", "slice"]),
        ("Describe a careful storage migration plan.", ["move", "cache", "compact"]),
        ("State why full-QKV stress should not be cited.", ["timeout", "invalid", "result"]),
        ("Describe a conservative systems claim.", ["packed", "runtime", "evidence"]),
        ("Name three things the GPU guard records.", ["memory", "disk", "timeout"]),
        ("State why role policies need multiple splits.", ["split", "quality", "regression"]),
    ]
    return [
        {
            "id": f"stress_contains_all_extra_{idx:03d}",
            "type": "contains_all",
            "prompt": prompt,
            "answer": required,
            "source_style": "deterministic_stress_contains_all_extra",
        }
        for idx, (prompt, required) in enumerate(specs)
    ]


def _extra_contains_none_tasks() -> list[dict[str, Any]]:
    specs = [
        ("Describe the Q-only candidate without using the word final.", ["final"]),
        ("Describe a stress benchmark without using the word proof.", ["proof"]),
        ("Describe C-drive pressure without using the word panic.", ["panic"]),
        ("Describe local artifacts without using the word upload.", ["upload"]),
        ("Describe a selector rejection without using the word bad.", ["bad"]),
        ("Describe GPU memory guarding without using the word unlimited.", ["unlimited"]),
    ]
    return [
        {
            "id": f"stress_contains_none_extra_{idx:03d}",
            "type": "contains_none",
            "prompt": prompt,
            "answer": forbidden,
            "source_style": "deterministic_stress_contains_none_extra",
        }
        for idx, (prompt, forbidden) in enumerate(specs)
    ]


def _extra_length_tasks() -> list[dict[str, Any]]:
    sentence_specs = [
        ("Explain why Q-only is still provisional in exactly one sentence.", 1),
        ("Explain why cache migration needs care in exactly two sentences.", 2),
        ("Explain why timeout results are excluded in exactly one sentence.", 1),
    ]
    word_specs = [
        ("Answer in exactly three words: selected role policy.", 3),
        ("Answer in exactly four words: storage cleanup summary.", 4),
        ("Answer in exactly five words: why three splits matter.", 5),
    ]
    tasks: list[dict[str, Any]] = []
    for idx, (prompt, count) in enumerate(sentence_specs):
        tasks.append(
            {
                "id": f"stress_sentence_count_extra_{idx:03d}",
                "type": "sentence_count",
                "prompt": prompt,
                "answer": {"count": count},
                "source_style": "deterministic_stress_sentence_count_extra",
            }
        )
    for idx, (prompt, count) in enumerate(word_specs):
        tasks.append(
            {
                "id": f"stress_word_count_extra_{idx:03d}",
                "type": "word_count",
                "prompt": prompt,
                "answer": {"count": count},
                "source_style": "deterministic_stress_word_count_extra",
            }
        )
    return tasks


def _extra_all_of_tasks() -> list[dict[str, Any]]:
    specs: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "Return one JSON object with keys policy and status, and do not use the word final.",
            [
                {"type": "json_keys", "answer": ["policy", "status"]},
                {"type": "contains_none", "answer": ["final"]},
            ],
        ),
        (
            "Write one sentence that mentions q-only and stress, and do not use the word proof.",
            [
                {"type": "sentence_count", "answer": {"count": 1}},
                {"type": "contains_all", "answer": ["q-only", "stress"]},
                {"type": "contains_none", "answer": ["proof"]},
            ],
        ),
        (
            "Return one JSON object with keys disk and action, and do not use the word delete.",
            [
                {"type": "json_keys", "answer": ["disk", "action"]},
                {"type": "contains_none", "answer": ["delete"]},
            ],
        ),
        (
            "Write exactly four words that include guard and memory.",
            [
                {"type": "word_count", "answer": {"count": 4}},
                {"type": "contains_all", "answer": ["guard", "memory"]},
            ],
        ),
        (
            "Return one JSON object with keys timeout and citation, and do not use the word valid.",
            [
                {"type": "json_keys", "answer": ["timeout", "citation"]},
                {"type": "contains_none", "answer": ["valid"]},
            ],
        ),
        (
            "Write one sentence that mentions three split and avoids the word universal.",
            [
                {"type": "sentence_count", "answer": {"count": 1}},
                {"type": "contains_all", "answer": ["three", "split"]},
                {"type": "contains_none", "answer": ["universal"]},
            ],
        ),
    ]
    return [
        {
            "id": f"stress_all_of_extra_{idx:03d}",
            "type": "all_of",
            "prompt": prompt,
            "answer": checks,
            "source_style": "deterministic_stress_all_of_extra",
        }
        for idx, (prompt, checks) in enumerate(specs)
    ]


def generate_tasks(count: int = DEFAULT_TASK_COUNT) -> list[dict[str, Any]]:
    if count <= 0:
        raise ValueError("count must be positive")
    pools = [
        _mcq_tasks(),
        _number_tasks(),
        _json_key_tasks(),
        _contains_all_tasks(),
        _contains_none_tasks(),
        _length_tasks(),
        _all_of_tasks(),
        _extra_mcq_tasks(),
        _extra_number_tasks(),
        _extra_json_key_tasks(),
        _extra_contains_all_tasks(),
        _extra_contains_none_tasks(),
        _extra_length_tasks(),
        _extra_all_of_tasks(),
    ]
    tasks = [task for pool in pools for task in pool]
    if count > len(tasks):
        raise ValueError(f"count must be <= {len(tasks)}")
    return tasks[:count]


def write_jsonl(tasks: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(task, ensure_ascii=False, separators=(",", ":")) for task in tasks) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic native JSONL stress slice.")
    parser.add_argument("--out", default=str(repo_root() / "data_eval" / "chat_task_stress_v2.jsonl"))
    parser.add_argument("--count", type=int, default=DEFAULT_TASK_COUNT)
    args = parser.parse_args(argv)

    tasks = generate_tasks(args.count)
    write_jsonl(tasks, Path(args.out))
    print(json.dumps({"out": args.out, "tasks": len(tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
