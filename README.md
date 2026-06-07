# HybridSkill Bypass Runtime

This repository is the planned artifact for deterministic skill bypass and
LLM-plus-parser routing on low-entropy, numerically strict tasks.

## Research Question

For low-entropy tasks with strict numerical or schema constraints, when should
an LLM delegate to a deterministic parser or micro-kernel instead of learning
the task through short LoRA fine-tuning?

## Paper Target

- Efficient AI / edge AI workshop.
- Negative-results workshop.
- Software engineering for AI systems paper.

## Source Of Truth

First migration candidates from `../eigenskill-research-pack`:

- `train_python/hybrid_eval_*`
- `train_python/generate_*skill*`
- `data_eval/chat_task_benchmark_v1.jsonl`
- `data_eval/chat_task_stress_v2.jsonl`
- `data_eval/chat_task_stress_v3_84.jsonl`
- C++ deterministic policy parser code and reports

Historical leaked data must be marked as historical negative evidence only.

## Non-Goals

- No PTQ method claim.
- No CSI theory claim.
- No swarm/acoustic vision claim.

