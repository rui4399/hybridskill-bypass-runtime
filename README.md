# HybridSkill Bypass Runtime

This repository isolates the deterministic-bypass research line: when a
low-entropy task is numerically strict, a small LLM may be the wrong execution
substrate. A deterministic parser or micro-kernel can be more reliable,
cheaper, and easier to audit.

## Core Question

For strict policy, formatting, and routing tasks, when should a system delegate
to deterministic code instead of forcing a compact LLM to learn the rule through
short LoRA fine-tuning?

## Paper Track

- Efficient AI / edge AI workshop.
- Negative-results or empirical software-engineering venue.
- Possible systems paper only after latency, coverage, and maintainability
  evidence is expanded.

## Repository Contents

- `data_eval/eigenskill_quant_v1/`: leak-free quant-policy data.
- `data_eval/chat_task_*.jsonl`: deterministic chat/task stress fixtures.
- `train_python/`: hybrid evaluation, task generation, regression analysis, and
  selector gates.
- `outputs/`: lightweight policy and hybrid-task evidence summaries.

## Claim Boundary

This repo is not a quantization paper and should not reuse CSI as its main
claim. It is about **delegation boundaries**: model-only versus deterministic
bypass under strict exact-match requirements.

Historical leaked v0/v2 data must never be used as headline evidence.
