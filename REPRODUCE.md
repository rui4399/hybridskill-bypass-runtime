# Reproduction Plan

Minimum gates after migration:

```bash
python train_python/eval_chat_task_benchmark.py --help
python train_python/hybrid_eval_quant_policy.py --help
```

The first release should expose deterministic tasks, scoring, parser coverage,
and latency summaries. Leaked historical v2 artifacts should not be included as
official data.

