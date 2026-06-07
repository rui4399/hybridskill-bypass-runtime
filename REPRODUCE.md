# Reproduction

## Quick Checks

```bash
python -m unittest discover -s train_python -p "test_*.py"
```

## Leak Discipline

Only `data_eval/eigenskill_quant_v1/` is intended for current policy-learning
claims. Historical leaked data may be discussed as a cautionary audit result,
not as evidence for model quality.

## Suggested Runs

```bash
python train_python/hybrid_eval_quant_policy.py --help
python train_python/gate_chat_task_regression_analysis.py --help
```

Record exact model, decoding mode, seed, prompt template, and parser coverage
for every result.
