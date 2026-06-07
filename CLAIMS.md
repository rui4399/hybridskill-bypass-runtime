# Claims Firewall

## Supported After Migration

- Deterministic parsers can provide exactness for selected low-entropy tasks.
- Model-only short LoRA attempts can be reported as negative evidence when the
  data split is leak-free and the scoring is deterministic.
- Hybrid routing can be evaluated with coverage, exactness, latency, and
  fallback-rate metrics.

## Rejected Until Proven

- "The LLM learns strict quantization policies reliably."
- "The bypass generalizes to arbitrary skills."
- "This proves spectral/eigen routing."

## Required Before Submission

- Leak-free datasets.
- Completion-only loss or constrained decoding ablations.
- C++ parser benchmark against model-only and Python parser baselines.

