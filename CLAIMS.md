# Claims Firewall

## Supported Claims

- Leak-free deterministic fixtures can compare model-only, parser-only, and
  hybrid execution under exact-match metrics.
- Short LoRA failure can be reported as a negative result only for the measured
  model, data, training budget, and decoding setup.
- Deterministic bypass is appropriate for low-entropy strict tasks when coverage
  and latency gates pass.

## Pending Claims

- Completion-only or constrained-decoding training closes the model-only gap.
- Parser maintainability remains acceptable as task diversity grows.
- Hybrid execution improves end-to-end latency on an actual edge runtime.

## Rejected Claims

- A failed LoRA proves LLMs cannot learn policies.
- Deterministic bypass is universally better than model reasoning.
- Historical leaked v0/v2 results are valid headline evidence.
- This repo proves quantization quality, CSI, or swarm communication.
