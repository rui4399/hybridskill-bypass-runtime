# Paper Plan: HybridSkill Bypass

## One-Sentence Thesis

For strict, low-entropy skills, deterministic bypass should be treated as a
first-class execution path rather than as an afterthought to small-LLM
fine-tuning.

## Main Contributions

1. Leak-free task fixtures for strict policy and formatting behaviors.
2. Model-only versus deterministic-bypass comparison protocol.
3. Negative-result analysis for short LoRA policy learning.
4. Selector and regression gates for hybrid execution.

## Reviewer Risks

- A parser can look trivial unless the paper emphasizes delegation criteria,
  coverage, latency, and failure containment.
- Model failure must not be overgeneralized.
- Leaked historical data must be quarantined.
