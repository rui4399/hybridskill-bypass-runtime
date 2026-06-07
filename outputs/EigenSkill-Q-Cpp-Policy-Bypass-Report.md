# EigenSkill-Q C++ Policy Bypass Stage Report

Date: 2026-06-04

## 1. Stage Result

This stage adds a standalone C++ evaluator for the cleaner quantization-policy
track. The result is intentionally narrow:

```text
deterministic quantization-policy skill -> C++ policy kernel -> JSON decision fields
```

It does not claim end-to-end LLM quantization quality, edge-board latency, NPU
integration, or spectral/eigen-routing through Transformer nonlinearities.

## 2. Verified Artifacts

Code:

```text
inference_cpp/src/quant_policy_bypass.cpp
inference_cpp/build-msvc.ps1
```

Data:

```text
data_eval/eigenskill_quant_v1/eval.jsonl
data_eval/eigenskill_quant_v1/test.jsonl
data_eval/eigenskill_quant_v1/audit.json
```

Outputs:

```text
outputs/eigenskill_quant_v1_eval_cpp_policy_summary.json
outputs/eigenskill_quant_v1_test_cpp_policy_summary.json
outputs/eigenskill_quant_v1_eval_hybrid_policy_summary.json
outputs/eigenskill_quant_v1_test_hybrid_policy_summary.json
```

## 3. Dataset Integrity

The committed `eigenskill_quant_v1` split contains five policy skills:

```text
outlier_detect
bit_allocate
rotation_select
residual_patch
kv_policy
```

Split sizes:

```text
train: 1200
eval:   400
test:   400
```

The audit reports zero exact-row overlap and zero input overlap across
train/eval/test. This is stronger than the older v2 routing split, which had
severe train/eval leakage and must remain an engineering proof-of-concept only.

## 4. C++ Evaluation

Build:

```powershell
powershell -ExecutionPolicy Bypass -File .\inference_cpp\build-msvc.ps1 -Target quant-policy
```

Run:

```powershell
.\inference_cpp\build\quant_policy_bypass.exe `
  --data data_eval\eigenskill_quant_v1\eval.jsonl `
  --out outputs\eigenskill_quant_v1_eval_cpp_policy_summary.json

.\inference_cpp\build\quant_policy_bypass.exe `
  --data data_eval\eigenskill_quant_v1\test.jsonl `
  --out outputs\eigenskill_quant_v1_test_cpp_policy_summary.json
```

Local Windows/MSVC result:

| split | rows | policy_fields_exact | decision_exact | parse_error | throughput |
|---|---:|---:|---:|---:|---:|
| eval | 400 | 1.000000 | 1.000000 | 0.000000 | 8261 rows/s |
| test | 400 | 1.000000 | 1.000000 | 0.000000 | 6924 rows/s |

`policy_fields_exact` checks the fields that determine the downstream
quantization decision. It is not a strict canonical JSON metric.

## 5. Python Strict Baseline

The Python evaluator remains the stricter full JSON reference. It checks
auxiliary fields such as `risk` and `score`.

| split | rows | exact_json | decision_exact | parse_error |
|---|---:|---:|---:|---:|
| eval | 400 | 1.000000 | 1.000000 | 0.000000 |
| test | 400 | 1.000000 | 1.000000 | 0.000000 |

This gives a clean separation:

```text
Python: strict dataset/rule correctness oracle.
C++: low-overhead policy-kernel executor for decision fields.
```

## 6. Research Interpretation

The strongest current paper direction is:

```text
Sensitivity-rate-distortion policy bypass for mixed-precision LLM quantization.
```

The C++ evaluator supports the systems side of this thesis: if a policy decision
is low-entropy, numeric, and verifiable, it should not be generated token by
token by a small LLM. The LLM can route or explain; a deterministic kernel can
execute the policy.

The mathematical contribution should be framed as a constrained policy layer:

```text
minimize   D(W, Q_b(W; r, p)) + alpha C(b, r, p)
subject to memory(b) <= B, latency(b, r, p) <= L, risk(b, r, p) <= epsilon
```

where `b` is the bit-width assignment, `r` is the rotation/preconditioner
choice, and `p` is the residual/KV-cache policy. This can be extended to a
contextual bandit or constrained MDP when the policy is chosen online.

## 7. Claim Boundary

Supported now:

- no-leak synthetic quantization-policy split;
- deterministic Python full-JSON oracle;
- deterministic C++ policy-field evaluator;
- negative evidence that a short SmolLM2-360M LoRA smoke run does not learn
  the numeric policy rules;
- synthetic low-rank C++ arithmetic benchmark.

Not supported yet:

- superiority over GPTQ, AWQ, SmoothQuant, QuaRot, SpinQuant, QuIP, AQLM, or
  KV-cache quantization baselines;
- real mixed-precision LLM compression quality;
- real ARM/NPU/edge-board latency or energy results;
- spectral/eigen-routing through nonlinear Transformer blocks;
- acoustic/swarm hardware claims.

## 8. Next Experiments

Minimum next bundle for a serious CCF-A attempt:

1. Implement a no-leak semantic router split where the LLM only decides whether
   to call the policy kernel, not the final numeric output.
2. Add calibration-set sensitivity estimation for real model layers:
   activation MSE, Hessian/diagonal Fisher proxy, outlier score, and KV-cache
   sensitivity.
3. Compare static heuristics, learned policy, and constrained bandit policy.
4. Run at least one real quantization baseline on a small but recognized model:
   Qwen2.5-0.5B/1.5B, Llama-3.2-1B/3B, or SmolLM2-1.7B.
5. Report perplexity, task accuracy, memory, policy overhead, and end-to-end
   latency.
6. Only after those pass, prepare the formal submission version for ICML,
   IJCAI, AAAI, ACL, AI, TPAMI, or JMLR depending on the final evidence.
