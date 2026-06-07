# HybridSkill: Deterministic Bypass for Strict Low-Entropy LLM Skills

## Target

Primary: efficient-AI / edge-AI workshop, negative-results workshop, or
software-engineering-for-AI venue.

## Abstract Draft

Small LLMs are increasingly used for local automation, but not all tasks should
be learned by the model. Strict low-entropy tasks such as policy thresholding,
schema repair, and numerical routing often require exact decisions where a
short fine-tuning run may be unreliable. We study this delegation boundary by
comparing model-only execution, deterministic parser execution, and hybrid
LLM-plus-bypass execution on leak-free strict-task fixtures. The resulting
artifact treats deterministic bypass as a first-class systems mechanism rather
than a hidden post-processing trick. The goal is not to prove that LLMs cannot
learn these behaviors; instead, it is to identify when deterministic code is
more auditable, lower latency, and safer under exact-match requirements.

## Core Claim

> For strict low-entropy skills, a deterministic bypass can be the correct
> execution substrate; the research question is when to delegate, not whether
> rules are more glamorous than models.

## Contributions

1. Provide leak-free strict-task fixtures for policy and chat-stress evaluation.
2. Compare model-only, parser-only, and hybrid execution under exact-match
   metrics.
3. Report short-LoRA failures only as scoped negative evidence.
4. Provide regression gates for parser coverage, task failures, and selector
   behavior.

## Method Skeleton

Let an input \(x\) be routed by a selector \(r(x)\in\{\text{LLM},
\text{BYPASS}\}\). For low-entropy skills, the bypass function \(f_b(x)\)
returns a structured answer with exact validation. The LLM path \(f_m(x)\) is
used when semantic interpretation or open-ended response is needed.

The hybrid output is:

\[
f(x)=
\begin{cases}
f_b(x), & r(x)=\text{BYPASS},\\
f_m(x), & r(x)=\text{LLM}.
\end{cases}
\]

Evaluation must report:

- exact match;
- parser coverage;
- abstention/fallback rate;
- latency;
- failure class;
- leak audit.

## Experimental Plan

### Current Evidence

- Leak-free `eigenskill_quant_v1` fixtures.
- Deterministic task stress fixtures.
- C++ policy-bypass reports and Python gates.
- Model-only and hybrid evaluation scripts.

### Missing Work

1. Completion-only loss and constrained-decoding ablations.
2. Parser latency and maintainability measurements.
3. Larger and more diverse strict-task suite.
4. Edge runtime integration after ESMP matures.

## Figure Plan

1. Figure 1: delegation boundary between LLM and deterministic bypass.
2. Table 1: model-only versus parser-only versus hybrid exact match.
3. Table 2: failure taxonomy.
4. Figure 2: latency/coverage tradeoff.

## Limitations

This paper must not overgeneralize from failed short LoRA runs. Historical
leaked data can only be discussed as an audit caution, never as headline
evidence.
