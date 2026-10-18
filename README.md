# AI Evaluation Lab

**A hands-on research notebook for AI quality engineering, LLM evaluation, and risk-based testing.**

AI Evaluation Lab is a growing collection of mathematical notes, open-source implementation investigations, controlled experiments, and practical testing insights focused on understanding how AI systems are evaluated—and where evaluation metrics can fail.

The goal is to go beyond calling evaluation libraries: understand the mathematics, trace the implementation, inspect intermediate outputs, and validate behavior through experiments.

## Research Principles

- **Understand before using:** Study the mathematical foundations and assumptions behind each metric.
- **Inspect the implementation:** Trace how open-source frameworks translate metric definitions into executable code.
- **Experiment with evidence:** Record inputs, model configurations, intermediate outputs, scores, warnings, and limitations.
- **Test risks, not just happy paths:** Explore edge cases, false positives, false negatives, incomplete answers, and evaluator instability.
- **Prefer reproducibility:** Document environment details, model versions, configurations, and experiment procedures.
- **Learn from first principles:** Implement simplified versions of metrics to verify their mathematics independently.

## Current Research

### Answer Relevancy

The initial research investigates how an LLM-generated answer can be evaluated for relevance to its original question.

Two implementations are being studied:

- **Ragas:** Generates questions from an answer and compares their embeddings with the original question's embedding using cosine similarity.
- **DeepEval:** Its standard LLM-as-a-judge approach extracts answer statements and evaluates their relevance to the input.

The central research question is not simply which framework produces a higher score, but how each metric works, what assumptions it makes, and how reliably its outputs reflect the intended evaluation objective.

### Initial Experiment

A local Ragas experiment uses an Ollama-hosted Gemma model as the judge and a local Nomic embedding model.

| Test case | Observed score |
|---|---:|
| Relevant answer about exercise benefits | 1.0000 |
| Unrelated answer about the Pacific Ocean | 0.3071 |

These are preliminary observations from two examples, not a benchmark or a general performance claim.

The run also reported fewer generated responses than the configured strictness requested. Investigating this discrepancy and capturing intermediate outputs are part of the ongoing research.

## Research Workflow

1. **Define the metric** — document its purpose, mathematical expression, inputs, and assumptions.
2. **Inspect the source code** — identify the relevant classes, functions, prompts, and scoring logic.
3. **Design controlled tests** — use identical input cases when comparing evaluators.
4. **Capture evidence** — preserve intermediate model outputs, scores, warnings, timing, and configuration.
5. **Investigate failure modes** — examine ambiguous inputs, irrelevant content, missing information, and unstable judgments.
6. **Validate from first principles** — implement the core mathematics independently where practical.
7. **Document conclusions** — distinguish verified findings from hypotheses and open questions.

## Repository Structure

```text
ai-evaluation-lab/
├── README.md
├── ai-evaluation/
│   └── relevance.md
├── experiments/
├── bug-investigations/
└── templates/
    └── research-note.md
