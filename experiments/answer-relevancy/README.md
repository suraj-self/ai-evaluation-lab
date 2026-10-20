# Answer Relevancy Experiments

## Objective

Explore how different evaluation approaches measure whether an LLM-generated answer addresses the user's question.

This experiment compares two approaches:

- **DeepEval:** statement-level relevance judgments using an LLM judge.
- **Ragas:** reverse-question generation followed by embedding-based cosine similarity.

The objective is to understand the techniques, inspect intermediate results, verify the mathematics independently, and identify limitations—not merely compare final scores.

## Experiment Design

Both frameworks use the same test cases from [`cases.json`](cases.json).

| Case | Question | Expected behavior |
|---|---|---|
| Relevant | What are the benefits of regular exercise? | The answer addresses exercise benefits. |
| Irrelevant | What are the benefits of regular exercise? | The answer discusses an unrelated topic: the Pacific Ocean. |

The expected behavior is qualitative. Actual scores and intermediate outputs depend on the evaluation method and model.

## Local-Only Requirement

All experiments use local Ollama models. No paid APIs or real API keys are required.

Models used in the current experiments:

- `gemma3n:e2b` — evaluation judge and question generation.
- `nomic-embed-text:latest` — embedding model for Ragas.

Prepare the models:

```bash
ollama pull gemma3n:e2b
ollama pull nomic-embed-text:latest
ollama list
```

Ensure Ollama is running and accessible at `http://localhost:11434`.

## Repository Structure

```text
answer-relevancy/
├── README.md
├── cases.json
├── deepeval/
│   ├── README.md
│   ├── run_experiment.py
│   └── requirements.txt
└── ragas/
    ├── README.md
    ├── run_experiment.py
    └── pyproject.toml
```

The Ragas environment and experiment are still exploratory; consult its README for known setup limitations.

## Run an Experiment

Follow the framework-specific instructions:

- [DeepEval experiment](deepeval/README.md)
- [Ragas experiment](ragas/README.md)

Keep the frameworks' dependencies separate so installing or updating one does not unintentionally change the other.

## What to Record

For each run, capture:

- Framework and package versions.
- Model names, configuration, and evaluation settings.
- Original question and generated answer.
- Intermediate statements or generated questions.
- Verdicts, noncommittal flags, or individual cosine similarities.
- Final score, warnings, and exceptions.

## Important Interpretation Notes

Scores from different frameworks are not necessarily calibrated to the same scale.

Answer Relevancy alone does not establish factual correctness, faithfulness to supplied context, or completeness.

See [`../../ai-evaluation/relevance.md`](../../ai-evaluation/relevance.md) for the concepts, mathematical foundations, techniques, strengths, and limitations.
