# DeepEval — Answer Relevancy

## Purpose

Run Answer Relevancy evaluations using a local Ollama judge and inspect the extracted statements, relevance verdicts, final score, and judge explanation.

## Tested Configuration

- Python: 3.12
- DeepEval: `4.3.1`
- Judge: `gemma3n:e2b`
- Ollama endpoint: `http://localhost:11434`
- Temperature: `0`
- Metric threshold: `0.7`
- Async mode: disabled

These values describe the observed experiment configuration, not a guarantee that every environment will produce identical results.

## Prerequisites

Install Python 3.12 and Ollama. Pull the judge model:

```bash
ollama pull gemma3n:e2b
ollama list
```

Ensure Ollama is running and reachable locally.

## Install Dependencies

From this directory:

```bash
cd experiments/answer-relevancy/deepeval

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file pins DeepEval to the version used in the experiment. Transitive dependencies are not yet fully locked.

## Run

```bash
python run_experiment.py
```

The script evaluates the shared test cases and prints the score, pass/fail result, and reason.

The configured threshold is `0.7`. Passing the threshold is distinct from the raw score itself.

## Observed Results

The original experiment produced:

| Case | Extracted statements | Verdicts | Score |
|---|---:|---|---:|
| Relevant exercise answer | 3 | Yes, yes, yes | 1.0 |
| Irrelevant Pacific Ocean answer | 1 | No | 0.0 |

The exercise answer contained one sentence but was decomposed into three claims by the judge. The number of statements is model-driven, not a fixed sentence count.

## Interpretation

The standard statement-level calculation is based on the proportion of statements judged relevant. Inspect the actual statements and verdicts before interpreting the score.

A high score does not prove that the answer is true, supported by evidence, or complete.

## Reproducibility Notes

- Use the exact test cases in `../cases.json`.
- Keep the judge model and metric settings explicit.
- Preserve intermediate outputs when investigating changes.
- Scores may vary with framework version, prompt, judge model, or model behavior.

All inference should remain local. Do not add real API keys or cloud fallbacks.
