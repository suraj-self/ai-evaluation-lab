# Ragas — Answer Relevancy

## Purpose

Investigate Ragas Answer Relevancy through reverse-question generation, text embeddings, cosine similarity, mean aggregation, and noncommittal-answer handling.

The goal is to understand the metric's implementation, capture intermediate values, and independently reproduce its mathematical calculation.

## Local Environment

| Component | Configuration |
|---|---|
| Python | 3.12.x |
| Environment manager | `uv` |
| Ragas | `0.4.3` |
| Judge LLM | `gemma3n:e2b` |
| Embedding model | `nomic-embed-text:latest` |
| Ollama endpoint | `http://localhost:11434` |
| OpenAI-compatible API | `http://localhost:11434/v1` |
| Generation temperature | `0` |
| Ragas `strictness` | `3` |

The project uses `pyproject.toml` to declare direct dependencies and `uv.lock` to record their resolved versions.

**Compatibility note:** The working environment uses `langchain-community==0.4.1`. The initial environment with `langchain-community==0.4.2` failed during the Ragas import because `langchain_community.chat_models.vertexai` was unavailable. The pin is retained as a tested compatibility workaround; reassess it when upgrading Ragas or LangChain dependencies.

## Prerequisites

Install these tools before running the experiment:

- Python 3.12.
- [`uv`](https://docs.astral.sh/uv/).
- [Ollama](https://ollama.com/) running locally.

No paid API or real API key is required. The Python clients use Ollama's local OpenAI-compatible API, with `ollama` as the placeholder API key.

## Run Locally

Run the following commands from the repository root.

### 1. Enter the experiment directory

```bash
cd experiments/answer-relevancy/ragas
```

### 2. Start Ollama

If Ollama is already running, skip this step.

On Linux systems configured with the Ollama system service:

```bash
sudo systemctl start ollama
```

Alternatively, if Ollama is not running as a service, start it in a separate terminal:

```bash
ollama serve
```

Use one startup method, not both.

### 3. Download the required models

```bash
ollama pull gemma3n:e2b
ollama pull nomic-embed-text:latest
```

Verify that both models are available:

```bash
ollama list
```

The names must match the model names configured in `run_experiment.py`.

### 4. Install project dependencies

```bash
uv sync --locked
```

This creates the project-local `.venv` and installs dependencies from the committed lockfile. The lockfile must be kept in Git alongside `pyproject.toml`.

You do not need to activate `.venv` manually when using `uv run`.

If dependency constraints have intentionally changed and the lockfile needs to be regenerated, review the dependency changes and update `uv.lock` before committing. Do not routinely upgrade dependencies just to run the experiment.

### 5. Run the experiment

```bash
uv run python run_experiment.py
```

The script evaluates two candidate answers against the same question:

> What are the benefits of regular exercise?

The cases are defined in `run_experiment.py`:

- **RELEVANT:** An answer describing benefits of exercise.
- **IRRELEVANT:** An answer stating that the Pacific Ocean is the largest ocean on Earth.

### 6. Verify the environment

Check the installed Ragas version:

```bash
uv run python -c "import ragas; print(ragas.__version__)"
```

Expected:

```text
0.4.3
```

Check the Git state to make sure the environment is not tracked:

```bash
git status --short
```

The virtual environment should be ignored by the repository's `.gitignore`. Commit `pyproject.toml`, `uv.lock`, `README.md`, and the experiment source—not `.venv`.

## Observed Results

The successful local run returned:

| Case | Observed score |
|---|---:|
| Relevant exercise answer | `1.0000` |
| Irrelevant Pacific Ocean answer | `0.3071` |

These are observations from the local `gemma3n:e2b` judge and the configured embedding model. Do not treat them as fixed expected values for every environment or model revision.

Both cases produced this warning:

```text
LLM returned 1 generations instead of requested 3.
Proceeding with 1 generations.
```

Although `strictness=3` requests three generations, the observed run proceeded with one generation per case. Therefore, these scores are **single-generation observations**, not verified averages across three generated questions.

## Method to Investigate

The intended scoring flow is:

1. Generate questions that the candidate answer could address.
2. Embed the original user question and each generated question.
3. Calculate cosine similarity between the original question embedding and each generated-question embedding.
4. Average the resulting similarity values.
5. Apply any noncommittal-answer adjustment implemented by the configured Ragas version.

For a single original question $q$ and generated questions $g_1,\ldots,g_n$, the mean similarity can be expressed as:

$$
S_{\text{mean}} =
\frac{1}{n}\sum_{i=1}^{n}
\frac{q\cdot g_i}{\lVert q \rVert \, \lVert g_i \rVert}
$$

Here, $q$ and $g_i$ represent embedding vectors, not raw text. This expression describes the mean cosine-similarity component; the final metric may also apply noncommittal-answer handling.

The configured strictness is a requested generation count, not a guarantee of the number actually returned.

### Values to Capture Next

To independently reproduce the result, capture:

- Requested and returned generation counts.
- Every generated question and its noncommittal flag.
- Original-question and generated-question embedding dimensions.
- Individual cosine similarities.
- Mean similarity before any adjustment.
- Any noncommittal adjustment.
- Final metric score.

Then independently calculate the score from the recorded intermediate values and compare it with Ragas' output.

## Current Experimental Status

**Status: Local smoke test successful; intermediate mathematical verification remains in progress.**

The standalone environment now installs through `uv`, and the experiment has completed successfully with Ragas `0.4.3`. The `langchain-community==0.4.1` workaround is part of the project dependency configuration.

The following limitations remain:

- Intermediate generated questions, embeddings, and cosine similarities have not yet been captured by the experiment script.
- The single-generation warning means the observed results do not verify three-generation aggregation.
- The `LangchainEmbeddingsWrapper` currently produces a deprecation warning. It does not prevent this experiment from running, but the embedding integration should be migrated and retested in a future change.

## Known Limitations

The two test cases are a small exploratory sample. They do not establish general metric quality, calibration, or reliability.

A nonzero similarity score for an unrelated answer does not automatically indicate a defect. Cosine similarity is a continuous semantic signal, not a binary relevance judgment.

When changing Ragas, LangChain dependencies, Ollama models, the embedding adapter, or metric configuration:

1. Update and review dependency constraints and the lockfile as appropriate.
2. Rerun the same test cases.
3. Record the dependency and model versions alongside the scores.
4. Investigate any change in generation count or scoring behavior before interpreting score differences.