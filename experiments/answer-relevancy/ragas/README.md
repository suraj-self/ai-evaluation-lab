# Ragas — Answer Relevancy

## Purpose

Investigate Ragas Answer Relevancy through reverse-question generation, text embeddings, cosine similarity, mean aggregation, and noncommittal-answer handling.

The aim is to explain the final score from its intermediate values and independently reproduce its mathematics.

## Local Models

The observed experiment used:

- Judge: `gemma3n:e2b`
- Embedding model: `nomic-embed-text:latest`
- Ollama endpoint: `http://localhost:11434`
- Generation temperature: `0`
- Ragas `strictness`: `3`

Prepare the models:

```bash
ollama pull gemma3n:e2b
ollama pull nomic-embed-text:latest
ollama list
```

No paid API or real API key is required. The Python clients communicate with Ollama's local OpenAI-compatible endpoints.

## Current Experimental Status

**Status: Exploratory — generation behavior still under investigation.**

The initial experiment was run from a separate Ragas development checkout:

`0.4.4.dev8+g298b68274`

The local environment also needed a dependency workaround involving `langchain-community==0.4.1`. The environment and dependencies have not yet been packaged as a verified, self-contained installation for this experiment directory.

Do not treat this experiment as a guaranteed clone-and-run setup until its environment and dependency lock are established.

## Observed Results

The shared question was:

> What are the benefits of regular exercise?

| Case | Observed score |
|---|---:|
| Relevant exercise answer | 1.0000 |
| Irrelevant Pacific Ocean answer | 0.3071 |

Both cases produced this warning:

```text
LLM returned 1 generations instead of requested 3.
Proceeding with 1 generations.
```

Therefore, these results are single-generation observations, not verified three-question averages.

## Method to Investigate

The intended scoring path is:

1. Generate questions that the answer could address.
2. Embed the original question and generated questions.
3. Calculate cosine similarity for each pair.
4. Average the similarities.
5. Apply the noncommittal-answer adjustment.

The configured `strictness=3` requests three generations; it does not guarantee that three usable generations are returned.

For the next experiment, capture:

- Requested and returned generation counts.
- Every generated question and its noncommittal flag.
- The original-question and generated-question embedding dimensions.
- Individual cosine similarities.
- The mean similarity and any noncommittal adjustment.
- The final metric score.

Independently recalculate the final score from those captured values.

## Known Limitations

The two observed test cases are a small exploratory sample. They do not establish general metric quality or score calibration.

A nonzero similarity score for an unrelated answer does not automatically indicate a defect: cosine similarity is a continuous semantic signal, not a binary relevance judgment.

Document future dependency changes separately and rerun the same test cases when changing the adapter, model, or configuration.
