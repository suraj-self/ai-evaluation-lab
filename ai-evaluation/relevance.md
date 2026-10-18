# Answer Relevancy

**Status:** Foundation / Mathematical understanding
**Category:** LLM evaluation / RAG evaluation
**Purpose:** Measure whether a generated response addresses the user's input.

## 1. Definition

Answer Relevancy measures how well a generated response addresses the intent of the user's question.

A response may be:
- Relevant but factually incorrect.
- Factually correct but irrelevant to the question.
- Relevant but incomplete.
- Mostly relevant but padded with unrelated information.

**Important:** Answer relevancy is not the same as factual correctness, faithfulness to retrieved context, or completeness. These require separate evaluation criteria.

## 2. What data does it require?

Typical inputs:
- `user_input`: the original user question.
- `response` / `actual_output`: the generated answer.

A reference answer is not required by the standard Ragas or DeepEval Answer Relevancy metrics described below.

The exact inputs and calculation depend on the evaluation framework and mode.

## 3. Method A — Ragas: Reverse-question embedding similarity

### Intuition

If an answer addresses the original question, questions reconstructed from that answer should have similar meaning to the original question.

### Algorithm

1. Give the generated response to an evaluation LLM.
2. Ask it to generate N artificial questions that the response could answer.
3. Embed the original question and each generated question.
4. Calculate cosine similarity between each generated-question embedding and the original-question embedding.
5. Average the similarity scores.

### Mathematical expression

\[
R_{\text{Ragas}} =
\frac{1}{N}\sum_{i=1}^{N}
\operatorname{cos}(E_{g_i}, E_o)
\]

Cosine similarity is:

\[
\operatorname{cos}(A,B) =
\frac{A \cdot B}{\|A\|\|B\|}
\]

Therefore:

\[
R_{\text{Ragas}} =
\frac{1}{N}\sum_{i=1}^{N}
\frac{E_{g_i}\cdot E_o}
{\|E_{g_i}\|\|E_o\|}
\]

### Variables

- \(R_{\text{Ragas}}\): mean answer-relevancy score.
- \(N\): number of artificial questions generated.
- \(E_{g_i}\): embedding vector for generated question i.
- \(E_o\): embedding vector for the original user input.
- \(A \cdot B\): dot product of two vectors.
- \(\|A\|\): magnitude (Euclidean norm) of vector A.

The documented default for N is 3 in this implementation; it is configurable.

### Interpretation

A higher mean similarity suggests the response captures the original question's intent.

Cosine similarity mathematically ranges from -1 to 1. Ragas notes that scores are usually between 0 and 1 in practice, but this is not guaranteed.

### Limitations

- Measures semantic alignment, not factual correctness.
- Depends on the question-generation LLM and embedding model.
- Generated questions may fail to represent important parts of the answer.
- Similarity can conceal missing details in multi-part questions.
- Results may vary with model choice and configuration.
- A high score alone does not prove the answer is complete or useful.

## 4. Method B — DeepEval: LLM-as-a-judge

### Intuition

Break the generated answer into statements and judge how many of those statements are relevant to the original input.

### Algorithm

1. Provide `input` and `actual_output`.
2. Use an evaluation LLM to extract statements from the answer.
3. Classify each extracted statement as relevant or irrelevant to the input.
4. Calculate the ratio of relevant statements to all extracted statements.

### Mathematical expression

\[
R_{\text{DeepEval}} =
\frac{S_{\text{relevant}}}{S_{\text{total}}}
\]

Where:

- \(S_{\text{relevant}}\): number of statements classified as relevant.
- \(S_{\text{total}}\): total number of extracted statements.
- \(R_{\text{DeepEval}}\): Answer Relevancy score in the documented LLM-as-a-judge mode.

For a non-empty set of statements, this ratio lies between 0 and 1.

### Example

Question: "What is the refund policy?"

Answer:
1. "Customers can request a refund within 30 days."
2. "Refunds are processed to the original payment method."
3. "Our company was founded in 2010."

If the judge marks statements 1 and 2 relevant and statement 3 irrelevant:

\[
R = \frac{2}{3} \approx 0.667
\]

This score reflects the proportion of relevant statements, not whether the policy is factually true or complete.

### Limitations

- Depends on statement extraction and the judge's classifications.
- Different judges or prompts can produce different scores.
- Statement splitting can affect the denominator and therefore the result.
- A response may omit a required policy detail while all its existing statements remain relevant.
- A relevant statement may still be false or unsupported.
- A ratio does not necessarily represent the importance or severity of irrelevant content.

**Mode warning:** DeepEval also supports hybrid and Jev-as-a-judge modes. Do not assume the ratio above describes every mode; check the implementation and documentation for the mode being used.

## 5. What Answer Relevancy does NOT establish

| Question | Is Answer Relevancy sufficient? |
|---|---|
| Does the answer address the question? | Yes, this is its primary purpose. |
| Is the answer factually correct? | No. |
| Is the answer supported by retrieved context? | No. |
| Did it include every required detail? | Not reliably. |
| Is the retrieved context itself relevant? | No; evaluate contextual relevancy separately. |

Useful complementary evaluations:
- **Faithfulness / groundedness:** Are claims supported by the supplied context?
- **Correctness:** Does the answer match reliable facts or an appropriate reference?
- **Completeness:** Are all required parts of the question answered?
- **Contextual relevancy:** Is the retrieved context relevant to the query?

Metric names and exact definitions vary by framework.

## 6. QA test design considerations

When testing an Answer Relevancy evaluator, include cases such as:

1. Direct, on-topic answer.
2. Completely off-topic answer.
3. On-topic answer containing irrelevant padding.
4. Multi-part question with one part omitted.
5. Relevant but factually false answer.
6. Correct facts that do not answer the question.
7. Empty or very short response.
8. Same answer evaluated with different judge models or configurations.

For each case, record:
- Input question.
- Generated answer.
- Expected qualitative behavior.
- Actual score and judge explanation.
- Framework, metric version, evaluation mode, and model.
- Whether the result supports the intended conclusion.

Do not assume one universal numeric threshold works across frameworks or datasets.

## 7. Implementation from first principles

Before using a package, implement each method separately.

### DeepEval-style ratio

1. Start with a fixed list of statements.
2. Label each statement relevant or irrelevant.
3. Count relevant statements.
4. Divide by total statements.
5. Test empty input and ambiguous statements explicitly.

This first version tests the mathematics only; it does not reproduce the LLM's statement extraction or judging behavior.

### Ragas-style similarity

1. Start with fixed original-question and generated-question embeddings.
2. Implement dot product and vector magnitude.
3. Calculate cosine similarity.
4. Average the similarities.
5. Test identical vectors, orthogonal vectors, and opposite vectors.

This first version tests the mathematics only; it does not reproduce the question-generation or embedding models.

## 8. Sources

### Ragas — Answer Relevancy
https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/

Supports the reverse-question generation method, cosine-similarity equation, variables, and score-range caveat.

### DeepEval — Answer Relevancy
https://deepeval.com/docs/metrics-answer-relevancy

Supports the statement-ratio equation, statement extraction and classification, required inputs, and differences between evaluation modes.

## 9. Findings and open questions

### Verified from documentation
- Ragas and DeepEval's standard Answer Relevancy implementations use different calculation procedures.
- Relevancy does not establish factual correctness.
- Results depend on the evaluation procedure and its configuration.

### Local Ragas experiment — 2026-10-18

**Observed setup**
- Framework: Ragas, executed from the local repository using `uv run --no-sync`.
- Judge: Ollama-hosted `gemma3n:e2b`, accessed through the local OpenAI-compatible endpoint `http://localhost:11434/v1`.
- Judge temperature: `0`.
- Embedding model: Ollama-hosted `nomic-embed-text:latest`, accessed through the same OpenAI-compatible endpoint.
- Ragas metric configuration: `AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3)`.
- The code wrapped `langchain_openai.OpenAIEmbeddings` in `LangchainEmbeddingsWrapper`.

**Dataset and observed scores**

| Case | Question | Answer | Observed Ragas score |
|---|---|---|---:|
| RELEVANT | What are the benefits of regular exercise? | Regular exercise improves cardiovascular health, strengthens muscles, and supports mental well-being. | 1.0000 |
| IRRELEVANT | What are the benefits of regular exercise? | The Pacific Ocean is the largest ocean on Earth. | 0.3071 |

**Runtime warning**
The run printed `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` for both cases. Although `strictness=3` was configured, the log therefore does **not** establish that three usable generated questions contributed to either score. Investigate the actual generated output and the current Ragas implementation before interpreting these scores as a three-question average.

**Deprecation warning**
`LangchainEmbeddingsWrapper` is deprecated in this Ragas version. The warning recommends using modern embedding providers instead. Do not change the experiment implementation silently: first inspect the current repository API, then record any migration as a separate, explicit setup change.

**Interpretation limits**
- These are two observations, not a validation study.
- The relevant answer scoring 1.0000 and the unrelated answer scoring 0.3071 are consistent with the intended direction for these examples, but do not prove general metric quality.
- A non-zero score for the unrelated answer is not automatically a bug; semantic embeddings can assign similarity to unrelated text, and the generated-question path may affect the result.
- The log does not include the generated questions, embedding vectors, cosine similarities, raw Ollama request/response bodies, latency, or model digest. Those are still required for full observability.
- Do not compare these Ragas values directly with DeepEval scores as if the numbers were calibrated on the same scale. The algorithms differ.

### Open questions
- Why did Ragas report one generation when `strictness=3` was configured?
- What exact generated question(s) were used for each case?
- Which embedding vectors and cosine similarity values produced the final scores?
- Are results stable across repeated runs with the same local model?
- How much does statement splitting affect the DeepEval-style ratio?
- Can either evaluator detect omitted parts of multi-part questions?
- How do relevancy, completeness, correctness, and faithfulness scores complement one another?

## 10. Implementation from first principles

Before relying on a package, implement each method separately.

### DeepEval-style ratio
1. Start with a fixed list of statements.
2. Label each statement relevant or irrelevant.
3. Count relevant statements.
4. Divide by total statements.
5. Test empty input and ambiguous statements explicitly.

This first version tests the mathematics only; it does not reproduce the LLM's statement extraction or judging behavior.

### Ragas-style similarity
1. Start with fixed original-question and generated-question embeddings.
2. Implement dot product and vector magnitude.
3. Calculate cosine similarity.
4. Average the similarities.
5. Test identical vectors, orthogonal vectors, and opposite vectors.

This first version tests the mathematics only; it does not reproduce the question-generation or embedding models.

## 11. Experiment log: exact run details

The submitted log shows the following implementation pattern:

```python
LOCAL_URL = "http://localhost:11434/v1"

client = OpenAI(api_key="ollama", base_url=LOCAL_URL)
judge = llm_factory(
    model="gemma3n:e2b",
    provider="openai",
    client=client,
    temperature=0,
)

local_embeddings = OpenAIEmbeddings(
    model="nomic-embed-text:latest",
    base_url=LOCAL_URL,
    api_key="ollama",
    check_embedding_ctx_length=False,
)
embeddings = LangchainEmbeddingsWrapper(local_embeddings)

metric = AnswerRelevancy(
    llm=judge,
    embeddings=embeddings,
    strictness=3,
)
```

Here, `"ollama"` is a placeholder API key used by the OpenAI-compatible client; the endpoint points to the local Ollama service. This run therefore does not, by itself, indicate a paid OpenAI API call.

**Reproducibility details still to capture**
- Git commit SHA for the Ragas checkout.
- Ragas, OpenAI client, LangChain OpenAI integration, and Python versions.
- Ollama server version and exact model tags/digests.
- Full raw model requests and responses, generated questions, embedding dimensions/vectors (or saved vectors), per-question cosine similarities, score, duration, and exceptions.
- Exact dataset file and a hash of its contents.
- Repeat count and whether repeated runs return identical intermediate outputs.

## 12. Next steps

1. Inspect the current `AnswerRelevancy` implementation and trace how `strictness` maps to the requested generation count and how returned generations are parsed.
2. Capture the generated question(s) for each case before changing any configuration.
3. Add raw Ollama request/response logging or instrument the framework/client path so prompts, outputs, and embedding results can be inspected.
4. Keep the dataset identical for Ragas and DeepEval; record each framework's own intermediate artifacts separately.
5. Only after the warning is understood, decide whether to migrate away from the deprecated embedding wrapper. Document the change so score differences are not mistakenly attributed only to the metric.

