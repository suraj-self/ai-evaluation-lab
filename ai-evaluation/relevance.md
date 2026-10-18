# Answer Relevancy — Learning Notebook

- **Status:** Foundation / mathematical understanding
- **Category:** LLM evaluation / RAG evaluation
- **Purpose:** Understand how answer relevancy is defined, implemented, debugged, and validated before applying it to a local AI agent.
- **Learning approach:** Source-first, small experiments, independently verify mathematics, and record observed behavior separately from assumptions.
- **Cost constraint:** Use local Ollama models only. Do not use paid APIs or real API keys. Pull another lightweight local model only if the existing models cannot support a necessary experiment.

## 1. Learning objective

Learn Answer Relevancy deeply enough to explain and test its behavior without blindly trusting a library score. For each implementation, document:

1. What the metric measures and what it does not measure.
2. Required inputs and intermediate representations.
3. The mathematical expression and how each value is produced.
4. The source-code execution path.
5. Edge cases, limitations, and sources of variability.
6. Experiments that could confirm or falsify our expectations.

The current focus is DeepEval's implementation. The next session will study Ragas's different technique and investigate its generation-count warning.

## 2. Definition and required data

Answer Relevancy evaluates whether a generated response addresses the intent of the user's input.

Typical inputs:

- **Question / input:** the user's original question.
- **Response / actual output:** the answer being evaluated.

A reference answer is not required by the standard Answer Relevancy procedures discussed here. Exact fields and behavior depend on the framework, version, and mode.

An answer may be relevant but factually incorrect, relevant but incomplete, or mostly relevant while containing unrelated padding. Relevancy alone does not establish truth, faithfulness to supplied context, or completeness.

## 3. Distinguish related evaluation concepts

| Evaluation concept | Main question | Does Answer Relevancy establish it? |
|---|---|---|
| Answer relevancy | Does the response address the user's question? | This is its primary purpose. |
| Factual correctness | Are the claims true? | No. |
| Faithfulness / groundedness | Are the claims supported by supplied context or evidence? | No. |
| Completeness | Does the answer cover all required parts? | Not reliably. |
| Contextual relevancy | Is the retrieved context relevant to the query? | No; it evaluates context, not just the answer. |

Useful complementary evaluations include correctness against trusted facts or a reference, faithfulness against retrieved context, completeness for multi-part requests, and contextual relevancy for retrieval quality.

## 4. Method A — DeepEval 4.3.1: statement-level LLM judging

### 4.1 Intuition

DeepEval's standard Answer Relevancy path asks an evaluation LLM to break the answer into statements, then judge each statement's relevance to the original input. The metric aggregates the statement-level verdicts into a score.

### 4.2 Experiment setup

Observed local setup:

- **Repository / package:** local `~/deepeval` checkout.
- **DeepEval version:** `4.3.1`.
- **Python environment:** Poetry environment `(deepeval-py3.12)`; Python `3.12.3` in the recorded setup.
- **Judge:** local Ollama `gemma3n:e2b`.
- **Judge temperature:** `0`.
- **Metric configuration used in the experiment:** `AnswerRelevancyMetric(model=judge, threshold=0.7, include_reason=True, verbose_mode=True, async_mode=False)`.
- **Cost constraint:** local Ollama endpoint only; no paid API is used.

The two test cases use the same question:

> What are the benefits of regular exercise?

**Relevant answer:**

> Regular exercise improves cardiovascular health, strengthens muscles, and supports mental well-being.

**Irrelevant answer:**

> The Pacific Ocean is the largest ocean on Earth.

### 4.3 Execution path traced in `measure()`

The important sequence in the local implementation is:

```python
input = test_case.input
actual_output = test_case.actual_output

self.statements = self._generate_statements(
    actual_output, test_case.multimodal
)
self.verdicts = self._generate_verdicts(
    input, test_case.multimodal
)
self.score = self._calculate_score()
self.reason = self._generate_reason(input, test_case.multimodal)
```

In plain language:

1. Read the question and answer from the test case.
2. Extract statements from the answer.
3. Judge the extracted statements against the question.
4. Aggregate verdicts into a numeric score.
5. Generate a natural-language explanation of the evaluation.

The statement and verdict lists are stored on `self`, so later methods can use them. A type annotation such as `List[str]` describes the expected type; it does not itself split text or determine the number of statements.

### 4.4 Who splits the answer into statements?

The local `_generate_statements()` implementation builds a prompt and calls `generate_with_schema_and_extract(...)`. The configured judge LLM—Gemma in this experiment—generates the statement list. DeepEval orchestrates the process, structures the expected output, parses the result, and extracts the list.

The prompt printed in the experiment began with:

> Given the text, breakdown and generate a list of statements presented. Ambiguous statements and single words can be considered as statements, but only if outside of a coherent statement.

It also provides a laptop-product example. The example splits the sentence about a one-year warranty and 24/7 customer support into two statements. This suggests that the intended unit is a claim or fact, not necessarily a grammatical sentence.

The schema is:

```python
class Statements(BaseModel):
    statements: List[str]
```

This schema requires a `statements` field containing a list of strings. It does **not** declare a minimum or maximum list length. It therefore does not fix the number of statements or define their semantic granularity.

**Observed extraction results:**

For the one-sentence exercise answer, Gemma produced three statements:

1. `Regular exercise improves cardiovascular health.`
2. `It strengthens muscles.`
3. `It supports mental well-being.`

For the Pacific Ocean answer, Gemma produced one statement:

1. `The Pacific Ocean is the largest ocean on Earth.`

The observed sentence counts were one for each answer, but the extracted statement counts were three and one, respectively. This demonstrates that extraction is not simple splitting on sentence punctuation. It is model-driven semantic decomposition guided by the prompt and example. It does not guarantee that every run or judge will make the same decomposition.

**QA implication:** Statement decomposition affects which items are judged and the denominator of a ratio-based score. Inspect the prompt and extracted statements before interpreting a surprising score.

### 4.5 Verdict generation

DeepEval next calls `_generate_verdicts(input, test_case.multimodal)`. The question is supplied here, while the statements are already stored on the metric object.

Observed relevant case:

- `Regular exercise improves cardiovascular health.` → `yes`
- `It strengthens muscles.` → `yes`
- `It supports mental well-being.` → `yes`

Observed irrelevant case:

- `The Pacific Ocean is the largest ocean on Earth.` → `no`
- The judge's reason: `The statement about the Pacific Ocean is irrelevant to the benefits of regular exercise.`

The relevant verdicts had `reason: null` in this run. That is an observed output detail, not a rule that all relevant verdicts must have a null reason.

A `yes` verdict means that the judge considers a statement relevant to the input; it does not mean that the statement is true.

### 4.6 Score calculation and mathematical expression

The local `_calculate_score()` method delegates to the shared `score_qag_verdicts()` helper. The inspected helper does the following:

1. Filters out entries that are literally `None`.
2. If no verdict entries remain, returns `empty_score`, whose default is `1`.
3. Counts verdicts accepted by `normalize_qag_verdict(v.verdict, passing)`.
4. Divides the count by the number of remaining verdict entries.
5. If `strict_mode` is enabled and the score is below `metric.threshold`, clamps the score to zero.

For a non-empty verdict list, let:

- \(N\) = number of verdict entries remaining after filtering.
- \(P\) = number of those verdicts considered passing by the normalizer and `passing` configuration.

Then the raw score is:

\[
R_{\text{raw}} = \frac{P}{N}
\]

For Answer Relevancy, `yes` is the intended passing judgment. Confirm the exact `passing` argument in the local `_calculate_score()` call when revisiting the source.

If strict mode is enabled, the helper applies this additional rule to a non-empty list:

\[
R_{\text{final}} =
\begin{cases}
0, & \text{if strict mode is enabled and } R_{\text{raw}} < \text{threshold}\\
R_{\text{raw}}, & \text{otherwise}
\end{cases}
\]

The raw score and the pass/fail `success` property are different concepts. The score is numerical; threshold logic determines whether the result passes. Strict-mode clamping is an additional behavior and should not be confused with ordinary threshold-based success.

**Observed cases:**

| Case | Verdicts | Manual calculation | Observed score |
|---|---|---:|---:|
| Relevant exercise answer | yes, yes, yes | \(3/3=1.0\) | `1.0` |
| Irrelevant Pacific Ocean answer | no | \(0/1=0.0\) | `0.0` |
| Hypothetical mixed answer | yes, yes, no | \(2/3\approx0.6667\) | Not run; manual illustration only |

Each counted statement contributes equally to this fraction. The helper's default empty score of `1` deserves an explicit edge-case test, but it does **not**, by itself, establish what the full metric will do for every empty-answer scenario; that depends on the rest of the pipeline.

### 4.7 Reason generation

After calculating the score, the metric calls `_generate_reason(...)`. The reason is a natural-language explanation produced through the evaluation pipeline. It is not the mathematical score and should not be treated as independent proof that the answer is correct, complete, or well grounded.

Observed reasons:

- **Relevant case:** `The answer is highly relevant to the input. It directly addresses the question about the benefits of regular exercise and provides a comprehensive explanation. There are no irrelevant statements present, resulting in a score of 1.00.`
- **Irrelevant case:** `The answer is currently at a 0.00 relevance score because it includes a statement about the Pacific Ocean, which is irrelevant to the question about the benefits of regular exercise. The irrelevance of this statement prevents the response from fully addressing the prompt.`

These are outputs from the local judge and should be treated as generated explanations, not ground truth.

### 4.8 DeepEval experiment results

| Case | Extracted statements | Relevant verdicts | Total verdicts | Score | Passed at threshold 0.7? |
|---|---:|---:|---:|---:|---|
| Relevant exercise answer | 3 | 3 | 3 | `1.0` | Yes |
| Irrelevant Pacific Ocean answer | 1 | 0 | 1 | `0.0` | No |

These results agree with the observed verdicts and the fraction-of-passing-verdicts calculation. Two cases are an initial sanity check, not a validation study.

### 4.9 DeepEval limitations and QA risks

- **Judge dependence:** another judge model or prompt can classify statements differently.
- **Extraction dependence:** alternative decomposition changes which statements are scored and can affect the denominator.
- **Equal weighting:** a ratio does not capture the relative importance or severity of irrelevant statements.
- **Completeness gap:** an answer can consist entirely of relevant statements and still omit a required part of a multi-part question.
- **Correctness gap:** a statement can be relevant but false.
- **Grounding gap:** relevance does not prove that a claim is supported by supplied context.
- **Empty-list behavior:** the shared helper defaults to a score of `1` when no verdict entries remain; test the full metric path for empty and very short answers.
- **Threshold and strict mode:** record both settings and distinguish the raw ratio from any strict-mode clamp and pass/fail outcome.
- **Mode differences:** DeepEval may offer other Answer Relevancy modes. Do not assume that the standard statement-ratio path describes every mode; record the chosen mode and inspect its implementation.

## 5. Method B — Ragas: reverse-question embedding similarity

This is the focus for the next learning session. The current notes preserve the earlier experiment so we can pick it up without rerunning or guessing.

### 5.1 Intuition and algorithm

If an answer addresses the original question, questions reconstructed from that answer may have semantics similar to the original question.

The implementation path inspected in the local Ragas checkout (`0.4.4.dev8+g298b68274`) follows this general sequence:

1. Give the response to an evaluation LLM.
2. Ask it to generate one or more artificial questions that the response could answer and classify whether the answer is noncommittal.
3. Embed the original input and each generated question.
4. Calculate cosine similarity between each generated-question embedding and the original-question embedding.
5. Average the similarities and apply the noncommittal adjustment.

The inspected implementation has `strictness=3` by default/configuration. The experiment below warned that only one generation was returned despite requesting three; therefore, the observed run did not establish a three-question average.

### 5.2 Mathematical expression

Let \(N\) be the number of generated questions actually used, \(E_o\) the embedding of the original input, and \(E_{g_i}\) the embedding for generated question \(i\). The mean cosine similarity is:

\[
R_{\text{mean}} = \frac{1}{N}\sum_{i=1}^{N}
\operatorname{cos}(E_{g_i}, E_o)
\]

where:

\[
\operatorname{cos}(A,B) =
\frac{A\cdot B}{\|A\|\|B\|}
\]

In the inspected implementation, the mean is multiplied by zero if **all** generated results are marked noncommittal; otherwise the mean is retained. If \(A\) denotes the event that all generated results are noncommittal, this can be written as:

\[
R_{\text{Ragas}} = R_{\text{mean}}\times\mathbf{1}[\neg A]
\]

The cosine similarity of two nonzero vectors mathematically ranges from \(-1\) to \(1\). Ragas notes that scores are usually between zero and one in practice, but that range is not a mathematical guarantee for cosine similarity.

### 5.3 Local Ragas experiment — 2026-10-18

**Observed setup:**

- Repository: local `~/ragas` checkout, branch `learning/answer-relevancy`.
- Ragas version: `0.4.4.dev8+g298b68274`.
- Judge: local Ollama `gemma3n:e2b` through `http://localhost:11434/v1`.
- Judge temperature: `0`.
- Embedder: local Ollama `nomic-embed-text:latest` through `http://localhost:11434/v1`.
- The Ollama embedding endpoint returned a vector with `768` dimensions in a direct endpoint test.
- Metric configuration: `AnswerRelevancy(llm=judge, embeddings=embeddings, strictness=3)`.
- The run used `langchain_openai.OpenAIEmbeddings` with the local endpoint and wrapped it in `LangchainEmbeddingsWrapper`.
- The placeholder string `ollama` was supplied to the OpenAI-compatible client; the base URL was localhost. It is not a paid OpenAI credential, and the intended calls were to the local Ollama server.

**Observed scores:**

| Case | Question | Answer | Observed Ragas score |
|---|---|---|---:|
| Relevant | What are the benefits of regular exercise? | Regular exercise improves cardiovascular health, strengthens muscles, and supports mental well-being. | `1.0000` |
| Irrelevant | What are the benefits of regular exercise? | The Pacific Ocean is the largest ocean on Earth. | `0.3071` |

**Generation-count warning:** For both cases, the run printed:

```text
LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
```

Although `strictness=3` was configured, the run's warning indicates that only one generation was returned. Do not interpret these scores as evidence of a three-question average.

**Important setup warning:** `LangchainEmbeddingsWrapper` emitted a deprecation warning in this Ragas checkout. Do not silently change the setup; first inspect the current provider API and determine whether migration is necessary. If changed, record it as a separate experiment setup change.

**Interpretation limits:**

- These are two observations, not a validation study.
- A nonzero score for the unrelated answer is not automatically a defect: embedding similarity can be nonzero for text that is not relevant, and generated-question behavior also affects the score.
- The run did not capture generated questions, noncommittal flags, individual cosine similarities, embedding vectors used in the metric, timing, or full model request/response traces.
- Do not compare Ragas and DeepEval scores as if the numeric scales were calibrated to mean the same thing. Their algorithms differ.

### 5.4 Next-session investigation: why was only one generation returned?

The next session should resolve this before making conclusions about the configured `strictness=3`.

Questions to answer from the exact local source:

1. How does `PydanticPrompt.generate_multiple()` handle `n` for the selected LLM class?
2. Does the configured `llm_factory()` return an `InstructorBaseRagasLLM` that follows a single-generation path, or a client with true multi-generation support?
3. At what point does Ragas log that it received one generation instead of three?
4. Can the current locally available Ollama model/client configuration return multiple distinct structured generations through a supported local-only path?
5. How do we capture the exact generated question(s) and `noncommittal` flags before embeddings are calculated?

Start with these source searches from `~/ragas`:

```bash
rg -n "def generate_multiple|LLM returned .* generations|is_langchain_llm|InstructorBaseRagasLLM" src/ragas/prompt src/ragas
```

Then inspect the matching method and follow the branch actually selected by the local configured judge. Keep the experiment local-only. Do not install LiteLLM, pull another model, or change dependencies unless source inspection shows a necessary reason and the trade-off is made explicit.

## 6. QA test design

Test each evaluator with controlled cases and record the intermediate values, not only the final score:

1. Direct, on-topic answer.
2. Completely off-topic answer.
3. On-topic answer with irrelevant padding.
4. Multi-part question with one required part omitted.
5. Relevant but factually false answer.
6. Factually correct information that does not answer the question.
7. Empty and very short responses.
8. Compound sentences that could be decomposed in different ways.
9. Vague or noncommittal responses such as “I don't know.”
10. Repeated runs and, later, alternate judge configurations to examine stability.

For every case, record the input, response, qualitative expectation, actual intermediate outputs, score, explanation, framework version, evaluation mode, model names/configuration, and any warning or exception. Do not assume one universal numeric threshold works across frameworks or datasets.

## 7. First-principles implementation exercises

These are mathematical exercises, not replacements for the full LLM pipelines.

### DeepEval-style ratio

1. Start with a fixed list of statements.
2. Label each statement relevant or irrelevant.
3. Count the verdicts that count as passing.
4. Divide by the total counted verdicts.
5. Test mixed cases, an empty list, and threshold/strict-mode behavior separately.

### Ragas-style similarity

1. Start with fixed original-question and generated-question embeddings.
2. Implement dot product and vector magnitude.
3. Calculate cosine similarity.
4. Average the similarities.
5. Apply the noncommittal adjustment.
6. Test identical, orthogonal, and opposite nonzero vectors, and verify what happens for all-noncommittal outputs.

## 8. Sources

### Ragas — Answer Relevancy

https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/

Reference for reverse-question generation, cosine-similarity scoring, configurable strictness, and noncommittal handling. The local source remains authoritative for the checked-out experiment version.

### DeepEval — Answer Relevancy

https://deepeval.com/docs/metrics-answer-relevancy

Reference for statement extraction, relevance verdicts, score interpretation, and the framework's supported modes. The local source remains authoritative for DeepEval `4.3.1` behavior in this experiment.

## 9. Findings ledger

### Verified so far

- DeepEval and Ragas use different standard Answer Relevancy procedures.
- DeepEval 4.3.1 in the local experiment uses LLM-driven statement extraction followed by statement-level relevance verdicts and score aggregation.
- DeepEval's `Statements` schema is `statements: List[str]` and does not set a fixed statement count.
- The exercise answer produced three statements, all judged relevant; the Pacific Ocean answer produced one statement judged irrelevant.
- The observed DeepEval scores were `1.0` and `0.0`, consistent with the passing-verdict fraction for these test cases.
- The shared DeepEval score helper filters literal `None` entries, returns `empty_score` (default `1`) if no verdicts remain, computes a passing-verdict fraction otherwise, and can apply a strict-mode clamp.
- The local Ragas experiment returned `1.0000` and `0.3071` for the same two cases, but warned that only one generation was returned despite requesting three.
- Both experiments used local Ollama models; no paid API key was used.
- Relevance does not establish factual correctness, faithfulness to context, or completeness.

### Open questions

- Ragas: Why did the selected generation path produce one generation when `strictness=3` was requested?
- Ragas: Which exact generated question(s), noncommittal flags, and cosine similarities produced the observed scores?
- Ragas: Can the current supported local-only setup produce and use three generations, and what changes would that require?
- DeepEval: How does statement decomposition vary over repeated runs or with a different local judge?
- DeepEval: What are the full metric-pipeline results for empty answers and empty verdict lists?
- Both: How reliably do the metrics expose incomplete answers, irrelevant padding, and relevant-but-false claims?
- Both: How stable are the intermediate outputs and scores across repeated local runs?

## 10. Next session plan — Ragas only

**Goal:** Understand Ragas's Answer Relevancy implementation at code level and explain its result from intermediate values, not just from the final score.

1. Inspect the local `PydanticPrompt.generate_multiple()` method and trace why one generation was returned when three were requested.
2. Follow the selected LLM adapter path and identify the exact call that requests and parses generations.
3. Add minimal temporary diagnostics to capture generated questions and `noncommittal` flags.
4. Capture the embedding/cosine calculation path and independently verify the score mathematically.
5. Decide whether the current local-only configuration supports three generations. Do not add dependencies or pull another model until there is an evidence-based reason.
6. Update this notebook with source locations, actual intermediate outputs, the score derivation, limitations, and any remaining uncertainty.

**Starting command:**

```bash
cd ~/ragas
rg -n "def generate_multiple|LLM returned .* generations|is_langchain_llm|InstructorBaseRagasLLM" src/ragas/prompt src/ragas
```
