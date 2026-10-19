# Answer Relevancy — Concepts and Techniques
- **Purpose:** Evaluate whether a generated answer addresses the user's question.
- **Focus:** Reusable techniques, mathematics, strengths, limitations, and QA validation.
- **Learning approach:** Understand intermediate results, independently verify calculations, and separate observations from assumptions.
- **Cost constraint:** Used local Ollama models only. No paid APIs or real API keys; pull another lightweight model only if genuinely required.

## 1. Definition and scope

Answer relevancy asks: **Does this response address the intent of the user's input?** It is not one universal algorithm. Evaluators approximate relevance via combinations of language-model judgments, semantic representations, and score aggregation.
Typical inputs are the original question and generated answer. A reference answer is not required by the standard approaches covered here, though other tasks (e.g., faithfulness, correctness) require reference or context.
An answer can be relevant but false, relevant but incomplete, or mostly relevant while padded with unrelated content. Relevancy alone does not establish correctness, grounding, completeness, or usefulness.

| Related concept | Question it answers | Why relevancy alone is insufficient |
|---|---|---|
| **Factual correctness** | Are the claims true? | A false claim can still be on topic. |
| **Faithfulness / groundedness** | Are claims supported by supplied evidence? | An on-topic claim may be unsupported by context. |
| **Completeness** | Are all requested parts answered? | All claims may be relevant while a required part is omitted. |
| **Contextual relevancy** | Is retrieved context relevant to the query? | This evaluates context, not the generated answer. |

## 2. Technique: LLM-based semantic decomposition

**Definition:** A language model turns a response into separate statements or claims for individual evaluation. This is semantic decomposition, not necessarily splitting on punctuation.

$$D(y)=[s_1,s_2,\ldots,s_m]$$

Here, $y$ is the answer, $s_i$ an extracted statement, and $m$ the model-produced statement count. Prompts and examples guide granularity; a list-of-strings schema alone does not define the correct number of claims.

**Example:** "Exercise improves cardiovascular health, strengthens muscles, and supports mental well-being" may become three claims even though it is one sentence.

**Strengths:** Intermediate claims are inspectable; mixed answers can be evaluated claim by claim.
**Limitations:** Over-splitting, under-splitting, duplicate or omitted claims, and model/prompt variation change which items are judged and the score denominator.
**Framework association:** Used in DeepEval's standard LLM-as-a-judge Answer Relevancy path.

## 3. Technique: LLM-as-a-judge classification

**Definition:** A prompted LLM classifies an item against a criterion, such as whether a statement addresses the original question.

$$v_i=J(q,s_i)$$

Here, $q$ is the question, $s_i$ a statement, and $J$ the prompted judge procedure. This is a model-generated judgment, not a deterministic mathematical operation.

**Example:** For "What are the benefits of regular exercise?", "Exercise can improve cardiovascular health" is relevant; "The Pacific Ocean is the largest ocean on Earth" is irrelevant.

**Strengths:** Flexible and explainable when reasons are returned.
**Limitations:** Judge may misunderstand intent, follow prompt bias, or vary across models/runs. A relevant verdict does not mean true, grounded, or complete.
**Framework association:** Statement-level judgments in DeepEval's standard approach.

## 4. Technique: verdict aggregation

For a yes/no statement-level score, divide passing verdicts by counted verdicts:

$$R_{\text{judge}}=\frac{P}{N}$$

- $P$: verdicts counted as passing/relevant.
- $N$: verdicts included in the denominator.

**Example:** Two relevant and one irrelevant statement give $R=2/3\approx0.667$.

```python
def relevancy_ratio(verdicts):
    if not verdicts:
        raise ValueError("No verdicts to score")
    return sum(v == "yes" for v in verdicts) / len(verdicts)
print(relevancy_ratio(["yes", "yes", "no"]))  # 0.6667...
```

This is a math illustration, not a complete replica of framework edge-case handling.

**Strengths:** Simple to reproduce from verdicts.
**Limitations:** All counted statements are weighted equally; a trivial aside counts as much as a major irrelevant claim. The score depends on decomposition and judge quality.
**Verified edge case (DeepEval shared helper):** literal `None` verdicts are filtered; if none remain, the helper returns `empty_score` (default `1.0`); in strict mode a below-threshold score can be clamped to zero.
**QA risk:** an empty or fully-filtered verdict list may score `1.0` at the helper level—a potential false positive. Test end-to-end; do not assume the helper alone determines the final pipeline score.

## 5. Technique: text embeddings

**Definition:** An embedding model maps text to a numeric vector intended to represent learned semantic features:

$$E(t)=\mathbf{x}\in\mathbb{R}^{d}$$

Here, $t$ is text and $d$ the embedding dimension. Interpret the vector as a whole; individual coordinates usually have no simple human-readable meaning.

**Observed local example:** `nomic-embed-text:latest` returned a 768-dimensional vector for "What are the benefits of regular exercise?" First five coordinates: approximately `[0.0091, 0.0885, -0.1764, 0.0283, 0.0017]`.

**Strengths:** Captures semantic relationships beyond exact word overlap; makes text numerically comparable.
**Limitations:** Similarity is not entailment or relevance proof; unrelated text can score nonzero; embeddings may miss negation, intent, or fine details. Results depend on the embedding model.
**Framework association:** Used by Ragas; not used by DeepEval's standard statement-ratio calculation.

## 6. Technique: cosine similarity

**Definition:** Cosine similarity measures the cosine of the angle between two nonzero vectors. It compares direction, not magnitude.

$$\cos(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}$$

The dot product is $\mathbf a\cdot\mathbf b=\sum_j a_jb_j$; the Euclidean norm is $\lVert\mathbf a\rVert=\sqrt{\sum_j a_j^2}$.

**Toy example:** For $\mathbf a=[1,0]$ and $\mathbf b=[0.8,0.6]$, the dot product is `0.8`, both norms are `1`, so cosine similarity is `0.8`. These are toy vectors, not model embeddings.

```python
from math import sqrt
def cosine_similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Vector dimensions must match")
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        raise ValueError("Cosine is undefined for a zero vector")
    return dot / (na * nb)
print(cosine_similarity([1, 0], [0.8, 0.6]))  # 0.8
```

Cosine similarity ranges from $-1$ to $1$ for nonzero vectors. Text embeddings often yield nonnegative scores, but this is not guaranteed.

**Limitation:** High similarity suggests semantic proximity; it does not prove the answer is correct, complete, or responsive.
**Framework association:** Used in the Ragas approach.

## 7. Technique: reverse-question generation

**Definition:** Give the answer to an LLM and ask it to generate a question the answer could answer. Compare that question with the user's original question.

**Example:** Answer "Albert Einstein was born in Germany" → generated question "Where was Albert Einstein born?" (illustrative, not from our experiment).

**Why use it?** Converts the answer into a question-shaped semantic representation, then estimates whether the reconstructed intent matches the original input. Differs from judging every answer statement individually.

**Strengths:** No reference answer needed; uses semantic alignment instead of binary claim classification.
**Limitations:** Generated questions can be broad, incomplete, or poorly phrased; may represent only one part of a multi-part answer. Depends on both generator and embedder.
**Framework association:** Central to Ragas Answer Relevancy.

## 8. Technique: mean aggregation and noncommittal adjustment

For original question $q$, generated questions $q_1,\ldots,q_N$, and embedding function $E$, let $E_o=E(q)$ and $E_{g_i}=E(q_i)$. Mean cosine similarity is:

$$R_{\text{mean}}=\frac{1}{N}\sum_{i=1}^{N}\cos(E_{g_i},E_o)$$

**Example:** Similarities `0.82`, `0.62`, `0.74` → $(0.82+0.62+0.74)/3\approx0.727$. These are invented similarity values, not from model embeddings.

Use the number of valid questions actually scored, not the requested generation count. In our experiment, `strictness=3` produced only one generation per case, so $N=1$ above—do not describe these as three-question averages.

The inspected Ragas implementation also flags **noncommittal** outputs—evasive, vague, or ambiguous responses (e.g., "not sure", "unclear", "I don't know")—and, if all generated items are noncommittal, forces the score to zero:

$$R_{\text{Ragas}}=R_{\text{mean}}\times\mathbb{1}[\text{not all items noncommittal}]$$

where $\mathbb{1}[\cdot]$ is the indicator function ($1$ if the condition holds, else $0$).

If every generated question string is empty, the inspected implementation returns `NaN` rather than computing cosine similarity.

**Strengths:** Multiple questions can probe different aspects of the answer.
**Limitations:** More generations cost more and need not be diverse; averaging can hide one poor match; the all-noncommittal rule overrides the similarity mean.

## 9. Compare the approaches

| Aspect | Statement judging (DeepEval standard) | Reverse-question similarity (Ragas) |
|---|---|---|
| Main question | Which answer statements address the input? | What questions could the answer answer, and how similar are they to the input? |
| Intermediate data | Statements and judge verdicts | Generated questions, noncommittal flags, vectors, cosine similarities |
| Score idea | Fraction of verdicts relevant | Mean cosine similarity, noncommittal adjustment |
| Main strength | Individual on/off-topic claims inspectable | Continuous semantic-alignment signal |
| Main risk | Decomposition and judge errors | Question-generation and embedding-similarity errors |
| Cannot prove | Truth, grounding, completeness | Truth, grounding, completeness |
| Cost (local) | One judge call per statement | One LLM call per generated question + embeddings |

Scores are not automatically calibrated across methods. `0.8` need not mean the same relevance level in both.

## 10. Local experiment findings

Both experiments used local Ollama models only; no paid API was used.

**Shared question:** "What are the benefits of regular exercise?"
- **Relevant answer:** "Regular exercise improves cardiovascular health, strengthens muscles, and supports mental well-being."
- **Irrelevant answer:** "The Pacific Ocean is the largest ocean on Earth."

**DeepEval 4.3.1:** Judge `gemma3n:e2b`, temperature `0`.

| Case | Intermediate result | Score |
|---|---|---:|
| Relevant | Three statements extracted, all `yes` | `1.0` |
| Irrelevant | One statement extracted, `no` | `0.0` |

One sentence decomposed into three statements in this run—evidence of semantic decomposition, not a guarantee across runs.

**Ragas 0.4.4.dev8+g298b68274:** Judge `gemma3n:e2b`, temperature `0`; embedder `nomic-embed-text:latest`; `strictness=3`. Direct embedding endpoint check returned a 768-dimensional vector.

| Case | Score |
|---|---:|
| Relevant exercise answer | `1.0000` |
| Irrelevant Pacific Ocean answer | `0.3071` |

**Unresolved generation warning:** Both cases logged `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` Generated questions, noncommittal flags, and individual cosine similarities were **not captured**, so the internal score path cannot yet be verified end-to-end. Until then, treat these as **single-generation observations**, not multi-generation averages.

**Threats to validity:**
- **Sample size.** Two hand-picked cases; not a validation study.
- **Version reproducibility.** The Ragas version is a development build (`.dev8`); others may not reproduce identical behavior.
- **No "related but different" control.** Both cases are semantically far apart. A partially related answer (e.g., "benefits of diet" for the exercise question) is needed to probe the semantic boundary.
- **No multi-part question.** Relevancy against a question requiring multiple distinct answers is untested.
- **Nonzero unrelated score is not automatically a defect.** Semantic similarity is a proxy, not a binary proof of relevance.

**Setup observed:** `langchain_openai.OpenAIEmbeddings` at `http://localhost:11434/v1` with `ollama` placeholder, wrapped in a deprecated `LangchainEmbeddingsWrapper`. Preserve as observed; any migration is a separate experiment.

## 11. QA checklist and first-principles exercises

**Test cases:** (1) direct on-topic answer, (2) off-topic answer, (3) relevant with irrelevant padding, (4) multi-part question with a part omitted, (5) relevant but false, (6) true facts that do not answer the question, (7) empty/short/vague answer, (8) compound sentences with alternative decompositions, (9) partially related answer (semantic boundary), (10) repeated runs for stability.

**Capture:** question, answer, expected behavior, intermediate statements or questions, verdicts or noncommittal flags, cosine values where applicable, final score, model/framework versions, configuration, warnings.

**First-principles exercises:**
- **Judge ratio:** Label a fixed statement list, count passing verdicts, divide by the denominator; test mixed and empty cases.
- **Cosine:** Compute dot product and norms; test identical, orthogonal, opposite, and zero vectors.
- **Ragas aggregation:** Average fixed similarities; separately test all-noncommittal and empty-question behavior.

Do not assume a universal threshold works across frameworks or datasets. Keep raw score, threshold decision, and strict-mode transformation distinct.

## 12. References

- Ragas Answer Relevancy: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/
- DeepEval Answer Relevancy: https://deepeval.com/docs/metrics-answer-relevancy
- Sentence Transformers — semantic textual similarity: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Hugging Face — sentence similarity: https://huggingface.co/tasks/sentence-similarity

Use docs for general context; use the exact local version and captured intermediate outputs when explaining observed behavior.