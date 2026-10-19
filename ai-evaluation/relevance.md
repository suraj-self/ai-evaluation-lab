# Answer Relevancy — Concepts and Techniques

- **Purpose:** Evaluate whether a generated answer addresses the user's question.
- **Focus:** Reusable techniques, mathematics, strengths, limitations, and QA validation—not framework source-code walkthroughs.
- **Learning approach:** Understand intermediate results, independently verify calculations, and separate observations from assumptions.
- **Cost constraint:** Use local Ollama models only. No paid APIs or real API keys; pull another lightweight model only if genuinely required.

## 1. Definition and scope

Answer relevancy asks: **Does this response address the intent of the user's input?** It is not one universal algorithm. Evaluators approximate relevance using different combinations of language-model judgments, semantic representations, and score aggregation.
Typical inputs are the original question and generated answer. A reference answer is not required by the standard approaches covered here, though other evaluation tasks may require a reference or retrieved context.
An answer can be relevant but false, relevant but incomplete, or mostly relevant while padded with unrelated content. Relevancy alone does not establish correctness, grounding, completeness, or usefulness.

| Related concept | Question it answers | Why relevancy alone is insufficient |
|---|---|---|
| **Factual correctness** | Are the claims true? | A false claim can still be on topic. |
| **Faithfulness / groundedness** | Are claims supported by supplied evidence? | An on-topic claim may be unsupported by context. |
| **Completeness** | Are all requested parts answered? | All existing claims may be relevant while a required part is omitted. |
| **Contextual relevancy** | Is retrieved context relevant to the query? | This evaluates context, not solely the generated answer. |

## 2. Technique: LLM-based semantic decomposition

**Definition:** A language model turns a response into separate statements or claims for individual evaluation. This is semantic decomposition, not necessarily splitting on punctuation.

\[
D(y)=[s_1,s_2,\ldots,s_m]
\]
Here, \(y\) is the answer, \(s_i\) an extracted statement, and \(m\) the model-produced statement count. Prompts and examples guide the granularity; a list-of-strings schema alone does not define the correct number of claims.
**Example:** “Exercise improves cardiovascular health, strengthens muscles, and supports mental well-being” may become three claims even though it is one sentence.
**Why use it?** It enables claim-level judging and helps isolate unrelated padding within an otherwise relevant response.
**Strengths:** Intermediate claims are inspectable; mixed answers can be evaluated claim by claim. **Limitations:** Over-splitting, under-splitting, duplicate or omitted claims, and variation across models/prompts can change which items are judged and the score denominator. Inspect extracted statements during QA.
**Framework association:** Used by the standard LLM-as-a-judge Answer Relevancy path in DeepEval.
## 3. Technique: LLM-as-a-judge classification

**Definition:** A prompted LLM classifies an item against a criterion, such as whether a statement addresses the original question.

\[
v_i=J(q,s_i)
\]
Here, \(q\) is the user question, \(s_i\) a statement, and \(J\) the prompted judge procedure. This is a model-generated judgment, not a deterministic mathematical operation.
**Example:** For “What are the benefits of regular exercise?”, “Exercise can improve cardiovascular health” is likely relevant; “The Pacific Ocean is the largest ocean on Earth” is irrelevant.
**Why use it?** Natural-language judging handles varied wording without a manually written rule for every formulation. A judge may also provide a reason.
**Strengths:** Flexible and explainable when reasons are available. **Limitations:** The judge may misunderstand intent, follow prompt bias, or vary across models/runs. A relevant verdict does not mean true, grounded, or complete. Inspect prompts, verdicts, and reasons when debugging.
**Framework association:** Used for statement-level judgments in DeepEval's standard approach.
## 4. Technique: verdict aggregation

For a basic yes/no statement-level score, divide the number of relevant statements by the number of statements counted:

\[
R_{\text{judge}}=\frac{P}{N}
\]
- \(P\): verdicts counted as passing/relevant.
- \(N\): verdicts included in the denominator.

**Example:** Two relevant statements and one irrelevant statement give \(R=2/3\approx0.667\).

```python
def relevancy_ratio(verdicts):
    if not verdicts:
        raise ValueError("No verdicts to score")
    return sum(v == "yes" for v in verdicts) / len(verdicts)
print(relevancy_ratio(["yes", "yes", "no"]))  # 0.6667...
```

This is a math illustration, not a complete replica of framework edge-case handling.
**Why use it?** It condenses multiple local judgments into an interpretable number. **Strength:** Simple to reproduce from verdicts. **Limitations:** All counted statements have equal weight; a trivial aside may count as much as a major irrelevant claim. The result depends on decomposition and judge quality.
**Verified edge case:** The inspected DeepEval shared helper filters literal `None` entries, returns `empty_score` (default `1.0`) if no verdicts remain, and can clamp a below-threshold score to zero in strict mode. These helper behaviors do not prove that every empty answer scores 1.0; the full pipeline determines which verdicts are produced.
## 5. Technique: text embeddings

**Definition:** An embedding model maps text to a numeric vector intended to represent learned semantic features:

\[
E(t)=\mathbf{x}\in\mathbb{R}^{d}
\]
Here, \(t\) is text and \(d\) the embedding dimension. Interpret the vector as a whole; individual coordinates usually have no simple human-readable meaning.
**Observed local example:** `nomic-embed-text:latest` returned a 768-dimensional vector for “What are the benefits of regular exercise?” The first five coordinates were approximately `[0.0091, 0.0885, -0.1764, 0.0283, 0.0017]`.
**Why use it here?** Ragas compares embeddings of the original question and questions generated from the answer, using their similarity as a numerical proxy for semantic alignment.
**Strengths:** Captures semantic relationships beyond exact word overlap and makes text comparable numerically. **Limitations:** Similarity is not entailment or relevance proof; unrelated text can score nonzero, and embeddings may miss negation, intent, or fine details. Results depend on the embedding model.
**Framework association:** Embeddings are used by the Ragas approach studied here, not by the standard DeepEval statement-ratio calculation.
## 6. Technique: cosine similarity

**Definition:** Cosine similarity measures the cosine of the angle between two nonzero vectors. It compares direction rather than directly comparing vector magnitude.

\[
\operatorname{cos}(\mathbf a,\mathbf b)=\frac{\mathbf a\cdot\mathbf b}{\lVert\mathbf a\rVert\lVert\mathbf b\rVert}
\]
The dot product is \(\mathbf a\cdot\mathbf b=\sum_j a_jb_j\); the Euclidean norm is \(\lVert\mathbf a\rVert=\sqrt{\sum_j a_j^2}\).
**Toy example:** For \(\mathbf a=[1,0]\) and \(\mathbf b=[0.8,0.6]\), the dot product is `0.8`, both norms are `1`, so cosine similarity is `0.8`. These are toy vectors, not actual model embeddings.

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

Cosine similarity ranges from \(-1\) to \(1\) for nonzero vectors. Text embeddings often yield nonnegative scores in practice, but this is not guaranteed.
**Why use it here?** It gives a normalized comparison between original-question and generated-question embeddings. **Limitation:** High similarity suggests semantic proximity; it does not prove the answer is correct, complete, or responsive. **Framework association:** Used in the Ragas approach studied here.
## 7. Technique: reverse-question generation

**Definition:** Give the answer to an LLM and ask it to generate a question that the answer could answer. Compare that question with the user's original question.
**Example:** Answer: “Albert Einstein was born in Germany.” Possible generated question: “Where was Albert Einstein born?” This is an illustrative example, not an output from our exercise experiment.
**Why use it?** It converts the answer into a question-shaped semantic representation, then estimates whether the reconstructed intent matches the original input. It differs from judging every answer statement individually.
**Strengths:** No reference answer is needed; the approach uses semantic alignment instead of only binary claim classification. **Limitations:** Generated questions can be broad, incomplete, or poorly phrased and may represent only one part of a multi-part answer. The method depends on both generator and embedder.
**Framework association:** Central to the Ragas Answer Relevancy approach studied here.
## 8. Technique: mean aggregation and noncommittal adjustment

For original question \(q\), generated questions \(q_1,\ldots,q_N\), and embedding function \(E\), let \(E_o=E(q)\) and \(E_{g_i}=E(q_i)\). Mean cosine similarity is:

\[
R_{\text{mean}}=\frac{1}{N}\sum_{i=1}^{N}\operatorname{cos}(E_{g_i},E_o)
\]
**Example:** Similarities `0.82`, `0.62`, and `0.74` yield \((0.82+0.62+0.74)/3\approx0.727\).
A mean summarizes several comparisons, but use the number of valid questions actually scored—not merely the requested generation count.
The inspected Ragas implementation also flags evasive, vague, or ambiguous outputs as noncommittal. If all generated items are noncommittal, its adjustment makes the score zero:

\[
R_{\text{Ragas}}=R_{\text{mean}}\times\mathbf 1[\text{not all items are noncommittal}]
\]
If every generated question string is empty, the inspected implementation returns `NaN` rather than calculating cosine similarity.
**Strengths:** Multiple questions can probe different aspects of the answer. **Limitations:** More generations cost more and need not be diverse or better; averaging can hide one poor match; the all-noncommittal rule overrides the similarity mean.
## 9. Compare the approaches

| Aspect | Statement judging (DeepEval standard mode) | Reverse-question similarity (Ragas implementation studied) |
|---|---|---|
| Main question | Which answer statements address the input? | What questions could the answer answer, and how similar are they to the input? |
| Intermediate data | Statements and judge verdicts | Generated questions, noncommittal flags, vectors, cosine similarities |
| Score idea | Fraction of verdicts counted as relevant | Mean cosine similarity plus noncommittal adjustment |
| Main strength | Makes individual on-topic/off-topic claims inspectable | Gives a continuous semantic-alignment signal |
| Main risk | Decomposition and judge errors | Question-generation and embedding-similarity errors |
| Cannot prove | Truth, grounding, completeness | Truth, grounding, completeness |

These scores are not automatically calibrated to the same scale. A value of `0.8` need not mean the same level of relevance across methods.
## 10. Local experiment findings

Both experiments used local Ollama models only; no paid API was intended or required.
**Shared question:** “What are the benefits of regular exercise?”
- **Relevant answer:** “Regular exercise improves cardiovascular health, strengthens muscles, and supports mental well-being.”
- **Irrelevant answer:** “The Pacific Ocean is the largest ocean on Earth.”

**DeepEval 4.3.1:** Judge `gemma3n:e2b`, temperature `0`.

| Case | Observed intermediate result | Score |
|---|---|---:|
| Relevant | Three extracted statements, all judged `yes` | `1.0` |
| Irrelevant | One extracted statement, judged `no` | `0.0` |

The relevant answer was one sentence but decomposed into three statements in this run. That is evidence of semantic decomposition, not a guarantee that every run will be identical.
**Ragas 0.4.4.dev8+g298b68274:** Judge `gemma3n:e2b`, temperature `0`; embedder `nomic-embed-text:latest`; `strictness=3`. A direct embedding endpoint check returned a 768-dimensional vector.

| Case | Observed score |
|---|---:|
| Relevant exercise answer | `1.0000` |
| Irrelevant Pacific Ocean answer | `0.3071` |

**Unresolved generation warning:** Both cases logged `LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.` Do not describe these as three-question averages until the generation path and outputs are verified. Generated questions, flags, and individual cosine similarities were not captured.
The experiment used `langchain_openai.OpenAIEmbeddings` pointed at `http://localhost:11434/v1`, with `ollama` as a local placeholder, wrapped in a deprecated `LangchainEmbeddingsWrapper`. Keep this as the observed setup; any migration should be a separately recorded experiment. These two cases are observations, not a validation study. A nonzero score for an unrelated answer is not automatically a defect: semantic similarity is a proxy, not a binary proof of relevance.
## 11. QA checklist and first-principles exercises

Test: (1) direct on-topic answer, (2) off-topic answer, (3) relevant answer with irrelevant padding, (4) multi-part question with a part omitted, (5) relevant but false answer, (6) true facts that do not answer the question, (7) empty/short/vague answer, (8) compound sentences with alternative decompositions, and (9) repeated runs.
Capture the question, answer, expected behavior, intermediate statements or questions, verdicts or flags, cosine values where applicable, final score, model/framework versions, configuration, and warnings.

- **Judge ratio:** Label a fixed statement list, count passing verdicts, divide by the denominator, and test mixed and empty cases.
- **Cosine:** Calculate dot product and norms; test identical, orthogonal, opposite, and zero vectors.
- **Ragas aggregation:** Average fixed similarities; separately test all-noncommittal and empty-question behavior.

Do not assume a universal threshold works across frameworks or datasets. Keep raw score, threshold decision, and strict-mode transformation distinct.

## 12. References

- Ragas Answer Relevancy: https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/answer_relevance/
- DeepEval Answer Relevancy: https://deepeval.com/docs/metrics-answer-relevancy
- Sentence Transformers — semantic textual similarity: https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- Hugging Face — sentence similarity: https://huggingface.co/tasks/sentence-similarity

Use docs for general context; use the exact local version and captured intermediate outputs when explaining observed behavior.
