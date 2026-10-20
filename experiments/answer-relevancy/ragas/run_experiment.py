
"""Local Ragas Answer Relevancy check against Ollama.

Prerequisites:
    ollama pull gemma3n:e2b
    ollama pull nomic-embed-text:latest

Run from this directory:
    uv run python ragas_answer_relevancy.py
"""

import asyncio

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics._answer_relevance import AnswerRelevancy

LOCAL_URL = "http://localhost:11434/v1"
API_KEY = "ollama"


def build_metric() -> AnswerRelevancy:
    """Configure the local judge and embedding models."""
    client = OpenAI(api_key=API_KEY, base_url=LOCAL_URL)

    judge = llm_factory(
        model="gemma3n:e2b",
        provider="openai",
        client=client,
        temperature=0,
    )

    local_embeddings = OpenAIEmbeddings(
        model="nomic-embed-text:latest",
        base_url=LOCAL_URL,
        api_key=API_KEY,
        check_embedding_ctx_length=False,
    )

    embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    return AnswerRelevancy(
        llm=judge,
        embeddings=embeddings,
        strictness=3,
    )


async def score_cases(
    metric: AnswerRelevancy,
    question: str,
    cases: list[tuple[str, str]],
) -> None:
    for name, answer in cases:
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
        )
        score = await metric.single_turn_ascore(sample)

        print(f"\n{name}")
        print(f"Answer: {answer}")
        print(f"Ragas score: {score:.4f}")


async def main() -> None:
    metric = build_metric()

    question = "What are the benefits of regular exercise?"

    cases = [
        (
            "RELEVANT",
            "Regular exercise improves cardiovascular health, strengthens "
            "muscles, and supports mental well-being.",
        ),
        (
            "IRRELEVANT",
            "The Pacific Ocean is the largest ocean on Earth.",
        ),
    ]

    await score_cases(metric, question, cases)


if __name__ == "__main__":
    asyncio.run(main())
