"""Run a small Answer Relevancy experiment with a local Ollama judge only."""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE.parent / "cases.json"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:e2b")
THRESHOLD = 0.7
TEMPERATURE = 0


def verify_local_ollama(base_url: str, model_name: str) -> None:
    """Require a loopback Ollama endpoint and verify the model is installed."""
    host = urlparse(base_url).hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            f"Local-only policy: OLLAMA_BASE_URL must use localhost/loopback, got {base_url!r}"
        )

    try:
        with urlopen(f"{base_url}/api/tags", timeout=5) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not contact local Ollama at {base_url}. Start Ollama and retry."
        ) from exc

    installed = {item.get("name") for item in payload.get("models", [])}
    if model_name not in installed:
        choices = ", ".join(sorted(name for name in installed if name)) or "(none found)"
        raise RuntimeError(
            f"Ollama model {model_name!r} is not installed. Available models: {choices}.\n"
            f"Install it with: ollama pull {model_name}"
        )


def load_cases() -> list[dict[str, str]]:
    with CASES_FILE.open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Expected a non-empty JSON list in {CASES_FILE}")

    for index, case in enumerate(cases, start=1):
        for field in ("name", "input", "actual_output"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                raise ValueError(f"Case {index} must have a non-empty string field {field!r}")
    return cases


def main() -> int:
    try:
        verify_local_ollama(OLLAMA_BASE_URL, OLLAMA_MODEL)
        cases = load_cases()
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"PRE-FLIGHT FAILED: {exc}", file=sys.stderr)
        return 2

    try:
        deepeval_version = version("deepeval")
    except PackageNotFoundError:
        print("DeepEval is not installed in the active Python environment.", file=sys.stderr)
        return 2

    print(f"DeepEval version: {deepeval_version}")
    print(f"Ollama endpoint:  {OLLAMA_BASE_URL} (local only)")
    print(f"Judge model:      {OLLAMA_MODEL}")
    print(f"Temperature:      {TEMPERATURE}")
    print(f"Threshold:        {THRESHOLD}")
    print(f"Test cases:       {len(cases)}")

    judge = OllamaModel(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
    )

    for case in cases:
        test_case = LLMTestCase(
            input=case["input"],
            actual_output=case["actual_output"],
        )
        metric = AnswerRelevancyMetric(
            model=judge,
            threshold=THRESHOLD,
            include_reason=True,
            verbose_mode=True,
            async_mode=False,
        )

        print(f"\n--- {case['name']} ---")
        print("Input:", test_case.input)
        print("Actual output:", test_case.actual_output)
        metric.measure(test_case)
        print("Score:", metric.score)
        print("Passed threshold:", metric.success)
        print("Reason:", metric.reason)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())