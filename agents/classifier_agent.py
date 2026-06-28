"""
EU AI Act Compliance Agent — Classifier Agent (Agent 1).

Analyses a natural-language description of an AI system and classifies it
against the EU AI Act risk tiers using Gemini and the project knowledge base.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from google import genai
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANNEX_III_PATH = PROJECT_ROOT / "knowledge_base" / "annex_iii.json"
RISK_MATRIX_PATH = PROJECT_ROOT / "knowledge_base" / "risk_matrix.json"
MODEL_NAME = "gemini-2.5-flash-lite"
MOCK_MODE = True

ANNEX_III_DATA: dict[str, Any] = {}
RISK_MATRIX_DATA: dict[str, Any] = {}


def load_knowledge_base() -> None:
    """Load annex_iii.json and risk_matrix.json into module-level caches."""
    global ANNEX_III_DATA, RISK_MATRIX_DATA

    for path, label in (
        (ANNEX_III_PATH, "Annex III"),
        (RISK_MATRIX_PATH, "risk matrix"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} file not found: {path}")
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if path == ANNEX_III_PATH:
            ANNEX_III_DATA = data
        else:
            RISK_MATRIX_DATA = data
        logger.info("Loaded %s from %s", label, path)


def _build_classification_prompt(system_description: str) -> str:
    """Build the user prompt sent to Gemini with full knowledge-base context."""
    classification_steps = RISK_MATRIX_DATA.get("classification_steps", [])
    risk_levels = RISK_MATRIX_DATA.get("risk_levels", {})

    return f"""You are an expert EU AI Act compliance classifier. Analyse the AI system description below and classify it according to the EU AI Act.

Follow this classification logic IN ORDER (stop at the first match):
{json.dumps(classification_steps, indent=2)}

Risk level definitions:
{json.dumps(risk_levels, indent=2)}

Full Annex III high-risk categories (knowledge base):
{json.dumps(ANNEX_III_DATA, indent=2)}

AI system to classify:
\"\"\"{system_description.strip()}\"\"\"

Important rules:
- Real-time remote biometric identification in publicly accessible spaces for law enforcement or general surveillance (e.g., facial recognition in a shopping mall) is PROHIBITED under Article 5 → UNACCEPTABLE_RISK.
- CV screening, hiring, recruitment, or worker monitoring tools match Annex III Employment (HR4) → HIGH_RISK.
- General product recommendation chatbots without high-risk use cases are LIMITED_RISK under Article 52 (transparency obligations for AI interaction).
- Only classify as HIGH_RISK if the system clearly falls under an Annex III category (HR1–HR8).

Respond with ONLY valid JSON (no markdown fences) using this exact schema:
{{
  "risk_tier": "UNACCEPTABLE_RISK | HIGH_RISK | LIMITED_RISK | MINIMAL_RISK",
  "confidence": "high | medium | low",
  "reasoning": "2-3 sentence explanation",
  "matched_category": "HR id and name if applicable, else null",
  "applicable_articles": ["Article 5", "Article 6", "Article 52", ...]
}}"""


def _parse_model_response(raw_text: str) -> dict[str, Any]:
    """Extract and parse a JSON object from the model's response text."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned non-JSON response: {raw_text[:500]}") from exc


def classify_system(system_description: str) -> dict[str, Any]:
    """
    Classify an AI system description using Gemini and the knowledge base.

    Returns a dict with risk_tier, confidence, reasoning, matched_category,
    and applicable_articles.
    """
    if MOCK_MODE:
        logger.info("[MOCK MODE] Bypassing classifier API call")
        return {
            "risk_tier": "HIGH_RISK",
            "confidence": "high",
            "reasoning": "Mocked classifier output for CV screening tool.",
            "matched_category": "HR4 Employment, Workers Management and Access to Self-Employment",
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"]
        }

    if not system_description or not system_description.strip():
        raise ValueError("system_description must be a non-empty string")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. Add it to your .env file."
        )

    client = genai.Client(api_key=api_key)
    prompt = _build_classification_prompt(system_description)

    logger.info("Sending classification request to Gemini (%s)", MODEL_NAME)
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            break
        except Exception as exc:
            last_exc = exc
            if "429" in str(exc) and attempt < 2:
                wait = 60
                logger.warning("Rate limited — retrying in %ds (attempt %d/3)", wait, attempt + 2)
                time.sleep(wait)
            else:
                logger.error("Gemini API error: %s", exc)
                raise
    else:
        raise last_exc  # type: ignore[misc]

    if not response.text:
        raise ValueError("Gemini returned an empty response")

    result = _parse_model_response(response.text)

    required_keys = {
        "risk_tier",
        "confidence",
        "reasoning",
        "matched_category",
        "applicable_articles",
    }
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Model response missing fields: {', '.join(sorted(missing))}")

    valid_tiers = {
        "UNACCEPTABLE_RISK",
        "HIGH_RISK",
        "LIMITED_RISK",
        "MINIMAL_RISK",
    }
    if result["risk_tier"] not in valid_tiers:
        raise ValueError(f"Invalid risk_tier: {result['risk_tier']}")

    logger.info("Classification result: %s", result["risk_tier"])
    return result


def _print_result(label: str, description: str, result: dict[str, Any]) -> None:
    """Pretty-print a single test classification result."""
    print(f"\n{'=' * 60}")
    print(f"Test: {label}")
    print(f"Description: {description}")
    print(f"Risk tier: {result['risk_tier']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Matched category: {result.get('matched_category')}")
    print(f"Applicable articles: {result.get('applicable_articles')}")
    print(f"Reasoning: {result['reasoning']}")


def main() -> None:
    """Run three example classifications to verify the classifier agent."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_knowledge_base()

    test_cases = [
        (
            "CV screening tool for hiring",
            "An AI-powered CV screening tool that automatically ranks and filters "
            "job applicants based on their résumés, work history, and skills "
            "for recruitment and hiring decisions.",
        ),
        (
            "Product recommendation chatbot",
            "A customer-facing product recommendation chatbot on an e-commerce "
            "website that suggests items based on browsing history and answers "
            "product questions in natural language.",
        ),
        (
            "Real-time facial recognition in shopping mall",
            "A real-time facial recognition system deployed in a shopping mall "
            "that identifies shoppers and tracks their movements across stores "
            "without their explicit consent.",
        ),
    ]

    print("EU AI Act Classifier Agent — Test Run")
    print(f"Annex III categories loaded: {len(ANNEX_III_DATA.get('high_risk_systems', []))}")

    failures = 0
    for label, description in test_cases:
        try:
            result = classify_system(description)
            _print_result(label, description, result)
        except Exception as exc:
            failures += 1
            logger.error("Classification failed for '%s': %s", label, exc)
            print(f"\n[-] FAILED: {label} — {exc}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
