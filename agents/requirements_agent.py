"""
EU AI Act Compliance Agent — Requirements Agent (Agent 2).

Takes a risk tier (from Agent 1) and returns the specific legal obligations
that apply, using the articles knowledge base and Gemini where needed.
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

import google.generativeai as genai
from dotenv import load_dotenv

from classifier_agent import classify_system, load_knowledge_base as load_classifier_kb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTICLES_PATH = PROJECT_ROOT / "knowledge_base" / "articles_obligations.json"
MODEL_NAME = "gemini-2.5-flash-lite"
MOCK_MODE = True

ARTICLES_DATA: dict[str, Any] = {}

VALID_TIERS = {
    "UNACCEPTABLE_RISK",
    "HIGH_RISK",
    "LIMITED_RISK",
    "MINIMAL_RISK",
}


def load_articles_obligations() -> None:
    """Load articles_obligations.json into a module-level cache."""
    global ARTICLES_DATA

    if not ARTICLES_PATH.exists():
        raise FileNotFoundError(f"Articles file not found: {ARTICLES_PATH}")

    with open(ARTICLES_PATH, encoding="utf-8") as handle:
        ARTICLES_DATA = json.load(handle)

    count = len(ARTICLES_DATA.get("articles", []))
    logger.info("Loaded %d articles from %s", count, ARTICLES_PATH)


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


def _unacceptable_response() -> dict[str, Any]:
    """Return a fixed response for prohibited AI systems (no API call)."""
    return {
        "applicable_articles": ["Article 5"],
        "requirements": [
            {
                "article": "Article 5",
                "title": "Prohibited AI practices",
                "specific_requirements": [
                    "This AI system falls under prohibited practices under the EU AI Act.",
                    "The system must NOT be placed on the market or put into service.",
                    "There is no compliance path — development and deployment must cease.",
                ],
                "checklist_questions": [
                    "Has all development and deployment of this system been halted?",
                    "Has legal counsel confirmed the prohibition applies?",
                ],
            }
        ],
        "summary": (
            "This system is banned under Article 5 of the EU AI Act. Prohibited AI "
            "practices may not be placed on the market or put into service under any "
            "circumstances. There is no compliance path — the only lawful action is "
            "to discontinue the system."
        ),
    }


def _minimal_risk_response() -> dict[str, Any]:
    """Return a fixed response for minimal-risk systems (no API call)."""
    return {
        "applicable_articles": [],
        "requirements": [],
        "summary": (
            "This AI system is classified as minimal risk under the EU AI Act. "
            "There are no mandatory legal requirements. Providers are encouraged "
            "to voluntarily apply codes of conduct and best practices."
        ),
    }


def _build_requirements_prompt(risk_tier: str, matched_category: str | None) -> str:
    """Build the Gemini prompt for HIGH_RISK and LIMITED_RISK systems."""
    category_line = matched_category or "Not specified"
    tier_guidance = {
        "HIGH_RISK": (
            "This is a HIGH-RISK system under Annex III. Apply Chapter 2 obligations "
            "(Articles 9–15): risk management, data governance, technical documentation, "
            "transparency, human oversight, accuracy/robustness, and conformity assessment."
        ),
        "LIMITED_RISK": (
            "This is a LIMITED-RISK system. Focus primarily on Article 52 transparency "
            "obligations (inform users they are interacting with AI)."
        ),
    }

    return f"""You are an expert EU AI Act compliance advisor. Based on the knowledge base below, generate the specific legal requirements that apply to this AI system.

Risk tier: {risk_tier}
Matched Annex III category: {category_line}

Guidance: {tier_guidance.get(risk_tier, "")}

Articles knowledge base (articles_obligations.json):
{json.dumps(ARTICLES_DATA, indent=2)}

For HIGH_RISK employment (HR4) systems, applicable articles typically include 6, 9, 10, 11, 13, 14, and 15.
For LIMITED_RISK chatbots, Article 52 is the primary obligation.

Respond with ONLY valid JSON (no markdown fences) using this exact schema:
{{
  "applicable_articles": ["Article 9", "Article 10", ...],
  "requirements": [
    {{
      "article": "Article 9",
      "title": "Risk management system",
      "specific_requirements": ["requirement 1", "requirement 2"],
      "checklist_questions": ["question 1?", "question 2?"]
    }}
  ],
  "summary": "One paragraph explaining what this system must do to comply with the EU AI Act."
}}

Be specific to the risk tier and matched category. Include 2-4 checklist questions per requirement article."""


def get_requirements(
    risk_tier: str,
    matched_category: str | None = None,
) -> dict[str, Any]:
    """
    Return legal requirements for a classified AI system.

    Args:
        risk_tier: One of UNACCEPTABLE_RISK, HIGH_RISK, LIMITED_RISK, MINIMAL_RISK.
        matched_category: Annex III category id/name from Agent 1 (e.g. HR4).

    Returns:
        Dict with applicable_articles, requirements, and summary.
    """
    if risk_tier not in VALID_TIERS:
        raise ValueError(
            f"Invalid risk_tier '{risk_tier}'. Must be one of: {', '.join(sorted(VALID_TIERS))}"
        )

    if MOCK_MODE:
        logger.info("[MOCK MODE] Bypassing requirements API call")
        return {
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"],
            "requirements": [
                {
                    "article": "Article 6",
                    "title": "Classification rules for high-risk AI systems",
                    "specific_requirements": [
                        "The AI system meets the criteria specified in Annex III, category HR4, making it a high-risk AI system.",
                        "A conformity assessment must be completed for the AI system before it is placed on the market or put into service."
                    ],
                    "checklist_questions": [
                        "Does the system clearly meet the criteria for high-risk AI as defined in Annex III, specifically for employment and workers management?",
                        "Has a conformity assessment been completed for this high-risk AI system?"
                    ]
                },
                {
                    "article": "Article 9",
                    "title": "Risk management system",
                    "specific_requirements": [
                        "Establish, implement, document, and maintain a comprehensive risk management system.",
                        "Identify and analyze all known and reasonably foreseeable risks associated with the AI system.",
                        "Adopt suitable risk mitigation measures, including testing and post-market monitoring."
                    ],
                    "checklist_questions": [
                        "Is there a documented risk management system in place for this AI system?",
                        "Have all potential risks related to employment decisions, worker management, and access to self-employment been identified and assessed?",
                        "Are effective risk mitigation measures, including testing and post-market surveillance, actively implemented?"
                    ]
                },
                {
                    "article": "Article 10",
                    "title": "Data and data governance",
                    "specific_requirements": [
                        "Establish rigorous data governance covering design, collection, and bias analysis.",
                        "Ensure datasets are relevant, representative, and free of discriminatory bias."
                    ],
                    "checklist_questions": [
                        "Are the training, validation, and testing datasets for this AI system subject to clear data governance policies?",
                        "Do data governance practices adequately address data collection, preparation, and the potential for bias?",
                        "Has a comprehensive bias analysis been conducted on the datasets used, particularly concerning protected characteristics relevant to employment?"
                    ]
                },
                {
                    "article": "Article 11",
                    "title": "Technical documentation",
                    "specific_requirements": [
                        "Draw up comprehensive technical documentation before placement on the market.",
                        "Ensure the documentation demonstrates compliance with Chapter 2 obligations."
                    ],
                    "checklist_questions": [
                        "Is complete technical documentation available and up-to-date for this high-risk AI system?",
                        "Does the documentation include a detailed description of the system's functionality, intended use, and its risk management measures?",
                        "Does the documentation clearly demonstrate how the system complies with all Chapter 2 obligations for high-risk AI?"
                    ]
                },
                {
                    "article": "Article 13",
                    "title": "Transparency and provision of information",
                    "specific_requirements": [
                        "Ensure the system's operation is sufficiently transparent for deployers to interpret outputs.",
                        "Accompany the system with clear, accessible instructions for use."
                    ],
                    "checklist_questions": [
                        "Can deployers effectively interpret the outputs and decisions of this AI system in relation to employment matters?",
                        "Are instructions for use provided to deployers that are clear, understandable, and readily accessible?",
                        "Does the system's design facilitate understanding of how it arrives at its employment-related recommendations or decisions?"
                    ]
                },
                {
                    "article": "Article 14",
                    "title": "Human oversight",
                    "specific_requirements": [
                        "Design the system to enable effective human oversight during its period of use.",
                        "Implement override and interruption mechanisms."
                    ],
                    "checklist_questions": [
                        "Are there effective mechanisms for human oversight of the AI system's operation in employment contexts?",
                        "Can users (deployers) understand the AI system's capabilities, limitations, and the reasoning behind its employment-related outputs?",
                        "Are there clear procedures and capabilities to override or interrupt the system if its operation leads to potentially unfair or non-compliant employment decisions?"
                    ]
                },
                {
                    "article": "Article 15",
                    "title": "Accuracy, robustness and cybersecurity",
                    "specific_requirements": [
                        "Achieve appropriate levels of accuracy, robustness, and cybersecurity.",
                        "Ensure resilience against errors, faults, and unauthorized alterations."
                    ],
                    "checklist_questions": [
                        "Does the AI system demonstrate an appropriate level of accuracy and robustness for its intended use in employment decisions?",
                        "Are there measures in place to ensure the system's resilience against errors, faults, and inconsistencies (e.g., redundancy, backups)?",
                        "Is the AI system adequately protected against cybersecurity threats that could compromise its performance or lead to unauthorized alterations in employment contexts?"
                    ]
                }
            ],
            "summary": "Mocked requirements list for high-risk HR tool."
        }

    if risk_tier == "UNACCEPTABLE_RISK":
        logger.info("UNACCEPTABLE_RISK — returning Article 5 ban (no API call)")
        return _unacceptable_response()

    if risk_tier == "MINIMAL_RISK":
        logger.info("MINIMAL_RISK — returning voluntary guidance only (no API call)")
        return _minimal_risk_response()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = _build_requirements_prompt(risk_tier, matched_category)

    logger.info(
        "Sending requirements request to Gemini (%s) for %s",
        MODEL_NAME,
        risk_tier,
    )
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            response = model.generate_content(prompt)
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

    required_keys = {"applicable_articles", "requirements", "summary"}
    missing = required_keys - set(result.keys())
    if missing:
        raise ValueError(f"Model response missing fields: {', '.join(sorted(missing))}")

    logger.info(
        "Requirements generated: %d articles, %d requirement blocks",
        len(result.get("applicable_articles", [])),
        len(result.get("requirements", [])),
    )
    return result


def _print_requirements(label: str, classification: dict[str, Any], requirements: dict[str, Any]) -> None:
    """Pretty-print classification and requirements for a test scenario."""
    print(f"\n{'=' * 60}")
    print(f"Scenario: {label}")
    print(f"Risk tier: {classification['risk_tier']}")
    print(f"Matched category: {classification.get('matched_category')}")
    print(f"\nArticles applicable: {requirements.get('applicable_articles')}")
    print("\nRequirements:")
    for req in requirements.get("requirements", []):
        print(f"  - {req.get('article')}: {req.get('title')}")
        for item in req.get("specific_requirements", []):
            print(f"      • {item}")
    print(f"\nSummary: {requirements.get('summary')}")


def main() -> None:
    """Run Agent 1 then Agent 2 in sequence for three test scenarios."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_classifier_kb()
    load_articles_obligations()

    test_cases = [
        (
            "HIGH_RISK employment system",
            "An AI-powered CV screening tool that automatically ranks and filters "
            "job applicants based on their résumés, work history, and skills "
            "for recruitment and hiring decisions.",
        ),
        (
            "LIMITED_RISK chatbot",
            "A customer-facing product recommendation chatbot on an e-commerce "
            "website that suggests items based on browsing history and answers "
            "product questions in natural language.",
        ),
        (
            "UNACCEPTABLE_RISK facial recognition",
            "A real-time facial recognition system deployed in a shopping mall "
            "that identifies shoppers and tracks their movements across stores "
            "without their explicit consent.",
        ),
    ]

    print("EU AI Act Compliance Agent — Agent 1 + Agent 2 Pipeline")
    print(f"Articles loaded: {len(ARTICLES_DATA.get('articles', []))}")

    failures = 0
    for label, description in test_cases:
        try:
            print(f"\n--- Running pipeline for: {label} ---")
            classification = classify_system(description)
            logger.info(
                "Agent 1 result: %s (%s)",
                classification["risk_tier"],
                classification.get("matched_category"),
            )

            matched = classification.get("matched_category")
            if matched == "None" or matched == "null":
                matched = None

            requirements = get_requirements(
                risk_tier=classification["risk_tier"],
                matched_category=matched,
            )
            _print_requirements(label, classification, requirements)

            if label != test_cases[-1][0]:
                time.sleep(15)

        except Exception as exc:
            failures += 1
            logger.error("Pipeline failed for '%s': %s", label, exc)
            print(f"\n[-] FAILED: {label} — {exc}")

    if failures:
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("All pipeline tests completed successfully.")


if __name__ == "__main__":
    main()
