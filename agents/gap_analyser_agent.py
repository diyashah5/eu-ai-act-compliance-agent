"""
EU AI Act Compliance Agent — Gap Analyser Agent (Agent 3).

Compares the legal requirements against the user's actual system answers,
identifying compliance gaps, calculating a compliance score, and suggesting fixes.
Supports a MOCK_MODE to run without Gemini API calls when quota is limited.
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
from requirements_agent import get_requirements, load_articles_obligations

# Mock Mode Configuration
MOCK_MODE = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "gemini-2.5-flash-lite"


def get_article_num(article_str: str) -> int | None:
    """Extract article number from article string."""
    if not article_str:
        return None
    match = re.search(r"\b(\d+)\b", article_str)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", article_str)
    return int(match.group(1)) if match else None


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


def _build_gap_analysis_prompt(requirements: dict[str, Any], user_responses: dict[str, str]) -> str:
    """Build prompt for Gemini to analyze gaps at the Article/Requirement level."""
    return f"""You are an expert EU AI Act compliance auditor.
Your task is to analyze the compliance gaps between the legal requirements and the user's answers.

Requirements and checklist questions:
{json.dumps(requirements, indent=2)}

User Responses (question -> answer):
{json.dumps(user_responses, indent=2)}

For each article/requirement where at least one checklist question was NOT answered "yes" (i.e. is "no", "partial", or "unknown"), identify a compliance gap for that article.
For articles/requirements where all checklist questions were answered "yes", list them in passed_checks.

Respond with ONLY valid JSON (no markdown fences) using this exact schema:
{{
  "gaps": [
    {{
      "article": "Article number (e.g. Article 9)",
      "requirement": "Requirement title (e.g. Risk management system)",
      "gap_description": "A comprehensive summary of the gaps for this article based on the user's responses (e.g., risk assessment is missing and testing is not implemented).",
      "fix_suggestion": "Actionable recommendations to fully resolve the gaps and comply with this article."
    }}
  ],
  "passed_checks": [
    "Requirement Title (Article X)"
  ]
}}"""


def analyse_gaps(
    requirements: dict[str, Any],
    user_responses: dict[str, str],
    risk_tier: str | None = None,
) -> dict[str, Any]:
    """
    Analyse the gaps between requirements and user responses.

    Args:
        requirements: The requirements dict from requirements_agent.py.
        user_responses: A dict mapping checklist questions to answers (yes/no/partial/unknown).
        risk_tier: The risk tier string (optional, will detect from requirements if None).
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE is True - returning mock gap analysis output without API calls")
        return {
            "compliance_score": 35,
            "gaps": [
                {
                    "article": "Article 9",
                    "requirement": "Risk management system",
                    "severity": "critical",
                    "gap_description": "No documented risk management process is established or maintained.",
                    "fix_suggestion": "Establish and document a systematic risk management process for the AI system."
                },
                {
                    "article": "Article 10",
                    "requirement": "Data governance",
                    "severity": "critical",
                    "gap_description": "Training, validation, and testing datasets are not fully documented or subject to proper governance.",
                    "fix_suggestion": "Implement strict data governance covering design, collection, and bias analysis."
                },
                {
                    "article": "Article 14",
                    "requirement": "Human oversight",
                    "severity": "critical",
                    "gap_description": "No mechanisms are in place to allow effective human override or oversight during operations.",
                    "fix_suggestion": "Integrate override functions and define clear human-in-the-loop oversight workflows."
                },
                {
                    "article": "Article 11",
                    "requirement": "Technical documentation",
                    "severity": "major",
                    "gap_description": "Technical documentation demonstrating compliance has not been drawn up.",
                    "fix_suggestion": "Compile comprehensive technical documentation detailing system architecture and compliance before launch."
                },
                {
                    "article": "Article 13",
                    "requirement": "Transparency",
                    "severity": "major",
                    "gap_description": "The system does not provide clear instructions or transparency to enable deployers to interpret outputs.",
                    "fix_suggestion": "Draft detailed instructions for use and user-facing documentation to ensure transparent operations."
                }
            ],
            "passed_checks": [
                "Accuracy & robustness (Article 15) - met",
                "Classification rules (Article 6) - met"
            ],
            "overall_status": "CRITICAL_GAPS"
        }

    # Detect or normalize risk tier
    if risk_tier is None:
        applicable = requirements.get("applicable_articles", [])
        if "Article 5" in applicable:
            risk_tier = "UNACCEPTABLE_RISK"
        elif not applicable and not requirements.get("requirements"):
            risk_tier = "MINIMAL_RISK"
        elif len(applicable) == 1 and "Article 52" in applicable:
            risk_tier = "LIMITED_RISK"
        else:
            risk_tier = "HIGH_RISK"

    # Fast-path for UNACCEPTABLE_RISK
    if risk_tier == "UNACCEPTABLE_RISK":
        logger.info("UNACCEPTABLE_RISK system - returning immediate BANNED status without API call")
        return {
            "compliance_score": 0,
            "gaps": [
                {
                    "article": "Article 5",
                    "requirement": "Prohibited AI practices",
                    "severity": "critical",
                    "gap_description": "The AI system description matches a prohibited category under Article 5.",
                    "fix_suggestion": "Stop all development and deployment. Banned systems have no lawful compliance path."
                }
            ],
            "passed_checks": [],
            "overall_status": "BANNED"
        }

    # Fast-path for MINIMAL_RISK
    if risk_tier == "MINIMAL_RISK":
        logger.info("MINIMAL_RISK system - returning immediate COMPLIANT status without API call")
        return {
            "compliance_score": 100,
            "gaps": [],
            "passed_checks": [
                "System is Minimal Risk: no mandatory requirements under the EU AI Act."
            ],
            "overall_status": "COMPLIANT"
        }

    # Ensure all checklist questions are answered (default to unknown if not specified)
    full_responses = user_responses.copy()
    for req in requirements.get("requirements", []):
        for q in req.get("checklist_questions", []):
            if q not in full_responses:
                full_responses[q] = "unknown"

    # Query Gemini to analyze gaps and generate descriptions/suggestions
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = _build_gap_analysis_prompt(requirements, full_responses)

    logger.info("Sending gap analysis request to Gemini (%s)", MODEL_NAME)
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

    gaps = result.get("gaps", [])
    passed_checks = result.get("passed_checks", [])

    # Group checklist questions and answers by Article to apply severity logic on Article-level
    article_to_answers: dict[str, list[str]] = {}
    article_to_req_title: dict[str, str] = {}
    for req in requirements.get("requirements", []):
        art = req.get("article", "")
        title = req.get("title", "")
        article_to_req_title[art] = title
        for q in req.get("checklist_questions", []):
            ans = full_responses.get(q, "unknown").lower()
            if art not in article_to_answers:
                article_to_answers[art] = []
            article_to_answers[art].append(ans)

    processed_gaps = []
    compliance_score = 100

    # Process gaps at the Article level
    for art, answers in article_to_answers.items():
        if "no" in answers:
            worst_ans = "no"
        elif "unknown" in answers:
            worst_ans = "unknown"
        elif "partial" in answers:
            worst_ans = "partial"
        else:
            worst_ans = "yes"

        if worst_ans == "yes":
            req_title = article_to_req_title.get(art, "Requirement")
            passed_msg = f"{req_title} ({art})"
            if passed_msg not in passed_checks:
                passed_checks.append(passed_msg)
            continue

        req_title = article_to_req_title.get(art, "Requirement")

        gemini_gap = next((g for g in gaps if g.get("article") == art), None)
        desc = gemini_gap.get("gap_description") if gemini_gap else None
        fix = gemini_gap.get("fix_suggestion") if gemini_gap else None

        if not desc:
            desc = f"Compliance check for {art} was not fully met. Answers included: {', '.join(set(answers))}."
        if not fix:
            fix = f"Review and implement all obligations required under {art}."

        # Severity logic:
        # - if article is 9, 10, or 14 and answer is no/unknown — mark as CRITICAL
        # - if article is 11 or 13 and answer is no — mark as MAJOR
        # - Everything else — MINOR
        art_num = get_article_num(art)
        severity = "minor"
        if art_num in (9, 10, 14) and worst_ans in ("no", "unknown"):
            severity = "critical"
        elif art_num in (11, 13) and worst_ans == "no":
            severity = "major"

        # Compliance score subtraction:
        # - CRITICAL: subtract 20
        # - MAJOR: subtract 10
        # - MINOR: subtract 5
        if severity == "critical":
            compliance_score -= 20
        elif severity == "major":
            compliance_score -= 10
        elif severity == "minor":
            compliance_score -= 5

        processed_gaps.append({
            "article": art,
            "requirement": req_title,
            "severity": severity,
            "gap_description": desc,
            "fix_suggestion": fix,
            "worst_answer": worst_ans
        })

    compliance_score = max(0, compliance_score)

    # Determine status
    if any(g["severity"] == "critical" for g in processed_gaps):
        overall_status = "CRITICAL_GAPS"
    elif compliance_score < 100:
        overall_status = "NEEDS_WORK"
    else:
        overall_status = "COMPLIANT"

    return {
        "compliance_score": compliance_score,
        "gaps": processed_gaps,
        "passed_checks": passed_checks,
        "overall_status": overall_status
    }


def format_gap_label(gap: dict[str, Any]) -> str:
    """Format the gap output string to match the requested Art prefix format."""
    art = gap.get("article", "")
    art_clean = art.replace("Article ", "Art.")
    art_num = get_article_num(art)

    if art_num == 9:
        return f"Risk management ({art_clean})"
    elif art_num == 10:
        return f"Data governance ({art_clean})"
    elif art_num == 11:
        return f"Technical documentation ({art_clean})"
    elif art_num == 13:
        return f"Transparency ({art_clean})"
    elif art_num == 14:
        return f"Human oversight ({art_clean})"
    elif art_num == 15:
        return f"Accuracy & robustness ({art_clean})"
    else:
        req = gap.get("requirement", "")
        return f"{req} ({art_clean})"


def main() -> None:
    """Run the full pipeline: classifier_agent -> requirements_agent -> gap_analyser_agent in sequence."""
    load_dotenv(PROJECT_ROOT / ".env")

    print("EU AI Act Compliance Agent — Full Pipeline Run (Classifier -> Requirements -> Gap Analyser)")

    # Test scenario description
    scenario_desc = (
        "An AI-powered CV screening tool that automatically ranks and filters "
        "job applicants based on their résumés, work history, and skills "
        "for recruitment and hiring decisions."
    )

    if MOCK_MODE:
        print("\n[MOCK MODE ACTIVE] Skipping real API calls for Classifier and Requirements agents...")
        classification = {
            "risk_tier": "HIGH_RISK",
            "confidence": "high",
            "reasoning": "Mocked classifier output for CV screening tool.",
            "matched_category": "HR4 Employment, Workers Management and Access to Self-Employment",
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"]
        }
        requirements = {
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"],
            "requirements": [
                {
                    "article": "Article 6",
                    "title": "Classification rules for high-risk AI systems",
                    "checklist_questions": ["Question 6.1", "Question 6.2"]
                },
                {
                    "article": "Article 9",
                    "title": "Risk management system",
                    "checklist_questions": ["Question 9.1", "Question 9.2"]
                },
                {
                    "article": "Article 10",
                    "title": "Data governance",
                    "checklist_questions": ["Question 10.1", "Question 10.2"]
                },
                {
                    "article": "Article 11",
                    "title": "Technical documentation",
                    "checklist_questions": ["Question 11.1", "Question 11.2"]
                },
                {
                    "article": "Article 13",
                    "title": "Transparency",
                    "checklist_questions": ["Question 13.1", "Question 13.2"]
                },
                {
                    "article": "Article 14",
                    "title": "Human oversight",
                    "checklist_questions": ["Question 14.1", "Question 14.2"]
                },
                {
                    "article": "Article 15",
                    "title": "Accuracy & robustness",
                    "checklist_questions": ["Question 15.1", "Question 15.2"]
                }
            ],
            "summary": "Mocked requirements list for high-risk HR tool."
        }
    else:
        load_classifier_kb()
        load_articles_obligations()

        print("\n[Step 1] Running Classifier Agent...")
        classification = classify_system(scenario_desc)
        print(f"Risk Tier: {classification['risk_tier']}")
        print(f"Matched Category: {classification['matched_category']}")

        print("\n[Step 2] Running Requirements Agent...")
        matched = classification.get("matched_category")
        if matched == "None" or matched == "null":
            matched = None
        requirements = get_requirements(
            risk_tier=classification["risk_tier"],
            matched_category=matched,
        )
        print(f"Applicable Articles: {requirements['applicable_articles']}")

    # Map checklist questions dynamically to test answers
    print("\n[Step 3] Mapping user responses for high-risk employment system...")
    user_responses = {}
    for req in requirements.get("requirements", []):
        art_str = req.get("article", "")
        art_num = get_article_num(art_str)
        for q in req.get("checklist_questions", []):
            q_lower = q.lower()
            if art_num == 9 or "risk" in q_lower:
                user_responses[q] = "no"
            elif art_num == 10 or "data" in q_lower or "dataset" in q_lower:
                user_responses[q] = "partial"
            elif art_num == 11 or "documentation" in q_lower or "technical" in q_lower:
                user_responses[q] = "no"
            elif art_num == 13 or "transparency" in q_lower or "instructions" in q_lower or "inform" in q_lower:
                user_responses[q] = "yes"
            elif art_num == 14 or "override" in q_lower or "human" in q_lower or "oversight" in q_lower:
                user_responses[q] = "no"
            elif art_num == 15 or "accuracy" in q_lower or "robustness" in q_lower or "cybersecurity" in q_lower or "metrics" in q_lower:
                user_responses[q] = "partial"
            else:
                user_responses[q] = "unknown"

    print("User responses mapped to checklist questions:")
    for q, ans in user_responses.items():
        print(f"  - {q}: {ans}")

    print("\n[Step 4] Running Gap Analyser Agent...")
    analysis = analyse_gaps(requirements, user_responses, risk_tier=classification["risk_tier"])

    print(f"\n{'=' * 60}")
    print(f"Compliance Score: {analysis['compliance_score']}/100")
    print(f"Overall Status: {analysis['overall_status']}")

    # Gather critical, major, minor gaps
    critical_gaps = []
    major_gaps = []
    minor_gaps = []

    for gap in analysis["gaps"]:
        label = format_gap_label(gap)
        sev = gap["severity"]
        if sev == "critical":
            critical_gaps.append(label)
        elif sev == "major":
            major_gaps.append(label)
        elif sev == "minor":
            minor_gaps.append(label)

    # De-duplicate lists
    critical_gaps = sorted(list(set(critical_gaps)))
    major_gaps = sorted(list(set(major_gaps)))
    minor_gaps = sorted(list(set(minor_gaps)))

    print(f"Critical gaps: {', '.join(critical_gaps) if critical_gaps else 'None'}")
    print(f"Major gaps: {', '.join(major_gaps) if major_gaps else 'None'}")
    print(f"Minor gaps: {', '.join(minor_gaps) if minor_gaps else 'None'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
