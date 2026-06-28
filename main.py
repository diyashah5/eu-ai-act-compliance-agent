"""
EU AI Act Compliance Agent - Master Pipeline (main.py).

Connects all 4 agents into a single pipeline function that can be called
by the UI layer. Includes audit logging, input sanitisation, and error handling.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Setup paths so we can import from agents/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
AGENTS_DIR = PROJECT_ROOT / "agents"
sys.path.insert(0, str(AGENTS_DIR))

# ---------------------------------------------------------------------------
# Import agent modules
# ---------------------------------------------------------------------------
from agents import classifier_agent  # type: ignore[import-untyped]
from agents import requirements_agent  # type: ignore[import-untyped]
from agents import gap_analyser_agent  # type: ignore[import-untyped]
from agents import report_generator_agent  # type: ignore[import-untyped]

from agents.classifier_agent import classify_system, load_knowledge_base as load_classifier_kb
from agents.requirements_agent import get_requirements, load_articles_obligations
from agents.gap_analyser_agent import analyse_gaps, get_article_num
from agents.report_generator_agent import generate_report, generate_pdf_report

# ---------------------------------------------------------------------------
# MCP Client - retrieves knowledge base via the MCP server instead of local JSON
# ---------------------------------------------------------------------------
from mcp_server.client import get_kb_data_sync

# ---------------------------------------------------------------------------
# Mock Mode - set True to bypass all Gemini API calls
# ---------------------------------------------------------------------------
MOCK_MODE = True

classifier_agent.MOCK_MODE = MOCK_MODE
requirements_agent.MOCK_MODE = MOCK_MODE
gap_analyser_agent.MOCK_MODE = MOCK_MODE
report_generator_agent.MOCK_MODE = MOCK_MODE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
SESSION_LOG_PATH = LOGS_DIR / "session_log.json"

INJECTION_PATTERNS = [
    r"IGNORE\s+INSTRUCTIONS",
    r"SYSTEM\s+OVERRIDE",
    r"BYPASS",
    r"DISREGARD\s+PREVIOUS",
    r"FORGET\s+EVERYTHING",
    r"IGNORE\s+ALL\s+RULES",
    r"OVERRIDE\s+SAFETY",
    r"JAILBREAK",
]


# ---------------------------------------------------------------------------
# Input Sanitisation
# ---------------------------------------------------------------------------
def sanitise_input(text: str) -> str | None:
    """
    Check for prompt injection patterns in the input text.

    Returns None if the input is safe, or the matched pattern string if
    a potential injection attempt is detected.
    """
    if not text:
        return None
    upper = text.upper()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, upper):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Session Logger
# ---------------------------------------------------------------------------
def append_session_log(entry: dict[str, Any]) -> None:
    """Append a pipeline run entry to logs/session_log.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if SESSION_LOG_PATH.exists():
        try:
            with open(SESSION_LOG_PATH, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Corrupted session log - starting fresh")
            existing = []

    existing.append(entry)

    with open(SESSION_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    logger.info("Session log updated (%d entries) -> %s", len(existing), SESSION_LOG_PATH)


# ---------------------------------------------------------------------------
# Mock data generators for different system types
# ---------------------------------------------------------------------------
def _mock_classifier(system_type: str) -> dict[str, Any]:
    """Return mock classifier result based on system type keyword."""
    if "hiring" in system_type or "cv" in system_type or "recruit" in system_type:
        return {
            "risk_tier": "HIGH_RISK",
            "confidence": "high",
            "reasoning": "CV screening tool matches Annex III HR4 (Employment).",
            "matched_category": "HR4 Employment, Workers Management and Access to Self-Employment",
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"],
        }
    elif "medical" in system_type or "diagnos" in system_type or "health" in system_type:
        return {
            "risk_tier": "HIGH_RISK",
            "confidence": "high",
            "reasoning": "Medical diagnosis AI matches Annex III HR1 (Biometrics/Health).",
            "matched_category": "HR1 Biometrics and Health",
            "applicable_articles": ["Article 6", "Article 9", "Article 10", "Article 11", "Article 13", "Article 14", "Article 15"],
        }
    elif "recommend" in system_type or "chatbot" in system_type or "product" in system_type:
        return {
            "risk_tier": "LIMITED_RISK",
            "confidence": "high",
            "reasoning": "Product recommendation chatbot falls under Article 52 transparency obligations.",
            "matched_category": None,
            "applicable_articles": ["Article 52"],
        }
    else:
        return {
            "risk_tier": "MINIMAL_RISK",
            "confidence": "medium",
            "reasoning": "System does not match any high-risk or limited-risk category.",
            "matched_category": None,
            "applicable_articles": [],
        }


def _mock_requirements(risk_tier: str) -> dict[str, Any]:
    """Return mock requirements based on risk tier."""
    if risk_tier == "LIMITED_RISK":
        return {
            "applicable_articles": ["Article 52"],
            "requirements": [
                {
                    "article": "Article 52",
                    "title": "Transparency obligations",
                    "specific_requirements": [
                        "Inform users they are interacting with an AI system.",
                        "Ensure AI-generated content is labelled where applicable.",
                    ],
                    "checklist_questions": [
                        "Are users clearly informed they are interacting with an AI system?",
                        "Is AI-generated content properly labelled?",
                    ],
                }
            ],
            "summary": "Limited-risk system must meet Article 52 transparency obligations.",
        }
    elif risk_tier == "MINIMAL_RISK":
        return {
            "applicable_articles": [],
            "requirements": [],
            "summary": "Minimal risk - no mandatory requirements.",
        }
    else:
        # HIGH_RISK - delegate to requirements_agent mock
        return get_requirements(risk_tier=risk_tier)


def _mock_gap_analysis(
    risk_tier: str,
    requirements: dict[str, Any],
    user_responses: dict[str, str],
) -> dict[str, Any]:
    """Return mock gap analysis tuned by risk tier."""
    if risk_tier == "LIMITED_RISK":
        has_gap = any(v.lower() != "yes" for v in user_responses.values())
        if has_gap:
            return {
                "compliance_score": 85,
                "gaps": [
                    {
                        "article": "Article 52",
                        "requirement": "Transparency obligations",
                        "severity": "minor",
                        "gap_description": "AI-generated content labelling is not fully implemented.",
                        "fix_suggestion": "Add clear AI disclosure labels to all generated outputs.",
                    }
                ],
                "passed_checks": [
                    "Users are informed they are interacting with AI (Article 52)."
                ],
                "overall_status": "NEEDS_WORK",
            }
        return {
            "compliance_score": 100,
            "gaps": [],
            "passed_checks": ["All Article 52 transparency checks passed."],
            "overall_status": "COMPLIANT",
        }
    elif risk_tier == "MINIMAL_RISK":
        return {
            "compliance_score": 100,
            "gaps": [],
            "passed_checks": ["Minimal risk - no mandatory requirements."],
            "overall_status": "COMPLIANT",
        }
    # HIGH_RISK - delegate to gap_analyser mock
    return analyse_gaps(requirements, user_responses, risk_tier=risk_tier)


def _mock_report(
    classifier_result: dict[str, Any],
    requirements_result: dict[str, Any],
    gap_result: dict[str, Any],
    system_description: str,
) -> dict[str, Any]:
    """Return mock report tuned by gap analysis results."""
    score = gap_result.get("compliance_score", 0)
    status = gap_result.get("overall_status", "NEEDS_WORK")
    tier = classifier_result.get("risk_tier", "HIGH_RISK")

    critical_actions = []
    major_actions = []
    minor_actions = []

    for gap in gap_result.get("gaps", []):
        sev = gap.get("severity", "minor")
        action_item = {
            "action": gap.get("fix_suggestion", "Address compliance gap."),
            "article": gap.get("article", "Unknown"),
            "deadline": "1 month" if sev == "critical" else ("2 months" if sev == "major" else "3 months"),
            "priority": 1 if sev == "critical" else (2 if sev == "major" else 3),
        }
        if sev == "critical":
            critical_actions.append(action_item)
        elif sev == "major":
            major_actions.append(action_item)
        else:
            minor_actions.append(action_item)

    return {
        "report_title": f"EU AI Act Compliance Report - {system_description[:50]}",
        "executive_summary": f"Compliance audit for: {system_description[:100]}. Risk tier: {tier}. Score: {score}/100. Status: {status}.",
        "risk_tier": tier,
        "compliance_score": score,
        "overall_status": status,
        "critical_actions": critical_actions,
        "major_actions": major_actions,
        "minor_actions": minor_actions,
        "estimated_compliance_timeline": "3-4 months" if score < 50 else ("1-2 months" if score < 100 else "N/A"),
        "passed_checks": gap_result.get("passed_checks", []),
        "disclaimer": "This report does not constitute legal advice. Providers are responsible for full regulatory compliance.",
    }


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------
def run_compliance_check(
    system_description: str,
    user_responses: dict[str, str],
) -> dict[str, Any]:
    """
    Run the full 4-agent compliance pipeline.

    Args:
        system_description: Natural language description of the AI system.
        user_responses: Dict mapping checklist questions to yes/no/partial/unknown.

    Returns:
        Combined result dict with classifier_result, requirements_result,
        gap_analysis_result, report_result, pipeline_status, and timestamp.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "timestamp": timestamp,
        "system_description": system_description,
        "classifier_result": None,
        "requirements_result": None,
        "gap_analysis_result": None,
        "report_result": None,
        "pipeline_status": "failed",
    }

    # --- Input Sanitisation ---
    injection = sanitise_input(system_description)
    if injection:
        logger.error("Prompt injection attempt detected: pattern=%s", injection)
        result["error"] = f"Input rejected: potential prompt injection detected (matched: {injection})"
        _log_session(result)
        return result

    desc_lower = system_description.lower()

    try:
        # --- Agent 1: Classifier ---
        logger.info("Agent 1 (Classifier) starting...")
        if MOCK_MODE:
            classifier_result = _mock_classifier(desc_lower)
        else:
            classifier_result = classify_system(system_description)
        result["classifier_result"] = classifier_result
        logger.info("Agent 1 complete: %s", classifier_result.get("risk_tier"))

        risk_tier = classifier_result["risk_tier"]
        matched_category = classifier_result.get("matched_category")

        # --- Agent 2: Requirements ---
        logger.info("Agent 2 (Requirements) starting...")
        if MOCK_MODE:
            requirements_result = _mock_requirements(risk_tier)
        else:
            requirements_result = get_requirements(
                risk_tier=risk_tier,
                matched_category=matched_category,
            )
        result["requirements_result"] = requirements_result
        logger.info(
            "Agent 2 complete: %d articles",
            len(requirements_result.get("applicable_articles", [])),
        )

        # --- Build user_responses dynamically if empty ---
        effective_responses = user_responses.copy()
        if not effective_responses:
            for req in requirements_result.get("requirements", []):
                for q in req.get("checklist_questions", []):
                    effective_responses[q] = "unknown"

        # --- Agent 3: Gap Analyser ---
        logger.info("Agent 3 (Gap Analyser) starting...")
        if MOCK_MODE:
            gap_result = _mock_gap_analysis(risk_tier, requirements_result, effective_responses)
        else:
            gap_result = analyse_gaps(requirements_result, effective_responses, risk_tier=risk_tier)
        result["gap_analysis_result"] = gap_result
        logger.info(
            "Agent 3 complete: score=%s, status=%s",
            gap_result.get("compliance_score"),
            gap_result.get("overall_status"),
        )

        # --- Agent 4: Report Generator ---
        logger.info("Agent 4 (Report Generator) starting...")
        if MOCK_MODE:
            report_result = _mock_report(
                classifier_result, requirements_result, gap_result, system_description,
            )
        else:
            report_result = generate_report(
                classifier_result=classifier_result,
                requirements_result=requirements_result,
                gap_analysis_result=gap_result,
                system_description=system_description,
            )

        # Generate PDF
        pdf_path = str(LOGS_DIR / "compliance_report.pdf")
        generate_pdf_report(report_result, pdf_path)

        result["report_result"] = report_result
        result["pipeline_status"] = "success"
        logger.info("Agent 4 complete. Pipeline finished successfully.")

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        result["error"] = str(exc)
        result["pipeline_status"] = "failed"

    # --- Session Logging ---
    _log_session(result)

    return result


def _log_session(result: dict[str, Any]) -> None:
    """Extract key fields and append to session log."""
    gap = result.get("gap_analysis_result") or {}
    classifier = result.get("classifier_result") or {}

    entry = {
        "timestamp": result.get("timestamp"),
        "system_description_summary": (result.get("system_description") or "")[:100],
        "risk_tier": classifier.get("risk_tier"),
        "compliance_score": gap.get("compliance_score"),
        "overall_status": gap.get("overall_status"),
        "pipeline_status": result.get("pipeline_status"),
    }
    if result.get("error"):
        entry["error"] = result["error"]

    try:
        append_session_log(entry)
    except Exception as exc:
        logger.error("Failed to write session log: %s", exc)


# ---------------------------------------------------------------------------
# Main - test with 3 different AI systems
# ---------------------------------------------------------------------------
def main() -> None:
    """Test the pipeline with 3 different AI system scenarios."""
    load_dotenv(PROJECT_ROOT / ".env")

    # --- Load knowledge base via MCP server ---
    logger.info("Fetching knowledge base from MCP server...")
    try:
        kb_data = get_kb_data_sync()
        logger.info(
            "MCP KB loaded: %d high-risk systems, %d articles, %d classification steps",
            len(kb_data.get("annex_iii", {}).get("high_risk_systems", [])),
            len(kb_data.get("articles_obligations", {}).get("articles", [])),
            len(kb_data.get("risk_matrix", {}).get("classification_steps", [])),
        )
    except Exception as exc:
        logger.warning("MCP server unavailable (%s), falling back to local KB files", exc)
        kb_data = None

    if not MOCK_MODE:
        load_classifier_kb()
        load_articles_obligations()

    print("=" * 60)
    print("EU AI Act Compliance Agent - Master Pipeline")
    print("=" * 60)

    test_systems = [
        {
            "name": "CV Screener (Hiring)",
            "description": (
                "An AI-powered CV screening tool that automatically ranks and filters "
                "job applicants based on their resumes, work history, and skills "
                "for recruitment and hiring decisions."
            ),
            "responses": {
                "Is there a documented risk management system in place for this AI system?": "no",
                "Are the training, validation, and testing datasets for this AI system subject to clear data governance policies?": "partial",
                "Is complete technical documentation available and up-to-date for this high-risk AI system?": "no",
                "Can deployers effectively interpret the outputs and decisions of this AI system in relation to employment matters?": "yes",
                "Are there effective mechanisms for human oversight of the AI system's operation in employment contexts?": "no",
                "Does the AI system demonstrate an appropriate level of accuracy and robustness for its intended use in employment decisions?": "partial",
            },
        },
        {
            "name": "Medical Diagnosis Chatbot",
            "description": (
                "An AI-powered medical diagnosis assistant chatbot that analyses "
                "patient symptoms, medical history, and lab results to suggest "
                "potential diagnoses and treatment options for healthcare professionals."
            ),
            "responses": {
                "Is there a documented risk management system in place?": "no",
                "Are training datasets properly governed?": "partial",
                "Is technical documentation complete?": "no",
                "Is the system transparent to users?": "yes",
                "Is human oversight implemented?": "no",
                "Are accuracy metrics documented?": "partial",
            },
        },
        {
            "name": "Product Recommender",
            "description": (
                "A customer-facing product recommendation engine on an e-commerce "
                "website that suggests items based on browsing history and purchase "
                "patterns using machine learning."
            ),
            "responses": {
                "Are users clearly informed they are interacting with an AI system?": "yes",
                "Is AI-generated content properly labelled?": "partial",
            },
        },
    ]

    results = []

    for i, system in enumerate(test_systems):
        print(f"\n--- Pipeline run {i + 1}/3: {system['name']} ---")
        result = run_compliance_check(
            system_description=system["description"],
            user_responses=system["responses"],
        )
        gap = result.get("gap_analysis_result") or {}
        classifier = result.get("classifier_result") or {}
        results.append({
            "name": system["name"],
            "risk_tier": classifier.get("risk_tier", "N/A"),
            "score": gap.get("compliance_score", "N/A"),
            "status": gap.get("overall_status", "N/A"),
        })

    # Print summary table
    print(f"\n{'=' * 60}")
    print("=== EU AI Act Compliance Pipeline Results ===")
    print(f"{'System':<28}| {'Risk Tier':<14}| {'Score':<6}| Status")
    print("-" * 60)
    for r in results:
        tier_display = r["risk_tier"].replace("_RISK", "").replace("_", " ")
        print(f"{r['name']:<28}| {tier_display:<14}| {str(r['score']):<6}| {r['status']}")
    print(f"{'=' * 60}")
    print(f"Session log saved to logs/session_log.json")


if __name__ == "__main__":
    main()
