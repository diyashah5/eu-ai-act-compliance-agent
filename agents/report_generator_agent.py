"""
EU AI Act Compliance Agent — Report Generator Agent (Agent 4).

Aggregates outputs from the Classifier, Requirements, and Gap Analyser agents
to produce a detailed compliance report, including prioritized action plans and
a PDF export (saved to logs/compliance_report.pdf).
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
from fpdf import FPDF

import classifier_agent
import requirements_agent
import gap_analyser_agent
from classifier_agent import classify_system, load_knowledge_base as load_classifier_kb
from requirements_agent import get_requirements, load_articles_obligations
from gap_analyser_agent import analyse_gaps

# Mock Mode Configuration
MOCK_MODE = True

# Force mock mode on dependent agents if active
if MOCK_MODE:
    classifier_agent.MOCK_MODE = True
    requirements_agent.MOCK_MODE = True
    gap_analyser_agent.MOCK_MODE = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "gemini-2.5-flash-lite"


def clean_unicode(text: str) -> str:
    """Replace non-latin1 characters with standard ASCII equivalents for FPDF compatibility."""
    if not isinstance(text, str):
        return text
    replacements = {
        "\u2013": "-",  # en-dash
        "\u2014": "-",  # em-dash
        "\u2018": "'",  # curly single open quote
        "\u2019": "'",  # curly single close quote
        "\u201c": '"',  # curly double open quote
        "\u201d": '"',  # curly double close quote
        "\u2022": "-",  # bullet point
        "\u20ac": "EUR", # euro symbol
        "\u00a0": " ",   # non-breaking space
    }
    cleaned = text
    for unicode_char, ascii_char in replacements.items():
        cleaned = cleaned.replace(unicode_char, ascii_char)
    # Force encode to latin-1 and replace unhandled chars with '?'
    return cleaned.encode("latin-1", "replace").decode("latin-1")


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


def _build_report_prompt(
    classifier_result: dict[str, Any],
    requirements_result: dict[str, Any],
    gap_analysis_result: dict[str, Any],
    system_description: str,
) -> str:
    """Build the prompt for Gemini to compile the compliance report."""
    return f"""You are an expert EU AI Act compliance officer. Your task is to compile a comprehensive, professional compliance report based on the results from the classification, requirements gathering, and gap analysis agents.

AI System Description:
\"\"\"{system_description}\"\"\"

Classifier Agent Result:
{json.dumps(classifier_result, indent=2)}

Requirements Agent Result:
{json.dumps(requirements_result, indent=2)}

Gap Analysis Agent Result:
{json.dumps(gap_analysis_result, indent=2)}

Write an executive summary of 2-3 detailed paragraphs.
Based on the gaps identified in the gap analysis, create lists of prioritized actions:
- `critical_actions`: Priority 1, high severity gaps (typically Article 9, 10, 14 gaps).
- `major_actions`: Priority 2, medium severity gaps (typically Article 11, 13 gaps).
- `minor_actions`: Priority 3, lower severity gaps.

For each action item, provide a concrete action description, the applicable article, a realistic deadline (e.g., '1 month', '2 weeks'), and the priority (1, 2, or 3).
Determine an estimated compliance timeline (e.g. '3-4 months').
Retrieve or summarize passed checks.
Include a standard legal disclaimer.

Respond with ONLY valid JSON (no markdown fences) using this exact schema:
{{
  "report_title": "Compliance Audit Report - [System Name]",
  "executive_summary": "Paragraph 1\\n\\nParagraph 2\\n\\nParagraph 3",
  "risk_tier": "HIGH_RISK | LIMITED_RISK | MINIMAL_RISK | UNACCEPTABLE_RISK",
  "compliance_score": 35,
  "overall_status": "CRITICAL_GAPS",
  "critical_actions": [
    {{
      "action": "Action description",
      "article": "Article 9",
      "deadline": "1 month",
      "priority": 1
    }}
  ],
  "major_actions": [
    {{
      "action": "Action description",
      "article": "Article 11",
      "deadline": "2 months",
      "priority": 2
    }}
  ],
  "minor_actions": [
    {{
      "action": "Action description",
      "article": "Article 15",
      "deadline": "3 months",
      "priority": 3
    }}
  ],
  "estimated_compliance_timeline": "3-4 months",
  "passed_checks": [
    "List of requirements already met"
  ],
  "disclaimer": "The legal disclaimer string"
}}"""


def get_mock_report(
    classifier_result: dict[str, Any],
    requirements_result: dict[str, Any],
    gap_analysis_result: dict[str, Any],
) -> dict[str, Any]:
    """Return hardcoded realistic compliance report matching the test scenario."""
    return {
        "report_title": "EU AI Act Compliance Report - Hiring AI System",
        "executive_summary": (
            "This compliance report evaluates the hiring AI-powered CV screening tool "
            "against the obligations set out in the European Union Artificial Intelligence Act (EU AI Act). "
            "Based on the system description and user response checklist, the system has been classified "
            "as a High-Risk AI system under Annex III (HR4 - Employment, recruitment, and worker selection). "
            "Consequently, it is subject to the stringent obligations detailed in Chapter 2 of the Act.\n\n"
            "Our audit has identified several critical and major compliance gaps that must be addressed before "
            "the system can be legally placed on the market or put into service. Critical gaps were identified "
            "in the areas of Risk Management (Article 9), Data Governance (Article 10), and Human Oversight (Article 14). "
            "Specifically, the lack of a documented risk management framework, incomplete training data governance, "
            "and missing human override capabilities constitute critical compliance failures.\n\n"
            "Immediate remediation is required to bring the system into compliance. The provider must establish "
            "a risk management process, formally document training datasets, draw up comprehensive technical documentation, "
            "and implement robust human oversight controls. Implementing these recommendations is estimated to "
            "take approximately 3-4 months, after which a full conformity assessment must be performed."
        ),
        "risk_tier": classifier_result.get("risk_tier", "HIGH_RISK"),
        "compliance_score": gap_analysis_result.get("compliance_score", 35),
        "overall_status": gap_analysis_result.get("overall_status", "CRITICAL_GAPS"),
        "critical_actions": [
            {
                "action": "Establish and document a comprehensive risk management process to continuously identify and mitigate risks.",
                "article": "Article 9",
                "deadline": "1 month",
                "priority": 1
            },
            {
                "action": "Implement rigorous data governance policies and document bias analysis for training, validation, and testing datasets.",
                "article": "Article 10",
                "deadline": "1 month",
                "priority": 1
            },
            {
                "action": "Design and integrate mechanisms for effective human oversight, including override and interruption options.",
                "article": "Article 14",
                "deadline": "2 weeks",
                "priority": 1
            }
        ],
        "major_actions": [
            {
                "action": "Compile complete and detailed technical documentation demonstrating compliance with all obligations under Chapter 2.",
                "article": "Article 11",
                "deadline": "2 months",
                "priority": 2
            },
            {
                "action": "Ensure operations are transparent to deployers and draft clear, comprehensive instructions for use.",
                "article": "Article 13",
                "deadline": "1 month",
                "priority": 2
            }
        ],
        "minor_actions": [],
        "estimated_compliance_timeline": "3-4 months",
        "passed_checks": [
            "Conformity Classification (Article 6) - High-risk tier correctly identified.",
            "Accuracy, Robustness and Cybersecurity (Article 15) - Partial documentation and measures in place."
        ],
        "disclaimer": (
            "Disclaimer: This compliance report is generated based on the information provided by the user and "
            "does not constitute formal legal advice. Compliance with the EU AI Act requires detailed legal "
            "and technical review. The providers remain fully liable for ensuring their systems comply with "
            "all applicable regulatory frameworks."
        )
    }


def generate_pdf_report(report: dict[str, Any], output_path: str) -> None:
    """Generate a formatted PDF compliance report using fpdf2."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # PAGE 1: Cover Page
    pdf.add_page()

    # Dark blue background header bar
    pdf.set_fill_color(26, 54, 93)  # Deep Navy Blue
    pdf.rect(0, 0, 210, 100, "F")

    pdf.set_xy(15, 30)
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(255, 255, 255)
    pdf.multi_cell(0, 12, clean_unicode(report.get("report_title", "EU AI Act Compliance Report")))

    pdf.set_xy(15, 65)
    pdf.set_font("helvetica", "I", 14)
    pdf.set_text_color(200, 200, 200)
    pdf.cell(0, 10, "Automated Compliance Auditing Report")

    # Cover Metadata Block
    pdf.set_xy(15, 110)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 10, "Project Information:")
    pdf.ln(10)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(50, 6, "Risk Classification:", 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, clean_unicode(report.get("risk_tier", "HIGH_RISK")), 0)
    pdf.ln(6)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(50, 6, "Overall Status:", 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, clean_unicode(report.get("overall_status", "CRITICAL_GAPS")), 0)
    pdf.ln(6)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(50, 6, "Compliance Score:", 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, f"{report.get('compliance_score', 0)}/100", 0)
    pdf.ln(6)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(50, 6, "Estimated Timeline:", 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, clean_unicode(report.get("estimated_compliance_timeline", "N/A")), 0)
    pdf.ln(6)

    pdf.set_font("helvetica", "", 10)
    pdf.cell(50, 6, "Date Generated:", 0)
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, time.strftime("%Y-%m-%d"), 0)

    # PAGE 2: Executive Summary
    pdf.add_page()
    pdf.set_text_color(26, 54, 93)  # Deep Navy Blue
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "1. Executive Summary", 0, 1, "L")
    pdf.ln(3)

    pdf.set_text_color(60, 60, 60)
    pdf.set_font("helvetica", "", 10)
    summary_text = clean_unicode(report.get("executive_summary", ""))
    pdf.multi_cell(0, 6, summary_text)
    pdf.ln(10)

    # Compliance Score Box
    pdf.set_fill_color(254, 242, 242)  # Light Red background
    pdf.set_draw_color(248, 113, 113)  # Red border
    pdf.rect(15, pdf.get_y(), 180, 28, "DF")

    pdf.set_xy(20, pdf.get_y() + 4)
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(185, 28, 28)  # Dark Red text
    pdf.cell(0, 6, f"Compliance Score: {report.get('compliance_score', 0)}/100")
    pdf.ln(6)
    pdf.set_xy(20, pdf.get_y())
    pdf.set_font("helvetica", "B", 10)
    pdf.cell(0, 6, f"Overall Status: {clean_unicode(report.get('overall_status', 'CRITICAL_GAPS'))}")
    pdf.ln(15)

    # PAGE 3: Gaps and Priority Actions
    pdf.add_page()
    pdf.set_text_color(26, 54, 93)  # Deep Navy Blue
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "2. Priority Action Plan", 0, 1, "L")
    pdf.ln(3)

    # Critical Actions
    pdf.set_text_color(185, 28, 28)  # Red
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2.1 Critical Action Items (Priority 1 - Immediate Action Required)", 0, 1, "L")
    pdf.ln(2)

    crit_actions = report.get("critical_actions", [])
    if crit_actions:
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        for idx, act in enumerate(crit_actions, 1):
            text = f"{idx}. {act.get('action')} [Article: {act.get('article')}] (Deadline: {act.get('deadline')})"
            pdf.multi_cell(0, 5, clean_unicode(text))
            pdf.ln(3)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "No critical action items identified.", 0, 1)
    pdf.ln(6)

    # Major Actions
    pdf.set_text_color(217, 119, 6)  # Amber
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2.2 Major Action Items (Priority 2 - High Priority)", 0, 1, "L")
    pdf.ln(2)

    major_actions = report.get("major_actions", [])
    if major_actions:
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        for idx, act in enumerate(major_actions, 1):
            text = f"{idx}. {act.get('action')} [Article: {act.get('article')}] (Deadline: {act.get('deadline')})"
            pdf.multi_cell(0, 5, clean_unicode(text))
            pdf.ln(3)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "No major action items identified.", 0, 1)
    pdf.ln(6)

    # Minor Actions
    pdf.set_text_color(37, 99, 235)  # Blue
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "2.3 Minor Action Items (Priority 3 - Recommended)", 0, 1, "L")
    pdf.ln(2)

    minor_actions = report.get("minor_actions", [])
    if minor_actions:
        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        for idx, act in enumerate(minor_actions, 1):
            text = f"{idx}. {act.get('action')} [Article: {act.get('article')}] (Deadline: {act.get('deadline')})"
            pdf.multi_cell(0, 5, clean_unicode(text))
            pdf.ln(3)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "No minor action items identified.", 0, 1)
    pdf.ln(6)

    # PAGE 4: Passed Checks & Disclaimer
    pdf.add_page()
    pdf.set_text_color(26, 54, 93)  # Deep Navy Blue
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "3. Passed Compliance Checks", 0, 1, "L")
    pdf.ln(3)

    pdf.set_font("helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    passed_checks = report.get("passed_checks", [])
    if passed_checks:
        for idx, check in enumerate(passed_checks, 1):
            pdf.multi_cell(0, 5, clean_unicode(f"- {check}"))
            pdf.ln(2)
    else:
        pdf.set_font("helvetica", "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 6, "No passed compliance checks recorded.", 0, 1)
    pdf.ln(20)

    # Bottom line & disclaimer
    pdf.set_y(-45)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, clean_unicode(report.get("disclaimer", "Disclaimer: This compliance audit report does not constitute legal advice.")))

    pdf.output(output_path)
    logger.info("PDF compliance report written to %s", output_path)


def generate_report(
    classifier_result: dict[str, Any],
    requirements_result: dict[str, Any],
    gap_analysis_result: dict[str, Any],
    system_description: str,
) -> dict[str, Any]:
    """
    Generate compliance report from auditing pipeline results.

    Args:
        classifier_result: Dict classification result.
        requirements_result: Dict requirements list.
        gap_analysis_result: Dict gaps and score.
        system_description: User described AI system description.
    """
    if MOCK_MODE:
        logger.info("MOCK_MODE is True - returning mock report without API calls")
        return get_mock_report(
            classifier_result,
            requirements_result,
            gap_analysis_result,
        )

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    prompt = _build_report_prompt(
        classifier_result,
        requirements_result,
        gap_analysis_result,
        system_description,
    )

    logger.info("Sending report compile request to Gemini (%s)", MODEL_NAME)
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
    return result


def main() -> None:
    """Run the full sequence: classifier -> requirements -> gap analyser -> report generator."""
    load_dotenv(PROJECT_ROOT / ".env")

    print("EU AI Act Compliance Agent — Full Pipeline Sequence Run")

    scenario_desc = (
        "An AI-powered CV screening tool that automatically ranks and filters "
        "job applicants based on their résumés, work history, and skills "
        "for recruitment and hiring decisions."
    )

    # Load Knowledge Bases if not mocking
    if not MOCK_MODE:
        load_classifier_kb()
        load_articles_obligations()

    # Step 1: Classifier
    classification = classify_system(scenario_desc)

    # Step 2: Requirements
    matched = classification.get("matched_category")
    if matched == "None" or matched == "null":
        matched = None
    requirements = get_requirements(
        risk_tier=classification["risk_tier"],
        matched_category=matched,
    )

    # Step 3: Map user responses & Run Gap Analyser
    user_responses = {}
    for req in requirements.get("requirements", []):
        art_str = req.get("article", "")
        # Fallback regex extraction for article number
        match = re.search(r"(\d+)", art_str)
        art_num = int(match.group(1)) if match else None

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

    gap_analysis = analyse_gaps(requirements, user_responses, risk_tier=classification["risk_tier"])

    # Step 4: Report Generator
    report = generate_report(
        classifier_result=classification,
        requirements_result=requirements,
        gap_analysis_result=gap_analysis,
        system_description=scenario_desc,
    )

    # Export to PDF
    pdf_path = os.path.join(PROJECT_ROOT, "logs", "compliance_report.pdf")
    generate_pdf_report(report, pdf_path)

    # Print Clean Summary
    print(f"\nReport Title: {report.get('report_title')}")
    print(f"Compliance Score: {report.get('compliance_score')}/100")
    print(f"Overall Status: {report.get('overall_status')}")
    print(f"Critical Actions: {len(report.get('critical_actions', []))}")
    print(f"Major Actions: {len(report.get('major_actions', []))}")
    print(f"Minor Actions: {len(report.get('minor_actions', []))}")
    print(f"PDF saved to: logs/compliance_report.pdf")


if __name__ == "__main__":
    main()
