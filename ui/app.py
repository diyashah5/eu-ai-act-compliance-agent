"""
EU AI Act Compliance Agent — Streamlit Web UI.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Resolve project root so we can import main.py and agents/
# ---------------------------------------------------------------------------
UI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "agents"))

# Force mock mode before importing anything from the pipeline
os.environ["MOCK_MODE"] = "1"

from main import run_compliance_check  # noqa: E402
import agents.classifier_agent as classifier_agent  # noqa: E402
import agents.requirements_agent as requirements_agent  # noqa: E402
import agents.gap_analyser_agent as gap_analyser_agent  # noqa: E402
import agents.report_generator_agent as report_generator_agent  # noqa: E402

# Ensure MOCK_MODE is on for all agents when running from UI
classifier_agent.MOCK_MODE = True
requirements_agent.MOCK_MODE = True
gap_analyser_agent.MOCK_MODE = True
report_generator_agent.MOCK_MODE = True

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOGS_DIR = PROJECT_ROOT / "logs"
SESSION_LOG_PATH = LOGS_DIR / "session_log.json"
PDF_REPORT_PATH = LOGS_DIR / "compliance_report.pdf"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EU AI Act Compliance Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Score card colours */
    .score-critical { color: #FF4B4B; font-size: 3rem; font-weight: 800; }
    .score-warning  { color: #FFA500; font-size: 3rem; font-weight: 800; }
    .score-good     { color: #21C354; font-size: 3rem; font-weight: 800; }

    /* Risk badges */
    .badge-high       { background: #FF4B4B; color: white; padding: 4px 14px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }
    .badge-unacceptable { background: #B80000; color: white; padding: 4px 14px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }
    .badge-limited    { background: #FFA500; color: white; padding: 4px 14px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }
    .badge-minimal    { background: #21C354; color: white; padding: 4px 14px; border-radius: 12px; font-weight: 700; font-size: 0.9rem; }

    /* Gap severity pills */
    .gap-critical { border-left: 4px solid #FF4B4B; padding-left: 12px; margin-bottom: 8px; }
    .gap-major    { border-left: 4px solid #FFA500; padding-left: 12px; margin-bottom: 8px; }
    .gap-minor    { border-left: 4px solid #3B82F6; padding-left: 12px; margin-bottom: 8px; }

    /* Pipeline step boxes */
    .pipeline-step {
        text-align: center; padding: 8px; border-radius: 8px;
        background: #262730; border: 1px solid #4A4A5A; margin: 2px;
        font-size: 0.82rem;
    }
    .pipeline-arrow { text-align: center; font-size: 1.3rem; color: #888; padding-top: 6px; }

    div[data-testid="stMetric"] { background: #1E1E2E; padding: 16px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# SIDEBAR
# ===================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/shield--v1.png", width=60)
    st.markdown("### 🛡️ EU AI Act Agent")
    st.caption("Built for **Kaggle AI Agents Capstone**")
    st.divider()

    # Pipeline diagram
    st.markdown("##### Agent Pipeline")
    cols = st.columns([3, 1, 3, 1, 3, 1, 3])
    labels = ["🔍 Classifier", "→", "📋 Requirements", "→", "🔎 Gap Analyser", "→", "📄 Report"]
    for col, label in zip(cols, labels):
        if label == "→":
            col.markdown(f"<div class='pipeline-arrow'>{label}</div>", unsafe_allow_html=True)
        else:
            col.markdown(f"<div class='pipeline-step'>{label}</div>", unsafe_allow_html=True)

    st.divider()

    # Session history
    st.markdown("##### 📜 Session History")
    if SESSION_LOG_PATH.exists():
        try:
            history = json.loads(SESSION_LOG_PATH.read_text(encoding="utf-8"))
            if history:
                for entry in reversed(history[-10:]):  # last 10, newest first
                    ts = entry.get("timestamp", "")[:19].replace("T", " ")
                    tier = entry.get("risk_tier", "N/A")
                    score = entry.get("compliance_score", "–")
                    status = entry.get("pipeline_status", "–")
                    desc = entry.get("system_description_summary", "Unnamed")[:40]
                    icon = "🟢" if status == "success" else "🔴"
                    with st.expander(f"{icon} {desc}", expanded=False):
                        st.markdown(f"**Time:** {ts}")
                        st.markdown(f"**Risk:** {tier}  •  **Score:** {score}")
                        st.markdown(f"**Status:** {status}")
            else:
                st.info("No runs yet. Submit a check to start.")
        except Exception:
            st.warning("Could not load session history.")
    else:
        st.info("No runs yet. Submit a check to start.")


# ===================================================================
# HEADER
# ===================================================================
st.markdown("# 🛡️ EU AI Act Compliance Agent")
st.markdown(
    "Analyse your AI system against the **EU Artificial Intelligence Act** — "
    "get a risk classification, applicable requirements, gap analysis, and an actionable compliance report."
)
st.info(
    "⏰ **EU AI Act deadline: 2 August 2026** — Providers of high-risk AI systems must comply "
    "with Chapter 2 obligations (Articles 9–15) by this date. Use this tool to identify gaps early.",
    icon="📅",
)
st.divider()


# ===================================================================
# INPUT FORM
# ===================================================================
st.markdown("## 📝 Describe Your AI System")

with st.form("compliance_form", clear_on_submit=False):
    col_left, col_right = st.columns(2)

    with col_left:
        system_name = st.text_input(
            "System Name",
            placeholder="e.g. CV Screening Tool",
            help="A short name identifying your AI system.",
        )
        system_description = st.text_area(
            "System Description",
            height=140,
            placeholder="Describe what your AI system does, its purpose, inputs, outputs, and who it affects…",
            help="Be as specific as possible — this drives the risk classification.",
        )
        industry_sector = st.selectbox(
            "Industry Sector",
            options=[
                "Healthcare",
                "Finance/Banking",
                "Education",
                "Employment/HR",
                "Law Enforcement",
                "Transportation",
                "Retail/E-commerce",
                "Other",
            ],
        )

    with col_right:
        affects_people = st.radio(
            "Does this system make decisions affecting people?",
            options=["Yes", "No"],
            horizontal=True,
        )
        processes_eu_data = st.radio(
            "Does it process personal data of EU residents?",
            options=["Yes", "No"],
            horizontal=True,
        )

        st.markdown("---")
        st.markdown("##### ✅ Compliance Checklist")
        checklist_labels = {
            "risk_mgmt": "Risk management process documented",
            "data_quality": "Training data quality verified",
            "tech_docs": "Technical documentation exists",
            "user_informed": "Users informed about AI interaction",
            "human_override": "Human can override system decisions",
            "accuracy_metrics": "Accuracy metrics documented",
        }
        answer_options = ["unknown", "yes", "no", "partial"]
        checklist_answers = {}
        for key, label in checklist_labels.items():
            checklist_answers[key] = st.selectbox(
                label,
                options=answer_options,
                key=f"chk_{key}",
            )

    submitted = st.form_submit_button(
        "🚀 Run Compliance Check",
        use_container_width=True,
        type="primary",
    )


# ===================================================================
# PIPELINE EXECUTION & RESULTS
# ===================================================================
if submitted:
    # Validate inputs
    if not system_description.strip():
        st.error("Please enter a system description before running the check.")
        st.stop()

    # Map checklist to the question/answer dict expected by the pipeline
    question_map = {
        "risk_mgmt": "Is there a documented risk management system in place for this AI system?",
        "data_quality": "Are the training, validation, and testing datasets for this AI system subject to clear data governance policies?",
        "tech_docs": "Is complete technical documentation available and up-to-date for this high-risk AI system?",
        "user_informed": "Can deployers effectively interpret the outputs and decisions of this AI system?",
        "human_override": "Are there effective mechanisms for human oversight of the AI system's operation?",
        "accuracy_metrics": "Does the AI system demonstrate an appropriate level of accuracy and robustness for its intended use?",
    }
    user_responses = {question_map[k]: v for k, v in checklist_answers.items()}

    # Enrich description with metadata
    full_description = (
        f"{system_description.strip()} "
        f"[Sector: {industry_sector}] "
        f"[Affects people: {affects_people}] "
        f"[Processes EU personal data: {processes_eu_data}]"
    )

    # Run pipeline
    with st.spinner("🔄 Running 4-agent compliance pipeline…"):
        result = run_compliance_check(
            system_description=full_description,
            user_responses=user_responses,
        )

    # --- Unpack results ---
    classifier = result.get("classifier_result") or {}
    requirements = result.get("requirements_result") or {}
    gap = result.get("gap_analysis_result") or {}
    report = result.get("report_result") or {}
    pipeline_ok = result.get("pipeline_status") == "success"

    if not pipeline_ok:
        st.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
        st.stop()

    st.divider()
    st.markdown("## 📊 Compliance Results")

    # ---- Score + Risk Tier row ----
    metric_col, tier_col, status_col = st.columns([1, 1, 1])

    score = gap.get("compliance_score", 0)
    with metric_col:
        score_class = "score-critical" if score < 50 else ("score-warning" if score < 75 else "score-good")
        st.markdown(f"<p style='margin-bottom:0; font-size:0.9rem; color:#888;'>Compliance Score</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='{score_class}'>{score}/100</p>", unsafe_allow_html=True)

    risk_tier = classifier.get("risk_tier", "UNKNOWN")
    with tier_col:
        tier_display = risk_tier.replace("_RISK", "").replace("_", " ")
        badge_map = {
            "HIGH": "badge-high",
            "UNACCEPTABLE": "badge-unacceptable",
            "LIMITED": "badge-limited",
            "MINIMAL": "badge-minimal",
        }
        badge_class = badge_map.get(tier_display, "badge-limited")
        st.markdown(f"<p style='margin-bottom:4px; font-size:0.9rem; color:#888;'>Risk Tier</p>", unsafe_allow_html=True)
        st.markdown(f"<span class='{badge_class}'>{tier_display} RISK</span>", unsafe_allow_html=True)
        st.caption(classifier.get("reasoning", ""))

    overall_status = gap.get("overall_status", "UNKNOWN")
    with status_col:
        status_emoji = {"COMPLIANT": "✅", "NEEDS_WORK": "⚠️", "CRITICAL_GAPS": "🚨"}.get(overall_status, "❓")
        st.markdown(f"<p style='margin-bottom:4px; font-size:0.9rem; color:#888;'>Overall Status</p>", unsafe_allow_html=True)
        st.markdown(f"### {status_emoji} {overall_status.replace('_', ' ')}")

    st.divider()

    # ---- Passed Checks ----
    passed = gap.get("passed_checks", [])
    if passed:
        with st.expander(f"✅ Passed Checks ({len(passed)})", expanded=False):
            for check in passed:
                st.markdown(f"- {check}")

    # ---- Gaps: Critical / Major / Minor ----
    st.markdown("### 🔍 Gap Analysis")

    gaps = gap.get("gaps", [])
    critical_gaps = [g for g in gaps if g.get("severity") == "critical"]
    major_gaps = [g for g in gaps if g.get("severity") == "major"]
    minor_gaps = [g for g in gaps if g.get("severity") == "minor"]

    gap_col1, gap_col2, gap_col3 = st.columns(3)

    with gap_col1:
        st.markdown(f"#### 🔴 Critical ({len(critical_gaps)})")
        if critical_gaps:
            for g in critical_gaps:
                with st.expander(f"⛔ {g.get('article', 'N/A')}: {g.get('requirement', '')[:50]}"):
                    st.markdown(f"**Gap:** {g.get('gap_description', '')}")
                    st.markdown(f"**Fix:** {g.get('fix_suggestion', '')}")
        else:
            st.success("No critical gaps")

    with gap_col2:
        st.markdown(f"#### 🟠 Major ({len(major_gaps)})")
        if major_gaps:
            for g in major_gaps:
                with st.expander(f"⚠️ {g.get('article', 'N/A')}: {g.get('requirement', '')[:50]}"):
                    st.markdown(f"**Gap:** {g.get('gap_description', '')}")
                    st.markdown(f"**Fix:** {g.get('fix_suggestion', '')}")
        else:
            st.success("No major gaps")

    with gap_col3:
        st.markdown(f"#### 🔵 Minor ({len(minor_gaps)})")
        if minor_gaps:
            for g in minor_gaps:
                with st.expander(f"ℹ️ {g.get('article', 'N/A')}: {g.get('requirement', '')[:50]}"):
                    st.markdown(f"**Gap:** {g.get('gap_description', '')}")
                    st.markdown(f"**Fix:** {g.get('fix_suggestion', '')}")
        else:
            st.success("No minor gaps")

    st.divider()

    # ---- Action Plan ----
    st.markdown("### 📋 Action Plan")

    all_actions = (
        report.get("critical_actions", [])
        + report.get("major_actions", [])
        + report.get("minor_actions", [])
    )
    all_actions.sort(key=lambda a: a.get("priority", 99))

    if all_actions:
        for i, action in enumerate(all_actions, 1):
            priority = action.get("priority", "–")
            prio_icon = {1: "🔴", 2: "🟠", 3: "🔵"}.get(priority, "⬜")
            st.markdown(
                f"{prio_icon} **#{i}** — _{action.get('article', '')}_ — "
                f"{action.get('action', '')}  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;📅 Deadline: **{action.get('deadline', 'TBD')}**"
            )
    else:
        st.success("🎉 No actions needed — system is compliant!")

    # ---- Timeline ----
    timeline = report.get("estimated_compliance_timeline", "")
    if timeline and timeline != "N/A":
        st.info(f"⏱️ **Estimated compliance timeline:** {timeline}")

    st.divider()

    # ---- PDF Download ----
    st.markdown("### 📥 Download Report")
    if PDF_REPORT_PATH.exists():
        pdf_bytes = PDF_REPORT_PATH.read_bytes()
        report_name = f"compliance_report_{system_name or 'system'}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            label="⬇️ Download PDF Compliance Report",
            data=pdf_bytes,
            file_name=report_name,
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.warning("PDF report not found. Run the pipeline to generate it.")

    # ---- Raw JSON (collapsible) ----
    with st.expander("🔧 Raw Pipeline Output (JSON)", expanded=False):
        st.json(result)


# ===================================================================
# FOOTER
# ===================================================================
st.divider()
st.caption(
    "⚖️ **Disclaimer:** This tool provides preliminary analysis only and does not constitute legal advice. "
    "Providers are responsible for full regulatory compliance with the EU AI Act. "
    "Consult qualified legal counsel for binding guidance."
)
st.caption("Built with ❤️ for the Kaggle AI Agents Capstone  •  Powered by Gemini + MCP")
