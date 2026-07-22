"""
AgentResearch — Streamlit Cloud Entry Point
Multi-Agent AI Research Platform with LangGraph + DeepSeek.
Runs graph directly (no HTTP to localhost FastAPI).
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# ── Ensure backend is importable ──────────────────────────
BACKEND_PATH = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="AgentResearch",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-researching { background: #e3f2fd; border-left: 4px solid #2196f3; }
    .status-analyzing { background: #f3e5f5; border-left: 4px solid #9c27b0; }
    .status-review { background: #fff3e0; border-left: 4px solid #ff9800; }
    .status-writing { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .status-completed { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .status-error { background: #ffebee; border-left: 4px solid #f44336; }
    .agent-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #444;
        margin-bottom: 0.5rem;
    }
    .report-container {
        background: #1e1e1e;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Imports (after path setup) ────────────────────────────
from config import settings
from state import create_initial_state
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent
from agents.fact_checker import fact_checker_agent
from database import init_database, save_session, load_session, list_sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit")

# ── Init DB (once) ────────────────────────────────────────
@st.cache_resource
def _init_db():
    try:
        init_database()
        logger.info("Database initialized.")
        return True
    except Exception as e:
        logger.warning("DB init failed: %s. Running without persistence.", e)
        return False

_ = _init_db()

# ── Session State Init ────────────────────────────────────
DEFAULTS = {
    "session_id": None,
    "topic": "",
    "status": "idle",
    "research_findings": "",
    "analysis": "",
    "human_feedback": "",
    "human_approved": False,
    "fact_check_results": "",
    "fact_check_score": 0.0,
    "final_report": "",
    "report_json": {},
    "error_message": "",
    "agent_log": [],
    "enable_fact_checker": False,
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


def reset_session():
    """Reset all session state to defaults."""
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


def save_current_state():
    """Persist current state to DB."""
    try:
        state = {k: st.session_state[k] for k in DEFAULTS if k != "agent_log"}
        save_session(st.session_state.session_id, state)
    except Exception as e:
        logger.warning("Failed to save session: %s", e)


# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.8rem;">🔬 AgentResearch</p>',
                unsafe_allow_html=True)
    st.caption("Multi-Agent AI Research Platform")

    # API Key status
    if settings.deepseek_api_key:
        st.success("🔑 DeepSeek API configured")
    else:
        st.error("🔴 DeepSeek API key missing — set DEEPSEEK_API_KEY")
        st.stop()

    st.divider()

    # Past sessions
    st.subheader("📋 Past Sessions")
    sessions = list_sessions(limit=10)
    if sessions:
        for s in sessions:
            label = s.get("topic", f"Session {s['id'][:8]}")
            if st.button(f"📝 {label[:35]}", key=f"sess_{s['id']}", use_container_width=True):
                loaded = load_session(s["id"])
                if loaded:
                    st.session_state.session_id = loaded.get("id", s["id"])
                    st.session_state.topic = loaded.get("topic", "")
                    st.session_state.status = loaded.get("status", "idle")
                    st.session_state.research_findings = loaded.get("research_findings", "")
                    st.session_state.analysis = loaded.get("analysis", "")
                    st.session_state.final_report = loaded.get("final_report", "")
                    st.session_state.report_json = loaded.get("report_json", {})
                    st.session_state.fact_check_results = loaded.get("fact_check_results", "")
                    st.session_state.fact_check_score = loaded.get("fact_check_score", 0.0)
                st.rerun()
    else:
        st.caption("No saved sessions.")

    st.divider()

    # Fact-checker toggle
    st.session_state.enable_fact_checker = st.toggle("🔎 Enable Fact-Checker", value=False)


# ── Main Header ───────────────────────────────────────────
st.markdown('<p class="main-header">🔬 AgentResearch</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Agent AI Research · LangGraph + DeepSeek</p>',
            unsafe_allow_html=True)

# ── Status Display ────────────────────────────────────────
status = st.session_state.status

if status != "idle" and status != "completed" and status != "error":
    status_class = {
        "researching": "status-researching",
        "analyzing": "status-analyzing",
        "awaiting_review": "status-review",
        "writing": "status-writing",
    }.get(status, "")
    st.markdown(
        f'<div class="status-box {status_class}"><strong>Status:</strong> {status.replace("_", " ").title()}</div>',
        unsafe_allow_html=True,
    )

# ── Topic Input ───────────────────────────────────────────
if status == "idle":
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "**Research Topic**",
            placeholder="e.g., 'Impact of quantum computing on cybersecurity by 2030'",
            key="topic_input",
        )
    with col2:
        st.write("")
        st.write("")
        start_disabled = not topic or len(topic) < 3
        if st.button("🚀 Start Research", type="primary", disabled=start_disabled,
                      use_container_width=True):
            st.session_state.topic = topic
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.status = "researching"
            st.rerun()

# ── Workflow Execution ────────────────────────────────────
if status == "researching":
    with st.spinner("🔍 Researcher Agent: Searching the web and gathering information..."):
        try:
            state = create_initial_state(st.session_state.topic, st.session_state.session_id)
            result = researcher_agent(state)
            st.session_state.research_findings = result.get("research_findings", "")
            st.session_state.status = result.get("status", "analyzing")
            st.session_state.agent_log.append({"agent": "researcher", "status": "completed"})
            save_current_state()
            st.rerun()
        except Exception as e:
            logger.exception("Researcher failed")
            st.session_state.status = "error"
            st.session_state.error_message = str(e)
            st.rerun()

if status == "analyzing":
    with st.spinner("🧠 Analyst Agent: Finding patterns and extracting insights..."):
        try:
            state = {
                "topic": st.session_state.topic,
                "research_findings": st.session_state.research_findings,
                "session_id": st.session_state.session_id,
            }
            result = analyst_agent(state)
            st.session_state.analysis = result.get("analysis", "")
            st.session_state.status = result.get("status", "awaiting_review")
            st.session_state.agent_log.append({"agent": "analyst", "status": "completed"})
            save_current_state()
            st.rerun()
        except Exception as e:
            logger.exception("Analyst failed")
            st.session_state.status = "error"
            st.session_state.error_message = str(e)
            st.rerun()

# ── Research Findings ─────────────────────────────────────
if st.session_state.research_findings:
    with st.expander("🔍 Research Findings", expanded=False):
        st.markdown(st.session_state.research_findings)

# ── Analysis ──────────────────────────────────────────────
if st.session_state.analysis:
    with st.expander("🧠 Strategic Analysis", expanded=(status == "awaiting_review")):
        st.markdown(st.session_state.analysis)

# ── Human Review ──────────────────────────────────────────
if status == "awaiting_review":
    st.divider()
    st.markdown("### 👤 Human Review")
    st.info("Review the analysis above. Provide feedback or approve to continue.")

    col1, col2 = st.columns(2)
    with col1:
        feedback = st.text_area(
            "**Feedback** (optional — suggestions, corrections, angle adjustments)",
            placeholder="e.g., 'Focus more on European regulations' or 'Add more quantitative data'",
            height=100,
            key="review_feedback",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✅ Approve & Generate Report", type="primary", use_container_width=True):
            st.session_state.human_feedback = feedback
            st.session_state.human_approved = True
            st.session_state.status = "writing"
            save_current_state()
            st.rerun()

        if st.button("🔄 Request Revision"):
            st.session_state.human_feedback = feedback
            st.warning("Revision requested. Update your feedback and try again.")
            st.rerun()

# ── Writing ───────────────────────────────────────────────
if status == "writing":
    with st.spinner("✍️ Writer Agent: Generating the final report..."):
        try:
            state = {
                "topic": st.session_state.topic,
                "research_findings": st.session_state.research_findings,
                "analysis": st.session_state.analysis,
                "human_feedback": st.session_state.human_feedback,
                "human_approved": st.session_state.human_approved,
            }
            result = writer_agent(state)
            st.session_state.final_report = result.get("final_report", "")
            st.session_state.report_json = result.get("report_json", {})

            # Optional: run fact-checker
            if st.session_state.enable_fact_checker:
                fc_state = {
                    "topic": st.session_state.topic,
                    "research_findings": st.session_state.research_findings,
                    "analysis": st.session_state.analysis,
                    "session_id": st.session_state.session_id,
                }
                fc_result = fact_checker_agent(fc_state)
                st.session_state.fact_check_results = fc_result.get("fact_check_results", "")
                st.session_state.fact_check_score = fc_result.get("fact_check_score", 0.0)

            st.session_state.status = "completed"
            st.session_state.agent_log.append({"agent": "writer", "status": "completed"})
            save_current_state()
            st.rerun()
        except Exception as e:
            logger.exception("Writer failed")
            st.session_state.status = "error"
            st.session_state.error_message = str(e)
            st.rerun()

# ── Fact-Check Results ────────────────────────────────────
if st.session_state.fact_check_results:
    with st.expander(f"🔎 Fact-Check Results (Score: {st.session_state.fact_check_score:.2f})"):
        st.markdown(st.session_state.fact_check_results)

# ── Final Report Display ──────────────────────────────────
if st.session_state.final_report and status == "completed":
    st.divider()
    st.markdown("### 📄 Final Report")
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Report (Markdown)",
            data=st.session_state.final_report,
            file_name=f"research_report_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col2:
        report_json_str = json.dumps(st.session_state.report_json, indent=2, ensure_ascii=False)
        st.download_button(
            "📥 Download Report (JSON)",
            data=report_json_str,
            file_name=f"research_report_{st.session_state.session_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    if st.button("🔍 Start New Research", type="secondary"):
        reset_session()
        st.rerun()

# ── Error Display ─────────────────────────────────────────
if status == "error":
    st.error(f"**Error:** {st.session_state.error_message}")
    if st.button("🔄 Try Again"):
        reset_session()
        st.rerun()

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption(
    "AgentResearch · Built with LangGraph + DeepSeek + Streamlit · "
    "[GitHub](https://github.com) · MIT License"
)
