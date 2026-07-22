"""
AgentResearch Frontend: Streamlit UI
Multi-Agent AI Research Platform with real-time progress visualization,
human review, and report generation.
"""

from __future__ import annotations

import json
import logging
import time
import sys
from pathlib import Path

import streamlit as st

# Add backend to path
BACKEND_PATH = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_PATH))

# ── Page Config ──
st.set_page_config(
    page_title="AgentResearch",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
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

# ── Imports (after path setup) ──
from config import settings
from state import create_initial_state
from agents.researcher import researcher_agent
from agents.analyst import analyst_agent
from agents.writer import writer_agent
from agents.fact_checker import fact_checker_agent
from database import init_database, save_session, load_session, list_sessions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit")

# ── Session State Init ──
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
    """Clear session state for a new research."""
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


def log_agent(agent_name: str, status: str):
    """Add an agent log entry."""
    st.session_state.agent_log.append(
        {"agent": agent_name, "status": status, "time": time.strftime("%H:%M:%S")}
    )


# ── Sidebar ──
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/brain.png", width=64)
    st.markdown("## AgentResearch 🔬")
    st.caption("Multi-Agent AI Research Platform")

    st.divider()

    # Environment check
    st.markdown("### ⚙️ Configuration")
    missing = settings.validate()
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
    else:
        st.success("API keys configured ✓")

    st.checkbox(
        "Enable Fact-Checker",
        key="enable_fact_checker",
        help="Add a fact-checking agent to verify claims (takes longer).",
    )

    st.divider()

    # Agent status log
    st.markdown("### 📋 Agent Log")
    if st.session_state.agent_log:
        for entry in st.session_state.agent_log:
            icon = {"started": "🔄", "completed": "✅", "error": "❌"}.get(entry["status"], "⏳")
            st.text(f"{icon} {entry['agent']} — {entry['time']}")
    else:
        st.caption("No agents running.")

    st.divider()

    # Session history
    st.markdown("### 📜 History")
    try:
        sessions = list_sessions(10)
        if sessions:
            for s in sessions:
                sid = s.get("id", "")[:8]
                topic = s.get("topic", "")[:40]
                status = s.get("status", "")
                if st.button(f"📝 {topic}... ({status})", key=f"hist_{sid}"):
                    loaded = load_session(s.get("id"))
                    if loaded:
                        st.session_state.session_id = loaded.get("id", "")
                        st.session_state.topic = loaded.get("topic", "")
                        st.session_state.status = loaded.get("status", "")
                        st.session_state.research_findings = loaded.get("research_findings", "")
                        st.session_state.analysis = loaded.get("analysis", "")
                        st.session_state.final_report = loaded.get("final_report", "")
                        st.session_state.report_json = loaded.get("report_json", {})
                        st.rerun()
        else:
            st.caption("No past sessions.")
    except Exception:
        st.caption("Database unavailable.")

# ── Main Content ──
st.markdown('<h1 class="main-header">🔬 AgentResearch</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-Agent AI Research Platform — '
    'Powered by LangGraph + DeepSeek</p>',
    unsafe_allow_html=True,
)

# ── Status Display ──
status = st.session_state.status
status_class = f"status-{status}" if status in [
    "researching", "analyzing", "awaiting_review", "writing", "completed", "error"
] else ""
if status != "idle":
    labels = {
        "researching": "🔍 Researching...",
        "analyzing": "🧠 Analyzing...",
        "awaiting_review": "👤 Awaiting Your Review",
        "writing": "✍️ Writing Report...",
        "completed": "✅ Report Complete!",
        "error": "❌ Error",
    }
    st.markdown(
        f'<div class="status-box {status_class}"><strong>{labels.get(status, status)}</strong></div>',
        unsafe_allow_html=True,
    )

# ── Input ──
if status in ("idle", "completed", "error"):
    col1, col2 = st.columns([4, 1])
    with col1:
        topic_input = st.text_input(
            "Research Topic",
            placeholder="E.g., 'Impact of AI on UAE financial services sector 2025'",
            key="topic_input",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("🚀 Start Research", type="primary", use_container_width=True)

    if start_btn and topic_input and len(topic_input.strip()) >= 3:
        with st.spinner("Running Research & Analysis agents..."):
            reset_session()
            st.session_state.topic = topic_input.strip()
            st.session_state.status = "researching"

            # Create initial state
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
            state = create_initial_state(st.session_state.topic, st.session_state.session_id)

            # ── Step 1: Researcher ──
            log_agent("Researcher", "started")
            status_placeholder = st.empty()
            status_placeholder.info("🔍 Researcher agent is searching the web...")
            try:
                result = researcher_agent(state)
                st.session_state.research_findings = result.get("research_findings", "")
                st.session_state.status = "analyzing"
                log_agent("Researcher", "completed")
                status_placeholder.success("✅ Researcher completed — found information")
            except Exception as e:
                st.session_state.error_message = str(e)
                st.session_state.status = "error"
                log_agent("Researcher", "error")
                st.error(f"Researcher failed: {e}")
                st.stop()

            # ── Step 2: Analyst ──
            state["research_findings"] = st.session_state.research_findings
            log_agent("Analyst", "started")
            status_placeholder.info("🧠 Analyst agent is processing findings...")
            try:
                result = analyst_agent(state)
                st.session_state.analysis = result.get("analysis", "")
                st.session_state.status = "awaiting_review"
                log_agent("Analyst", "completed")
                status_placeholder.success("✅ Analyst completed — review the insights below")
            except Exception as e:
                st.session_state.error_message = str(e)
                st.session_state.status = "error"
                log_agent("Analyst", "error")
                st.error(f"Analyst failed: {e}")
                st.stop()

            # Save to DB
            try:
                save_session(st.session_state.session_id, dict(st.session_state))
            except Exception:
                pass

            st.rerun()

# ── Research Findings Display ──
if st.session_state.research_findings:
    with st.expander("🔍 Research Findings", expanded=(status == "awaiting_review")):
        st.markdown(st.session_state.research_findings)

# ── Analysis Display ──
if st.session_state.analysis:
    with st.expander("🧠 Strategic Analysis", expanded=(status == "awaiting_review")):
        st.markdown(st.session_state.analysis)

# ── Human Review Section ──
if status == "awaiting_review":
    st.divider()
    st.markdown("### 👤 Human Review")
    st.markdown("Review the findings and analysis above, then provide your feedback.")

    feedback = st.text_area(
        "Your Feedback",
        placeholder=(
            "Add notes, request changes, or guide the report direction...\n"
            "E.g., 'Focus more on financial regulations', "
            "'The analysis of market trends needs more depth'"
        ),
        height=120,
        key="review_feedback",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("✅ Approve & Generate Report", type="primary"):
            st.session_state.human_feedback = feedback
            st.session_state.human_approved = True
            st.session_state.status = "writing"

            # Optional fact-checker
            if st.session_state.enable_fact_checker:
                with st.spinner("🔎 Fact-checker is verifying claims..."):
                    log_agent("Fact-Checker", "started")
                    state = {
                        "topic": st.session_state.topic,
                        "research_findings": st.session_state.research_findings,
                        "analysis": st.session_state.analysis,
                    }
                    try:
                        fc_result = fact_checker_agent(state)
                        st.session_state.fact_check_results = fc_result.get("fact_check_results", "")
                        st.session_state.fact_check_score = fc_result.get("fact_check_score", 0.0)
                        log_agent("Fact-Checker", "completed")
                    except Exception as e:
                        log_agent("Fact-Checker", "error")
                        st.warning(f"Fact-checker skipped: {e}")

            # Run writer
            with st.spinner("✍️ Writer agent is generating the final report..."):
                log_agent("Writer", "started")
                state = {
                    "topic": st.session_state.topic,
                    "research_findings": st.session_state.research_findings,
                    "analysis": st.session_state.analysis,
                    "human_feedback": st.session_state.human_feedback,
                    "human_approved": True,
                    "fact_check_results": st.session_state.fact_check_results,
                }
                try:
                    result = writer_agent(state)
                    st.session_state.final_report = result.get("final_report", "")
                    st.session_state.report_json = result.get("report_json", {})
                    st.session_state.status = "completed"
                    log_agent("Writer", "completed")

                    # Save to DB
                    try:
                        save_session(st.session_state.session_id, dict(st.session_state))
                    except Exception:
                        pass
                except Exception as e:
                    st.session_state.error_message = str(e)
                    st.session_state.status = "error"
                    log_agent("Writer", "error")
                    st.error(f"Report generation failed: {e}")

            st.rerun()

    with col2:
        if st.button("🔄 Request Revision"):
            st.session_state.human_feedback = feedback
            st.warning("Revision requested. Update your feedback and try again.")
            st.rerun()

# ── Fact-Check Results ──
if st.session_state.fact_check_results:
    with st.expander(f"🔎 Fact-Check Results (Score: {st.session_state.fact_check_score:.2f})"):
        st.markdown(st.session_state.fact_check_results)

# ── Final Report Display ──
if st.session_state.final_report and status == "completed":
    st.divider()
    st.markdown("### 📄 Final Report")
    st.markdown('<div class="report-container">', unsafe_allow_html=True)
    st.markdown(st.session_state.final_report)
    st.markdown("</div>", unsafe_allow_html=True)

    # Download buttons
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

    # New research button
    st.divider()
    if st.button("🔍 Start New Research", type="secondary"):
        reset_session()
        st.rerun()

# ── Error Display ──
if status == "error":
    st.error(f"**Error:** {st.session_state.error_message}")
    if st.button("🔄 Try Again"):
        reset_session()
        st.rerun()

# ── Footer ──
st.divider()
st.caption(
    "AgentResearch · Built with LangGraph + DeepSeek + Streamlit · "
    "[GitHub](https://github.com) · MIT License"
)
