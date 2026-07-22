"""
AgentResearch Backend Entry Point
FastAPI server exposing the multi-agent research workflow as REST endpoints.
Can also be run via Streamlit (frontend/app.py) directly.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from database import init_database, list_sessions, load_session, save_session
from graph import build_graph
from state import create_initial_state

# ── Logging ──
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI App ──
app = FastAPI(
    title="AgentResearch API",
    description="Multi-Agent AI Research Platform — LangGraph + DeepSeek",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response Models ──


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=500, description="Research topic")
    enable_fact_checker: bool = Field(default=False)
    session_id: Optional[str] = Field(default=None)


class ReviewRequest(BaseModel):
    session_id: str
    approved: bool
    feedback: str = Field(default="")


class ResearchResponse(BaseModel):
    session_id: str
    topic: str
    status: str
    research_findings: str = ""
    analysis: str = ""
    final_report: str = ""
    report_json: dict[str, Any] = {}
    error_message: str = ""


# ── Graph Instances ──
graph = build_graph(with_fact_checker=False, checkpoint=True)
graph_with_fc = build_graph(with_fact_checker=True, checkpoint=True)

# ── Endpoints ──


@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    logger.info("Starting AgentResearch API...")
    missing = settings.validate()
    if missing:
        logger.warning(f"Missing environment variables: {', '.join(missing)}")
    init_database()
    logger.info("AgentResearch API ready.")


@app.get("/")
async def root():
    return {"name": "AgentResearch API", "version": "1.0.0", "status": "running"}


@app.get("/health")
async def health():
    missing = settings.validate()
    return {
        "status": "healthy" if not missing else "degraded",
        "missing_env": missing,
        "db_available": bool(settings.database_url),
    }


@app.post("/research/start", response_model=ResearchResponse)
async def start_research(req: ResearchRequest):
    """Start a new research session. Runs researcher + analyst agents."""
    session_id = req.session_id or str(uuid.uuid4())
    initial_state = create_initial_state(req.topic, session_id)

    # Save initial state
    save_session(session_id, initial_state)

    try:
        # Run researcher + analyst (stops before human review)
        g = graph_with_fc if req.enable_fact_checker else graph
        config = {"configurable": {"thread_id": session_id}}

        # Run until we hit the human_review node (or complete)
        final_state = None
        for event in g.stream(initial_state, config):
            for node_name, node_output in event.items():
                logger.info(f"Node '{node_name}' completed")
                final_state = node_output if isinstance(node_output, dict) else None

        # After streaming, get the current state
        snapshot = g.get_state(config)
        if snapshot and snapshot.values:
            current_state = dict(snapshot.values)
            save_session(session_id, current_state)

            return ResearchResponse(
                session_id=session_id,
                topic=current_state.get("topic", req.topic),
                status=current_state.get("status", "idle"),
                research_findings=current_state.get("research_findings", ""),
                analysis=current_state.get("analysis", ""),
                final_report=current_state.get("final_report", ""),
                report_json=current_state.get("report_json", {}),
                error_message=current_state.get("error_message", ""),
            )

    except Exception as e:
        logger.error(f"Research failed for session {session_id}: {e}", exc_info=True)
        save_session(
            session_id,
            {
                "session_id": session_id,
                "topic": req.topic,
                "status": "error",
                "error_message": str(e),
            },
        )
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(
        session_id=session_id,
        topic=req.topic,
        status="error",
        error_message="No state returned from graph.",
    )


@app.post("/research/review", response_model=ResearchResponse)
async def submit_review(req: ReviewRequest):
    """Submit human review and continue the workflow to generate the final report."""
    g = graph_with_fc  # Use the graph; fact-checker availability is determined at build time

    session_id = req.session_id
    config = {"configurable": {"thread_id": session_id}}

    try:
        # Update state with human feedback
        g.update_state(
            config,
            {
                "human_feedback": req.feedback,
                "human_approved": req.approved,
                "status": "writing" if req.approved else "awaiting_review",
            },
        )

        if not req.approved:
            snapshot = g.get_state(config)
            current_state = dict(snapshot.values) if snapshot.values else {}
            save_session(session_id, current_state)
            return ResearchResponse(
                session_id=session_id,
                topic=current_state.get("topic", ""),
                status="awaiting_review",
                research_findings=current_state.get("research_findings", ""),
                analysis=current_state.get("analysis", ""),
                error_message="Awaiting approval. Update feedback and approve to continue.",
            )

        # Continue execution from current state
        for event in g.stream(None, config):
            for node_name, node_output in event.items():
                logger.info(f"Node '{node_name}' completed after review")

        snapshot = g.get_state(config)
        if snapshot and snapshot.values:
            current_state = dict(snapshot.values)
            save_session(session_id, current_state)

            return ResearchResponse(
                session_id=session_id,
                topic=current_state.get("topic", ""),
                status=current_state.get("status", "idle"),
                research_findings=current_state.get("research_findings", ""),
                analysis=current_state.get("analysis", ""),
                final_report=current_state.get("final_report", ""),
                report_json=current_state.get("report_json", {}),
                error_message=current_state.get("error_message", ""),
            )

    except Exception as e:
        logger.error(f"Review failed for session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return ResearchResponse(session_id=session_id, topic="", status="error", error_message="No state returned.")


@app.get("/research/sessions")
async def get_sessions(limit: int = 20):
    """List recent research sessions."""
    return {"sessions": list_sessions(limit)}


@app.get("/research/session/{session_id}")
async def get_session(session_id: str):
    """Get a specific research session by ID."""
    session = load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── Main ──
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
