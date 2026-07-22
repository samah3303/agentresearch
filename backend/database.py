"""
Database Layer: Neon PostgreSQL for session state and research history.
Uses asyncpg or sync fallback via psycopg2.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from config import settings

logger = logging.getLogger(__name__)

# Try psycopg2 first (synchronous, simpler)
try:
    import psycopg2
    import psycopg2.extras

    HAS_DB = True
    DB_LIB = "psycopg2"
except ImportError:
    try:
        import asyncpg

        HAS_DB = True
        DB_LIB = "asyncpg"
    except ImportError:
        HAS_DB = False
        DB_LIB = None
        logger.warning("No PostgreSQL driver found. Database features disabled.")


def _get_connection():
    """Create a psycopg2 connection."""
    if not HAS_DB or DB_LIB != "psycopg2":
        raise RuntimeError("psycopg2 not available")
    return psycopg2.connect(settings.database_url)


def init_database() -> bool:
    """
    Initialize database tables. Returns True on success.
    Creates tables if they don't exist.
    """
    if not HAS_DB:
        logger.warning("Database unavailable — skipping initialization.")
        return False

    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id UUID PRIMARY KEY,
                    topic TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'idle',
                    research_findings TEXT,
                    analysis TEXT,
                    human_feedback TEXT,
                    human_approved BOOLEAN DEFAULT FALSE,
                    fact_check_results TEXT,
                    fact_check_score FLOAT DEFAULT 0.0,
                    final_report TEXT,
                    report_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS agent_logs (
                    id SERIAL PRIMARY KEY,
                    session_id UUID REFERENCES research_sessions(id),
                    agent_name TEXT NOT NULL,
                    agent_status TEXT NOT NULL,
                    input_summary TEXT,
                    output_summary TEXT,
                    duration_ms INTEGER,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            conn.commit()
        conn.close()
        logger.info("Database tables initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


def save_session(session_id: str, state: dict[str, Any]) -> bool:
    """Save or update a research session in the database."""
    if not HAS_DB:
        return False

    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_sessions (
                    id, topic, status, research_findings, analysis,
                    human_feedback, human_approved, fact_check_results,
                    fact_check_score, final_report, report_json, updated_at
                ) VALUES (
                    %(id)s, %(topic)s, %(status)s, %(research_findings)s,
                    %(analysis)s, %(human_feedback)s, %(human_approved)s,
                    %(fact_check_results)s, %(fact_check_score)s,
                    %(final_report)s, %(report_json)s, NOW()
                )
                ON CONFLICT (id) DO UPDATE SET
                    topic = EXCLUDED.topic,
                    status = EXCLUDED.status,
                    research_findings = EXCLUDED.research_findings,
                    analysis = EXCLUDED.analysis,
                    human_feedback = EXCLUDED.human_feedback,
                    human_approved = EXCLUDED.human_approved,
                    fact_check_results = EXCLUDED.fact_check_results,
                    fact_check_score = EXCLUDED.fact_check_score,
                    final_report = EXCLUDED.final_report,
                    report_json = EXCLUDED.report_json,
                    updated_at = NOW(),
                    completed_at = CASE
                        WHEN EXCLUDED.status = 'completed' THEN NOW()
                        ELSE research_sessions.completed_at
                    END
                """,
                {
                    "id": session_id,
                    "topic": state.get("topic", ""),
                    "status": state.get("status", "idle"),
                    "research_findings": state.get("research_findings", ""),
                    "analysis": state.get("analysis", ""),
                    "human_feedback": state.get("human_feedback", ""),
                    "human_approved": state.get("human_approved", False),
                    "fact_check_results": state.get("fact_check_results", ""),
                    "fact_check_score": state.get("fact_check_score", 0.0),
                    "final_report": state.get("final_report", ""),
                    "report_json": json.dumps(state.get("report_json", {})),
                },
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save session {session_id}: {e}")
        return False


def load_session(session_id: str) -> Optional[dict[str, Any]]:
    """Load a research session from the database."""
    if not HAS_DB:
        return None

    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM research_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
        conn.close()

        if row:
            result = dict(row)
            # Parse JSONB fields
            if result.get("report_json") and isinstance(result["report_json"], str):
                result["report_json"] = json.loads(result["report_json"])
            # Convert datetime objects
            for key in ("created_at", "updated_at", "completed_at"):
                if result.get(key):
                    result[key] = result[key].isoformat()
            return result
        return None
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return None


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    """List recent research sessions."""
    if not HAS_DB:
        return []

    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, topic, status, created_at, updated_at, completed_at
                FROM research_sessions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return []


def log_agent_activity(
    session_id: str,
    agent_name: str,
    agent_status: str,
    input_summary: str = "",
    output_summary: str = "",
    duration_ms: int = 0,
) -> bool:
    """Log agent activity to the database."""
    if not HAS_DB:
        return False

    try:
        conn = _get_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_logs (
                    session_id, agent_name, agent_status,
                    input_summary, output_summary, duration_ms
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (session_id, agent_name, agent_status, input_summary, output_summary, duration_ms),
            )
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to log agent activity: {e}")
        return False
