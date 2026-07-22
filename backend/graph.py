"""
LangGraph State Graph: Multi-Agent Research Orchestration
Defines the workflow with conditional routing and human-in-the-loop.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents import analyst_agent, fact_checker_agent, researcher_agent, writer_agent
from state import AgentState

logger = logging.getLogger(__name__)


def researcher_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: runs the Researcher agent."""
    logger.info(">>> Researcher Node")
    return researcher_agent(state)


def analyst_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: runs the Analyst agent."""
    logger.info(">>> Analyst Node")
    return analyst_agent(state)


def human_review_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph node: human review checkpoint.
    In production with LangGraph's interrupt, this would pause execution.
    For Streamlit, we handle pausing in the UI layer by checking status.
    """
    logger.info(">>> Human Review Node (pass-through)")
    # The actual pause happens in the Streamlit UI.
    # This node simply marks the state for review.
    # If human_feedback is already set, we skip through.
    if state.get("human_approved"):
        return {"status": "writing"}
    # Otherwise, we stay in awaiting_review — the UI will prompt the user
    return {}


def writer_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: runs the Writer agent."""
    logger.info(">>> Writer Node")
    return writer_agent(state)


def fact_checker_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: runs the Fact-Checker agent (optional)."""
    logger.info(">>> Fact-Checker Node")
    return fact_checker_agent(state)


def should_continue_after_analyst(state: AgentState) -> Literal["human_review", "writer", "fact_checker"]:
    """Route after analyst: if approved directly, skip review; else go to review."""
    if state.get("human_approved"):
        return "writer"
    if state.get("status") == "error":
        return "writer"
    return "human_review"


def should_continue_after_review(state: AgentState) -> Literal["writer", "fact_checker", "human_review"]:
    """Route after human review: if approved, proceed; else stay in review."""
    if state.get("human_approved"):
        # Route to fact-checker if enabled, else writer
        if state.get("fact_check_results"):
            return "writer"
        return "writer"
    # Stay in review if not approved yet
    return "human_review"


def build_graph(
    with_fact_checker: bool = False,
    checkpoint: bool = True,
) -> StateGraph:
    """
    Build the LangGraph StateGraph for the multi-agent research workflow.

    Args:
        with_fact_checker: Include fact-checker agent after analysis.
        checkpoint: Enable memory checkpointing for interrupt/resume.

    Returns:
        A compiled StateGraph ready to invoke.
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("human_review", human_review_node)

    if with_fact_checker:
        workflow.add_node("fact_checker", fact_checker_node)

    workflow.add_node("writer", writer_node)

    # Define edges
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "analyst")

    # After analyst: conditional routing
    workflow.add_conditional_edges(
        "analyst",
        should_continue_after_analyst,
        {
            "human_review": "human_review",
            "writer": "writer",
            "fact_checker": "fact_checker" if with_fact_checker else "writer",
        },
    )

    # After human review: conditional routing
    if with_fact_checker:
        workflow.add_conditional_edges(
            "human_review",
            should_continue_after_review,
            {
                "writer": "writer",
                "fact_checker": "fact_checker",
                "human_review": "human_review",
            },
        )
        workflow.add_edge("fact_checker", "writer")
    else:
        workflow.add_conditional_edges(
            "human_review",
            should_continue_after_review,
            {
                "writer": "writer",
                "human_review": "human_review",
            },
        )

    # Writer → END
    workflow.add_edge("writer", END)

    # Compile with optional checkpointing
    if checkpoint:
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)
    else:
        compiled = workflow.compile()

    logger.info(
        f"Graph built successfully "
        f"(fact_checker={with_fact_checker}, checkpoint={checkpoint})"
    )
    return compiled


# Default graph instance
graph = build_graph(with_fact_checker=False)
