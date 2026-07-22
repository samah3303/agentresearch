"""
Analyst Agent: Pattern Analysis & Insight Extraction
Takes research findings and produces structured analysis with key insights.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from config import settings
from state import AgentState

logger = logging.getLogger(__name__)

ANALYST_SYSTEM_PROMPT = """You are a Principal Strategy Analyst at a top-tier management consulting firm (think McKinsey, BCG, Bain). Your role is to analyze research findings and extract actionable insights.

WORKFLOW:
1. Read the provided research findings carefully
2. Identify patterns, trends, and non-obvious connections
3. Extract the top 5-7 key insights as bullet points
4. Identify 2-4 knowledge gaps that need further investigation
5. Assess the quality and reliability of the sources
6. Produce a structured analysis report

GUIDELINES:
- Think like a strategy consultant: "So what?" for every finding
- Look for contradictions across sources — these are gold for insights
- Use frameworks when relevant (PESTLE, Porter's Five Forces, SWOT, etc.)
- Rate confidence levels: High / Medium / Low for each insight
- Challenge assumptions in the source material
- Quantify impact where possible

OUTPUT FORMAT:
Return your analysis with these sections:

## Executive Summary
2-3 sentence synthesis of the most important takeaway

## Key Insights
Numbered list of insights, each with:
- Confidence rating
- Supporting evidence from research
- Strategic implication

## Knowledge Gaps
What we still don't know — areas for further research

## Source Quality Assessment
Brief assessment of the reliability and diversity of sources

## Recommended Next Steps
What a decision-maker should do with this information
"""


def _call_deepseek(messages: list[dict]) -> str:
    """Call DeepSeek API and return text content."""
    for attempt in range(settings.max_retries):
        try:
            payload: dict[str, Any] = {
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": settings.deepseek_max_tokens,
            }
            resp = requests.post(
                f"{settings.deepseek_base_url}/chat/completions",
                headers=settings.deepseek_headers,
                json=payload,
                timeout=settings.agent_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.Timeout:
            logger.warning(f"Analyst DeepSeek timeout on attempt {attempt + 1}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
        except requests.exceptions.RequestException as e:
            logger.error(f"Analyst DeepSeek API error: {e}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise
    return "Analysis failed after multiple retries."


def _extract_insights(analysis_text: str) -> list[str]:
    """Extract numbered insights from the analysis text."""
    insights: list[str] = []
    for line in analysis_text.split("\n"):
        stripped = line.strip()
        # Match numbered items like "1. " or "1) "
        if stripped and (stripped[0].isdigit() and ". " in stripped[:4]):
            insights.append(stripped)
    return insights if insights else [analysis_text[:500] + "..."]


def _extract_gaps(analysis_text: str) -> list[str]:
    """Extract knowledge gaps from the analysis text."""
    gaps: list[str] = []
    in_gaps = False
    for line in analysis_text.split("\n"):
        if "knowledge gap" in line.lower():
            in_gaps = True
            continue
        if in_gaps and line.strip().startswith("-"):
            gaps.append(line.strip("- "))
        elif in_gaps and line.strip().startswith("#"):
            in_gaps = False
    return gaps if gaps else ["Further investigation needed across all areas."]


def analyst_agent(state: AgentState) -> dict[str, Any]:
    """
    Analyst agent node for LangGraph.
    Processes research findings and produces structured analysis.

    Args:
        state: Current workflow state with research_findings populated.

    Returns:
        Partial state update with analysis, key_insights, identified_gaps.
    """
    findings = state.get("research_findings", "")
    topic = state["topic"]
    logger.info(f"Analyst agent starting for topic: '{topic}'")

    if not findings:
        return {
            "analysis": "## Error\nNo research findings available to analyze.",
            "key_insights": ["No findings to analyze."],
            "identified_gaps": ["Research phase did not produce results."],
            "status": "error",
            "error_message": "No research findings to analyze.",
        }

    # Build prompt with full findings
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TOPIC: {topic}\n\n"
                f"RESEARCH FINDINGS:\n{findings}\n\n"
                f"Please provide your strategic analysis of these findings. "
                f"Focus on extracting actionable insights a decision-maker can use."
            ),
        },
    ]

    analysis_text = _call_deepseek(messages)
    key_insights = _extract_insights(analysis_text)
    identified_gaps = _extract_gaps(analysis_text)

    logger.info(
        f"Analyst completed: {len(analysis_text)} chars, "
        f"{len(key_insights)} insights, {len(identified_gaps)} gaps"
    )

    return {
        "analysis": analysis_text,
        "key_insights": key_insights,
        "identified_gaps": identified_gaps,
        "status": "awaiting_review",
    }
