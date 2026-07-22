"""
Writer Agent: Report Generation
Produces polished, structured Markdown reports from analyzed research.
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

WRITER_SYSTEM_PROMPT = """You are a Senior Report Writer at a prestigious consulting firm. You transform raw analysis into polished, executive-ready reports that drive decisions.

WORKFLOW:
1. Review the topic, research findings, analysis, and human feedback
2. Structure the report with professional sections
3. Write clear, concise, and impactful prose
4. Ensure all claims are backed by evidence from the research
5. Produce both a Markdown report and a structured JSON summary

SECTIONS TO INCLUDE:
1. Executive Summary (1-2 paragraphs)
2. Introduction & Background
3. Key Findings (with source citations)
4. Strategic Analysis
5. Opportunities & Risks
6. Recommendations (prioritized, actionable)
7. Conclusion
8. Sources & References

GUIDELINES:
- Write for a C-suite executive: clear, concise, no jargon
- Use bullet points for scannability
- Every recommendation must be specific and actionable
- Include a "Confidence Level" for major findings
- Charts/visual descriptions can be included as placeholders
- Honor any human feedback provided — adjust tone, focus, or content accordingly

OUTPUT FORMAT:
First, output the complete Markdown report.
Then, on a separate line, output `---JSON---` followed by a JSON object with:
{
  "title": "Report title",
  "executive_summary": "Brief summary",
  "key_findings": ["finding 1", "finding 2", ...],
  "recommendations": ["rec 1", "rec 2", ...],
  "confidence": "high/medium/low",
  "word_count": 1234
}
"""


def _call_deepseek(messages: list[dict]) -> str:
    """Call DeepSeek API and return text content."""
    for attempt in range(settings.max_retries):
        try:
            payload: dict[str, Any] = {
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": 0.3,
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
            logger.warning(f"Writer DeepSeek timeout on attempt {attempt + 1}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
        except requests.exceptions.RequestException as e:
            logger.error(f"Writer DeepSeek API error: {e}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise
    return "Report generation failed after multiple retries."


def _parse_report(report_text: str) -> tuple[str, dict[str, Any]]:
    """Split the report into Markdown and JSON components."""
    markdown = report_text
    report_json: dict[str, Any] = {}

    # Look for JSON separator
    if "---JSON---" in report_text:
        parts = report_text.split("---JSON---", 1)
        markdown = parts[0].strip()
        json_str = parts[1].strip()
        # Extract just the JSON object (in case there's trailing text)
        try:
            brace_start = json_str.index("{")
            brace_end = json_str.rindex("}") + 1
            report_json = json.loads(json_str[brace_start:brace_end])
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse report JSON: {e}")
            report_json = {"error": "JSON parsing failed", "raw": json_str[:500]}

    return markdown, report_json


def writer_agent(state: AgentState) -> dict[str, Any]:
    """
    Writer agent node for LangGraph.
    Generates the final structured research report.

    Args:
        state: Current workflow state with all previous outputs.

    Returns:
        Partial state update with final_report and report_json.
    """
    topic = state["topic"]
    findings = state.get("research_findings", "")
    analysis = state.get("analysis", "")
    feedback = state.get("human_feedback", "")
    approved = state.get("human_approved", False)
    fact_results = state.get("fact_check_results", "")

    logger.info(f"Writer agent starting for topic: '{topic}'")

    # Build the prompt with all context
    feedback_note = ""
    if feedback:
        feedback_note = (
            f"\n\nHUMAN FEEDBACK (incorporate this into the report):\n{feedback}\n"
            f"Approved: {'Yes' if approved else 'No'}\n"
            f"Adjust the report accordingly based on this feedback."
        )

    fact_note = ""
    if fact_results:
        fact_note = f"\n\nFACT-CHECK RESULTS (verify and correct claims):\n{fact_results}\n"

    user_content = (
        f"TOPIC: {topic}\n\n"
        f"RESEARCH FINDINGS:\n{findings}\n\n"
        f"STRATEGIC ANALYSIS:\n{analysis}\n"
        f"{fact_note}"
        f"{feedback_note}\n\n"
        f"Please generate the final report. Remember to include both the Markdown "
        f"report and the JSON summary separated by '---JSON---'."
    )

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    report_text = _call_deepseek(messages)
    markdown, report_json = _parse_report(report_text)

    logger.info(
        f"Writer completed: {len(markdown)} chars markdown, "
        f"{len(report_json)} JSON keys"
    )

    return {
        "final_report": markdown,
        "report_json": report_json,
        "status": "completed",
    }
