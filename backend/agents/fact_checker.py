"""
Fact-Checker Agent: Claim Verification
Verifies claims made by other agents against web search results.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from config import settings
from state import AgentState
from tools import TOOL_DEFINITIONS, TOOL_MAP

logger = logging.getLogger(__name__)

FACT_CHECKER_SYSTEM_PROMPT = """You are a Rigorous Fact-Checker. Your job is to verify the factual accuracy of claims made in research reports and analysis.

WORKFLOW:
1. Extract specific factual claims from the provided text
2. For each claim, search the web to verify or refute it
3. Rate each claim: Verified / Partially Verified / Unverified / Contradicted
4. Provide a source for your verification
5. Calculate an overall factuality score (0.0 = all false, 1.0 = all verified)

GUIDELINES:
- Focus on specific, verifiable claims (numbers, dates, names, events)
- Skip opinions, recommendations, and subjective statements
- Use multiple sources when possible for high-stakes claims
- Flag contradictory information prominently
- Be precise about what was verified vs. what was partially verified
- If you can't verify something, say so — don't guess

OUTPUT FORMAT:

## Claim Verification Report

| Claim | Status | Source | Notes |
|-------|--------|--------|-------|
| ... | ✅ Verified | [URL] | ... |
| ... | ⚠️ Partially | [URL] | ... |
| ... | ❌ Contradicted | [URL] | ... |

## Overall Factuality Score: X.X / 1.0

## Summary
Brief assessment of overall factual reliability.
"""


def _call_deepseek(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call DeepSeek API and return the response."""
    for attempt in range(settings.max_retries):
        try:
            payload: dict[str, Any] = {
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": settings.deepseek_max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            resp = requests.post(
                f"{settings.deepseek_base_url}/chat/completions",
                headers=settings.deepseek_headers,
                json=payload,
                timeout=settings.agent_timeout_seconds,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Fact-checker DeepSeek timeout on attempt {attempt + 1}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
        except requests.exceptions.RequestException as e:
            logger.error(f"Fact-checker DeepSeek API error: {e}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise
    return {"choices": [{"message": {"content": "Fact-checking failed after retries."}}]}


def _extract_score(text: str) -> float:
    """Extract the factuality score from the report text."""
    import re

    # Look for patterns like "X.X / 1.0" or "Score: 0.8"
    patterns = [
        r"(\d+\.?\d*)\s*/\s*1\.0",
        r"[Ss]core:\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)\s*out of\s*1",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass
    return 0.5  # Default neutral score


def fact_checker_agent(state: AgentState) -> dict[str, Any]:
    """
    Fact-checker agent node for LangGraph.
    Verifies claims in research findings and analysis.

    Args:
        state: Current workflow state with research and analysis.

    Returns:
        Partial state update with fact_check_results and fact_check_score.
    """
    topic = state["topic"]
    findings = state.get("research_findings", "")
    analysis = state.get("analysis", "")

    logger.info(f"Fact-checker agent starting for topic: '{topic}'")

    content_to_check = f"RESEARCH FINDINGS:\n{findings}\n\nANALYSIS:\n{analysis}"

    messages = [
        {"role": "system", "content": FACT_CHECKER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"TOPIC: {topic}\n\n"
                f"Please fact-check the following content:\n\n{content_to_check}\n\n"
                f"Extract specific factual claims and verify each one using web search."
            ),
        },
    ]

    # Function calling loop for fact-checking (max 8 turns)
    for turn in range(8):
        response = _call_deepseek(messages, TOOL_DEFINITIONS)
        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})

        if msg.get("tool_calls"):
            messages.append(msg)
            for tool_call in msg["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                tool_fn = TOOL_MAP.get(func_name)

                if tool_fn:
                    if func_name == "web_search":
                        arg = func_args.get("query", topic)
                        result = tool_fn(arg)
                    elif func_name == "calculator":
                        result = tool_fn(func_args.get("expression", ""))
                    else:
                        result = f"Unknown tool: {func_name}"

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "content": result,
                        }
                    )
        else:
            content = msg.get("content", "")
            if content:
                break
            messages.append(
                {"role": "user", "content": "Please complete the fact-checking report."}
            )
    else:
        messages.append(
            {
                "role": "user",
                "content": "Provide your complete fact-checking report now.",
            }
        )
        response = _call_deepseek(messages)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    fact_check_results = content or "Fact-checking completed."
    fact_check_score = _extract_score(fact_check_results)

    logger.info(
        f"Fact-checker completed: score={fact_check_score:.2f}, "
        f"{len(fact_check_results)} chars"
    )

    return {
        "fact_check_results": fact_check_results,
        "fact_check_score": fact_check_score,
    }
