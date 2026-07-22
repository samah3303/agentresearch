"""
Researcher Agent: Web Search & Information Gathering
Uses DeepSeek with function calling to search DuckDuckGo and synthesize findings.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

from config import settings
from state import AgentState, ResearchSource
from tools import TOOL_DEFINITIONS, TOOL_MAP

logger = logging.getLogger(__name__)

RESEARCHER_SYSTEM_PROMPT = """You are a Senior Research Analyst at a top consulting firm. Your job is to conduct thorough web research on any given topic.

WORKFLOW:
1. Generate 3-5 targeted search queries that cover different angles of the topic
2. Call the web_search function for each query
3. Synthesize ALL findings into a comprehensive research brief
4. Cite specific facts, statistics, and sources in your summary

GUIDELINES:
- Be thorough: explore different perspectives (technical, business, social, etc.)
- Prioritize recent and authoritative sources
- Extract concrete data points: numbers, dates, names, statistics
- Flag any contradictory information you find
- Structure the findings with clear section headers
- Always cite the source URL when presenting a specific fact

OUTPUT FORMAT:
Return your findings as a well-structured markdown document with sections:
## Overview
## Key Findings (with citations)
## Data & Statistics
## Perspectives & Debates
## Sources
"""


def _call_deepseek(messages: list[dict], tools: list[dict] | None = None) -> dict:
    """Call DeepSeek API and return the response. Handles retries."""
    for attempt in range(settings.max_retries):
        try:
            payload: dict[str, Any] = {
                "model": settings.deepseek_model,
                "messages": messages,
                "temperature": settings.deepseek_temperature,
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
            logger.warning(f"DeepSeek timeout on attempt {attempt + 1}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepSeek API error: {e}")
            if attempt < settings.max_retries - 1:
                time.sleep(2**attempt)
            else:
                raise
    return {"choices": [{"message": {"content": "Research failed after retries."}}]}


def researcher_agent(state: AgentState) -> dict[str, Any]:
    """
    Researcher agent node for LangGraph.
    Searches the web and produces a research findings summary.

    Args:
        state: Current workflow state with at least 'topic' set.

    Returns:
        Partial state update with research_findings and research_sources.
    """
    topic = state["topic"]
    logger.info(f"Researcher agent starting for topic: '{topic}'")

    # Step 1: Ask DeepSeek to generate search queries and then search
    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Research this topic thoroughly: {topic}\n\n"
            f"First, generate 3-5 search queries, then call web_search for each. "
            f"After gathering all results, synthesize a comprehensive research brief.",
        },
    ]

    all_sources: list[ResearchSource] = []
    all_tool_results: list[str] = []

    # Multi-turn function calling loop (max 6 turns)
    for turn in range(6):
        response = _call_deepseek(messages, TOOL_DEFINITIONS)
        choice = response.get("choices", [{}])[0]
        msg = choice.get("message", {})

        # Check if the model wants to call a function
        if msg.get("tool_calls"):
            messages.append(msg)  # assistant message with tool_calls

            for tool_call in msg["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])
                tool_fn = TOOL_MAP.get(func_name)

                if tool_fn:
                    # Extract the argument based on tool
                    if func_name == "web_search":
                        arg = func_args.get("query", topic)
                        result = tool_fn(arg)
                        all_tool_results.append(f"## Search: {arg}\n{result}")

                        # Parse results into sources
                        try:
                            parsed = json.loads(result)
                            for src in parsed:
                                all_sources.append(
                                    ResearchSource(
                                        title=src.get("title", "Untitled"),
                                        url=src.get("url", ""),
                                        snippet=src.get("snippet", ""),
                                        relevance_score=0.8,
                                    )
                                )
                        except json.JSONDecodeError:
                            pass
                    elif func_name == "calculator":
                        result = tool_fn(func_args.get("expression", ""))
                        all_tool_results.append(f"Calculation: {result}")
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
            # Model provided a text response — this is the final synthesis
            content = msg.get("content", "")
            if content:
                # If we have tool results, the content is the synthesis;
                # otherwise it might be a direct answer without search
                break
            else:
                # Empty response — continue loop
                messages.append(
                    {"role": "user", "content": "Please continue your research."}
                )
    else:
        # Loop exhausted without final answer — force summary
        messages.append(
            {
                "role": "user",
                "content": "Please provide your complete research findings now based on all the search results above.",
            }
        )
        response = _call_deepseek(messages)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")

    research_findings = content or "Research completed. See sources for details."

    logger.info(f"Researcher completed: {len(research_findings)} chars, {len(all_sources)} sources")

    return {
        "research_findings": research_findings,
        "research_sources": all_sources,
        "search_queries": [],
        "messages": messages,
        "status": "analyzing",
    }
