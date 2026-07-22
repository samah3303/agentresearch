"""
Tool definitions for AgentResearch agents.
Uses DuckDuckGo for web search and includes utility tools.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import requests
from ddgs import DDGS

from config import settings

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web using DuckDuckGo and return structured results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, url, snippet.
    """
    try:
        logger.info(f"Searching DuckDuckGo for: '{query}'")
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            structured = [
                {
                    "title": r.get("title", "Untitled"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
            logger.info(f"Found {len(structured)} results for '{query}'")
            return structured
    except Exception as e:
        logger.error(f"DuckDuckGo search failed for '{query}': {e}")
        return [{"title": "Search Error", "url": "", "snippet": str(e)}]


def web_search_tool(query: str) -> str:
    """
    Tool-compatible web search: returns JSON string for LLM consumption.
    """
    results = web_search(query, max_results=settings.max_search_results)
    return json.dumps(results, ensure_ascii=False, indent=2)


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    Args:
        expression: A mathematical expression as a string (e.g., "2 + 2 * 3").

    Returns:
        The result as a string.
    """
    import ast
    import operator

    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"Operation {op_type} not allowed")
            return allowed_ops[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_ops:
                raise ValueError(f"Operation {op_type} not allowed")
            return allowed_ops[op_type](_eval(node.operand))
        if isinstance(node, ast.Constant):
            return node.value
        raise ValueError(f"Node type {type(node)} not supported")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


# LangChain-compatible tool definitions for DeepSeek function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information on a given query. Use this to find facts, news, articles, and data relevant to the research topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for calculations, statistics, or numerical analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate, e.g. '2 + 2 * 3'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]

# Map tool names to actual callables
TOOL_MAP = {
    "web_search": web_search_tool,
    "calculator": calculator,
}
