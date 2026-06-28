"""
MCP Client for testing and connecting to the EU AI Act Knowledge Server.

FastMCP serialisation note:
  When a tool returns a List[Dict], FastMCP creates one TextContent item per
  list element.  Single-Dict returns produce one TextContent item as expected.
  All helpers below account for this by iterating ``resp.content``.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Resolve paths relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"

# Configure server execution parameters
server_params = StdioServerParameters(
    command="python",
    args=[str(SERVER_PATH)],
    env=os.environ.copy()
)


# ---------------------------------------------------------------------------
# Helpers to parse FastMCP responses
# ---------------------------------------------------------------------------
def _parse_list_response(resp) -> List[Dict[str, Any]]:
    """Parse an MCP tool response that returns a list (one content item per element)."""
    return [json.loads(item.text) for item in resp.content]


def _parse_dict_response(resp) -> Dict[str, Any]:
    """Parse an MCP tool response that returns a single dict."""
    return json.loads(resp.content[0].text)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
async def run_client_diagnostics() -> None:
    """Connects to the MCP server, runs diagnostics for all tools, and prints output."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call Health Check
            health_data = _parse_dict_response(
                await session.call_tool("server_health")
            )

            # List Tools
            tools_result = await session.list_tools()

            # Call High Risk Systems
            high_risk_data = _parse_list_response(
                await session.call_tool("get_high_risk_systems")
            )

            # Call Risk Classification Logic
            class_logic_data = _parse_list_response(
                await session.call_tool("get_risk_classification_logic")
            )

            # Call Article Obligations (Article 9)
            art9_data = _parse_dict_response(
                await session.call_tool("get_article_obligations", arguments={"article_number": "9"})
            )

            # Call Prohibited Systems
            prohibited_data = _parse_list_response(
                await session.call_tool("get_prohibited_systems")
            )

            # Format and print the exact expected output
            print(f"MCP Server Health: {health_data.get('status')}")
            custom_tools_count = len([t for t in tools_result.tools if t.name != "server_health"])
            print(f"Tools available: {custom_tools_count}")
            print(f"High risk systems loaded: {len(high_risk_data)}")

            # Classification steps (excluding the default fallback)
            step_count = len([s for s in class_logic_data if s.get("name") != "default"])
            print(f"Risk classification steps: {step_count}")

            # Article 9 obligations
            print(f"Article 9 obligations: {len(art9_data.get('obligations', []))} requirements")

            # Prohibited systems
            print(f"Prohibited systems: {len(prohibited_data)}")


# ---------------------------------------------------------------------------
# Knowledge-base data fetcher (used by main.py)
# ---------------------------------------------------------------------------
async def fetch_knowledge_base_data() -> Dict[str, Any]:
    """Connects to MCP server and fetches the complete knowledge base data."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Fetch classifications and categories
            classification_steps = _parse_list_response(
                await session.call_tool("get_risk_classification_logic")
            )

            high_risk_systems = _parse_list_response(
                await session.call_tool("get_high_risk_systems")
            )

            prohibited_systems = _parse_list_response(
                await session.call_tool("get_prohibited_systems")
            )

            # Fetch obligations for all standard articles
            articles_to_fetch = ["5", "6", "9", "10", "11", "13", "14", "15", "52"]
            articles_list = []
            for art_num in articles_to_fetch:
                art_data = _parse_dict_response(
                    await session.call_tool("get_article_obligations", arguments={"article_number": art_num})
                )
                if art_data:
                    articles_list.append(art_data)

            # Package data to match the format of annex_iii.json, risk_matrix.json, and articles_obligations.json
            return {
                "annex_iii": {
                    "high_risk_systems": high_risk_systems,
                    "prohibited_systems": prohibited_systems
                },
                "risk_matrix": {
                    "classification_steps": classification_steps,
                    "risk_levels": {
                        "unacceptable": {
                            "description": "AI practices presenting a clear threat to safety, livelihoods, and rights of people.",
                            "action": "Prohibited. System must not be placed on the market or put into service."
                        },
                        "high": {
                            "description": "AI systems listed under Annex III or safety components of regulated products.",
                            "action": "Obligatory compliance with Chapter 2 requirements (Articles 9-15) and conformity assessment."
                        },
                        "limited": {
                            "description": "AI systems presenting specific transparency risks (e.g., chatbots, deepfakes, emotion recognition).",
                            "action": "Ensure users are informed they are interacting with an AI system (Article 52)."
                        },
                        "minimal": {
                            "description": "AI systems representing low or no risk.",
                            "action": "No obligations under the Act, voluntary codes of conduct encouraged."
                        }
                    },
                    "mappings": [
                        {"category_id": f"HR{i}", "risk_level": "high", "required_articles": ["article_6", "article_9", "article_13"]}
                        for i in range(1, 9)
                    ]
                },
                "articles_obligations": {
                    "articles": articles_list
                }
            }


def get_kb_data_sync() -> Dict[str, Any]:
    """Synchronous wrapper to retrieve the knowledge base data from the MCP server."""
    return asyncio.run(fetch_knowledge_base_data())


if __name__ == "__main__":
    asyncio.run(run_client_diagnostics())
