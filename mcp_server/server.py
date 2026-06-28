"""
Model Context Protocol (MCP) Server for the EU AI Act Compliance Agent.

What is MCP?
Model Context Protocol (MCP) is an open-standard protocol designed to enable secure,
controlled, and standardized communication between Large Language Models (LLMs) and
external data sources, tools, or services. It acts as an integration layer, allowing
agents to query knowledge bases, run checks, and invoke tools through a uniform API.

Why is it used in this project?
In this project, the MCP server is used to decouple the EU AI Act regulatory knowledge base
(annex_iii.json, articles_obligations.json, and risk_matrix.json) from individual agent
implementations. Instead of each agent reading JSON files directly from disk, they call
the MCP server's standardized tools. This enables centralized management of the regulations,
proper access control, auditing, and potential transition to remote APIs/databases in the future.
"""

import json
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP

# Setup clean logging to stderr to prevent stdout pollution (which crashes stdio MCP transport)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("mcp-server")

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "knowledge_base"
ANNEX_III_PATH = KB_DIR / "annex_iii.json"
ARTICLES_PATH = KB_DIR / "articles_obligations.json"
RISK_MATRIX_PATH = KB_DIR / "risk_matrix.json"

# Load knowledge base files at startup with error handling
def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        logger.error("Knowledge base file missing: %s", path)
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to parse knowledge base file %s: %s", path, e)
        return {}

ANNEX_DATA = load_json_file(ANNEX_III_PATH)
ARTICLES_DATA = load_json_file(ARTICLES_PATH)
RISK_MATRIX_DATA = load_json_file(RISK_MATRIX_PATH)

# Initialize MCP server
mcp = FastMCP("eu-ai-act-knowledge-server")

@mcp.tool()
def get_risk_classification_logic() -> List[Dict[str, Any]]:
    """
    Retrieve the full risk classification logic steps from risk_matrix.json.
    
    Returns:
        List of classification steps containing step order, rule names, descriptions, and target risk tiers.
    """
    logger.info("Tool called: get_risk_classification_logic")
    steps = RISK_MATRIX_DATA.get("classification_steps", [])
    logger.info("get_risk_classification_logic returned %d steps", len(steps))
    return steps

@mcp.tool()
def get_high_risk_systems() -> List[Dict[str, Any]]:
    """
    Retrieve the list of high-risk AI systems categories from annex_iii.json.
    
    Returns:
        List of high-risk systems, each with ID, category, description, legal article, and examples.
    """
    logger.info("Tool called: get_high_risk_systems")
    systems = ANNEX_DATA.get("high_risk_systems", [])
    logger.info("get_high_risk_systems returned %d high-risk categories", len(systems))
    return systems

@mcp.tool()
def get_article_obligations(article_number: str) -> Dict[str, Any]:
    """
    Retrieve legal obligations and requirements for a specific article from articles_obligations.json.
    
    Args:
        article_number: The article identifier (e.g., '5', '9', 'Article 9', 'article_9').
        
    Returns:
        Dictionary containing the article title, ID, and list of specific obligations.
    """
    logger.info("Tool called: get_article_obligations(article_number=%r)", article_number)
    
    # Extract digits or normalize number from the string
    cleaned = article_number.lower().replace("article", "").replace("_", "").replace(" ", "").strip()
    
    articles = ARTICLES_DATA.get("articles", [])
    for art in articles:
        art_id = art.get("id", "").lower().replace("article", "").replace("_", "").replace(" ", "").strip()
        if art_id == cleaned:
            logger.info("get_article_obligations found article %s", art.get("id"))
            return art
            
    logger.warning("get_article_obligations: article %s not found", article_number)
    return {}

@mcp.tool()
def get_prohibited_systems() -> List[Dict[str, Any]]:
    """
    Retrieve the list of prohibited AI systems/practices from annex_iii.json.
    
    Returns:
        List of prohibited practices, each with ID, category/name, and description.
    """
    logger.info("Tool called: get_prohibited_systems")
    prohibited = ANNEX_DATA.get("prohibited_systems", [])
    logger.info("get_prohibited_systems returned %d prohibited categories", len(prohibited))
    return prohibited

@mcp.tool()
def server_health() -> Dict[str, Any]:
    """
    Retrieve the health status and metadata of the MCP server.
    
    Returns:
        Dictionary containing server status ('OK'), number of exposed tools, and current timestamp.
    """
    logger.info("Tool called: server_health")
    # Number of tools is 5 (get_risk_classification_logic, get_high_risk_systems, get_article_obligations, get_prohibited_systems, server_health)
    status = {
        "status": "OK",
        "tools_available": 5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "loaded_files": {
            "annex_iii": bool(ANNEX_DATA),
            "articles_obligations": bool(ARTICLES_DATA),
            "risk_matrix": bool(RISK_MATRIX_DATA)
        }
    }
    logger.info("server_health returned status OK")
    return status

if __name__ == "__main__":
    mcp.run(transport="stdio")
