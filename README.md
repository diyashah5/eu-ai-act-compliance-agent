# EU AI Act Compliance Agent

An automated compliance assistant and technical auditor for aligning AI systems with the EU AI Act.

## Project Structure

```
eu-ai-act-compliance-agent/
├── agents/            # AI agent logic and prompts
├── mcp_server/        # Model Context Protocol servers
├── knowledge_base/    # Machine-readable EU AI Act regulations
│   ├── annex_iii.json
│   ├── articles_obligations.json
│   └── risk_matrix.json
├── ui/                # Streamlit front-end
├── logs/              # Audit and compliance execution logs
├── tests/             # Unit and integration tests
├── .env               # Local configuration and keys
├── requirements.txt   # Python dependency list
├── validate_kb.py     # Script to check integrity of the knowledge base
└── README.md          # Project documentation
```

## Setup & Validation

To validate the structure and integrity of the compliance knowledge base, run:
```bash
python validate_kb.py
```
