# UDA-Hub: Multi-Agent AI Customer Support System

A sophisticated multi-agent assembly line built on **LangGraph** and **LangChain** for automated customer support ticket management. The system processes inquiries through specialized agents that handle data enrichment, classification, fact research, response drafting, and policy validation.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Using `uv` (Recommended)](#using-uv-recommended)
  - [Using `pip`](#using-pip)
- [Environment Configuration](#environment-configuration)
- [Setup & Database Initialization](#setup--database-initialization)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🎯 Overview

UDA-Hub is an intelligent customer support automation platform that:

- **Enriches customer data** by verifying user identity against databases
- **Classifies support tickets** by intent and urgency level
- **Researches facts** across multiple SQLite databases (UDAHub and CultPass)
- **Drafts empathetic responses** tailored to customer tier and context
- **Validates responses** against company policies to prevent hallucinations
- **Archives interactions** for audit trails and continuous improvement

The system uses a directed acyclic graph (DAG) with self-correction loops to ensure high-quality, policy-compliant responses while maintaining the ability to escalate high-risk or high-urgency tickets to human agents.

## 🏗️ Architecture

For a comprehensive overview of the system architecture, agents, and information flow, see [Design & Architecture Documentation](uda_hub/agentic/design/design_and_architecture.md).

### Key Components

| Agent | Responsibility |
|---|---|
| **Enricher** | Authenticates users and retrieves customer tier information |
| **Classifier** | Determines intent and urgency of support tickets |
| **Clarifier** | Requests missing information when identity is unknown |
| **Escalator** | Routes high-risk or VIP tickets to human agents |
| **Researcher** | Queries databases to find bookings and policies |
| **Drafter** | Synthesizes facts into empathetic, branded responses |
| **Policy Checker** | Audits drafts for policy violations and hallucinations |
| **Archivist** | Logs interactions and metadata to the core database |

## 📁 Project Structure

```
uda_hub/
├── 01_external_db_setup.ipynb          # Initialize external (CultPass) database
├── 02_core_db_setup.ipynb              # Initialize core (UDAHub) database
├── 03_agentic_app.ipynb                # Run the multi-agent system
├── utils.py                             # Utility functions for DB management
├── agentic/
│   ├── workflow.py                      # LangGraph workflow definition
│   ├── agents/                          # Individual agent implementations
│   │   ├── enricher.py
│   │   ├── classifier.py
│   │   ├── clarificator.py
│   │   ├── escalator.py
│   │   ├── researcher.py
│   │   ├── drafter.py
│   │   ├── policy_checker.py
│   │   ├── archivist.py
│   │   └── supervisor.py
│   ├── tools/                           # Shared tools and state management
│   │   ├── tools.py
│   │   └── udahub_state.py
│   └── design/
│       ├── design_and_architecture.md   # Detailed architecture documentation
│       └── multi-agents-architecture.mermaid
└── data/
    ├── core/                            # UDAHub core database
    ├── external/                        # CultPass external data
    │   ├── cultpass_articles.jsonl
    │   ├── cultpass_experiences.jsonl
    │   └── cultpass_users.jsonl
    └── models/                          # SQLAlchemy models
        ├── cultpass.py
        └── udahub.py
```

## 📌 Prerequisites

- **Python 3.12+**
- **SQLite 3**
- **OpenAI API Key** (for LLM integration)
- **Tavily API Key** (for web search in research agent)
- **Vocareum API Key** (environment-specific)

## 🚀 Installation

### Using `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package installer. It provides faster dependency resolution and cleaner environment management.

1. **Install `uv`** (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd uda_hub
   ```

3. **Create and activate a project virtual environment**:
   ```bash
   # Create a .venv using a specific Python version (recommended)
   uv venv --python 3.12
   # Activate the venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   # If this project contains a lockfile (uv.lock), sync from it
   uv sync

   # Or install the project in editable mode via uv's pip interface
   uv pip install -e .
   ```

## 🔐 Environment Configuration

Create a `.env` file in the project root directory. This repository currently includes a `VOCAREUM_API_KEY` entry; add other keys below as needed for external services.

Example `.env` (the project repo currently provides `VOCAREUM_API_KEY`):

```env
# Vocareum (Udacity Grading Platform) — provided in repository
VOCAREUM_API_KEY="voc-1573585998172997299766869ce3b7f87bd73.55451122"

# Optional / Add if you use these integrations
# OpenAI API Configuration
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4

# Optional: Database paths (defaults to data/ directory)
# UDAHUB_DB_PATH=data/core/udahub.db
# CULTPASS_DB_PATH=data/external/cultpass.db
```

### Environment Variable Descriptions

| Variable | Required (in repo `.env`) | Description |
|---|---:|---|
| `VOCAREUM_API_KEY` | Yes | Vocareum platform API key (present in project `.env`) |
| `OPENAI_API_KEY` | Optional | Your OpenAI API key for LLM calls (add if using OpenAI models) |
| `OPENAI_MODEL` | Optional | Model to use (default: gpt-4) |
| `UDAHUB_DB_PATH` | Optional | Path to core database (defaults to `data/core/udahub.db`) |
| `CULTPASS_DB_PATH` | Optional | Path to external database (defaults to `data/external/cultpass.db`) |

### Obtaining API Keys

1. **OpenAI API Key**: Visit [platform.openai.com](https://platform.openai.com/api-keys)
3. **Vocareum API Key**: Available through Udacity workspace environment

## 📊 Setup & Database Initialization

The project uses Jupyter notebooks to set up and manage the database schema and seed data.

### Step 1: Initialize External Database (CultPass)

```bash
jupyter notebook 01_external_db_setup.ipynb
```

This notebook:
- Loads CultPass data from JSONL files
- Creates the external database with user, article, and experience tables
- Seeds the database with sample data

### Step 2: Initialize Core Database (UDAHub)

```bash
jupyter notebook 02_core_db_setup.ipynb
```

This notebook:
- Creates the UDAHub core database schema
- Sets up customer, ticket, and metadata tables
- Initializes required indices for efficient querying

### Step 3: Run the Multi-Agent System

```bash
jupyter notebook 03_agentic_app.ipynb
```

This notebook:
- Loads the LangGraph workflow
- Demonstrates how to invoke the multi-agent system
- Provides examples of different ticket types and expected outputs

## 🎮 Running the Application

### Via Jupyter Notebook
The recommended way to interact with the system is through the provided notebooks:

```bash
jupyter notebook 03_agentic_app.ipynb
```

### Via Python Script
You can also run the workflow programmatically:

```python
from uda_hub.agentic.workflow import create_workflow
import json

# Create the workflow
workflow = create_workflow()

# Define a test ticket
ticket = {
    "user_email": "user@example.com",
    "ticket_text": "I'd like to cancel my booking for next week.",
    "ticket_id": "TKT-001"
}

# Run the workflow
result = workflow.invoke(ticket)
print(json.dumps(result, indent=2))
```

### Database Inspection

To inspect the databases directly:

```bash
# Inspect UDAHub database
sqlite3 data/core/udahub.db ".schema"

# Inspect CultPass database
sqlite3 data/external/cultpass.db ".schema"

# Query examples
sqlite3 data/core/udahub.db "SELECT * FROM customers LIMIT 5;"
```

## 🧪 Testing

Run tests to validate the system:

```bash
# Using pytest (if test suite is available)
pytest tests/
```

## 🛠️ Development

### Project Dependencies

Key dependencies include:

- **langgraph**: Multi-agent orchestration framework
- **langchain**: LLM integration and utilities
- **langchain-openai**: OpenAI models and embeddings
- **sqlalchemy**: ORM for database management
- **pandas**: Data manipulation and analysis
- **tavily-python**: Web search and research
- **ragas**: RAG evaluation framework
- **mlflow**: Experiment tracking and model management

### Adding New Agents

To add a new agent:

1. Create a new file in `uda_hub/agentic/agents/`
2. Implement the agent function following the existing pattern
3. Update `uda_hub/agentic/tools/udahub_state.py` to extend the state if needed
4. Integrate into `uda_hub/agentic/workflow.py`

## 📚 Documentation

- **Architecture**: See [design_and_architecture.md](uda_hub/agentic/design/design_and_architecture.md) for detailed system design
- **Mermaid Diagram**: View [multi-agents-architecture.mermaid](uda_hub/agentic/design/multi-agents-architecture.mermaid) in a Mermaid viewer

## 📝 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## ❓ Troubleshooting

### Issue: "API Key not found" error
**Solution**: Ensure your `.env` file is in the project root and contains all required API keys.

### Issue: Database file not found
**Solution**: Run `01_external_db_setup.ipynb` and `02_core_db_setup.ipynb` first to initialize the databases.

### Issue: ModuleNotFoundError
**Solution**: Ensure the project is installed in editable mode: `pip install -e .`

### Issue: Jupyter kernel not found
**Solution**: Install the IPython kernel: `pip install ipykernel` and select it in Jupyter.

## 📧 Support

For issues or questions, please [open an issue](issues) on the repository.
