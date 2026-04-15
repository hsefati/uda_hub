# UDA-Hub: Multi-Agent AI Customer Support System

A sophisticated multi-agent assembly line built on LangGraph and LangChain for automated customer support ticket management. The system utilizes Dynamic Anchoring and specialized nodes to process inquiries from initial contact to final archival.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Agent Roles](#agent-roles)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Setup & Running](#setup--running)

## 🎯 Overview

UDA-Hub is an intelligent customer support platform that treats every interaction as part of a structured "Assembly Line." Unlike basic chatbots, UDA-Hub maintains a "Dynamic Anchor"—a persistent understanding of the user's core problem—even if the user provides information across multiple turns or gets distracted by pleasantries.

## ✨ Key Features

- **Dynamic Anchoring:** Preserves the original support request as the "source of truth," preventing context-switching or "greeting hijacks."
- **History-Aware Enrichment:** Mines identity data (Email, Name, Membership ID) from the full conversation history to authenticate users against the `cultpass.db`.
- **Identity Gate:** A security layer that ensures data is only archived to the production `udahub.db` once a `user_id` has been verified.
- **Automated QA Loop:** A dedicated Policy Checker audits every AI draft against retrieved facts to eliminate hallucinations before the user sees the response.
- **Satisfaction Tracking:** Detects user gratitude or closure to officially "Seal" and close tickets in the database.

## 🏗️ Architecture

UDA-Hub is organized as a stateful directed graph (LangGraph) where each node is a specialized agent and edges encode routing and business logic. Responsibilities are separated across layers so the system is resilient, auditable, and easy to extend.

- Ingress & Triage: incoming messages are classified and anchored; the `Classifier` protects the Dynamic Anchor and routes messages to the correct flow.
- Enrichment & Research: the `Enricher` mines identity and conversation history while the `Researcher` retrieves facts from external/internal sources.
- Drafting & QA: the `Drafter` composes responses and the `Policy Checker` audits drafts against retrieved facts to prevent hallucinations.
- Persistence & Closure: the `Archivist` persists verified tickets to `udahub.db` and the `Closer` seals and closes completed tickets.

Data flow (high level):
1. Message arrives → `Classifier` (protect Anchor and route)
2. If identity or context is missing → `Clarifier` requests details
3. `Enricher`/`Researcher` gather facts and context
4. `Drafter` produces a response → `Policy Checker` audits it
5. After verification, `Archivist` persists the ticket → `Closer` seals the record

For detailed design and sequence diagrams, see:
> [uda_hub/agentic/design/design_and_architecture.md](uda_hub/agentic/design/design_and_architecture.md)

## 🏗️ Key Components

| Agent | Responsibility |
|---|---|
| Enricher | Mines history for identity (Name/ID) and performs SQL lookups via specialized tools. |
| Classifier | Intent triage that protects the "Anchor" from being overwritten by greetings. |
| Greeter | Handles pleasantries and "Hello" when no active ticket is present. |
| Clarifier | Requests missing info (Email/ID) with empathy and context awareness. |
| Escalator | Hand-off logic for high-risk or VIP tickets to human managers. |
| Researcher | Fact-finding agent that queries bookings and company policies. |
| Drafter | Synthesizes facts into professional, branded email responses. |
| Policy Checker | A 0-temperature auditor that validates drafts against retrieved research. |
| Closer | Detects satisfaction and triggers the "Closed" status for the database. |
| Archivist | A tool-driven agent that summarizes and seals the ticket in `udahub.db`. |

## 📁 Project Structure
uda_hub/
├── agentic/
│ ├── workflow.py # Core LangGraph definition & edges
│ ├── agents/ # Specialized Node implementations
│ │ ├── archivist.py # Summarization & SQL persistence
│ │ ├── classifier.py # Intent & Anchor management
│ │ ├── clarificator.py # Identity recovery
│ │ ├── drafter.py # Response synthesis
│ │ ├── enricher.py # Identity mining & DB lookup
│ │ ├── escalator.py # Human hand-off logic
│ │ ├── greater.py # Initial pleasantries
│ │ ├── closer.py # Satisfaction & Closure
│ │ ├── policy_checker.py # Hallucination prevention
│ │ ├── researcher.py # SQL-based fact finding
│ │ └── supervisor.py # Conditional routing logic
│ ├── tools/
│ │ ├── tools.py # Standalone SQL tools
│ │ └── udahub_state.py # Pydantic State definition
│ └── design/
│ └── design_and_architecture.md # Detailed design specs
└── data/
├── core/ # UDAHub ticket database
└── external/ # CultPass user/experience data


## 🚀 Installation

### Using uv (Recommended)

1. Clone the repository:

```bash
git clone <repository-url>
cd uda_hub
```

2. Setup Environment:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
uv pip install -e .
```

## 🔐 Environment Configuration

Ensure your `.env` contains the following:

- `OPENAI_API_KEY`: Required for LLM reasoning.
- `TAVILY_API_KEY`: Required for web-based research fallback.
- `VOCAREUM_API_KEY`: Required for platform-specific grading (if applicable).

## 📊 Setup & Database Initialization

Initialize the "Digital World" before running the agents:

1. **External Data (CultPass):** Run `01_external_db_setup.ipynb` to load users and experiences.
2. **Core Records (UDAHub):** Run `02_core_db_setup.ipynb` to initialize the ticket tracking schema.
3. **Run App:** Open `03_agentic_app.ipynb` to start the LangGraph chat interface.

## 🛠️ Development

### Adding a New Agent

1. Define the node function in `uda_hub/agentic/agents/`.
2. Add the node to the graph in `workflow.py`.
3. Update `supervisor.py` to include the new routing logic.
4. Ensure the node points to the Archivist if it is a terminal state.

## 🤝 Contributing

Contributions are welcome! Please ensure that any changes to the state schema are reflected in `udahub_state.py` and that the Mermaid diagram in the documentation is updated accordingly.

---

*UDA-Hub — Precision automation for the modern support desk.*