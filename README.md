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

UDA-Hub is an intelligent customer support platform built on a **Concierge-First model** that maximizes efficiency and data integrity. The system uses a Directed Acyclic Graph (DAG) with a **Concierge Agent at the entry point** that triages requests and enforces an **Identity Gate**. Only authenticated users proceed through the heavy path of enrichment, research, and resolution. Unauthenticated inquiries are handled gracefully by the Concierge with appropriate responses.

## ✨ Key Features

- **Concierge-First Triage:** The Concierge Agent handles initial classification and acts as the "Front Desk," enforcing the Identity Gate before heavy operations begin.
- **Identity Gate Security:** A strict separation between social interactions (greetings/closures) and authenticated support workflows. Only identified users access the full investigation path.
- **Durable Memory (Historian):** The Historian Agent scans past tickets to detect recurring issues and ensures repeat problems are escalated to human specialists rather than re-automated.
- **Multi-Stage Investigation:** The heavy path routes through Enricher → Historian → Classifier → Supervisor, ensuring data integrity and optimal routing.
- **Self-Correcting QA Loop:** The Policy Checker audits drafts against retrieved facts and can route failed responses back to the Drafter for revision.
- **Efficient Archival:** The Archivist finalizes interactions and logs metadata to `udahub.db` with verified accuracy.

## 🏗️ Architecture

UDA-Hub is organized as a **Directed Acyclic Graph (DAG)** with an **Identity Gatekeeper at the entry point**. The Concierge Agent enforces strict separation between social interactions and authenticated support workflows. Responsibilities are separated across layers so the system is resilient, auditable, and easy to extend.

**Core Flow Stages:**

1. **Triage & Identity Gate (Concierge):** Incoming messages are triaged. Greetings/closures/unidentified requests are handled immediately. Support requests with identification proceed to the heavy path.
2. **Enrichment & Memory (Enricher & Historian):** The Enricher authenticates users by querying the `cultpass.db`. The Historian scans past tickets to detect recurring issues.
3. **Classification & Routing (Classifier & Supervisor):** The Classifier identifies the technical domain (Billing, Tech, Account). The Supervisor routes to either automated (Researcher) or escalated (Escalator) paths.
4. **Research & Drafting (Researcher & Drafter):** The Researcher gathers grounded facts from the Knowledge Base. The Drafter synthesizes empathetic responses.
5. **Quality Assurance (Policy Checker):** Audits responses for compliance. Failed responses route back to Drafter for revision (self-correction loop).
6. **Finalization (Archivist):** Logs verified interactions and updates metadata in `udahub.db`.

**Data Flow (High Level):**
```
User Input → Concierge (Identity Gate)
  ├─ Greeting/Unidentified → Archivist → Resolved/Logged
  └─ Support + Identified → Enricher → Historian → Classifier → Supervisor
      ├─ Standard/Automated → Researcher → Drafter → Policy Checker
      │   ├─ FAIL → Drafter (self-correction)
      │   └─ PASS → Archivist → Resolved/Logged
      └─ High Urgency/Repeat → Escalator → Archivist → Resolved/Logged
```

For detailed design and sequence diagrams, see:
> [uda_hub/agentic/design/design_and_architecture.md](uda_hub/agentic/design/design_and_architecture.md)

## 🏗️ Key Components

| Agent | Responsibility | Key I/O |
|---|---|---|
| **Concierge** | Front Desk triage & Identity Gate enforcement. Routes greetings to Archivist; authenticated requests to Enricher. | Input: ticket_text, messages; Output: triage_intent, is_identified, ai_response |
| **Enricher** | Authenticates users by querying `cultpass.db` for identity data. | Input: user_email/name; Output: user_id, customer_tier, subscription_status |
| **Historian** | Retrieves "Durable Memory" by scanning past tickets for recurring issues. | Input: user_id; Output: user_history, is_recurring, similar_ticket_id |
| **Classifier** | Identifies technical domains (Billing, Tech, Account) and urgency levels. | Input: ticket_text; Output: support_category, urgency |
| **Supervisor** | Router node that decides between automated (Researcher) or escalated (Escalator) paths. | Input: urgency, is_recurring; Output: routing_decision |
| **Researcher** | Retrieves grounded facts from internal Knowledge Base and company policies. | Input: support_category; Output: research_facts, retrieval_confidence |
| **Escalator** | Hand-off logic for high-urgency or repeat issues to human managers. | Input: urgency, is_recurring; Output: escalation_summary |
| **Drafter** | Synthesizes research and history into empathetic, professional responses. | Input: research_facts, user_history; Output: ai_response |
| **Policy Checker** | Audits responses for compliance and accuracy. Can reject drafts for revision. | Input: ai_response, research_facts; Output: policy_grade (PASS/FAIL) |
| **Archivist** | Finalizes interactions by logging to `udahub.db` and updating metadata. | Input: ai_response, status; Output: archive_summary |

## 📁 Project Structure

```
uda_hub/
├── agentic/
│ ├── workflow.py                # Core LangGraph DAG definition & edges
│ ├── agents/                    # Specialized Node implementations
│ │ ├── concierge.py            # Entry point & Identity Gate enforcement
│ │ ├── enricher.py             # User authentication & cultpass.db lookup
│ │ ├── historian.py            # Durable Memory & recurring issue detection
│ │ ├── classifier.py           # Technical domain & urgency classification
│ │ ├── supervisor.py           # Router node (Researcher vs Escalator)
│ │ ├── researcher.py           # Fact-finding from Knowledge Base
│ │ ├── escalator.py            # High-urgency & repeat issue escalation
│ │ ├── drafter.py              # Response synthesis & empathetic composition
│ │ ├── policy_checker.py       # QA auditor & hallucination prevention
│ │ ├── archivist.py            # Finalization & udahub.db persistence
│ │ └── [deprecated] clarificator.py # Older identity recovery agent
│ ├── tools/
│ │ ├── tools.py                # Standalone SQL & utility functions
│ │ ├── logging.py              # Logging utilities
│ │ └── udahub_state.py         # Pydantic State definition
│ └── design/
│ └── design_and_architecture.md # Detailed design specs with Mermaid diagram
├── data/
│ ├── core/                     # UDAHub ticket & knowledge base schema
│ └── external/                 # CultPass user, subscription, and experience data
└── utils.py                    # Shared utilities
```


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
2. Add the node to the graph and edges in `workflow.py`.
3. Update routing logic in `supervisor.py` or `concierge.py` as appropriate.
4. Ensure the new node integrates into the DAG without breaking the Identity Gate or self-correction loop.
5. Update the Mermaid diagram in `design_and_architecture.md` to reflect changes.
6. Test with the notebooks (`03_agentic_app.ipynb`) before deployment.

### State Management

The system uses `UDAHubState` (defined in `udahub_state.py`) to maintain turn-based accuracy. Key considerations:

- **Preserved across turns:** `messages`, `user_id`, `ticket_id`
- **Reset each turn:** `policy_grade`, `research_facts`, `ai_response` (managed by Entry Bridge)
- **Concierge-Controlled:** `is_identified`, `triage_intent` determine routing

## 🤝 Contributing

Contributions are welcome! Please ensure that any changes to the state schema are reflected in `udahub_state.py` and that the Mermaid diagram in the documentation is updated accordingly.

---

*UDA-Hub — Precision automation for the modern support desk.*