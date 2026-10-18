# 🛒 Amazon E-Commerce AI Assistant & Agentic RAG

An end-to-end, production-grade **Agentic Retrieval-Augmented Generation (RAG)** platform and intelligent shopping assistant built for Amazon product catalogs.

The system features **hybrid vector + sparse search (Qdrant BM25 + OpenAI embeddings + RRF)**, **Cohere semantic reranking**, **LangGraph multi-agent orchestration with PostgreSQL memory checkpointing**, a **FastAPI** backend with automated request-ID tracking, and a feature-rich **Streamlit** user interface with real-time product cards, shopping cart management, and **LangSmith** feedback integration.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Agent Workflow](#-agent-workflow)
- [Technology Stack](#-technology-stack)
- [Repository Structure](#-repository-structure)
- [Prerequisites](#-prerequisites)
- [Environment Variables](#-environment-variables)
- [Quick Start](#-quick-start)
  - [Method 1: Docker Compose (Recommended)](#method-1-docker-compose-recommended)
  - [Method 2: Local Development with `uv`](#method-2-local-development-with-uv)
- [Data Ingestion Pipeline](#-data-ingestion-pipeline)
- [API Reference](#-api-reference)
- [Observability & Evaluation (LangSmith)](#-observability--evaluation-langsmith)
- [License](#-license)

---

## ✨ Key Features

- **Multi-Turn Stateful Agent (LangGraph)**:
  - Preserves full conversation history per `thread_id` backed by a persistent PostgreSQL checkpointer (`PostgresSaver`).
  - Capable of iterative tool execution and multi-step reasoning before concluding answers.
- **Intent Router & Guardrails**:
  - Automatically screens user inputs using structured outputs (`gpt-4.1-mini` via Instructor) to verify queries relate to in-stock inventory before executing database queries.
- **Hybrid Retrieval (Dense + BM25 + RRF)**:
  - Dense semantic search using OpenAI's `text-embedding-3-small` (1536 dims).
  - Lexical full-text search using Qdrant native BM25 sparse matching.
  - Reciprocal Rank Fusion (RRF, 3:1 weight ratio) to blend keyword precision with semantic depth.
- **Cohere Semantic Reranking**:
  - Re-evaluates top retrieved product chunks with `rerank-v4.0-fast` for maximum relevance.
- **Interactive Streamlit UI**:
  - Interactive chat interface with thread-based session persistence.
  - Dedicated **Suggestions Tab** displaying real-time product cards, ASINs, high-resolution imagery, and USD pricing.
  - **Shopping Cart Tab** for order tracking and basket calculations.
  - In-app feedback system (Thumbs up/down + optional comment prompt) synced to LangSmith.
- **Observability & Analytics**:
  - Deep instrumentation with LangSmith `@traceable` across LLM nodes, embeddings, retrieval steps, and user feedback submission.
- **Production-Ready Microservices**:
  - Full Docker containerization with `docker-compose` managing FastAPI, Streamlit, Qdrant vector database, and PostgreSQL.

---

## 🏛 System Architecture

```mermaid
flowchart TD
    User([User / Browser]) <--> StreamlitUI[Streamlit Chatbot UI\n:8502]
    StreamlitUI <-->|HTTP / REST\nwith Request-ID| FastAPIServer[FastAPI Backend\n:8002]

    subgraph Backend_Services [FastAPI & Agent Pipeline]
        FastAPIServer --> RequestMiddleware[RequestID Middleware]
        RequestMiddleware --> LangGraphApp[LangGraph Workflow\nStateGraph]
        
        LangGraphApp <-->|State Checkpoints| PostgresDB[(PostgreSQL 15\n:5434)]
        LangGraphApp --> IntentRouter[Intent Router Node\ngpt-4.1-mini]
        IntentRouter --> AgentNode[Agent Node\ngpt-4.1-mini + Instructor]
        AgentNode <-->|Tool Execution| ToolNode[Tool Node\nget_formatted_context]
    end

    subgraph Retrieval_Layer [Search & External Services]
        ToolNode --> OpenAIEmbeddings[OpenAI Embeddings\ntext-embedding-3-small]
        ToolNode --> QdrantDB[(Qdrant Vector DB\n:6335\nHybrid: Dense + BM25)]
        ToolNode --> CohereRerank[Cohere Rerank\nrerank-v4.0-fast]
    end

    subgraph Monitoring [Observability]
        LangGraphApp -.->|Traces & Usage| LangSmith[LangSmith Platform]
        FastAPIServer -.->|Feedback API| LangSmith
    end
```

---

## 🔄 Agent Workflow

The LangGraph orchestration flow operates as a cyclic state machine:

```mermaid
stateDiagram-v2
    [*] --> IntentRouterNode: User Query + Thread ID
    
    state IntentRouterNode {
        [*] --> ClassifyIntent: Check query relevance
        ClassifyIntent --> Relevant: Product Question
        ClassifyIntent --> Irrelevant: Out of Domain
    }

    Irrelevant --> [*]: Return Out-of-Stock / Clarification
    Relevant --> AgentNode: Route to Agent

    state AgentNode {
        [*] --> ReasonStep: LLM generates thoughts & tool calls
    }

    AgentNode --> ToolNode: Tool calls requested (final_answer=False)
    ToolNode --> AgentNode: Return formatted context & scores
    
    AgentNode --> [*]: Complete (final_answer=True or max_iterations reached)
```

1. **User input arrives** via `POST /rag/` accompanied by a `thread_id`.
2. **Intent Classification**: Evaluates if the question is relevant to retail inventory.
3. **Agent Reasoning**: The agent inspects available tools (dynamically parsed from docstrings via AST) and decides whether to retrieve inventory.
4. **Tool Execution**: `get_formatted_context` performs hybrid vector + BM25 search in Qdrant, applies RRF scoring, and formats context chunks.
5. **Synthesis**: The agent compiles a structured final response including bulleted specifications, item IDs, and citations.

---

## 🛠 Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.12 | Core programming environment |
| **Package Manager** | `uv` (Astral) | Ultra-fast dependency resolution and venv management |
| **API Framework** | FastAPI + Uvicorn | High-performance asynchronous API endpoints |
| **Agent Orchestration** | LangGraph & LangChain | Stateful graph workflows, tool loops, routing |
| **State Persistence** | `langgraph-checkpoint-postgres` | Multi-turn conversational memory in PostgreSQL |
| **LLMs & Schema Validation** | OpenAI `gpt-4.1-mini` + `instructor` | Structured Pydantic outputs and tool invocations |
| **Vector Database** | Qdrant | Hybrid search engine (Dense Cosine + Sparse BM25 + RRF) |
| **Embeddings** | OpenAI `text-embedding-3-small` | 1536-dimensional semantic representation |
| **Reranker** | Cohere `rerank-v4.0-fast` | High-precision cross-encoder reranking |
| **User Interface** | Streamlit | Chat frontend, suggestion cards, cart, and feedback |
| **Observability** | LangSmith | Distributed tracing, prompt latency, token counting, and RLHF feedback |
| **Containerization** | Docker & Docker Compose | Multi-container orchestration |

---

## 📁 Repository Structure

```plaintext
amazon_ecommerce/
├── api/
│   ├── Agent/
│   │   ├── agent.py               # Agent and Intent Router nodes (Instructor + OpenAI)
│   │   ├── graph.py               # LangGraph StateGraph, edges, and PostgreSQL checkpointer
│   │   ├── tools.py               # Retrieval tools, Qdrant hybrid query, Cohere reranking
│   │   └── utils.py               # AST docstring parser, AI message formatters
│   ├── processors/
│   │   └── submit_feedback.py     # LangSmith feedback dispatcher (thumbs / comments)
│   ├── app.py                     # FastAPI application setup and routing
│   ├── endpoints.py               # Alternative RAG endpoints
│   ├── middleware.py              # RequestIDMiddleware for distributed tracing
│   └── models.py                  # Pydantic schemas (RAGRequest, RAGResponse, Feedback)
├── rag/
│   ├── ingest.py                  # Dataset ingestion script into Qdrant collection
│   └── reterivalgeneration.py     # Standalone RAG pipeline & retriever functions
├── streamlit_app/
│   ├── chatbot.py                 # Streamlit UI with tabs, suggestions, cart, and feedback
│   └── config.py                  # Streamlit configuration and API URL binding
├── .dockerignore
├── .env.example                   # Environment configuration template
├── .gitignore
├── Dockerfile                     # Multi-stage/uv-based Python 3.12 container definition
├── docker-compose.yml             # Orchestration for Qdrant, Postgres, API, and UI
├── pyproject.toml                 # Project metadata and locked dependencies
└── README.md
```

---

## 📋 Prerequisites

Before running the application, make sure you have:

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/) installed.
- (Optional for local running) [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.
- API Keys for:
  - **OpenAI** (for embeddings and chat generation)
  - **LangSmith** (for tracing and evaluation)
  - **Cohere** (optional, for reranking)

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Populate the following variables:

```ini
# OpenAI Configuration
OPENAI_API_KEY="sk-proj-..."

# Qdrant Vector Database
QDRANT_URL="http://localhost:6335"
QDRANT_COLLECTION_NAME="Amazon-items-collection-02-hybrid-serach"

# Cohere Configuration (Optional Reranking)
CO_API_KEY="your-cohere-api-key"

# PostgreSQL Checkpointer (LangGraph Memory)
DB_URI="postgresql://user:password@localhost:5434/ecommerce"

# FastAPI Backend URL
API_URL="http://localhost:8002"

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY="lsv2_pt_..."
LANGCHAIN_PROJECT="amazon-ecommerce-rag"
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"

# Ingestion Dataset (Optional custom path)
INGEST_DATA_PATH="data/meta_Electronics_2022_23_with_categeory_rating_100_sample_2000.jsonl"
```

> **Note on Docker Ports**:
> Inside Docker containers, internal hostnames are used automatically via `docker-compose.yml` (`QDRANT_URL=http://ecommerceqdrant:6333`, `DB_URI=postgresql://user:password@ecommerce_postgres:5432/ecommerce`, `API_URL=http://ecommerceapi:8000`). Host-mapped ports on your machine will be `6335` (Qdrant), `5434` (Postgres), `8002` (FastAPI), and `8502` (Streamlit).

---

## 🚀 Quick Start

### Method 1: Docker Compose (Recommended)

1. **Clone the repository and prepare `.env`**:
   ```bash
   git clone https://github.com/ANURAGYADAV008/amazon_ecommerce.git
   cd amazon_ecommerce
   cp .env.example .env
   # Edit .env and supply your OPENAI_API_KEY and LANGCHAIN_API_KEY
   ```

2. **Launch all services**:
   ```bash
   docker compose up --build -d
   ```

3. **Verify running containers**:
   ```bash
   docker compose ps
   ```

4. **Access the services**:
   - **Streamlit Web UI**: [http://localhost:8502](http://localhost:8502)
   - **FastAPI Swagger Docs**: [http://localhost:8002/docs](http://localhost:8002/docs)
   - **Qdrant Web Dashboard**: [http://localhost:6335/dashboard](http://localhost:6335/dashboard)
   - **FastAPI Health Check**: [http://localhost:8002/health](http://localhost:8002/health)

5. **Stop services**:
   ```bash
   docker compose down
   ```

---

### Method 2: Local Development with `uv`

If you prefer to run services outside Docker for active debugging:

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Start Qdrant & PostgreSQL via Docker**:
   ```bash
   docker compose up -d ecommerceqdrant ecommerce_postgres
   ```

3. **Start the FastAPI backend**:
   ```bash
   uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Start the Streamlit frontend**:
   ```bash
   uv run streamlit run streamlit_app/chatbot.py --server.port 8501
   ```

---

## 📦 Data Ingestion Pipeline

To populate your Qdrant vector database with Amazon product data:

1. Place your Amazon electronics dataset (JSONL format) in `data/` or configure `INGEST_DATA_PATH` in `.env`.
2. Run the ingestion script:

**Via Docker**:
```bash
docker compose exec ecommerceapi uv run python -m rag.ingest
```

**Locally**:
```bash
uv run python -m rag.ingest
```

The script will:
- Initialize the Qdrant collection with cosine distance and 1536 vector dimension.
- Parse item titles, feature bullet points, and descriptions.
- Compute OpenAI batch embeddings.
- Upsert payload metadata (ASIN, description, average rating, image URL, price) into Qdrant.

---

## 🔌 API Reference

### 1. RAG Chat Endpoint

- **Endpoint**: `POST /rag/`
- **Description**: Submits a user question to the LangGraph agent and returns the generated answer with product context.

**Request Payload:**
```json
{
  "query": "Show me wireless noise cancelling headphones under $100",
  "thread_id": "session-unique-thread-uuid"
}
```

**Response Payload:**
```json
{
  "request_id": "c92e92c6-3ea8-4cf5-99bc-ba9fb94c25d8",
  "answer": "Here are the top wireless noise cancelling headphones currently in stock: ...",
  "used_context": [
    {
      "image_url": "https://m.media-amazon.com/images/I/...",
      "price": 79.99,
      "description": "Wireless Over-Ear Noise-Cancelling Headphones with 40H Playtime"
    }
  ],
  "trace_id": "018e6e5a-7b3b-7489-b00d-587fcf3ecda2"
}
```

---

### 2. User Feedback Endpoint

- **Endpoint**: `POST /submit_feedback/`
- **Description**: Submits user satisfaction scores and comments directly to LangSmith for evaluation and continuous learning.

**Request Payload:**
```json
{
  "run_id": "018e6e5a-7b3b-7489-b00d-587fcf3ecda2",
  "feedback_score": 1,
  "feedback_text": "Accurate recommendation and clear specifications!",
  "feedback_source_type": "api"
}
```

**Response Payload:**
```json
{
  "request_id": "d04b868e-5fa2-4c22-b5b6-7eb509749ba0",
  "status": "success"
}
```

---

### 3. Health Check

- **Endpoint**: `GET /health`
- **Response**: `{"status": "ok"}`

---

## 📊 Observability & Evaluation (LangSmith)

All operations are natively instrumented with **LangSmith**:
- **Execution Traces**: Inspect every prompt template, LLM invocation latency, input/output token usage, and retrieval results.
- **Agent Reasoning Loops**: Trace iterative decision-making in `agent_node` and tool execution in `tool_node`.
- **User Feedback Loop**: User ratings (👍 / 👎) and detailed remarks in the Streamlit UI automatically link to their exact trace run in LangSmith for offline evaluation, debugging, and dataset creation.

Access your telemetry at [smith.langchain.com](https://smith.langchain.com).

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
