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


---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
