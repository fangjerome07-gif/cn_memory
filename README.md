# cn_memory

A persistent, zero-dependency Chinese memory plugin for **Hermes Agent**, designed to give AI long-term cross-session memory through LLM-powered automatic extraction and hybrid retrieval.

## Why cn_memory?

Most AI agents forget everything between sessions. `cn_memory` solves this with two automatic lifecycles:

1. **Every conversation automatically extracts memories** — At the end of each conversation, the plugin analyzes the full context using an LLM, extracts high-value facts (preferences, rules, projects, health data, tools, etc. across 8 categories), and stores them in a local SQLite database. No manual `save` required.
2. **Every conversation automatically injects relevant memories** — At the start of each conversation, the plugin retrieves the most relevant memories using hybrid search and injects them directly into the system prompt. Your agent remembers what matters.

These two loops run silently in the background. The agent just... remembers.

## Features

### Core Architecture

- **Zero Infrastructure Dependency** — Runs entirely on **SQLite** + **numpy**. No Docker, no PostgreSQL, no Qdrant, no Neo4j. One `pip install` and you're running.
- **Chinese (CJK) Optimization** — Purpose-built for Chinese text with **Trigram tokenization** for FTS5 full-text search and a **LIKE fallback** mechanism for maximum recall on short queries and proper nouns.
- **Multi-Agent Profile Isolation** — Automatically creates separate SQLite databases per agent identity. No cross-contamination between profiles.

### Retrieval & Scoring

- **Hybrid Retrieval** — Combines dense vector search (cosine similarity via numpy) with sparse keyword matching (SQLite FTS5 BM25) and fuzzy LIKE search.
- **Multi-Dimensional Scoring Formula**:
  - Vector Similarity: **40%**
  - BM25 Keyword Rank: **30%**
  - LIKE Fuzzy Match: **20%**
  - Base Importance: **15%**
  - Time Decay (Recency): **15%**
  - Type Weight: **10%**
- **Rule Priority** — Memories tagged as `rule` type get the highest weight of **1.0**, ensuring critical guidelines are never overlooked.

### Memory Lifecycle

- **Automatic Deduplication & Superseding**:
  - Similarity > 0.98: automatically merged (no duplicates)
  - Similarity 0.92–0.98: older entry marked `superseded` by the newer, more accurate one
- **Profile Aggregation** — Every 50 write operations, the plugin automatically summarizes scattered facts into a structured user profile summary using LLM.
- **TTL Expiration & Auto-Archiving** — Optional Time-To-Live per memory entry. Expired memories are archived (not deleted) for historical reference.
- **8 Memory Categories**: `preference`, `profile`, `project`, `rule`, `health`, `tool`, `relationship`, `general`

### Bonus

- **Built-in Todo Tracking** — A state machine that automatically extracts and tracks task/todo items from conversations (`open` → `done` → `cancelled`).

## Installation

1. Copy or clone this plugin into your Hermes plugins directory:
   ```bash
   cp -r cn_memory ~/.hermes/plugins/
   ```
2. Install dependencies:
   ```bash
   pip install numpy pyyaml
   ```
   *(SQLite and other standard libraries are included in Python.)*

## Configuration

### Hermes config.yaml

```yaml
memory:
  provider: cn_memory
  memory_char_limit: 2200  # Max characters for injected memory context
```

### Plugin config

Create `~/.hermes/cn_memory/config.json`:

```json
{
  "embedding_endpoint": "http://127.0.0.1:18080/v1/embeddings",
  "embedding_model": "bge-small-zh-v1.5",
  "memory_char_limit": 2200,
  "llm_provider": "openai",
  "llm_base_url": "https://api.openai.com/v1",
  "llm_model": "gpt-4o-mini",
  "llm_api_key": "your-api-key-here",
  "llm_timeout": 8.0
}
```

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `CN_MEMORY_EMBEDDING_ENDPOINT` | Embedding API endpoint (OpenAI compatible) | `http://127.0.0.1:18080/v1/embeddings` |
| `CN_MEMORY_EMBEDDING_MODEL` | Embedding model name | `bge-small-zh-v1.5` |
| `CN_MEMORY_CHAR_LIMIT` | Max character limit for retrieved memory contexts | `2200` |
| `CN_MEMORY_LLM_BASE_URL` | LLM API base URL for fact extraction | None |
| `CN_MEMORY_LLM_MODEL` | LLM model name for fact extraction | None |
| `CN_MEMORY_LLM_API_KEY` | LLM API key for fact extraction | None |
| `CN_MEMORY_LLM_TIMEOUT` | Timeout (seconds) for LLM extraction requests | `8.0` |

## Usage

### Tools (registered automatically)

**`cn_memory_store`** — Store a structured fact:
- `content` (string, required): The fact to remember
- `memory_type` (string): `preference`, `profile`, `project`, `rule`, `health`, `tool`, `relationship`, `general`
- `importance` (int): 1 (low) to 5 (critical)
- `confidence` (float): 0.0 to 1.0
- `ttl_days` (float): Days before expiration (optional)

**`cn_memory_search`** — Search memories:
- `query` (string, required): Search term or semantic phrase

### CLI Viewer

```bash
python cn_memory_viewer.py --profile default
```

## How It Works

```
Conversation Ends
    │
    ▼
┌─────────────────────┐
│ LLM Fact Extraction  │  ← Analyzes full conversation
│ (8 categories)       │  ← Extracts structured facts
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ SQLite Storage       │  ← Vector embedding + FTS5 index
│ (per-profile DB)     │  ← Deduplication & superseding
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Profile Aggregation  │  ← Every 50 writes
│ (LLM summary)        │  ← Structured user profile
└─────────────────────┘


Conversation Starts
    │
    ▼
┌─────────────────────┐
│ Hybrid Retrieval     │  ← Vector + BM25 + LIKE
│ (query → memories)   │  ← Multi-dimensional scoring
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Inject to Prompt     │  ← Top-K memories in system prompt
│ (2200 char limit)    │  ← Agent "remembers" context
└─────────────────────┘
```

## Comparison

| Feature | cn_memory | Mem0 | Hindsight |
|---|---|---|---|
| Infrastructure | **SQLite only** | Qdrant + Neo4j | PostgreSQL + pgvector |
| Chinese (CJK) optimization | **Trigram + LIKE fallback** | ❌ | ❌ |
| Auto-extraction per conversation | **✅ LLM-powered** | ✅ | ✅ |
| Auto-injection per conversation | **✅ Into system prompt** | ✅ | ✅ |
| Scoring transparency | **Configurable weights** | Black box | Reflection-based |
| Todo tracking | **✅ Built-in** | ❌ | ❌ |
| Deployment | **pip install** | Docker recommended | Docker required |

## License

[MIT License](LICENSE)
