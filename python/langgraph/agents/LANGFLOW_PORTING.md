# Porting the RAG Agents to Langflow

How to rebuild [basic-RAG](basic-RAG/) and [corrective-rag](corrective-rag/) as
Langflow flows, node by node.

> Component names are verified against `langflow-ai/langflow` `main` as of
> **v1.13.2** (Nov 2026) — the `display_name` in each component's source under
> `src/lfx/src/lfx/components/` and `src/bundles/`. Names drift between
> versions; if your sidebar differs, check your installed version.

> **Chroma is gone.** Langflow 1.13 retired Chroma. The **Chroma DB** and
> **Local DB** nodes still exist only as legacy shims that raise
> `ChromaMigrationRequiredError`. The existing `langgraph-docs-db/` and
> `index/chroma/` stores in this repo can't be reused — re-ingest into a
> **Knowledge** base instead (backends: SQLite by default, Postgres, OpenSearch).

## Running Langflow locally

[langflow/docker-compose.yml](langflow/docker-compose.yml) runs one Langflow
container plus Postgres. Flows are JSON documents loaded into that one server —
there is no container per flow or per component.

```
  ┌──────────────────── langflow container ─────────────────────┐
  │                                                             │
  │   ./flows/*.json   ──► loaded on startup                    │
  │   ./components/    ──► custom components (e.g. a grader)    │
  │   built-in nodes   ──► Knowledge, Prompt Template, If-Else, │
  │                        Loop, DuckDuckGo Search, ...         │
  └──────────┬──────────────────────────────┬───────────────────┘
             │ HTTPS                        │
             ▼                              ▼
   ┌──────────────────────┐      ┌──────────────────────┐
   │ OpenAI / Gemini      │      │ postgres container   │
   │ DuckDuckGo/Firecrawl │      │ Langflow DB          │
   └──────────────────────┘      └──────────────────────┘
```

Create `langflow/.env` (git-ignored):

```
POSTGRES_PASSWORD=<any local password>
OPENAI_API_KEY=<for basic-RAG>
GOOGLE_API_KEY=<for corrective-rag>
FIRECRAWL_API_KEY=<optional, instead of DuckDuckGo>
LANGFLOW_HOST_PORT=<optional, defaults to 7860>
```

**Windows / WSL:** Hyper-V can reserve port ranges that include 7860. Docker
Desktop then fails with `ports are not available ... /forwards/expose returned
unexpected status: 500`. Check with
`netsh.exe interface ipv4 show excludedportrange protocol=tcp` and set
`LANGFLOW_HOST_PORT` to a free port such as `17860`.

Then, from `langflow/`:

```bash
docker compose up -d
```

The editor is at http://localhost:7860 (or your `LANGFLOW_HOST_PORT`).

Details:
- The image is `langflowai/langflow-all`, the full-bundles tier. The default
  `langflowai/langflow` image leaves out the DuckDuckGo and Firecrawl bundles.
- Knowledge bases are stored under `LANGFLOW_KNOWLEDGE_BASES_DIR`, which is set
  to a path on the `langflow-data` volume. The default
  (`~/.langflow/knowledge_bases`) is inside the container, so recreating it
  leaves the knowledge base listed but fails with "SQLite knowledge base
  storage is missing".
- Auto-login is on, which `LANGFLOW_LOAD_FLOWS_PATH` requires. That makes it
  local-only, so the port is bound to `127.0.0.1`.
- **Save your work into the repo:** in the editor, export each flow to
  `langflow/flows/`. On restart, Langflow loads those files but by default
  won't overwrite a flow with the same name that you've edited in the UI.
- Custom components go in `langflow/components/<category>/<name>.py`. The
  folder name becomes the sidebar category. Don't leave an empty category
  folder: Langflow logs `error[bundle-empty]` for a folder with no `.py` files.
- Verified on Langflow 1.13.2: every component named in this guide is present
  in the `langflow-all` image.

## Key insight: these are one flow in two stages

basic-RAG is already self-correcting: it grades each retrieved chunk and
rewrites the question when nothing relevant comes back. corrective-rag keeps
the grading but replaces the retry loop with a **web-search fallback**. Build
basic-RAG first, then turn it into corrective-rag by changing one condition
and one branch.

```
  basic-RAG                                corrective-rag
  ─────────                                ──────────────

  question                                 question
     │                                        │
     ▼                                        ▼
  retrieve ◄──────────┐                    retrieve (k=3)
     │                │                       │
     ▼                │                       ▼
  grade each chunk    │                    grade each chunk
     │                │                       │
     ▼                │                       ▼
  any relevant? ─no─► rewrite              >50% failed? ─yes─► rewrite
     │ yes                                    │ no                │
     ▼                                        │                   ▼
  generate (+history)                         │               web search
     │                                        │                   │
     ▼                                        ▼                   ▼
    END                                    generate ◄─────────────┘
                                              │
                                              ▼
                                             END
```

## Stage 0 — Ingest flow (shared)

Build indexing as its own flow; run it once, then the chat flow only reads.

| Source | Langflow |
|---|---|
| basic-RAG `utils.py`: ~50 doc URLs, `WebBaseLoader`, 200-token chunks, overlap 0, Chroma + OpenAI embeddings | **URL** → **Split Text** (Chunk Size 800, Chunk Overlap 0) → **Knowledge** (mode *Ingest*, OpenAI embedding model). Built: [langflow/flows/basic-rag-ingest.json](langflow/flows/basic-rag-ingest.json), knowledge base `langgraph-docs` (`text-embedding-3-small`) |
| corrective-rag `index_tool.py`: PDF upload, `PyMuPDFLoader`, `SemanticChunker`, Gemini embeddings | **Read File** → **Split Text** → **Knowledge** (mode *Ingest*, Google embedding model) |

Notes:
- **Knowledge** is one node with an *Ingest* / *Retrieve* tab. It replaced the
  separate **Knowledge Ingestion** and **Knowledge Base** nodes (now legacy).
  The embedding model is chosen when the knowledge base is created and stored
  with it, so the chat flow doesn't need its own embeddings node.
- The **File** component is now displayed as **Read File**.
- **Split Text** counts characters, not tokens. basic-RAG's 200 tiktoken
  tokens is roughly 800 characters.
- Calling a flow through `/api/v1/run/<flow-id>` needs a Langflow API key
  (`x-api-key` header), even with auto-login on. Create one under
  Settings → API Keys.
- Langflow has no semantic chunker; fixed-size **Split Text** is the practical
  substitute (or a custom component wrapping `SemanticChunker`).
- The bundled *Vector Store RAG* starter template is built on **Knowledge**,
  **Prompt Template**, **Parser** and **Agent** — a good reference to open
  alongside this guide.

## Stage 1 — basic-RAG ([agent.py](basic-RAG/agent.py))

| LangGraph node | Langflow components |
|---|---|
| `retrieve_documents` | **Chat Input** → **Knowledge** (mode *Retrieve*, `top_k`) |
| `grade_documents` — one LLM call per chunk, `GradeDocument{score: yes/no}` | **Loop** (outputs *Item* per chunk, *Done* when finished) → **Prompt Template** (`GRADE_PROMPT`) → **Structured Output** `{score}`. Simpler: one **Structured Output** call that grades all chunks and returns only the relevant ones. |
| `decide_to_generate` | **If-Else** (operators on text: relevant docs non-empty?). Alternative: **Smart Router**, which routes with an LLM and can merge grading and deciding into one node. |
| `rewrite_question` | **Prompt Template** (`REWRITE_PROMPT`) → **Language Model** → second **Knowledge** retrieve |
| `generate_response` + `messages` history | **Message History** + **Prompt Template** (`RAG_PROMPT` with `{history}`, `{context}`, `{question}`) → **Language Model** → **Chat Output** |
| Formatting chunks into `{context}` | **Parser** (turns retrieved data into text) |
| `MemorySaver` + `thread_id` | Langflow session ID — built in, nothing to configure |

**Fix during the port:** the `rewrite_question → retrieve_documents` cycle in
basic-RAG has no iteration limit — if no chunk ever passes grading, it loops
forever. Langflow flows are essentially DAGs anyway, so do **one** rewrite pass
and then generate (or answer "I don't know").

## Stage 2 — corrective-rag ([graph.py](corrective-rag/graph.py))

Starting from the Stage 1 flow, change three things:

1. **Threshold decision.** `grade_node.py` sets `web_search = "Yes"` when more
   than 50% of chunks fail (`CORRECTION_THRESHOLD = 0.5`), not only when all
   fail. Have the grader output a fail ratio and point **If-Else** at `> 0.5`.
2. **Correction branch.** Rewrite → web search → generate. No loop back to
   retrieval, so this is a plain DAG:
   **Prompt Template** (rewriter) → **Language Model** → **DuckDuckGo Search**
   (or **Firecrawl Search API**) → merge with the filtered chunks → generate.
   Both search components ship as bundles (`src/bundles/duckduckgo`,
   `src/bundles/firecrawl`).
3. **Models.** Gemini 2.5 Flash for the LLM and `gemini-embedding-001` for
   embeddings (`models/LLM.py`, `models/EM.py`). In Langflow use **Language
   Model** with the Google provider, and pick the Google embedding model when
   creating the knowledge base. The standalone **Google Generative AI
   Embeddings** component defaults to `models/text-embedding-004`, so change
   the model name if you use it. The prompts in `tools/` copy over unchanged.

## Build order

1. Ingest flow + plain retrieve → generate chat. Confirms the knowledge base
   is wired correctly. **Done:**
   [langflow/flows/basic-rag-chat.json](langflow/flows/basic-rag-chat.json):

   ```
   Chat Input ─┬─► Knowledge (Retrieve, top_k 4) ─► Parser ({content}) ─┐
               │                                                         ▼
               │   Message History (Retrieve, last 10) ───────────► Prompt Template ─► Language Model ─► Chat Output
               └──────────────────────────────────────────────────►  (RAG_PROMPT)      (gpt-6.1-terra)
   ```

   It answers basic-RAG's three test questions, and the follow-ups resolve
   "it" and "that" from the message history.
   **Model:** the flow uses `gpt-6.1-terra` rather than basic-RAG's `gpt-4o`.
   The 1.13 model picker only lists newer models (`gpt-6.1-*`, `gpt-6-*`,
   `gemini-4-*`, ...). If a flow names a model that isn't listed, the editor
   swaps in the first listed model and auto-saves the change. Pick a model
   from the list so the model shown on the canvas is the one that runs.

   **Generating flows in code:** `lfx` `Graph.dump()` writes the format the
   backend runs, not the one the editor draws. Nodes show up as empty boxes
   with no connection lines, and a Knowledge node in *Retrieve* mode keeps
   its *Ingest* fields visible. Run
   [langflow/tools/fix_flow_ui.py](langflow/tools/fix_flow_ui.py) on the
   dumped JSON before importing it. After you open a flow in the editor,
   export it again so the repo copy matches what the editor saved.
2. Add grading and If-Else with one rewrite retry → **basic-RAG**.
3. Switch the condition to the 50% threshold and add the web-search branch →
   **corrective-rag**.

## Mapping for the rest of the repo

| Pattern | Langflow equivalent |
|---|---|
| agentic-RAG | **Agent** component with **Knowledge** and web-search tools |
| arxiv-researcher | *Deep Research Agent* template + **arXiv** component |
| ecommerce-hierarchical-system | *Multi Agent Flow* template |
| langmem | *Memory Chatbot* template |
| mcp-agent-template, react-mcp-chat | Agent with **MCP Tools** |
| persistence | No direct equivalent (LangGraph checkpointing) |
| mcp-auth-demo | Mostly unnecessary — Langflow has its own auth |
| langsmith observability | Langflow's LangSmith / Langfuse integrations |
