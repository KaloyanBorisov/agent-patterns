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
PHOENIX_API_KEY=<optional, turns on Phoenix tracing; same value as basic-RAG/.env>
PHOENIX_COLLECTOR_ENDPOINT=<optional, same value as basic-RAG/.env>
LANGFLOW_API_KEY=<for tools/smoke_test.py; create under Settings -> API Keys>
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
| `grade_documents` — one LLM call per chunk, `GradeDocument{score: yes/no}` | Built: **Parser** (chunks joined by `---`) → **Prompt Template** (`GRADE_PROMPT`, changed to grade all chunks at once) → **Language Model**, which returns the relevant chunks or `NONE`. Per-chunk alternative: **Loop** (*Item* per chunk, *Done* when finished) → **Prompt Template** → **Structured Output** `{score}`. |
| `decide_to_generate` | **If-Else**, regex `\W*NONE\W*$` on the grader output. *False* carries the relevant chunks to generation; *True* carries the question (its *Case True* is wired to Chat Input) to the rewrite. Alternative: **Smart Router**, which routes with an LLM. |
| `rewrite_question` | **Prompt Template** (`REWRITE_PROMPT`) → **Language Model** → second **Knowledge** retrieve → second grade |
| `generate_response` + `messages` history | **Message History** + **Prompt Template** (`RAG_PROMPT` with `{history}`, `{context}`, `{question}`) → **Language Model** → **Chat Output** |
| Formatting chunks into `{context}` | **Parser** (turns retrieved data into text) |
| `MemorySaver` + `thread_id` | Langflow session ID — built in, nothing to configure |

**Fix during the port:** the `rewrite_question → retrieve_documents` cycle in
basic-RAG has no iteration limit — if no chunk ever passes grading, it loops
forever. The port unrolls it into **one** rewrite pass, and answers a fixed
"I don't know" if the retry also finds nothing.

**Why not a real loop?** Langflow does run cycles, but only one kind. An edge
may point backwards only into an output flagged `allows_loop=True`
(`lfx/custom/custom_component/component.py`, `_is_loop_connection`), and among
built-in components only **Loop**'s *Item* output has it. Loop is a
**for-each** over a list it already has, so it fits per-chunk grading. The
rewrite cycle is a **while** (repeat until a chunk is relevant), and If-Else
can't wire its *True* output back into an earlier Knowledge node. For N
retries instead of one, write a custom component that runs the loop inside a
single node.

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
2. Add grading and If-Else with one rewrite retry → **basic-RAG**. **Done:**
   same file, generated by
   [langflow/tools/build_basic_rag_chat.py](langflow/tools/build_basic_rag_chat.py):

   ```
   Chat Input ─► Knowledge ─► Parser ─► Prompt (GRADE) ─► LLM ─► If-Else: NONE?
                                                                   │
        ┌──────────── False: relevant chunks ──────────────────────┤
        ▼                                                          │ True: question
   Prompt (RAG_PROMPT) ─► LLM ─► Chat Output                       ▼
        ▲                                         Prompt (REWRITE) ─► LLM
   Message History                                                 │ rewritten question
                                                                   ▼
                                       Knowledge ─► Parser ─► Prompt (GRADE) ─► LLM ─► If-Else: NONE?
                                                                                         │
                     Prompt (RAG_PROMPT) ─► LLM ─► Chat Output ◄─── False ───────────────┤
                                                                                         │ True
                                                              Chat Output ("I don't know ...") ◄┘
   ```

   21 nodes. Each answer path ends in its own **Chat Output**; If-Else
   switches off the other branch, so only one of the three replies per
   question. The *generate* tail is duplicated per branch rather than merged
   into one Prompt Template, which keeps each input fed by a single edge.
   The answer prompt uses the user's original question, not the rewritten
   one (agent.py answers the rewritten one).

   Rebuild and re-import:

   ```bash
   cd langflow
   docker compose exec -T langflow python - < tools/build_basic_rag_chat.py > /tmp/chat.json
   python3 tools/fix_flow_ui.py /tmp/chat.json
   ```

   Then import `/tmp/chat.json` in the editor, or `PATCH /api/v1/flows/<id>`.

   Verified through `/api/v1/run`: the three test questions answer on the
   first branch; off-topic questions reach both the post-rewrite answer and
   the fixed reply. Findings:
   - **The grader is lenient and not deterministic.** `GRADE_PROMPT`'s
     "keywords or semantic meaning" rule passed a `get_weather(city)` example
     for "What city is the capital of France?", and the same question can
     grade differently between runs. This is the original prompt's behaviour.
   - **The fixed reply needs its own branch.** With context `NONE`, the model
     ignored RAG_PROMPT's "say that you don't know" and answered from its own
     knowledge, hence the second If-Else.
   - **`Graph.dump()` gotchas.** It walks back from a single end node, so the
     script dumps once per Chat Output and merges. It also writes a typed
     *Case True* text as a serialized `Message` object, which the backend
     reads as blank; the script stores it as a plain string, as the editor does.
   - **Edges into advanced fields are dropped by the editor.** *Case True* is
     an advanced (hidden) field, so it has no handle. The backend ran the
     Chat Input → *Case True* edge, but the editor removed it on open
     ("Some connections were removed because they were invalid") and
     auto-saved, which would have sent `NONE` to the rewriter. The script sets
     `advanced: False` on that field. Open a generated flow in the editor once
     before trusting API-only tests.
3. Switch the condition to the 50% threshold and add the web-search branch →
   **corrective-rag**.

## Tracing with Phoenix

basic-RAG registers Phoenix in `agent.py` (`phoenix.otel.register` +
`LangChainInstrumentor`). Langflow has this built in
(`langflow/services/tracing/arize_phoenix.py`), so there's no code to port:
[docker-compose.yml](langflow/docker-compose.yml) passes `PHOENIX_API_KEY` and
`PHOENIX_COLLECTOR_ENDPOINT` into the container, and tracing turns on when the
key is set. Langflow appends `/v1/traces` to the endpoint, as `agent.py`
does, so the same values work. Recreate the container after changing them
(`docker compose up -d`).

What shows up in Phoenix, checked against Phoenix Cloud:

- **One project per flow,** named after the flow with spaces turned into
  dashes: `basic-RAG-chat`, not basic-RAG's `basic-RAG`.
- **One trace per run,** with a span per node (`Knowledge (Knowledge-retrieve)`,
  `If-Else (IfElse-relevant)`, ...), not per LangGraph node.
- **LLM calls are separate traces.** Each Language Model call's
  `ChatOpenAI` / `ChatPromptTemplate` spans arrive as their own 3-span trace,
  not nested under the run, so prompts and token counts are not under the
  node that made the call.
- **Runs that take the retry branch have many repeated spans.** After an
  If-Else excludes a branch, the engine re-checks upstream nodes and the
  tracer records every check: one off-topic question gave 99 spans (17
  If-Else, 10 Chat Output). The repeats take about 1 ms and reuse results;
  the LLM still ran only 3 times (3 separate LLM traces).
- **Phoenix Cloud can return 503s.** The exporter retries and then drops the
  batch (`Failed to export spans batch` in `docker compose logs langflow`),
  so a trace can be missing spans.

## Smoke test

[langflow/tools/smoke_test.py](langflow/tools/smoke_test.py) replaces
`agent.py`'s `__main__`. It runs the three follow-up questions on one session
and one off-topic question through `/api/v1/run`, and prints which Chat
Output answered each:

```bash
cd langflow
python3 tools/smoke_test.py
```

It reads `LANGFLOW_API_KEY` and `LANGFLOW_HOST_PORT` from the environment or
`langflow/.env`, finds the flow by name (`basic-RAG chat`, or pass
`--flow-id`), and exits 1 if a question gets no answer or more than one
answer. Which branch a question takes can vary between runs because the
grader is an LLM, so branches are reported, not asserted.

## Component reference: one Python module per node

Every node on the canvas is one Python class in its own module. Each class
subclasses `Component` and declares `inputs` (the fields on the node),
`outputs` (the dots on the right edge), and one method per output that
returns `Message`, `Data` or `DataFrame`. Edges only carry those return
values. Your own `langflow/components/<category>/<name>.py` follows the same
pattern.

Paths below are relative to `site-packages/lfx/components/` in the
`langflow-all:1.13.2` image (`src/lfx/src/lfx/components/` in the Langflow
repo), checked in the running container.

```
  ported project                       Langflow module (lfx/components/...)
  ──────────────                       ────────────────────────────────────
  basic-RAG/utils.py
    WebBaseLoader(LANGGRAPH_DOCS) ───► data_source/url.py            URLComponent
    RecursiveCharacterTextSplitter ──► processing/split_text.py      SplitTextComponent
    Chroma.from_documents ───────────► files_and_knowledge/knowledge.py  (Ingest)
        └─ = langflow/flows/basic-rag-ingest.json

  basic-RAG/agent.py
    graph.invoke({"question"}) ──────► input_output/chat.py          ChatInput
    retrieve_documents ──────────────► files_and_knowledge/knowledge.py  (Retrieve)
    "\n\n".join(page_content) ───────► processing/parser.py          ParserComponent
    state["messages"] + MemorySaver ─► models_and_agents/memory.py   MemoryComponent
    RAG_PROMPT.format(...) ──────────► models_and_agents/prompt.py   PromptComponent
    llm.invoke (utils.llm) ──────────► models_and_agents/language_model.py
    grade_documents (GRADE_PROMPT) ──► prompt.py + language_model.py
    decide_to_generate ──────────────► flow_controls/conditional_router.py
    rewrite_question (REWRITE_PROMPT) ► prompt.py + language_model.py
    result["generation"] ────────────► input_output/chat_output.py   ChatOutput
        └─ = langflow/flows/basic-rag-chat.json
```

### Built: ingest flow ([basic-rag-ingest.json](langflow/flows/basic-rag-ingest.json))

| Node | Module · class | Idea | Ported from |
|---|---|---|---|
| **URL** | `data_source/url.py` · `URLComponent` | Fetches pages over HTTP and strips HTML to text. Output *Extracted Pages* is a `DataFrame` (one row per URL, `text` + `url`). Replaces `WebBaseLoader`. | `basic-RAG/utils.py` `LANGGRAPH_DOCS` + `WebBaseLoader` |
| **Split Text** | `processing/split_text.py` · `SplitTextComponent` | Character-based chunking over the `text` column, keeps the other columns as metadata. 800 chars ≈ 200 tokens. | `RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=200)` |
| **Knowledge** (*Ingest*) | `files_and_knowledge/knowledge.py` · `KnowledgeComponent` | Embeds the chunks and writes them to a named knowledge base. The embedding model is stored with the base, so readers don't configure one. Replaces the Chroma store + `OpenAIEmbeddings`. | `Chroma.from_documents(..., persist_directory=DB_PATH)` |

### Built: chat flow ([basic-rag-chat.json](langflow/flows/basic-rag-chat.json))

| Node | Module · class | Idea | Ported from |
|---|---|---|---|
| **Chat Input** | `input_output/chat.py` · `ChatInput` | Entry point. Turns the user's text into a `Message` with a session ID and stores it. Fans out to retrieval and the prompt. | `InputState.question` / `graph.invoke({"question": ...})` |
| **Knowledge** (*Retrieve*) | same module as Ingest | Same class, other tab: the `mode` field switches inputs and outputs. Embeds `search_query` and returns `top_k` chunks as a `DataFrame`. | `retrieve_documents` → `retriever.invoke(question)` |
| **Parser** | `processing/parser.py` · `ParserComponent` | Renders each row through a template (`{content}`) and joins rows into one text `Message`. Glue between `DataFrame` and prompt variables. | `"\n\n".join(doc.page_content ...)` in `generate_response` |
| **Message History** | `models_and_agents/memory.py` · `MemoryComponent` | *Retrieve* mode reads the last N messages of the current session as text; *Store* mode writes. Session ID is the thread. | `messages: Annotated[..., add_messages]` + `MemorySaver` + `thread_id` |
| **Prompt Template** | `models_and_agents/prompt.py` · `PromptComponent` | Parses `{variables}` out of the template and creates one input dot per variable. Output is the filled `Message`. | `RAG_PROMPT.format(history=, context=, question=)` |
| **Language Model** | `models_and_agents/language_model.py` · `LanguageModelComponent` | Provider-agnostic chat model (`build_model`). Provider and model are fields, so swapping OpenAI for Gemini needs no rewiring. | `utils.llm = ChatOpenAI("gpt-4o")` |
| **Chat Output** | `input_output/chat_output.py` · `ChatOutput` | Sends the answer to the Playground / API response and stores it in the session history, which is what Message History reads next turn. | `return {"generation": ..., "messages": [...]}` |

Grading and retry, also in the chat flow:

| Node | Module · class | Idea | Ported from |
|---|---|---|---|
| **If-Else** | `flow_controls/conditional_router.py` · `ConditionalRouterComponent` | Compares *Text Input* to *Match Text* (`regex` uses `re.match`) and keeps one of *True* / *False*; the other branch is excluded. Each output sends *Case True* / *Case False*, or the input text when blank. | `decide_to_generate` |
| **Prompt Template** + **Language Model** (grader, rewriter) | as above | Same classes, other prompts: `GRADE_PROMPT` (all chunks at once) and `REWRITE_PROMPT`. | `grade_documents`, `rewrite_question` |

### Not built yet: per-chunk grading and Stage 2

| Node | Module · class | Idea | Ported from |
|---|---|---|---|
| **Structured Output** | `llm_operations/structured_output.py` | LLM call with a declared output schema. | `llm.with_structured_output(GradeDocument)` in `grade_documents` |
| **Loop** | `flow_controls/loop.py` | Emits one *Item* per row, then *Done* with the collected results. | `for doc in documents:` in `grade_documents` |
| **Smart Router** | `llm_operations/llm_conditional_router.py` | LLM picks the route. | Grade + decide merged into one node |
| **Read File** | `files_and_knowledge/file.py` | Loads uploaded files (PDF included). | corrective-rag `tools/index_tool.py` (`PyMuPDFLoader`) |
| **DuckDuckGo Search** | `lfx_duckduckgo/components/duckduckgo/duck_duck_go_search_run.py` (bundle package) | Web search. | corrective-rag `tools/search_tool.py` / `nodes/search_node.py` |
| **Firecrawl Search API** | `lfx_firecrawl/components/firecrawl/firecrawl_search_api.py` (bundle package) | Web search with page content. | Same, alternative to DuckDuckGo |

Design difference: LangGraph shares one `GraphState` dict that every node
reads and writes. Langflow has no shared state. Each value travels along an
edge, which is why Chat Input has three outgoing edges and why Parser exists
at all.

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
| langsmith observability | Langflow's LangSmith / Langfuse integrations (Phoenix: see *Tracing with Phoenix*) |
