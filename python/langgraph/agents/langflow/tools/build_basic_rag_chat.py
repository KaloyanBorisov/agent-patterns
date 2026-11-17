"""Generate flows/basic-rag-chat.json: basic-RAG with grading and one rewrite retry.

Runs inside the Langflow container (needs lfx):
    docker compose exec -T langflow python - < tools/build_basic_rag_chat.py > /tmp/chat.json
    python3 tools/fix_flow_ui.py /tmp/chat.json

basic-RAG/agent.py loops rewrite -> retrieve until a chunk passes grading, with
no limit. Here the loop is unrolled into one retry:

  Chat Input -> Knowledge -> Parser -> grade -> If-Else (NONE?)
     no  (relevant chunks)  -> RAG prompt -> LLM -> Chat Output
     yes (nothing relevant) -> rewrite -> Knowledge -> Parser -> grade -> If-Else (NONE?)
                                 no  -> RAG prompt -> LLM -> Chat Output
                                 yes -> Chat Output ("I don't know ...")
"""
import json
import sys

from lfx.components.files_and_knowledge.knowledge import KnowledgeComponent
from lfx.components.flow_controls.conditional_router import ConditionalRouterComponent
from lfx.components.input_output.chat import ChatInput
from lfx.components.input_output.chat_output import ChatOutput
from lfx.components.models_and_agents.language_model import LanguageModelComponent
from lfx.components.models_and_agents.memory import MemoryComponent
from lfx.components.models_and_agents.prompt import PromptComponent
from lfx.components.processing.parser import ParserComponent
from lfx.graph import Graph

KB = "langgraph-docs"
TOP_K = 4
# The 1.13 editor only lists newer models; a model it doesn't list is swapped
# for the first listed one on open.
MODEL = [{
    "name": "gpt-6.1-terra", "icon": "OpenAI", "provider": "OpenAI",
    "metadata": {"context_length": 128000, "model_class": "ChatOpenAI", "model_name_param": "model",
                 "api_key_param": "api_key", "max_tokens_field_name": "max_tokens",
                 "reasoning": True, "reasoning_models": ["gpt-6.1-terra"]},
}]

# GRADE_PROMPT from basic-RAG/agent.py, changed to grade every chunk in one
# call (agent.py makes one call per chunk) and return the relevant ones.
GRADE_PROMPT = """You are a grader assessing whether retrieved documents are relevant to a user question.

Documents (separated by ---):
{documents}

Question: {question}

If a document contains keywords or semantic meaning related to the question, grade it as relevant.
Return the relevant documents exactly as written, separated by blank lines, with no other text.
If no document is relevant, return exactly: NONE"""

# REWRITE_PROMPT and RAG_PROMPT from basic-RAG/agent.py, unchanged.
REWRITE_PROMPT = """You are rewriting a question to improve retrieval from a vector database.

Look at the question and reason about the underlying semantic intent, then rewrite it to be more specific and retrieval-friendly.

Original question: {question}
Rewritten question:"""

RAG_PROMPT = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Conversation history:
{history}

Question: {question}
Context: {context}
Answer:"""

# If-Else uses re.match, so this matches "NONE", "NONE." or " NONE\n".
NONE_REGEX = r"\W*NONE\W*$"
NO_ANSWER = "I don't know. I couldn't find anything relevant in the LangGraph docs."


def llm(id_, prompt):
    m = LanguageModelComponent(_id=id_)
    m.set(input_value=prompt.build_prompt, model=MODEL, temperature=0)
    return m


def retrieve_and_grade(suffix, query, question):
    kb = KnowledgeComponent(_id=f"Knowledge-{suffix}")
    kb.set(knowledge_base=KB, mode="Retrieve", search_query=query, top_k=TOP_K)
    parser = ParserComponent(_id=f"Parser-{suffix}")
    parser.set(input_data=kb.retrieve_data, mode="Parser", pattern="{content}", sep="\n---\n")
    prompt = PromptComponent(_id=f"Prompt-grade-{suffix}")
    prompt.set(template=GRADE_PROMPT, documents=parser.parse_combined_text, question=question)
    return llm(f"LanguageModel-grade-{suffix}", prompt)


def generate(suffix, context, question, history):
    prompt = PromptComponent(_id=f"Prompt-rag-{suffix}")
    prompt.set(template=RAG_PROMPT, history=history, context=context, question=question)
    out = ChatOutput(_id=f"ChatOutput-{suffix}")
    out.set(input_value=llm(f"LanguageModel-rag-{suffix}", prompt).text_response)
    return out


chat = ChatInput(_id="ChatInput-rag")
memory = MemoryComponent(_id="Memory-history")
memory.set(mode="Retrieve", n_messages=10)
history = memory.retrieve_messages_as_text

# First pass: retrieve_documents + grade_documents.
grader = retrieve_and_grade("retrieve", chat.message_response, chat.message_response)

# decide_to_generate. True = nothing relevant. Case True carries the original
# question to the rewrite branch; False passes the relevant chunks through.
decide = ConditionalRouterComponent(_id="IfElse-relevant")
decide.set(input_text=grader.text_response, operator="regex", match_text=NONE_REGEX,
           true_case_message=chat.message_response)

# Relevant chunks found: generate_response.
out_rag = generate("rag", decide.false_response, chat.message_response, history)

# Nothing relevant: rewrite_question, then one more retrieve + grade, then generate.
rewrite_prompt = PromptComponent(_id="Prompt-rewrite")
rewrite_prompt.set(template=REWRITE_PROMPT, question=decide.true_response)
rewriter = llm("LanguageModel-rewrite", rewrite_prompt)
grader_retry = retrieve_and_grade("retry", rewriter.text_response, rewriter.text_response)

# Still nothing relevant after the retry: answer with a fixed message instead
# of looping (agent.py) or letting the model answer from its own knowledge
# (it ignores "say you don't know" when the context is just NONE).
decide_retry = ConditionalRouterComponent(_id="IfElse-retry")
decide_retry.set(input_text=grader_retry.text_response, operator="regex", match_text=NONE_REGEX,
                 true_case_message=NO_ANSWER)
out_none = ChatOutput(_id="ChatOutput-none")
out_none.set(input_value=decide_retry.true_response)
out_retry = generate("retry", decide_retry.false_response, chat.message_response, history)

# Graph.dump() only walks back from one end node, so dump once per Chat Output
# and merge.
meta = dict(name="basic-RAG chat",
            description="basic-RAG/agent.py: retrieve, grade, and one rewrite retry when nothing is relevant.")
d = Graph(start=chat, end=out_retry).dump(**meta)
edge_key = lambda e: (e["source"], e["target"], e["data"]["targetHandle"]["fieldName"])
for end in (out_rag, out_none):
    other = Graph(start=chat, end=end).dump(**meta)["data"]
    ids = {n["id"] for n in d["data"]["nodes"]}
    seen = {edge_key(e) for e in d["data"]["edges"]}
    d["data"]["nodes"] += [n for n in other["nodes"] if n["id"] not in ids]
    d["data"]["edges"] += [e for e in other["edges"] if edge_key(e) not in seen]

# dump() writes a typed Message value as a serialized Message object, which the
# backend reads back as blank (If-Else then routes the input text, "NONE").
# The editor stores typed text as a plain string, so do the same.
# Case True is an advanced (hidden) field, so it has no handle on the node and
# the editor drops the Chat Input -> Case True edge as invalid on open (the
# backend runs it fine). Show it on both If-Else nodes.
for n in d["data"]["nodes"]:
    if n["id"] in ("IfElse-relevant", "IfElse-retry"):
        n["data"]["node"]["template"]["true_case_message"]["advanced"] = False
    if n["id"] == "IfElse-retry":
        n["data"]["node"]["template"]["true_case_message"]["value"] = NO_ANSWER

# Grid layout, read left to right. Top row: first pass and the direct answer.
# Second row: the retry pass, starting under the answer branch. Bottom: Message
# History (feeds both answer prompts) and the fixed "I don't know" reply.
COL, ROW = 440, 850
GRID = {
    "ChatInput-rag": (0, 0), "Knowledge-retrieve": (1, 0), "Parser-retrieve": (2, 0),
    "Prompt-grade-retrieve": (3, 0), "LanguageModel-grade-retrieve": (4, 0), "IfElse-relevant": (5, 0),
    "Prompt-rag-rag": (6, 0), "LanguageModel-rag-rag": (7, 0), "ChatOutput-rag": (8, 0),
    "Prompt-rewrite": (6, 1), "LanguageModel-rewrite": (7, 1), "Knowledge-retry": (8, 1),
    "Parser-retry": (9, 1), "Prompt-grade-retry": (10, 1), "LanguageModel-grade-retry": (11, 1),
    "IfElse-retry": (12, 1), "Prompt-rag-retry": (13, 1), "LanguageModel-rag-retry": (14, 1),
    "ChatOutput-retry": (15, 1),
    "Memory-history": (5, 1.75), "ChatOutput-none": (13, 1.75),
}
POS = {k: (c * COL, int(r * ROW)) for k, (c, r) in GRID.items()}
for n in d["data"]["nodes"]:
    x, y = POS[n["id"]]
    n["position"] = {"x": x, "y": y}
d["data"]["viewport"] = {"x": 40, "y": 120, "zoom": 0.22}
json.dump(d, sys.stdout)
