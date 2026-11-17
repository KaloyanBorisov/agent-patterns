#!/usr/bin/env python3
"""Smoke test for the basic-RAG chat flow, via the Langflow run API.

Mirrors basic-RAG/agent.py's __main__: three follow-up questions on one
session (history must resolve "it" and "that"), plus one off-topic question
that should take the retry path. Prints which Chat Output answered each one.

    python3 tools/smoke_test.py            # from langflow/
    python3 tools/smoke_test.py --flow-id <id>

Reads LANGFLOW_API_KEY and LANGFLOW_HOST_PORT from the environment or
langflow/.env. Create the key in the editor under Settings -> API Keys.
Exits 1 if a question gets no answer or more than one Chat Output answers.
The grader is an LLM, so which branch a question takes can vary between runs;
only the off-topic question's branch is checked, and only loosely.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FLOW_NAME = "basic-RAG chat"
FOLLOW_UPS = [
    "What is LangGraph?",
    "How does it handle persistence and checkpointing?",
    "Can you give an example of how to use that in a multi-agent workflow?",
]
OFF_TOPIC = "How tall is Mount Everest?"
# Chat Output node ids set in build_basic_rag_chat.py.
BRANCH = {
    "ChatOutput-rag": "answered directly",
    "ChatOutput-retry": "answered after rewrite",
    "ChatOutput-none": "fixed 'I don't know' reply",
}


def load_env():
    env = dict(os.environ)
    path = Path(__file__).resolve().parent.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value)
    return env


def call(base, key, method, path, body=None):
    req = urllib.request.Request(
        base + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"x-api-key": key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
    except urllib.error.URLError as e:
        sys.exit(f"Can't reach Langflow at {base} ({e.reason}). Is `docker compose up -d` running?")


def find_flow(base, key):
    flows = call(base, key, "GET", "/api/v1/flows/?get_all=true&remove_example_flows=true")
    if isinstance(flows, dict):
        flows = flows.get("items", [])
    ids = [f["id"] for f in flows if f.get("name") == FLOW_NAME]
    if len(ids) != 1:
        sys.exit(f"Expected one flow named {FLOW_NAME!r}, found {len(ids)}. Pass --flow-id.")
    return ids[0]


def ask(base, key, flow_id, session, question):
    body = {"input_value": question, "input_type": "chat", "output_type": "chat", "session_id": session}
    result = call(base, key, "POST", f"/api/v1/run/{flow_id}", body)
    answers = []
    for out in result["outputs"][0]["outputs"]:
        msg = (out.get("results") or {}).get("message") or {}
        text = msg.get("text") if isinstance(msg, dict) else msg
        if text:
            answers.append((out.get("component_id"), text))
    return answers


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flow-id")
    args = parser.parse_args()

    env = load_env()
    key = env.get("LANGFLOW_API_KEY")
    if not key:
        sys.exit("Set LANGFLOW_API_KEY (env or langflow/.env).")
    base = env.get("LANGFLOW_URL") or f"http://127.0.0.1:{env.get('LANGFLOW_HOST_PORT') or 7860}"
    flow_id = args.flow_id or find_flow(base, key)
    stamp = int(time.time())

    failures = 0
    runs = [(f"smoke-{stamp}-history", q) for q in FOLLOW_UPS] + [(f"smoke-{stamp}-offtopic", OFF_TOPIC)]
    for session, question in runs:
        answers = ask(base, key, flow_id, session, question)
        print(f"Q: {question}")
        if len(answers) != 1:
            failures += 1
            print(f"  FAIL: expected one answer, got {len(answers)}: {answers}\n")
            continue
        node, text = answers[0]
        print(f"  [{BRANCH.get(node, node)}] {text.strip()[:300]}\n")
        if question == OFF_TOPIC and node == "ChatOutput-rag":
            print("  note: off-topic question passed the first grade (the grader is lenient)\n")

    print("PASS" if not failures else f"FAIL ({failures})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
