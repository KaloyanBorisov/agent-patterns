#!/usr/bin/env python3
"""Add the editor-only fields that lfx Graph.dump() omits (node type, handles, edge ids)."""
import json, sys

def handle(d):
    return json.dumps(d, separators=(",", ":")).replace('"', "œ")

for path in sys.argv[1:]:
    flow = json.load(open(path))
    g = flow["data"]
    for n in g["nodes"]:
        node = n["data"]["node"]
        n["type"] = "genericNode"
        n["data"].setdefault("display_name", node.get("display_name", n["data"]["type"]))
        n["data"].setdefault("description", node.get("description", ""))
        outs = node.get("outputs") or []
        if outs:
            n["data"].setdefault("selected_output", outs[0]["name"])
        n.setdefault("selected", False)
        n.setdefault("dragging", False)
    for e in g["edges"]:
        sh, th = e["data"]["sourceHandle"], e["data"]["targetHandle"]
        e["sourceHandle"], e["targetHandle"] = handle(sh), handle(th)
        e["id"] = f"reactflow__edge-{e['source']}{e['sourceHandle']}-{e['target']}{e['targetHandle']}"
        e.setdefault("animated", False)
        e.setdefault("className", "")
        e.setdefault("selected", False)
        # the edge's output must be the node's selected output, or the editor hides it
        for n in g["nodes"]:
            if n["id"] == e["source"]:
                n["data"]["selected_output"] = sh["name"]
    json.dump(flow, open(path, "w"), indent=2)
    print(path, len(g["nodes"]), "nodes", len(g["edges"]), "edges fixed")

# Knowledge nodes: show the fields for the selected mode tab (dump() keeps the Ingest defaults).
INGEST = ("input_df", "column_config", "chunk_size", "allow_duplicates", "metadata_json")
RETRIEVE = ("search_query", "top_k", "include_metadata", "include_embeddings", "metadata_filter")
LAYOUT = {}  # positions come from the build scripts
for path in sys.argv[1:]:
    flow = json.load(open(path))
    for n in flow["data"]["nodes"]:
        if n["data"]["type"] in LAYOUT:
            x, y = LAYOUT[n["data"]["type"]]; n["position"] = {"x": x, "y": y}
        if n["data"]["type"] != "Knowledge":
            continue
        t = n["data"]["node"]["template"]
        retrieve = t["mode"]["value"] == "Retrieve"
        for k in INGEST:
            if k in t: t[k]["show"] = not retrieve
        for k in RETRIEVE:
            if k in t: t[k]["show"] = retrieve
        n["data"]["selected_output"] = "retrieve_data" if retrieve else "dataframe_output"
    json.dump(flow, open(path, "w"), indent=2)

# Collapse nodes holding long list inputs (e.g. URL with dozens of URLs) so the
# canvas stays readable; the user can expand them in the editor.
LONG_LIST = 10
for path in sys.argv[1:]:
    flow = json.load(open(path))
    for n in flow["data"]["nodes"]:
        t = n["data"]["node"]["template"]
        if any(isinstance(f, dict) and isinstance(f.get("value"), list) and f.get("list")
               and len(f["value"]) > LONG_LIST for f in t.values()):
            n["data"]["showNode"] = False
    json.dump(flow, open(path, "w"), indent=2)
