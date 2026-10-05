#!/usr/bin/env python3
"""Protocol fixture: no model, GPU, network, or production service access."""
import json
import sys


def emit(**event):
    event["ev"] = event.pop("event")
    print(json.dumps(event), flush=True)


chat = len(sys.argv) > 1 and sys.argv[1] == "external"
emit(event="hello", protocol=1, worker="audit-fixture", engine="fixture" if chat else "native", backend_revision="audit", level="chat" if chat else "tokens")
for line in sys.stdin:
    request = json.loads(line)
    kind = request["op"]
    if kind == "load":
        emit(event="loaded", load_ms=0, n_ctx_seq=4096, n_seq=2, memory=[])
    elif kind == "tokenize":
        if request["text"] != "HANG":
            emit(event="tokens", id=request["id"], tokens=[1, 2])
    elif kind == "apply_template":
        emit(event="templated", id=request["id"], prompt="fixture", preserved_tokens=[], additional_stops=[])
    elif kind == "prefill":
        emit(event="token", req=request["req"], i=0, token=42, text="hello ", t_us=1)
        emit(event="finished", req=request["req"], reason="length", n_prompt=2, n_decoded=1, tail="</")
    elif kind == "chat":
        emit(event="chat_chunk", req=request["req"], t_us=1, chunk={"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_audit", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}, "finish_reason": None}]})
        emit(event="finished", req=request["req"], reason="eog", n_prompt=2, n_decoded=1, tail="")
    elif kind == "shutdown":
        break
