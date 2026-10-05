#!/usr/bin/env python3
"""Deterministic IPC fixture; no model, network, or GPU."""
import json
import sys


def emit(ev, **fields):
    print(json.dumps(dict(ev=ev, **fields)), flush=True)


chat = len(sys.argv) > 1 and sys.argv[1] == "external"
emit("hello", protocol=1, worker="test", engine="fixture" if chat else "native",
     backend_revision="audit", backend_build="audit", level="chat" if chat else "tokens")
mode = "normal"
for line in sys.stdin:
    r = json.loads(line)
    kind = r["op"]
    if kind == "load":
        mode = r["plan"]["architecture"]
        if mode == "load_error":
            emit("error", code="load_failed", message="injected load failure")
        else:
            emit("loaded", load_ms=0, n_ctx_seq=4096, n_seq=2, memory=[])
    elif kind == "tokenize" and r["text"] != "HANG":
        emit("tokens", id=r["id"], tokens=[1, 2])
    elif kind == "apply_template":
        emit("templated", id=r["id"], prompt="fixture", preserved_tokens=[], additional_stops=[])
    elif kind == "prefill":
        if mode == "malformed":
            print("invalid-json", flush=True)
            continue
        for i in range(80 if mode == "backpressure" else 1):
            emit("token", req=r["req"], i=i, token=42, text="" if mode == "crash" else "hello ", t_us=i+1)
        if mode == "crash":
            sys.exit(0)
        if mode != "stall":
            emit("finished", req=r["req"], reason="length", n_prompt=2, n_decoded=80 if mode == "backpressure" else 1, tail="</")
    elif kind == "chat":
        emit("chat_chunk", req=r["req"], t_us=1, chunk={"choices": [{"index": 0, "delta": {
            "reasoning_content": "thinking", "tool_calls": [{"index": 0, "id": "call_audit", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]})
        emit("finished", req=r["req"], reason="eog", n_prompt=2, n_decoded=1, tail="")
    elif kind == "shutdown":
        break
