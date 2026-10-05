#!/usr/bin/env python3
"""Collect greedy decode output and complete CUDA op tables at specified depths."""

import argparse
import copy
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import threading
import time


def read_events(stream, events):
    for line in stream:
        events.put(line)
    events.put(None)


def receive(events, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("profiling exceeded its time budget")
    try:
        line = events.get(timeout=remaining)
    except queue.Empty as error:
        raise TimeoutError("worker did not respond before the deadline") from error
    if line is None:
        raise RuntimeError("worker exited; inspect its log")
    event = json.loads(line)
    if event["ev"] == "error":
        raise RuntimeError(event["message"])
    return event


def send(worker, **request):
    worker.stdin.write(json.dumps(request) + "\n")
    worker.stdin.flush()


def until(events, deadline, kind):
    while True:
        event = receive(events, deadline)
        if event["ev"] == kind:
            return event


def op_table(log):
    header = re.compile(r"cuda op profile: (\S+) backend (\d+): graphs (\d+) wall ([\d.]+) ms/graph busy ([\d.]+)")
    row = re.compile(r"cuda op profile:\s+([\d.]+) ms/graph\s+([\d.]+) calls/graph\s+([\d.]+) us/call\s+(.+)")
    decode = False
    seen = set()
    batches = {}
    current = None
    for line in log.splitlines():
        if line == "flux profile phase: decode":
            decode = True
        match = header.search(line)
        if match:
            device, backend, graphs, wall, busy = match.groups()
            key = (device, backend)
            # Discard the first interval after prefill; it can contain prompt graphs.
            eligible = decode and key in seen
            if decode:
                seen.add(key)
            current = {"graphs": int(graphs), "wall_ms": float(wall), "busy_ms": float(busy), "ops": []} if eligible else None
            if current is not None:
                batches.setdefault(key, []).append(current)
        elif current is not None and (match := row.search(line)):
            milliseconds, calls, micros, name = match.groups()
            current["ops"].append({"name": name, "ms_per_graph": float(milliseconds), "calls_per_graph": float(calls), "us_per_call": float(micros)})
    return [{"device": device, "backend": int(backend), "intervals": intervals} for (device, backend), intervals in sorted(batches.items())]


def profile(args, plan, text, depth, deadline):
    log_path = args.output / f"{depth}.log"
    with log_path.open("w") as log:
        environment = dict(os.environ, FLUX_CUDA_OP_PROFILE="2")
        if args.freeze_cache:
            # the second-GPU expert tier otherwise reruns late jobs on the CPU, which rounds differently
            environment["FLUX_MOE_TIER_WAIT"] = "1"
        worker = subprocess.Popen([str(args.worker.resolve()), "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=environment)
        events = queue.Queue()
        reader = threading.Thread(target=read_events, args=(worker.stdout, events), daemon=True)
        reader.start()
        try:
            hello = until(events, deadline, "hello")
            if hello["engine"] != "native":
                raise RuntimeError("profiling requires the native worker")
            if hello["backend_build"] != plan["key"]["backend_build"] and not args.replay_placement:
                raise RuntimeError("plan is stale; replan with this backend first")
            send(worker, op="load", plan=plan)
            loaded = until(events, deadline, "loaded")
            if depth + args.tokens > loaded["n_ctx_seq"]:
                raise ValueError(f"{depth} prompt tokens plus {args.tokens} reply tokens exceed loaded context")
            send(worker, op="tokenize", id=1, text=text, add_special=True)
            prompt = until(events, deadline, "tokens")["tokens"]
            if len(prompt) < depth:
                raise ValueError(f"prompt file has {len(prompt)} tokens; {depth} required")
            prompt = prompt[:depth]
            send(worker, op="prefill", req="profile", prompt=prompt, sampling={"temperature": 0, "seed": 0, "ignore_eos": True}, stop=[], max_tokens=args.tokens)
            prefilled = until(events, deadline, "prefilled")
            log.write("flux profile phase: decode\n")
            log.flush()
            send(worker, op="decode", req="profile", n=args.tokens)
            emitted = []
            while True:
                event = receive(events, deadline)
                if event["ev"] == "token":
                    emitted.append({"token": event["token"], "t_us": event["t_us"], "text": event.get("text", "")})
                elif event["ev"] == "finished":
                    if event["reason"] != "length" or event["n_decoded"] != args.tokens:
                        raise RuntimeError(f"incomplete decode: {event}")
                    break
            send(worker, op="stats", id=2)
            worker_stats = until(events, deadline, "stats")["stats"]
            send(worker, op="shutdown")
            worker.wait(timeout=max(0.1, min(10, deadline - time.monotonic())))
            if worker.returncode:
                raise RuntimeError(f"worker exited with status {worker.returncode}")
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()
            worker.stdin.close()
            reader.join(timeout=1)
            worker.stdout.close()
    duration = (emitted[-1]["t_us"] - emitted[0]["t_us"]) / 1e6 if len(emitted) > 1 else 0
    table = op_table(log_path.read_text())
    if not table:
        raise RuntimeError(f"no complete decode-only profile interval in {log_path}; increase --tokens")
    return {"depth": depth, "backend_build": hello["backend_build"], "plan_id": plan["id"], "prefill": prefilled,
            "source_plan_build": plan["key"]["backend_build"], "placement_replay": args.replay_placement,
            "experiment": args.experiment,
            "expert_cache_frozen": args.freeze_cache,
            "memory": worker_stats["memory"], "kv_pages": worker_stats.get("kv_pages"),
            "decode_tps_under_profile": (len(emitted) - 1) / duration if duration > 0 else None,
            "tokens": emitted, "cuda": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--worker", type=Path, default=Path("target/release/flux-worker"))
    parser.add_argument("--depths", type=int, nargs="+", default=[19000, 57000])
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=int, default=1790)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-placement", action="store_true", help="Replay an older placement as an uncertified backend experiment; never rewrites or certifies the saved plan")
    parser.add_argument("--experiment", choices=["masked", "pooled", "blocks", "indexed", "all"], help="Override QSA runtime flags for an A/B experiment")
    parser.add_argument("--freeze-cache", action="store_true", help="Keep expert residency fixed for greedy-output comparisons")
    args = parser.parse_args()
    if args.tokens < 128 or any(depth <= 0 for depth in args.depths) or not 0 < args.timeout_s <= 7200:
        parser.error("depths must be positive, tokens >= 128, and total timeout <= 7200 seconds")
    plan = json.loads(args.plan.read_text())
    if plan["engine"] != "native":
        parser.error("the plan must use the native engine")
    if args.experiment:
        plan = copy.deepcopy(plan)
        plan["runtime"].update(qsa_pooled=args.experiment in {"pooled", "blocks", "all"},
                               qsa_blocks=args.experiment in {"blocks", "all"},
                               qsa_indexed=args.experiment in {"indexed", "all"})
    if args.freeze_cache and plan.get("expert_cache"):
        plan = copy.deepcopy(plan)
        plan["expert_cache"]["frozen"] = True
    if args.output.exists():
        parser.error("output directory already exists; choose a new directory to preserve earlier measurements")
    text = args.prompt_file.read_text()
    args.output.mkdir(parents=True)
    deadline = time.monotonic() + args.timeout_s
    try:
        for depth in args.depths:
            report = profile(args, plan, text, depth, deadline)
            path = args.output / f"{depth}.json"
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{path}: greedy tokens, prompt time and complete CUDA op intervals", flush=True)
    except (RuntimeError, TimeoutError, ValueError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
