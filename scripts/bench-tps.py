#!/usr/bin/env python3
"""Fixed-work native-worker benchmark. Keeps exact tokens and terminal state as an oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import statistics
import subprocess
import time


class Worker:
    def __init__(self, binary, log):
        self.log = log.open("w")
        self.process = subprocess.Popen([str(binary), "serve"], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        try:
            self.hello = self.receive()
            assert self.hello["ev"] == "hello", self.hello
            self.send(op="hello", protocol=1)
        except Exception:
            self.close()
            raise

    def send(self, **request):
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()

    def receive(self, timeout=300):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError("worker progress deadline")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                raise RuntimeError(f"worker exited: {self.process.poll()}")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        event = json.loads(line)
        if event["ev"] == "error":
            raise RuntimeError(event)
        return event

    def close(self):
        try:
            if self.process.poll() is None:
                self.send(op="shutdown")
                self.process.wait(timeout=15)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            pass
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.selector.close()
            self.process.stdin.close()
            self.process.stdout.close()
            self.log.close()


def plan(args, hello):
    return {
        "schema": 1, "id": "tps-experiment", "created": "2026-10-05T00:00:00Z",
        "key": {"model_identity": "local-benchmark", "topology": "local-benchmark",
                "backend_revision": hello["backend_revision"], "backend_build": hello["backend_build"],
                "driver": "local-benchmark", "ctx_bucket": args.context, "concurrency": args.concurrency,
                "policy": "uncertified-benchmark-candidate"},
        "model_files": [str(args.model)], "architecture": "qwen35", "engine": "native",
        "workload": {"n_ctx_seq": args.context, "concurrency": args.concurrency, "objective": "interactive"},
        "placement": {"devices": ["CUDA0", "CUDA1"], "layer_device": [], "output_device": "CUDA1",
                      "overrides": [], "n_gpu_layers": 999, "tensor_split": [args.split, 1 - args.split]},
        "runtime": {"n_batch": 8192, "n_ubatch": args.ubatch, "n_threads": 12, "n_threads_batch": 12,
                    "flash_attn": True, "type_k": "f16", "type_v": "f16", "mmap": True, "mlock": False,
                    "op_offload": True, "speculation": {"kind": "draft-mtp", "draft_model": None,
                    "n_max": args.drafts, "draft_vocab": None} if args.drafts else None},
        "budgets": [], "host": {"capacity": 0, "available_at_plan": 0, "os_reserve": 0,
                    "resident_weights": 0, "mirrored_weights": 0, "pinned_buffers": 0, "state_and_scratch": 0},
        "quality": {"kind": "exact"}, "decisions": [], "validation": None,
    }


def run(worker, request_id, prompt, generated):
    started = time.monotonic()
    worker.send(op="prefill", req=request_id, prompt=prompt, max_tokens=generated, stop=[],
                render_special=[], checkpoints=[], sampling={"temperature": 0, "seed": 1, "ignore_eos": True})
    worker.send(op="decode", req=request_id, n=generated)
    tokens, text, arrival, engine_times = [], [], [], []
    prefilled = None
    while True:
        event = worker.receive()
        if event.get("req") != request_id:
            continue
        if event["ev"] == "prefilled":
            prefilled = event
        elif event["ev"] == "token":
            tokens.append(event["token"])
            text.append(event["text"])
            arrival.append(time.monotonic() - started)
            engine_times.append(event["t_us"])
        elif event["ev"] == "finished":
            assert event["reason"] == "length", event
            assert len(tokens) == generated == event["n_decoded"], event
            text.append(event["tail"])
            break
    assert prefilled is not None
    assert len(prompt) == prefilled["n_prompt"]
    return {"prompt_tokens": len(prompt), "reused": prefilled["reused"], "prefill_ms": prefilled["ms"],
            "prefill_tps": (len(prompt) - prefilled["reused"]) * 1000 / prefilled["ms"],
            "ttft_s": arrival[0], "decode_tps": (generated - 1) / (arrival[-1] - arrival[0]),
            "engine_decode_tps": (generated - 1) * 1e6 / (engine_times[-1] - engine_times[0]),
            "total_s": time.monotonic() - started, "tokens": tokens,
            "text_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
            "token_times_s": arrival}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--oracle", type=Path, help="Compare tokens with a prior run on the same prompts")
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--drafts", type=int, default=0)
    parser.add_argument("--split", type=float, default=0.67)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=1, help="Reserved sequence slots; requests run sequentially")
    parser.add_argument("--prompts", default="512,2048,4096")
    parser.add_argument("--generated", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    assert args.generated > 1 and args.repeats > 0 and args.concurrency > 0
    assert args.ubatch > 0 and args.drafts >= 0 and 0 < args.split < 1
    sizes = [int(s) for s in args.prompts.split(",")]
    assert min(sizes) > 0 and max(sizes) + args.generated <= args.context
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.corpus is None:
        args.corpus = args.output.parent / "corpus.json"
    result = {"config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runs": [], "status": "running"}
    worker = Worker(args.worker.resolve(), args.output.with_suffix(".log"))
    try:
        result["hello"] = worker.hello
        worker.send(op="load", plan=plan(args, worker.hello), trace=False)
        loaded = worker.receive(timeout=900)
        assert loaded["ev"] == "loaded", loaded
        result["loaded"] = loaded
        print(json.dumps({"loaded": loaded}), flush=True)
        root = Path(__file__).resolve().parents[1]
        if args.corpus.exists():
            corpus = json.loads(args.corpus.read_text())
            assert Path(corpus["model"]).resolve() == args.model.resolve(), "corpus belongs to another model"
            pool = corpus["tokens"]
        else:
            source = "\n".join(p.read_text() for p in [root / "README.md", root / "crates/flux-worker/src/native.rs",
                                                         root / "crates/flux-core/src/worker.rs", root / "crates/flux-plan/src/planner.rs"])
            worker.send(op="tokenize", id=1, text=source, add_special=False)
            pool = worker.receive()["tokens"]
            args.corpus.write_text(json.dumps({"model": str(args.model), "tokens": pool}) + "\n")
        result["corpus_sha256"] = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
        assert len(pool) > max(sizes) + args.repeats + 100
        run(worker, "warmup", pool[-64:], 16)
        previous = None
        for size in sizes:
            for repeat in range(args.repeats):
                offset = repeat * 37
                while pool[offset] == previous:
                    offset += 1
                prompt = pool[offset:offset + size]
                previous = prompt[0]
                trial = run(worker, f"cold-{size}-{repeat}", prompt, args.generated)
                assert trial["reused"] == 0, trial["reused"]
                trial.update(mode="cold", repeat=repeat, prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest())
                result["runs"].append(trial)
                warm = run(worker, f"warm-{size}-{repeat}", prompt, args.generated)
                assert trial["tokens"] == warm["tokens"], "warm prefix changes greedy output"
                warm.update(mode="warm", repeat=repeat, prompt_sha256=trial["prompt_sha256"])
                result["runs"].append(warm)
                print(json.dumps({"prompt": size, "repeat": repeat, "prefill_tps": trial["prefill_tps"],
                                  "decode_tps": trial["decode_tps"], "warm_ttft_s": warm["ttft_s"]}), flush=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
        result["summary"] = [{"prompt": size, "prefill_tps": statistics.median(r["prefill_tps"] for r in result["runs"] if r["mode"] == "cold" and r["prompt_tokens"] == size),
                              "decode_tps": statistics.median(r["decode_tps"] for r in result["runs"] if r["mode"] == "cold" and r["prompt_tokens"] == size)} for size in sizes]
        print(json.dumps({"summary": result["summary"]}), flush=True)
        if args.oracle:
            reference = json.loads(args.oracle.read_text())
            oracle = {(r["prompt_sha256"], r["mode"], r["repeat"]): r for r in reference["runs"]}
            result["oracle"] = []
            for r in result["runs"]:
                gold = oracle.get((r["prompt_sha256"], r["mode"], r["repeat"]))
                if gold is None:
                    raise ValueError("oracle does not cover every benchmark prompt")
                if len(gold["tokens"]) != len(r["tokens"]):
                    raise ValueError("oracle has a different generated-token budget")
                result["oracle"].append({"prompt_tokens": r["prompt_tokens"], "repeat": r["repeat"],
                                         "mode": r["mode"], "exact_tokens": r["tokens"] == gold["tokens"],
                                         "exact_text": r["text_sha256"] == gold["text_sha256"]})
            result["oracle_exact"] = all(r["exact_tokens"] and r["exact_text"] for r in result["oracle"])
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = str(error)
        raise
    finally:
        worker.close()
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
