#!/usr/bin/env python3
"""Compare completed, fixed-work worker benchmarks without dropping failed trials."""
import argparse
import json
from pathlib import Path
import statistics


def load(path):
    value = json.loads(path.read_text())
    if value.get("status", "complete") != "complete" or not value.get("summary"):
        raise ValueError(f"incomplete benchmark: {path}")
    expected = len(value["config"]["prompts"].split(",")) * value["config"]["repeats"] * 2
    if len(value["runs"]) != expected:
        raise ValueError(f"missing trials: {path}")
    runs = {}
    for run in value["runs"]:
        if len(run["tokens"]) != value["config"]["generated"]:
            raise ValueError(f"different token budget: {path}")
        key = (run["prompt_sha256"], run["mode"], run["repeat"])
        if key in runs:
            raise ValueError(f"duplicate trial: {path}")
        runs[key] = run
    return value, runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline, gold = load(args.baseline)
    candidate, trials = load(args.candidate)
    for key in ["model", "ubatch", "split", "context", "concurrency", "generated", "repeats", "prompts"]:
        if baseline["config"][key] != candidate["config"][key]:
            raise ValueError(f"different fixed-work setting: {key}")
    if gold.keys() != trials.keys():
        raise ValueError("prompt oracle differs")
    pairs = [(gold[key], trials[key]) for key in gold]
    result = {"baseline": str(args.baseline), "candidate": str(args.candidate),
              "exact_token_trials": sum(a["tokens"] == b["tokens"] for a, b in pairs),
              "exact_text_trials": sum(a["text_sha256"] == b["text_sha256"] for a, b in pairs),
              "total_trials": len(pairs), "cells": []}
    for size in sorted({a["prompt_tokens"] for a, _ in pairs}):
        cell = [(a, b) for a, b in pairs if a["prompt_tokens"] == size and a["mode"] == "cold"]
        row = {"prompt_tokens": size, "trials": len(cell)}
        for field in ["prefill_tps", "decode_tps", "ttft_s", "total_s"]:
            before = statistics.median(a[field] for a, _ in cell)
            after = statistics.median(b[field] for _, b in cell)
            row[field] = {"baseline": before, "candidate": after, "ratio": after / before}
        warm = [(a, b) for a, b in pairs if a["prompt_tokens"] == size and a["mode"] == "warm"]
        row["warm_ttft_s"] = {"baseline": statistics.median(a["ttft_s"] for a, _ in warm),
                              "candidate": statistics.median(b["ttft_s"] for _, b in warm)}
        result["cells"].append(row)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
