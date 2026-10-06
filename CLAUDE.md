# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Flux is a measured execution planner and runtime for LLM inference over a pinned, patched llama.cpp. It measures the machine, searches placements of a model's layers and experts across GPUs and host memory, validates the fastest candidates on real prompts, saves the winner as an immutable plan, and serves that plan over an OpenAI-compatible API.

## Build

Follow [Build](README.md#build) in `README.md`. The Rust crates link a llama.cpp build, so `scripts/build-backend.sh` must run before `cargo build`. `crates/flux-native/build.rs` refuses to build unless the submodule's HEAD equals `backend.pin` and `build/bin/libllama.so` exists. `FLUX_LLAMA_DIR` can point at another checkout, but it must be at the same pin.

Format with `cargo fmt` (`rustfmt.toml`: 160 columns).

## Test

[Development](README.md#development) in `README.md` lists the test commands. Every crate needs the submodule checked out, because `crates/flux-ingest/build.rs` reads the supported architectures from the pinned `llama-arch.cpp`. Crates that link `flux-native` also need the backend built.

## Patching llama.cpp

Flux's backend changes live in `patches/llama.cpp/*.patch` as `git diff` output against `backend.pin`. `build-backend.sh` applies each patch unless it is already applied, so the submodule's working tree is expected to be dirty. Never commit inside the submodule.

To change the backend, follow [Changing the llama.cpp backend](README.md#changing-the-llamacpp-backend) in `README.md`: edit `third_party/llama.cpp` in place, rebuild, then regenerate both patches with the commands given there.

`flux-native/build.rs` hashes the patches, the submodule's working tree, the built libraries and the `flux-native` and `flux-worker` sources into `FLUX_BACKEND_BUILD`, which is part of every plan's `ProfileKey` and every probe report. A change to any of them therefore invalidates all saved plans and probes: `flux serve` refuses the old plan, so run `flux plan` again.

## Architecture

**Process boundary.** Only `flux-worker` links `flux-native`, the C++ bridge to llama.cpp, so a native crash never takes down the `flux` process. The bridge exposes a C ABI (`crates/flux-native/native/flux_native.h`) that passes JSON strings for complex values, so Rust never depends on `llama.h` struct layouts. The worker runs in two ways:

- **One-shot jobs.** Planning and probing call `oneshot_job(["measure"])` and `oneshot_job(["probe", kind])`, which run `flux-worker measure` or `flux-worker probe <kind>` with JSON on stdin.
- **Long-lived engine.** Serving and plan validation spawn `flux-worker serve`, which speaks the versioned JSON-lines protocol in `crates/flux-core/src/protocol.rs`. The supervisor side is `crates/flux-core/src/worker.rs`. `flux-worker external` drives llama-server or other OpenAI-compatible engines through the same protocol.

**Pipeline.** Each `flux` command maps to one crate:

| Command | Crate | Role |
|---|---|---|
| `flux inspect` | `flux-ingest` | Builds the GGUF or Hugging Face manifest; file hashes form the model identity |
| `flux probe` | `flux-probe` | Measures copy curves, contention, kernel shapes, CPU and storage bandwidth, and GPU reads of pinned RAM; saves a probe report keyed by hardware topology |
| `flux plan` | `flux-plan` | Plans placement, measures the finalists and saves the winner |
| `flux serve <plan>` | `flux-serve` | Runs an axum OpenAI API in front of a worker; admission control, journal, drift-triggered replanning |
| `flux bench …` | `flux-bench` | Paired trials against baselines, quality (KL divergence), conformance, soak tests |

**Planner** (`crates/flux-plan/src/planner.rs`). It works in stages: probe, prune, enumerate, verify memory, measure finalists, save.

- `layers.rs` turns the manifest into per-block byte and compute accounting.
- `cost.rs` predicts step times from probes. These predictions only rank and prune candidates: the final choice always comes from measured runs.
- `search.rs` places contiguous blocks across devices by dynamic programming.
- `experts.rs` sizes per-GPU expert caches and second-GPU tiers from routing counts traced on prose chat and on the coding-agent conversations in `agent_calibration.json`. Validation also times decoding on those conversations, and serving judges requests that carry tools against that rate.
- The backend's allocation dry run (`fx_measure`) verifies memory before anything loads.
- The context defaults to the model's trained one. The plan's `KvPaging` keeps the first `floor_tokens` (`KV_FLOOR_TOKENS`) of attention KV per sequence in VRAM and budgets pinned-RAM pages past it; the context is capped only when that memory cannot hold it. Interactive plans also time one long prompt on the best candidates. [Long contexts](README.md#long-contexts) in `README.md` covers the serving side.
- `store.rs` saves plans keyed by `ProfileKey`: model hashes, topology, backend revision and build, driver, context bucket, concurrency, and `policy`, a hash of the plan request, the `[plan]` config, `host_reserve_percent`, the registered engines and a planner revision (`PlanRequest::policy_key`).

**Plan to execution.** `crates/flux-core/src/plan.rs` defines the plan, which is never mutated after it is saved. The bridge translates its placement, expert cache, tiers and speculation settings into llama.cpp calls. Some of those calls exist only in the patch, for example `llama_moe_cache_tier`, `llama_set_offload_tuning` and `llama_set_draft_vocab`. `flux serve` loads a copy fitted to the GPU memory free at start (`planning::fit` in `flux-cli`), which drops the least-routed cached experts on a GPU that falls short.

## Configuration and diagnostics

The [Configuration](README.md#configuration) section of `README.md` covers `flux.toml`, the state and log directories, the environment variables, and why throughput is compared only on an idle machine. Defaults live in `crates/flux-core/src/config.rs`.
