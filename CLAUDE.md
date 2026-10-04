# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Flux is a measured execution planner and runtime for LLM inference over a pinned, patched llama.cpp. It measures the machine, searches placements of a model's layers and experts across GPUs and host memory, validates the fastest candidates on real prompts, saves the winner as an immutable plan, and serves that plan over an OpenAI-compatible API. `Flux_Inference_Proposal.pdf` is the original specification.

## Build

The Rust crates link a llama.cpp build that must exist first:

1. `git submodule update --init`
2. `scripts/build-backend.sh` builds `third_party/llama.cpp` with CUDA (architectures `75;86`). Set `CUDA_ARCHS` for other GPUs, or `GGML_CUDA=OFF` for a CPU-only build.
3. `cargo build --release -p flux-cli -p flux-worker` builds the `flux` and `flux-worker` binaries.

`crates/flux-native/build.rs` refuses to build unless the submodule's HEAD equals `backend.pin` and `build/bin/libllama.so` exists. `FLUX_LLAMA_DIR` can point at another checkout, but it must be at the same pin. `scripts/package.sh` builds a relocatable tarball in `dist/`.

Format with `cargo fmt` (`rustfmt.toml`: 160 columns).

## Test

- Workspace: `cargo test --release`
- One crate, filtered by test name: `cargo test --release -p flux-plan experts`

Every crate needs the submodule checked out, because `crates/flux-ingest/build.rs` reads the supported architectures from the pinned `llama-arch.cpp`. Crates that link `flux-native` also need the backend built.

## Patching llama.cpp

Flux's backend changes live in `patches/llama.cpp/*.patch`. They are plain `git diff` output against `backend.pin`. `build-backend.sh` applies each patch unless it is already applied, so the submodule's working tree is expected to be dirty. Never commit inside the submodule.

To change the backend, edit `third_party/llama.cpp` in place, rebuild with `scripts/build-backend.sh`, then regenerate both patches:

```sh
git -C third_party/llama.cpp diff -- ggml/src/ggml-cpu/arch-fallback.h ggml/src/ggml-cpu/arch/x86/quants.c \
  > patches/llama.cpp/0002-flux-q2_0-avx2.patch
git -C third_party/llama.cpp diff -- . ':!ggml/src/ggml-cpu/arch-fallback.h' ':!ggml/src/ggml-cpu/arch/x86/quants.c' \
  > patches/llama.cpp/0001-flux-backend-extensions.patch
```

`flux-native/build.rs` hashes the patches into `FLUX_BACKEND_BUILD`, which is part of every plan's `ProfileKey`. Any patch change therefore invalidates all saved plans, so run `flux plan` again (or pass `--replan`).

The default build compiles only the CPU and CUDA backends. Edits to Metal, Vulkan, SYCL and other backends in the patch are not compile-checked here.

## Architecture

**Process boundary.** Only `flux-worker` links `flux-native`, the C++ bridge to llama.cpp, so a native crash never takes down the `flux` process. The bridge exposes a C ABI (`crates/flux-native/native/flux_native.h`) that passes JSON strings for complex values, so Rust never depends on `llama.h` struct layouts. The worker runs in two ways:

- **One-shot jobs.** Planning and probing call `oneshot_job(["measure"])` and `oneshot_job(["probe", kind])`, which run `flux-worker measure` or `flux-worker probe <kind>` with JSON on stdin.
- **Long-lived engine.** Serving and plan validation spawn `flux-worker serve`, which speaks the versioned JSON-lines protocol in `crates/flux-core/src/protocol.rs`. The supervisor side is `crates/flux-core/src/worker.rs`. `flux-worker external` drives llama-server or other OpenAI-compatible engines through the same protocol.

**Pipeline.** Each `flux` command maps to one crate:

| Command | Crate | Role |
|---|---|---|
| `flux inspect` | `flux-ingest` | Builds the GGUF or Hugging Face manifest; file hashes form the model identity |
| `flux probe` | `flux-probe` | Measures copy curves, contention and kernel shapes; saves a probe report keyed by hardware topology |
| `flux plan` | `flux-plan` | Plans placement, measures the finalists and saves the winner |
| `flux serve <plan>` | `flux-serve` | Runs an axum OpenAI API in front of a worker; admission control, journal, drift-triggered replanning |
| `flux bench …` | `flux-bench` | Paired trials against baselines, quality (KL divergence), conformance, soak tests |

**Planner** (`crates/flux-plan/src/planner.rs`). It works in stages: probe, prune, enumerate, verify memory, measure finalists, save.

- `layers.rs` turns the manifest into per-block byte and compute accounting.
- `cost.rs` predicts step times from probes. These predictions only rank and prune candidates: the final choice always comes from measured runs.
- `search.rs` places contiguous blocks across devices by dynamic programming.
- `experts.rs` sizes per-GPU expert caches and second-GPU tiers from routing counts traced on prose chat and on the coding-agent conversations in `agent_calibration.json`. Validation also times decoding on those conversations, and serving judges requests that carry tools against that rate.
- The backend's allocation dry run (`fx_measure`) verifies memory before anything loads.
- `store.rs` saves plans keyed by `ProfileKey`: model hashes, topology, backend revision and build, driver, context bucket and concurrency.

**Plan to execution.** `crates/flux-core/src/plan.rs` defines the plan, which is never mutated after it is saved. The bridge translates its placement, expert cache, tiers and speculation settings into llama.cpp calls. Some of those calls exist only in the patch, for example `llama_moe_cache_tier`, `llama_set_offload_tuning` and `llama_set_draft_vocab`.

## Configuration and diagnostics

The [Configuration](README.md#configuration) section of `README.md` covers `flux.toml`, the state and log directories, the environment variables, and why throughput is compared only on an idle machine. Defaults live in `crates/flux-core/src/config.rs`.
