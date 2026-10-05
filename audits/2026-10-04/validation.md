Validation record — Flux audit, 2026-10-04

Historical results for the original revision. See [remediation.md](remediation.md) for verification after fixes.

Revision: `f18ac23e9636bd5a774587e13415932a9b02b8c9` with the pre-existing dirty llama.cpp submodule. Production source was not edited.

`cargo test --release --workspace -- --test-threads=2` exited 0:

| Target | Passed |
|---|---:|
| flux-bench | 23 |
| flux-cli | 1 |
| flux-core | 13 |
| flux-ingest | 16 |
| flux-native | 3 |
| flux-plan | 24 |
| flux-probe | 14 |
| flux-serve | 3 |
| flux-worker | 4 |
| Total | 101 |

All doc-test targets had zero tests. The native suite printed `ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected`; its successful CPU tests do not certify CUDA execution.

The final reproduction command exited 0 and printed:

```text
CONFIRMED: dropped admission future leaks waiting count and rejects future queueing
CONFIRMED: request already queued is admitted after memory-pressure closure
CONFIRMED: two concurrent drains each hold one permit and deadlock
CONFIRMED: rebuilding TextStream after token replay loses pending stop prefix
CONFIRMED: rebuilding TextStream after token replay corrupts split UTF-8
CONFIRMED: journal TTL does not evict until a subsequent begin()
MEASURED: journal replay clones n=1000, elapsed_ms=22.1
MEASURED: journal replay clones n=2000, elapsed_ms=88.1
MEASURED: journal replay clones n=4000, elapsed_ms=523.6
CONFIRMED: a 4096-context cache lookup returns a 3072-context plan
CONFIRMED: unresponsive worker RPC requires a caller-supplied timeout
CONFIRMED: completed HTTP reply includes tail that the resume journal omits
CONFIRMED: idle worker death leaves repeated chat failures, generation=0, admission open
CONFIRMED: cancelled replan leaves admission closed and the old worker stopped
CONFIRMED: external chat loses nonstream tool calls and leaves journal Running
```

The reproduction imports production admission, journal, service handlers, supervisor, worker, and plan-store code. It includes the production text-stream source directly. Its mock worker replaces inference so the scenarios need no GPU or model. The clone measurements isolate journal copying and allocation; they are not TPS benchmarks.

The audited and reproduction lockfiles resolve the same versions of the principal async/API dependencies: Tokio 1.53.1, Axum 0.8.9, futures 0.3.34, and serde_json 1.0.151.

Findings concerning validation fallback, SLO selection, control-queue starvation, incomplete benchmark completion checks, drift contamination, and build identity were established by source/control-flow analysis rather than full model-backed reproductions. Full GPU throughput, numerical quality, CUDA memory safety, and soak testing remain unverified.
