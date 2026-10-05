Flux audit remediation — 2026-10-04

The implementation addresses all 12 findings in [the original audit](report.md). The changes are in the serving, worker, planner, and benchmark crates; the pre-existing llama.cpp working-tree patches are separate.

| Finding | Implemented correction | Evidence |
|---|---|---|
| 1. Admission cancellation and closure | Queue reservations decrement on cancellation; admission checks blockers again after acquiring a permit. | Cancellation and queued-closure regression tests. |
| 2. Worker availability and deadlines | Readiness precedes preprocessing; an idle watchdog restarts failed workers. Hello, load, RPC, pipe write, generation progress, slow readers, and one-shot jobs have deadlines. Malformed worker output retires its connection. Monitoring stops with the server. | Lifecycle fixture exercises idle death, RPC stall, decode stall, malformed IPC, and an unread response. |
| 3. Replan lifecycle | A detached task owns the full serialized transaction; drain acquires all permits atomically. Callback cancellation, timeout, panic, and load failures restore service. Independent blocker owners preserve pressure closures. | Concurrent drains/replans, cancelled caller, timed-out callback, failed new-plan load, and closure ownership tests. |
| 4. Output integrity | Generation retries only before any token is committed. A partial reply fails explicitly, including an empty text token representing buffered bytes. Versioned journal events retain tails, parsed deltas, chat chunks, terminal message, usage, and status. | Empty-token crash, tail/journal equality, resume HTTP route, terminal metadata, and existing UTF-8/stop-buffer tests. |
| 5. Plan reuse | Complete request, planner policy, and engine registry fingerprints participate in the key; lookup also checks context capacity, objective, and engine eligibility. | Policy tests change context, objective, storage policy, and validation settings independently. |
| 6. Fail-open certification | Complete execution, held-out validation, finite positive rates, minimum decode TPS, and serving p95 are hard constraints. Failed long-context measurements cannot fall back to short-prompt ranking. | Certification predicate tests cover errors, nonfinite rates, minimum TPS, and SLO boundaries; final selection has explicit failure guards. |
| 7. Scheduler starvation | Native stdin is bounded to 64 commands; the loop handles at most eight commands before inference. Control HTTP traffic has four slots. Decode rows stay within batch capacity and rotate priority. | Source review of bounded loop and batch construction. Sustained GPU load remains a hardware validation task. |
| 8. Journal cost and retention | Replay reads one immutable event by index, without cloning prompt or history. Retention has a byte accounting budget, a 4096-entry cap, periodic expiry, and expiry checks on reads. Running entries are protected from eviction and fail explicitly on exhausted capacity. | Indexed-read, capacity, eviction, idle-expiry, and idempotency tests. |
| 9. External chat correctness | Accumulation preserves tool IDs, fragmented names/arguments, reasoning, finish reason, and usage. All outcomes close the journal. Chat-only measurement uses chat requests and requires per-token metadata. Engine children receive a parent-death signal; completed tasks cannot race registry insertion. | Tool/reasoning accumulation and external-chat lifecycle fixture. Actual external model certification remains unmeasured. |
| 10. Benchmark and soak verification | Fixed-length measurements require terminal completion and exact token/usage counts. Tail chunks do not count as tokens. Soak compares token IDs, text, and finish state; verified partial failures are expected only during injected worker kills. | Completion/truncation assertions and HTTP truncation fixture; model-backed soak remains unmeasured. |
| 11. Drift contamination | Decode rate uses worker timestamps. Automatic drift decisions exclude retries, credit pauses, response backpressure, and plans with more than one sequence. | Source review of measurement eligibility. Multi-sequence drift is conservatively disabled until comparable workload telemetry exists. |
| 12. Backend identity | Fingerprint includes compiled bridge, bridge/worker source, actual backend diff, CMake settings, build script, backend library hashes, and the llama-server executable. Startup verifies mapped library contents and server content; plan load checks worker identity. | Native runtime artifact verification and stale-plan identity rejection tests. |

Verification commands:

```sh
cargo test --release --workspace --offline -- --test-threads=2
cargo clippy --release --workspace --all-targets --offline -- -D warnings
cargo fmt --all -- --check
git diff --check
```

All 111 tests passed after remediation. The sandbox prohibits localhost binds, so the 110 other release tests ran with the HTTP fixture filtered out, and the isolated HTTP test passed in a socket-enabled run. Strict workspace Clippy, formatting, and whitespace checks passed. The release workspace build succeeded, and the rebuilt worker's `info` command verified its backend artifacts and returned the new build identity successfully.

| Test target | Passed |
|---|---:|
| flux-bench (including the isolated HTTP fixture) | 25 |
| flux-cli | 1 |
| flux-core | 13 |
| flux-ingest | 16 |
| flux-native | 3 |
| flux-plan | 26 |
| flux-probe | 14 |
| flux-serve unit tests | 8 |
| flux-serve lifecycle integration | 1 |
| flux-worker | 4 |
| Total | 111 |

Exact test invocations in the restricted environment:

```sh
cargo test --release --workspace --offline -- --test-threads=2 --skip client::tests::real_http_streams_require_complete_output
cargo test --release --offline -p flux-bench client::tests::real_http_streams_require_complete_output -- --exact
```

The second invocation received permission to use localhost sockets. The initial socket permission review timed out; the isolated retry was approved and passed. The audit runner also passed its independent Cargo check.

Patch coverage was checked by reconstructing pristine files from the pinned llama.cpp revision, applying both tracked patches in lexical order, and comparing all 61 affected files byte for byte with the current backend tree. Every file matched. No llama.cpp source or patch was changed during remediation.

Operational changes: regenerate old plans; resume uses output-event indexes, including tails and terminal events; worker loss after partial output produces an explicit failure; generic chat certification requires completion usage and per-token logprobs. Defaults for all deadlines and journal retention are documented in [README.md](../../README.md).

The journal is in memory and survives worker replacement, not HTTP-server replacement. Its budget is application memory accounting rather than an exact allocator/RSS bound. Multi-sequence drift monitoring is disabled conservatively. GPU execution was unavailable to the native test process, so these fixes establish lifecycle and measurement invariants without claiming an absolute TPS gain, CUDA numerical equivalence, or a completed GPU soak. Those release measurements must run on the intended model, device, and workload envelope.
