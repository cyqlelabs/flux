# Measured native-worker performance

The rollback reservation fix makes an 8K-context MTP configuration fit on this GPU pair. That configuration measured **69–95% higher generation TPS** than non-speculative decoding. Cold prefill was **6–11% slower**. These are local measurements, not a guarantee for other models, hardware, prompts, or traffic.

## Conditions and method

- Model: `Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ4_XS.gguf`, 15,705,860,224 bytes; `qwen35`, 64 trunk layers and one next-token head.
- Hardware: RTX 3060 12 GB + RTX 2060 6 GB, Ryzen 9 5900X. CUDA layer split `0.67,0.33`, 12 CPU threads, f16 KV, flash attention enabled.
- Runtime: context 8,192, one sequence, batch 8,192, micro-batch 128. Greedy sampling, seed 1, EOS ignored, exactly 96 emitted tokens per request.
- Three cold/warm pairs at each prompt length. `corpus.json` freezes token IDs from public repository source, so editing source cannot change the workload. These are raw source continuations; chat templates and tool calls were not measured. Request completion, finish reason, emitted count, prompt count and cold reuse count are checked. Every cold/warm pair produced identical token IDs and text.
- Generation TPS is `(96 - 1) / (last token time - first token time)`, measured at the IPC client. Prefill TPS uses uncached prompt tokens divided by the worker's elapsed prefill time, including checkpoint work. Total request time includes prefill, generation and completion.
- The initial non-speculative control ran before the candidate; the matched control ran after it. Overlapping control prompts produced identical outputs in all 12 trials. Models ran sequentially on the GPUs. These few repetitions establish observations, not a statistical guarantee or a mixed-traffic serving SLO.

The baseline is commit `1950b091794eb2e6a9634bfe31df75a5e54ef387`, worker build `427291b5b34c+5292b81c5bf2dac1fa7d2589`. The candidate build is `427291b5b34c+dc0634452f440b191b467c9e`. Both use llama.cpp pin `427291b5b34cd914a31b3fd3b61a68f6184f4b9f` with the same patch stack. These candidates are explicitly uncertified experiments; no production plan was saved.

## Matched 8K results

Medians from [comparison-ctx8k.json](comparison-ctx8k.json):

| Prompt tokens | Plain generation TPS | MTP generation TPS | Generation gain | Plain prefill TPS | MTP prefill TPS | Total time, plain → MTP |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 17.85 | 32.09 | +79.7% | 525.56 | 467.30 | 6.36 → 4.06 s |
| 2,048 | 17.74 | 34.56 | +94.8% | 466.53 | 436.79 | 9.81 → 7.41 s |
| 4,096 | 17.78 | 30.10 | +69.3% | 456.49 | 431.09 | 14.38 → 12.67 s |

The complete-request time reductions are 36.1%, 24.4% and 11.9%. Longer outputs can amortize the extra prompt work; short outputs and prefill-heavy traffic may favor non-speculative decoding. Warm-prefix TTFT was approximately 90–94 ms. Existing prefix reuse reduced the 4K control's TTFT from 9.04 s to 0.092 s; this is avoided repeated work, not a new cold-prefill kernel speedup.

**Output limits:** speculative verification still uses the target model and full vocabulary. Nevertheless, plain versus MTP token IDs and text hashes matched in only **10 of 18** trials (five of nine distinct prompts). Batch shapes change numerical execution, and this run does not isolate the cause of every divergence or establish a quality score. Do not interpret the TPS result as a bitwise-output or quality certification. Cold/warm reuse matched within each configuration. The before/after reservation regression at 2K matched **all eight** token streams and text hashes; see [comparison-mtp-ctx2k.json](comparison-mtp-ctx2k.json).

## Implemented change and memory evidence

Previously a native `n_max=2` plan reserved five rollback slots because context copying could draft five tokens. Each extra slot retains a full recurrent state per sequence. Both copying and head drafting now obey the plan's maximum, and the output-row reservation follows that same limit. Larger limits remain available for workloads that benefit from longer copies.

The original worker failed to load MTP at 8K with splits 0.67 and 0.72; the captured error/log files retain those failures. The revised worker loaded and completed the 8K trials. A separate cold/warm boundary check completed an 8,096-token prompt plus all 96 output tokens, with identical cold/warm output, at 421.11 prefill TPS and 34.67 generation TPS; see [bounded-mtp-2-full-context.json](bounded-mtp-2-full-context.json). At 2K, the original target alone reported 1,075,576,832 bytes of context state, while the revised target **plus drafter** reported 613,285,888 bytes: at least **441 MiB less**, despite the more complete accounting. The live engine memory report now includes draft state, scratch and staging without double-counting shared weights, using the same aggregation as dry-run planning.

The 2K before/after generation medians differed by about 2.5–2.7%; two repetitions are insufficient to claim a separate kernel speedup. The demonstrated implementation benefit is lower memory use and enabling the faster MTP configuration at a larger context.

Increasing micro-batch 128 → 512 alone was rejected as a general optimization: the initial two-repeat medians changed by −11.1%, +0.4% and +2.8% for 512/2K/4K prefill, with output differences on five of six prompts. Flux already grows supported prompt chunks adaptively. No cold-prefill TPS increase has been established here.

## Remaining opportunities

| Opportunity from implementation review | Evidence | Required experiment |
|---|---|---|
| Batch drafting across active sequences | `NativeWorker::step` invokes `fx_spec_draft` per sequence; the speculative manager supports a vector of sequences. | Concurrent model runs, accepted tokens/round, aggregate TPS, output conformance and rollback boundaries. The single-sequence results cannot establish a gain. |
| Give mixed prefill/decode a time budget | The native step fills spare batch capacity with one unfinished prompt. | Long and short prompt arrivals together; p95/p99 TTFT and inter-token latency under a serving SLO. A smaller chunk may reduce raw TPS while improving usable throughput. |
| Tune draft length and vocabulary per workload | The 2K profile observed roughly 10 ms drafting and 66–69 ms verification per round, with about 2.2–2.3 tokens emitted per round. | Compare draft lengths and optional draft-vocabulary limits on held-out prompts, with quality checks. No automatic vocabulary restriction was added. |
| Reduce host checkpoint/history copies | Prompt checkpoints retain about 149.6 MiB of recurrent state; drafting copies token history each round. | Attribute CPU, transfer and allocation time before changing storage. GPU checkpoints consume scarce VRAM, and host allocation savings may be negligible relative to target execution. |
| Extend the workload matrix | Measurements use one dense model, one concurrent sequence and one quantization. | MoE models, larger contexts, concurrency, mixed arrivals, perplexity/task quality and the existing conformance/soak suites. |

## Reproduce and validate

Keep an original release worker before rebuilding the changed bridge. Set `MODEL` to the tested local GGUF and run from the repository root on an idle machine. The saved corpus records its model path; reuse it with that same model path.

```bash
python3 scripts/bench-tps.py --worker /tmp/flux-tps-baseline-worker --model "$MODEL" \
  --output benchmarks/tps/2026-10-05/plain-ctx8k-control.json \
  --ubatch 128 --split 0.67 --context 8192 --prompts 512,2048,4096 --repeats 3 --generated 96
python3 scripts/bench-tps.py --worker target/release/flux-worker --model "$MODEL" \
  --output benchmarks/tps/2026-10-05/bounded-mtp-2-ctx8k.json \
  --ubatch 128 --drafts 2 --split 0.67 --context 8192 --prompts 512,2048,4096 --repeats 3 --generated 96
python3 scripts/compare-tps.py \
  benchmarks/tps/2026-10-05/plain-ctx8k-control.json \
  benchmarks/tps/2026-10-05/bounded-mtp-2-ctx8k.json
```

The comparison tool rejects incomplete runs, missing/duplicate trials, different prompts, generated-token budgets or fixed runtime settings. `--concurrency` reserves slots; the harness sends sequential requests and does not measure concurrent throughput. `--oracle` records exact token/text comparisons, independently of completion and timing checks.

Validation: 86 fixed-work requests completed across the GPU experiments, including the full-context boundary pair. All 111 workspace tests passed (the localhost HTTP fixture ran outside the socket-restricted sandbox), strict release Clippy, formatting, release worker build and diff checks passed. Existing patches reconstruct all 61 modified llama.cpp files byte-for-byte. This change edits Flux's tracked bridge, not llama.cpp source.
