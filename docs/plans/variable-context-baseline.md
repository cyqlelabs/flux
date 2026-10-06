# Variable-context baseline measurements

Measured 2026-10-05. Plan `7c02c48366687167`, backend `427291b5b34c+10ebedb9a31d74dd29a6425d`, Flash-Next IQ2_XS, 65,536 cells, f16 KV, six CPU decode threads. CUDA0 is the RTX 3060; CUDA1 is the RTX 2060. This baseline has no MTP heads and cannot establish the plan’s 50.4 tok/s target.

The profiler records 512 greedy tokens at each depth. Tables discard each backend’s first post-prefill interval and average the remaining 448 target graphs. CUDA events add overhead: these decode rates are instrumentation rates, not throughput acceptance results. Full tables, tokens and logs are in `/tmp/flux-variable-context/profile-m0/`.

| Depth | Prompt time | Decode under profiling |
|---|---:|---:|
| 19,000 | 58.085 s | 18.805 tok/s |
| 57,000 | 204.965 s | 14.214 tok/s |

| Device / operation | 19K ms/graph | 57K ms/graph |
|---|---:|---:|
| CUDA0 / MOE_HOST_WAIT ffn_moe_out | 3.963 | 4.580 |
| CUDA0 / MUL_MAT_ID ffn_moe_gate | 3.333 | 3.414 |
| CUDA0 / FLASH_ATTN_EXT node_# | 1.540 | 4.310 |
| CUDA0 / MUL_MAT_ID ffn_moe_down | 1.536 | 1.566 |
| CUDA0 / GET_ROWS indexer_gather | 1.105 | 3.109 |
| CUDA0 / CONT indexer_gather | 0.873 | 2.300 |
| CUDA0 / TOP_K indexer_select_cells | 0.744 | 0.787 |
| CUDA0 / MOE_HOST_POST ffn_moe_host_post | 0.309 | 0.315 |
| CUDA0 / ROPE indexer_k_rope | 0.289 | 0.792 |
| CUDA0 / SCALE indexer_k_pooled | 0.198 | 0.582 |
| CUDA0 / SET_ROWS indexer_mask_rows | 0.042 | 0.040 |
| CUDA0 / ADD indexer_mask | 0.042 | 0.041 |
| CUDA0 / CONT indexer_select_cells | 0.032 | 0.032 |
| CUDA1 / MUL_MAT_ID ffn_moe_gate | 0.815 | 0.816 |
| CUDA1 / MOE_HOST_WAIT ffn_moe_out | 0.356 | 0.398 |
| CUDA1 / MUL_MAT_ID ffn_moe_down | 0.320 | 0.320 |
| CUDA1 / FLASH_ATTN_EXT node_# | 0.284 | 0.769 |
| CUDA1 / GET_ROWS indexer_gather | 0.161 | 0.465 |
| CUDA1 / TOP_K indexer_select_cells | 0.154 | 0.147 |
| CUDA1 / MOE_HOST_POST ffn_moe_host_post | 0.110 | 0.107 |
| CUDA1 / CONT indexer_gather | 0.098 | 0.229 |
| CUDA1 / ROPE indexer_k_rope | 0.037 | 0.087 |
| CUDA1 / SCALE indexer_k_pooled | 0.025 | 0.068 |
| CUDA1 / ADD indexer_mask | 0.008 | 0.006 |
| CUDA1 / CONT indexer_select_cells | 0.007 | 0.005 |
| CUDA1 / SET_ROWS indexer_mask_rows | 0.007 | 0.006 |

On CUDA0 at 57K, gather and its contiguous copies take 5.409 ms/graph, pooled scaling 0.582 ms, and pooled-key rotation 0.792 ms. Those named operations give T4 an upper bound of 6.783 ms before accounting for cache updates; normalization and pooling additions also include unnamed fused nodes. Attention takes 4.310 ms. Top-k takes 0.787 ms, so even an ideal fourfold reduction saves only 0.590 ms. These are bounds on device work, not predicted end-to-end speedups.

The draft-layer profile and the heads-enabled throughput comparison remain pending.

The rebuilt 512 MiB host-page probe (four-cell scattered reads, six concurrent CPU GEMV threads) measured:

| GPU | Stream alone | Scattered alone | Stream contended | Scattered contended |
|---|---:|---:|---:|---:|
| RTX 3060 | 22.471 GB/s | 21.170 GB/s | 9.639 GB/s | 8.910 GB/s |
| RTX 2060 | 2.928 GB/s | 2.875 GB/s | 2.928 GB/s | 2.868 GB/s |

Report: `/tmp/flux-variable-context/data/probes/0168a722be8d844e-7dd8036e721e.json`, which did not survive a reboot. Placement must use the contended measurements.

## Pooled keys and block selection (T4, T5)

Each run replays the saved placement with expert residency frozen and decodes 512 greedy tokens. Masked and pooled ran on backend `427291b5b34c+cfd5f7cec5e24cfb9ed7e4cb`. Blocks ran on `427291b5b34c+ddac833564da26703ff851b7`, which adds one fix: block selection had never run on Flash-Next, because M-RoPE marks every ubatch as 2D and the contiguity check rejected them all.

| Depth | Masked | Pooled | Blocks (pooled plus block selection) |
|---|---:|---:|---:|
| 19K | 18.11, 17.72 tok/s | 18.72 tok/s | 18.60, 19.37 tok/s |
| 57K | 13.13, 13.28 tok/s | 15.48 tok/s | 17.68, 17.69 tok/s |

These are decode rates under profiling. Two values are two runs.

CUDA0 at 57K, in ms per graph:

| Operation | Masked | Pooled | Blocks |
|---|---:|---:|---:|
| Other indexer operations (gather, copies, pooling, rotation, masks) | 6.99 | 0.49 | 0.44 |
| Cell expansion of block scores (unnamed `GET_ROWS`) | 7.16 | 7.12 | 0.83 |
| Top-k | 0.80 (cells) | 0.82 (cells) | 0.87 (blocks) |

Greedy output is not reproducible run to run, even on the masked path: two masked runs on one build diverge at token 143 at 57K. Exact-token equality therefore cannot gate T4 or T5. Grouping runs by identical output:

- 19K: masked, its repeat and pooled agree on all 512 tokens. The two block runs leave that output at tokens 147 and 162, and differ from each other.
- 57K: masked, pooled and both block runs agree. The masked repeat leaves them at token 143.

Pooled keys match the masked path wherever it matches itself. Block selection matches it at 57K in both runs and leaves it at 19K in both runs. A designed difference predicts that. Every cell of a block shares its score, and the budget of 2,051 cells ends three cells into a block. CUDA's radix top-k breaks those ties in atomic order, so the masked path keeps an arbitrary three; block selection keeps the lowest three, as the CPU reference does.

Two questions remain open:

- The source of the run-to-run divergence. No Flux CUDA kernel adds float atomics. Repeating masked runs with `FLUX_MOE_HOST_SYNC=1` would test the CPU/GPU expert overlap first.
- T5's kernel target. The block top-k sorts every block score with CUB, so it is no faster than the cell top-k. A radix select over block scores, followed by a sort of the 514 candidates, would meet the target. Block selection pays without it, because it removes the cell expansion.

Raw results: `/tmp/flux-variable-context/{profile-m1-masked-repeat1,claude-masked,claude-masked2,claude-pooled,claude-blocks-fixed,claude-blocks-fixed2}`. These paths do not survive a reboot.

## Paged dry run at 262,144 tokens

`flux-worker measure` on the same placement, one sequence, 65,536-token floor, in MiB:

| Device | KV reserved at load, fixed | KV reserved at load, paged | Compute, fixed | Compute, paged |
|---|---:|---:|---:|---:|
| CUDA0 | 6,433 | 2,209 | 4,147 | 1,651 |
| CUDA1 | 592 | 208 | 3,042 | 930 |
| CPU | 0 | 0 | 801 | 223 |

The paged capacity splits into 768 MiB of full-read KV (the indexer keys, without V) and 6,144 MiB of dense attention KV. The paged load reserves the full-read pages plus the attention floor.

## Paging equivalence and prompt staging (T7, T11)

All runs at 57K replay the same placement with expert residency frozen and `FLUX_MOE_TIER_WAIT=1`, so outputs repeat exactly.

| Configuration | Prefill | Decode under profiling | Greedy output |
|---|---:|---:|---|
| Unpaged | 183.6 s | 18.81 tok/s | reference |
| Paged, VRAM floor at full capacity (65,536) | 183.3 s | 18.64 tok/s | identical to unpaged |
| Paged, 49,152-token floor, KV past it in VRAM | 194.8 s | 18.71 tok/s | identical to the RAM case below |
| Paged, 49,152-token floor, 156 MiB of KV in RAM | 198.2 s | 16.12 tok/s | identical to the VRAM case above |

Paging changes nothing below the floor. Past a floor below the prompt depth, chunks shrink to fit the floor-sized compute reserve, which rounds differently from unpaged chunks and costs 6% of prefill here.

## Indexed attention (T14)

At 57K with KV partly in RAM, decode attention falls from 11.72 to 1.85 ms per graph on CUDA0 and decode rises from 16.73 to 19.60 tok/s under profiling. On prompt chunks the indexed kernel is slower: a 57K prompt took 471 s against 183 s masked, so prompts stay masked.

## Long prompts on the 262K plan

Plan `e9bf576d751564b2` (Flash-Next with MTP heads), replayed on the final backend, 128 greedy tokens after the prompt:

| Prompt | Prefill | Mean rate | Decode under profiling | KV in RAM |
|---|---:|---:|---:|---:|
| 131,000 tokens | 991 s | 132 tok/s | 12.76 tok/s | 1.7 GiB |
| 200,000 tokens | 2,586 s | 77 tok/s | 10.70 tok/s | 3.3 GiB |

Both outputs are coherent and no step failed. The staging trial chose staging both times: 8.6 against 12.0 ms per token at about 100K, and 14.9 against 45.9 ms at about 190K.

After these runs, prompt chunks also borrow each GPU's KV page budget that no page holds yet. The 131,000-token prompt then prefilled in 758 s (173 tok/s, 24% faster): chunks at about 100K grew from 256 to 512 tokens and took 5.9 instead of 8.6 ms per token. Decode at that depth measured 20.4 tok/s.

Before borrowing, prefill past the floor was slow because the chunks shrink: 256 tokens at 131K and 128 at 262K, against 1,024 at 65K. At about 190K, attention and staging take roughly 3 of the 14.9 ms per token; the rest is work each chunk repeats, such as copying the host experts it uses to the GPU, so the rate follows the chunk size. The RTX 2060 limits the chunk: it holds attention layer 3 and has no expert cache to lend, so the RTX 3060's prompt loans never apply.

## Serving checks

`flux serve` on plan `889d2e30867ecbd1` (262,144 tokens), driven through the OpenAI API:

| Check | Result |
|---|---|
| `/v1/models` | `context_length`, `max_model_len` and `meta.n_ctx` report 262,144 |
| 140,000-token completion | HTTP 200 in 935 s; 1.5 GiB of KV in RAM, 2.2 GiB in VRAM |
| Next turn, same conversation plus 500 tokens | HTTP 200 in 10 s: the cached prefix is reused |
| Prompt 100 tokens short of the context, `max_tokens` 4,096 | HTTP 400 `context_length_exceeded`, "This model's maximum context length is 262144 tokens. However, your messages resulted in 262044 tokens." |
| 200,000 tokens with the reserve raised to leave no RAM for new pages | HTTP 400 `context_length_exceeded` in 0.6 s, naming the 118,784 tokens that fit |
| 3 MB request body | Accepted; axum's own 2 MB limit used to refuse it before `serve.max_body_bytes` applied |
