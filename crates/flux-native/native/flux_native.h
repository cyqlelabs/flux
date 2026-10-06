// Flux-owned C ABI over the pinned llama.cpp/ggml. Complex inputs and outputs are JSON
// strings so that Rust never depends on llama.h struct layouts; hot paths use plain C types.
// Every returned char* is malloc'd: release it with fx_free. JSON results carry "error" on failure.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct fx_engine fx_engine;
typedef struct fx_sampler fx_sampler;

void fx_free(char * p);

// {"commit","build_number","compiler","target","system_info","devices":[...],"max_devices","max_overrides"}
char * fx_backend_info(void);

// Backend allocation dry run for a candidate (no weights loaded): per-device model/context/compute bytes.
char * fx_measure(const char * params_json);

// Kernel probes: weight-matrix products of a given encoding and shape on one device.
char * fx_probe_matmul(const char * request_json);
// Copy bandwidth curves (host<->device, device<->device).
char * fx_probe_copy(const char * request_json);
// The same activities alone and concurrently, to expose shared-link and memory-bandwidth contention.
char * fx_probe_contention(const char * request_json);
char * fx_probe_host_pages(const char * request_json);
// Whether a device implements MUL_MAT / MUL_MAT_ID for each encoding.
char * fx_supports(const char * request_json);

fx_engine * fx_engine_load(const char * params_json, char ** error);
void fx_engine_free(fx_engine * e);
// {"n_ctx","n_ctx_seq","n_seq","n_batch","n_vocab","memory":[...]}
char * fx_engine_info(fx_engine * e);
bool fx_seq_reserve(fx_engine * e, int32_t seq, uint32_t cells);
void fx_seq_release(fx_engine * e, int32_t seq);
void fx_host_reserve(fx_engine * e, uint64_t bytes);

// Returns the token count, or -(required) when cap is too small.
int32_t fx_tokenize(fx_engine * e, const char * text, int32_t len, bool add_special, int32_t * out, int32_t cap);
// Raw bytes of a token's text (may be a partial UTF-8 sequence); returns length or -(required).
int32_t fx_token_piece(fx_engine * e, int32_t token, bool special, char * buf, int32_t cap);
bool fx_is_eog(fx_engine * e, int32_t token);
// {"messages":[...],"tools":[...]?,"add_generation_prompt":bool} -> {"prompt","preserved_tokens":[ids],
// "additional_stops":[...],"parser":{...},"checkpoints":[...]}: "parser" is the spec for fx_chat_parser_new and
// "checkpoints" the prompt positions later prompts are likely to share (end of the system prompt, of the last message).
char * fx_apply_template(fx_engine * e, const char * request_json);

// Splits a chat reply into reasoning, content and tool calls as llama-server does, for one request.
typedef struct fx_chat_parser fx_chat_parser;
fx_chat_parser * fx_chat_parser_new(const char * spec_json, char ** error);
void fx_chat_parser_free(fx_chat_parser * p);
// Appends reply text and returns {"deltas":[OpenAI chunk deltas]}; final parses the whole reply as complete
// and adds "message" (the assistant message with reasoning_content and tool_calls).
char * fx_chat_parser_push(fx_chat_parser * p, const char * text, int32_t len, bool final);

// One llama_decode over n tokens; returns llama_decode's status (0 = ok), -100 for a batch over n_batch,
// -101 when the drafter fails to follow it, -102 when llama_decode throws.
int32_t fx_decode(fx_engine * e, int32_t n, const int32_t * tokens, const int32_t * pos, const int32_t * seq, const int8_t * logits);
// Time attribution of decode steps per device and op (engine loaded with "trace": true).
// {"prompt":[ids],"steps":N,"seq":0} -> {"plain_step_us":[...],"traced_step_us":[...],"ops":[{"device","op","us","count"}],...}
char * fx_trace(fx_engine * e, const char * request_json);
// Expert selections per MoE layer while prefilling `prompt` and greedily decoding `steps` tokens
// (engine loaded with "trace": true): {"prefill":{layer:[count per expert]},"decode":{...}}.
char * fx_route_stats(fx_engine * e, const char * request_json);
void fx_seq_clear(fx_engine * e, int32_t seq);
// Prompt reuse. Saves the sequence's recurrent state at its current end (up to four checkpoints per sequence).
bool fx_seq_checkpoint(fx_engine * e, int32_t seq);
// Keeps the sequence's first `keep` positions where its state can be recovered: by trimming (attention-only
// models) or from the latest checkpoint within `keep` (recurrent models), else not at all. Returns how many it kept.
int32_t fx_seq_keep(fx_engine * e, int32_t seq, int32_t keep);
// Conversation cache. Bytes a copy of the sequence's whole state takes: KV, recurrent and drafter state, checkpoints.
uint64_t fx_seq_state_size(fx_engine * e, int32_t seq);
// Copies the sequence's whole state to host memory under `id`, leaving the sequence as it is; returns the bytes
// held, 0 when the copy failed.
uint64_t fx_seq_park(fx_engine * e, int32_t seq, int64_t id);
// Replaces the sequence with a copy of parked state `id`, which stays parked. False leaves the sequence empty.
bool fx_seq_restore(fx_engine * e, int32_t seq, int64_t id);
void fx_park_drop(fx_engine * e, int64_t id);

// Speculation (engines loaded with a draft-mtp plan). Drafts up to n_max tokens after `last`, which sits at
// `pos`, given the n_hist tokens before it; returns how many were written to out.
int32_t fx_spec_draft(fx_engine * e, int32_t seq, int32_t pos, int32_t last, const int32_t * hist, int32_t n_hist, int32_t n_max, int32_t * out);
// After verification: drops the sequence from `pos` on (target and drafter) and tells the drafter how many
// draft tokens the target accepted. False when the target could not roll back.
bool fx_spec_accept(fx_engine * e, int32_t seq, int32_t pos, int32_t n_accepted);

// Sampling chain identical to llama-server's for the same settings.
fx_sampler * fx_sampler_new(fx_engine * e, const char * sampling_json);
void fx_sampler_free(fx_sampler * s);
// Feeds a prompt token into penalty history without advancing grammar state.
void fx_sampler_accept_prompt(fx_sampler * s, int32_t token);
// Samples from the logits of batch row `idx` and accepts the result.
int32_t fx_sampler_sample(fx_sampler * s, fx_engine * e, int32_t idx);
// The most likely token of row idx other than `chosen` (by raw logits), or -1.
int32_t fx_runner_up(fx_engine * e, int32_t idx, int32_t chosen);
// Samples rows row..row+n_draft against the draft, stopping at the first disagreement; returns the accepted
// draft tokens plus the token sampled after them (1..n_draft+1), written to out, or -1 when sampling fails.
int32_t fx_sampler_sample_draft(fx_sampler * s, fx_engine * e, int32_t row, const int32_t * draft, int32_t n_draft, int32_t * out);

#ifdef __cplusplus
}
#endif
