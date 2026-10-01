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
// Whether a device implements MUL_MAT / MUL_MAT_ID for each encoding.
char * fx_supports(const char * request_json);

fx_engine * fx_engine_load(const char * params_json, char ** error);
void fx_engine_free(fx_engine * e);
// {"n_ctx","n_ctx_seq","n_seq","n_batch","n_vocab","memory":[...]}
char * fx_engine_info(fx_engine * e);

// Returns the token count, or -(required) when cap is too small.
int32_t fx_tokenize(fx_engine * e, const char * text, int32_t len, bool add_special, int32_t * out, int32_t cap);
// Raw bytes of a token's text (may be a partial UTF-8 sequence); returns length or -(required).
int32_t fx_token_piece(fx_engine * e, int32_t token, bool special, char * buf, int32_t cap);
bool fx_is_eog(fx_engine * e, int32_t token);
// {"messages":[...],"tools":[...]?,"add_generation_prompt":bool} -> {"prompt","preserved_tokens":[ids],"additional_stops":[...]}
char * fx_apply_template(fx_engine * e, const char * request_json);

// One llama_decode over n tokens; returns llama_decode's status (0 = ok).
int32_t fx_decode(fx_engine * e, int32_t n, const int32_t * tokens, const int32_t * pos, const int32_t * seq, const int8_t * logits);
// Time attribution of decode steps per device and op (engine loaded with "trace": true).
// {"prompt":[ids],"steps":N,"seq":0} -> {"plain_step_us":[...],"traced_step_us":[...],"ops":[{"device","op","us","count"}],...}
char * fx_trace(fx_engine * e, const char * request_json);
// Expert selections per MoE layer while prefilling `prompt` and greedily decoding `steps` tokens
// (engine loaded with "trace": true): {"prefill":{layer:[count per expert]},"decode":{...}}.
char * fx_route_stats(fx_engine * e, const char * request_json);
void fx_seq_clear(fx_engine * e, int32_t seq);

// Sampling chain identical to llama-server's for the same settings.
fx_sampler * fx_sampler_new(fx_engine * e, const char * sampling_json);
void fx_sampler_free(fx_sampler * s);
// Feeds a prompt token into penalty history without advancing grammar state.
void fx_sampler_accept_prompt(fx_sampler * s, int32_t token);
// Samples from the logits of batch row `idx` and accepts the result.
int32_t fx_sampler_sample(fx_sampler * s, fx_engine * e, int32_t idx);

#ifdef __cplusplus
}
#endif
