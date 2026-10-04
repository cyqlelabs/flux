#include "flux_native.h"

#include "build-info.h"
#include "chat.h"
#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama-ext.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <strings.h>
#include <thread>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

char * dup(const std::string & s) {
    char * p = static_cast<char *>(malloc(s.size() + 1));
    memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

char * err_json(const std::string & msg) {
    return dup(json{{"error", msg}}.dump());
}

void log_cb(ggml_log_level level, const char * text, void *) {
    static const int min_level = [] {
        const char * v = getenv("FLUX_NATIVE_LOG");
        if (!v) {
            return (int) GGML_LOG_LEVEL_WARN;
        }
        for (const auto & [name, level] : {std::pair{"debug", GGML_LOG_LEVEL_DEBUG}, std::pair{"info", GGML_LOG_LEVEL_INFO},
                                           std::pair{"warn", GGML_LOG_LEVEL_WARN}, std::pair{"error", GGML_LOG_LEVEL_ERROR}}) {
            if (strcasecmp(v, name) == 0) {
                return (int) level;
            }
        }
        return atoi(v);
    }();
    if (level >= min_level || level == GGML_LOG_LEVEL_CONT) {
        fputs(text, stderr);
    }
}

void ensure_init() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(log_cb, nullptr);
        llama_backend_init();
    });
}

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The batch size from which the backend offloads host-weight ops to a GPU (GGML_OP_OFFLOAD_MIN_BATCH, default 32).
int32_t op_offload_min_batch() {
    static const int32_t n = [] {
        const char * v = getenv("GGML_OP_OFFLOAD_MIN_BATCH");
        return v ? std::max(atoi(v), 1) : 32;
    }();
    return n;
}

ggml_backend_dev_t dev_by_name(const std::string & name) {
    if (name == "CPU") {
        return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    ggml_backend_dev_t d = ggml_backend_dev_by_name(name.c_str());
    if (!d) {
        throw std::runtime_error("unknown backend device " + name);
    }
    return d;
}

const char * kind_name(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU: return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return "igpu";
        default: return "accel";
    }
}

json devices_json() {
    json out = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        ggml_backend_dev_props p;
        ggml_backend_dev_get_props(d, &p);
        out.push_back({
            {"name", p.name},
            {"description", p.description ? p.description : ""},
            {"kind", kind_name(p.type)},
            {"mem_free", p.memory_free},
            {"mem_total", p.memory_total},
            {"pci_bus_id", p.device_id ? json(p.device_id) : json(nullptr)},
        });
    }
    return out;
}

ggml_type type_by_name(const std::string & name) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        if (ggml_blck_size((ggml_type) t) > 0 && strcasecmp(ggml_type_name((ggml_type) t), name.c_str()) == 0) {
            return (ggml_type) t;
        }
    }
    throw std::runtime_error("unknown ggml type " + name);
}

// Model and context parameters from a Flux plan, with the storage their pointers refer to.
struct load_params {
    std::string model;
    llama_model_params mp;
    llama_context_params cp;
    std::vector<ggml_backend_dev_t> devs;
    std::vector<float> split;
    std::vector<std::string> patterns;
    std::vector<llama_model_tensor_buft_override> overrides;

    explicit load_params(const json & j) {
        model = j.at("model").get<std::string>();
        mp = llama_model_default_params();
        cp = llama_context_default_params();

        for (const auto & n : j.value("devices", json::array())) {
            devs.push_back(dev_by_name(n.get<std::string>()));
        }
        devs.push_back(nullptr);
        mp.devices = devs.data();
        const std::string sm = j.value("split_mode", std::string("layer"));
        mp.split_mode = sm == "row" ? LLAMA_SPLIT_MODE_ROW : sm == "tensor" ? LLAMA_SPLIT_MODE_TENSOR : LLAMA_SPLIT_MODE_LAYER;
        mp.main_gpu = 0;
        mp.n_gpu_layers = j.value("n_gpu_layers", 0);
        split.assign(llama_max_devices(), 0.0f);
        const json ts = j.value("tensor_split", json::array());
        for (size_t i = 0; i < ts.size() && i < split.size(); i++) {
            split[i] = ts[i].get<float>();
        }
        mp.tensor_split = split.data();

        const json ov = j.value("overrides", json::array());
        patterns.reserve(ov.size());
        for (const auto & o : ov) {
            patterns.push_back(o.at("pattern").get<std::string>());
            const std::string dev = o.at("device").get<std::string>();
            ggml_backend_buffer_type_t buft = dev == "CPU" ? ggml_backend_cpu_buffer_type() : ggml_backend_dev_buffer_type(dev_by_name(dev));
            overrides.push_back({patterns.back().c_str(), buft});
            host_overrides |= dev == "CPU";
        }
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();

        const bool mmap = j.value("mmap", true), mlock = j.value("mlock", false);
        mp.load_mode = mmap ? (mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MMAP) : (mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE);

        const uint32_t n_seq = j.value("n_seq", 1u);
        cp.n_ctx = j.value("n_ctx_seq", 4096u) * n_seq;
        cp.n_seq_max = n_seq;
        cp.n_batch = j.value("n_batch", 2048u);
        cp.n_ubatch = j.value("n_ubatch", 512u);
        cp.n_threads = j.value("n_threads", (int) std::thread::hardware_concurrency() / 2);
        cp.n_threads_batch = j.value("n_threads_batch", (int) std::thread::hardware_concurrency());
        cp.flash_attn_type = j.value("flash_attn", true) ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.type_k = type_by_name(j.value("type_k", std::string("f16")));
        cp.type_v = type_by_name(j.value("type_v", std::string("f16")));
        cp.offload_kqv = true;
        cp.op_offload = j.value("op_offload", true);
        cp.kv_unified = j.value("kv_unified", false);
        cp.no_perf = false;

        // Expert cache: per layer, the initial cached experts (the first `hot` of `order`).
        const json ec = j.value("expert_cache", json());
        if (ec.is_object()) {
            for (const auto & l : ec.at("layers")) {
                const int32_t hot = l.at(2).get<int32_t>();
                cache_layers.push_back(l.at(0).get<int32_t>());
                cache_slots.push_back(hot);
                cache_offsets.push_back((int32_t) cache_experts.size());
                for (int32_t k = 0; k < hot; k++) {
                    cache_experts.push_back(l.at(1).at(k).get<int32_t>());
                }
            }
            cache_offsets.push_back((int32_t) cache_experts.size());
            // Tiers: experts another GPU serves for cached layers placed elsewhere.
            for (const auto & t : ec.value("tiers", json::array())) {
                tier_spec & ts = cache_tiers.emplace_back();
                ts.device = t.at("device").get<std::string>();
                for (const auto & l : t.at("layers")) {
                    ts.layers.push_back(l.at(0).get<int32_t>());
                    ts.offsets.push_back((int32_t) ts.experts.size());
                    for (const auto & x : l.at(1)) {
                        ts.experts.push_back(x.get<int32_t>());
                    }
                }
                ts.offsets.push_back((int32_t) ts.experts.size());
            }
        }
        cache_frozen = j.value("expert_cache_frozen", false);

        // Speculation drafts with the model's own next-token heads; the target keeps one recurrent-state
        // snapshot per draft token so a partly rejected draft rolls back without re-decoding.
        const json sp = j.value("speculation", json());
        if (sp.is_object()) {
            if (sp.value("kind", std::string()) != "draft-mtp") {
                throw std::runtime_error("the native engine drafts only with the model's next-token heads (draft-mtp)");
            }
            spec_n_max = sp.value("n_max", 3);
            spec_draft_vocab = sp.value("draft_vocab", 0);
            mp.load_mtp = true;
            cp.n_rs_seq = (uint32_t) spec_window();
        }
        // A batch requests logits only for each decoding sequence's verification rows and a prompt's last
        // token; reserving vocabulary-wide rows for every token of a prompt chunk would cost a GPU-sized slab.
        cp.n_outputs_max = n_seq * (uint32_t) (spec_window() + 1) + 1;
    }

    // Drafts a round may verify: the heads' n_max, or a longer copy of earlier context (ngram_draft).
    int32_t spec_window() const { return spec_n_max > 0 ? std::max(spec_n_max, ngram_draft) : 0; }

    // Copying drafts: when the last ngram_match tokens occurred earlier in the context, the tokens that followed
    // them there. Agent replies copy code and tool output, where such drafts are mostly accepted; a long match
    // keeps them from replacing the heads' drafts elsewhere.
    static constexpr int32_t ngram_match = 8, ngram_draft = 5;

    int32_t spec_n_max = 0, spec_draft_vocab = 0;
    bool host_overrides = false;
    std::vector<int32_t> cache_layers, cache_offsets, cache_experts, cache_slots;
    struct tier_spec {
        std::string device;
        std::vector<int32_t> layers, offsets, experts;
    };
    std::vector<tier_spec> cache_tiers;
    bool cache_frozen = false;
};

// The next-token heads' draft context: same placement and threads as the target, MTP graph, no rollback
// snapshots of its own. Its micro-batch only has to carry a prompt's hidden states through the heads after
// prefill: 128 tokens take few passes while keeping its buffers small on the heads' GPU.
llama_context_params draft_context_params(const llama_context_params & target, llama_context * ctx_tgt) {
    llama_context_params dp = target;
    dp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    dp.n_ctx = llama_n_ctx(ctx_tgt);
    dp.n_rs_seq = 0;
    dp.ctx_other = ctx_tgt;
    dp.n_ubatch = std::min<uint32_t>(dp.n_ubatch, 128);
    return dp;
}

// `draft_vocab` > 0 limits drafts to the first token ids (the plan measured that the model's output stays
// there): the heads' output head over the rest would cost most of each draft step. Verification uses every id.
llama_context * new_draft_context(llama_model * model, const llama_context_params & target, llama_context * ctx_tgt, int32_t draft_vocab) {
    llama_context * ctx = llama_init_from_model(model, draft_context_params(target, ctx_tgt));
    if (ctx && draft_vocab > 0) {
        llama_set_draft_vocab(ctx, draft_vocab);
    }
    return ctx;
}

// Plans with host-resident weights (overridden tensors or CPU layers) stage the weights they offload and send
// offloaded ops to the GPU with the fastest measured host link; upstream llama.cpp does neither by default.
void tune_offload(llama_context * ctx, const load_params & p, const llama_model * model) {
    if (p.host_overrides || p.mp.n_gpu_layers <= llama_model_n_layer(model)) {
        llama_set_offload_tuning(ctx, true);
    }
}

// Per-device totals of a context's buffers. Host-visible buffers (including CUDA pinned host) count as CPU.
json memory_json(const llama_context * ctx) {
    std::map<std::string, llama_memory_breakdown_data> totals;
    for (const auto & [buft, mb] : llama_get_memory_breakdown(ctx)) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        const bool host = ggml_backend_buft_is_host(buft) || (dev && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) || !dev;
        auto & t = totals[host ? std::string("CPU") : std::string(ggml_backend_dev_name(dev))];
        t.model += mb.model;
        t.context += mb.context;
        t.compute += mb.compute;
        t.staging += mb.staging;
    }
    json out = json::array();
    for (const auto & [name, t] : totals) {
        out.push_back({{"device", name}, {"model", t.model}, {"context", t.context}, {"compute", t.compute}, {"staging", t.staging}});
    }
    return out;
}

void set_threads(ggml_backend_t backend, ggml_backend_dev_t dev, int n) {
    auto * fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
    if (fn && n > 0) {
        fn(backend, n);
    }
}

// Fills a weight tensor with real model bytes when a source is given, else with quantized noise.
void fill_weight(ggml_tensor * w, const json & src) {
    const size_t nbytes = ggml_nbytes(w);
    std::vector<uint8_t> host(nbytes);
    if (src.is_object()) {
        std::ifstream f(src.at("path").get<std::string>(), std::ios::binary);
        f.seekg((std::streamoff) src.at("offset").get<uint64_t>());
        f.read(reinterpret_cast<char *>(host.data()), (std::streamsize) nbytes);
        if (!f) {
            throw std::runtime_error("could not read probe weights from source file");
        }
    } else {
        const int64_t k = w->ne[0], n = w->ne[1];
        std::vector<float> data((size_t) (k * n));
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);
        for (auto & x : data) {
            x = dist(rng);
        }
        std::vector<float> imatrix;
        if (ggml_quantize_requires_imatrix(w->type)) {
            imatrix.assign((size_t) k, 1.0f);
        }
        ggml_quantize_chunk(w->type, data.data(), host.data(), 0, n, k, imatrix.empty() ? nullptr : imatrix.data());
    }
    ggml_backend_tensor_set(w, host.data(), 0, nbytes);
}

void fill_f32(ggml_tensor * t) {
    std::vector<float> v((size_t) ggml_nelements(t));
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & x : v) {
        x = dist(rng);
    }
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

struct ctx_guard {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ~ctx_guard() {
        if (buf) ggml_backend_buffer_free(buf);
        if (ctx) ggml_free(ctx);
    }
};

ggml_context * small_ctx() {
    ggml_init_params ip = {ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true};
    return ggml_init(ip);
}

// Times W(k x n, type) * X(k x batch) on a backend; returns microseconds per run.
std::vector<double> time_matmul(ggml_backend_t backend, ggml_type type, int64_t k, int64_t n, int64_t batch, int iters, const json & src) {
    ctx_guard g;
    g.ctx = small_ctx();
    ggml_tensor * w = ggml_new_tensor_2d(g.ctx, type, k, n);
    ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, batch);
    ggml_tensor * y = ggml_mul_mat(g.ctx, w, x);
    ggml_cgraph * gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(gf, y);
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, backend);
    if (!g.buf) {
        throw std::runtime_error("could not allocate probe tensors");
    }
    fill_weight(w, src);
    fill_f32(x);
    for (int i = 0; i < 2; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    std::vector<double> out;
    for (int i = 0; i < iters; i++) {
        const double t0 = now_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        out.push_back(now_us() - t0);
    }
    return out;
}

// A device-resident byte tensor plus host memory (pinned or pageable) for copy probes.
struct copy_rig {
    ctx_guard dev;
    ggml_tensor * t = nullptr;
    ggml_backend_buffer_t host_buf = nullptr;
    std::vector<uint8_t> pageable;
    uint8_t * host = nullptr;

    copy_rig(ggml_backend_dev_t d, size_t bytes, bool pinned) {
        dev.ctx = small_ctx();
        t = ggml_new_tensor_1d(dev.ctx, GGML_TYPE_I8, (int64_t) bytes);
        dev.buf = ggml_backend_alloc_ctx_tensors_from_buft(dev.ctx, ggml_backend_dev_buffer_type(d));
        if (!dev.buf) {
            throw std::runtime_error("could not allocate device copy buffer");
        }
        ggml_backend_buffer_type_t hb = pinned ? ggml_backend_dev_host_buffer_type(d) : nullptr;
        if (hb) {
            host_buf = ggml_backend_buft_alloc_buffer(hb, bytes);
            host = static_cast<uint8_t *>(ggml_backend_buffer_get_base(host_buf));
        } else {
            pageable.assign(bytes, 1);
            host = pageable.data();
        }
        memset(host, 1, bytes);
    }
    ~copy_rig() {
        if (host_buf) ggml_backend_buffer_free(host_buf);
    }
    bool pinned() const { return host_buf != nullptr; }
};

double copy_once(copy_rig & r, size_t bytes, bool h2d) {
    const double t0 = now_us();
    if (h2d) {
        ggml_backend_tensor_set(r.t, r.host, 0, bytes);
    } else {
        ggml_backend_tensor_get(r.t, r.host, 0, bytes);
    }
    return now_us() - t0;
}

// Runs `fn` repeatedly for `seconds` and returns achieved GB/s given bytes per call.
double sustained_gbps(const std::function<double()> & fn, size_t bytes, double seconds) {
    const double t_end = now_us() + seconds * 1e6;
    double busy = 0;
    size_t moved = 0;
    while (now_us() < t_end) {
        busy += fn();
        moved += bytes;
    }
    return busy > 0 ? moved / (busy * 1e3) : 0;
}

ggml_tensor * mul_mat_id_op(ggml_context * ctx, ggml_type type) {
    const int64_t k = 256 * 4, n = 512, n_expert = 8, n_used = 2, batch = 4;
    ggml_tensor * as = ggml_new_tensor_3d(ctx, type, k, n, n_expert);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, batch);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, batch);
    return ggml_mul_mat_id(ctx, as, b, ids);
}

// Recommended sampling stored in the model, applied before request settings as llama-server does
// (mirrors the static common_init_sampler_from_model of the pinned common library).
void model_sampling_defaults(const llama_model * model, common_params_sampling & p) {
    char buf[512];
    auto meta = [&](llama_model_meta_key k) -> const char * {
        buf[0] = 0;
        return llama_model_meta_val_str(model, llama_model_meta_key_str(k), buf, sizeof(buf)) > 0 ? buf : nullptr;
    };
    auto as_int = [&](llama_model_meta_key k, int32_t & dst) {
        if (const char * v = meta(k)) {
            char * end = nullptr;
            const long x = strtol(v, &end, 10);
            if (end != v) dst = (int32_t) x;
        }
    };
    auto as_float = [&](llama_model_meta_key k, float & dst) {
        if (const char * v = meta(k)) {
            char * end = nullptr;
            const float x = strtof(v, &end);
            if (end != v) dst = x;
        }
    };
    if (const char * seq = meta(LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE)) {
        const auto names = string_split<std::string>(std::string(seq), ';');
        if (!names.empty()) p.samplers = common_sampler_types_from_names(names);
    }
    as_int(LLAMA_MODEL_META_KEY_SAMPLING_TOP_K, p.top_k);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_TOP_P, p.top_p);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_MIN_P, p.min_p);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY, p.xtc_probability);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD, p.xtc_threshold);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_TEMP, p.temp);
    as_int(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N, p.penalty_last_n);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT, p.penalty_repeat);
    as_int(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT, p.mirostat);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU, p.mirostat_tau);
    as_float(LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA, p.mirostat_eta);
}

} // namespace

// Time attribution through the scheduler's eval callback. Device mode first discovers the device
// of every node, then observes only the last node of each device run, so each split still executes
// as one piece from its first node (observing mid-split makes the backend run the remainder far
// slower than normal). Ops mode observes every computing node and is only indicative.
struct trace_state {
    bool active = false;
    bool per_op = false;
    // Routing mode: count the experts each MoE layer selects (from the ffn_moe_topk-<layer> nodes).
    bool routes = false;
    std::map<int, std::vector<int64_t>> route_counts;
    bool discovering = false;
    size_t ask_idx = 0;
    std::vector<std::string> devs;
    std::vector<bool> observe;
    double last_us = 0;
    std::string last_dev;
    std::map<std::pair<std::string, std::string>, std::pair<double, int64_t>> by_dev_op;
    double switch_us = 0;
    int64_t switches = 0;
};

static std::string tensor_device(const ggml_tensor * t) {
    if (!t->buffer) {
        return "CPU";
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(t->buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (ggml_backend_buft_is_host(buft) || !dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return "CPU";
    }
    return ggml_backend_dev_name(dev);
}

static bool trace_cb(ggml_tensor * t, bool ask, void * ud) {
    auto * st = static_cast<trace_state *>(ud);
    if (!st->active) {
        return false;
    }
    if (st->routes) {
        const char * name = ggml_get_name(t);
        if (strncmp(name, "ffn_moe_topk-", 13) != 0 || t->type != GGML_TYPE_I32) {
            return false;
        }
        if (ask) {
            return true;
        }
        // The top-k selection is a strided view into the full argsort: copy its span, index by strides.
        std::vector<uint8_t> span(ggml_nbytes(t));
        ggml_backend_tensor_get(t, span.data(), 0, span.size());
        auto & counts = st->route_counts[atoi(name + 13)];
        for (int64_t i3 = 0; i3 < t->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < t->ne[2]; i2++) {
                for (int64_t i1 = 0; i1 < t->ne[1]; i1++) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; i0++) {
                        int32_t id;
                        memcpy(&id, span.data() + i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3], sizeof(id));
                        if (id >= 0) {
                            if ((size_t) id >= counts.size()) counts.resize((size_t) id + 1, 0);
                            counts[(size_t) id]++;
                        }
                    }
                }
            }
        }
        return true;
    }
    if (ask && !st->per_op) {
        const size_t i = st->ask_idx++;
        if (st->discovering) {
            st->devs.push_back(tensor_device(t));
            return false;
        }
        return i < st->observe.size() && st->observe[i];
    }
    if (ask) {
        // Views do no work; their (zero) time folds into the next observed node.
        switch (t->op) {
            case GGML_OP_NONE:
            case GGML_OP_VIEW:
            case GGML_OP_RESHAPE:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                return false;
            default:
                return true;
        }
    }
    const double now = now_us();
    const std::string dev = tensor_device(t);
    const double dt = now - st->last_us;
    if (!st->per_op) {
        // A device run just ended: the interval covers its input copies and its compute.
        auto & slot = st->by_dev_op[{dev, "split"}];
        slot.first += dt;
        slot.second++;
        st->switches++;
        st->last_us = now;
        st->last_dev = dev;
        return true;
    }
    if (!st->last_dev.empty() && dev != st->last_dev) {
        st->switch_us += dt;
        st->switches++;
    }
    auto & slot = st->by_dev_op[{dev, ggml_op_desc(t)}];
    slot.first += dt;
    slot.second++;
    st->last_us = now;
    st->last_dev = dev;
    return true;
}

struct fx_engine {
    trace_state trace;
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    common_chat_templates_ptr tmpls;
    llama_batch batch{};
    int32_t n_batch = 0;
    // Speculation: the next-token heads run in their own context against the target's hidden states.
    llama_context * ctx_dft = nullptr;
    common_speculative * spec = nullptr;
    // Drafts a round may verify, and those the heads draft: a longer draft copies earlier context.
    int32_t spec_n_max = 0, spec_heads_max = 0;
    // Verification rounds, drafted and accepted tokens, and time spent drafting, verifying, sampling the
    // verified rows and following the target (microseconds), logged periodically.
    int64_t spec_rounds = 0, spec_drafted = 0, spec_accepted = 0, spec_draft_us = 0, spec_verify_us = 0, spec_sample_us = 0, spec_follow_us = 0;
    // The rounds that verified copied context, their drafted and accepted tokens, and whether each sequence's
    // last draft was one.
    int64_t spec_copy_rounds = 0, spec_copy_drafted = 0, spec_copy_accepted = 0;
    std::vector<bool> spec_copied;
    int32_t n_threads = 0, n_threads_batch = 0;
    // Recurrent state cannot be trimmed back to an arbitrary position, only restored from a checkpoint.
    bool recurrent = false;
    // An expert cache serves the plan's cached layers; its hit rate is logged every `cache_log_every` decodes.
    bool cached = false;
    int64_t decodes = 0;
    static constexpr int64_t cache_log_every = 256;
    // Prompt reuse: per sequence, the recurrent state saved at `n` positions, ascending by `n`. Several per
    // sequence, so that requests sharing a shorter prefix (a new turn, a side request) still find one.
    struct checkpoint {
        int32_t n = 0;
        std::vector<uint8_t> data;
        // The drafter's pending hidden state at the same position, so drafts after reuse pair with it.
        std::vector<uint8_t> draft;
    };
    std::map<int32_t, std::vector<checkpoint>> checkpoints;
    static constexpr size_t max_checkpoints = 4;
    bool checkpoint_size_logged = false;

    ~fx_engine() {
        if (spec) common_speculative_free(spec);
        if (ctx_dft) llama_free(ctx_dft);
        if (batch.token) llama_batch_free(batch);
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
};

struct fx_sampler {
    common_sampler * s = nullptr;
    ~fx_sampler() {
        if (s) common_sampler_free(s);
    }
};

extern "C" {

void fx_free(char * p) {
    free(p);
}

char * fx_backend_info(void) {
    try {
        ensure_init();
        return dup(json{
            {"commit", llama_commit()},
            {"build_number", llama_build_number()},
            {"compiler", llama_compiler()},
            {"target", llama_build_target()},
            {"system_info", llama_print_system_info()},
            {"devices", devices_json()},
            {"max_devices", llama_max_devices()},
            {"max_overrides", llama_max_tensor_buft_overrides()},
            {"supports_mmap", llama_supports_mmap()},
            {"supports_mlock", llama_supports_mlock()},
        }.dump());
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

char * fx_measure(const char * params_json) {
    try {
        ensure_init();
        load_params p(json::parse(params_json));
        p.mp.no_alloc = true;
        p.mp.load_mode = LLAMA_LOAD_MODE_NONE;
        llama_model * model = llama_model_load_from_file(p.model.c_str(), p.mp);
        if (!model) {
            return err_json("backend rejected the model with these parameters");
        }
        llama_context * ctx = llama_init_from_model(model, p.cp);
        if (!ctx) {
            llama_model_free(model);
            return err_json("backend could not create a context with these parameters");
        }
        tune_offload(ctx, p, model);
        llama_context * ctx_dft = nullptr;
        if (p.spec_n_max > 0) {
            ctx_dft = new_draft_context(model, p.cp, ctx, p.spec_draft_vocab);
            if (!ctx_dft) {
                llama_free(ctx);
                llama_model_free(model);
                return err_json("backend could not create the draft context with these parameters");
            }
        }
        json memory = memory_json(ctx);
        if (ctx_dft) {
            // The draft context shares the model's weights: add only its own cache and compute buffers.
            for (const auto & d : memory_json(ctx_dft)) {
                auto it = std::find_if(memory.begin(), memory.end(), [&](const json & m) { return m["device"] == d["device"]; });
                if (it == memory.end()) {
                    memory.push_back({{"device", d["device"]}, {"model", 0}, {"context", d["context"]}, {"compute", d["compute"]}});
                    continue;
                }
                for (const char * k : {"context", "compute"}) {
                    (*it)[k] = (*it)[k].get<uint64_t>() + d[k].get<uint64_t>();
                }
            }
            llama_free(ctx_dft);
        }
        json out = {
            {"memory", memory},
            {"n_ctx", llama_n_ctx(ctx)},
            {"n_ctx_seq", llama_n_ctx_seq(ctx)},
            {"n_layer", llama_model_n_layer(model)},
            {"n_expert", llama_model_n_expert(model)},
        };
        llama_free(ctx);
        llama_model_free(model);
        return dup(out.dump());
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

char * fx_probe_matmul(const char * request_json) {
    try {
        ensure_init();
        const json j = json::parse(request_json);
        ggml_backend_dev_t dev = dev_by_name(j.at("device").get<std::string>());
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            return err_json("could not initialize backend");
        }
        set_threads(backend, dev, j.value("threads", 0));
        const ggml_type type = type_by_name(j.at("type").get<std::string>());
        const int64_t k = j.at("k").get<int64_t>(), n = j.at("n").get<int64_t>();
        const int iters = j.value("iters", 10);
        json results = json::array();
        try {
            for (const auto & b : j.at("batches")) {
                results.push_back({{"batch", b}, {"micros", time_matmul(backend, type, k, n, b.get<int64_t>(), iters, j.value("source", json()))}});
            }
        } catch (...) {
            ggml_backend_free(backend);
            throw;
        }
        ggml_backend_free(backend);
        return dup(json{{"results", results}}.dump());
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

char * fx_probe_copy(const char * request_json) {
    try {
        ensure_init();
        const json j = json::parse(request_json);
        ggml_backend_dev_t dev = dev_by_name(j.at("device").get<std::string>());
        const std::string dir = j.value("direction", std::string("h2d"));
        const int iters = j.value("iters", 10);
        std::vector<size_t> sizes = j.at("sizes").get<std::vector<size_t>>();
        const size_t max_size = *std::max_element(sizes.begin(), sizes.end());
        json results = json::array();

        if (dir == "d2d") {
            ggml_backend_dev_t peer = dev_by_name(j.at("peer").get<std::string>());
            for (size_t bytes : sizes) {
                copy_rig a(dev, bytes, false), b(peer, bytes, false);
                ggml_backend_tensor_copy(a.t, b.t);
                std::vector<double> us;
                for (int i = 0; i < iters; i++) {
                    const double t0 = now_us();
                    ggml_backend_tensor_copy(a.t, b.t);
                    us.push_back(now_us() - t0);
                }
                results.push_back({{"bytes", bytes}, {"micros", us}});
            }
            return dup(json{{"results", results}, {"pinned", false}}.dump());
        }

        copy_rig rig(dev, max_size, j.value("pinned", true));
        const bool h2d = dir == "h2d";
        for (size_t bytes : sizes) {
            copy_once(rig, bytes, h2d);
            std::vector<double> us;
            for (int i = 0; i < iters; i++) {
                us.push_back(copy_once(rig, bytes, h2d));
            }
            results.push_back({{"bytes", bytes}, {"micros", us}});
        }
        return dup(json{{"results", results}, {"pinned", rig.pinned()}}.dump());
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

char * fx_probe_contention(const char * request_json) {
    try {
        ensure_init();
        const json j = json::parse(request_json);
        const std::string scenario = j.at("scenario").get<std::string>();
        const size_t bytes = j.value("bytes", (size_t) 64 << 20);
        const double seconds = j.value("seconds", 1.5);

        if (scenario == "h2d_pair") {
            // Two GPUs pulling from host at once: shared root complex or CPU memory limits show up here.
            const auto names = j.at("devices").get<std::vector<std::string>>();
            copy_rig a(dev_by_name(names.at(0)), bytes, true), b(dev_by_name(names.at(1)), bytes, true);
            auto fa = [&] { return copy_once(a, bytes, true); };
            auto fb = [&] { return copy_once(b, bytes, true); };
            const double alone_a = sustained_gbps(fa, bytes, seconds), alone_b = sustained_gbps(fb, bytes, seconds);
            double both_a = 0, both_b = 0;
            std::thread ta([&] { both_a = sustained_gbps(fa, bytes, seconds); });
            both_b = sustained_gbps(fb, bytes, seconds);
            ta.join();
            return dup(json{{"results", json::array({
                {{"label", names[0] + " h2d"}, {"alone_gbps", alone_a}, {"contended_gbps", both_a}},
                {{"label", names[1] + " h2d"}, {"alone_gbps", alone_b}, {"contended_gbps", both_b}},
            })}}.dump());
        }
        if (scenario == "h2d_cpu") {
            // Host->GPU transfers while CPU threads stream expert weights: both compete for DRAM bandwidth.
            const std::string name = j.at("device").get<std::string>();
            copy_rig rig(dev_by_name(name), bytes, true);
            ggml_backend_dev_t cpu = dev_by_name("CPU");
            ggml_backend_t be = ggml_backend_dev_init(cpu, nullptr);
            set_threads(be, cpu, j.value("threads", 8));
            const ggml_type type = type_by_name(j.value("type", std::string("q4_K")));
            const int64_t k = j.value("k", (int64_t) 4096), n = j.value("n", (int64_t) 14336);
            const double wbytes = (double) ggml_row_size(type, k) * n;
            ctx_guard g;
            g.ctx = small_ctx();
            ggml_tensor * w = ggml_new_tensor_2d(g.ctx, type, k, n);
            ggml_tensor * x = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, k, 1);
            ggml_cgraph * gf = ggml_new_graph(g.ctx);
            ggml_build_forward_expand(gf, ggml_mul_mat(g.ctx, w, x));
            g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, be);
            fill_weight(w, json());
            fill_f32(x);
            auto fcpu = [&] {
                const double t0 = now_us();
                ggml_backend_graph_compute(be, gf);
                return now_us() - t0;
            };
            auto fcopy = [&] { return copy_once(rig, bytes, true); };
            const double cpu_alone = sustained_gbps(fcpu, (size_t) wbytes, seconds), copy_alone = sustained_gbps(fcopy, bytes, seconds);
            double cpu_both = 0, copy_both = 0;
            std::thread tc([&] { cpu_both = sustained_gbps(fcpu, (size_t) wbytes, seconds); });
            copy_both = sustained_gbps(fcopy, bytes, seconds);
            tc.join();
            ggml_backend_free(be);
            return dup(json{{"results", json::array({
                {{"label", "cpu gemv weight read"}, {"alone_gbps", cpu_alone}, {"contended_gbps", cpu_both}},
                {{"label", name + " h2d"}, {"alone_gbps", copy_alone}, {"contended_gbps", copy_both}},
            })}}.dump());
        }
        return err_json("unknown contention scenario " + scenario);
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

char * fx_supports(const char * request_json) {
    try {
        ensure_init();
        const json j = json::parse(request_json);
        ggml_backend_dev_t dev = dev_by_name(j.at("device").get<std::string>());
        json out = json::object();
        for (const auto & tn : j.at("types")) {
            const ggml_type type = type_by_name(tn.get<std::string>());
            ggml_init_params ip = {ggml_tensor_overhead() * 16, nullptr, true};
            ggml_context * ctx = ggml_init(ip);
            ggml_tensor * w = ggml_new_tensor_2d(ctx, type, 256 * 4, 512);
            ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256 * 4, 8);
            const bool mm = ggml_backend_dev_supports_op(dev, ggml_mul_mat(ctx, w, x));
            const bool mmid = ggml_backend_dev_supports_op(dev, mul_mat_id_op(ctx, type));
            ggml_free(ctx);
            out[tn.get<std::string>()] = {{"mul_mat", mm}, {"mul_mat_id", mmid}};
        }
        return dup(json{{"support", out}}.dump());
    } catch (const std::exception & e) {
        return err_json(e.what());
    }
}

fx_engine * fx_engine_load(const char * params_json, char ** error) {
    try {
        ensure_init();
        load_params p(json::parse(params_json));
        auto * e = new fx_engine();
        if (json::parse(params_json).value("trace", false)) {
            // Observation splits the graph; re-capturing CUDA graphs for every piece would dominate.
            setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
            p.cp.cb_eval = trace_cb;
            p.cp.cb_eval_user_data = &e->trace;
        }
        e->model = llama_model_load_from_file(p.model.c_str(), p.mp);
        if (!e->model) {
            delete e;
            *error = dup("backend failed to load the model (see worker log)");
            return nullptr;
        }
        e->ctx = llama_init_from_model(e->model, p.cp);
        if (!e->ctx) {
            delete e;
            *error = dup("backend failed to create the context (insufficient memory?)");
            return nullptr;
        }
        tune_offload(e->ctx, p, e->model);
        e->vocab = llama_model_get_vocab(e->model);
        e->tmpls = common_chat_templates_init(e->model, "");
        e->n_batch = (int32_t) llama_n_batch(e->ctx);
        e->batch = llama_batch_init(e->n_batch, 0, 1);
        e->n_threads = p.cp.n_threads;
        e->n_threads_batch = p.cp.n_threads_batch;
        e->recurrent = llama_model_is_recurrent(e->model) || llama_model_is_hybrid(e->model);
        if (!p.cache_layers.empty()) {
            if (llama_moe_cache_init(e->ctx, (int32_t) p.cache_layers.size(), p.cache_layers.data(), p.cache_offsets.data(), p.cache_experts.data(),
                                     p.cache_slots.data()) != 0) {
                delete e;
                *error = dup("backend failed to set up the expert cache (see worker log)");
                return nullptr;
            }
            for (const auto & t : p.cache_tiers) {
                if (llama_moe_cache_tier(e->ctx, t.device.c_str(), (int32_t) t.layers.size(), t.layers.data(), t.offsets.data(), t.experts.data()) != 0) {
                    delete e;
                    *error = dup(("backend failed to set up the expert tier on " + t.device + " (see worker log)").c_str());
                    return nullptr;
                }
            }
            if (p.cache_frozen) {
                llama_moe_cache_freeze(e->ctx, true);
            }
            // Prompt uploads and the tiers reach full speed only once the host experts are page-locked: wait,
            // so a loaded engine runs (and is measured) at its real speed from the first request.
            llama_moe_cache_wait_pinned(e->ctx);
            e->cached = true;
        }
        if (p.spec_n_max > 0) {
            if (llama_model_n_layer_nextn(e->model) == 0) {
                delete e;
                *error = dup("speculation needs next-token heads, and the model has none");
                return nullptr;
            }
            e->ctx_dft = new_draft_context(e->model, p.cp, e->ctx, p.spec_draft_vocab);
            if (!e->ctx_dft) {
                delete e;
                *error = dup("backend failed to create the draft context (insufficient memory?)");
                return nullptr;
            }
            common_params_speculative sp;
            // A copying draft comes first; the heads draft whenever the context offers none.
            sp.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
            sp.ngram_simple.size_n = load_params::ngram_match;
            sp.ngram_simple.size_m = load_params::ngram_match;
            sp.draft.n_max = p.spec_n_max;
            // Always draft n_max tokens: a fixed verification batch keeps the target graph reused.
            sp.draft.p_min = 0.0f;
            sp.draft.ctx_tgt = e->ctx;
            sp.draft.ctx_dft = e->ctx_dft;
            e->spec = common_speculative_init(sp, llama_n_seq_max(e->ctx));
            if (!e->spec) {
                delete e;
                *error = dup("backend failed to initialize drafting with the next-token heads");
                return nullptr;
            }
            e->spec_n_max = p.spec_window();
            e->spec_heads_max = p.spec_n_max;
            e->spec_copied.assign(llama_n_seq_max(e->ctx), false);
        }
        return e;
    } catch (const std::exception & ex) {
        *error = dup(ex.what());
        return nullptr;
    } catch (...) {
        *error = dup("backend raised a non-standard exception while loading (see worker log)");
        return nullptr;
    }
}

void fx_engine_free(fx_engine * e) {
    delete e;
}

char * fx_engine_info(fx_engine * e) {
    try {
        return dup(json{
            {"n_ctx", llama_n_ctx(e->ctx)},
            {"n_ctx_seq", llama_n_ctx_seq(e->ctx)},
            {"n_seq", llama_n_seq_max(e->ctx)},
            {"n_batch", e->n_batch},
            {"n_vocab", llama_vocab_n_tokens(e->vocab)},
            {"add_bos", llama_vocab_get_add_bos(e->vocab)},
            {"spec_n_max", e->spec_n_max},
            {"recurrent", e->recurrent},
            {"memory", memory_json(e->ctx)},
        }.dump());
    } catch (const std::exception & ex) {
        return err_json(ex.what());
    }
}

int32_t fx_tokenize(fx_engine * e, const char * text, int32_t len, bool add_special, int32_t * out, int32_t cap) {
    const std::vector<llama_token> toks = common_tokenize(e->vocab, std::string(text, (size_t) len), add_special, true);
    if ((int32_t) toks.size() > cap) {
        return -(int32_t) toks.size();
    }
    std::copy(toks.begin(), toks.end(), out);
    return (int32_t) toks.size();
}

int32_t fx_token_piece(fx_engine * e, int32_t token, bool special, char * buf, int32_t cap) {
    const std::string piece = common_token_to_piece(e->vocab, token, special);
    if ((int32_t) piece.size() > cap) {
        return -(int32_t) piece.size();
    }
    memcpy(buf, piece.data(), piece.size());
    return (int32_t) piece.size();
}

bool fx_is_eog(fx_engine * e, int32_t token) {
    return llama_vocab_is_eog(e->vocab, token);
}

char * fx_apply_template(fx_engine * e, const char * request_json) {
    try {
        const json j = json::parse(request_json);
        common_chat_templates_inputs in;
        in.messages = common_chat_msgs_parse_oaicompat(common_json::parse(j.at("messages").dump()));
        if (j.contains("tools") && !j.at("tools").is_null()) {
            in.tools = common_chat_tools_parse_oaicompat(common_json::parse(j.at("tools").dump()));
        }
        in.add_generation_prompt = j.value("add_generation_prompt", true);
        in.use_jinja = true;
        // As llama-server: reasoning goes to reasoning_content, in streamed deltas too.
        in.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        const common_chat_params cp = common_chat_templates_apply(e->tmpls.get(), in);
        json preserved = json::array();
        for (const auto & t : cp.preserved_tokens) {
            const auto ids = common_tokenize(e->vocab, t, false, true);
            if (ids.size() == 1) {
                preserved.push_back(ids[0]);
            }
        }
        json out = {
            {"prompt", cp.prompt},
            {"preserved_tokens", preserved},
            {"additional_stops", cp.additional_stops},
            {"parser",
             {{"format", (int) cp.format}, {"generation_prompt", cp.generation_prompt}, {"parser", cp.parser}, {"parse_tool_calls", !in.tools.empty()}}},
        };
        // Positions where later prompts still agree with this one, for recurrent models to checkpoint: the end of
        // the system prompt (shared by a client's side requests) and the end of the last message (the next turn
        // agrees up to there even when the template rewrites the reply, e.g. drops its reasoning). Each is where
        // a variant of these messages starts to differ.
        const auto full = common_tokenize(e->vocab, cp.prompt, true, true);
        const auto agreed = [&](common_chat_templates_inputs variant) -> size_t {
            try {
                const auto other = common_tokenize(e->vocab, common_chat_templates_apply(e->tmpls.get(), variant).prompt, true, true);
                size_t n = 0;
                while (n < full.size() && n < other.size() && full[n] == other[n]) {
                    n++;
                }
                return n < full.size() ? n : 0;
            } catch (const std::exception &) {
                return 0;
            }
        };
        json checkpoints = json::array();
        size_t n_sys = 0;
        while (n_sys < in.messages.size() && (in.messages[n_sys].role == "system" || in.messages[n_sys].role == "developer")) {
            n_sys++;
        }
        if (n_sys > 0 && n_sys < in.messages.size()) {
            common_chat_templates_inputs variant = in;
            variant.messages.resize(n_sys);
            common_chat_msg probe;
            probe.role = "user";
            probe.content = "\x01";
            variant.messages.push_back(probe);
            variant.add_generation_prompt = false;
            if (const size_t n = agreed(variant); n > 0) {
                checkpoints.push_back(n);
            }
        }
        if (in.add_generation_prompt) {
            common_chat_templates_inputs variant = in;
            variant.add_generation_prompt = false;
            if (const size_t n = agreed(variant); n > 0) {
                checkpoints.push_back(n);
            }
        }
        out["checkpoints"] = checkpoints;
        return dup(out.dump());
    } catch (const std::exception & ex) {
        return err_json(ex.what());
    }
}

struct fx_chat_parser {
    common_chat_parser_params params;
    std::string text;
    common_chat_msg msg;
    std::vector<std::string> tool_call_ids;
};

static std::string random_tool_call_id() {
    static thread_local std::mt19937 rng{std::random_device{}()};
    static const char chars[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::uniform_int_distribution<int> pick(0, (int) sizeof(chars) - 2);
    std::string id = "call_";
    for (int i = 0; i < 24; i++) {
        id += chars[pick(rng)];
    }
    return id;
}

// One OpenAI chat.completion.chunk delta, shaped as llama-server's server_chat_msg_diff_to_json_oaicompat.
static json diff_json(const common_chat_msg_diff & d) {
    json delta = json::object();
    if (!d.reasoning_content_delta.empty()) {
        delta["reasoning_content"] = d.reasoning_content_delta;
    }
    if (!d.content_delta.empty()) {
        delta["content"] = d.content_delta;
    }
    if (d.tool_call_index != std::string::npos) {
        json call = {{"index", d.tool_call_index}};
        if (!d.tool_call_delta.id.empty()) {
            call["id"] = d.tool_call_delta.id;
            call["type"] = "function";
        }
        if (!d.tool_call_delta.name.empty() || !d.tool_call_delta.arguments.empty()) {
            json function = json::object();
            if (!d.tool_call_delta.name.empty()) {
                function["name"] = d.tool_call_delta.name;
            }
            if (!d.tool_call_delta.arguments.empty()) {
                function["arguments"] = d.tool_call_delta.arguments;
            }
            call["function"] = function;
        }
        delta["tool_calls"] = json::array({call});
    }
    return delta;
}

fx_chat_parser * fx_chat_parser_new(const char * spec_json, char ** error) {
    try {
        const json s = json::parse(spec_json);
        auto p = std::make_unique<fx_chat_parser>();
        p->params.format = static_cast<common_chat_format>(s.at("format").get<int>());
        p->params.generation_prompt = s.value("generation_prompt", std::string());
        p->params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        p->params.parse_tool_calls = s.value("parse_tool_calls", false);
        const std::string peg = s.value("parser", std::string());
        if (!peg.empty()) {
            p->params.parser.load(peg);
        }
        return p.release();
    } catch (const std::exception & ex) {
        if (error) {
            *error = dup(ex.what());
        }
        return nullptr;
    }
}

void fx_chat_parser_free(fx_chat_parser * p) {
    delete p;
}

char * fx_chat_parser_push(fx_chat_parser * p, const char * text, int32_t len, bool final) {
    try {
        p->text.append(text, (size_t) len);
        // As llama-server's task_result_state::update_chat_msg: parse everything so far, send what changed.
        json deltas = json::array();
        common_chat_msg m = common_chat_parse(p->text, !final, p->params);
        if (!m.empty()) {
            m.set_tool_call_ids(p->tool_call_ids, random_tool_call_id);
            for (const auto & d : common_chat_msg_diff::compute_diffs(p->msg, m)) {
                deltas.push_back(diff_json(d));
            }
            p->msg = std::move(m);
        }
        json out = {{"deltas", deltas}};
        if (final) {
            out["message"] = json::parse(p->msg.to_json_oaicompat().dump());
        }
        return dup(out.dump());
    } catch (const std::exception & ex) {
        // the parse lists only what it understood; the raw tail shows what the model wrote instead
        const size_t from = p->text.size() > 600 ? p->text.size() - 600 : 0;
        return err_json(std::string(ex.what()) + "Generated text (tail):\n" + p->text.substr(from));
    }
}

int32_t fx_decode(fx_engine * e, int32_t n, const int32_t * tokens, const int32_t * pos, const int32_t * seq, const int8_t * logits) {
    if (n > e->n_batch) {
        return -100;
    }
    e->batch.n_tokens = n;
    for (int32_t i = 0; i < n; i++) {
        e->batch.token[i] = tokens[i];
        e->batch.pos[i] = pos[i];
        e->batch.n_seq_id[i] = 1;
        e->batch.seq_id[i][0] = seq[i];
        e->batch.logits[i] = logits[i];
    }
    // Batches below the op-offload threshold (decode steps, draft verification) keep host work on the CPU
    // like single-token decode, so they use the decode thread count; prompt chunks use the batch count.
    const bool small = n < op_offload_min_batch();
    llama_set_n_threads(e->ctx, e->n_threads, small ? e->n_threads : e->n_threads_batch);
    const int64_t t_verify = ggml_time_us();
    int32_t rc;
    try {
        rc = llama_decode(e->ctx, e->batch);
    } catch (const std::exception & ex) {
        fprintf(stderr, "decode failed: %s\n", ex.what());
        return -102;
    }
    if (e->spec && small) {
        // The drafter reads the target's hidden states next, so waiting here only moves the wait into this timer.
        llama_synchronize(e->ctx);
        e->spec_verify_us += ggml_time_us() - t_verify;
    }
    if (rc == 0 && e->cached && ++e->decodes % fx_engine::cache_log_every == 0) {
        int64_t hits = 0, lookups = 0, swaps = 0;
        llama_moe_cache_stats(e->ctx, &hits, &lookups, &swaps);
        fprintf(stderr, "expert cache: %.1f%% of %lld selections served from GPU memory, %lld swaps\n", 100.0 * hits / std::max<int64_t>(lookups, 1),
                (long long) lookups, (long long) swaps);
    }
    // The drafter consumes the target's hidden state for every decoded row, prompt included.
    try {
        const int64_t t0 = ggml_time_us();
        if (rc == 0 && e->spec && !common_speculative_process(e->spec, e->batch)) {
            return -101;
        }
        if (small) {
            e->spec_follow_us += ggml_time_us() - t0;
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "drafter failed: %s\n", ex.what());
        return -101;
    }
    return rc;
}

char * fx_trace(fx_engine * e, const char * request_json) {
    try {
        const json j = json::parse(request_json);
        const auto prompt = j.at("prompt").get<std::vector<llama_token>>();
        const int steps = j.value("steps", 16);
        const llama_seq_id seq = j.value("seq", 0);
        e->trace.per_op = j.value("per_op", false);
        llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
        for (size_t i = 0; i < prompt.size(); i += (size_t) e->n_batch) {
            const int32_t n = (int32_t) std::min(prompt.size() - i, (size_t) e->n_batch);
            e->batch.n_tokens = n;
            for (int32_t k = 0; k < n; k++) {
                e->batch.token[k] = prompt[i + k];
                e->batch.pos[k] = (llama_pos) (i + k);
                e->batch.n_seq_id[k] = 1;
                e->batch.seq_id[k][0] = seq;
                e->batch.logits[k] = (i + k + 1 == prompt.size());
            }
            if (llama_decode(e->ctx, e->batch) != 0) {
                return err_json("prefill failed");
            }
        }
        llama_sampler * greedy = llama_sampler_init_greedy();
        llama_pos pos = (llama_pos) prompt.size();
        llama_token tok = llama_sampler_sample(greedy, e->ctx, -1);
        auto step = [&](bool traced) -> double {
            e->trace.active = traced;
            e->batch.n_tokens = 1;
            e->batch.token[0] = tok;
            e->batch.pos[0] = pos++;
            e->batch.n_seq_id[0] = 1;
            e->batch.seq_id[0][0] = seq;
            e->batch.logits[0] = 1;
            const double t0 = now_us();
            e->trace.last_us = t0;
            e->trace.last_dev.clear();
            e->trace.ask_idx = 0;
            const int rc = llama_decode(e->ctx, e->batch);
            llama_synchronize(e->ctx);
            const double t1 = now_us();
            if (traced && !e->trace.per_op && !e->trace.discovering && !e->trace.last_dev.empty()) {
                auto & slot = e->trace.by_dev_op[{e->trace.last_dev, "split"}];
                slot.first += t1 - e->trace.last_us;
                slot.second++;
            }
            const double dt = t1 - t0;
            e->trace.active = false;
            if (rc != 0) {
                throw std::runtime_error("decode failed during trace");
            }
            tok = llama_sampler_sample(greedy, e->ctx, -1);
            return dt;
        };
        std::vector<double> plain, traced;
        for (int i = 0; i < steps; i++) {
            plain.push_back(step(false));
        }
        if (!e->trace.per_op) {
            e->trace.devs.clear();
            e->trace.discovering = true;
            step(true);
            e->trace.discovering = false;
            const auto & d = e->trace.devs;
            e->trace.observe.assign(d.size(), false);
            for (size_t i = 0; i < d.size(); i++) {
                e->trace.observe[i] = i + 1 == d.size() || d[i + 1] != d[i];
            }
        }
        e->trace.by_dev_op.clear();
        e->trace.switch_us = 0;
        e->trace.switches = 0;
        for (int i = 0; i < steps; i++) {
            traced.push_back(step(true));
        }
        llama_sampler_free(greedy);
        llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
        json ops = json::array();
        for (const auto & [k, v] : e->trace.by_dev_op) {
            ops.push_back({{"device", k.first}, {"op", k.second}, {"us", v.first / steps}, {"count", v.second / steps}});
        }
        return dup(json{{"steps", steps},
                        {"plain_step_us", plain},
                        {"traced_step_us", traced},
                        {"device_switch_us", e->trace.switch_us / steps},
                        {"device_switches", e->trace.switches / steps},
                        {"ops", ops}}
                       .dump());
    } catch (const std::exception & ex) {
        e->trace.active = false;
        return err_json(ex.what());
    }
}

char * fx_route_stats(fx_engine * e, const char * request_json) {
    try {
        const json j = json::parse(request_json);
        const auto prompt = j.at("prompt").get<std::vector<llama_token>>();
        const int steps = j.value("steps", 64);
        const llama_seq_id seq = j.value("seq", 0);
        e->trace.route_counts.clear();
        e->trace.routes = true;
        e->trace.active = true;
        llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
        llama_sampler * greedy = llama_sampler_init_greedy();
        llama_pos pos = 0;
        auto run = [&](const llama_token * toks, int32_t n) {
            e->batch.n_tokens = n;
            for (int32_t k = 0; k < n; k++) {
                e->batch.token[k] = toks[k];
                e->batch.pos[k] = pos++;
                e->batch.n_seq_id[k] = 1;
                e->batch.seq_id[k][0] = seq;
                e->batch.logits[k] = k + 1 == n;
            }
            if (llama_decode(e->ctx, e->batch) != 0) {
                throw std::runtime_error("decode failed while tracing routes");
            }
        };
        json prefill_counts;
        for (size_t i = 0; i < prompt.size(); i += (size_t) e->n_batch) {
            run(prompt.data() + i, (int32_t) std::min(prompt.size() - i, (size_t) e->n_batch));
        }
        prefill_counts = e->trace.route_counts;
        e->trace.route_counts.clear();
        for (int i = 0; i < steps; i++) {
            llama_token tok = llama_sampler_sample(greedy, e->ctx, -1);
            run(&tok, 1);
        }
        e->trace.active = false;
        e->trace.routes = false;
        llama_sampler_free(greedy);
        llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
        json decode_counts = e->trace.route_counts;
        return dup(json{{"prefill_tokens", prompt.size()}, {"decode_tokens", steps}, {"prefill", prefill_counts}, {"decode", decode_counts}}.dump());
    } catch (const std::exception & ex) {
        e->trace.active = false;
        e->trace.routes = false;
        return err_json(ex.what());
    }
}

void fx_seq_clear(fx_engine * e, int32_t seq) {
    llama_memory_seq_rm(llama_get_memory(e->ctx), seq, -1, -1);
    if (e->ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(e->ctx_dft), seq, -1, -1);
    }
    e->checkpoints.erase(seq);
}

bool fx_seq_checkpoint(fx_engine * e, int32_t seq) {
    try {
        fx_engine::checkpoint c;
        c.data.resize(llama_state_seq_get_size_ext(e->ctx, seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        c.n = llama_memory_seq_pos_max(llama_get_memory(e->ctx), seq) + 1;
        if (llama_state_seq_get_data_ext(e->ctx, c.data.data(), c.data.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) != c.data.size()) {
            return false;
        }
        if (!e->spec || !common_speculative_get_state(e->spec, seq, c.draft)) {
            c.draft.clear();
        }
        if (!e->checkpoint_size_logged) {
            e->checkpoint_size_logged = true;
            fprintf(stderr, "prompt reuse: a checkpoint holds %.1f MiB of recurrent state (up to %zu per sequence)\n",
                    (c.data.size() + c.draft.size()) / 1048576.0, fx_engine::max_checkpoints);
        }
        auto & list = e->checkpoints[seq];
        list.erase(std::remove_if(list.begin(), list.end(), [&](const fx_engine::checkpoint & o) { return o.n == c.n; }), list.end());
        list.push_back(std::move(c));
        std::sort(list.begin(), list.end(), [](const fx_engine::checkpoint & a, const fx_engine::checkpoint & b) { return a.n < b.n; });
        if (list.size() > fx_engine::max_checkpoints) {
            list.erase(list.begin(), list.end() - fx_engine::max_checkpoints);
        }
        return true;
    } catch (const std::exception & ex) {
        fprintf(stderr, "checkpoint failed: %s\n", ex.what());
        return false;
    }
}

int32_t fx_seq_keep(fx_engine * e, int32_t seq, int32_t keep) {
    llama_memory_t mem = llama_get_memory(e->ctx);
    const int32_t end = llama_memory_seq_pos_max(mem, seq) + 1;
    const auto trim_draft = [&](int32_t n) {
        if (e->ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(e->ctx_dft), seq, n, -1);
        }
    };
    if (keep >= end) {
        return end;
    }
    // Attention caches trim anywhere. Recurrent rollback snapshots may predate the last step (a short
    // final ubatch leaves deeper slots stale), so recurrent state comes back only from the checkpoint.
    if (!e->recurrent && keep > 0 && llama_memory_seq_rm(mem, seq, keep, -1)) {
        trim_draft(keep);
        return keep;
    }
    // The latest checkpoint within `keep`; those past it describe positions the trim removes.
    const auto it = e->checkpoints.find(seq);
    if (it != e->checkpoints.end()) {
        auto & list = it->second;
        const auto c = std::find_if(list.rbegin(), list.rend(), [&](const fx_engine::checkpoint & o) { return o.n > 0 && o.n <= keep; });
        if (c != list.rend() &&
            llama_state_seq_set_data_ext(e->ctx, c->data.data(), c->data.size(), seq, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == c->data.size() &&
            llama_memory_seq_rm(mem, seq, c->n, -1)) {
            const int32_t n = c->n;
            trim_draft(n);
            if (e->spec && !c->draft.empty()) {
                common_speculative_set_state(e->spec, seq, c->draft);
            }
            list.erase(std::remove_if(list.begin(), list.end(), [&](const fx_engine::checkpoint & o) { return o.n > n; }), list.end());
            return n;
        }
    }
    fx_seq_clear(e, seq);
    return 0;
}

int32_t fx_spec_draft(fx_engine * e, int32_t seq, int32_t pos, int32_t last, const int32_t * hist, int32_t n_hist, int32_t n_max, int32_t * out) {
    llama_tokens draft;
    try {
        const int64_t t0 = ggml_time_us();
        // the copying drafter searches the context for the tokens before `last`
        const llama_tokens prompt(hist, hist + std::max(n_hist, 0));
        common_speculative_get_draft_params(e->spec, seq) = {true, n_max, pos, last, &prompt, &draft};
        common_speculative_draft(e->spec);
        e->spec_draft_us += ggml_time_us() - t0;
    } catch (const std::exception & ex) {
        // Drafting is an optimization: a failed draft verifies nothing extra.
        fprintf(stderr, "drafting failed: %s\n", ex.what());
        draft.clear();
    }
    // Verification re-enters these positions with the target's hidden states.
    llama_memory_seq_rm(llama_get_memory(e->ctx_dft), seq, pos, -1);
    draft.resize(std::min<size_t>(draft.size(), (size_t) std::max(n_max, 0)));
    e->spec_drafted += (int64_t) draft.size();
    e->spec_copied[seq] = (int32_t) draft.size() > e->spec_heads_max;
    e->spec_copy_drafted += e->spec_copied[seq] ? (int64_t) draft.size() : 0;
    std::copy(draft.begin(), draft.end(), out);
    return (int32_t) draft.size();
}

bool fx_spec_accept(fx_engine * e, int32_t seq, int32_t pos, int32_t n_accepted) {
    try {
        const bool ok = llama_memory_seq_rm(llama_get_memory(e->ctx), seq, pos, -1);
        llama_memory_seq_rm(llama_get_memory(e->ctx_dft), seq, pos, -1);
        common_speculative_accept(e->spec, seq, (uint16_t) n_accepted);
        e->spec_accepted += n_accepted;
        if (e->spec_copied[seq]) {
            e->spec_copy_rounds++;
            e->spec_copy_accepted += n_accepted;
        }
        if (++e->spec_rounds % 64 == 0) {
            fprintf(stderr, "speculation: %lld rounds, %lld of %lld drafts accepted (%.1f tokens per round), %.1f ms drafting, %.1f ms verifying, %.1f ms sampling and %.1f ms following per round; %lld rounds copied context, %lld of %lld accepted\n",
                    (long long) e->spec_rounds, (long long) e->spec_accepted, (long long) e->spec_drafted, 1.0 + (double) e->spec_accepted / (double) e->spec_rounds,
                    e->spec_draft_us / 1e3 / e->spec_rounds, e->spec_verify_us / 1e3 / e->spec_rounds, e->spec_sample_us / 1e3 / e->spec_rounds,
                    e->spec_follow_us / 1e3 / e->spec_rounds, (long long) e->spec_copy_rounds, (long long) e->spec_copy_accepted, (long long) e->spec_copy_drafted);
        }
        return ok;
    } catch (const std::exception & ex) {
        fprintf(stderr, "rollback failed: %s\n", ex.what());
        return false;
    }
}

fx_sampler * fx_sampler_new(fx_engine * e, const char * sampling_json) {
    try {
        const json j = json::parse(sampling_json);
        common_params_sampling p;
        model_sampling_defaults(e->model, p);
        p.temp = j.value("temperature", p.temp);
        p.top_k = j.value("top_k", p.top_k);
        p.top_p = j.value("top_p", p.top_p);
        p.min_p = j.value("min_p", p.min_p);
        p.penalty_repeat = j.value("repeat_penalty", p.penalty_repeat);
        p.penalty_last_n = j.value("repeat_last_n", p.penalty_last_n);
        p.penalty_present = j.value("presence_penalty", p.penalty_present);
        p.penalty_freq = j.value("frequency_penalty", p.penalty_freq);
        p.seed = j.value("seed", p.seed);
        if (j.value("ignore_eos", false)) {
            for (llama_token t = 0; t < llama_vocab_n_tokens(e->vocab); t++) {
                if (llama_vocab_is_eog(e->vocab, t)) {
                    p.logit_bias.push_back({t, -INFINITY});
                }
            }
        }
        auto * s = new fx_sampler();
        s->s = common_sampler_init(e->model, p);
        if (!s->s) {
            delete s;
            return nullptr;
        }
        return s;
    } catch (const std::exception &) {
        return nullptr;
    }
}

void fx_sampler_free(fx_sampler * s) {
    delete s;
}

void fx_sampler_accept_prompt(fx_sampler * s, int32_t token) {
    common_sampler_accept(s->s, token, false);
}

int32_t fx_sampler_sample(fx_sampler * s, fx_engine * e, int32_t idx) {
    try {
        const llama_token t = common_sampler_sample(s->s, e->ctx, idx);
        common_sampler_accept(s->s, t, true);
        return t;
    } catch (const std::exception & ex) {
        fprintf(stderr, "sampling failed: %s\n", ex.what());
        return -1;
    }
}

int32_t fx_runner_up(fx_engine * e, int32_t idx, int32_t chosen) {
    const float * logits = llama_get_logits_ith(e->ctx, idx);
    if (!logits) {
        return -1;
    }
    int32_t best = -1;
    for (int32_t t = 0, n = llama_vocab_n_tokens(e->vocab); t < n; t++) {
        if (t != chosen && (best < 0 || logits[t] > logits[best])) {
            best = t;
        }
    }
    return best;
}

int32_t fx_sampler_sample_draft(fx_sampler * s, fx_engine * e, int32_t row, const int32_t * draft, int32_t n_draft, int32_t * out) {
    try {
        std::vector<int> idxs(n_draft + 1);
        for (int32_t i = 0; i <= n_draft; i++) {
            idxs[i] = row + i;
        }
        const int64_t t0 = ggml_time_us();
        const auto ids = common_sampler_sample_and_accept_n(s->s, e->ctx, idxs, llama_tokens(draft, draft + n_draft));
        e->spec_sample_us += ggml_time_us() - t0;
        std::copy(ids.begin(), ids.end(), out);
        return (int32_t) ids.size();
    } catch (const std::exception & ex) {
        fprintf(stderr, "sampling failed: %s\n", ex.what());
        return -1;
    }
}

} // extern "C"
