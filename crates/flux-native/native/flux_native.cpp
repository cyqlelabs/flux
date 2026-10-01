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
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
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
        return v ? atoi(v) : GGML_LOG_LEVEL_WARN;
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
    }
};

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
    }
    json out = json::array();
    for (const auto & [name, t] : totals) {
        out.push_back({{"device", name}, {"model", t.model}, {"context", t.context}, {"compute", t.compute}});
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

    ~fx_engine() {
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
        json out = {
            {"memory", memory_json(ctx)},
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
        e->vocab = llama_model_get_vocab(e->model);
        e->tmpls = common_chat_templates_init(e->model, "");
        e->n_batch = (int32_t) llama_n_batch(e->ctx);
        e->batch = llama_batch_init(e->n_batch, 0, 1);
        return e;
    } catch (const std::exception & ex) {
        *error = dup(ex.what());
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
        const common_chat_params cp = common_chat_templates_apply(e->tmpls.get(), in);
        json preserved = json::array();
        for (const auto & t : cp.preserved_tokens) {
            const auto ids = common_tokenize(e->vocab, t, false, true);
            if (ids.size() == 1) {
                preserved.push_back(ids[0]);
            }
        }
        return dup(json{{"prompt", cp.prompt}, {"preserved_tokens", preserved}, {"additional_stops", cp.additional_stops}}.dump());
    } catch (const std::exception & ex) {
        return err_json(ex.what());
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
    return llama_decode(e->ctx, e->batch);
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
    const llama_token t = common_sampler_sample(s->s, e->ctx, idx);
    common_sampler_accept(s->s, t, true);
    return t;
}

} // extern "C"
