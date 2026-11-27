// Drives the Inkling 2 flash-prefill lane under the CUDA execution shim: the production tensor
// layer (inkling/fa_prefill_torch.cpp), its launcher (inkling/fa_prefill.cu) and its kernel, all
// compiled from the shipped sources. One scenario per process, chosen by argv[1]; controls come as
// key=value arguments. Prints `CALL ...` (from the shim), `RESULT key=value` for the fence, and —
// for the numerics scenario — dumps every tensor it fed to the kernel so the fence can recompute
// the same attention in float64 and compare. Nothing here reimplements the lane's decisions.

#include "../../../exllamav3/exllamav3_ext/inkling/fa_prefill.cuh"
#include "../../../exllamav3/exllamav3_ext/inkling/fa_prefill_torch.cuh"
#include "shim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

cudaStream_t g_shim_stream = (cudaStream_t) 0x5eed;      // the handle the launch must carry
bool g_q0 = false, g_rel0 = false, g_sink0 = false;       // fixture overrides (isolation arms)

uint16_t em_f32_to_f16(float f);
float em_f16_to_f32(uint16_t h);

namespace
{
struct Buf : std::vector<uint8_t>
{
    void* p() { return data(); }
    const void* p() const { return data(); }
};

// Fixture tensors: contiguous, 16-byte aligned strides, exactly the layouts the served seam hands
// the extension (q/rel/o head-major (h, *, *), k/v (S, kvh, d), sink (T, h), positions int64).
struct Fixture
{
    int h = 4, kvh = 2, d = 128, T = 64, S = 256, ext = 32;
    double scale = 1.0 / 16.0;
    Buf q, k, v, rel, sink, o, nfin, qpos, kpos;
    at::Tensor tq, tk, tv, tre, tsink, to, tnfin, tqpos, tkpos;

    void build(unsigned seed)
    {
        unsigned x = seed ? seed : 12345u;
        auto rnd = [&x]() { x = x * 1103515245u + 12345u; return ((x >> 8) & 0xffffu) / 32768.0f - 1.0f; };

        q.assign(size_t(h) * T * d * 2, 0);
        for (size_t i = 0; i < size_t(h) * T * d; ++i)
            ((uint16_t*) q.p())[i] = em_f32_to_f16(rnd() * 0.6f);
        k.assign(size_t(S) * kvh * d * 2, 0);
        v.assign(size_t(S) * kvh * d * 2, 0);
        for (size_t i = 0; i < size_t(S) * kvh * d; ++i)
        {
            ((uint16_t*) k.p())[i] = em_f32_to_f16(rnd() * 0.6f);
            ((uint16_t*) v.p())[i] = em_f32_to_f16(rnd() * 0.9f);
        }
        rel.assign(size_t(h) * T * ext * 4, 0);
        for (size_t i = 0; i < size_t(h) * T * ext; ++i) ((float*) rel.p())[i] = rnd() * 2.0f;
        sink.assign(size_t(T) * h * 4, 0);
        for (size_t i = 0; i < size_t(T) * h; ++i) ((float*) sink.p())[i] = rnd() * 1.5f;
        o.assign(size_t(h) * T * d * 2, 0xEE);            // poison: a unwritten row must show up
        nfin.assign(4, 0);
        qpos.assign(size_t(T) * 8, 0);
        kpos.assign(size_t(S) * 8, 0);
        for (int i = 0; i < T; ++i) ((int64_t*) qpos.p())[i] = i;
        for (int j = 0; j < S; ++j) ((int64_t*) kpos.p())[j] = j;

        tq = at::Tensor(q.p(), {h, T, d}, {int64_t(T) * d, d, 1}, at::kHalf);
        tk = at::Tensor(k.p(), {S, kvh, d}, {int64_t(kvh) * d, d, 1}, at::kHalf);
        tv = at::Tensor(v.p(), {S, kvh, d}, {int64_t(kvh) * d, d, 1}, at::kHalf);
        tre = at::Tensor(rel.p(), {h, T, ext}, {int64_t(T) * ext, ext, 1}, at::kFloat);
        tsink = at::Tensor(sink.p(), {T, h}, {h, 1}, at::kFloat);
        to = at::Tensor(o.p(), {h, T, d}, {int64_t(T) * d, d, 1}, at::kHalf);
        tnfin = at::Tensor(nfin.p(), {1}, {1}, at::kInt);
        tqpos = at::Tensor(qpos.p(), {T}, {1}, at::kLong);
        tkpos = at::Tensor(kpos.p(), {S}, {1}, at::kLong);
    }

    void dump(const char* path) const
    {
        FILE* f = std::fopen(path, "w");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", path); std::exit(2); }
        std::fprintf(f, "SHAPE h %d kvh %d d %d T %d S %d ext %d scale %.17g\n",
                     h, kvh, d, T, S, ext, scale);
        std::fprintf(f, "NFIN %d\n", ((int32_t*) nfin.p())[0]);
        // 0 = fp16 storage (decoded), 1 = fp32, 2 = int64
        auto put = [&](const char* name, const void* p, size_t n, int kind)
        {
            std::fprintf(f, "%s", name);
            for (size_t i = 0; i < n; ++i)
            {
                const float val = kind == 0 ? em_f16_to_f32(((const uint16_t*) p)[i])
                              : kind == 1 ? ((const float*) p)[i]
                                          : float(((const int64_t*) p)[i]);
                std::fprintf(f, " %.9g", val);
            }
            std::fprintf(f, "\n");
        };
        put("Q", q.p(), size_t(h) * T * d, 0);
        put("K", k.p(), size_t(S) * kvh * d, 0);
        put("V", v.p(), size_t(S) * kvh * d, 0);
        put("REL", rel.p(), size_t(h) * T * ext, 1);
        put("SINK", sink.p(), size_t(T) * h, 1);
        put("QPOS", qpos.p(), size_t(T), 2);
        put("KPOS", kpos.p(), size_t(S), 2);
        put("O", o.p(), size_t(h) * T * d, 0);
        std::fclose(f);
    }
};

void print_plan(const std::map<std::string, int64_t>& p)
{
    for (const auto& kv : p) std::printf("RESULT %s=%lld\n", kv.first.c_str(), (long long) kv.second);
    const std::string why = exl3_inkling_fa_reason();
    std::printf("RESULT reason=%s\n", why.empty() ? "-" : why.c_str());
}

int64_t kb(const char* v) { return strtoll(v, nullptr, 10); }
}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) { std::printf("RESULT error=no-scenario\n"); return 2; }
    const std::string scene = argv[1];
    Fixture fx;
    unsigned seed = 12345u;
    std::string dump_path;
    int64_t budget = 0;
    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto eq = a.find('=');
        const std::string key = a.substr(0, eq), val = a.substr(eq + 1);
        if (key == "h") fx.h = int(kb(val.c_str()));
        else if (key == "kvh") fx.kvh = int(kb(val.c_str()));
        else if (key == "d") fx.d = int(kb(val.c_str()));
        else if (key == "ext") fx.ext = int(kb(val.c_str()));
        else if (key == "T") fx.T = int(kb(val.c_str()));
        else if (key == "S") fx.S = int(kb(val.c_str()));
        else if (key == "budget") budget = kb(val.c_str());
        else if (key == "seed") seed = unsigned(kb(val.c_str()));
        else if (key == "dump") dump_path = val;
        else if (key == "sm")
        {
            shim_ctl().sm_major = int(strtol(val.substr(0, val.size() - 1).c_str(), nullptr, 10));
            shim_ctl().sm_minor = int(strtol(val.substr(val.size() - 1).c_str(), nullptr, 10));
        }
        else if (key == "attrs_fail") shim_ctl().attrs_fail = true;
        else if (key == "setattr_refuse") shim_ctl().setattr_refuses = true;
        else if (key == "q0") g_q0 = true;            // fixture knobs for the fence's isolation arms
        else if (key == "rel0") g_rel0 = true;
        else if (key == "sink0") g_sink0 = true;
        else if (key == "smem_optin") shim_ctl().smem_optin = size_t(kb(val.c_str()));
        else if (key == "smem_per_sm") shim_ctl().smem_per_sm = size_t(kb(val.c_str()));
    }
    fx.build(seed);
    if (g_q0) std::memset(fx.q.p(), 0, fx.q.size());
    if (g_rel0) std::memset(fx.rel.p(), 0, fx.rel.size());
    if (g_sink0) std::memset(fx.sink.p(), 0, fx.sink.size());

    if (scene == "plan")
    {
        print_plan(exl3_inkling_fa_plan(fx.h, fx.kvh, fx.d, fx.ext, fx.T, fx.S, budget, 0));
        return 0;
    }
    if (scene == "run" || scene == "decline")
    {
        const int64_t rc = exl3_inkling_fa_prefill(fx.tq, fx.tk, fx.tv, fx.tre, fx.tsink,
                                                   fx.tqpos, fx.tkpos, fx.to, fx.tnfin,
                                                   fx.scale, fx.ext, budget);
        std::printf("RESULT launched=%lld reason=%s nfin=%d\n", (long long) rc,
                    exl3_inkling_fa_reason().empty() ? "-" : exl3_inkling_fa_reason().c_str(),
                    ((int32_t*) fx.nfin.p())[0]);
        std::printf("CALL_COUNT launch=%d setattr=%d attrs=%d\n",
                    shim_call_count("launch"), shim_call_count("cudaFuncSetAttribute"),
                    shim_call_count("cudaDeviceGetAttribute"));
        if (!dump_path.empty()) fx.dump(dump_path.c_str());
        return 0;
    }
    if (scene == "dtype_bad")       // q handed in as fp32: the layer must refuse, not run
    {
        fx.tq = at::Tensor(fx.q.p(), {fx.h, fx.T, fx.d}, {int64_t(fx.T) * fx.d, fx.d, 1}, at::kFloat);
        const int64_t rc = exl3_inkling_fa_prefill(fx.tq, fx.tk, fx.tv, fx.tre, fx.tsink,
                                                   fx.tqpos, fx.tkpos, fx.to, fx.tnfin,
                                                   fx.scale, fx.ext, budget);
        std::printf("RESULT launched=%lld reason=%s\n", (long long) rc,
                    exl3_inkling_fa_reason().c_str());
        return 0;
    }
    if (scene == "stride_bad")      // k's row stride not 16-byte aligned (odd kvh*d): refuse
    {
        fx.tq = at::Tensor(fx.q.p(), {fx.h, fx.T, fx.d}, {int64_t(fx.T) * fx.d + 1, fx.d, 1},
                           at::kHalf);
        const int64_t rc = exl3_inkling_fa_prefill(fx.tq, fx.tk, fx.tv, fx.tre, fx.tsink,
                                                   fx.tqpos, fx.tkpos, fx.to, fx.tnfin,
                                                   fx.scale, fx.ext, budget);
        std::printf("RESULT launched=%lld reason=%s\n", (long long) rc,
                    exl3_inkling_fa_reason().c_str());
        return 0;
    }
    if (scene == "shape_bad")       // output shorter than q
    {
        fx.to = at::Tensor(fx.o.p(), {fx.h, fx.T - 1, fx.d}, {int64_t(fx.T) * fx.d, fx.d, 1},
                           at::kHalf);
        const int64_t rc = exl3_inkling_fa_prefill(fx.tq, fx.tk, fx.tv, fx.tre, fx.tsink,
                                                   fx.tqpos, fx.tkpos, fx.to, fx.tnfin,
                                                   fx.scale, fx.ext, budget);
        std::printf("RESULT launched=%lld reason=%s\n", (long long) rc,
                    exl3_inkling_fa_reason().c_str());
        return 0;
    }
    std::printf("RESULT error=unknown-scenario\n");
    return 2;
}