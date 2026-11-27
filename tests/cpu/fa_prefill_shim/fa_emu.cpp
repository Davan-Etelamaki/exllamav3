// CUDA execution shim: runs the production kernel body of
// exllamav3/exllamav3_ext/inkling/fa_prefill.cu on a CPU — one block at a time, one SIMT thread at
// a time, cooperatively scheduled with ucontexts — so that
// tests/cpu/inkling_fa_prefill_fence.py EXECUTES the real kernel (the fragment choreography, the
// in-loop relative-bias fold, the sink-seeded online softmax, the tile predicates, the pipeline
// barriers) instead of judging a transcription of it.
//
// Determinism: the scheduler walks threads in ascending tid order, and every cross-lane operation
// (shuffle, warp barrier, block barrier, mma, ldmatrix) publishes its inputs and then waits for the
// whole participating group, so an emulation race cannot hide a real one and a wrong fragment
// layout cannot cancel out.
//
// Instruction models, from the PTX ISA (mma.sync.aligned.m16n8k8, ldmatrix.m8n8.x4, cp.async.cg),
// with lane = 32*w + r, gid = r>>2, tig = r&3:
//   mma A   a0={A[gid][2*tig], A[gid][2*tig+1]}  a1={A[gid+8][2*tig], A[gid+8][2*tig+1]}
//   mma B   b0={B[2*tig][gid], B[2*tig+1][gid]}                     (.col: the k axis is the pair)
//   mma C   {C[gid][2*tig], C[gid][2*tig+1], C[gid+8][2*tig], C[gid+8][2*tig+1]}
//   ldmatrix x4: address group g (lanes 8g..8g+7) supplies row (lane&7) of 8x8 matrix g (16 B per
//                row, each lane's own address is that row's base — the swizzle is honoured because
//                the rows come from the lanes' addresses, not from a recomputed offset);
//                output register g of lane r is {M_g[r>>2][2*(r&3)], M_g[r>>2][2*(r&3)+1]}, with
//                .trans {M_g[2*(r&3)][r>>2], M_g[2*(r&3)+1][r>>2]}.
//   cp.async 16 B global->shared; with src-size 0 the destination is ZERO-FILLED (which is what
//                keeps a tile row that runs off the end of the tensor out of the fp16 products).
// Model limit, stated: the fp16-accumulate mma rounds once per call rather than per k-term, the way
// no public document specifies the hardware's internal order; the difference is bounded by eight
// fp16 roundings and the fence's tolerance, which is two orders above it.

#include "../../../exllamav3/exllamav3_ext/inkling/fa_prefill.cuh"
#include "../../../exllamav3/exllamav3_ext/inkling/fa_prims.cuh"
#include "cuda_runtime.h"
#include "shim.h"

#include <ucontext.h>
#include <cstdarg>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define EM_MAX_THREADS 1024
#define EM_STACK_BYTES (1024 * 1024)
#define EM_SMEM_BYTES (256 * 1024)
#define EM_MAX_WARPS (EM_MAX_THREADS / 32)

namespace
{
enum { EM_READY = 0, EM_BLOCKED = 1, EM_DONE = 2 };

struct Barrier
{
    int count = 0;
    int gen = 0;                    // one release == one generation
    const char* name = "?";
};

struct EmThread
{
    ucontext_t ctx;
    unsigned char* stack = nullptr;
    int state = EM_DONE;
    Barrier* waiting = nullptr;     // the barrier this thread is parked on, and which generation
    int wait_gen = 0;
};

EmThread g_t[EM_MAX_THREADS];
ucontext_t g_sched;
int g_cur = 0;                      // tid of the thread that is actually running
int g_n = 0;                        // threads in the current block
Barrier g_blk;                      // __syncthreads
Barrier g_warp[EM_MAX_WARPS];       // __syncwarp, one per warp
int g_wsize[EM_MAX_WARPS];
int g_nwarps = 0;

float g_pub[EM_MAX_WARPS][32];      // shuffle staging
struct Frag { uint32_t a0, a1, b; uint32_t addr; };
Frag g_frag[EM_MAX_WARPS][32];      // mma / ldmatrix publish slots

int g_grid_x = 0, g_grid_y = 0, g_threads = 0, g_bx = 0, g_by = 0;
unsigned char g_smem[EM_SMEM_BYTES];
void (*g_fn)(FaPrefillArgs) = nullptr;
FaPrefillArgs g_arg;

void em_to_sched() { swapcontext(&g_t[g_cur].ctx, &g_sched); }

void em_trampoline()
{
    const int self = g_cur;
    g_fn(g_arg);
    g_t[self].state = EM_DONE;
    em_to_sched();
}

// One barrier, one generation. The releaser wakes exactly the threads parked on THIS barrier and
// generation — a block barrier must never release a thread that is waiting for its warp, which is
// how the first cut of this scheduler deadlocked a single warp at 16 alive threads.
void em_barrier_wait(Barrier& b, int participants)
{
    const int self = g_cur;              // g_cur changes under the scheduler: pin our own tid
    const int my_gen = b.gen;
    if (++b.count >= participants)
    {
        b.count = 0;
        ++b.gen;
        for (int i = 0; i < g_n; ++i)
            if (g_t[i].waiting == &b && g_t[i].wait_gen == my_gen)
            {
                g_t[i].waiting = nullptr;
                g_t[i].state = EM_READY;
            }
        return;                                  // the releaser keeps running; peers run next scan
    }
    g_t[self].waiting = &b;
    g_t[self].wait_gen = my_gen;
    g_t[self].state = EM_BLOCKED;
#ifdef EM_DEBUG
    if (b.count <= 3)
        std::fprintf(stderr, "PARK tid %d warp %d %s count=%d/%d gen=%d\n", self, self >> 5,
                     b.name, b.count, participants, my_gen);
#endif
    em_to_sched();
    while (b.gen == my_gen)                      // woken early? park again until the release lands
    {
        g_t[self].state = EM_BLOCKED;
        em_to_sched();
    }
    g_t[self].waiting = nullptr;
}

void em_warp_barrier()
{
    const int w = g_cur >> 5;
    if (w >= g_nwarps) return;
    em_barrier_wait(g_warp[w], g_wsize[w]);
}

void em_block_barrier() { em_barrier_wait(g_blk, g_n); }
}  // namespace

// ------------------------------------------------------------------ fp16 codec

float em_f16_to_f32(uint16_t h)
{
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t man = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0)
    {
        if (man == 0) bits = sign;
        else
        {
            int e = 0;
            while ((man & 0x400u) == 0) { ++e; man <<= 1; }
            man &= 0x3ffu;
            bits = sign | (uint32_t(127 - 15 - e) << 23) | (man << 13);
        }
    }
    else if (exp == 31) bits = sign | 0x7f800000u | (man << 13);
    else bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint16_t em_f32_to_f16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t man = bits & 0x7fffffu;
    if (exp == 0xff) return uint16_t(sign | 0x7c00u | (man ? 0x200u : 0u));
    int e = int(exp) - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7c00u);
    if (e <= 0)
    {
        man |= 0x800000u;
        const int shift = 14 - e;
        if (shift > 25) return uint16_t(sign);
        uint32_t m = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (m & 1u))) ++m;
        return uint16_t(sign | m);
    }
    uint32_t m = man >> 13;
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u)))
    {
        ++m;
        if (m == 0x400u) { m = 0; ++e; if (e >= 31) return uint16_t(sign | 0x7c00u); }
    }
    return uint16_t(sign | (uint32_t(e) << 10) | (m & 0x3ffu));
}

namespace
{
inline float f16_at(uint32_t packed, int j)   // j = 0 low half, 1 high half
{
    const uint16_t h = j ? uint16_t(packed >> 16) : uint16_t(packed & 0xffffu);
    return em_f16_to_f32(h);
}
}  // namespace

// ------------------------------------------------------------------ execution model API

Dim3 em_thread_idx() { return Dim3{ uint32_t(g_cur), 0u, 1u }; }
Dim3 em_block_idx() { return Dim3{ uint32_t(g_bx), uint32_t(g_by), 1u }; }
Dim3 em_block_dim() { return Dim3{ uint32_t(g_threads), 1u, 1u }; }
Dim3 em_grid_dim() { return Dim3{ uint32_t(g_grid_x), uint32_t(g_grid_y), 1u }; }

void em_syncthreads() { em_block_barrier(); }
void em_syncwarp() { em_warp_barrier(); }

uint8_t* em_smem_base() { return g_smem; }
uint32_t em_smem_addr(const void* p) { return uint32_t((unsigned char*) p - g_smem); }

void em_cp_async16(void* smem_dst, const void* src, bool pred)
{
    if (pred) std::memcpy(smem_dst, src, 16);
    else std::memset(smem_dst, 0, 16);
}

void em_atomic_add_i32(int32_t* p, int32_t v) { *p += v; }

uint32_t em_pack_h2(float x, float y)
{
    uint16_t h[2] = { em_f32_to_f16(x), em_f32_to_f16(y) };
    return uint32_t(h[0]) | (uint32_t(h[1]) << 16);
}

float2 em_h2_to_f2(uint32_t h)
{
    float2 r;
    r.x = f16_at(h, 0);
    r.y = f16_at(h, 1);
    return r;
}

float em_shfl_xor(float v, int lane_mask)
{
    const int w = g_cur >> 5, l = g_cur & 31;
    g_pub[w][l] = v;
    em_warp_barrier();
    const float r = g_pub[w][l ^ lane_mask];
    em_warp_barrier();
    return r;
}

void em_ldsm_x4(uint32_t out[4], uint32_t saddr, int transpose)
{
    const int w = g_cur >> 5, l = g_cur & 31;
    g_frag[w][l].addr = saddr;              // each lane publishes the row base it addresses
    em_warp_barrier();

    uint16_t m[4][8][8];
    for (int g = 0; g < 4; ++g)
        for (int rr = 0; rr < 8; ++rr)
        {
            const uint32_t row = g_frag[w][g * 8 + rr].addr;
            const uint16_t* src = (const uint16_t*) (g_smem + row);
            for (int cc = 0; cc < 8; ++cc) m[g][rr][cc] = src[cc];
        }
    em_warp_barrier();                        // the next call may overwrite the address slots

    for (int g = 0; g < 4; ++g)
    {
        uint16_t p[2];
        if (!transpose) { p[0] = m[g][l >> 2][2 * (l & 3)]; p[1] = m[g][l >> 2][2 * (l & 3) + 1]; }
        else { p[0] = m[g][2 * (l & 3)][l >> 2]; p[1] = m[g][2 * (l & 3) + 1][l >> 2]; }
        out[g] = uint32_t(p[0]) | (uint32_t(p[1]) << 16);
    }
}

void em_mma_1688_f32(float c[4], uint32_t a0, uint32_t a1, uint32_t b)
{
    const int w = g_cur >> 5, l = g_cur & 31;
    g_frag[w][l].a0 = a0; g_frag[w][l].a1 = a1; g_frag[w][l].b = b;
    em_warp_barrier();

    const int gid = l >> 2, tig = l & 3;
    float acc[4] = { c[0], c[1], c[2], c[3] };
    for (int i = 0; i < 4; ++i)
    {
        const int row = gid + (i >= 2 ? 8 : 0);
        const int col = 2 * tig + (i & 1);
        float sum = 0.0f;
        for (int k = 0; k < 8; ++k)
        {
            // A[row][k]: lane (row&7)*4 + (k>>1); a0 for rows 0..7, a1 for rows 8..15
            const Frag& fa = g_frag[w][(row & 7) * 4 + (k >> 1)];
            const float a = f16_at(row >= 8 ? fa.a1 : fa.a0, k & 1);
            // B[k][col]: lane col*4 + (k>>1), element k&1 of its b register
            const Frag& fb = g_frag[w][col * 4 + (k >> 1)];
            const float bv = f16_at(fb.b, k & 1);
            sum += a * bv;
        }
        acc[i] += sum;
    }
    em_warp_barrier();
    c[0] = acc[0]; c[1] = acc[1]; c[2] = acc[2]; c[3] = acc[3];
}

void em_mma_1688_f16(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t b)
{
    const uint16_t* cp = (const uint16_t*) c;
    float cf[4] = { em_f16_to_f32(cp[0]), em_f16_to_f32(cp[1]),
                    em_f16_to_f32(cp[2]), em_f16_to_f32(cp[3]) };
    em_mma_1688_f32(cf, a0, a1, b);
    uint16_t* op = (uint16_t*) c;
    for (int i = 0; i < 4; ++i) op[i] = em_f32_to_f16(cf[i]);   // the accumulator is fp16
}

// ------------------------------------------------------------------ block / grid driver

void em_launch_kernel(void (*fn)(FaPrefillArgs), const FaPrefillArgs* args,
                      int grid_x, int grid_y, int threads, size_t smem_bytes)
{
    shim_record(("launch kernel=inkling_fa_prefill_kernel grid=(" + std::to_string(grid_x) + "," +
                 std::to_string(grid_y) + ") block=" + std::to_string(threads) +
                 " smem=" + std::to_string(smem_bytes) +
                 " stages=" + std::to_string(args->stages) +
                 " grp=" + std::to_string(args->grp)).c_str());
    if (threads > EM_MAX_THREADS || smem_bytes > EM_SMEM_BYTES)
    {
        std::fprintf(stderr, "INKLING-FLASHEMU request above shim capacity\n");
        std::exit(3);
    }
    g_fn = fn;
    g_arg = *args;
    g_grid_x = grid_x;
    g_grid_y = grid_y;
    g_threads = threads;
    g_n = threads;                                // barrier participant count and wake-scan bound
    g_nwarps = (threads + 31) / 32;

    for (int by = 0; by < grid_y; ++by)
        for (int bx = 0; bx < grid_x; ++bx)
        {
            g_bx = bx;
            g_by = by;
            g_blk = Barrier{0, 0, "__syncthreads"};
            for (int w = 0; w < EM_MAX_WARPS; ++w)
                g_warp[w] = Barrier{0, 0, "__syncwarp"};
            for (int w = 0; w < EM_MAX_WARPS; ++w)
                g_wsize[w] = w < g_nwarps ? (threads - w * 32 < 32 ? threads - w * 32 : 32) : 0;
            for (int i = 0; i < threads; ++i)
            {
                if (!g_t[i].stack) g_t[i].stack = (unsigned char*) std::malloc(EM_STACK_BYTES);
                g_t[i].state = EM_READY;
                getcontext(&g_t[i].ctx);
                g_t[i].ctx.uc_stack.ss_sp = g_t[i].stack;
                g_t[i].ctx.uc_stack.ss_size = EM_STACK_BYTES;
                g_t[i].ctx.uc_link = &g_sched;
                makecontext(&g_t[i].ctx, em_trampoline, 0);
            }
            int alive = threads;
            while (alive > 0)
            {
                bool progress = false;
                for (int i = 0; i < threads; ++i)
                {
                    if (g_t[i].state != EM_READY) continue;
                    g_cur = i;
                    swapcontext(&g_sched, &g_t[i].ctx);
                    progress = true;
                }
                alive = 0;
                for (int i = 0; i < threads; ++i) if (g_t[i].state != EM_DONE) ++alive;
                if (!progress && alive > 0)
                {
                    std::fprintf(stderr, "INKLING-FLASHEMU DEADLOCK alive=%d grid=(%d,%d)\n",
                                 alive, bx, by);
                    std::fprintf(stderr, "  blk count=%d gen=%d\n", g_blk.count, g_blk.gen);
                    for (int w = 0; w < g_nwarps; ++w)
                        std::fprintf(stderr, "  warp %d count=%d gen=%d\n", w,
                                     g_warp[w].count, g_warp[w].gen);
                    for (int i = 0; i < threads; ++i)
                        if (g_t[i].state != EM_DONE)
                            std::fprintf(stderr, "  tid %d state %d waiting %s gen %d\n", i,
                                         g_t[i].state,
                                         g_t[i].waiting ? g_t[i].waiting->name : "-",
                                         g_t[i].wait_gen);
                    std::exit(3);
                }
            }
        }
}

// ------------------------------------------------------------------ recorder + CUDA API

namespace
{
ShimCtl g_ctl;
#define SHIM_LOG_MAX 256
char g_calls[SHIM_LOG_MAX][320];
int g_call_n = 0;
}  // namespace

ShimCtl& shim_ctl() { return g_ctl; }

void shim_record(const char* call)
{
    if (g_call_n < SHIM_LOG_MAX) std::snprintf(g_calls[g_call_n++], sizeof(g_calls[0]), "%s", call);
    std::printf("CALL %s\n", call);
    std::fflush(stdout);
}

int shim_call_count(const char* prefix)
{
    int n = 0;
    for (int i = 0; i < g_call_n; ++i)
        if (std::strncmp(g_calls[i], prefix, std::strlen(prefix)) == 0) ++n;
    return n;
}

const char* shim_last_call(const char* prefix)
{
    const char* found = "";
    for (int i = 0; i < g_call_n; ++i)
        if (std::strncmp(g_calls[i], prefix, std::strlen(prefix)) == 0) found = g_calls[i];
    return found;
}

cudaError_t cudaGetDevice(int* device)
{
    *device = g_ctl.device;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)
{
    const ShimCtl& c = g_ctl;
    switch (attr)
    {
    case cudaDevAttrComputeCapabilityMajor: *value = c.sm_major; break;
    case cudaDevAttrComputeCapabilityMinor: *value = c.sm_minor; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin: *value = int(c.smem_optin); break;
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor: *value = int(c.smem_per_sm); break;
    case cudaDevAttrMaxSharedMemoryPerBlock: *value = int(c.smem_per_block); break;
    default: *value = 0; return cudaErrorInvalidValue;
    }
    shim_record(("cudaDeviceGetAttribute attr=" + std::to_string(int(attr)) + " device=" +
                 std::to_string(device) + " -> " + std::to_string(*value)).c_str());
    return g_ctl.attrs_fail ? cudaErrorInvalidValue : cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute attr, int amount)
{
    shim_record(("cudaFuncSetAttribute attr=" + std::to_string(int(attr)) +
                 " amount=" + std::to_string(amount)).c_str());
    return g_ctl.setattr_refuses ? cudaErrorInvalidValue : cudaSuccess;
}

cudaError_t cudaGetLastError() { return cudaSuccess; }