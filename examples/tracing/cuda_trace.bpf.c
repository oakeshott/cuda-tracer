// cuda_trace.bpf.c - eBPF uprobes for CUDA Driver/Runtime API
#include "vmlinux.h"
#include "cuda_evt.h"
#include "helper.h"
/* #include <linux/bpf.h> */

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
/* #include <linux/sched.h> */

char LICENSE[] SEC("license") = "GPL";

/* Additional NCCL event IDs kept local so the existing shared event ABI stays
 * unchanged.  The IDs are between the existing NCCL range and CPU events in
 * cuda_evt.h. */
#define F_NCCL_BROADCAST              57
#define F_NCCL_BCAST                  58
#define F_NCCL_REDUCE                 59
#define F_NCCL_ALL_TO_ALL             60
#define F_NCCL_GATHER                 61
#define F_NCCL_SCATTER                62
#define F_NCCL_GROUP_SIMULATE_END     63

#define BPF_CORE_READ(src, a, ...) ({                       \
    ___type((src), a, ##__VA_ARGS__) __r;                   \
    BPF_CORE_READ_INTO(&__r, (src), a, ##__VA_ARGS__);          \
    __r;                                    \
})

struct inflight_launch {
    __u64 func_ptr;
    __u32 grid_x, grid_y, grid_z;
    __u32 block_x, block_y, block_z;
    __u32 shared_mem;
    __u64 stream;
};

struct inflight_mem {
    __u64 bytes;
};

struct inflight_graph {
    __u64 graph_exec;
    __u64 stream;
};

struct inflight_memcpy_runtime {
    __u64 count;
    __s32 kind;
};

struct inflight_mem_async {
  __u64 bytes;
  __u64 stream;
};

struct inflight_memcpy_async {
  __u64 bytes;
  __u64 stream;
  __s32 kind;
};

// trace_id を hi/lo の u64×2 で持つ（OTel 32hex も入る）
struct trace_id { __u64 hi, lo; };

// TID -> trace_id の LRU
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 16384);
    __type(key, __u32);             // tid
    __type(value, struct trace_id); // trace_id
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} active_traces SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 24); // 16 MiB
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32); // tid
    __type(value, struct inflight_launch);
} inflight_launch_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32); // tid
    __type(value, struct inflight_graph);
} inflight_graph_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32);
    __type(value, struct inflight_mem);
} inflight_mem_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32);
    __type(value, struct inflight_memcpy_runtime);
} inflight_memcpy_rt_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u64);  // devptr
    __type(value, __u64); // size
} devptr_sz_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32); // tid
    __type(value, struct inflight_mem_async);
} inflight_mem_async_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32); // tid
    __type(value, struct inflight_memcpy_async);
} inflight_memcpy_async_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 8192);
    __type(key, __u64);   // event handle
    __type(value, __u64); // stream handle
} event_stream_map SEC(".maps");

struct inflight_free { __u64 dptr; };
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 8192);
    __type(key, __u32); // tid
    __type(value, struct inflight_free);
} inflight_free_map SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} cfg_pid_filter SEC(".maps");

struct cpu_state {
    __u64 last_on_ns;
    __u64 last_off_ns;
    __u64 last_wakeup_ns;
    __u64 futex_enter_ns;
};

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);              // tid
    __type(value, struct cpu_state);
} cpu_state_map SEC(".maps");

static __always_inline bool tid_pass(__u32 tid)
{
    __u32 key = 0, *flt = bpf_map_lookup_elem(&cfg_pid_filter, &key);
    __u32 v = flt ? *flt : 0;
    return (v == 0) || (v == tid);
}

static __always_inline __u32 get_pid(void) {
    return (__u32)(bpf_get_current_pid_tgid() >> 32);
}
static __always_inline __u32 get_tid(void) {
    return (__u32)(bpf_get_current_pid_tgid());
}

static __always_inline void fill_common(struct event *e, __u32 func) {
    e->ts_ns = bpf_ktime_get_ns();
    e->pid = get_pid();
    e->tid = get_tid();
    e->func = func;
    e->ret = 0;
    bpf_get_current_comm(&e->comm, sizeof(e->comm));

    struct task_struct *task;
    task = (struct task_struct *)bpf_get_current_task();
    if (!task)
      return;
    e->ppid = BPF_CORE_READ(task, real_parent, tgid);
}

static __always_inline void set_trace_id(struct event *e) {
    // pyhook publishes the host TID immediately before GPUModelRunner runs.
    // Parent PID cannot identify a request across Ray processes or batches.
    // A multi-request batch has its own UID; resolve its member request UIDs
    // through batch_uids. The active_traces map and event ABI are unchanged.
    __u32 tid = e->tid;
    struct trace_id *t = bpf_map_lookup_elem(&active_traces, &tid);
    if (t) { e->trace_hi = t->hi; e->trace_lo = t->lo; }
    else   { e->trace_hi = 0;    e->trace_lo = 0; }
}

/* ---------------- Driver API: cuMemAlloc_v2(void **dptr, size_t bytesize) ---------------- */
SEC("uprobe")
int BPF_KPROBE(cuMemAlloc_v2_enter)
{
    __u32 tid = get_tid();
    struct inflight_mem v = {};
    v.bytes = PT_REGS_PARM2(ctx);

    bpf_map_update_elem(&inflight_mem_map, &tid, &v, BPF_ANY);
    return 0;
}


SEC("uretprobe")
int BPF_KPROBE(cuMemAlloc_v2_exit)
{
    __u32 tid = get_tid();
    struct inflight_mem *vp = bpf_map_lookup_elem(&inflight_mem_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CU_MEMALLOC_V2);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);

    // 1引数目は void** dptr
    __u64 dptr_addr = PT_REGS_PARM1(ctx);
    __u64 devptr = 0;
    bpf_probe_read_user(&devptr, sizeof(devptr), (void *)dptr_addr);

    if (vp) {
        e->u.mem_ex.bytes = vp->bytes;
        e->u.mem_ex.dptr  = devptr;

        if (e->ret == 0 && devptr) {
            __u64 sz = vp->bytes;
            bpf_map_update_elem(&devptr_sz_map, &devptr, &sz, BPF_ANY);
        }
    }

    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_mem_map, &tid);
    return 0;
}

/* ---------------- Driver API: cuMemFree_v2(CUdeviceptr dptr) ---------------- */
SEC("uprobe")
int BPF_KPROBE(cuMemFree_v2_enter)
{
    __u32 tid = get_tid();
    struct inflight_free v = { .dptr = PT_REGS_PARM1(ctx) };
    bpf_map_update_elem(&inflight_free_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuMemFree_v2_exit)
{
    __u32 tid = get_tid();
    struct inflight_free *fp = bpf_map_lookup_elem(&inflight_free_map, &tid);

    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CU_MEMFREE_V2);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);

    __u64 bytes = 0, dptr = fp ? fp->dptr : 0;
    if (dptr) {
        __u64 *bp = bpf_map_lookup_elem(&devptr_sz_map, &dptr);
        if (bp) bytes = *bp;
        if (e->ret == 0 && bp) bpf_map_delete_elem(&devptr_sz_map, &dptr);
    }
    e->u.mem_ex.dptr  = dptr;
    e->u.mem_ex.bytes = bytes;

    bpf_ringbuf_submit(e, 0);
out:
    if (fp) bpf_map_delete_elem(&inflight_free_map, &tid);
    return 0;
}

/* ---------------- Driver API: cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) ---------------- */
SEC("uprobe")
int BPF_KPROBE(cuMemcpyHtoD_v2_enter)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;
    fill_common(e, F_CU_MEMCPY_HTOD_V2);
    set_trace_id(e);
    e->u.mem.bytes = PT_REGS_PARM3(ctx);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuMemcpyHtoD_v2_exit)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;
    fill_common(e, F_CU_MEMCPY_HTOD_V2);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

/* ---------------- Driver API: cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount) ---------------- */
SEC("uprobe")
int BPF_KPROBE(cuMemcpyDtoH_v2_enter)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;
    fill_common(e, F_CU_MEMCPY_DTOH_V2);
    set_trace_id(e);
    e->u.mem.bytes = PT_REGS_PARM3(ctx);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuMemcpyDtoH_v2_exit)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;
    fill_common(e, F_CU_MEMCPY_DTOH_V2);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

/* ---------------- Driver API: cuLaunchKernel(...) ----------------
 * CUresult cuLaunchKernel(CUfunction f,
 *   unsigned gridX, gridY, gridZ,
 *   unsigned blockX, blockY, blockZ,
 *   unsigned sharedMemBytes, CUstream hStream, void** kernelParams, void** extra)
 * The register/stack layout depends on the target architecture.
 */
static __always_inline __u64 read_user_u64(void *addr)
{
    __u64 v = 0;
    bpf_probe_read_user(&v, sizeof(v), addr);
    return v;
}

SEC("uprobe")
int BPF_KPROBE(cuLaunchKernel_enter)
{
    __u32 tid = get_tid();
    struct inflight_launch v = {};

    // regs
    v.func_ptr = PT_REGS_PARM1(ctx);
    v.grid_x   = (__u32)PT_REGS_PARM2(ctx);
    v.grid_y   = (__u32)PT_REGS_PARM3(ctx);
    v.grid_z   = (__u32)PT_REGS_PARM4(ctx);
    v.block_x  = (__u32)PT_REGS_PARM5(ctx);
    v.block_y  = (__u32)PT_REGS_PARM6(ctx);

#if defined(__TARGET_ARCH_x86)
    /* x86_64 SysV: SP + 8 is blockZ, +16 is sharedMem, +24 is stream. */
    __u64 sp = PT_REGS_SP(ctx);
    v.block_z   = (__u32)read_user_u64((void *)(sp + 8));
    v.shared_mem= (__u32)read_user_u64((void *)(sp + 16));
    v.stream    = read_user_u64((void *)(sp + 24));
#elif defined(__TARGET_ARCH_arm64)
    /* AAPCS64: blockZ and sharedMem are x6/x7; stream is stack argument 9. */
    v.block_z   = (__u32)PT_REGS_PARM7(ctx);
    v.shared_mem= (__u32)PT_REGS_PARM8(ctx);
    v.stream    = read_user_u64((void *)PT_REGS_SP(ctx));
#else
    v.block_z = 0;
    v.shared_mem = 0;
    v.stream = 0;
#endif

    bpf_map_update_elem(&inflight_launch_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuLaunchKernel_exit)
{
    __u32 tid = get_tid();
    struct inflight_launch *vp = bpf_map_lookup_elem(&inflight_launch_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;


    fill_common(e, F_CU_LAUNCH_KERNEL);
    set_trace_id(e);

    if (vp) {
        e->u.launch.func_ptr = vp->func_ptr;
        e->u.launch.grid_x = vp->grid_x;
        e->u.launch.grid_y = vp->grid_y;
        e->u.launch.grid_z = vp->grid_z;
        e->u.launch.block_x = vp->block_x;
        e->u.launch.block_y = vp->block_y;
        e->u.launch.block_z = vp->block_z;
        e->u.launch.shared_mem = vp->shared_mem;
        e->u.launch.stream = vp->stream;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_launch_map, &tid);
    return 0;
}

/* ---------------- Driver API: cuGraphLaunch(CUgraphExec, CUstream) ----------------
 * Used by vLLM (and other frameworks) to replay a captured CUDA graph each
 * decode/prefill step when CUDA graphs are enabled. One call replays many
 * kernels, so this is the launch event we see during steady-state inference.
 */
SEC("uprobe")
int BPF_KPROBE(cuGraphLaunch_enter)
{
    __u32 tid = get_tid();
    struct inflight_graph v = {};
    v.graph_exec = PT_REGS_PARM1(ctx);
    v.stream     = PT_REGS_PARM2(ctx);
    bpf_map_update_elem(&inflight_graph_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuGraphLaunch_exit)
{
    __u32 tid = get_tid();
    struct inflight_graph *vp = bpf_map_lookup_elem(&inflight_graph_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CU_GRAPH_LAUNCH);
    set_trace_id(e);

    if (vp) {
        e->u.graph.graph_exec = vp->graph_exec;
        e->u.graph.stream     = vp->stream;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_graph_map, &tid);
    return 0;
}

/* ---------------- Runtime API: cudaLaunchKernel(...) ----------------
 * cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
 *                              void **args, size_t sharedMem, cudaStream_t stream)
 * This (and its _ptsz variant) is the hot launch path for PyTorch/vLLM.
 * AAPCS64: dim3 (12B) is passed packed across two X registers, so:
 *   x0=func  x1={grid.x,grid.y} x2={grid.z} x3={block.x,block.y} x4={block.z}
 *   x5=args  x6=sharedMem       x7=stream
 */
SEC("uprobe")
int BPF_KPROBE(cudaLaunchKernel_enter)
{
    __u32 tid = get_tid();
    struct inflight_launch v = {};
    __u64 p1 = PT_REGS_PARM2(ctx);   /* grid.x | grid.y */
    __u64 p2 = PT_REGS_PARM3(ctx);   /* grid.z          */
    __u64 p3 = PT_REGS_PARM4(ctx);   /* block.x | block.y */
    __u64 p4 = PT_REGS_PARM5(ctx);   /* block.z         */

    v.func_ptr  = PT_REGS_PARM1(ctx);
    v.grid_x    = (__u32)p1;
    v.grid_y    = (__u32)(p1 >> 32);
    v.grid_z    = (__u32)p2;
    v.block_x   = (__u32)p3;
    v.block_y   = (__u32)(p3 >> 32);
    v.block_z   = (__u32)p4;
    v.shared_mem= (__u32)PT_REGS_PARM7(ctx);
    v.stream    = PT_REGS_PARM8(ctx);

    bpf_map_update_elem(&inflight_launch_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaLaunchKernel_exit)
{
    __u32 tid = get_tid();
    struct inflight_launch *vp = bpf_map_lookup_elem(&inflight_launch_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CUDA_LAUNCH_KERNEL);
    set_trace_id(e);

    if (vp) {
        e->u.launch.func_ptr = vp->func_ptr;
        e->u.launch.grid_x = vp->grid_x;
        e->u.launch.grid_y = vp->grid_y;
        e->u.launch.grid_z = vp->grid_z;
        e->u.launch.block_x = vp->block_x;
        e->u.launch.block_y = vp->block_y;
        e->u.launch.block_z = vp->block_z;
        e->u.launch.shared_mem = vp->shared_mem;
        e->u.launch.stream = vp->stream;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_launch_map, &tid);
    return 0;
}

/* ---------------- Runtime API: cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) ----
 * PyTorch CUDA-graph replay path (and its _ptsz variant).
 */
SEC("uprobe")
int BPF_KPROBE(cudaGraphLaunch_enter)
{
    __u32 tid = get_tid();
    struct inflight_graph v = {};
    v.graph_exec = PT_REGS_PARM1(ctx);
    v.stream     = PT_REGS_PARM2(ctx);
    bpf_map_update_elem(&inflight_graph_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaGraphLaunch_exit)
{
    __u32 tid = get_tid();
    struct inflight_graph *vp = bpf_map_lookup_elem(&inflight_graph_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CUDA_GRAPH_LAUNCH);
    set_trace_id(e);

    if (vp) {
        e->u.graph.graph_exec = vp->graph_exec;
        e->u.graph.stream     = vp->stream;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_graph_map, &tid);
    return 0;
}

/* ---------------- Runtime API (optional): cudaMalloc/cudaFree/cudaMemcpy ---------------- */
SEC("uprobe")
int BPF_KPROBE(cudaMalloc_enter)
{
    __u32 tid = get_tid();
    struct inflight_mem v = {};
    v.bytes = PT_REGS_PARM2(ctx);
    bpf_map_update_elem(&inflight_mem_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaMalloc_exit)
{
    __u32 tid = get_tid();
    struct inflight_mem *vp = bpf_map_lookup_elem(&inflight_mem_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CUDA_MALLOC);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);

    // 1引数目 void** devPtr
    __u64 devptr_addr = PT_REGS_PARM1(ctx);
    __u64 devptr = 0;
    bpf_probe_read_user(&devptr, sizeof(devptr), (void *)devptr_addr);

    if (vp) {
        e->u.mem_ex.bytes = vp->bytes;
        e->u.mem_ex.dptr  = devptr;
        if (e->ret == 0 && devptr) {
            __u64 sz = vp->bytes;
            bpf_map_update_elem(&devptr_sz_map, &devptr, &sz, BPF_ANY);
        }
    }
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_mem_map, &tid);
    return 0;
}

// cudaFree enter/exit
SEC("uprobe")
int BPF_KPROBE(cudaFree_enter)
{
    __u32 tid = get_tid();
    struct inflight_free v = { .dptr = PT_REGS_PARM1(ctx) };
    bpf_map_update_elem(&inflight_free_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaFree_exit)
{
    __u32 tid = get_tid();
    struct inflight_free *fp = bpf_map_lookup_elem(&inflight_free_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;

    fill_common(e, F_CUDA_FREE);
    set_trace_id(e);
    e->ret = PT_REGS_RC(ctx);

    __u64 bytes = 0, dptr = fp ? fp->dptr : 0;
    if (dptr) {
        __u64 *bp = bpf_map_lookup_elem(&devptr_sz_map, &dptr);
        if (bp) bytes = *bp;
        if (e->ret == 0 && bp) bpf_map_delete_elem(&devptr_sz_map, &dptr);
    }
    e->u.mem_ex.dptr  = dptr;
    e->u.mem_ex.bytes = bytes;

    bpf_ringbuf_submit(e, 0);
out:
    if (fp) bpf_map_delete_elem(&inflight_free_map, &tid);
    return 0;
}


SEC("uprobe")
int BPF_KPROBE(cudaMemcpy_enter)
{
    __u32 tid = get_tid();
    struct inflight_memcpy_runtime v = {};
    v.count = PT_REGS_PARM3(ctx);
    v.kind  = (__s32)PT_REGS_PARM4(ctx);
    bpf_map_update_elem(&inflight_memcpy_rt_map, &tid, &v, BPF_ANY);
    return 0;
}
SEC("uretprobe")
int BPF_KPROBE(cudaMemcpy_exit)
{
    __u32 tid = get_tid();
    struct inflight_memcpy_runtime *vp = bpf_map_lookup_elem(&inflight_memcpy_rt_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;
    fill_common(e, F_CUDA_MEMCPY);
    set_trace_id(e);
    if (vp) {
        e->u.memcpy_runtime.count = vp->count;
        e->u.memcpy_runtime.kind  = vp->kind;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_memcpy_rt_map, &tid);
    return 0;
}

/* Runtime API: cudaMemcpyAsync(dst, src, count, kind, stream)  (+ _ptsz variant)
 * This is the async transfer path PyTorch/vLLM use. Direction is in `kind`:
 *   1=HostToDevice 2=DeviceToHost 3=DeviceToDevice
 * ABI: x0=dst x1=src x2=count(PARM3) x3=kind(PARM4) x4=stream(PARM5)
 */
SEC("uprobe")
int BPF_KPROBE(cudaMemcpyAsync_enter)
{
    __u32 tid = get_tid();
    struct inflight_memcpy_async v = {
        .bytes  = PT_REGS_PARM3(ctx),
        .kind   = (__s32)PT_REGS_PARM4(ctx),
        .stream = PT_REGS_PARM5(ctx),
    };
    bpf_map_update_elem(&inflight_memcpy_async_map, &tid, &v, BPF_ANY);
    return 0;
}
SEC("uretprobe")
int BPF_KPROBE(cudaMemcpyAsync_exit)
{
    __u32 tid = get_tid();
    struct inflight_memcpy_async *vp = bpf_map_lookup_elem(&inflight_memcpy_async_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;
    fill_common(e, F_CUDA_MEMCPY_ASYNC);
    set_trace_id(e);
    if (vp) {
        e->u.memcpy_async.bytes  = vp->bytes;
        e->u.memcpy_async.stream = vp->stream;
        e->u.memcpy_async.kind   = vp->kind;
    }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_memcpy_async_map, &tid);
    return 0;
}

/* cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) */
SEC("uprobe")
int BPF_KPROBE(cuMemcpyHtoDAsync_v2_enter)
{
    __u32 tid = get_tid();
    struct inflight_mem_async v = {
        .bytes  = PT_REGS_PARM3(ctx),
        .stream = PT_REGS_PARM4(ctx),
    };
    bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuMemcpyHtoDAsync_v2_exit)
{
    __u32 tid = get_tid();
    struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;
    fill_common(e, F_CU_MEMCPY_HTOD_ASYNC_V2);
    set_trace_id(e);
    if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid);
    return 0;
}

/* cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) */
SEC("uprobe")
int BPF_KPROBE(cuMemcpyDtoHAsync_v2_enter)
{
    __u32 tid = get_tid();
    struct inflight_mem_async v = {
        .bytes  = PT_REGS_PARM3(ctx),
        .stream = PT_REGS_PARM4(ctx),
    };
    bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY);
    return 0;
}
SEC("uretprobe")
int BPF_KPROBE(cuMemcpyDtoHAsync_v2_exit)
{
    __u32 tid = get_tid();
    struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;
    fill_common(e, F_CU_MEMCPY_DTOH_ASYNC_V2);
    set_trace_id(e);
    if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid);
    return 0;
}

/* cuMemcpyDtoDAsync_v2(dst, src, bytes, stream) */
SEC("uprobe")
int BPF_KPROBE(cuMemcpyDtoDAsync_v2_enter)
{
    __u32 tid = get_tid();
    struct inflight_mem_async v = {
        .bytes  = PT_REGS_PARM3(ctx),
        .stream = PT_REGS_PARM4(ctx),
    };
    bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY);
    return 0;
}
SEC("uretprobe")
int BPF_KPROBE(cuMemcpyDtoDAsync_v2_exit)
{
    __u32 tid = get_tid();
    struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid);
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) goto out;
    fill_common(e, F_CU_MEMCPY_DTOD_ASYNC_V2);
    set_trace_id(e);
    if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; }
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
out:
    if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid);
    return 0;
}

/* ★ stream 同期: cuStreamSynchronize(stream) */
SEC("uprobe")
int BPF_KPROBE(cuStreamSynchronize_enter) { return 0; }
SEC("uretprobe")
int BPF_KPROBE(cuStreamSynchronize_exit)
{
  bpf_printk("cuStreamSynchronize");
  struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
  if (!e) return 0;
  fill_common(e, F_CU_STREAM_SYNCHRONIZE);
  set_trace_id(e);
  e->u.fence.stream = PT_REGS_PARM1(ctx);
  e->ret = PT_REGS_RC(ctx);
  bpf_ringbuf_submit(e, 0);
  return 0;
}

/* runtime: cudaStreamSynchronize(stream) */
SEC("uprobe")
int BPF_KPROBE(cudaStreamSynchronize_enter) { return 0; }
/* ★ runtime: cudaStreamSynchronize(stream) */
SEC("uretprobe")
int BPF_KPROBE(cudaStreamSynchronize_exit)
{
  bpf_printk("cudaStreamSynchronize");
  struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
  if (!e) return 0;
  fill_common(e, F_CUDA_STREAM_SYNCHRONIZE);
  set_trace_id(e);
  e->u.fence.stream = PT_REGS_PARM1(ctx);
  e->ret = PT_REGS_RC(ctx);
  bpf_ringbuf_submit(e, 0);
  return 0;
}

/* ★ ctx/device 同期：ストリーム全体をフェンス（stream=0） */
/* driver: cuCtxSynchronize() */
SEC("uprobe")
int BPF_KPROBE(cuCtxSynchronize_enter) { return 0; }
SEC("uretprobe")
int BPF_KPROBE(cuCtxSynchronize_exit)
{
  bpf_printk("cuCtxSynchronize");
  struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
  if (!e) return 0;
  fill_common(e, F_CU_CTX_SYNCHRONIZE);
  set_trace_id(e);
  e->u.fence.stream = 0;
  e->ret = PT_REGS_RC(ctx);
  bpf_ringbuf_submit(e, 0);
  return 0;
}

/* runtime: cudaDeviceSynchronize() */
SEC("uprobe")
int BPF_KPROBE(cudaDeviceSynchronize_enter) { return 0; }

SEC("uretprobe")
int BPF_KPROBE(cudaDeviceSynchronize_exit)
{
  bpf_printk("cuDeviceSynchronize");
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;
    fill_common(e, F_CUDA_DEVICE_SYNCHRONIZE);
    set_trace_id(e);
    e->u.fence.stream = 0;
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);
    return 0;
}

/* driver: cuEventRecord(event, stream) */
SEC("uprobe")
int BPF_KPROBE(cuEventRecord_enter)
{
    __u64 ev     = PT_REGS_PARM1(ctx);
    __u64 stream = PT_REGS_PARM2(ctx);
    if (ev)
        bpf_map_update_elem(&event_stream_map, &ev, &stream, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuEventRecord_exit)
{
    return 0;
}

/* runtime: cudaEventRecord(event, stream) */
SEC("uprobe")
int BPF_KPROBE(cudaEventRecord_enter)
{
    __u64 ev     = PT_REGS_PARM1(ctx);
    __u64 stream = PT_REGS_PARM2(ctx);
    if (ev)
        bpf_map_update_elem(&event_stream_map, &ev, &stream, BPF_ANY);
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaEventRecord_exit)
{
    return 0;
}

/* driver: cuEventSynchronize(event) */
SEC("uprobe")
int BPF_KPROBE(cuEventSynchronize_enter)
{
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cuEventSynchronize_exit)
{
    __u64 ev = PT_REGS_PARM1(ctx);
    __u64 *p_stream = bpf_map_lookup_elem(&event_stream_map, &ev);

    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    fill_common(e, F_CU_EVENT_SYNCHRONIZE);
    set_trace_id(e);
    e->u.fence.stream = p_stream ? *p_stream : 0;  // stream が取れなければ 0
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);

    return 0;
}

/* runtime: cudaEventSynchronize(event) */
SEC("uprobe")
int BPF_KPROBE(cudaEventSynchronize_enter)
{
    return 0;
}

SEC("uretprobe")
int BPF_KPROBE(cudaEventSynchronize_exit)
{
    __u64 ev = PT_REGS_PARM1(ctx);
    __u64 *p_stream = bpf_map_lookup_elem(&event_stream_map, &ev);

    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return 0;

    fill_common(e, F_CUDA_EVENT_SYNCHRONIZE);
    set_trace_id(e);
    e->u.fence.stream = p_stream ? *p_stream : 0;
    e->ret = PT_REGS_RC(ctx);
    bpf_ringbuf_submit(e, 0);

    return 0;
}

/* #<{(| -------- sched_switch: on/off-CPU を出す -------- |)}># */
/* SEC("tracepoint/sched/sched_switch") */
/* int tp_sched_switch(struct trace_event_raw_sched_switch *ctx) */
/* { */
/*     __u64 now = bpf_ktime_get_ns(); */
/*     __u32 cpu = bpf_get_smp_processor_id(); */
/*  */
/*     __u32 prev_pid   = ctx->prev_pid; */
/*     __u32 prev_state = ctx->prev_state; */
/*     if (tid_pass(prev_pid)) { */
/*         struct cpu_state *st = bpf_map_lookup_elem(&cpu_state_map, &prev_pid); */
/*         if (!st) { */
/*             struct cpu_state init = {}; */
/*             bpf_map_update_elem(&cpu_state_map, &prev_pid, &init, BPF_ANY); */
/*             st = bpf_map_lookup_elem(&cpu_state_map, &prev_pid); */
/*         } */
/*         if (st) { */
/*             #<{(| on-CPU 終了 = now - last_on_ns |)}># */
/*             if (st->last_on_ns) { */
/*                 struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*                 if (e) { */
/*                     fill_common(e, F_CPU_OFFCPU); */
/*                     set_trace_id(e); */
/*                     e->u.cpu.cpu    = cpu; */
/*                     e->u.cpu.state  = prev_state; */
/*                     e->u.cpu.dur_ns = now - st->last_on_ns; */
/*                     e->u.cpu.aux_ns = 0; #<{(| 予備 |)}># */
/*                     bpf_ringbuf_submit(e, 0); */
/*                 } */
/*             } */
/*             st->last_off_ns = now; */
/*         } */
/*     } */
/*  */
/*     #<{(| next (= これから乗る側) |)}># */
/*     __u32 next_pid = ctx->next_pid; */
/*     if (tid_pass(next_pid)) { */
/*         struct cpu_state *st = bpf_map_lookup_elem(&cpu_state_map, &next_pid); */
/*         if (!st) { */
/*             struct cpu_state init = {}; */
/*             bpf_map_update_elem(&cpu_state_map, &next_pid, &init, BPF_ANY); */
/*             st = bpf_map_lookup_elem(&cpu_state_map, &next_pid); */
/*         } */
/*         if (st) { */
/*             #<{(| runqueue滞在時間 = now - last_off_ns, さらに wakeup→switch-in の遅延も |)}># */
/*             __u64 offcpu_ns = st->last_off_ns ? (now - st->last_off_ns) : 0; */
/*             __u64 rq_lat_ns = st->last_wakeup_ns ? (now - st->last_wakeup_ns) : 0; */
/*  */
/*             struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*             if (e) { */
/*                 fill_common(e, F_CPU_ONCPU); */
/*                 set_trace_id(e); */
/*                 e->u.cpu.cpu    = cpu; */
/*                 e->u.cpu.state  = 0; */
/*                 e->u.cpu.dur_ns = offcpu_ns;   #<{(| off-CPU(=runqueue滞在) |)}># */
/*                 e->u.cpu.aux_ns = rq_lat_ns;   #<{(| wakeup→onCPU のレイテンシ |)}># */
/*                 bpf_ringbuf_submit(e, 0); */
/*             } */
/*             st->last_on_ns = now; */
/*             st->last_wakeup_ns = 0; #<{(| 消費したのでクリア |)}># */
/*         } */
/*     } */
/*     return 0; */
/* } */
/*  */
/* #<{(| -------- sched_wakeup: runnable になった時刻を記録 -------- |)}># */
/* SEC("tracepoint/sched/sched_wakeup") */
/* int tp_sched_wakeup(struct trace_event_raw_sched_wakeup *ctx) */
/* { */
/*     __u32 pid = ctx->pid; */
/*     if (!tid_pass(pid)) return 0; */
/*     __u64 now = bpf_ktime_get_ns(); */
/*     struct cpu_state *st = bpf_map_lookup_elem(&cpu_state_map, &pid); */
/*     if (!st) { */
/*         struct cpu_state init = {}; */
/*         bpf_map_update_elem(&cpu_state_map, &pid, &init, BPF_ANY); */
/*         st = bpf_map_lookup_elem(&cpu_state_map, &pid); */
/*     } */
/*     if (st) { */
/*         st->last_wakeup_ns = now; */
/*         #<{(| 任意: wakeup イベント自体も流す |)}># */
/*         struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*         if (e) { */
/*             fill_common(e, F_CPU_WAKEUP); */
/*             set_trace_id(e); */
/*             e->u.cpu.cpu = bpf_get_smp_processor_id(); */
/*             e->u.cpu.state = 0; */
/*             e->u.cpu.dur_ns = 0; */
/*             e->u.cpu.aux_ns = 0; */
/*             bpf_ringbuf_submit(e, 0); */
/*         } */
/*     } */
/*     return 0; */
/* } */
/*  */
/* #<{(| -------- futex 待ち時間（syscalls） -------- |)}># */
/* SEC("tracepoint/syscalls/sys_enter_futex") */
/* int tp_enter_futex(struct trace_event_raw_sys_enter *ctx) */
/* { */
/*     __u32 tid = get_tid(); */
/*     if (!tid_pass(tid)) return 0; */
/*     struct cpu_state *st = bpf_map_lookup_elem(&cpu_state_map, &tid); */
/*     if (!st) { */
/*         struct cpu_state init = {}; */
/*         bpf_map_update_elem(&cpu_state_map, &tid, &init, BPF_ANY); */
/*         st = bpf_map_lookup_elem(&cpu_state_map, &tid); */
/*     } */
/*     if (st) st->futex_enter_ns = bpf_ktime_get_ns(); */
/*     return 0; */
/* } */
/*  */
/* SEC("tracepoint/syscalls/sys_exit_futex") */
/* int tp_exit_futex(struct trace_event_raw_sys_exit *ctx) */
/* { */
/*     __u32 tid = get_tid(); */
/*     if (!tid_pass(tid)) return 0; */
/*     struct cpu_state *st = bpf_map_lookup_elem(&cpu_state_map, &tid); */
/*     if (!st || !st->futex_enter_ns) return 0; */
/*  */
/*     __u64 now = bpf_ktime_get_ns(); */
/*     __u64 dur = now - st->futex_enter_ns; */
/*     st->futex_enter_ns = 0; */
/*  */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CPU_FUTEX_WAIT); */
/*     set_trace_id(e); */
/*     e->u.cpu.cpu    = bpf_get_smp_processor_id(); */
/*     e->u.cpu.state  = 0; */
/*     e->u.cpu.dur_ns = dur;    #<{(| futex 待ち時間 |)}># */
/*     e->u.cpu.aux_ns = 0; */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */

/* ---------------- NCCL collectives (enter-only) ----------------
 * Signatures (libnccl, AAPCS64 x0..x7 = PARM1..PARM8):
 *   ncclSend       (buf, count, dtype, peer, comm, stream)
 *   ncclRecv       (buf, count, dtype, peer, comm, stream)
 *   ncclAllReduce  (sbuf, rbuf, count, dtype, op, comm, stream)
 *   ncclBroadcast  (sbuf, rbuf, count, dtype, root, comm, stream)
 *   ncclBcast      (buf, count, dtype, root, comm, stream)
 *   ncclReduce     (sbuf, rbuf, count, dtype, op, root, comm, stream)
 *   ncclAllGather  (sbuf, rbuf, sendcount, dtype, comm, stream)
 *   ncclReduceScatter(sbuf, rbuf, recvcount, dtype, op, comm, stream)
 *   ncclAlltoAll   (sbuf, rbuf, count, dtype, comm, stream)
 *   ncclGather     (sbuf, rbuf, count, dtype, root, comm, stream)
 *   ncclScatter    (sbuf, rbuf, count, dtype, root, comm, stream)
 *   ncclGroupStart () / ncclGroupEnd ()
 *   ncclGroupSimulateEnd (simInfo)
 *
 * The nccl event's peer field stores the peer rank for Send/Recv and the root
 * rank for root-based collectives. */
static __always_inline void emit_nccl(struct pt_regs *ctx, __u32 fid,
                                      __u64 count, __u64 stream,
                                      __s32 dtype, __s32 peer,
                                      __u64 comm, __s32 redop)
{
    struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
    if (!e) return;
    fill_common(e, fid);
    set_trace_id(e);
    e->u.nccl.count  = count;
    e->u.nccl.stream = stream;
    e->u.nccl.dtype  = dtype;
    e->u.nccl.peer   = peer;
    e->u.nccl.comm   = comm;
    e->u.nccl.redop  = redop;
    bpf_ringbuf_submit(e, 0);
}

SEC("uprobe")
int BPF_KPROBE(ncclSend_enter)
{
    emit_nccl(ctx, F_NCCL_SEND,
              (__u64)PT_REGS_PARM2(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM3(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__u64)PT_REGS_PARM5(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclRecv_enter)
{
    emit_nccl(ctx, F_NCCL_RECV,
              (__u64)PT_REGS_PARM2(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM3(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__u64)PT_REGS_PARM5(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclAllReduce_enter)
{
    /* 7 args; stream is PARM7. AAPCS64 passes x0..x7 in registers. */
    emit_nccl(ctx, F_NCCL_ALL_REDUCE,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              -1,
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM5(ctx));
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclBroadcast_enter)
{
    emit_nccl(ctx, F_NCCL_BROADCAST,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__s32)PT_REGS_PARM5(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclBcast_enter)
{
    emit_nccl(ctx, F_NCCL_BCAST,
              (__u64)PT_REGS_PARM2(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM3(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__u64)PT_REGS_PARM5(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclReduce_enter)
{
    emit_nccl(ctx, F_NCCL_REDUCE,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM8(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__s32)PT_REGS_PARM6(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM5(ctx));
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclAllGather_enter)
{
    emit_nccl(ctx, F_NCCL_ALL_GATHER,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              -1,
              (__u64)PT_REGS_PARM5(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclReduceScatter_enter)
{
    emit_nccl(ctx, F_NCCL_REDUCE_SCATTER,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              -1,
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM5(ctx));
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclAlltoAll_enter)
{
    emit_nccl(ctx, F_NCCL_ALL_TO_ALL,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              -1,
              (__u64)PT_REGS_PARM5(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclGather_enter)
{
    emit_nccl(ctx, F_NCCL_GATHER,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__s32)PT_REGS_PARM5(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclScatter_enter)
{
    emit_nccl(ctx, F_NCCL_SCATTER,
              (__u64)PT_REGS_PARM3(ctx),
              (__u64)PT_REGS_PARM7(ctx),
              (__s32)PT_REGS_PARM4(ctx),
              (__s32)PT_REGS_PARM5(ctx),
              (__u64)PT_REGS_PARM6(ctx),
              -1);
    return 0;
}


SEC("uprobe")
int BPF_KPROBE(ncclGroupStart_enter)
{
    emit_nccl(ctx, F_NCCL_GROUP_START, 0, 0, -1, -1, 0, -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclGroupEnd_enter)
{
    emit_nccl(ctx, F_NCCL_GROUP_END, 0, 0, -1, -1, 0, -1);
    return 0;
}

SEC("uprobe")
int BPF_KPROBE(ncclGroupSimulateEnd_enter)
{
    emit_nccl(ctx, F_NCCL_GROUP_SIMULATE_END, 0, 0, -1, -1, 0, -1);
    return 0;
}
