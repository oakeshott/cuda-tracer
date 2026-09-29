// nvml_trace.bpf.c - eBPF uprobes for CUDA Driver/Runtime API
#include "vmlinux.h"
#include "cuda_evt.h"
#include "helper.h"
/* #include <linux/bpf.h> */

#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>
/* #include <linux/sched.h> */

char LICENSE[] SEC("license") = "GPL";

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

struct inflight_memcpy_runtime {
    __u64 count;
    __s32 kind;
};

struct inflight_mem_async {
  __u64 bytes;
  __u64 stream;
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
  __u32 ppid = e->ppid;
    struct trace_id *t = bpf_map_lookup_elem(&active_traces, &ppid);
    if (t) { e->trace_hi = t->hi; e->trace_lo = t->lo; }
    else   { e->trace_hi = 0;    e->trace_lo = 0; }
}

/* #<{(| ---------------- Driver API: cuMemAlloc_v2(void **dptr, size_t bytesize) ---------------- |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemAlloc_v2_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem v = {}; */
/*     v.bytes = PT_REGS_PARM2(ctx); */
/*  */
/*     bpf_map_update_elem(&inflight_mem_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemAlloc_v2_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem *vp = bpf_map_lookup_elem(&inflight_mem_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*  */
/*     fill_common(e, F_CU_MEMALLOC_V2); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*  */
/*     // 1引数目は void** dptr */
/*     __u64 dptr_addr = PT_REGS_PARM1(ctx); */
/*     __u64 devptr = 0; */
/*     bpf_probe_read_user(&devptr, sizeof(devptr), (void *)dptr_addr); */
/*  */
/*     if (vp) { */
/*         e->u.mem_ex.bytes = vp->bytes; */
/*         e->u.mem_ex.dptr  = devptr; */
/*  */
/*         if (e->ret == 0 && devptr) { */
/*             __u64 sz = vp->bytes; */
/*             bpf_map_update_elem(&devptr_sz_map, &devptr, &sz, BPF_ANY); */
/*         } */
/*     } */
/*  */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_mem_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ---------------- Driver API: cuMemFree_v2(CUdeviceptr dptr) ---------------- |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemFree_v2_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_free v = { .dptr = PT_REGS_PARM1(ctx) }; */
/*     bpf_map_update_elem(&inflight_free_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemFree_v2_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_free *fp = bpf_map_lookup_elem(&inflight_free_map, &tid); */
/*  */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*  */
/*     fill_common(e, F_CU_MEMFREE_V2); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*  */
/*     __u64 bytes = 0, dptr = fp ? fp->dptr : 0; */
/*     if (dptr) { */
/*         __u64 *bp = bpf_map_lookup_elem(&devptr_sz_map, &dptr); */
/*         if (bp) bytes = *bp; */
/*         if (e->ret == 0 && bp) bpf_map_delete_elem(&devptr_sz_map, &dptr); */
/*     } */
/*     e->u.mem_ex.dptr  = dptr; */
/*     e->u.mem_ex.bytes = bytes; */
/*  */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (fp) bpf_map_delete_elem(&inflight_free_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ---------------- Driver API: cuMemcpyHtoD_v2(dstDevice, srcHost, ByteCount) ---------------- |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemcpyHtoD_v2_enter) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_MEMCPY_HTOD_V2); */
/*     set_trace_id(e); */
/*     e->u.mem.bytes = PT_REGS_PARM3(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemcpyHtoD_v2_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_MEMCPY_HTOD_V2); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ---------------- Driver API: cuMemcpyDtoH_v2(dstHost, srcDevice, ByteCount) ---------------- |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemcpyDtoH_v2_enter) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_MEMCPY_DTOH_V2); */
/*     set_trace_id(e); */
/*     e->u.mem.bytes = PT_REGS_PARM3(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemcpyDtoH_v2_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_MEMCPY_DTOH_V2); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* #ifdef __x86_64__ */
/* #<{(| ---------------- Driver API: cuLaunchKernel(...) ---------------- */
/*  * CUresult cuLaunchKernel(CUfunction f, */
/*  *   unsigned gridX, gridY, gridZ, */
/*  *   unsigned blockX, blockY, blockZ, */
/*  *   unsigned sharedMemBytes, CUstream hStream, void** kernelParams, void** extra) */
/*  * x86_64 SysV: 第7引数以降はユーザスタック (SP) 側 */
/*  |)}># */
/* static __always_inline __u64 read_user_u64(void *addr) */
/* { */
/*     __u64 v = 0; */
/*     bpf_probe_read_user(&v, sizeof(v), addr); */
/*     return v; */
/* } */
/* #endif */
/*  */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuLaunchKernel_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_launch v = {}; */
/*  */
/*     // regs */
/*     v.func_ptr = PT_REGS_PARM1(ctx); */
/*     v.grid_x   = (__u32)PT_REGS_PARM2(ctx); */
/*     v.grid_y   = (__u32)PT_REGS_PARM3(ctx); */
/*     v.grid_z   = (__u32)PT_REGS_PARM4(ctx); */
/*     v.block_x  = (__u32)PT_REGS_PARM5(ctx); */
/*     v.block_y  = (__u32)PT_REGS_PARM6(ctx); */
/*  */
/* #ifdef __x86_64__ */
/*     // SP + 8: 7th arg (blockZ), +16: 8th (sharedMem), +24: 9th (stream) */
/*     __u64 sp = PT_REGS_SP(ctx); */
/*     v.block_z   = (__u32)read_user_u64((void *)(sp + 8)); */
/*     v.shared_mem= (__u32)read_user_u64((void *)(sp + 16)); */
/*     v.stream    = read_user_u64((void *)(sp + 24)); */
/* #else */
/*     v.block_z = 0; */
/*     v.shared_mem = 0; */
/*     v.stream = 0; */
/* #endif */
/*  */
/*     bpf_map_update_elem(&inflight_launch_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuLaunchKernel_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_launch *vp = bpf_map_lookup_elem(&inflight_launch_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*  */
/*  */
/*     fill_common(e, F_CU_LAUNCH_KERNEL); */
/*     set_trace_id(e); */
/*  */
/*     if (vp) { */
/*         e->u.launch.func_ptr = vp->func_ptr; */
/*         e->u.launch.grid_x = vp->grid_x; */
/*         e->u.launch.grid_y = vp->grid_y; */
/*         e->u.launch.grid_z = vp->grid_z; */
/*         e->u.launch.block_x = vp->block_x; */
/*         e->u.launch.block_y = vp->block_y; */
/*         e->u.launch.block_z = vp->block_z; */
/*         e->u.launch.shared_mem = vp->shared_mem; */
/*         e->u.launch.stream = vp->stream; */
/*     } */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_launch_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ---------------- Runtime API (optional): cudaMalloc/cudaFree/cudaMemcpy ---------------- |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cudaMalloc_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem v = {}; */
/*     v.bytes = PT_REGS_PARM2(ctx); */
/*     bpf_map_update_elem(&inflight_mem_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cudaMalloc_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem *vp = bpf_map_lookup_elem(&inflight_mem_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*  */
/*     fill_common(e, F_CUDA_MALLOC); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*  */
/*     // 1引数目 void** devPtr */
/*     __u64 devptr_addr = PT_REGS_PARM1(ctx); */
/*     __u64 devptr = 0; */
/*     bpf_probe_read_user(&devptr, sizeof(devptr), (void *)devptr_addr); */
/*  */
/*     if (vp) { */
/*         e->u.mem_ex.bytes = vp->bytes; */
/*         e->u.mem_ex.dptr  = devptr; */
/*         if (e->ret == 0 && devptr) { */
/*             __u64 sz = vp->bytes; */
/*             bpf_map_update_elem(&devptr_sz_map, &devptr, &sz, BPF_ANY); */
/*         } */
/*     } */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_mem_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* // cudaFree enter/exit */
/* SEC("uprobe") */
/* int BPF_KPROBE(cudaFree_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_free v = { .dptr = PT_REGS_PARM1(ctx) }; */
/*     bpf_map_update_elem(&inflight_free_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cudaFree_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_free *fp = bpf_map_lookup_elem(&inflight_free_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*  */
/*     fill_common(e, F_CUDA_FREE); */
/*     set_trace_id(e); */
/*     e->ret = PT_REGS_RC(ctx); */
/*  */
/*     __u64 bytes = 0, dptr = fp ? fp->dptr : 0; */
/*     if (dptr) { */
/*         __u64 *bp = bpf_map_lookup_elem(&devptr_sz_map, &dptr); */
/*         if (bp) bytes = *bp; */
/*         if (e->ret == 0 && bp) bpf_map_delete_elem(&devptr_sz_map, &dptr); */
/*     } */
/*     e->u.mem_ex.dptr  = dptr; */
/*     e->u.mem_ex.bytes = bytes; */
/*  */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (fp) bpf_map_delete_elem(&inflight_free_map, &tid); */
/*     return 0; */
/* } */
/*  */
/*  */
/* SEC("uprobe") */
/* int BPF_KPROBE(cudaMemcpy_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_memcpy_runtime v = {}; */
/*     v.count = PT_REGS_PARM3(ctx); */
/*     v.kind  = (__s32)PT_REGS_PARM4(ctx); */
/*     bpf_map_update_elem(&inflight_memcpy_rt_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cudaMemcpy_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_memcpy_runtime *vp = bpf_map_lookup_elem(&inflight_memcpy_rt_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*     fill_common(e, F_CUDA_MEMCPY); */
/*     set_trace_id(e); */
/*     if (vp) { */
/*         e->u.memcpy_runtime.count = vp->count; */
/*         e->u.memcpy_runtime.kind  = vp->kind; */
/*     } */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_memcpy_rt_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemcpyHtoDAsync_v2_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async v = { */
/*         .bytes  = PT_REGS_PARM3(ctx), */
/*         .stream = PT_REGS_PARM4(ctx), */
/*     }; */
/*     bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemcpyHtoDAsync_v2_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*     fill_common(e, F_CU_MEMCPY_HTOD_ASYNC_V2); */
/*     set_trace_id(e); */
/*     if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; } */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemcpyDtoHAsync_v2_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async v = { */
/*         .bytes  = PT_REGS_PARM3(ctx), */
/*         .stream = PT_REGS_PARM4(ctx), */
/*     }; */
/*     bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemcpyDtoHAsync_v2_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*     fill_common(e, F_CU_MEMCPY_DTOH_ASYNC_V2); */
/*     set_trace_id(e); */
/*     if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; } */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| cuMemcpyDtoDAsync_v2(dst, src, bytes, stream) |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuMemcpyDtoDAsync_v2_enter) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async v = { */
/*         .bytes  = PT_REGS_PARM3(ctx), */
/*         .stream = PT_REGS_PARM4(ctx), */
/*     }; */
/*     bpf_map_update_elem(&inflight_mem_async_map, &tid, &v, BPF_ANY); */
/*     return 0; */
/* } */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuMemcpyDtoDAsync_v2_exit) */
/* { */
/*     __u32 tid = get_tid(); */
/*     struct inflight_mem_async *vp = bpf_map_lookup_elem(&inflight_mem_async_map, &tid); */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) goto out; */
/*     fill_common(e, F_CU_MEMCPY_DTOD_ASYNC_V2); */
/*     set_trace_id(e); */
/*     if (vp) { e->u.mem_async.bytes = vp->bytes; e->u.mem_async.stream = vp->stream; } */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/* out: */
/*     if (vp) bpf_map_delete_elem(&inflight_mem_async_map, &tid); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ★ stream 同期: cuStreamSynchronize(stream) |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuStreamSynchronize_enter) { return 0; } */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuStreamSynchronize_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_STREAM_SYNCHRONIZE); */
/*     set_trace_id(e); */
/*     e->u.fence.stream = PT_REGS_PARM1(ctx); */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* #<{(| runtime: cudaStreamSynchronize(stream) |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cudaStreamSynchronize_enter) { return 0; } */
/* #<{(| ★ runtime: cudaStreamSynchronize(stream) |)}># */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cudaStreamSynchronize_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CUDA_STREAM_SYNCHRONIZE); */
/*     set_trace_id(e); */
/*     e->u.fence.stream = PT_REGS_PARM1(ctx); */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* #<{(| ★ ctx/device 同期：ストリーム全体をフェンス（stream=0） |)}># */
/* #<{(| driver: cuCtxSynchronize() |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cuCtxSynchronize_enter) { return 0; } */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cuCtxSynchronize_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CU_CTX_SYNCHRONIZE); */
/*     set_trace_id(e); */
/*     e->u.fence.stream = 0; */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
/* #<{(| runtime: cudaDeviceSynchronize() |)}># */
/* SEC("uprobe") */
/* int BPF_KPROBE(cudaDeviceSynchronize_enter) { return 0; } */
/*  */
/* SEC("uretprobe") */
/* int BPF_KPROBE(cudaDeviceSynchronize_exit) */
/* { */
/*     struct event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0); */
/*     if (!e) return 0; */
/*     fill_common(e, F_CUDA_DEVICE_SYNCHRONIZE); */
/*     set_trace_id(e); */
/*     e->u.fence.stream = 0; */
/*     e->ret = PT_REGS_RC(ctx); */
/*     bpf_ringbuf_submit(e, 0); */
/*     return 0; */
/* } */
/*  */
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
