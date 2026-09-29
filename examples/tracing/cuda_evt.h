// cuda_evt.h
#pragma once

#if defined(__KERNEL__) || defined(__BPF_TRACING__) || defined(__BPF__)
#else
#include <stdint.h>
#endif


enum func_id {
    F_CU_MEMALLOC_V2 = 1,
    F_CU_MEMFREE_V2,
    F_CU_MEMCPY_HTOD_V2,
    F_CU_MEMCPY_DTOH_V2,
    F_CU_LAUNCH_KERNEL,
    F_CUDA_MALLOC,
    F_CUDA_FREE,
    F_CUDA_MEMCPY,

    // 非同期コピー
    F_CU_MEMCPY_HTOD_ASYNC_V2,
    F_CU_MEMCPY_DTOH_ASYNC_V2,
    F_CU_MEMCPY_DTOD_ASYNC_V2,

    // 同期(フェンス)類
    F_CU_STREAM_SYNCHRONIZE,
    F_CUDA_STREAM_SYNCHRONIZE,
    F_CU_CTX_SYNCHRONIZE,
    F_CUDA_DEVICE_SYNCHRONIZE,
    F_CU_EVENT_SYNCHRONIZE,
    F_CUDA_EVENT_SYNCHRONIZE,

    // CUDA Graph replay (used by vLLM when CUDA graphs are enabled)
    F_CU_GRAPH_LAUNCH,

    // Runtime API launch/graph (what PyTorch/vLLM actually call, in libcudart)
    F_CUDA_LAUNCH_KERNEL,
    F_CUDA_GRAPH_LAUNCH,
    F_CUDA_MEMCPY_ASYNC,   /* runtime cudaMemcpyAsync; direction via .kind */

    /* NCCL collectives (enter-only; ret not tracked) */
    F_NCCL_SEND = 50,
    F_NCCL_RECV,
    F_NCCL_ALL_REDUCE,
    F_NCCL_ALL_GATHER,
    F_NCCL_REDUCE_SCATTER,
    F_NCCL_GROUP_START,
    F_NCCL_GROUP_END,

    F_CPU_ONCPU = 100,        /* 次回の switch-in ＝ run-queue 滞在(=off-CPU)が終わった */
    F_CPU_OFFCPU,             /* switch-out ＝ on-CPU 実行が終わった */
    F_CPU_WAKEUP,             /* sched_wakeup で runnable に入った */
    F_CPU_FUTEX_WAIT,         /* futex 待ちが完了（sys_exit_futex） */
    F_CPU_MMAP,

};

struct event {

    uint64_t ts_ns;
    uint32_t ppid;
    uint32_t pid;
    uint32_t tid;
    char     comm[16];

    uint32_t func;  /* enum func_id */
    uint32_t pad;

    int64_t  ret;   /* uretprobe の戻り値（entry時は0） */

    uint64_t trace_hi;
    uint64_t trace_lo;
    union {
        struct { uint64_t bytes; } mem; /* 旧: 同期 memcpy/alloc 用 */
        struct {                       /* 旧: メモリ拡張（alloc/free dptr など）*/
            uint64_t bytes;
            uint64_t dptr;
        } mem_ex;

        struct {                       /* ★ 非同期 memcpy 用 */
            uint64_t bytes;
            uint64_t stream;
        } mem_async;

        struct {                       /* ★ runtime cudaMemcpyAsync 用 */
            uint64_t bytes;
            uint64_t stream;
            int32_t  kind;             /* cudaMemcpyKind: 1=H2D 2=D2H 3=D2D */
        } memcpy_async;

        struct {                       /* 既存: カーネル投入情報 */
            uint64_t func_ptr;
            uint32_t grid_x, grid_y, grid_z;
            uint32_t block_x, block_y, block_z;
            uint32_t shared_mem;
            uint64_t stream;
        } launch;

        struct { uint64_t size; }  malloc_runtime;
        struct { uint64_t count; int32_t kind; } memcpy_runtime;
        struct { uint64_t ptr; }   free_runtime;

        struct {                    /* ★ 同期(フェンス) */
            uint64_t stream;        /* stream同期はstream、ctx/device同期は0 */
        } fence;

        struct {                    /* ★ CUDA Graph replay (cuGraphLaunch) */
            uint64_t graph_exec;    /* CUgraphExec ハンドル */
            uint64_t stream;        /* 投入先 stream */
        } graph;

        struct {                    /* ★ NCCL collectives */
            uint64_t count;         /* element count (0 for GroupStart/End) */
            uint64_t stream;        /* cudaStream_t (0 for GroupStart/End) */
            int32_t  dtype;         /* ncclDataType_t (-1 if N/A) */
            int32_t  peer;          /* peer rank for Send/Recv (-1 otherwise) */
        } nccl;

        struct {
          uint32_t cpu;           /* 実行CPU (switch時は bpf_get_smp_processor_id()) */
          uint32_t state;         /* sched_switch の prev_state 等（任意） */
          uint64_t dur_ns;        /* on/off/runq/futex 等の継続時間 */
          uint64_t aux_ns;        /* 例: runqueue latency(ns) を on-CPU で出す */
        } cpu;
    } u;
};
