// userspace: libbpf-bootstrap style
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <fcntl.h>
#include <stdbool.h>
#include <sys/resource.h>

#include <stdarg.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#include <gelf.h>
#include <libelf.h>
#include <pthread.h>
#include <time.h>
#include <nvml.h>
#include <inttypes.h>

#include "cuda_evt.h"
#include "nvml_trace.skel.h"

#define MAX_ITEMS 65536
#define OP_KERNEL 1
#define OP_HTOD   2
#define OP_DTOH   3
#define OP_DTOD   4

struct pid_usage {
  uint32_t pid;
  uint64_t cur;
  uint64_t peak;
};

static volatile sig_atomic_t exiting = 0;
#define MAX_PIDS 4096

static struct pid_usage usages[MAX_PIDS];
static int usages_cnt = 0;

static struct pid_usage* get_usage(uint32_t pid) {
  for (int i=0;i<usages_cnt;i++) if (usages[i].pid==pid) return &usages[i];
  if (usages_cnt < MAX_PIDS) {
    usages[usages_cnt] = (struct pid_usage){ .pid = pid, .cur = 0, .peak = 0 };
    return &usages[usages_cnt++];
  }
  return NULL;
}

static void add_mem(uint32_t pid, uint64_t bytes) {
  struct pid_usage *u = get_usage(pid);
  if (!u) return;
  u->cur += bytes;
  if (u->cur > u->peak) u->peak = u->cur;
}
static void sub_mem(uint32_t pid, uint64_t bytes) {
  struct pid_usage *u = get_usage(pid);
  if (!u) return;
  if (u->cur >= bytes) u->cur -= bytes;
  else u->cur = 0;
}

static void on_sigint(int signo) {
  (void)signo;
  exiting = 1;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *fmt, va_list args) {
  if (level == LIBBPF_DEBUG)  // 詳細が欲しければここを出す
    return 0;
  return vfprintf(stderr, fmt, args);
}

static inline uint64_t mono_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);        // ← bpf_ktime_get_ns と揃う
  return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

struct work_item {
  uint64_t token;
  uint64_t ts_start_ns;
  uint64_t stream;
  uint64_t bytes;
  uint32_t pid;
  uint32_t op;     /* 上の OP_* */
  uint8_t  done;   /* 0/1 */
};
static struct work_item items[MAX_ITEMS];
static uint64_t items_cnt = 0;
static uint64_t next_token = 1;

static void enq_item(uint32_t pid, uint64_t stream, uint32_t op, uint64_t bytes, uint64_t ts_start_ns) {
  if (items_cnt >= MAX_ITEMS) return;
  items[items_cnt++] = (struct work_item){
    .token = next_token++,
      .ts_start_ns = ts_start_ns,
      .stream = stream,
      .bytes = bytes,
      .pid = pid,
      .op = op,
      .done = 0
  };
}

static void complete_stream(uint64_t stream, uint64_t ts_end_ns){
  for (uint64_t i=0;i<items_cnt;i++){
    if (!items[i].done && items[i].stream == stream){
      items[i].done = 1;
      double ms = (ts_end_ns - items[i].ts_start_ns) / 1e6;
      const char* kind = items[i].op==OP_KERNEL?"KERNEL":
        items[i].op==OP_HTOD?"HtoD":
        items[i].op==OP_DTOH?"DtoH":"DtoD";
      uint64_t sec  = (uint64_t)(ts_end_ns / 1000000000ULL);
      uint64_t nsec = (uint64_t)(ts_end_ns % 1000000000ULL);
      fprintf(stdout,
          "[%" PRIu64 ".%06" PRIu64 "] COMPLETE stream=%p op=%s pid=%u token=%llu latency=%.3f ms bytes=%llu\n",
          sec, nsec,
          (void*)(uintptr_t)stream, kind, items[i].pid,
          (unsigned long long)items[i].token, ms,
          (unsigned long long)items[i].bytes);
    }
  }
}

static void complete_all(uint64_t ts_end_ns){
  for (uint64_t i=0;i<items_cnt;i++){
    if (!items[i].done){
      items[i].done = 1;
      double ms = (ts_end_ns - items[i].ts_start_ns)/1e6;
      const char* kind = items[i].op==OP_KERNEL?"KERNEL":
        items[i].op==OP_HTOD?"HtoD":
        items[i].op==OP_DTOH?"DtoH":"DtoD";
      uint64_t sec  = (uint64_t)(ts_end_ns / 1000000000ULL);
      uint64_t nsec = (uint64_t)(ts_end_ns % 1000000000ULL);
      fprintf(stdout,
          "[%" PRIu64 ".%06" PRIu64 "] COMPLETE_ALL stream=%p op=%s pid=%u token=%llu latency=%.3f ms bytes=%llu\n",
          sec, nsec,
          (void*)(uintptr_t)items[i].stream, kind, items[i].pid,
          (unsigned long long)items[i].token, ms,
          (unsigned long long)items[i].bytes);

    }
  }
}

/* #ifndef NO_NVML */
/* #<{(| ==== NVML thread ==== |)}># */
/* static void* nvml_thread(void* arg){ */
/*   (void)arg; */
/*   if (nvmlInit_v2() != NVML_SUCCESS) { */
/*     fprintf(stderr, "NVML init failed\n"); */
/*     return NULL; */
/*   } */
/*   unsigned int count = 0; */
/*   nvmlDeviceGetCount_v2(&count); */
/*   while (1) { */
/*     for (unsigned int i=0;i<count;i++){ */
/*       nvmlDevice_t h; */
/*       if (nvmlDeviceGetHandleByIndex_v2(i, &h) != NVML_SUCCESS) continue; */
/*       nvmlUtilization_t u = {0}; */
/*       nvmlMemory_t m = {0}; */
/*       unsigned int temp = 0, sm_clk=0, mem_clk=0; */
/*       unsigned int power_mw = 0; */
/*  */
/*       nvmlDeviceGetUtilizationRates(h, &u); */
/*       nvmlDeviceGetMemoryInfo(h, &m); */
/*       nvmlDeviceGetTemperature(h, NVML_TEMPERATURE_GPU, &temp); */
/*       nvmlDeviceGetClockInfo(h, NVML_CLOCK_SM, &sm_clk); */
/*       nvmlDeviceGetClockInfo(h, NVML_CLOCK_MEM, &mem_clk); */
/*       nvmlDeviceGetPowerUsage(h, &power_mw); */
/*  */
/*       #<{(| fprintf(stdout, |)}># */
/*       #<{(|     "NVML gpu=%u util.gpu=%u%% util.mem=%u%% mem.used=%.2f/%.2f GiB temp=%uC power=%.1fW clocks sm=%uMHz mem=%uMHz\n", |)}># */
/*       #<{(|     i, u.gpu, u.memory, |)}># */
/*       #<{(|     (double)m.used/1024/1024/1024, (double)m.total/1024/1024/1024, |)}># */
/*       #<{(|     temp, power_mw/1000.0, sm_clk, mem_clk); |)}># */
/*       uint64_t tns = mono_ns(); */
/*       uint64_t sec  = (uint64_t)(tns / 1000000000ULL); */
/*       #<{(| uint64_t nsec = (uint64_t)(tns % 1000000000ULL); |)}># */
/*       uint64_t usec = (tns / 1000ull) % 1000000ull; */
/*       fprintf(stdout, */
/*           "[%" PRIu64 ".%06" PRIu64 "] NVML gpu=%u util.gpu=%u%% util.mem=%u%% " */
/*           "mem.used=%.2f/%.2f GiB temp=%uC power=%.1fW clocks sm=%uMHz mem=%uMHz\n", */
/*           sec, usec, */
/*           i, u.gpu, u.memory, */
/*           (double)m.used/1024/1024/1024, (double)m.total/1024/1024/1024, */
/*           temp, power_mw/1000.0, sm_clk, mem_clk); */
/*     } */
/*     fflush(stdout); */
/*     struct timespec ts = {.tv_sec=0,.tv_nsec=500*1000*1000}; // 500ms */
/*     nanosleep(&ts, NULL); */
/*   } */
/*   nvmlShutdown(); */
/*   return NULL; */
/* } */
/* #endif */

static int handle_event(void *ctx, void *data, size_t len)
{
  const struct event *e = data;
  const char *fname = "UNKNOWN";
  switch (e->func) {
    case F_CU_MEMALLOC_V2:   fname="cuMemAlloc_v2"; break;
    case F_CU_MEMFREE_V2:    fname="cuMemFree_v2";  break;
    case F_CUDA_MALLOC:      fname="cudaMalloc";    break;
    case F_CUDA_FREE:        fname="cudaFree";      break;
    case F_CU_MEMCPY_HTOD_V2:fname="cuMemcpyHtoD_v2"; break;
    case F_CU_MEMCPY_DTOH_V2:fname="cuMemcpyDtoH_v2"; break;
    case F_CU_LAUNCH_KERNEL: fname="cuLaunchKernel";  break;
    case F_CU_MEMCPY_HTOD_ASYNC_V2: fname="cuMemcpyHtoDAsync_v2"; break;
    case F_CU_MEMCPY_DTOH_ASYNC_V2: fname="cuMemcpyDtoHAsync_v2"; break;
    case F_CU_MEMCPY_DTOD_ASYNC_V2: fname="cuMemcpyDtoDAsync_v2"; break;
    case F_CU_STREAM_SYNCHRONIZE: fname="cuStreamSynchronize"; break;
    case F_CUDA_STREAM_SYNCHRONIZE: fname="cudaStreamSynchronize"; break;
    case F_CU_CTX_SYNCHRONIZE: fname="cuCtxSynchronize"; break;
    case F_CUDA_DEVICE_SYNCHRONIZE: fname="cudaDeviceSynchronize"; break;
    case F_CPU_ONCPU:          fname="CPU_ONCPU"; break;
    case F_CPU_OFFCPU:         fname="CPU_OFFCPU"; break;
    case F_CPU_WAKEUP:         fname="CPU_WAKEUP"; break;
    case F_CPU_FUTEX_WAIT:     fname="CPU_FUTEX_WAIT"; break;
    default: break;
  }

  printf("[%ld.%06ld] ppid=%u, pid=%u tid=%u trace_id=%lu%lu, comm=%s func=%s",
      (long)(e->ts_ns/1000000000ULL),
      (long)((e->ts_ns/1000ULL)%1000000ULL),
      e->ppid, e->pid, e->tid,
      e->trace_hi, e->trace_lo,
      e->comm, fname);

  switch (e->func) {
    case F_CU_MEMALLOC_V2:
    case F_CUDA_MALLOC: {
                          uint64_t bytes = e->u.mem_ex.bytes ? e->u.mem_ex.bytes : e->u.malloc_runtime.size;
                          printf(" dptr=%p bytes=%llu ret=%lld",
                              (void*)(uintptr_t)e->u.mem_ex.dptr,
                              (unsigned long long)bytes, (long long)e->ret);
                          if (e->ret==0 && bytes) add_mem(e->pid, bytes);
                          break;
                        }
    case F_CU_MEMFREE_V2:
    case F_CUDA_FREE: {
                        uint64_t bytes = e->u.mem_ex.bytes ? e->u.mem_ex.bytes : 0;
                        printf(" dptr=%p bytes=%llu ret=%lld",
                            (void*)(uintptr_t)e->u.mem_ex.dptr,
                            (unsigned long long)bytes, (long long)e->ret);
                        if (e->ret==0 && bytes) sub_mem(e->pid, bytes);
                        break;
                      }
    case F_CU_MEMCPY_HTOD_V2:
    case F_CU_MEMCPY_DTOH_V2:
                      printf(" bytes=%llu ret=%lld",
                          (unsigned long long)e->u.mem.bytes, (long long)e->ret);
                      break;
    case F_CU_LAUNCH_KERNEL:
                      enq_item(e->pid, e->u.launch.stream, OP_KERNEL, 0, e->ts_ns);
                      printf(" f=%p grid=%ux%ux%u block=%ux%ux%u shmem=%u stream=%p ret=%lld",
                          (void*)(uintptr_t)e->u.launch.func_ptr,
                          e->u.launch.grid_x, e->u.launch.grid_y, e->u.launch.grid_z,
                          e->u.launch.block_x, e->u.launch.block_y, e->u.launch.block_z,
                          e->u.launch.shared_mem, (void*)(uintptr_t)e->u.launch.stream,
                          (long long)e->ret);
                      break;
    case F_CU_MEMCPY_HTOD_ASYNC_V2:
                      enq_item(e->pid, e->u.mem_async.stream, OP_HTOD, e->u.mem_async.bytes, e->ts_ns);
                      break;
    case F_CU_MEMCPY_DTOH_ASYNC_V2:
                      enq_item(e->pid, e->u.mem_async.stream, OP_DTOH, e->u.mem_async.bytes, e->ts_ns);
                      break;
    case F_CU_MEMCPY_DTOD_ASYNC_V2:
                      enq_item(e->pid, e->u.mem_async.stream, OP_DTOD, e->u.mem_async.bytes, e->ts_ns);
                      break;
    case F_CU_STREAM_SYNCHRONIZE:
                      if (e->ret==0) complete_stream(e->u.fence.stream, e->ts_ns);
                      break;
    case F_CUDA_STREAM_SYNCHRONIZE:
                      if (e->ret==0) complete_stream(e->u.fence.stream, e->ts_ns);
                      break;
    case F_CU_CTX_SYNCHRONIZE:
    case F_CUDA_DEVICE_SYNCHRONIZE:
                      if (e->ret==0) complete_all(e->ts_ns);
                      break;
    case F_CPU_OFFCPU:
                      printf(" cpu=%u oncpu=%.3f ms", e->u.cpu.cpu, e->u.cpu.dur_ns/1e6);
                      break;
    case F_CPU_ONCPU:
                      printf(" cpu=%u offcpu=%.3f ms rq_latency=%.3f ms",
                          e->u.cpu.cpu, e->u.cpu.dur_ns/1e6, e->u.cpu.aux_ns/1e6);
                      break;
    case F_CPU_WAKEUP:
                      printf(" cpu=%u", e->u.cpu.cpu);
                      break;
    case F_CPU_FUTEX_WAIT:
                      printf(" cpu=%u futex_wait=%.3f ms", e->u.cpu.cpu, e->u.cpu.dur_ns/1e6);
                      break;
    default: break;
  }
  printf("\n");
  /*  */
  /* struct pid_usage *u = get_usage(e->pid); */
  /* if (u) */
  /*   printf("  => MEM[pid=%u] cur=%.2f MiB, peak=%.2f MiB\n", */
  /*       u->pid, mib(u->cur), mib(u->peak)); */

  return 0;
}

/* static struct ring_buffer* open_ringbuf_any(struct nvml_trace_bpf *skel) { */
/*   struct bpf_object *obj = skel->obj; */
/*   struct bpf_map *m; */
/*   struct ring_buffer *rb = NULL; */
/*  */
/*   bpf_object__for_each_map(m, obj) { */
/*     enum bpf_map_type t = bpf_map__type(m); */
/*     if (t == BPF_MAP_TYPE_RINGBUF) { */
/*       int fd = bpf_map__fd(m); */
/*       rb = ring_buffer__new(fd, handle_event, NULL, NULL); */
/*       if (!rb) { */
/*         fprintf(stderr, "ring_buffer__new failed for map %s: %s\n", */
/*             bpf_map__name(m), strerror(errno)); */
/*       } else { */
/*         fprintf(stderr, "using ringbuf map: %s\n", bpf_map__name(m)); */
/*         return rb; */
/*       } */
/*     } */
/*   } */
/*   return NULL; */
/* } */

static int bump_memlock_rlimit(void) {
  struct rlimit rlim = {
    .rlim_cur = RLIM_INFINITY,
    .rlim_max = RLIM_INFINITY,
  };
  return setrlimit(RLIMIT_MEMLOCK, &rlim);
}

/* static int open_elf(const char *path, Elf **elf, int *fd, GElf_Ehdr *ehdr) */
/* { */
/*   *fd = open(path, O_RDONLY); */
/*   if (*fd < 0) { */
/*     fprintf(stderr, "open(%s): %s\n", path, strerror(errno)); */
/*     return -1; */
/*   } */
/*   if (elf_version(EV_CURRENT) == EV_NONE) { */
/*     fprintf(stderr, "libelf version error\n"); */
/*     close(*fd); */
/*     return -1; */
/*   } */
/*   *elf = elf_begin(*fd, ELF_C_READ, NULL); */
/*   if (!*elf) { */
/*     fprintf(stderr, "elf_begin(%s): %s\n", path, elf_errmsg(-1)); */
/*     close(*fd); */
/*     return -1; */
/*   } */
/*   if (!gelf_getehdr(*elf, ehdr)) { */
/*     fprintf(stderr, "gelf_getehdr: %s\n", elf_errmsg(-1)); */
/*     elf_end(*elf); */
/*     close(*fd); */
/*     return -1; */
/*   } */
/*   return 0; */
/* } */

/* static long find_sym_offset(const char *path, const char *symname) */
/* { */
/*   Elf *elf = NULL; */
/*   int fd = -1; */
/*   GElf_Ehdr ehdr; */
/*   if (open_elf(path, &elf, &fd, &ehdr)) return -1; */
/*  */
/*   size_t shstrndx; */
/*   if (elf_getshdrstrndx(elf, &shstrndx) != 0) { */
/*     fprintf(stderr, "elf_getshdrstrndx: %s\n", elf_errmsg(-1)); */
/*     goto out_err; */
/*   } */
/*  */
/*   Elf_Scn *scn = NULL; */
/*   while ((scn = elf_nextscn(elf, scn)) != NULL) { */
/*     GElf_Shdr shdr; */
/*     if (!gelf_getshdr(scn, &shdr)) continue; */
/*     if (shdr.sh_type != SHT_SYMTAB && shdr.sh_type != SHT_DYNSYM) continue; */
/*  */
/*     Elf_Data *data = elf_getdata(scn, NULL); */
/*     if (!data) continue; */
/*  */
/*     size_t cnt = shdr.sh_size / shdr.sh_entsize; */
/*     for (size_t i = 0; i < cnt; i++) { */
/*       GElf_Sym sym; */
/*       if (!gelf_getsym(data, (int)i, &sym)) continue; */
/*       const char *name = elf_strptr(elf, shdr.sh_link, sym.st_name); */
/*       if (!name) continue; */
/*       if (strcmp(name, symname) == 0) { */
/*         long off = (long)sym.st_value; */
/*         elf_end(elf); */
/*         close(fd); */
/*         return off; */
/*       } */
/*     } */
/*   } */
/* out_err: */
/*   if (elf) elf_end(elf); */
/*   if (fd >= 0) close(fd); */
/*   return -1; */
/* } */
/*  */
/* struct link_holder { */
/*   struct bpf_link *enter; */
/*   struct bpf_link *exit; */
/* }; */

/* static int attach_one(struct nvml_trace_bpf *skel, pid_t pid, const char *lib_path, */
/*     const char *sym, const char *prog_enter, const char *prog_exit, */
/*     struct link_holder *out) */
/* { */
/*   struct bpf_object *obj = skel->obj; */
/*   long off = find_sym_offset(lib_path, sym); */
/*   if (off < 0) { */
/*     fprintf(stderr, "symbol %s not found in %s\n", sym, lib_path); */
/*     return -1; */
/*   } */
/*  */
/*   struct bpf_program *pe = bpf_object__find_program_by_name(obj, prog_enter); */
/*   struct bpf_program *px = bpf_object__find_program_by_name(obj, prog_exit); */
/*   if (!pe || !px) { */
/*     fprintf(stderr, "bpf program not found: %s / %s\n", prog_enter, prog_exit); */
/*     return -1; */
/*   } */
/*  */
/*   LIBBPF_OPTS(bpf_uprobe_opts, opts_e, */
/*       .retprobe = false */
/*       ); */
/*   LIBBPF_OPTS(bpf_uprobe_opts, opts_x, */
/*       .retprobe = true */
/*       ); */
/*  */
/*   out->enter = bpf_program__attach_uprobe_opts(pe, pid, lib_path, off, &opts_e); */
/*   if (!out->enter) { */
/*     int err = -errno; */
/*     fprintf(stderr, "attach enter %s@%s failed: %s\n", sym, lib_path, strerror(-err)); */
/*     return -1; */
/*   } */
/*   out->exit  = bpf_program__attach_uprobe_opts(px, pid, lib_path, off, &opts_x); */
/*   if (!out->exit) { */
/*     int err = -errno; */
/*     fprintf(stderr, "attach exit %s@%s failed: %s\n", sym, lib_path, strerror(-err)); */
/*     bpf_link__destroy(out->enter); */
/*     out->enter = NULL; */
/*     return -1; */
/*   } */
/*   return 0; */
/* } */


static void usage(const char *prog) {
  fprintf(stderr,
      "Usage: %s [--verbose] [--duration SEC]\n"
      "  --verbose       Verbose libbpf logs (info+warn)\n"
      "  --duration SEC  Exit after SEC seconds (default: run until Ctrl-C)\n"
      "\n"
      "Note:\n"
      "  - If your BPF uprobe SEC() strings already point to absolute\n"
      "    shared objects (e.g. libcuda.so.1), skeleton __attach() will auto-attach.\n"
      "  - Otherwise, manually attach uprobes in the marked section below.\n",
      prog);
}

int main(int argc, char **argv) {
  int err = 0;
  bool verbose = false;
  int duration = 0;

  const char *libcuda   = getenv("LIBCUDA_PATH");
  const char *libcudart = getenv("LIBCUDART_PATH");
  /* const char *bpf_obj   = "./nvml_trace.bpf.o"; */
  pid_t pid = -1;
  if (!libcuda)   libcuda   = "/usr/lib/aarch64-linux-gnu/libcuda.so.1";
  if (!libcudart) libcudart = "/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so";

  static const struct option opts[] = {
    {"pid", required_argument, 0, 'p'},
    {"libcuda", required_argument, 0, 'd'},
    {"libcudart", required_argument, 0, 'r'},
    {"bpf-obj", required_argument, 0, 'o'},
    {"help", no_argument, 0, 'h'},
    {0,0,0,0}
  };

  int c;
  while ((c = getopt_long(argc, argv, "p:d:r:o:h", (struct option*)opts, NULL)) != -1) {
    switch(c) {
      case 'p': pid = (pid_t)strtol(optarg, NULL, 10); break;
      case 'd': libcuda = optarg; break;
      case 'r': libcudart = optarg; break;
      case 'h': default: usage(argv[0]); return c=='h'?0:1;
    }
  }


  if (bump_memlock_rlimit()) {
    perror("setrlimit(RLIMIT_MEMLOCK)");
    // 続行は可能なことが多いので warn のみにする
  }

  libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
  libbpf_set_print(libbpf_print_fn);
  if (verbose) {
    // info 以上だけ可視化（debug はノイズなので引き続き無効）
    libbpf_set_print(libbpf_print_fn);
  }


  struct nvml_trace_bpf *skel = NULL;
  struct ring_buffer *rb = NULL;
  struct bpf_map *rb_map;
  int rb_fd;

  skel = nvml_trace_bpf__open();
  if (!skel) {
    fprintf(stderr, "failed to open skeleton\n");
    err = 1;
    goto cleanup;
  }

  // BPF オプションを load 前にセットしたい場合は skel->rodata / skel->bss に代入
  // 例:
  // skel->rodata->target_tgid = my_pid;

  err = nvml_trace_bpf__load(skel);
  if (err) {
    fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
    goto cleanup;
  }

  // --- 自動アタッチ（BPF 側の SEC 文字列が絶対パスや soname を含む場合） ---
  err = nvml_trace_bpf__attach(skel);
  if (err) {
    fprintf(stderr, "auto-attach failed (%d). If your BPF uprobe SEC() "
        "doesn't specify full path, attach manually in userspace.\n", err);
    goto cleanup;
  }

  {
    int cfg_fd = bpf_object__find_map_fd_by_name(skel->obj, "cfg_pid_filter");
    if (cfg_fd >= 0) {
      __u32 key = 0;
      __u32 val = (pid > 0) ? (uint32_t)pid : 0;
      if (bpf_map_update_elem(cfg_fd, &key, &val, BPF_ANY) != 0) {
        fprintf(stderr, "warn: set cfg_pid_filter failed\n");
      }
    }
  }

  /* { */
  /*   struct bpf_map *m = bpf_object__find_map_by_name(skel->obj, "active_traces"); */
  /*   const char *pin = "/sys/fs/bpf/active_traces"; */
  /*   if (m) { */
  /*     unlink(pin); */
  /*     if (bpf_map__pin(m, pin)) { */
  /*       fprintf(stderr, "pin active_traces failed\n"); */
  /*     } */
  /*     printf("active_traces is pinned.\n"); */
  /*   } */
  /*   if (bpf_map_update_elem(cfg_fd, &key, &val, BPF_ANY) != 0) { */
  /*     fprintf(stderr, "warn: set cfg_pid_filter failed\n"); */
  /*   } */
  /* } */
  /* // ringbuf を開く（マップ名は BPF 側と一致させる） */
  /* rb = open_ringbuf_any(skel); */
  /* if (!rb) { */
  /*   fprintf(stderr, "no ring buffer map found (BPF_MAP_TYPE_RINGBUF)\n"); */
  /*   err = 1; */
  /*   goto cleanup; */
  /* } */
/*   struct link_holder links[16] = {}; */
/*   int lidx = 0; */
/*  */
/*   if (attach_one(skel, pid, libcuda,   "cuMemAlloc_v2", */
/*         "cuMemAlloc_v2_enter", "cuMemAlloc_v2_exit", &links[lidx++])) goto after_attach; */
/*   if (attach_one(skel, pid, libcuda,   "cuMemFree_v2", */
/*         "cuMemFree_v2_enter", "cuMemFree_v2_exit", &links[lidx++])) goto after_attach; */
/*   if (attach_one(skel, pid, libcuda,   "cuMemcpyHtoD_v2", */
/*         "cuMemcpyHtoD_v2_enter", "cuMemcpyHtoD_v2_exit", &links[lidx++])) goto after_attach; */
/*   if (attach_one(skel, pid, libcuda,   "cuMemcpyDtoH_v2", */
/*         "cuMemcpyDtoH_v2_enter", "cuMemcpyDtoH_v2_exit", &links[lidx++])) goto after_attach; */
/*   if (attach_one(skel, pid, libcuda,   "cuLaunchKernel", */
/*         "cuLaunchKernel_enter", "cuLaunchKernel_exit", &links[lidx++])) goto after_attach; */
/*  */
/*   attach_one(skel, pid, libcudart, "cudaMalloc", */
/*       "cudaMalloc_enter", "cudaMalloc_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcudart, "cudaFree", */
/*       "cudaFree_enter", "cudaFree_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcudart, "cudaMemcpy", */
/*       "cudaMemcpy_enter", "cudaMemcpy_exit", &links[lidx++]); */
/*   // Driver API: 非同期 memcpy */
/*   attach_one(skel, pid, libcuda, "cuMemcpyHtoDAsync_v2", */
/*       "cuMemcpyHtoDAsync_v2_enter", "cuMemcpyHtoDAsync_v2_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcuda, "cuMemcpyDtoHAsync_v2", */
/*       "cuMemcpyDtoHAsync_v2_enter", "cuMemcpyDtoHAsync_v2_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcuda, "cuMemcpyDtoDAsync_v2", */
/*       "cuMemcpyDtoDAsync_v2_enter", "cuMemcpyDtoDAsync_v2_exit", &links[lidx++]); */
/*  */
/*   attach_one(skel, pid, libcuda,   "cuStreamSynchronize", */
/*       "cuStreamSynchronize_enter", "cuStreamSynchronize_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcudart, "cudaStreamSynchronize", */
/*       "cudaStreamSynchronize_enter", "cudaStreamSynchronize_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcuda,   "cuCtxSynchronize", */
/*       "cuCtxSynchronize_enter", "cuCtxSynchronize_exit", &links[lidx++]); */
/*   attach_one(skel, pid, libcudart, "cudaDeviceSynchronize", */
/*       "cudaDeviceSynchronize_enter", "cudaDeviceSynchronize_exit", &links[lidx++]); */
/*  */
/* after_attach: */

  rb_map = bpf_object__find_map_by_name(skel->obj, "events");
  if (!rb_map) {
    fprintf(stderr, "map 'events' not found\n");
    goto cleanup;
  }
  rb_fd = bpf_map__fd(rb_map);
  rb = ring_buffer__new(rb_fd, handle_event, NULL, NULL);
  if (!rb) {
    fprintf(stderr, "ring_buffer__new failed\n");
    goto cleanup;
  }
  signal(SIGINT, on_sigint);
  signal(SIGTERM, on_sigint);


  fprintf(stderr, "running.. press Ctrl-C to stop");
  if (duration > 0)
    fprintf(stderr, " (auto-exit in %d sec)", duration);
  fprintf(stderr, "\n");

/* #ifndef NO_NVML */
/*   #<{(| nvml thread |)}># */
/*   pthread_t th; */
/*   pthread_create(&th, NULL, nvml_thread, NULL); */
/*   pthread_detach(th); */
/* #endif */

  // メインループ
  /* unsigned long long start_ns = 0; */
  /* if (duration > 0) { */
  /*   struct timespec ts; */
  /*   clock_gettime(CLOCK_MONOTONIC, &ts); */
  /*   start_ns = (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec; */
  /* } */

  while (!exiting) {
    /* err = ring_buffer__poll(rb, 200 #<{(| ms |)}>#); */
    /* if (err == -EINTR) { */
    /*   err = 0; */
    /*   break; */
    /* } */
    /* if (err < 0) { */
    /*   fprintf(stderr, "ring_buffer__poll: %d\n", err); */
    /*   break; */
    /* } */
    /*  */
    /* if (duration > 0) { */
    /*   struct timespec ts; */
    /*   clock_gettime(CLOCK_MONOTONIC, &ts); */
    /*   unsigned long long now_ns = */
    /*     (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec; */
    /*   if ((now_ns - start_ns) / 1000000000ull >= (unsigned long long)duration) */
    /*     break; */
    /* } */
    struct timespec ts = {.tv_sec=0,.tv_nsec=500*1000*1000}; // 500000ms
    nanosleep(&ts, NULL);
  }

cleanup:
  /* for (int i = 0; i < lidx; i++) { */
  /*   if (links[i].enter) bpf_link__destroy(links[i].enter); */
  /*   if (links[i].exit)  bpf_link__destroy(links[i].exit); */
  /* } */
  ring_buffer__free(rb);
  nvml_trace_bpf__destroy(skel);
  return err != 0;
}
