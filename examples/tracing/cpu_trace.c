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
#include "cpu_trace.skel.h"

#define CSV_MODE

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


static int handle_event(void *ctx, void *data, size_t len)
{
  const struct event *e = data;
  const char *fname = "UNKNOWN";
  switch (e->func) {
    case F_CPU_ONCPU:          fname="CPU_ONCPU"; break;
    case F_CPU_OFFCPU:         fname="CPU_OFFCPU"; break;
    case F_CPU_WAKEUP:         fname="CPU_WAKEUP"; break;
    case F_CPU_FUTEX_WAIT:     fname="CPU_FUTEX_WAIT"; break;
    case F_CPU_MMAP:           fname = "CPU_MMAP"; break;
    default: break;
  }

  switch (e->func) {
    default: {
               #ifdef CSV_MODE
                   printf("%ld.%06ld,%u,%u,%u,%lu%lu,%s,%s",
                       (long)(e->ts_ns/1000000000ULL),
                       (long)((e->ts_ns/1000ULL)%1000000ULL),
                       e->ppid, e->pid, e->tid,
                       e->trace_hi, e->trace_lo,
                       e->comm, fname);
               #else
                   printf("[%ld.%06ld] ppid=%u, pid=%u tid=%u trace_id=%lu%lu, comm=%s func=%s",
                       (long)(e->ts_ns/1000000000ULL),
                       (long)((e->ts_ns/1000ULL)%1000000ULL),
                       e->ppid, e->pid, e->tid,
                       e->trace_hi, e->trace_lo,
                       e->comm, fname);
               #endif
               break;
             }
  }

  switch (e->func) {
    case F_CPU_OFFCPU:
#ifdef CSV_MODE
      printf(",%u,%ld,", e->u.cpu.cpu, e->u.cpu.dur_ns);
#else
      printf(" cpu=%u oncpu=%.3f ms", e->u.cpu.cpu, e->u.cpu.dur_ns/1e6);
#endif
      break;
    case F_CPU_ONCPU:
#ifdef CSV_MODE
      printf(",%u,%ld,%ld",
          e->u.cpu.cpu, e->u.cpu.dur_ns, e->u.cpu.aux_ns);
#else
      printf(" cpu=%u offcpu=%.3f ms rq_latency=%.3f ms",
          e->u.cpu.cpu, e->u.cpu.dur_ns/1e6, e->u.cpu.aux_ns/1e6);
#endif
      break;
    case F_CPU_WAKEUP:
#ifdef CSV_MODE
      printf(",%u,,", e->u.cpu.cpu);
#else
      printf(" cpu=%u", e->u.cpu.cpu);
#endif
      break;
    case F_CPU_FUTEX_WAIT:
#ifdef CSV_MODE
      printf(",%u,%ld,", e->u.cpu.cpu, e->u.cpu.dur_ns);
#else
      printf(" cpu=%u futex_wait=%.3f ms", e->u.cpu.cpu, e->u.cpu.dur_ns/1e6);
#endif
      break;
    case F_CPU_MMAP:  // 新規追加: mmapイベントの出力
#ifdef CSV_MODE
      printf(",%u,%ld,", e->u.cpu.cpu, e->u.cpu.dur_ns);
#else
      printf(" cpu=%u mmap=%.3f ms", e->u.cpu.cpu, e->u.cpu.dur_ns/1e6);
#endif

    default: break;
  }
  struct pid_usage *u = get_usage(e->pid);
  if (u)
    printf(",%ld,%ld",
        u->cur, u->peak);
  else
    printf(",,");
  printf("\n");
  fflush(stdout);

  return 0;
}

static int bump_memlock_rlimit(void) {
  struct rlimit rlim = {
    .rlim_cur = RLIM_INFINITY,
    .rlim_max = RLIM_INFINITY,
  };
  return setrlimit(RLIMIT_MEMLOCK, &rlim);
}

struct link_holder {
  struct bpf_link *enter;
  struct bpf_link *exit;
};


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
  /* const char *bpf_obj   = "./cuda_trace.bpf.o"; */
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


  struct cpu_trace_bpf *skel = NULL;
  struct ring_buffer *rb = NULL;
  struct bpf_map *rb_map;
  int rb_fd;

  skel = cpu_trace_bpf__open();
  if (!skel) {
    fprintf(stderr, "failed to open skeleton\n");
    err = 1;
    goto cleanup;
  }

  {
    int pinned_fd = bpf_obj_get("/sys/fs/bpf/active_traces");
    if (pinned_fd < 0) {
      perror("bpf_obj_get");
      return 1;
    }

    err = bpf_map__reuse_fd(skel->maps.active_traces, pinned_fd);
    if (err) {
      fprintf(stderr, "reuse_fd failed: %d\n", err);
      return 1;
    }
  }

  // BPF オプションを load 前にセットしたい場合は skel->rodata / skel->bss に代入
  // 例:
  // skel->rodata->target_tgid = my_pid;

  err = cpu_trace_bpf__load(skel);
  if (err) {
    fprintf(stderr, "failed to load BPF skeleton: %d\n", err);
    goto cleanup;
  }

  // --- 自動アタッチ（BPF 側の SEC 文字列が絶対パスや soname を含む場合） ---
  err = cpu_trace_bpf__attach(skel);
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

  printf("timestamp,ppid,pid,tid,trace_id,comm,fname,cpu,dur_ns,aux_ns,cur,peak\n");
  // メインループ
  unsigned long long start_ns = 0;
  if (duration > 0) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    start_ns = (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec;
  }

  while (!exiting) {
    err = ring_buffer__poll(rb, 200 /* ms */);
    if (err == -EINTR) {
      err = 0;
      break;
    }
    if (err < 0) {
      fprintf(stderr, "ring_buffer__poll: %d\n", err);
      break;
    }

    if (duration > 0) {
      struct timespec ts;
      clock_gettime(CLOCK_MONOTONIC, &ts);
      unsigned long long now_ns =
        (unsigned long long)ts.tv_sec * 1000000000ull + ts.tv_nsec;
      if ((now_ns - start_ns) / 1000000000ull >= (unsigned long long)duration)
        break;
    }
  }

cleanup:
  ring_buffer__free(rb);
  cpu_trace_bpf__destroy(skel);
  return err != 0;
}
