// #ifdef __has_include
// #  if __has_include("vmlinux.h")
// #    include "vmlinux.h"
// #  else
// struct trace_event_raw_sched_switch {
//     char prev_comm[16];
//     int prev_pid;
//     int prev_prio;
//     long long prev_state;
//     char next_comm[16];
//     int next_pid;
//     int next_prio;
// };
struct trace_event_raw_sched_wakeup {
    char comm[16];
    int pid;
    int prio;
    int success;
    int target_cpu;
};
// #  endif
// #endif
