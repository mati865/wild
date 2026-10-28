//#Object:runtime.c
//#Mode:dynamic
//#Shared:shared.c
//#CompSoArgs:-fPIC
//#CompArgs:-fno-pie
//#LinkArgs:-no-pie
//#ReferenceLinkers:bfd,lld
//#DiffIgnore:.dynamic.DT_RELA
//#DiffIgnore:.dynamic.DT_RELAENT
//#DiffIgnore:.dynamic.DT_NEEDED

#include "../common/runtime.h"

extern int call_func(void);
extern int call_default(void);

int my_func(void) { return 42; }

void _start(void) {
  runtime_init();
  if (call_func() != 42) {
    exit_syscall(1);
  }
  if (call_default() != 10) {
    exit_syscall(2);
  }
  exit_syscall(42);
}
