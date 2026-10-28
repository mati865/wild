//#AbstractConfig:base
//#ReferenceLinkers:bfd,lld
//#Object:runtime.c
//#ExpectSym:absolute_value is_abs=true,address=0x1234
//#ExpectSym:abs_outside is_abs=true,address=0x5678
//#DiffIgnore:segment.INTERP.alignment
//#DiffIgnore:.dynamic.DT_RELAENT
//#DiffIgnore:.dynamic.DT_RELA
//#SkipArch:riscv64

//#Config:pie:base
//#Mode:dynamic
//#CompArgs:-fPIE
//#LinkArgs:-pie
//#LinkerScript:linker-script-absolute-pie.ld

//#Config:pie-got-relax:pie
//#CompArgs:-DTEST_GOT
//#Arch:x86_64
//#DiffEnabled:false
// BFD incorrectly applies the load base to this absolute symbol's GOT entry.
//#ReferenceLinkers:lld

//#Config:pie-got:pie-got-relax
//#LinkArgs:--no-relax

//#Config:pie-got-relr:pie-got
//#LinkArgs:-z pack-relative-relocs

//#Config:no-pie:base
//#CompArgs:-fno-pie
//#LinkArgs:-no-pie
//#LinkerScript:linker-script-absolute-no-pie.ld

#include <stdint.h>

#include "../common/runtime.h"

extern char absolute_value;
extern char abs_outside;
char* volatile stored_pointer = &absolute_value;
char* volatile stored_pointer_outside = &abs_outside;

void _start(void) {
  runtime_init();
  if ((uintptr_t)stored_pointer != 0x1234) {
    exit_syscall(1);
  }
  if ((uintptr_t)stored_pointer_outside != 0x5678) {
    exit_syscall(2);
  }
#ifdef TEST_GOT
  uintptr_t got_value;
  __asm__("mov absolute_value@GOTPCREL(%%rip), %0" : "=r"(got_value));
  if (got_value != 0x1234) {
    exit_syscall(3);
  }
  uintptr_t got_value_outside;
  __asm__("mov abs_outside@GOTPCREL(%%rip), %0" : "=r"(got_value_outside));
  if (got_value_outside != 0x5678) {
    exit_syscall(4);
  }
#endif
  exit_syscall(42);
}
