//#AbstractConfig:base
//#Arch:x86_64
//#ReferenceLinkers:lld
//#CompArgs:-fPIC
//#LinkArgs:-shared
//#AugmentLinkerScript:absolute.ld
//#RunDynSym:check_value
//#DiffEnabled:false
//#ExpectSym:absolute_value is_abs=true,address=0x1234

//#Config:relax:base

//#Config:no-relax:base
//#LinkArgs:--no-relax

#include <stdint.h>

int check_value(void) {
  uintptr_t value;
  __asm__("mov absolute_value@GOTPCREL(%%rip), %0" : "=r"(value));
  return value == 0x1234 ? 42 : 1;
}
