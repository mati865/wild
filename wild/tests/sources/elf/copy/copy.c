//#AbstractConfig:base
//#CompArgs:-fno-pie
//#Mode:static
//#Object:runtime.c
//#Copy:value.c:staged/value.c
//#Object:as(staged/value.o),noadd:$OUT_DIR/staged/value.c
//#Copy:$OUT_DIR/staged/value.o:staged/linked.o
//#ExpectSym:copied_value

//#Config:working-directory:base
//#LinkerWorkingDirectory:$OUT_DIR
//#LinkArgs:staged/linked.o

//#Config:script:base
//#AugmentLinkerScript:as(script.ld):script.ld

#include "../common/runtime.h"

int copied_value(void);

void _start(void) {
  runtime_init();
  exit_syscall(copied_value());
}
