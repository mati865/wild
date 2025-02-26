//#Object:exit.c
//#EnableLinker:lld
//#Static:false
//#LinkArgs:-z now
//#Shared:trivial-dynamic-2.c
//#EnableLinker:lld
// We're linking different .so files, so this is expected.
//#DiffIgnore:.dynamic.DT_NEEDED
// We put a GLOB_DAT in .rela.dyn, other linkers use a JUMP_SLOT in .rela.plt.
//#DiffIgnore:.dynamic.DT_RELA
//#DiffIgnore:.dynamic.DT_RELAENT
// On aarch64, GNU ld seems to emit a GOT in the shared object even though it isn't needed.
//#DiffIgnore:section.got

#include "exit.h"

int foo(void);

void _start(void) {
    if (foo() != 10) {
        exit_syscall(20);
    }
    exit_syscall(42);
}
