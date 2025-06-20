//#AbstractConfig:default
//#Object:runtime.c
//#Mode:dynamic
//#Shared:shlib-undefined-2.c
//#DiffIgnore:.dynamic.DT_RELA
//#DiffIgnore:.dynamic.DT_RELAENT
//#DiffIgnore:.dynamic.DT_NEEDED

//#Config:allow:default
//#LinkArgs:--allow-shlib-undefined -z now
//#RunEnabled:false

//#Config:disallow:default
//#LinkArgs:--no-allow-shlib-undefined
//#ExpectError:def2

//#Config:shared:default
//#LinkArgs:-z now -shared
//#RunEnabled:false
// TODO: GNU ld sets the entry to _start even though we're writing a shared object. We probably
// should too.
//#DiffIgnore:file-header.entry

//#Config:weak:default
//#LinkArgs:-z now --allow-shlib-undefined
//#Mode:dynamic
//#Shared:shlib-undefined-2.c
//#Archive:shlib-undefined-3.c

#include "runtime.h"

int def1(void) {
    return 100;
}

int call_def1(void);
__attribute__((weak)) int foo(void);

void _start(void) {
    runtime_init();
    if (foo() == 40) {
        exit_syscall(0);
    }
    exit_syscall(call_def1());
}
