// Tests that .relr.dyn correctly covers RELATIVE relocations across three
// section types when -z pack-relative-relocs is active in a PIE:
//
//   .init_array — constructor function pointers (one per object file)
//   .fini_array — destructor function pointers  (one per object file)
//   .got        — variable addresses from cross-file GOT references
//
// Two object files place symbols in different parallel linker groups, so the
// linker must merge and sort RELR entries from all three section types across
// both groups.
//
// With -fpic the compiler emits R_X86_64_REX_GOTPCRELX for extern variable
// accesses.  --no-relax prevents the linker from relaxing those into direct
// PC-relative loads, so each variable keeps its GOT entry and receives a
// RELATIVE relocation packed into .relr.dyn.
//
// DiffEnabled is false: --no-relax causes Wild to keep GOT entries that other
// linkers relax away, so the two outputs legitimately differ.  The meaningful
// check is runtime correctness: wrong RELR coverage either crashes before
// _start (bad .init_array pointer) or produces wrong values afterwards.
//
// Note: ld.so does not walk .init_array for the main executable; that is
// __libc_start_main's job.  Since this test bypasses libc, call_init_functions()
// walks the section manually.  This also means the .init_array function pointers
// must be correctly RELATIVE-relocated (via .relr.dyn) before we call them.

//#Object:pack-relative-relocs-2.c
//#Object:init.c
//#Object:runtime.c
//#CompArgs:-fpic
//#Arch:x86_64
//#Mode:dynamic
//#LinkArgs:--no-relax -pie -z pack-relative-relocs -z now
//#DiffEnabled:false
//#RequiresGlibc:true
//#RequiresGlibcVersion:2.36

#include "init.h"
#include "runtime.h"

// Flag set by ctor_a; checked in _start to confirm the constructor ran.
int ctor_a_ran = 0;

// Flag set by ctor_b (in pack-relative-relocs-2.c); accessing it cross-file
// from _start generates a GOT load → another RELATIVE relocation → RELR.
extern int ctor_b_ran;

// --- .init_array entry (group A) -------------------------------------------
// The function pointer stored in .init_array is a RELATIVE relocation packed
// into .relr.dyn.  call_init_functions() calls it; if the relocation is wrong
// the pointer is garbage and we crash or get a bad value.
__attribute__((constructor)) static void ctor_a(void) { ctor_a_ran = 1; }

// --- .fini_array entry (group A) -------------------------------------------
// Likewise a RELATIVE relocation in .fini_array covered by .relr.dyn.
// We bypass libc's exit so it never executes, but its address is still
// relocated at startup, exercising the RELR path for .fini_array.
__attribute__((destructor)) static void dtor_a(void) {}

// --- .got entries (group A) -------------------------------------------------
// val_a1/val_a2 are accessed cross-file from pack-relative-relocs-2.c.
// With -fpic the compiler in that TU emits R_X86_64_REX_GOTPCRELX for each;
// --no-relax keeps those GOT entries → RELATIVE relocations → RELR.
int val_a1 = 1;
int val_a2 = 2;

extern int get_val_a1(void);
extern int get_val_a2(void);
extern int get_val_b1(void);
extern int get_val_b2(void);

void _start(void) {
    runtime_init();

    // Walk .init_array manually (ld.so does not do this for the main
    // executable when libc is bypassed).  The function pointers in
    // .init_array are RELATIVE-relocated via .relr.dyn; calling through
    // them validates that those relocations were applied correctly.
    call_init_functions();

    // Both constructors must have run.
    if (ctor_a_ran != 1) exit_syscall(1);
    if (ctor_b_ran != 1) exit_syscall(2);   // cross-file GOT load → RELR

    // Cross-file GOT RELR: val_a1/val_a2 accessed from group B's TU.
    if (get_val_a1() != 1)  exit_syscall(3);
    if (get_val_a2() != 2)  exit_syscall(4);

    // Same-file GOT RELR (group B): val_b1/val_b2 with --no-relax.
    if (get_val_b1() != 10) exit_syscall(5);
    if (get_val_b2() != 20) exit_syscall(6);

    exit_syscall(42);
}
