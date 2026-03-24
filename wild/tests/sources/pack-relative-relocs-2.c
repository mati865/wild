// Second object file for pack-relative-relocs.c (parallel group B).
//
// Contributes:
//   .init_array — ctor_b function pointer → RELATIVE relocation → RELR
//   .fini_array — dtor_b function pointer → RELATIVE relocation → RELR
//   .got        — GOT entries for val_a1/val_a2 (cross-file) and
//                 val_b1/val_b2 (same-file) via R_X86_64_REX_GOTPCRELX;
//                 kept by --no-relax → RELATIVE relocations → RELR

extern int val_a1;
extern int val_a2;

int val_b1 = 10;
int val_b2 = 20;

// Flag checked in _start; defined here so the cross-file access from _start
// generates a GOT load (R_X86_64_REX_GOTPCRELX) → RELATIVE relocation → RELR.
int ctor_b_ran = 0;

// --- .init_array entry (group B) -------------------------------------------
__attribute__((constructor)) static void ctor_b(void) { ctor_b_ran = 1; }

// --- .fini_array entry (group B) -------------------------------------------
__attribute__((destructor)) static void dtor_b(void) {}

// --- .got accessors (group B) -----------------------------------------------
// val_a1/val_a2: cross-file access → R_X86_64_REX_GOTPCRELX → GOT RELR
int get_val_a1(void) { return val_a1; }
int get_val_a2(void) { return val_a2; }

// val_b1/val_b2: same-file access with -fpic → R_X86_64_REX_GOTPCRELX;
// --no-relax keeps the GOT entry → RELATIVE relocation → RELR
int get_val_b1(void) { return val_b1; }
int get_val_b2(void) { return val_b2; }
