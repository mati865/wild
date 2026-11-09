//#Object:runtime.c
//#Object:ctor0.c
//#Object:ctor1.c
//#Object:xi-sentinels.c
//#Object:xl.c
//#TestUpdateInPlace:true

void crt_ctor(void) {}

extern void (*const ctor_i0)(void);
extern void (*const ctor_i1)(void);
extern void (*const ctor_l0)(void);

extern void (*const ctor_b0)(void);
extern void (*const ctor_b1)(void);

__attribute__((used, section(".CRT$XCA"))) void (*const ctor_a)(void) = crt_ctor;

__attribute__((used, section(".CRT$XD"))) void (*const ctor_c)(void) = crt_ctor;

int main() {
  // Ordered by suffix
  if (&ctor_a > &ctor_b0) return 101;
  if (&ctor_b1 > &ctor_c) return 102;
  // Ordered by link order
  if (&ctor_b0 > &ctor_b1) return 103;
  // Three-letter suffixes order by the two letters after `X`: XIA < XIZ < XLC. In particular,
  // `.CRT$XLC` must not land between the `.CRT$XIA` and `.CRT$XIZ` sentinels, since that range
  // is walked by the CRT as the C-initializer list (`_initterm_e`). Real CRT objects supply the
  // XIA/XIZ sentinels (crt2.o / cinitexe.o) while other CRT objects hang TLS-callback pointers
  // off XLA/XLC/XLD, so getting this wrong aborts every mingw program at startup.
  if (&ctor_i0 > &ctor_i1) return 104;
  if (&ctor_l0 < &ctor_i1) return 105;
  return 42;
}

//#ExpectSym: ctor_a section=".rdata"
//#ExpectSym: ctor_b0 section=".rdata"
//#ExpectSym: ctor_b1 section=".rdata"
//#ExpectSym: ctor_c section=".rdata"
//#ExpectSym: ctor_i0 section=".rdata"
//#ExpectSym: ctor_i1 section=".rdata"
//#ExpectSym: ctor_l0 section=".rdata"
