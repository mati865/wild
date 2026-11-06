//#Object:runtime.c
//#Object:ctor0.c
//#Object:ctor1.c

void crt_ctor(void) {}

extern void (*const ctor_b0)(void);
extern void (*const ctor_b1)(void);

__attribute__((used, section(".CRT$XCA")))
void (*const ctor_a)(void) = crt_ctor;

__attribute__((used, section(".CRT$XD")))
void (*const ctor_c)(void) = crt_ctor;

int main() {
  // Ordered by suffix
  if (&ctor_a > &ctor_b0) return 101;
  if (&ctor_b1 > &ctor_c) return 102;
  // Ordered by link order
  if (&ctor_b0 > &ctor_b1) return 103;
  return 42;
}

//#ExpectSym: ctor_a section=".rdata"
//#ExpectSym: ctor_b0 section=".rdata"
//#ExpectSym: ctor_b1 section=".rdata"
//#ExpectSym: ctor_c section=".rdata"
