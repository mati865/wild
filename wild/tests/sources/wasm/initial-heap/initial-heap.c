//#Config:default
//#LinkArgs: --initial-heap=33554432 -z stack-size=65536 --stack-first

extern char __heap_base;
extern char __heap_end;
volatile char data[17];

void _start(void) {
  data[0] = 1;
  unsigned long expected = ((unsigned long)&__heap_base + 33554432UL + 65535UL) & ~65535UL;
  if ((unsigned long)&__heap_end != expected) {
    __builtin_trap();
  }
}
