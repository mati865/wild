//#Mode:dynamic
//#RunEnabled:false
//#CompArgs:-fPIE -fPIC
//#LinkArgs:-shared -z now -T ./linker-script-phdrs.ld
//#DiffIgnore:section.got
//#DiffIgnore:section.riscv.attributes
//#DiffIgnore:segment.RISCV_ATTRIBUTES.*
// GNU ld emits `.riscv.attributes`, but Wild does not
//#DiffIgnore:riscv_attributes.*
//#ExpectProgramHeader:LOAD flags=RX,sections=[.text]
//#ExpectProgramHeader:LOAD flags=RW,sections=[*]
//#ExpectProgramHeader:LOAD flags=R,sections=[.rodata,*]
//#NoProgramHeader:DYNAMIC
//#NoProgramHeader:PHDR
//#NoProgramHeader:NOTE
//#NoProgramHeader:GNU_STACK
//#NoProgramHeader:GNU_RELRO
//#NoProgramHeader:GNU_PROPERTY

const char message[] = "Hello PHDRS";

int foo(void) { return 42; }

const char* bar() { return &message[0]; }
