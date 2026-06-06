//#Config:default
//#Mode:dynamic
//#RunEnabled:false
//#CompArgs:-fPIE -fPIC
//#LinkArgs:-shared -z now -T ./linker-script-phdrs.ld
//#DiffIgnore:section.got
//#ExpectProgramHeader:LOAD flags=RX,sections=[.text]
//#ExpectProgramHeader:LOAD flags=RW,sections=[*]
//#ExpectProgramHeader:LOAD flags=R,sections=[.rodata,*]
//#NoProgramHeader:DYNAMIC
//#NoProgramHeader:PHDR
//#NoProgramHeader:NOTE
//#NoProgramHeader:GNU_STACK
//#NoProgramHeader:GNU_RELRO
//#NoProgramHeader:GNU_PROPERTY

//#Config:riscv:default
//#Arch:riscv64
//#ExpectProgramHeader:RISCV_ATTRIBUTES flags=R,sections=[.riscv.attributes]

const char message[] = "Hello PHDRS";

int foo(void) { return 42; }

const char* bar() { return &message[0]; }
