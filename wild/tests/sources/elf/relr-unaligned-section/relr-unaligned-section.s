//#AbstractConfig:default
//#Arch:x86_64
//#LinkArgs:-shared -z pack-relative-relocs
//#LinkerScript:layout.ld
//#ReferenceLinkers:lld
//#RunEnabled:false

//#Config:even-offset:default
//#CompArgs:-Wa,--defsym,offset=0

//#Config:odd-offset:default
//#CompArgs:-Wa,--defsym,offset=1

.section .data,"aw",@progbits
target:
    .quad 42

.section .odd,"aw",@progbits
.space offset
.globl pointer
pointer:
    .quad target

.section .note.GNU-stack,"",@progbits
