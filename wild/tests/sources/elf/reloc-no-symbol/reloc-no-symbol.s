/*
//#AbstractConfig:default
//#Arch:x86_64
//#ReferenceLinkers:bfd,lld

//#Config:static:default
//#LinkArgs:-no-pie
//#ExpectSectionBytes:.rodata=0x0000000000000000887766554433221178563412
//#ExpectSectionBytes:.data=0x66554433

//#Config:shared:default
//#ReferenceLinkers:bfd
//#LinkArgs:-shared
//#ExpectError:can( |)not be used when making a shared object
*/

.text
.globl _start
_start:
    mov rodata_val(%rip), %rax
    mov tpoff_val(%rip), %rbx
    mov $60, %rax
    mov $42, %rdi
    syscall

.section .rodata,"a",@progbits
rodata_val:
    .reloc ., R_X86_64_64, 0
    .quad 0
    .reloc ., R_X86_64_64, 0x1122334455667788
    .quad 0
    .reloc ., R_X86_64_32, 0x12345678
    .long 0

.section .data,"aw",@progbits
tpoff_val:
    .reloc ., R_X86_64_TPOFF32, 0x33445566
    .long 0
