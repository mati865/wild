.text
.weak missing
.hidden missing
.globl check_offset
.type check_offset,@function
check_offset:
    mov missing@gottpoff(%rip), %rax
    mov defined_tls@gottpoff(%rip), %rdx
    cmp %rdx, %rax
    jne .Lfail
    mov $42, %eax
    ret
.Lfail:
    xor %eax, %eax
    ret

.section .tdata,"awT",@progbits
.globl defined_tls
.hidden defined_tls
.type defined_tls,@tls_object
defined_tls:
    .quad 42

.section .note.GNU-stack,"",@progbits
