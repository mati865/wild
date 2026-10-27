//#Arch:x86_64
//#Compiler:clang
//#LinkArgs:-shared -z now
//#ReferenceLinkers:bfd
//#ExpectError:absolute symbol|GotTlsOffset resolves to address not in TLS segment
//#ExpectErrorWild:GotTlsOffset resolves to address not in TLS segment 0x1234

.text
.globl get_offset
get_offset:
    mov absolute@gottpoff(%rip), %rax
    ret

.weak absolute
.hidden absolute
.type absolute,@tls_object
.set absolute, 0x1234

.section .note.GNU-stack,"",@progbits
