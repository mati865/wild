// An AArch64 object explicitly marked as supporting BTI.

.section .note.gnu.property, "a", %note
.balign 8
.long 4
.long 16
.long 5
.asciz "GNU"
.balign 8
.long 0xc0000000
.long 4
.long 1
.long 0

.text
.globl marked_function
.type marked_function, %function
marked_function:
    bti c
    ret
