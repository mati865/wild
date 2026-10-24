ENTRY(_start)
SECTIONS {
    .text 0x10004 : {
        unaligned_dot_align = ALIGN(0x10);
        unaligned_expr_align = ALIGN(., 0x10);
        *(.text*)
    }
}
