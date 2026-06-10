ENTRY(main)

SECTIONS {
    .text 0x100000 : AT(0x200000) {
        *(.text .text.*)
    }
    .data 0x200000 : AT(0x300000) {
        *(.data)
    }
}

ASSERT(ADDR(.text) == 0x100000, "VMA of .text should be 0x100000");
ASSERT(ADDR(.data) == 0x200000, "VMA of .data should be 0x200000");
