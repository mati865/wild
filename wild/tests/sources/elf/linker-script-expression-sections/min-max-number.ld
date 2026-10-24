ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        *(.text*)
        /* GNU ld compares section offsets with numeric operands here. */
        min_right = MIN(section_start, 0x10);
        min_left = MIN(0x10, section_start);
        max_right = MAX(section_start, 0x10);
        max_left = MAX(0x10, section_start);
    }
}
