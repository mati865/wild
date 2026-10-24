ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        *(.text*)
        /* ADDR retains the referenced section, including through arithmetic. */
        section_addr = ADDR(.text);
        after_addr = section_addr + 0x10;
    }
}
