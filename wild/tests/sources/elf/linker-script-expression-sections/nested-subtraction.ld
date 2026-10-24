ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        section_end = section_start + 0x20;
        difference = MIN(section_end - section_start, LOADADDR(.text));
        *(.text*)
    }
}
