ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        mixed_and = section_start && LOADADDR(.text);
        mixed_or = section_start || (LOADADDR(.text) & 0);
        mixed_nor = !(section_start || (LOADADDR(.text) & 0));
        in_sec_nor = !LOADADDR(.text);
        *(.text*)
    }
}
out_sec_nor = !LOADADDR(.text);
