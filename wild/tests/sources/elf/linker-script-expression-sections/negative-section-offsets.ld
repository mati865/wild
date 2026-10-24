ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        section_start = .;
        before_start = section_start - 1;
        restored_start = before_start + 1;
        minimum = MIN(before_start, section_start);
        in_sec_and = (section_start + 1) && 1;
        in_sec_zero_and = section_start && 1;
        in_sec_not = !section_start;
        *(.text*)
    }
}
