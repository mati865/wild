ENTRY(_start)

SECTIONS {
    .text 0x10000 : {
        sec_start = .;
        sec_end = sec_start + 0x20;
        min_seg = MIN(sec_end - sec_start, SEGMENT_START("text-segment", 0x10000));
        max_seg = MAX(sec_end - sec_start, SEGMENT_START("text-segment", 0x10000));
        *(.text*)
    }
}
