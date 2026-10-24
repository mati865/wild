ENTRY(_start)

MEMORY {
    ram : ORIGIN = 0x10000, LENGTH = 0x1000
}

SECTIONS {
    .text : {
        sym_mul = SEGMENT_START("text-segment", 0x10000) * 2;
        sym_mul_offset = SEGMENT_START("text-segment", 0x10020) * 2;
        sym_shl = SEGMENT_START("text-segment", 0x10000) << 1;
        sym_and = SEGMENT_START("text-segment", 0x10000) & 0xff00;
        sym_orig_mul = ORIGIN(ram) * 2;
        neg_seg = -SEGMENT_START("text-segment", 0x10000);
        neg_seg_offset = -SEGMENT_START("text-segment", 0x10020);
        neg_orig = -ORIGIN(ram);
        *(.text*)
    } > ram
}
