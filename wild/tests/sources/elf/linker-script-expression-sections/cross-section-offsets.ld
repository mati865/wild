ENTRY(_start)
SECTIONS {
    .data 0x20000 : {
        data_start = .;
        KEEP(*(.data*))
    }
    .text 0x10000 : {
        cross_min = MIN(data_start, 0x10);
        cross_mul = data_start * 2;
        cross_neg = -data_start;
        *(.text*)
    }
}
