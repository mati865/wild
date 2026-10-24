ENTRY(_start)
SECTIONS {
    .text 0x10000 : {
        *(.text*)
        /* The same conversion must work during early layout evaluation. */
        . = MAX(., 0x100);
        padded_end = .;
    }
}
