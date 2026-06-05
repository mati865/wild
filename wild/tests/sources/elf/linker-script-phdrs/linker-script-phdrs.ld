PHDRS {
    text PT_LOAD;
    data PT_LOAD FLAGS(6);   /* R + W */
    rodata PT_LOAD FLAGS(4); /* R */
}

SECTIONS {
    .text : {
        *(.text*)
    } :text

    .data : {
        *(.data*)
    } :data

    .bss : {
        *(.bss*)
    } :data

    .rodata : {
        *(.rodata*)
    } :rodata
}
