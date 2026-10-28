PHDRS {
    text PT_LOAD FLAGS(5);
    data PT_LOAD FLAGS(6);
}

SECTIONS {
    /DISCARD/ : {
        *(.discard_before)
    }
    .text : {
        *(.text*)
    } :text
    .data : {
        *(.data*)
    } :data
    /DISCARD/ : {
        *(.discard_after)
    }
}
