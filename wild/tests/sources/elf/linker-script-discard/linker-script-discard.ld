SECTIONS {
    /DISCARD/ : {
        *(.text)
        *(.discard_before)
        *(.discard_after)
    }
}
