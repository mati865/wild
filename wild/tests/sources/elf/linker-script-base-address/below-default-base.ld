ENTRY(_start)

PHDRS {
    text PT_LOAD FLAGS(7) FILEHDR PHDRS;
}

SECTIONS {
    .text 0x3fe800 : {
        *(.text)
    } :text
}
